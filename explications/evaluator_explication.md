# 📖 Explication de `src/evaluation/evaluator.py`

---

## C'est quoi ce fichier ?

C'est l'**orchestrateur de l'évaluation**. Il ne calcule pas les métriques lui-même (c'est le rôle de `metrics.py`), mais il **coordonne** le pipeline complet :
1. Retrieval (chercher les chunks)
2. Generation (formuler la réponse)
3. Evaluation (calculer CP, CR, Faithfulness, RAGAS)

> **Analogie** : c'est le chef de projet qui dit "d'abord cherchez les documents, puis donnez-les au rédacteur, puis soumettez le résultat au jury".

---

## Structure de la classe

```
RAGEvaluator
│
├── __init__(llm, embedder, vector_store)
│
├── evaluate_single(question, answer, context_chunks, ground_truth)
│   └── Calcule CP + CR + F + RAGAS sur une réponse déjà générée
│
├── evaluate_query_end_to_end(question, ground_truth, ...)
│   └── Pipeline complet : retrieval → generation → evaluate_single
│
├── evaluate_dataset(test_questions, ...)
│   └── Boucle sur toutes les questions + calcule les moyennes
│
├── generate_report(results, output_path, config)
│   └── Sauvegarde le rapport JSON
│
└── print_summary(results)
    └── Affiche le tableau dans la console
```

---

## `__init__` — initialisation

```python
def __init__(self, llm, embedder=None, vector_store=None):
    self.llm = llm                     # pour générer les réponses ET juger les métriques
    self.embedder = embedder           # pour encoder les questions (optionnel)
    self.vector_store = vector_store   # pour chercher les chunks (optionnel)
```

`embedder` et `vector_store` sont optionnels car `evaluate_single` peut fonctionner sans eux (si on a déjà la réponse et les chunks produits par ailleurs).

---

## `evaluate_single` — évaluer une réponse déjà générée

```python
def evaluate_single(self, question, answer, context_chunks, ground_truth="",
                    compute_precision=True, compute_recall=True, compute_faithfulness=True):
```

**Ce qu'elle reçoit :**
- `question` : la question posée
- `answer` : la réponse que le RAG a générée
- `context_chunks` : les chunks que le RAG a récupérés
- `ground_truth` : la vraie réponse attendue (nécessaire pour Context Recall)

**Ce qu'elle fait :**

```python
results = {
    "question": question,
    "answer_preview": answer[:200],
    "num_context_chunks": len(context_chunks),
    "ground_truth_provided": bool(ground_truth),
}

# Context Precision
if compute_precision:
    cp_result = ContextPrecision.compute(question, context_chunks, self.llm, ground_truth)
    results["context_precision"] = cp_result
    cp_score = cp_result.get("score")

# Context Recall (seulement si ground_truth fournie)
if compute_recall and ground_truth:
    cr_result = ContextRecall.compute(question, context_chunks, self.llm, ground_truth)
    results["context_recall"] = cr_result
    cr_score = cr_result.get("score")
elif compute_recall and not ground_truth:
    results["context_recall"] = {"score": None, "reason": "ground_truth manquante"}
    # ↑ on signale explicitement pourquoi CR est absent

# Faithfulness
if compute_faithfulness:
    f_result = Faithfulness.compute(question, answer, context_chunks, self.llm)
    results["faithfulness"] = f_result
    f_score = f_result.get("score")

# Score RAGAS global
ragas = RAGASScore.compute(cp_score, cr_score, f_score)
results["ragas_score"] = ragas
results["evaluation_time_seconds"] = round(time.time() - start_time, 2)

return results
```

---

## `evaluate_query_end_to_end` — pipeline complet

C'est la méthode appelée par `api.py` pour l'endpoint `/evaluate`.

```python
def evaluate_query_end_to_end(self, question, ground_truth="", top_k=5,
                               similarity_threshold=0.3, temperature=0.2,
                               max_tokens=300, search_mode="hybrid",
                               hybrid_alpha=0.6, hybrid_candidate_factor=4):
```

**Les 3 étapes :**

### Étape 1 : Retrieval

```python
query_embedding = self.embedder.embed_text(question)

if search_mode == "hybrid":
    context_chunks = self.vector_store.hybrid_search(
        query_embedding=query_embedding,
        query_text=question,
        top_k=top_k, alpha=hybrid_alpha,
        candidate_factor=hybrid_candidate_factor,
    )
else:
    context_chunks = self.vector_store.search(query_embedding, top_k=top_k)

context_chunks = [c for c in context_chunks if c.get("similarity", 0) >= similarity_threshold]

# Si aucun chunk suffisamment similaire :
if not context_chunks:
    return {"question": question, "answer": "", "error": "Aucun chunk trouvé"}
```

### Étape 2 : Generation

```python
gen_start = time.time()
response_data = self.llm.generate_with_context(
    query=question,
    context_chunks=context_chunks,
    temperature=temperature,
    max_tokens=max_tokens,
)
generation_time = time.time() - gen_start
answer = response_data.get("answer", "")
```

### Étape 3 : Evaluation

```python
eval_results = self.evaluate_single(
    question=question,
    answer=answer,
    context_chunks=context_chunks,
    ground_truth=ground_truth,
)

# Enrichit les résultats avec les temps de traitement
eval_results["answer"] = answer
eval_results["retrieval_time"] = round(retrieval_time, 3)
eval_results["generation_time"] = round(generation_time, 3)
eval_results["total_time"] = round(time.time() - start, 3)
return eval_results
```

---

## `evaluate_dataset` — évaluer tout le jeu de test

```python
def evaluate_dataset(self, test_questions, top_k=5, ...):
    scores_aggregate = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
        "ragas_score": [],
    }

    for i, tq in enumerate(test_questions, 1):
        question = tq.get("query", "")
        ground_truth = tq.get("ground_truth", "")
        qid = tq.get("id", i)

        logger.info(f"[{i}/{total}] Évaluation question #{qid} : {question[:60]}...")

        try:
            result = self.evaluate_query_end_to_end(question=question,
                                                    ground_truth=ground_truth, ...)
            result["question_id"] = qid

            # Collecte les scores pour calculer la moyenne à la fin
            for metric_key in scores_aggregate:
                s = result.get(metric_key, {}).get("score")
                if s is not None:
                    scores_aggregate[metric_key].append(s)

            results_per_question.append(result)

        except Exception as e:
            logger.error(f"Erreur question #{qid} : {e}")
            results_per_question.append({"question_id": qid, "error": str(e)})
            # ↑ on enregistre l'erreur mais on continue les autres questions
```

**Calcul des moyennes :**

```python
averages = {}
for metric_key, values in scores_aggregate.items():
    if values:
        averages[metric_key] = round(sum(values) / len(values), 4)
    else:
        averages[metric_key] = None

# Ex: {"context_precision": 0.5717, "context_recall": 0.355, ...}
```

---

## `generate_report` — sauvegarde du rapport JSON

```python
@staticmethod
def generate_report(evaluation_results, output_path="./logs/evaluation_report.json", config=None):
    report = {
        "report_type": "RAGAS Evaluation",
        "generated_at": datetime.now().isoformat(),
    }

    if config:
        report["model_config"] = {
            "model_name": config.get("model_name"),
            "embedding_model": config.get("embeddings", {}).get("model_name"),
            ...
        }

    report["evaluation"] = evaluation_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        # ↑ ensure_ascii=False : garde les accents (é, à...) en clair
        # ↑ indent=2 : indentation de 2 espaces → fichier lisible par un humain
```

---

## `print_summary` — affichage dans la console

```python
@staticmethod
def print_summary(evaluation_results):
    for key, label in metrics_labels.items():
        score = avg.get(key)
        if score is not None:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            # ↑ barre de progression ASCII : score=0.57 → ███████████░░░░░░░░░
            print(f"  {label:<25} {score:>8.4f}  {bar}")
```

`"█" * int(0.57 * 20)` = `"█" * 11` = `"███████████"` → représentation visuelle du score.

---

## Flux complet en schéma

```
evaluate_dataset(test_questions)
    │
    │  pour chaque question
    ▼
evaluate_query_end_to_end(question, ground_truth)
    │
    ├── 1. embed_text(question) → vecteur
    ├── 2. hybrid_search(vecteur, question) → 5 chunks
    ├── 3. generate_with_context(question, chunks) → réponse
    │
    └── evaluate_single(question, réponse, chunks, ground_truth)
            │
            ├── ContextPrecision.compute()  → CP score + détails
            ├── ContextRecall.compute()     → CR score + détails
            ├── Faithfulness.compute()      → F score + détails
            └── RAGASScore.compute()        → score global
    │
    ▼
scores_aggregate → calculate averages
    │
    ▼
generate_report() → logs/evaluation_report.json
    │
    ▼
print_summary() → affichage console
```
