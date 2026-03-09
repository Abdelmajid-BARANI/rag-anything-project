# 📖 Explication de `src/evaluation/metrics.py`

---

## C'est quoi ce fichier ?

Ce fichier implémente les **4 métriques RAGAS** — des algorithmes qui évaluent automatiquement la qualité du pipeline RAG. Chaque métrique utilise le LLM comme juge ("LLM-as-Judge") : on envoie des questions au LLM et on interprète ses réponses pour calculer un score.

---

## Les constantes globales

```python
CHUNK_PREVIEW_LEN = 300   # taille max d'un chunk dans les prompts de jugement
MAX_CONTEXT_LEN = 3000    # contexte total max (évite de dépasser la fenêtre du LLM)
JUDGE_TIMEOUT = 600       # 10 minutes max par appel LLM
DECOMPOSE_MAX_TOKENS = 400
JUDGE_MAX_TOKENS = 300
```

Ces limites existent pour deux raisons :

- Ne pas dépasser la **fenêtre de contexte** de Llama (8k tokens pour llama3.1:8b)
- Limiter le temps d'évaluation (chaque appel LLM prend ~30 secondes sur CPU)

---

## `_parse_verdicts` — parser les réponses du LLM juge

Le LLM est censé répondre `["oui", "non", "oui"]` mais il peut répondre n'importe quoi. Cette fonction essaie de comprendre sa réponse de plusieurs façons :

```python
def _parse_verdicts(text: str, n_expected: int) -> List[bool]:
```

**Tentative 1 : JSON array**

```python
json_match = re.search(r'\[.*?\]', text, re.DOTALL)
# Cherche quelque chose comme ["oui", "non", "oui"] dans le texte
if json_match:
    parsed = json.loads(json_match.group(0))
    # Convertit "oui"/"non"/"yes"/"true"/"1"/"vrai" → True/False
    verdicts.append(str(v).strip().lower() in ("oui", "yes", "1", "true", "vrai"))
```

**Tentative 2 : lignes numérotées**

```python
# Ex: "1. oui  2. non  3. oui"
cleaned = re.sub(r'^[\d]+[\.\)\-:\s]+', '', line)   # enlève "1. " ou "1) "
verdicts.append(v.startswith(("oui", "yes", "true", "vrai")) or "oui" in v)
```

**Tentative 3 : séparés par virgule**

```python
# Ex: "oui, non, oui"
parts = re.split(r'[,;]+', text.lower())
```

**Tentative 4 (fallback) : cherche oui/non dans tout le texte dans l'ordre**

```python
for match in re.finditer(r'\b(oui|non|yes|no)\b', text.lower()):
    verdicts.append(match.group(1) in ("oui", "yes"))
```

**Si pas assez de verdicts trouvés :**

```python
while len(verdicts) < n_expected:
    verdicts.append(False)   # complète avec False = "non pertinent"
```

---

## `_safe_llm_call` — appel LLM sécuris

```python
def _safe_llm_call(llm, prompt, max_tokens=500, temperature=0.0, timeout=600, retries=1):
    for attempt in range(retries + 1):
        try:
            return llm.generate(prompt=prompt, temperature=temperature,
                                max_tokens=max_tokens, timeout=timeout)
        except Exception as e:
            if attempt < retries:
                time.sleep(2)   # attend 2 secondes avant de réessayer
                continue
            logger.warning(f"Erreur lors de l'appel LLM pour évaluation : {e}")
            return ""   # retourne chaîne vide en cas d'échec total
```

Retourner `""` au lieu de lever une exception : ça permet à l'évaluation de **continuer** même si un appel LLM échoue — le score sera 0 mais les autres questions seront quand même évaluées.

---

## Métrique 1 : `ContextPrecision` (1 appel LLM)

> **Question :** Les chunks récupérés sont-ils pertinents pour la question ?

### Le prompt

```python
BATCH_PROMPT = """Tu es un évaluateur de systèmes de recherche documentaire.
Pour la question ci-dessous, détermine si CHAQUE passage est utile pour y répondre.

Question : {question}

Passage 1 : [texte du chunk 1]
Passage 2 : [texte du chunk 2]
...

Réponds UNIQUEMENT avec une liste JSON de "oui" ou "non", un par passage, dans l'ordre.
Exemple pour 3 passages : ["oui", "non", "oui"]

Liste JSON :"""
```

1 seul appel LLM pour tous les chunks → efficace.

### Le calcul du score

```python
num_relevant = sum(relevance_flags)   # nombre de "oui"

if num_relevant == 0:
    score = 0.0
else:
    weighted_sum = 0.0
    cumulative_relevant = 0
    for k, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            cumulative_relevant += 1
            precision_at_k = cumulative_relevant / k   # proportion de pertinents jusqu'au rang k
            weighted_sum += precision_at_k
    score = weighted_sum / num_relevant
```

**Exemple :**

```
Chunks : [✅ pertinent, ✅ pertinent, ❌ non, ✅ pertinent, ❌ non]
k=1 : ✅ → precision@1 = 1/1 = 1.0
k=2 : ✅ → precision@2 = 2/2 = 1.0
k=3 : ❌ → rien
k=4 : ✅ → precision@4 = 3/4 = 0.75
k=5 : ❌ → rien

weighted_sum = 1.0 + 1.0 + 0.75 = 2.75
score = 2.75 / 3 = 0.917
```

Les chunks pertinents bien classés (en tête) donnent un meilleur score.

---

## Métrique 2 : `ContextRecall` (2 appels LLM)

> **Question :** Le contexte couvre-t-il toute l'info de la réponse attendue ?

### Appel 1 : décomposer la ground truth

```python
DECOMPOSE_PROMPT = """Décompose le texte suivant en une liste d'énoncés factuels courts.
Retourne un tableau JSON de chaînes de caractères. Maximum 5 énoncés.

Texte : {ground_truth}

Tableau JSON :"""
```

**Exemple :**

- Ground truth : *"La TVA est calculée sur le prix hors taxe. Le taux standard est 20%."*
- Énoncés : `["La TVA est calculée sur le prix HT", "Le taux standard de TVA est 20%"]`

### Appel 2 : vérifier chaque énoncé contre le contexte

```python
BATCH_SUPPORT_PROMPT = """Étant donné le contexte ci-dessous, détermine si CHAQUE énoncé
est soutenu par le contexte.

Contexte : [texte des chunks]

Énoncé 1 : La TVA est calculée sur le prix HT
Énoncé 2 : Le taux standard de TVA est 20%

Réponds UNIQUEMENT avec une liste JSON de "oui" ou "non"..."""
```

### Calcul

```python
score = supported_count / len(statements)   # énoncés couverts / total énoncés
```

---

## Métrique 3 : `Faithfulness` (2 appels LLM)

> **Question :** La réponse inventée-t-elle des informations ?

### Appel 1 : extraire les affirmations de la réponse

```python
EXTRACT_CLAIMS_PROMPT = """Extrais les affirmations factuelles de la réponse ci-dessous.
Retourne un tableau JSON de phrases courtes. Maximum 5 affirmations.

Question : {question}
Réponse : {answer}

Tableau JSON :"""
```

Si la réponse dit *"Le taux de TVA est 20% selon l'article 289 et la directive européenne 2006/112"*, on extrait :

1. `"Le taux de TVA est 20%"`
2. `"Cela est selon l'article 289"`
3. `"Cela est selon la directive européenne 2006/112"`

### Appel 2 : vérifier chaque affirmation contre le contexte

```python
claims_block = "\n".join(f"Affirmation {i+1} : {c}" for i, c in enumerate(claims))
# → "Affirmation 1 : Le taux de TVA est 20%
#    Affirmation 2 : Cela est selon l'article 289
#    ..."
```

### Calcul

```python
score = supported_count / len(claims)
# Si 2 affirmations sur 3 sont dans le contexte → score = 0.667
```

Une affirmation absente du contexte = le LLM a inventé cette information (**hallucination**).

---

## Métrique 4 : `RAGASScore` — la moyenne harmonique

```python
@staticmethod
def compute(context_precision, context_recall, faithfulness):
    scores = {}   # ne garde que les scores non-None
    if context_precision is not None: scores["cp"] = context_precision
    if context_recall is not None:    scores["cr"] = context_recall
    if faithfulness is not None:      scores["f"]  = faithfulness

    valid_scores = [s for s in scores.values() if s is not None and s > 0]

    if not valid_scores:
        harmonic_mean = 0.0
    else:
        n = len(valid_scores)
        harmonic_mean = n / sum(1.0 / s for s in valid_scores)
        # formule : n / (1/s1 + 1/s2 + 1/s3)
```

**Pourquoi la moyenne harmonique et pas la moyenne classique ?**

| Scores CP, CR, F | Moyenne classique | Moyenne harmonique |
| ---------------- | ----------------- | ------------------ |
| 1.0, 1.0, 1.0    | 1.0 ✅            | 1.0 ✅             |
| 1.0, 1.0, 0.5    | 0.833             | 0.75               |
| 1.0, 0.0, 1.0    | 0.667 ⚠️        | **0.0** ✅   |

La moyenne harmonique **punit sévèrement** un score nul : si une métrique est 0, le score global est 0. Ça force le système à être bon sur *tous* les critères.

---

## Résumé

```
                      1 appel LLM
Context Precision ──────────────► score [0-1]
                                           │
                      2 appels LLM         │
Context Recall ──────────────────► score [0-1] ou None (sans ground truth)
                                           │
                      2 appels LLM         │
Faithfulness ────────────────────► score [0-1]
                                           │
                                           ▼
                         RAGAS Score = moyenne harmonique
                                    (score global [0-1])
```
