# 📚 Explication du Projet — RAG Benchmark V2

---

## 🤔 C'est quoi ce projet en résumé ?

Ce projet est un **système de questions-réponses intelligent** basé sur des documents PDF, avec une couche d'**évaluation automatique** de la qualité des réponses.

Concrètement : on dépose des fichiers PDF (par exemple des textes sur la TVA ou la fiscalité), on pose des questions en français comme *"Quels sont les trois formats obligatoires pour transmettre les factures électroniques ?"*, et le système :

1. Trouve les passages les plus pertinents dans les documents (recherche **hybride sémantique + lexicale**)
2. Génère une réponse précise avec un LLM local (Llama 3.1:8b)
3. **Évalue automatiquement la qualité de cette réponse** avec les métriques RAGAS

C'est ce qu'on appelle un système **RAG** (Retrieval-Augmented Generation) avec **benchmark d'évaluation intégré**.

> **Identifiant du modèle courant :** `mpnet_faiss_bm25_llama3.1_8b_v2`

---

## 🧱 Les trois grandes phases du système

### ⚙️ Phase 1 — Ingestion des documents (une seule fois)

```
Fichiers PDF (dossier "données rag/")
    │
    ▼
Lecture du texte (PyPDF, page par page)
    │
    ▼
Découpage en chunks (~1000 caractères, overlap 200)
    via RecursiveCharacterTextSplitter (LangChain)
    │
    ▼
Transformation en vecteurs numériques 768 dimensions
    (MPNet multilingue : paraphrase-multilingual-mpnet-base-v2)
    │
    ▼
Normalisation L2 des vecteurs
    │
    ▼
Stockage dans l'index FAISS IndexFlatIP (data/vector_store/)
    + Construction de l'index BM25 (en mémoire)
```

> **Analogie** : construire le double index d'une bibliothèque — un index sémantique (FAISS) pour le sens, un index lexical (BM25) pour les mots exacts.

---

### 💬 Phase 2 — Réponse à une question (à chaque requête)

```
Question de l'utilisateur
    │
    ▼
Transformation en vecteur MPNet 768d
    │
    ┌────────────────────┬────────────────────────┐
    ▼                    ▼
Recherche FAISS      Recherche BM25
(sémantique)         (lexicale / mots-clés)
    │                    │
    └────────────────────┘
                  │
                  ▼
    Fusion hybride (alpha=0.6)
    score_final = 0.6 × FAISS + 0.4 × BM25
                  │
                  ▼
    Top K×4 candidats → re-ranking → Top K
                  │
                  ▼
    Filtrage par seuil de similarité (≥ 0.3)
                  │
                  ▼
    Construction du contexte (passages pertinents)
                  │
                  ▼
    Envoi [contexte + question] à Llama 3.1:8b
    via Ollama (http://localhost:11434)
                  │
                  ▼
    Réponse JSON : answer + context_chunks + latency
```

---

### 📊 Phase 3 — Évaluation RAGAS (à la demande)

```
Question + Réponse générée + Chunks récupérés + Ground Truth
    │
    ┌──────────────┬───────────────┬──────────────┐
    ▼              ▼               ▼              ▼
Context       Context          Faithfulness   RAGAS
Precision     Recall           (Fidélité)     Score
(LLM juge)    (LLM juge ×2)   (LLM juge ×2)  (harmonique)
    │              │               │              │
    └──────────────┴───────────────┴──────────────┘
                        │
                        ▼
              Rapport JSON (logs/evaluation_report.json)
```

---

## 🗂️ Structure complète des fichiers

```
RAG benchmarkV2/
│
├── api.py                     ← Serveur FastAPI (8 endpoints REST)
├── ingest_documents.py        ← Ingestion PDFs → index FAISS + BM25
├── run_evaluation.py          ← Évaluation RAGAS standalone
├── test_system.py             ← Tests rapides de tous les composants
├── _debug_search.py           ← Utilitaire de debug pour la recherche
│
├── config.yaml                ← Configuration centralisée (modèle, seuils, etc.)
├── requirements.txt           ← Dépendances Python
├── test_questions.yaml        ← Jeu de test avec ground truths (8 questions fiscales)
├── postman_collection.json    ← Collection Postman prête à l'emploi
├── pyrightconfig.json         ← Configuration Pyright (type checking)
│
├── run_api.bat                ← Démarrer l'API (Windows)
├── run_ingest.bat             ← Lancer l'ingestion (Windows)
├── run_evaluation.bat         ← Lancer l'évaluation (Windows)
├── setup.bat                  ← Installation de l'environnement
│
├── données rag/               ← 📂 Déposer ici les PDFs à indexer
├── data/
│   └── vector_store/
│       └── faiss_index.index  ← Index vectoriel FAISS généré
├── logs/
│   ├── rag_benchmark.log      ← Logs d'exécution (loguru)
│   └── evaluation_report.json ← Rapport d'évaluation RAGAS
│
└── src/
    ├── ingestion/
    │   ├── document_loader.py ← Lecture PDF page par page (PyPDF)
    │   ├── chunker.py         ← Découpage RecursiveCharacterTextSplitter
    │   └── embedder.py        ← Embeddings MPNet 768d (sentence-transformers)
    ├── retrieval/
    │   └── vector_store.py    ← FAISS IndexFlatIP + BM25 + fusion hybride
    ├── generation/
    │   └── llm_interface.py   ← Client Ollama → Llama 3.1:8b
    ├── evaluation/
    │   ├── metrics.py         ← Implémentation RAGAS (CP, CR, F, Score)
    │   └── evaluator.py       ← Orchestrateur RAGEvaluator
    └── utils/
        └── helpers.py         ← Chargement config YAML, setup logging
```

---

## 🔍 La recherche hybride FAISS + BM25

C'est la principale amélioration de cette version V2. Le système combine deux approches complémentaires :

### Recherche sémantique (FAISS)

- Encode la question en vecteur MPNet 768 dimensions
- Calcule la **similarité cosinus** avec tous les chunks indexés (`IndexFlatIP` sur vecteurs normalisés L2)
- Capture le **sens et la sémantique** : "facture électronique" ≈ "dématérialisation des factures"

### Recherche lexicale (BM25 Okapi)

- Tokenise la question et les chunks en mots significatifs (≥ 2 caractères, minuscules)
- Calcule un **score BM25** basé sur la fréquence des termes et la longueur des documents
- Capture les **correspondances exactes** : noms propres, codes fiscaux, abréviations

### Fusion hybride

```
score_final = alpha × score_FAISS + (1 - alpha) × score_BM25_normalisé
            = 0.6 × score_FAISS + 0.4 × score_BM25
```

Le paramètre `hybrid_candidate_factor: 4` fait récupérer `top_k × 4` candidats de chaque méthode avant fusion, ce qui améliore la qualité du re-ranking.

| Paramètre                  | Valeur | Description                         |
| --------------------------- | ------ | ----------------------------------- |
| `search_mode`             | hybrid | FAISS + BM25 combinés              |
| `hybrid_alpha`            | 0.6    | Poids sémantique (FAISS)           |
| `hybrid_candidate_factor` | 4      | Sur-récupération avant fusion     |
| `top_k`                   | 5      | Chunks finaux retournés            |
| `similarity_threshold`    | 0.3    | Score minimum pour retenir un chunk |

---

## 📐 Les métriques d'évaluation RAGAS

Le système évalue automatiquement la qualité du pipeline RAG avec 4 métriques. Chacune utilise le LLM comme **juge** (LLM-as-Judge).

### 1. Context Precision (Précision du contexte)

> **Question :** Les chunks récupérés sont-ils pertinents pour la question ?

- Pour chaque chunk retourné, le LLM juge s'il est utile ou non
- On calcule une **Precision@k pondérée** : les chunks pertinents placés en tête comptent davantage
- Score de 0 à 1 (1 = tous les chunks sont pertinents et bien classés)

**Exemple :** 5 chunks récupérés, 4 pertinents avec les meilleurs en tête → score élevé.

```
Precision@k pondérée = (1 / N_pertinents) × Σ (precision@k × relevance_k)
```

**Ground truth requise :** Non (1 appel LLM)

---

### 2. Context Recall (Rappel du contexte)

> **Question :** Le contexte récupéré couvre-t-il toute l'information de la réponse attendue ?

- Nécessite une **ground truth** (réponse attendue)
- Étape 1 : le LLM décompose la ground truth en énoncés atomiques
- Étape 2 : pour chaque énoncé, le LLM vérifie s'il est couvert par le contexte récupéré
- Score = nb_énoncés_couverts / nb_total_énoncés

**Exemple :** Ground truth contient 5 énoncés, le contexte en couvre 4 → score = 0.80

**Ground truth requise :** Oui (2 appels LLM)

---

### 3. Faithfulness (Fidélité)

> **Question :** La réponse générée est-elle basée sur le contexte récupéré ?

- Étape 1 : le LLM extrait les affirmations factuelles de la réponse générée
- Étape 2 : pour chaque affirmation, le LLM vérifie si le contexte la soutient
- Score = nb_affirmations_soutenues / nb_total_affirmations
- Détecte les **hallucinations** (le LLM invente des informations absentes des documents)

**Exemple :** La réponse fait 4 affirmations, 3 sont dans le contexte → score = 0.75

**Ground truth requise :** Non (2 appels LLM)

---

### 4. RAGAS Score (Score global)

> **Résumé :** Score global combinant les 3 métriques précédentes

Calculé comme la **moyenne harmonique** de Context Precision, Context Recall et Faithfulness :

```
RAGAS Score = 3 / (1/CP + 1/CR + 1/F)
```

La moyenne harmonique pénalise les scores très faibles : si une métrique est nulle, le score global tombe à zéro.

---

### Tableau récapitulatif des métriques

| Métrique                   | Ce qu'elle mesure                                                | Ground Truth ? | Appels LLM |
| --------------------------- | ---------------------------------------------------------------- | :------------: | :--------: |
| **Context Precision** | Les bons chunks sont-ils récupérés et bien classés ?         |      Non      |     1     |
| **Context Recall**    | Toute l'info attendue est-elle dans le contexte ?                | **Oui** |     2     |
| **Faithfulness**      | La réponse est-elle fidèle au contexte (pas d'hallucination) ? |      Non      |     2     |
| **RAGAS Score**       | Score global (moyenne harmonique des 3 métriques)               |    Partiel    |     —     |

---

## 🛠️ Technologies utilisées

| Technologie                     | Version / Modèle                         | Rôle                                                     |
| ------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| **Python**                | 3.10+                                     | Langage de programmation principal                        |
| **sentence-transformers** | `paraphrase-multilingual-mpnet-base-v2` | Embeddings 768 dimensions multilingues (MPNet)            |
| **FAISS** (Facebook AI)   | `IndexFlatIP`                           | Index vectoriel — recherche cosinus ultra-rapide         |
| **BM25 Okapi**            | `rank_bm25`                             | Recherche lexicale complémentaire                        |
| **Llama 3.1:8b**          | via Ollama                                | LLM open-source local pour génération et jugement RAGAS |
| **Ollama**                | localhost:11434                           | Gestionnaire de LLMs locaux (CPU/GPU)                     |
| **FastAPI**               | —                                        | API REST avec documentation Swagger auto-générée       |
| **PyPDF**                 | —                                        | Extraction du texte des fichiers PDF                      |
| **LangChain**             | `RecursiveCharacterTextSplitter`        | Découpage hiérarchique du texte en chunks               |
| **loguru**                | —                                        | Logging structuré et coloré                             |
| **Pydantic**              | v2                                        | Validation et sérialisation des modèles de l'API        |

---

## 🔄 Architecture complète

```
┌───────────────────────────────────────────────────────────────────┐
│                      PHASE 1 : INGESTION                          │
│                                                                   │
│  📄 PDFs ──► 📝 Texte (PyPDF) ──► ✂️ Chunks (RecursiveCharText)  │
│                                          │                        │
│                         ┌────────────────┤                        │
│                         ▼                ▼                        │
│              🔢 Vecteurs MPNet 768d    📋 Corpus BM25             │
│                         │                │                        │
│                         ▼                ▼                        │
│              💾 FAISS IndexFlatIP    🗂️ BM25Okapi (mémoire)       │
└───────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                       PHASE 2 : REQUÊTE                           │
│                                                                   │
│  ❓ Question ──► 🔢 Vecteur MPNet ──► 🔍 FAISS (sémantique)       │
│                       │                                           │
│                       └──────────────► 🔍 BM25 (lexical)         │
│                                              │                    │
│                              Fusion hybride (α = 0.6)             │
│                                              │                    │
│                              Top K chunks pertinents              │
│                                              │                    │
│                         🤖 Llama 3.1:8b (Ollama)                 │
│                      [contexte + question → réponse]              │
│                                              │                    │
│                              ✅ Réponse JSON                      │
└───────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                  PHASE 3 : ÉVALUATION RAGAS                       │
│                                                                   │
│  [Question + Réponse + Chunks + Ground Truth]                     │
│        │              │              │                            │
│        ▼              ▼              ▼                            │
│   Context         Context       Faithfulness                      │
│   Precision       Recall        (Fidélité)                        │
│  (1 appel LLM)  (2 appels LLM) (2 appels LLM)                   │
│        │              │              │                            │
│        └──────────────┴──────────────┘                           │
│                        │                                          │
│                        ▼                                          │
│             📊 RAGAS Score (moyenne harmonique)                   │
│                        │                                          │
│                        ▼                                          │
│          📄 logs/evaluation_report.json                           │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Utilisation pas à pas

### Étape 1 — Installer l'environnement

```bash
setup.bat         # Windows (installe venv + dépendances + vérifie Ollama)
# ou manuellement :
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Étape 2 — Démarrer Ollama et télécharger Llama 3.1:8b

```bash
# Installer Ollama : https://ollama.ai  (ou : winget install Ollama.Ollama)
ollama pull llama3.1:8b
ollama serve          # doit tourner sur http://localhost:11434
```

### Étape 3 — Placer les PDFs

Copier les fichiers PDF dans le dossier `données rag/`.

### Étape 4 — Indexer les documents

```bash
python ingest_documents.py
# ou : run_ingest.bat
```

Génère `data/vector_store/faiss_index.index` et construit l'index BM25 en mémoire.

### Étape 5 — Démarrer l'API

```bash
python api.py
# ou : run_api.bat
```

API disponible sur `http://localhost:8000`.
Documentation interactive Swagger : `http://localhost:8000/docs`.

### Étape 6 — Poser une question

```bash
# Via curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelles sont les obligations de facturation?", "top_k": 5}'

# Via Postman : importer postman_collection.json
```

### Étape 7 — Évaluer la qualité (RAGAS)

```bash
# Évaluer 1 question
python run_evaluation.py --max-questions 1

# Évaluer toutes les questions du jeu de test
python run_evaluation.py

# Via l'API :
POST http://localhost:8000/evaluate          # une question avec ground_truth
POST http://localhost:8000/evaluate/dataset  # tout le jeu de test YAML
```

---

## 📊 Endpoints de l'API

| URL                   | Méthode | Description                                                   |
| --------------------- | -------- | ------------------------------------------------------------- |
| `/`                 | GET      | Page d'accueil + version + identifiant du modèle             |
| `/health`           | GET      | État de santé de tous les composants (embedder, FAISS, LLM) |
| `/config`           | GET      | Configuration complète chargée depuis `config.yaml`       |
| `/stats`            | GET      | Statistiques du vector store (nb chunks, dimensions)          |
| `/query`            | POST     | Poser une question — retourne réponse + chunks + latence    |
| `/evaluate`         | POST     | Évaluer une question avec métriques RAGAS                   |
| `/evaluate/dataset` | POST     | Évaluer tout le jeu de test `test_questions.yaml`          |
| `/docs`             | GET      | Documentation interactive Swagger UI                          |

### Exemple de requête `/query`

```json
POST /query
{
  "query": "Quelles sont les règles de TVA pour les entreprises ?",
  "top_k": 5,
  "temperature": 0.2,
  "max_tokens": 300
}
```

### Exemple de réponse `/query`

```json
{
  "query": "...",
  "answer": "...",
  "context_chunks": [
    {"text": "...", "score": 0.82, "rank": 1, "source": "doc.pdf", "page": 3}
  ],
  "metrics": {"num_chunks": 5, "avg_similarity": 0.74},
  "latency_seconds": 2.31
}
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
model_name: "mpnet_faiss_bm25_llama3.1_8b_v2"
description: "RAG avec MPNet (768d), FAISS + BM25 Hybrid Search et Llama 3.1:8b"

ingestion:
  chunk_size: 1000        # caractères (RecursiveCharacterTextSplitter)
  chunk_overlap: 200      # chevauchement entre chunks consécutifs
  chunk_mode: "chars"
  data_source: "./données rag"

embeddings:
  model_name: "paraphrase-multilingual-mpnet-base-v2"  # MPNet multilingue 768d
  dimension: 768
  device: "cpu"           # "cuda" si GPU disponible

vector_store:
  type: "faiss"
  index_type: "IndexFlatIP"   # produit scalaire sur vecteurs L2-normalisés = cosine

retrieval:
  top_k: 5
  similarity_threshold: 0.3   # seuil bas pour ne pas rater de chunks pertinents
  search_mode: "hybrid"        # "semantic" (FAISS seul) | "hybrid" (FAISS + BM25)
  hybrid_alpha: 0.6            # 0 = tout BM25, 1 = tout FAISS
  hybrid_candidate_factor: 4   # sur-récupération : top_k × 4 avant fusion

llm:
  provider: "ollama"
  model: "llama3.1:8b"
  host: "http://localhost:11434"
  temperature: 0.2            # bas = réponses factuelles et déterministes
  max_tokens: 300

evaluation:
  metrics: [context_precision, context_recall, faithfulness, ragas_score]
  llm_judge_temperature: 0.0  # température basse pour le jugement LLM
  llm_judge_max_tokens: 500
  report_output: "./logs/evaluation_report.json"

logging:
  level: "INFO"
  file: "./logs/rag_benchmark.log"
```

---

## 🧪 Jeu de test (`test_questions.yaml`)

Contient 8 questions sur la fiscalité française avec :

- `query` : la question posée au système
- `ground_truth` : la réponse attendue (utilisée pour Context Recall)
- `keywords` : mots-clés attendus dans le contexte récupéré
- `expected_topics` : thèmes devant être couverts par la réponse

---

## 🔧 Choix des modèles d'embeddings

Le fichier `config.yaml` documente plusieurs alternatives selon le cas d'usage :

| Modèle                                      | Dimensions | Langue          | Usage recommandé                                   |
| -------------------------------------------- | ---------- | --------------- | --------------------------------------------------- |
| `paraphrase-multilingual-mpnet-base-v2` ✅ | 768        | FR/EN/multi     | **Défaut — meilleure qualité sémantique** |
| `paraphrase-multilingual-MiniLM-L12-v2`    | 384        | FR/EN/multi     | Version légère si ressources limitées            |
| `OrdalieTech/Solon-embeddings-large-0.1`   | 1024       | Droit français | Spécialisé textes juridiques français            |
| `dangvantuan/sentence-camembert-large`     | 1024       | Français       | Très bon pour le français général               |
| `all-MiniLM-L6-v2`                         | 384        | Anglais         | ❌ Ne pas utiliser pour des documents en français  |

> Pour changer de modèle : modifier `embeddings.model_name` et `embeddings.dimension` dans `config.yaml`, puis relancer l'ingestion.

---

## 📝 Résumé en une phrase

> Ce projet est un assistant IA juridique/fiscal qui lit des PDFs, les indexe avec MPNet (768d) + FAISS + BM25 pour une recherche hybride sémantique et lexicale, génère des réponses via Llama 3.1:8b en local (Ollama), et évalue automatiquement la qualité du pipeline avec les métriques RAGAS (Context Precision, Context Recall, Faithfulness, RAGAS Score) — le tout exposé via une API REST FastAPI et testable avec Postman.
