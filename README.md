# RAG Benchmark - Premier Modèle

Système RAG (Retrieval-Augmented Generation) avec BERT, FAISS et Llama 3.1:8b pour l'analyse de documents fiscaux français.

## 📋 Configuration

- **Embeddings**: BERT (bert-base-multilingual-cased)
- **Vector Store**: FAISS (IndexFlatL2)
- **LLM**: Llama 3.1:8b (via Ollama)
- **Chunk Size**: 256 caractères
- **Chunk Overlap**: 50 caractères

## 🚀 Installation

### 1. Prérequis

- Python 3.10+
- Ollama installé et en cours d'exécution
- Modèle Llama 3.1:8b téléchargé dans Ollama

### 2. Installation d'Ollama

```bash
# Windows : Télécharger depuis https://ollama.ai
# Ou via winget
winget install Ollama.Ollama

# Télécharger le modèle Llama 3.1:8b
ollama pull llama3.1:8b
```

### 3. Installation des dépendances Python

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 4. Configuration

```bash
# Copier le fichier d'exemple
copy .env.example .env

# Éditer .env si nécessaire (par défaut convient)
```

## 📊 Utilisation

### Étape 1 : Ingestion des documents

Cette étape charge les PDFs du dossier `données rag`, les découpe en chunks, génère les embeddings et les indexe dans FAISS.

```bash
python ingest_documents.py
```

**Avec test de recherche :**

```bash
python ingest_documents.py --test
```

### Étape 2 : Démarrer l'API

```bash
python api.py
```

L'API sera disponible sur `http://localhost:8000`

### Étape 3 : Tester avec Postman

#### Vérifier l'état de santé

```
GET http://localhost:8000/health
```

#### Envoyer une requête

```
POST http://localhost:8000/query
Content-Type: application/json

{
  "query": "Quelles sont les règles de TVA pour les entreprises?",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 512
}
```

#### Obtenir les statistiques

```
GET http://localhost:8000/stats
```

#### Voir la configuration

```
GET http://localhost:8000/config
```

## 📁 Structure du Projet

```
RAG benchmark/
├── données rag/              # Documents PDF source
├── src/
│   ├── ingestion/           # Modules de chargement et préparation
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/           # Vector store et recherche
│   │   └── vector_store.py
│   ├── generation/          # Interface LLM
│   │   └── llm_interface.py
│   ├── evaluation/          # Métriques et évaluation
│   │   └── evaluator.py
│   └── utils/               # Utilitaires
│       └── helpers.py
├── data/
│   └── vector_store/        # Index FAISS sauvegardé
├── logs/                    # Fichiers de log
├── config.yaml              # Configuration du modèle
├── api.py                   # API FastAPI
├── ingest_documents.py      # Script d'ingestion
├── requirements.txt         # Dépendances
└── README.md               # Ce fichier
```

## 🔍 Endpoints API

### `GET /`

Page d'accueil avec informations sur l'API

### `GET /health`

Vérification de l'état de tous les composants

### `POST /query`

**Body:**

```json
{
  "query": "Votre question",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Response:**

```json
{
  "query": "Votre question",
  "answer": "Réponse générée",
  "context_chunks": [...],
  "metrics": {...},
  "latency_seconds": 2.5
}
```

### `GET /stats`

Statistiques du vector store et des évaluations

### `POST /reset-metrics`

Réinitialise les métriques d'évaluation

### `GET /config`

Configuration actuelle du système

## 📊 Métriques d'Évaluation

Le système évalue automatiquement chaque requête sur plusieurs dimensions :

### Retrieval

- Nombre de chunks récupérés
- Distance moyenne/min/max des chunks
- Couverture des mots-clés (si fournis)

### Génération

- Longueur de la réponse
- Score de fidélité au contexte (faithfulness)
- Similarité sémantique avec ground truth (si fourni)
- Détection de refus de réponse

### Performance

- Latence totale (retrieval + génération)
- Timestamp de la requête

## 🛠️ Dépannage

### Problème : "Impossible de se connecter à Ollama"

**Solution :** Vérifiez qu'Ollama est en cours d'exécution

```bash
# Vérifier le statut
ollama list

# Démarrer Ollama si nécessaire (il démarre généralement automatiquement)
```

### Problème : "Vector store vide"

**Solution :** Exécutez d'abord l'ingestion

```bash
python ingest_documents.py
```

### Problème : "Out of memory" lors de l'ingestion

**Solution :** Réduire la taille du batch dans le code ou utiliser un GPU

```python
# Dans ingest_documents.py, ligne de génération des embeddings
enriched_chunks = embedder.embed_chunks(chunks, batch_size=16)  # Réduire de 32 à 16
```

### Problème : Le modèle est lent

**Solutions :**

- Utiliser un GPU si disponible (changer `device: "cuda"` dans config.yaml)
- Réduire `max_tokens` dans les requêtes
- Réduire `top_k` pour récupérer moins de chunks

## 📈 Prochaines Étapes

1. **Créer un jeu de test** avec questions et réponses attendues
2. **Comparer avec d'autres modèles** (GPT, Claude, autres LLMs)
3. **Optimiser les hyperparamètres** (chunk_size, top_k, temperature)
4. **Ajouter des métriques avancées** (RAGAS, TruLens)
5. **Développer un dashboard** de visualisation

## 📝 Notes

- Les documents sont automatiquement rechargés depuis `données rag/`
- L'index FAISS est sauvegardé dans `data/vector_store/`
- Les logs sont dans `logs/rag_benchmark.log`
- La configuration peut être modifiée dans `config.yaml`

## 📄 Licence

Projet interne Tessi - 2026
