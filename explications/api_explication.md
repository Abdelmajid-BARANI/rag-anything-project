# 📖 Explication de `api.py` — pas à pas pour débutant

---

## 🗺️ C'est quoi ce fichier ?

`api.py` est le **serveur web** du projet. Il expose le système RAG sous forme d'une **API REST** :
- une API REST, c'est comme un serveur de restaurant — tu envoies une commande (requête HTTP), il te prépare et te retourne quelque chose (réponse JSON)
- ici, Postman ou n'importe quel client HTTP joue le rôle du client qui passe commande

Le serveur tourne sur `http://localhost:8000` quand on lance `python api.py`.

---

## 🔢 Découpage du fichier

```
api.py
│
├── 1. Imports (lignes 1–25)
├── 2. Configuration globale (lignes 27–46)
├── 3. Fonction utilitaire numpy (lignes 49–64)
├── 4. Modèles de données Pydantic (lignes 67–100)
├── 5. Lifespan — démarrage/arrêt (lignes 103–154)
├── 6. Création de l'app FastAPI (lignes 156–171)
├── 7. Endpoints GET (lignes 174–225)
├── 8. Endpoint POST /query (lignes 228–308)
├── 9. Endpoints GET /stats et /config (lignes 310–325)
├── 10. Endpoint POST /evaluate (lignes 328–380)
├── 11. Endpoint POST /evaluate/dataset (lignes 382–457)
└── 12. Point d'entrée __main__ (lignes 459–470)
```

---

## 1️⃣ Les imports (lignes 1–25)

```python
import time
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
from loguru import logger
```

> **Analogie** : comme apporter ses outils avant de commencer un chantier.

| Import | Rôle |
|--------|------|
| `time` | Mesurer combien de temps une requête prend |
| `os`, `sys` | Manipuler les chemins de fichiers et le système |
| `asynccontextmanager` | Gérer le démarrage/arrêt du serveur proprement |
| `FastAPI` | Le framework qui crée le serveur web |
| `HTTPException` | Renvoyer une erreur HTTP avec un code et un message |
| `CORSMiddleware` | Autoriser d'autres sites (Postman, front-end) à appeler l'API |
| `BaseModel`, `Field` | Définir la structure exacte des données attendues/retournées |
| `logger` | Afficher des messages dans la console (avec couleurs) |

```python
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
```

Ces 3 lignes disent à Python : *"cherche aussi les modules dans le dossier `src/`"*.  
Sans ça, `from ingestion import BERTEmbedder` produirait une erreur `ModuleNotFoundError`.

```python
from ingestion import BERTEmbedder
from retrieval import FAISSVectorStore
from generation import OllamaLLM
from evaluation import RAGEvaluator
from utils import load_config, setup_logging
```

Import des 5 modules du projet :
- `BERTEmbedder` → transforme du texte en vecteurs numériques
- `FAISSVectorStore` → base vectorielle + recherche hybride
- `OllamaLLM` → client pour parler à Llama 3.1
- `RAGEvaluator` → calcule les métriques RAGAS
- `load_config`, `setup_logging` → charge `config.yaml`, configure les logs

---

## 2️⃣ Configuration globale (lignes 27–46)

```python
setup_logging(log_level="INFO")

config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
```

Ces deux lignes s'exécutent **une seule fois au démarrage** du script :
1. Active le système de logs colorés (loguru)
2. Lit `config.yaml` et le stocke dans `config` (un dictionnaire Python)

```python
embedder: Optional["BERTEmbedder"] = None
vector_store: Optional["FAISSVectorStore"] = None
llm: Optional["OllamaLLM"] = None
similarity_threshold = 0.3
config_top_k = 5
search_mode = "hybrid"
hybrid_alpha = 0.6
hybrid_candidate_factor = 4
```

**Variables globales** : déclarées ici à `None`, elles seront remplies plus tard dans le `lifespan`.  
`Optional["BERTEmbedder"]` signifie : *"soit un objet BERTEmbedder, soit None"* — c'est une indication de type pour le lecteur, pas une contrainte d'exécution.

---

## 3️⃣ Fonction utilitaire `_numpy_to_python` (lignes 49–64)

```python
def _numpy_to_python(obj) -> Any:
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    ...
```

**Problème qu'elle résout** : numpy utilise ses propres types (`np.float32`, `np.int64`…) que Python ne sait pas convertir en JSON natif. FastAPI essaie de sérialiser (convertir en JSON) les réponses, et plante si elle rencontre un `np.float32`.

**Solution** : cette fonction parcourt récursivement n'importe quel objet (dict, liste, valeur) et remplace chaque type numpy par son équivalent Python standard (`int`, `float`, `list`).

> **Analogie** : c'est un traducteur — il transforme le "dialecte numpy" en "français JSON standard".

---

## 4️⃣ Modèles Pydantic (lignes 67–100)

Pydantic permet de décrire exactement **la forme** des données qui entrent et sortent de l'API.  
Si quelqu'un envoie un `top_k` qui n'est pas un entier, Pydantic renvoie automatiquement une erreur 422 avant même que le code s'exécute.

### Modèle d'entrée — `QueryRequest`

```python
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question à poser au système RAG")
    top_k: int = Field(5, description="Nombre de chunks à récupérer", ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(300, ge=50, le=2000)
```

Ce que ça signifie :
- `query: str` → obligatoire (le `...` dans `Field(...)` = "requis"), doit être une chaîne de caractères
- `top_k: int = Field(5, ge=1, le=20)` → entier, **valeur par défaut = 5**, doit être **≥ 1 et ≤ 20** (`ge` = greater or equal, `le` = less or equal)
- `temperature: float = Field(0.2, ge=0.0, le=2.0)` → nombre décimal entre 0 et 2

### Modèle de sortie — `QueryResponse`

```python
class QueryResponse(BaseModel):
    query: str
    answer: str
    context_chunks: List[Dict]
    metrics: Dict
    latency_seconds: float
```

Décrit exactement ce que `/query` retournera : la question, la réponse, les chunks utilisés, des métriques, et le temps de traitement.

### Les autres modèles

| Classe | Utilisée par | Description |
|--------|-------------|-------------|
| `HealthResponse` | `GET /health` | Status + composants + config |
| `StatsResponse` | `GET /stats` | Statistiques du vector store |
| `EvalSingleRequest` | `POST /evaluate` | Question + ground_truth à évaluer |
| `EvalDatasetRequest` | `POST /evaluate/dataset` | Paramètres pour évaluer tout le jeu de test |

---

## 5️⃣ Le `lifespan` — démarrage et arrêt du serveur (lignes 103–154)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code AVANT yield : exécuté au DÉMARRAGE ---
    global embedder, vector_store, llm, ...

    embedder = BERTEmbedder(model_name=..., device=...)
    vector_store = FAISSVectorStore(embedding_dim=..., persist_directory=...)
    vector_store.load()
    llm = OllamaLLM(model=..., host=...)
    similarity_threshold = retrieval_config.get("similarity_threshold", 0.3)
    ...

    yield   # ← le serveur tourne ici et répond aux requêtes

    # --- Code APRÈS yield : exécuté à l'ARRÊT (Ctrl+C) ---
    logger.info("Arrêt de l'application...")
```

**C'est le "constructeur" du serveur.**  
Tout ce qui précède `yield` s'exécute **une seule fois** quand on lance `python api.py`.

Étapes dans l'ordre :
1. **Crée l'embedder MPNet** — charge le modèle sentence-transformers en mémoire (~400 Mo)
2. **Crée le vector store FAISS** — prépare la structure en mémoire
3. **Charge l'index FAISS depuis le disque** (`data/vector_store/faiss_index.index`) — si absent, avertit sans bloquer
4. **Connecte Ollama/Llama** — si Ollama n'est pas lancé, avertit sans bloquer (le serveur démarre quand même)
5. **Lit les paramètres** de retrieval depuis `config.yaml` et les stocke dans les variables globales

> **Pourquoi `global` ?** Sans `global`, les modifications seraient locales à la fonction et les autres endpoints ne verraient pas `embedder`, `llm`, etc.

---

## 6️⃣ Création de l'app FastAPI (lignes 156–171)

```python
app = FastAPI(
    title="RAG Benchmark API",
    description="API pour tester le système RAG...",
    version="1.0.0",
    lifespan=lifespan,
)
```

`app` est **l'objet central** — c'est lui qui reçoit toutes les requêtes HTTP et les route vers le bon endpoint.  
Le paramètre `lifespan=lifespan` branche la fonction de démarrage/arrêt vue ci-dessus.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    ...
)
```

**CORS** (Cross-Origin Resource Sharing) : sans ça, un navigateur web qui tente d'appeler l'API depuis un autre domaine serait bloqué. `"*"` signifie "tout le monde peut appeler l'API" — acceptable en développement local.

---

## 7️⃣ Endpoint `GET /` et `GET /health` (lignes 174–225)

### `GET /`

```python
@app.get("/", tags=["General"])
async def root():
    return {
        "message": "RAG Benchmark API",
        "version": "1.0.0",
        "model": config.get("model_name", "bert_faiss_llama3.1_v1"),
        "docs": "/docs"
    }
```

**Le décorateur `@app.get("/")`** dit à FastAPI : *"quand quelqu'un fait un GET sur `/`, exécute cette fonction"*.  
`async def` signifie que la fonction est **asynchrone** — elle peut être mise en pause pendant qu'elle attend (ex: réponse d'Ollama), permettant au serveur de traiter d'autres requêtes en parallèle.

### `GET /health`

```python
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    vs_ready = (
        vector_store is not None
        and vector_store.index.ntotal > 0
    )
    components_status = {
        "embedder": embedder is not None,
        "vector_store": vs_ready,
        "llm": llm is not None
    }
    overall_status = "healthy" if all(components_status.values()) else "degraded"
    return {"status": overall_status, "components": components_status, "config": {...}}
```

Vérifie que les 3 composants sont prêts :
- `embedder is not None` → le modèle MPNet est chargé
- `vector_store.index.ntotal > 0` → l'index FAISS contient des vecteurs (documents indexés)
- `llm is not None` → Ollama est connecté

Retourne `"healthy"` si tout va bien, `"degraded"` si un composant manque.

---

## 8️⃣ Endpoint principal `POST /query` (lignes 228–308)

C'est **le cœur du système RAG**. Voici chaque étape :

```python
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
```

`request: QueryRequest` → FastAPI lit automatiquement le corps JSON de la requête et le valide contre `QueryRequest`.

### Étape 0 — Vérification des prérequis

```python
_check_component("embedder", embedder)
_check_component("vector_store", vector_store)
_check_component("llm", llm)

if vector_store.index.ntotal == 0:
    raise HTTPException(status_code=503, detail="Vector store vide...")
```

`_check_component` lève une `HTTPException 503` si un composant est `None`.  
`HTTPException` arrête immédiatement la fonction et renvoie une réponse d'erreur HTTP au client.

### Étape 1 — Encoder la question en vecteur

```python
query_embedding = embedder.embed_text(request.query)
```

Transforme la question textuelle en un vecteur de 768 nombres (MPNet).

### Étape 2 — Recherche hybride dans FAISS + BM25

```python
effective_top_k = max(request.top_k, config_top_k)

if search_mode == "hybrid":
    retrieved_chunks = vector_store.hybrid_search(
        query_embedding=query_embedding,
        query_text=request.query,
        top_k=effective_top_k,
        alpha=hybrid_alpha,           # 0.6 = 60% poids sémantique
        candidate_factor=hybrid_candidate_factor,  # récupère top_k×4 avant fusion
    )
else:
    retrieved_chunks = vector_store.search(query_embedding, top_k=effective_top_k)
```

`effective_top_k = max(request.top_k, config_top_k)` : on prend le maximum entre ce que l'utilisateur demande et ce qui est configuré — pour ne jamais récupérer moins que ce que la config préconise.

### Étape 3 — Filtrage par seuil de similarité

```python
retrieved_chunks = [c for c in retrieved_chunks if c.get("similarity", 0) >= similarity_threshold]
```

**List comprehension** Python : garde seulement les chunks dont le score de similarité cosinus est ≥ 0.3.  
Les chunks trop éloignés sémantiquement de la question sont écartés.

### Étape 4 — Génération de la réponse par le LLM

```python
response_data = llm.generate_with_context(
    query=request.query,
    context_chunks=retrieved_chunks,
    temperature=request.temperature,
    max_tokens=request.max_tokens
)
```

Envoie à Ollama/Llama : `[contexte des chunks + question]` → reçoit une réponse en texte.

### Étape 5 — Retourner le résultat

```python
return QueryResponse(
    query=request.query,
    answer=response_data["answer"],
    context_chunks=_numpy_to_python(retrieved_chunks),
    metrics={"num_chunks_retrieved": len(retrieved_chunks)},
    latency_seconds=round(end_time - start_time, 3)
)
```

Construit et retourne l'objet `QueryResponse` (FastAPI le sérialise en JSON automatiquement).

---

## 9️⃣ Endpoints `GET /stats` et `GET /config` (lignes 310–325)

```python
@app.get("/stats")
async def get_stats():
    return StatsResponse(
        vector_store_stats=_numpy_to_python(vector_store.get_stats())
    )

@app.get("/config")
async def get_config():
    return config
```

- `/stats` → retourne le nombre de vecteurs dans FAISS, les dimensions, etc.
- `/config` → retourne tout le contenu de `config.yaml` tel quel

---

## 🔟 Endpoint `POST /evaluate` (lignes 328–380)

```python
@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_single(request: EvalSingleRequest):
    evaluator = RAGEvaluator(llm=llm, embedder=embedder, vector_store=vector_store)

    result = evaluator.evaluate_query_end_to_end(
        question=request.query,
        ground_truth=request.ground_truth,
        top_k=...,
        similarity_threshold=...,
        ...
    )
    return _numpy_to_python(result)
```

Crée un `RAGEvaluator` (à chaque requête, pas une fois au démarrage — car l'évaluation est ponctuelle), puis lance le pipeline complet :
1. Encode la question → recherche hybrid → génère la réponse
2. Calcule CP, CR, Faithfulness, RAGAS Score

---

## 1️⃣1️⃣ Endpoint `POST /evaluate/dataset` (lignes 382–457)

```python
@app.post("/evaluate/dataset", tags=["Evaluation"])
async def evaluate_dataset(request: EvalDatasetRequest):
    # 1. Lire test_questions.yaml
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = _yaml.safe_load(f)
    questions = test_data.get("questions", [])

    # 2. Limiter le nombre de questions si demandé
    if request.max_questions > 0:
        questions = questions[:request.max_questions]

    # 3. Évaluer chaque question
    report = evaluator.evaluate_dataset(test_questions=questions, ...)

    # 4. Sauvegarder le rapport
    RAGEvaluator.generate_report(
        evaluation_results=report,
        output_path="./logs/evaluation_report.json",
        config=config,
    )

    return _numpy_to_python(report)
```

Lit les 10 questions de `test_questions.yaml`, évalue chacune, sauvegarde le résumé dans `logs/evaluation_report.json` et retourne le rapport au client.

---

## 1️⃣2️⃣ Point d'entrée `__main__` (lignes 459–470)

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
```

`if __name__ == "__main__"` : ce bloc s'exécute **seulement** quand on lance directement `python api.py` (pas quand un autre fichier importe `api.py`).

**uvicorn** est le serveur ASGI qui fait réellement tourner FastAPI :
- `"api:app"` → module `api`, objet `app`
- `host="0.0.0.0"` → écoute sur toutes les interfaces réseau (pas juste localhost)
- `port=8000` → port TCP 8000
- `reload=False` → ne pas redémarrer automatiquement à chaque modification du code (activer en dev : `reload=True`)

---

## 🗺️ Schéma de flux complet d'une requête `/query`

```
Client (Postman)
    │
    │  POST /query  {"query": "...", "top_k": 5}
    ▼
FastAPI (api.py)
    │
    ├─ Validation Pydantic (QueryRequest) ──── erreur 422 si données invalides
    │
    ├─ _check_component() ─────────────────── erreur 503 si composant absent
    │
    ├─ embedder.embed_text(query) ──────────► vecteur 768d
    │
    ├─ vector_store.hybrid_search() ────────► 5 chunks pertinents
    │
    ├─ filtrage similarity >= 0.3 ──────────► chunks filtrés
    │
    ├─ llm.generate_with_context() ─────────► réponse texte (Ollama)
    │
    └─ return QueryResponse(...) ───────────► JSON au client
```

---

## 💡 Ce qu'il faut retenir

| Concept | Explication simple |
|---------|-------------------|
| `@app.get("/route")` | Décorateur qui associe une URL à une fonction Python |
| `async def` | Fonction asynchrone — peut attendre sans bloquer le serveur |
| `BaseModel` Pydantic | Contrat de données : définit ce qu'on accepte/retourne |
| `HTTPException` | Arrête la fonction et renvoie une erreur HTTP au client |
| Variables globales | `embedder`, `llm`, `vector_store` partagés entre tous les endpoints |
| `lifespan` | Initialise les composants lourds une seule fois au démarrage |
| `_numpy_to_python` | Convertit les types numpy → JSON-compatible avant de retourner |
