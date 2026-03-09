# Intégrer RAG-Anything dans ce projet (100% open source avec Ollama)

---

## Ce que ça change concrètement

RAG-Anything est basé sur **LightRAG**, qui supporte n'importe quel LLM compatible API OpenAI.  
Ollama expose exactement ce format sur `http://localhost:11434/v1` → **tout fonctionne localement, sans clé API.**

| Composant | Actuel | Après intégration RAG-Anything (open source) |
|-----------|--------|----------------------------------------------|
| Parser PDF | `pdfplumber` (texte) | **MinerU** (texte + images + tableaux + équations) |
| Index | FAISS + BM25 | **LightRAG** (Knowledge Graph + vecteurs) |
| LLM texte | Llama 3.1:8b via Ollama | **Llama 3.1:8b** via Ollama (inchangé ✅) |
| LLM vision (images) | ❌ non supporté | **LLaVA** via Ollama (open source ✅) |
| Embeddings | MPNet 768d local | **nomic-embed-text** via Ollama (768d ✅) |
| Requêtes | `hybrid_search()` | `rag.aquery()` : local / global / hybrid / naive |
| Coût | 0 € | **0 €** (100% local ✅) |

---

## Modèles Ollama nécessaires

Télécharger les modèles avant de commencer :

```bash
# LLM principal (déjà installé)
ollama pull llama3.1:8b

# LLM vision pour analyser les images dans les PDFs
ollama pull llava:13b          # recommandé (meilleure qualité)
# ou plus léger :
ollama pull llava:7b

# Modèle d'embeddings (remplace MPNet)
ollama pull nomic-embed-text   # 768 dimensions, rapide
```

Vérifier qu'ils sont disponibles :
```bash
ollama list
```

---

## Prérequis à installer

### 1. Python packages

```bash
pip install raganything lightrag-hku
pip install "raganything[all]"
```

### 2. MinerU (le parser de documents)

```bash
pip install mineru
mineru --version
```

MinerU télécharge ses modèles (~3-5 Go) au premier lancement. Il tourne en **CPU** (lent mais fonctionnel) ou **GPU** si disponible.

### 3. LibreOffice (optionnel — pour Word/Excel/PowerPoint)

Télécharger depuis : https://www.libreoffice.org/download/download/  
Nécessaire seulement si tu ajoutes des `.docx`, `.xlsx` etc.

---

## Plan d'intégration étape par étape

### Étape 1 — Créer un fichier `.env`

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llava:13b
OLLAMA_EMBED_MODEL=nomic-embed-text
OUTPUT_DIR=./output
PARSER=mineru
PARSE_METHOD=auto
```

---

### Étape 2 — Créer `src/ingestion/rag_anything_pipeline.py`

```python
"""
Pipeline d'ingestion RAG-Anything (multimodal, 100% open source via Ollama)
"""
import asyncio
import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

load_dotenv()

OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_BASE_URL     = f"{OLLAMA_HOST}/v1"   # endpoint compatible OpenAI
LLM_MODEL           = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
VISION_MODEL        = os.getenv("OLLAMA_VISION_MODEL", "llava:13b")
EMBED_MODEL         = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OUTPUT_DIR          = os.getenv("OUTPUT_DIR", "./output")
DATA_DIR            = "./données rag"
FAKE_API_KEY        = "ollama"   # Ollama n'exige pas de vraie clé, mais le paramètre est requis


# ─────────────────────────────────────────────
# 1. Fonction LLM texte — Llama 3.1 via Ollama
# ─────────────────────────────────────────────
def make_llm_func():
    def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # Retire les paramètres non supportés par Ollama
        kwargs.pop("hashing_kv", None)
        return openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=FAKE_API_KEY,
            base_url=OLLAMA_BASE_URL,
            **kwargs,
        )
    return llm_func


# ─────────────────────────────────────────────
# 2. Fonction LLM vision — LLaVA via Ollama
#    Appelée pour décrire les images extraites des PDFs
# ─────────────────────────────────────────────
def make_vision_func():
    def vision_func(prompt, system_prompt=None, history_messages=[],
                    image_data=None, messages=None, **kwargs):
        kwargs.pop("hashing_kv", None)

        # Format messages OpenAI complet (VLM enhanced query)
        if messages:
            return openai_complete_if_cache(
                VISION_MODEL, "",
                system_prompt=None, history_messages=[],
                messages=messages,
                api_key=FAKE_API_KEY, base_url=OLLAMA_BASE_URL, **kwargs,
            )

        # Image en base64 : on envoie via l'API Ollama /api/generate
        elif image_data:
            payload = {
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [image_data],   # Ollama accepte base64 directement
                "stream": False,
                "options": {"temperature": 0.1}
            }
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=120
            )
            return response.json().get("response", "")

        # Texte seul → on utilise le LLM texte
        else:
            return make_llm_func()(prompt, system_prompt, history_messages, **kwargs)

    return vision_func


# ─────────────────────────────────────────────
# 3. Fonction embeddings — nomic-embed-text via Ollama
#    Ollama expose /api/embeddings compatible avec LightRAG
# ─────────────────────────────────────────────
def make_embedding_func():
    def embed(texts):
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60
            )
            embeddings.append(response.json()["embedding"])
        return embeddings

    return EmbeddingFunc(
        embedding_dim=768,        # nomic-embed-text = 768 dimensions (comme MPNet)
        max_token_size=8192,
        func=embed,
    )


# ─────────────────────────────────────────────
# 4. Ingestion de tous les PDFs
# ─────────────────────────────────────────────
async def ingest_all_documents():
    config = RAGAnythingConfig(
        working_dir="./data/rag_anything_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,    # LLaVA décrira les images
        enable_table_processing=True,    # tableaux → texte structuré
        enable_equation_processing=True, # équations LaTeX → texte
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=make_llm_func(),
        vision_model_func=make_vision_func(),
        embedding_func=make_embedding_func(),
    )

    await rag.process_folder_complete(
        folder_path=DATA_DIR,
        output_dir=OUTPUT_DIR,
        file_extensions=[".pdf"],
        recursive=False,
        max_workers=1,   # 1 seul worker en local pour ne pas surcharger Ollama
    )

    print("✅ Ingestion multimodale terminée.")
    return rag


# ─────────────────────────────────────────────
# 5. Requête
# ─────────────────────────────────────────────
async def query(question: str, mode: str = "hybrid"):
    """
    mode options :
      "naive"  → RAG classique (chunks vectoriels uniquement)
      "local"  → graphe local (entités proches de la question)
      "global" → graphe global (thèmes transversaux)
      "hybrid" → local + global (recommandé)
    """
    config = RAGAnythingConfig(working_dir="./data/rag_anything_storage")

    rag = RAGAnything(
        config=config,
        llm_model_func=make_llm_func(),
        vision_model_func=make_vision_func(),
        embedding_func=make_embedding_func(),
    )

    result = await rag.aquery(question, mode=mode)
    return result


if __name__ == "__main__":
    asyncio.run(ingest_all_documents())
```

---

### Étape 3 — Adapter `api.py` pour cohabitation

Les deux pipelines coexistent. Ajouter un paramètre `engine` :

```python
# Dans QueryRequest
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    engine: str = "faiss"   # "faiss" (actuel) ou "raganything" (nouveau)

# Dans l'endpoint /query
@app.post("/query")
async def query_documents(request: QueryRequest):
    if request.engine == "raganything":
        from src.ingestion.rag_anything_pipeline import query as ra_query
        answer = await ra_query(request.query, mode="hybrid")
        return {"query": request.query, "answer": answer, "engine": "raganything"}
    else:
        # Pipeline actuel FAISS+BM25+Llama (inchangé)
        ...
```

---

### Étape 4 — Structure finale du projet

```
RAG benchmarkV2/
│
├── données rag/              ← PDFs source (inchangé)
│
├── data/
│   ├── vector_store/         ← Index FAISS+BM25 (pipeline actuel)
│   └── rag_anything_storage/ ← Knowledge Graph LightRAG (nouveau)
│       ├── graph_chunk_entity_relation.graphml
│       ├── kv_store_*.json
│       └── vdb_*.json
│
├── output/                   ← Résultats parsing MinerU
│   └── document.pdf/
│       ├── images/           ← Images extraites
│       ├── tables/           ← Tableaux extraits
│       └── content_list.json ← Structure complète
│
├── src/
│   └── ingestion/
│       ├── document_loader.py           ← Pipeline FAISS (actuel)
│       └── rag_anything_pipeline.py     ← Pipeline multimodal (nouveau)
│
├── api.py                    ← Modifié pour supporter les 2 engines
├── ingest_documents.py       ← Pipeline FAISS (inchangé)
└── .env                      ← Config Ollama (nouveau)
```

---

## Comparaison des deux pipelines

```
PIPELINE ACTUEL (FAISS)              PIPELINE RAG-ANYTHING (Ollama)
─────────────────────────            ──────────────────────────────────
PDF → pdfplumber                     PDF → MinerU
  → texte brut                         → texte
                                        → images → LLaVA:13b → description texte
                                        → tableaux → texte structuré
                                        → équations → LaTeX → description
  → chunks (1000 chars)               → entities + relations (Knowledge Graph)
  → MPNet 768d                         → nomic-embed-text 768d
  → FAISS IndexFlatIP                  → LightRAG (vecteurs + graphe Neo4j-like)
  → BM25Okapi
  → Hybrid Search (0.6/0.4)           → aquery(mode="hybrid")
  → Llama 3.1:8b                       → Llama 3.1:8b
  → Réponse                            → Réponse avec contexte multimodal

Coût : 0 €                           Coût : 0 € ✅
Vitesse : ~5-10s/question            Vitesse : ~15-30s/question (LLaVA + graphe)
RAM requise : ~8 Go                  RAM requise : ~16 Go (Llama + LLaVA en même temps)
```

---

## Limitation de LLaVA vs GPT-4o pour les images

LLaVA est open source et tourne localement, mais est **moins précis que GPT-4o** pour décrire des images complexes :

| Type d'image | LLaVA:13b | GPT-4o |
|-------------|-----------|--------|
| Texte dans une image | ✅ Correct | ✅ Excellent |
| Tableaux scannés | ⚠️ Partiel | ✅ Excellent |
| Graphiques / courbes | ⚠️ Basique | ✅ Détaillé |
| Schémas techniques | ⚠️ Basique | ✅ Détaillé |

Pour des documents fiscaux (CGI, BOFIP), les images sont rares → **LLaVA est suffisant**.

---

## Ce qui justifie le passage à RAG-Anything

À considérer **seulement si** tes documents contiennent :

| Contenu | Exemple fiscal | Valeur ajoutée |
|---------|---------------|----------------|
| Tableaux complexes | Barèmes IR, grilles TVA | ⭐⭐⭐ Élevée |
| Schémas/organigrammes | Flux de facturation | ⭐⭐ Moyenne |
| Questions transversales | "Comparer les taux de TVA entre articles" | ⭐⭐⭐ Élevée (Knowledge Graph) |
| Équations mathématiques | Formules de calcul | ⭐ Faible (rare dans le CGI) |

Le **Knowledge Graph** de LightRAG est particulièrement utile pour les questions qui nécessitent de **relier des informations de plusieurs articles** — ce que FAISS gère mal car il cherche par similarité locale.
