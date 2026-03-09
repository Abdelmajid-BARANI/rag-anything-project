# RAG-Anything — Explication complète du projet

---

## Vue d'ensemble

RAG-Anything est un **pipeline RAG multimodal** qui combine deux technologies :

- **MinerU** : parser de documents avancé (texte, images, tableaux, équations)
- **LightRAG** : moteur de Knowledge Graph (graphe de connaissance) basé sur des vecteurs

Il remplace ou coexiste avec le pipeline FAISS+BM25 existant, et tourne **100% en local** via Ollama — sans clé API, sans coût.

---

## Architecture globale

```
PDFs (données rag/)
       │
       ▼
  ┌─────────────┐
  │   MinerU    │  ← Parser intelligent (OCR, VLM optionnel)
  └──────┬──────┘
         │ content_list.json
         │  (texte + images + tableaux structurés)
         ▼
  ┌─────────────────────┐
  │   LightRAG          │
  │  ┌───────────────┐  │
  │  │ Llama 3.1:8b  │  │  ← Extrait entités et relations
  │  │  (via Ollama) │  │
  │  └───────────────┘  │
  │  ┌───────────────┐  │
  │  │ nomic-embed   │  │  ← Vectorise les chunks
  │  │  (via Ollama) │  │
  │  └───────────────┘  │
  └──────────┬──────────┘
             │
             ▼
  ┌────────────────────────────┐
  │  data/rag_anything_storage │
  │  ├── graph_chunk_entity_   │  ← Knowledge Graph (.graphml)
  │  │   relation.graphml      │
  │  ├── kv_store_*.json       │  ← Cache des chunks
  │  └── vdb_*.json            │  ← Index vectoriel
  └────────────────────────────┘
             │
             ▼
      Requête utilisateur
      aquery(mode="hybrid")
             │
             ▼
      Llama 3.1:8b → Réponse
```

---

## Composants détaillés

### 1. MinerU — Le parser

MinerU remplace `pdfplumber`. Il est beaucoup plus puissant :

| Fonctionnalité | pdfplumber | MinerU (mode txt) | MinerU (mode auto) |
|---------------|-----------|-------------------|-------------------|
| Texte brut | ✅ | ✅ | ✅ |
| Tableaux structurés | ⚠️ Basique | ✅ | ✅ |
| Images extraites | ❌ | ❌ | ✅ |
| Équations LaTeX | ❌ | ❌ | ✅ |
| VRAM nécessaire | 0 | 0 | ~8 GB |
| Vitesse | Rapide | Rapide | Lent (CPU) |

**Mode utilisé ici** : `txt` (défini dans `.env`) — texte + tableaux, sans VLM, adapté si peu de VRAM GPU.

MinerU télécharge ~500MB de modèles OCR (PaddleOCR) au **premier lancement uniquement**, puis les réutilise depuis le cache.

---

### 2. LightRAG — Le Knowledge Graph

LightRAG est le cœur de l'intelligence du système. Il ne fait pas qu'indexer des chunks de texte — il **crée un graphe de connaissances** :

```
                ┌─────────────┐
                │  Article 289│
                └──────┬──────┘
                       │ "oblige"
              ┌────────┴────────┐
              ▼                 ▼
    ┌──────────────┐    ┌──────────────┐
    │  Entreprises │    │  Factures    │
    │  assujetties │    │ électroniques│
    └──────────────┘    └──────┬───────┘
                               │ "format"
                        ┌──────┴──────┐
                     UBL/CII     Factur-X
```

Ce graphe permet de répondre à des questions **transversales** entre plusieurs documents, ce que FAISS ne peut pas faire.

---

### 3. Les modèles Ollama

| Rôle | Modèle | Usage |
|------|--------|-------|
| **LLM principal** | `llama3.1:8b` | Génère les réponses, extrait entités/relations |
| **Vision** | `llava:7b` | Décrit les images (si mode `auto`) |
| **Embeddings** | `nomic-embed-text` | Vectorise texte (768 dimensions) |

Tous tournent localement via Ollama sur `http://localhost:11434/v1` (API compatible OpenAI).

---

### 4. Les 4 modes de requête

LightRAG propose 4 modes pour interroger la base :

```
mode="naive"   → RAG classique : recherche par similarité vectorielle seule
                 Similaire à FAISS. Bon pour questions précises.

mode="local"   → Graphe local : explore les entités proches de la question
                 Bon pour : "Que dit l'article X ?"

mode="global"  → Graphe global : analyse les thèmes transversaux
                 Bon pour : "Comparer les règles de TVA entre articles"

mode="hybrid"  → local + global (recommandé)
                 Meilleure couverture, légèrement plus lent
```

---

## Structure des fichiers

```
RAG benchmark_RAG-AnythingV3/
│
├── .env                              ← Configuration (modèles, chemins, mode MinerU)
│
├── src/ingestion/
│   └── rag_anything_pipeline.py     ← Pipeline complet (ingestion + requête)
│
├── data/
│   ├── vector_store/                ← Index FAISS (pipeline actuel)
│   └── rag_anything_storage/        ← Knowledge Graph LightRAG (nouveau)
│       ├── graph_chunk_entity_relation.graphml
│       ├── kv_store_full_docs.json
│       ├── kv_store_text_chunks.json
│       └── vdb_chunks.json
│
├── output/                          ← Résultats parsing MinerU
│   └── document.pdf/
│       ├── content_list.json        ← Structure complète extraite
│       └── images/                  ← Images (si mode auto)
│
└── api.py                           ← API avec dual engine (faiss / raganything)
```

---

## Configuration `.env`

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llava:7b
OLLAMA_EMBED_MODEL=nomic-embed-text

# MinerU
OUTPUT_DIR=./output
PARSER=mineru
PARSE_METHOD=txt      # txt (rapide) ou auto (multimodal, nécessite VRAM)

# Miroir HuggingFace
HF_ENDPOINT=https://hf-mirror.com   # évite les timeouts réseau
```

---

## Utilisation

### Ingestion (premier lancement)

```bash
# Via le script Python directement
venv\Scripts\python.exe src\ingestion\rag_anything_pipeline.py

# Via l'API (après démarrage du serveur)
POST http://localhost:8000/raganything/ingest
```

**Durée estimée** :
- 4 PDFs fiscaux (~50 pages chacun) : 5-30 minutes selon le CPU
- Les lancements suivants ne réingèrent pas si les fichiers sont déjà traités

### Vérification de l'état

```bash
GET http://localhost:8000/raganything/status
```

### Requêtes via l'API

```json
POST /query
{
  "query": "Quelles sont les obligations de facturation électronique ?",
  "engine": "raganything",
  "rag_mode": "hybrid"
}
```

---

## Comparaison FAISS vs RAG-Anything

| Critère | Pipeline FAISS | RAG-Anything |
|---------|---------------|--------------|
| **Parser** | pdfplumber (texte brut) | MinerU (texte + tableaux + images*) |
| **Index** | FAISS + BM25 | Knowledge Graph + vecteurs |
| **Recherche** | Similarité cosinus | Graphe d'entités + vecteurs |
| **Questions transversales** | ⚠️ Limité | ✅ Excellent |
| **Vitesse ingestion** | ~1 min | 5-30 min |
| **Vitesse requête** | ~5-10s | ~15-30s |
| **RAM** | ~6 GB | ~8 GB |
| **GPU** | Non requis | Non requis (mode txt) |
| **Coût** | 0 € | 0 € |

*Images disponibles uniquement en mode `auto`

---

## Problèmes courants et solutions

| Problème | Cause | Solution |
|----------|-------|----------|
| `Fetching 14 files` au démarrage | MinerU télécharge ses modèles OCR (1 seule fois) | Attendre, ou utiliser `HF_ENDPOINT=https://hf-mirror.com` |
| `Read timed out` sur HuggingFace | Connexion lente vers `hf.co` | `HF_ENDPOINT=https://hf-mirror.com` dans `.env` |
| `gpu_memory: 1GB, batch_size: 1` | MinerU essaie d'utiliser le GPU en mode `auto` | Passer `PARSE_METHOD=txt` dans `.env` |
| `ModuleNotFoundError: raganything` | Package non installé ou mauvais venv | `venv\Scripts\pip.exe install raganything lightrag-hku mineru` |
| Ingestion lente | CPU seul, LLM qui extrait le graphe | Normal, ~1-5 min/PDF. Lancer en arrière-plan. |
