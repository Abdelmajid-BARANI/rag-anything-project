# 📖 Explication de `ingest_documents.py`

---

## C'est quoi ce fichier ?

C'est le **script de préparation** — il se lance **une seule fois** (ou à chaque fois qu'on ajoute de nouveaux PDFs) pour :
1. Lire les PDFs
2. Les découper
3. Générer les vecteurs
4. Construire et sauvegarder l'index FAISS + BM25

Sans ce script, l'API n'a rien à chercher. `ingest_documents.py` est à lancer **avant** `api.py`.

> **Analogie** : c'est l'archiviste qui lit tous les documents, les découpe, les classe et range l'index dans le tiroir. Après, n'importe qui peut venir chercher rapidement grâce à cet index.

---

## Les 5 étapes du pipeline d'ingestion

### Étape 0 — Charger la configuration

```python
config = load_config("config.yaml")
setup_logging(
    log_level=config.get("logging", {}).get("level", "INFO"),
    log_file=config.get("logging", {}).get("file")
)
ensure_directories([
    config.get("vector_store", {}).get("persist_directory", "./data/vector_store"),
    "./logs"
])
```

Première chose : tout lire depuis `config.yaml` pour que les paramètres soient centralisés. On crée aussi les dossiers de sortie si absents.

---

### Étape 1 — Charger les documents PDF

```python
logger.info("\n[1/5] Chargement des documents PDF...")
data_source = config.get("ingestion", {}).get("data_source", "./données rag")
loader = DocumentLoader(data_source)
documents = loader.load_all_pdfs()

if not documents:
    logger.error("Aucun document chargé. Arrêt du processus.")
    return   # ← sort de main() proprement si dossier vide
```

**Sortie de cette étape :**
```python
[
  {"filename": "CGI_Article 289.pdf",        "content": "...", "metadata": {...}},
  {"filename": "CGI_Article 242 nonies.pdf", "content": "...", "metadata": {...}},
  ...
]
```

---

### Étape 2 — Découper en chunks

```python
logger.info("\n[2/5] Découpage des documents en chunks...")
chunk_size   = config.get("ingestion", {}).get("chunk_size", 1000)
chunk_overlap = config.get("ingestion", {}).get("chunk_overlap", 200)

chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = chunker.chunk_documents(documents)
```

Valeurs depuis `config.yaml` : `chunk_size=1000`, `chunk_overlap=200`.

**Sortie de cette étape :**
```python
[
  {"text": "Code général des impôts\nArticle 289...", "chunk_id": 0,
   "metadata": {"filename": "CGI_Article 289.pdf", ...}},
  {"text": "...suite avec overlap...",                "chunk_id": 1, ...},
  ...  # quelques centaines de chunks
]
```

---

### Étape 3 — Générer les embeddings

```python
logger.info("\n[3/5] Génération des embeddings avec BERT...")
embedding_config = config.get("embeddings", {})
embedder = BERTEmbedder(
    model_name=embedding_config.get("model_name", "bert-base-multilingual-cased"),
    device=embedding_config.get("device", "cpu")
)

enriched_chunks = embedder.embed_chunks(chunks, batch_size=32)
```

**C'est l'étape la plus lente** : chaque chunk est transformé en vecteur de 768 nombres. Avec plusieurs centaines de chunks sur CPU, ça prend quelques minutes.

**Sortie de cette étape :**
```python
[
  {"text": "...", "chunk_id": 0, "metadata": {...},
   "embedding": array([0.23, -0.11, ..., 0.31])},  # 768 nombres
  ...
]
```

---

### Étape 4 — Créer et remplir le vector store

```python
logger.info("\n[4/5] Création du vector store FAISS...")
vector_store_config = config.get("vector_store", {})
vector_store = FAISSVectorStore(
    embedding_dim=embedder.get_embedding_dimension(),   # 768
    persist_directory=vector_store_config.get("persist_directory", "./data/vector_store")
)

vector_store.add_chunks(enriched_chunks)
# ↑ normalise L2 les vecteurs, les ajoute à l'index FAISS, construit l'index BM25
```

---

### Étape 5 — Sauvegarder sur disque

```python
logger.info("\n[5/5] Sauvegarde du vector store...")
vector_store.save()
# ↑ écrit :
#   data/vector_store/faiss_index.index  (index vectoriel binaire FAISS)
#   data/vector_store/faiss_index.pkl    (textes des chunks en pickle Python)
```

**Note :** L'index BM25 n'est pas sauvegardé — il sera reconstruit à chaque chargement depuis les textes pickle. C'est rapide (quelques secondes).

---

## La fonction `test_search`

```python
def test_search():
    """Fonction de test pour vérifier la recherche"""
    # Charge l'embedder et le vector store
    ...
    vector_store.load()

    test_query = "Qu'est-ce que la TVA?"
    query_embedding = embedder.embed_text(test_query)
    results = vector_store.search(query_embedding, top_k=3)

    for i, result in enumerate(results):
        logger.info(f"[{i+1}] Score: {result['score']:.4f}")
        logger.info(f"Texte: {result['text'][:200]}...")
```

Activée avec `python ingest_documents.py --test`.

---

## Le bloc `if __name__ == "__main__"`

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion de documents RAG")
    parser.add_argument("--test", action="store_true",
                        help="Tester la recherche après l'ingestion")
    args = parser.parse_args()

    main()   # toujours exécuter l'ingestion

    if args.test:
        test_search()   # optionnellement tester la recherche
```

**`argparse`** : gère les arguments de la ligne de commande.  
`--test` est un **flag booléen** : `action="store_true"` signifie que si `--test` est présent dans la commande, `args.test` vaut `True`, sinon `False`.

```bash
python ingest_documents.py          # ingestion seule
python ingest_documents.py --test   # ingestion + vérification de la recherche
```

---

## Résumé du flux complet

```
python ingest_documents.py
    │
    │ config.yaml
    ▼
1. DocumentLoader.load_all_pdfs()
       données rag/*.pdf  →  texte brut

    ▼
2. DocumentChunker.chunk_documents()
       texte brut  →  chunks (~1000 chars, overlap 200)

    ▼
3. BERTEmbedder.embed_chunks()
       chunks  →  chunks + vecteurs 768d
       (modèle MPNet, batch=32, peut prendre 2-5 min sur CPU)

    ▼
4. FAISSVectorStore.add_chunks()
       normalisation L2 + ajout à IndexFlatIP + construction BM25

    ▼
5. FAISSVectorStore.save()
       → data/vector_store/faiss_index.index
       → data/vector_store/faiss_index.pkl

    ✅ PRÊT — on peut lancer api.py
```

---

## Ce qui se passe si on reouvre `api.py` sans avoir ingéré

```python
# Dans lifespan (api.py) :
try:
    vector_store.load()
    logger.success("Vector store chargé depuis le disque")
except FileNotFoundError:
    logger.warning("Aucun index FAISS trouvé. Veuillez d'abord ingérer des documents.")
```

L'API démarre quand même mais `/query` retournera une erreur 503 :
```json
{"detail": "Vector store vide. Veuillez d'abord ingérer des documents via le script d'ingestion."}
```

La solution : lancer `python ingest_documents.py` (ou `run_ingest.bat`) avant l'API.
