# 📖 Explication de `src/ingestion/embedder.py`

---

## C'est quoi ce fichier ?

C'est le **traducteur texte → vecteur numérique**. Il prend une phrase ou un paragraph en français, et la transforme en une liste de **768 nombres** (un vecteur) qui représente le "sens" de ce texte dans un espace mathématique.

**Pourquoi transformer en vecteurs ?**  
Les ordinateurs ne comprennent pas le texte directement. En les convertissant en vecteurs, on peut calculer une **distance mathématique** entre deux textes. Deux textes similaires auront des vecteurs proches l'un de l'autre.

> **Analogie** : imagine que chaque phrase est un point dans l'espace. Les phrases sur la "TVA" sont regroupées dans un coin, celles sur "les factures électroniques" dans un autre coin. Un vecteur, c'est juste les coordonnées de ce point.

---

## Le modèle utilisé : MPNet multilingue

```python
model_name = "paraphrase-multilingual-mpnet-base-v2"
```

Ce modèle a été entraîné sur des milliards de paires de phrases pour apprendre à placer des textes similaires **proches** dans l'espace vectoriel.

- **Multilingue** : comprend le français, l'anglais, l'espagnol et 50+ langues
- **768 dimensions** : chaque texte devient un vecteur de 768 nombres
- **MPNet** : architecture plus performante que BERT classique pour la similarité sémantique

---

## Ligne par ligne

### Le constructeur `__init__`

```python
def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # ↑ Si une carte graphique NVIDIA est disponible → utilise le GPU (beaucoup plus rapide)
    # Sinon → CPU (plus lent mais fonctionne partout)

    self.model = SentenceTransformer(model_name, device=device)
    self.embedding_dim = self.model.get_sentence_embedding_dimension()
    # ↑ récupère la dimension (768 pour MPNet, 384 pour MiniLM)
```

`torch.cuda.is_available()` : retourne `True` si PyTorch détecte un GPU compatible CUDA, `False` sinon.

---

### `embed_text` — encoder une seule phrase

```python
def embed_text(self, text: str) -> np.ndarray:
    embedding = self.model.encode(text, convert_to_numpy=True)
    return embedding
```

- `self.model.encode(text)` : passe le texte dans le réseau de neurones → retourne un vecteur
- `convert_to_numpy=True` : retourne un tableau numpy au lieu d'un tenseur PyTorch (plus simple à manipuler)
- **Retourne** : un `np.ndarray` de forme `(768,)` — un vecteur de 768 nombres décimaux

**Exemple (simplifié) :**
```python
embed_text("Quels sont les taux de TVA ?")
# → [0.23, -0.11, 0.45, 0.07, ..., 0.31]  ← 768 nombres
```

---

### `embed_batch` — encoder plusieurs textes d'un coup

```python
def embed_batch(self, texts: List[str], batch_size=32, show_progress=True) -> np.ndarray:
    embeddings = self.model.encode(
        texts,
        batch_size=batch_size,             # traite 32 textes à la fois
        show_progress_bar=show_progress,   # affiche une barre de progression
        convert_to_numpy=True
    )
    return embeddings
```

**Pourquoi un `batch_size` ?**  
Si on a 500 chunks à encoder, envoyer les 500 en une seule fois peut dépasser la mémoire RAM/GPU. On envoie par paquets de 32 — assez grand pour être efficace, assez petit pour tenir en mémoire.

**Retourne** : une matrice `np.ndarray` de forme `(N, 768)` où N = nombre de textes.

```
         dim 0    dim 1   dim 2  ... dim 767
Chunk 0 [ 0.23,  -0.11,   0.45, ...,  0.31 ]
Chunk 1 [ 0.18,   0.07,  -0.22, ...,  0.55 ]
Chunk 2 [ 0.41,  -0.33,   0.12, ..., -0.08 ]
...
Chunk N [ ... ]
```

---

### `embed_chunks` — encoder une liste de chunks (dictionnaires)

```python
def embed_chunks(self, chunks: List[Dict], batch_size=32) -> List[Dict]:
    texts = [chunk["text"] for chunk in chunks]   # extrait juste les textes

    embeddings = self.embed_batch(texts, batch_size=batch_size)

    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        enriched_chunk = chunk.copy()            # copie pour ne pas modifier l'original
        enriched_chunk["embedding"] = embeddings[i]  # ajoute le vecteur
        enriched_chunks.append(enriched_chunk)

    return enriched_chunks
```

**Ce que ça fait :**  
Chaque chunk (qui était un dict avec `"text"`, `"chunk_id"`, `"metadata"`) reçoit une nouvelle clé `"embedding"` contenant son vecteur de 768 nombres.

**Avant `embed_chunks` :**
```python
{"text": "Article 289...", "chunk_id": 0, "metadata": {...}}
```

**Après `embed_chunks` :**
```python
{"text": "Article 289...", "chunk_id": 0, "metadata": {...},
 "embedding": array([0.23, -0.11, 0.45, ..., 0.31])}  # 768 nombres
```

---

### `get_embedding_dimension`

```python
def get_embedding_dimension(self) -> int:
    return self.embedding_dim   # retourne 768
```

Utilisée par `FAISSVectorStore` pour savoir combien de dimensions son index doit avoir.

---

## Ce que produit ce module dans le pipeline

```
[chunks sans vecteurs]
    │
    ▼ embed_chunks()
    
[chunks enrichis avec vecteurs]
    │
    ▼ → FAISSVectorStore.add_chunks()
```

Les vecteurs sont ensuite **normalisés L2** par FAISS avant d'être indexés — c'est la normalisation qui permet de calculer la similarité cosinus via un simple produit scalaire.
