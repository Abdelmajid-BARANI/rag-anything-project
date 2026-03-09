# 📖 Explication de `src/ingestion/chunker.py`

---

## C'est quoi ce fichier ?

C'est le **découpeur de texte**. Il prend les textes bruts extraits des PDFs (parfois des milliers de caractères) et les coupe en **petits morceaux appelés "chunks"** d'environ 1000 caractères.

**Pourquoi découper ?**
- Le LLM (Llama) a une limite de taille de contexte — on ne peut pas lui envoyer tout un PDF
- FAISS cherche par similarité sur des vecteurs : un vecteur représente mieux un paragraphe précis qu'un document entier
- Des chunks courts = réponses plus précises et sources plus ciblées

> **Analogie** : tu fais des photocopies d'un livre, mais seulement par tranches d'une page à la fois, avec un léger chevauchement entre deux pages consécutives pour ne pas couper une phrase en plein milieu.

---

## Le concept de `chunk_overlap`

```
Texte original : [====A====][====B====][====C====]
                      ↓ avec overlap 200 chars
Chunk 1 : [====A====|==début_B==]
Chunk 2 :         [==fin_A==|====B====|==début_C==]
Chunk 3 :                         [==fin_B==|====C====]
```

Le **chevauchement de 200 caractères** fait que chaque chunk partage une partie avec ses voisins. Ça évite de couper une information importante exactement entre deux chunks.

---

## `RecursiveCharacterTextSplitter` — le séparateur intelligent

Ce n'est pas un découpage bête "tous les 1000 caractères". LangChain essaie de couper **dans un ordre de priorité** :

1. D'abord sur les **paragraphes** (`\n\n`)
2. Puis sur les **lignes** (`\n`)
3. Puis sur les **espaces** (mots)
4. En dernier recours, caractère par caractère

→ Il respecte la structure naturelle du texte autant que possible.

---

## Ligne par ligne

### Le constructeur `__init__`

```python
def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    splitter_kwargs = dict(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,          # mesure la taille en nb de caractères
        is_separator_regex=False,     # les séparateurs sont des chaînes, pas des regex
    )
    if separators is not None:
        splitter_kwargs["separators"] = separators

    self._splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
```

- `length_function=len` : on mesure la taille des chunks en **nombre de caractères** (pas en tokens)
- `**splitter_kwargs` : l'opérateur `**` "déplie" le dictionnaire en arguments nommés — équivalent à écrire `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, ...)`

---

### `chunk_text` — découper un texte

```python
def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
    text = text.strip()     # enlève les espaces/sauts de ligne en début et fin
    if not text:
        return []

    raw_chunks = self._splitter.create_documents([text])
    chunks = []
    for chunk_id, doc in enumerate(raw_chunks):
        chunks.append({
            "text": doc.page_content,          # le texte du chunk
            "chunk_id": chunk_id,              # numéro d'ordre (0, 1, 2...)
            "n_chars": len(doc.page_content),  # taille en caractères
            "metadata": dict(metadata) if metadata else {},
        })
    return chunks
```

- `self._splitter.create_documents([text])` : retourne une liste d'objets LangChain `Document`, chacun ayant un `.page_content`
- `enumerate(raw_chunks)` : donne `(0, chunk0), (1, chunk1), ...` — le numéro et l'objet en même temps

---

### `chunk_documents` — découper une liste de documents

```python
def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
    all_chunks = []

    for doc in documents:
        metadata = dict(doc.get("metadata", {}))
        metadata["filename"] = doc.get("filename", "unknown")
        # ↑ on ajoute le nom du fichier dans les métadonnées de chaque chunk
        # pour savoir plus tard d'où vient chaque chunk

        chunks = self.chunk_text(doc["content"], metadata)
        all_chunks.extend(chunks)     # ajoute tous les chunks à la liste globale

    return all_chunks
```

`all_chunks.extend(chunks)` vs `all_chunks.append(chunks)` :
- `append` ajouterait une **liste dans la liste** : `[[chunk1, chunk2], [chunk3, chunk4]]`
- `extend` **aplatit** : `[chunk1, chunk2, chunk3, chunk4]` ✅

---

### `get_chunk_stats` — statistiques

```python
def get_chunk_stats(self, chunks):
    lengths = [c.get("n_chars", len(c["text"])) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
        "total_chars": sum(lengths),
    }
```

Utile pour comprendre la distribution : si `min_chunk_chars` est très bas (ex: 10), il y a peut-être des chunks quasi-vides.

---

## Ce que produit ce module dans le pipeline

```
Document "CGI_Article 289.pdf" (10 000 caractères)
    │
    ▼ chunk_documents()
    
[
  {"text": "Code général des impôts\nArticle 289...", "chunk_id": 0, "n_chars": 987,
   "metadata": {"filename": "CGI_Article 289.pdf", "source": "...", "pages": 12}},

  {"text": "...suite de l'article 289 avec overlap...", "chunk_id": 1, "n_chars": 1000,
   "metadata": {"filename": "CGI_Article 289.pdf", ...}},

  ...
]
```

Cette liste est ensuite passée à `BERTEmbedder.embed_chunks()`.
