# 📖 Explication de `src/ingestion/document_loader.py`

---

## C'est quoi ce fichier ?

C'est le **lecteur de PDFs** du projet. Il ouvre les fichiers PDF du dossier `données rag/`, extrait tout le texte page par page, et retourne ce texte sous forme de dictionnaires Python prêts à être utilisés par le reste du pipeline.

> **Analogie** : c'est le bibliothécaire qui ouvre chaque livre, recopie tout le texte, et pose les feuilles sur la table pour qu'on puisse les découper ensuite.

---

## Structure du fichier

```
DocumentLoader
│
├── __init__(data_dir)         ← vérifie que le dossier existe
├── load_pdf(pdf_path)         ← lit UN fichier PDF
├── load_all_pdfs()            ← lit TOUS les PDFs du dossier
└── get_document_stats(docs)   ← calcule des statistiques
```

---

## Ligne par ligne

### Le constructeur `__init__`

```python
def __init__(self, data_dir: str):
    self.data_dir = Path(data_dir)
    if not self.data_dir.exists():
        raise ValueError(f"Le répertoire {data_dir} n'existe pas")
```

- `Path(data_dir)` : convertit la chaîne `"./données rag"` en objet `Path` — ça permet d'utiliser `.exists()`, `.glob()` etc. proprement
- Si le dossier n'existe pas, on lève immédiatement une `ValueError` pour éviter une erreur bizarre plus tard

---

### `load_pdf` — lire un seul PDF

```python
def load_pdf(self, pdf_path: Path) -> Dict[str, str]:
    reader = PdfReader(pdf_path)       # ouvre le PDF avec pypdf
    text = ""

    for page in reader.pages:          # parcourt chaque page
        text += page.extract_text() + "\n"   # extrait le texte + saut de ligne

    return {
        "filename": pdf_path.name,     # ex: "CGI_Article 289.pdf"
        "content": text,               # tout le texte du PDF
        "metadata": {
            "source": str(pdf_path),   # chemin complet
            "pages": len(reader.pages) # nombre de pages
        }
    }
```

**Ce que retourne cette fonction :**

```python
{
    "filename": "CGI_Article 289.pdf",
    "content": "Code général des impôts\nArticle 289\n...",
    "metadata": {
        "source": "./données rag/CGI_Article 289.pdf",
        "pages": 12
    }
}
```

⚠️ **Limite de `extract_text()`** : pypdf extrait le texte brut — les tableaux, images et formules ne sont pas extraits fidèlement.

---

### `load_all_pdfs` — lire tout le dossier

```python
def load_all_pdfs(self) -> List[Dict[str, str]]:
    documents = []
    pdf_files = list(self.data_dir.glob("*.pdf"))   # cherche tous les *.pdf
```

`self.data_dir.glob("*.pdf")` retourne un itérateur sur tous les fichiers `.pdf` du dossier.
`list(...)` le convertit en liste pour pouvoir en compter le nombre.

```python
    for pdf_path in pdf_files:
        try:
            doc = self.load_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Impossible de charger {pdf_path.name}: {e}")
            continue   # ← passe au PDF suivant sans planter tout le programme
```

Le `try/except` avec `continue` est important : si un PDF est corrompu, le programme continue avec les autres au lieu de s'arrêter.

---

### `get_document_stats` — statistiques

```python
def get_document_stats(self, documents):
    total_chars = sum(len(doc["content"]) for doc in documents)
    total_words = sum(len(doc["content"].split()) for doc in documents)
    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_chars_per_doc": total_chars / len(documents) if documents else 0,
        "avg_words_per_doc": total_words / len(documents) if documents else 0
    }
```

- `sum(len(doc["content"]) for doc in documents)` → **generator expression** : parcourt tous les docs et additionne les longueurs — équivalent d'une boucle `for` mais en une ligne
- `.split()` sépare le texte par espaces → donne une liste de mots → `len()` compte les mots

---

## Ce que produit ce module dans le pipeline

```
"données rag/"
    │  CGI_Article 289.pdf  (12 pages)
    │  CGI_Article 242 nonies.pdf  (25 pages)
    │  ...
    │
    ▼ load_all_pdfs()
  
[
  {"filename": "CGI_Article 289.pdf",       "content": "...", "metadata": {...}},
  {"filename": "CGI_Article 242 nonies.pdf", "content": "...", "metadata": {...}},
  ...
]
```

Cette liste est ensuite passée au `DocumentChunker`.
