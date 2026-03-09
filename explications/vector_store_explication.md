# 📖 Explication de `src/retrieval/vector_store.py`

---

## C'est quoi ce fichier ?

C'est le **moteur de recherche** du projet. Il stocke tous les vecteurs des chunks et, quand on pose une question, il trouve les chunks les plus "proches" de cette question.

Il combine **deux méthodes de recherche** :
- **FAISS** : cherche par similarité sémantique (le sens)
- **BM25** : cherche par mots-clés (les termes exacts)

> **Analogie FAISS** : chercher dans une bibliothèque par thème — "je cherche quelque chose sur la fiscalité" → on vous guide vers le bon rayon.  
> **Analogie BM25** : chercher dans l'index d'un livre — "je cherche le mot exact TVA" → pages 12, 45, 103.

---

## Structure de la classe

```
FAISSVectorStore
│
├── __init__()              ← crée l'index FAISS vide
├── add_chunks()            ← ajoute des vecteurs à l'index
│
├── _tokenize()             ← découpe un texte en mots pour BM25
├── _build_bm25_index()     ← construit l'index BM25 en mémoire
│
├── search()                ← recherche sémantique seule (FAISS)
├── bm25_search()           ← recherche lexicale seule (BM25)
├── hybrid_search()         ← ★ fusion des deux (utilisée en production)
│
├── save()                  ← sauvegarde sur disque
├── load()                  ← charge depuis le disque
├── get_stats()             ← statistiques
└── clear()                 ← vide l'index
```

---

## 1. `__init__` — initialisation

```python
def __init__(self, embedding_dim: int, persist_directory="./data/vector_store"):
    self.embedding_dim = embedding_dim   # 768 pour MPNet
    self.persist_directory = Path(persist_directory)
    self.persist_directory.mkdir(parents=True, exist_ok=True)

    self.index = faiss.IndexFlatIP(embedding_dim)
    # ↑ "IndexFlatIP" = index plat avec produit scalaire (Inner Product)
    # Sur des vecteurs normalisés L2, le produit scalaire = similarité cosinus

    self.chunks = []      # liste des textes originaux (sans les vecteurs)
    self.bm25 = None      # index BM25 (construit après add_chunks)
    self._tokenized_corpus = []
```

**Pourquoi `IndexFlatIP` et pas `IndexFlatL2` ?**

| Index | Mesure | Quand l'utiliser |
|-------|--------|-----------------|
| `IndexFlatL2` | distance euclidienne | espace géométrique classique |
| `IndexFlatIP` | produit scalaire | vecteurs normalisés → similarité cosinus |

Sur des vecteurs normalisés à longueur 1, le produit scalaire = cosinus de l'angle entre deux vecteurs. Plus l'angle est petit (vecteurs dans la même direction), plus les textes sont similaires.

---

## 2. `add_chunks` — ajouter des chunks

```python
def add_chunks(self, chunks: List[Dict]):
    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype('float32')
    # ↑ extrait tous les vecteurs et les met dans une matrice numpy float32
    # (FAISS exige float32, pas float64)

    faiss.normalize_L2(embeddings)
    # ↑ normalise chaque vecteur à longueur 1 (division par sa norme)
    # Nécessaire pour que le produit scalaire = similarité cosinus

    self.index.add(embeddings)
    # ↑ ajoute tous les vecteurs à l'index FAISS

    for chunk in chunks:
        chunk_copy = chunk.copy()
        del chunk_copy["embedding"]    # supprime le vecteur pour économiser la RAM
        self.chunks.append(chunk_copy)

    self._build_bm25_index()           # reconstruit l'index BM25 avec les nouveaux chunks
```

**Pourquoi supprimer l'embedding du chunk stocké ?**  
Les vecteurs sont déjà dans l'index FAISS — les garder aussi dans `self.chunks` doublerait l'utilisation mémoire inutilement.

---

## 3. `_tokenize` et `_build_bm25_index`

```python
@staticmethod
def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'[A-Za-zÀ-ÿ0-9/]{2,}', text)]
    # Extrait tous les mots de ≥ 2 caractères, en minuscules
    # "Article 289 du CGI." → ["article", "289", "du", "cgi"]
```

`re.findall(r'[A-Za-zÀ-ÿ0-9/]{2,}', text)` : expression régulière qui capture les séquences de lettres (y compris accentuées), chiffres et `/` d'au moins 2 caractères.

```python
def _build_bm25_index(self):
    self._tokenized_corpus = [self._tokenize(c.get("text", "")) for c in self.chunks]
    # ↑ tokenise TOUS les chunks → liste de listes de mots
    # [["article", "289", ...], ["tva", "facture", ...], ...]

    self.bm25 = BM25Okapi(self._tokenized_corpus)
    # ↑ BM25Okapi calcule automatiquement les fréquences et longueurs des documents
```

---

## 4. `search` — recherche sémantique FAISS seule

```python
def search(self, query_embedding: np.ndarray, top_k=5) -> List[Dict]:
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_vector)   # normalize la requête aussi

    scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
    # ↑ retourne les top_k indices et leurs scores (produit scalaire = cosinus)
    # scores : ex. [[0.91, 0.85, 0.72, 0.68, 0.55]]
    # indices : ex. [[42, 7, 18, 3, 99]]

    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        result = self.chunks[idx].copy()
        result["score"] = float(score)          # similarité cosinus ∈ [-1, 1]
        result["similarity"] = float(score)
        result["rank"] = i + 1
        results.append(result)

    return results
```

`zip(scores[0], indices[0])` couple les scores et indices : `[(0.91, 42), (0.85, 7), ...]`.

---

## 5. `hybrid_search` — ★ la vraie méthode utilisée en production

C'est l'algorithme le plus complexe du fichier. Il combine FAISS et BM25 en 5 étapes :

### Étape 1 : Sur-récupération FAISS

```python
n_candidates = min(top_k * candidate_factor, self.index.ntotal)
# top_k=5, candidate_factor=4 → récupère 20 candidats FAISS
semantic_results = self.search(query_embedding, top_k=n_candidates)
```

On récupère plus que nécessaire (20 au lieu de 5) pour avoir un meilleur bassin de candidats.

### Étape 2 : Scores BM25 sur tout le corpus

```python
tokenized_query = self._tokenize(query_text)
bm25_scores_all = self.bm25.get_scores(tokenized_query)
# ↑ retourne un score BM25 pour CHAQUE chunk du corpus (pas seulement le top)
```

### Étape 3 : Union des candidats

```python
bm25_top_indices = set(np.argsort(bm25_scores_all)[::-1][:n_candidates].tolist())
all_candidate_indices = set(sem_scores_map.keys()) | bm25_top_indices
# ↑ l'opérateur | fait l'union des deux ensembles
# On garde tous les chunks qui apparaissent dans FAISS OU dans BM25
```

### Étape 4 : Normalisation min-max et score hybride

```python
# Normalise chaque score de 0 à 1 pour pouvoir les combiner
sem_norm = (sem_score - sem_min) / (sem_max - sem_min)
bm25_norm = (bm25_score - bm25_min) / (bm25_max - bm25_min)

hybrid_score = alpha * sem_norm + (1 - alpha) * bm25_norm
#            = 0.6 * sem_norm + 0.4 * bm25_norm
```

**Pourquoi normaliser ?** Les scores FAISS sont entre -1 et 1, les scores BM25 peuvent être 0, 3.5, 12.7... Il faut les mettre sur la même échelle avant de les fusionner.

### Étape 5 : Tri final

```python
scored_candidates.sort(key=lambda r: r["score"], reverse=True)
final = scored_candidates[:top_k]   # garde les 5 meilleurs
```

---

## 6. `save` et `load` — persistance sur disque

```python
def save(self, filename="faiss_index"):
    # Sauvegarde l'index vectoriel FAISS (format binaire propre à FAISS)
    faiss.write_index(self.index, str(index_path))         # → faiss_index.index
    # Sauvegarde les textes des chunks (format pickle Python)
    pickle.dump(self.chunks, f)                            # → faiss_index.pkl

def load(self, filename="faiss_index"):
    self.index = faiss.read_index(str(index_path))         # charge .index
    self.chunks = pickle.load(f)                           # charge .pkl
    self._build_bm25_index()                               # reconstruit BM25 en mémoire
```

**Note** : BM25 n'est pas sauvegardé sur disque — il est **reconstruit à chaque chargement** depuis les textes des chunks. C'est rapide et évite un fichier supplémentaire.

---

## Résumé visuel

```
add_chunks()
    │  vecteurs (float32, normalisés L2)
    ▼
FAISS IndexFlatIP  ←→  save/load  ←→  faiss_index.index
    │
    │  textes des chunks
    ▼
self.chunks []     ←→  save/load  ←→  faiss_index.pkl
    │
    ▼ _build_bm25_index()
BM25Okapi (RAM uniquement)

─────────────────────────────────

hybrid_search(vecteur_question, texte_question)
    │
    ├── FAISS → top 20 candidats sémantiques
    ├── BM25  → scores lexicaux sur tout le corpus
    ├── Union + normalisation min-max
    ├── score = 0.6 × FAISS + 0.4 × BM25
    └── Top 5 résultats retournés
```
