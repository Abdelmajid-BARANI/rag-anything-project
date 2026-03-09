# 📖 Explication de `src/utils/helpers.py`

---

## C'est quoi ce fichier ?

C'est la **boîte à outils partagée** du projet. Il contient 3 fonctions utilitaires que tous les autres fichiers importent. Ce n'est pas un composant métier — il ne fait ni recherche, ni génération, ni évaluation — mais il fournit les fondations communes.

---

## `load_config` — lire le fichier de configuration

```python
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration chargée depuis {config_path}")
    return config
```

### Pourquoi `yaml.safe_load` et pas `yaml.load` ?

`yaml.load()` peut exécuter du code arbitraire intégré dans le YAML si quelqu'un manipule le fichier (`!!python/object`...). `yaml.safe_load()` désactive cette fonctionnalité → plus sécurisé.

### Ce que retourne cette fonction

Le fichier `config.yaml` est transformé en dictionnaire Python imbriqué :

```yaml
# config.yaml
embeddings:
  model_name: "paraphrase-multilingual-mpnet-base-v2"
  dimension: 768
retrieval:
  top_k: 5
```

↓ devient ↓

```python
{
    "embeddings": {
        "model_name": "paraphrase-multilingual-mpnet-base-v2",
        "dimension": 768
    },
    "retrieval": {
        "top_k": 5
    }
}
```

### Comment les autres fichiers l'utilisent

```python
config = load_config("config.yaml")
top_k = config.get("retrieval", {}).get("top_k", 5)
# ↑ le double .get() avec valeur par défaut protège si la clé est absente
```

---

## `setup_logging` — configurer le système de logs

```python
_logging_configured = False   # variable module-level (hors de la fonction)

def setup_logging(log_level="INFO", log_file=None):
    global _logging_configured

    if _logging_configured:
        return   # ← empêche de configurer deux fois (idempotent)
```

**`_logging_configured`** est une variable **au niveau du module** (pas dans une classe). Elle persiste entre les appels à la fonction — c'est ce qu'on appelle un état de module.

Le `global _logging_configured` dans la fonction est nécessaire pour modifier cette variable depuis l'intérieur de la fonction (sinon Python créerait une variable locale du même nom).

### Configuration loguru

```python
logger.remove()   # supprime le handler par défaut de loguru

logger.add(
    lambda msg: print(msg, end=""),   # écrit dans la console sans double saut de ligne
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
```

**Le format de log :**
```
2026-03-03 13:45:12 | INFO     | ingestion.embedder:embed_text - Chargement du modèle...
```

Les balises `<green>`, `<cyan>`, `<level>` sont des balises de couleur loguru → affichage coloré dans le terminal.

**Handler fichier (optionnel) :**
```python
if log_file:
    logger.add(
        log_file,
        level=log_level,
        format="...",       # format sans couleurs pour le fichier texte
        rotation="100 MB"  # crée un nouveau fichier log quand l'actuel dépasse 100 Mo
    )
```

`rotation="100 MB"` : évite que le fichier log grossisse indéfiniment.

---

## `ensure_directories` — créer les dossiers nécessaires

```python
def ensure_directories(directories: list):
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Répertoire créé/vérifié: {directory}")
```

`mkdir(parents=True, exist_ok=True)` :
- `parents=True` : crée tous les dossiers intermédiaires si nécessaire (`./data/vector_store/` crée `data/` puis `data/vector_store/`)
- `exist_ok=True` : ne lève pas d'erreur si le dossier existe déjà

Utilisée dans `ingest_documents.py` :
```python
ensure_directories([
    config.get("vector_store", {}).get("persist_directory", "./data/vector_store"),
    "./logs"
])
```

---

## Pourquoi un fichier `helpers.py` séparé ?

Sans ce fichier, chaque module devrait :
- réimplémenter la lecture du YAML
- reconfigurer loguru à sa façon
- créer les dossiers à sa façon

En centralisant : si on décide de changer le format de log ou de passer de YAML à TOML, on modifie **un seul fichier** au lieu de tous les modules.

---

## Résumé

| Fonction | Appelée par | Ce qu'elle fait |
|----------|-------------|----------------|
| `load_config("config.yaml")` | `api.py`, `ingest_documents.py`, `run_evaluation.py` | Lit et retourne `config.yaml` comme dict Python |
| `setup_logging(level, file)` | `api.py`, `ingest_documents.py` | Configure loguru (console + fichier, une seule fois) |
| `ensure_directories([...])` | `ingest_documents.py` | Crée `data/vector_store/` et `logs/` si absents |
