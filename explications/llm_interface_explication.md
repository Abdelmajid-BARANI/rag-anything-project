# 📖 Explication de `src/generation/llm_interface.py`

---

## C'est quoi ce fichier ?

C'est le **client qui parle à Ollama/Llama**. Il envoie des requêtes HTTP au serveur Ollama (qui tourne en local) et récupère les réponses générées par Llama 3.1:8b.

> **Analogie** : Ollama est un serveur de restaurant, Llama 3.1 est le chef. Ce fichier, c'est le serveur qui prend ta commande et te rapporte le plat.

---

## Architecture de la communication

```
Python (ce fichier)
    │
    │  POST http://localhost:11434/api/generate
    │  {"model": "llama3.1:8b", "prompt": "...", "options": {...}}
    ▼
Ollama (serveur local)
    │
    │  fait tourner Llama 3.1:8b
    ▼
Réponse JSON : {"response": "La réponse du LLM..."}
```

Ollama expose une **API REST locale** — le code lui envoie des requêtes HTTP exactement comme on appellerait une API web distante, sauf que tout tourne sur la même machine.

---

## Constantes en haut du fichier

```python
MAX_RETRIES = 2      # nombre de tentatives en cas d'échec
RETRY_DELAY = 2      # secondes d'attente entre deux tentatives
```

Ces constantes à la racine (pas dans la classe) signifient qu'elles s'appliquent à toute la classe et sont faciles à modifier en un seul endroit.

---

## `__init__` — initialisation

```python
def __init__(self, model="llama3.1:8b", host="http://localhost:11434"):
    self.model = model
    self.host = host
    self.api_url = f"{host}/api/generate"    # endpoint pour la génération
    self.chat_url = f"{host}/api/chat"       # endpoint chat (non utilisé ici)
    self.is_connected = False

    self._check_connection()   # vérifie immédiatement qu'Ollama répond
```

---

## `_check_connection` — vérification de la connexion

```python
def _check_connection(self):
    try:
        response = requests.get(f"{self.host}/api/tags", timeout=5)
        # ↑ GET /api/tags retourne la liste des modèles installés

        if response.status_code == 200:
            self.is_connected = True
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self.model not in model_names:
                logger.warning(f"Le modèle '{self.model}' n'est pas téléchargé...")
                # ↑ Avertit mais ne plante pas — l'init continue
            else:
                logger.success(f"Connexion à Ollama établie, modèle '{self.model}' disponible")

    except requests.exceptions.ConnectionError:
        logger.error("Impossible de se connecter à Ollama sur ...")
        # ↑ Ollama n'est pas lancé — on logue l'erreur mais on ne plante pas
```

Le `timeout=5` est crucial : sans lui, si Ollama n'est pas lancé, le programme resterait bloqué indéfiniment à attendre une réponse qui ne viendra jamais.

---

## `generate` — générer une réponse

C'est la méthode de base, utilisée par toutes les autres.

```python
def generate(self, prompt, temperature=0.7, max_tokens=512, stream=False, timeout=600):
    payload = {
        "model": self.model,              # "llama3.1:8b"
        "prompt": prompt,                 # le texte à envoyer
        "stream": stream,                 # False = réponse d'un bloc, True = flux
        "keep_alive": "10m",              # garde le modèle chargé 10 minutes
        "options": {
            "temperature": temperature,   # 0.0 = déterministe, 2.0 = très créatif
            "num_predict": max_tokens     # nombre max de tokens à générer
        }
    }
```

**`keep_alive: "10m"`** : Ollama décharge le modèle de la RAM après inactivité. En mettant 10 minutes, le modèle reste chargé entre deux requêtes — évite le rechargement de ~5 Go à chaque fois.

**`temperature`** :
- `0.0` → le LLM choisit toujours le mot le plus probable → réponses déterministes et factuelles
- `0.7` → un peu d'aléatoire → réponses plus variées et fluides
- `2.0` → très aléatoire → réponses créatives mais souvent incohérentes

Pour ce projet on utilise `0.2` (réponses factuelles proches du contexte).

### La boucle de retry

```python
for attempt in range(MAX_RETRIES + 1):   # 0, 1, 2 → 3 essais
    try:
        response = requests.post(self.api_url, json=payload, timeout=timeout)
        response.raise_for_status()       # lève une exception si code HTTP >= 400
        result = response.json()
        return result.get("response", "")  # retourne juste le texte généré

    except requests.exceptions.ConnectionError as e:
        last_error = e
        if attempt < MAX_RETRIES:
            logger.warning(f"Tentative {attempt + 1} échouée, retry dans {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        continue

    except requests.exceptions.Timeout:
        ...  # gestion du timeout avec retry
```

`response.raise_for_status()` : si Ollama retourne un code d'erreur HTTP (500, 404...), cette méthode lève automatiquement une exception Python — évite de traiter silencieusement une erreur.

---

## `generate_with_context` — méthode RAG

C'est celle appelée par `api.py`. Elle assemble le contexte et construit le prompt avant d'appeler `generate`.

```python
def generate_with_context(self, query, context_chunks, temperature=0.7, max_tokens=512):
    # 1. Construire le contexte
    context_text = "\n\n".join([
        f"[Document {i+1}]: {chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])
    # Résultat :
    # "[Document 1]: texte du chunk 1\n\n[Document 2]: texte du chunk 2\n\n..."

    # 2. Construire le prompt complet
    prompt = self._build_rag_prompt(query, context_text)

    # 3. Envoyer à Ollama
    response = self.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)

    return {
        "query": query,
        "answer": response,
        "context_chunks": context_chunks,
        "num_chunks_used": len(context_chunks)
    }
```

---

## `_build_rag_prompt` — le prompt système

```python
def _build_rag_prompt(self, query, context):
    prompt = f"""Tu es un assistant juridique spécialisé en droit fiscal français.
Réponds UNIQUEMENT à partir des documents fournis ci-dessous.

RÈGLES STRICTES :
- Réponds EXCLUSIVEMENT avec les informations présentes dans le CONTEXTE ci-dessous.
- Si l'information demandée se trouve dans le contexte, extrais-la et cite le document source.
- Si l'information ne se trouve PAS dans le contexte, dis simplement :
  « L'information demandée n'est pas présente dans les documents fournis. »
- Ne refuse JAMAIS de répondre.
- Sois concis et précis.

CONTEXTE :
{context}

QUESTION : {query}

RÉPONSE :"""
    return prompt
```

Ce prompt est conçu pour :
1. **Contraindre le LLM** à n'utiliser que le contexte fourni → évite les hallucinations
2. **Lui donner un rôle** ("assistant juridique") → oriente le style de réponse
3. **Forcer la citation des sources** → traçabilité des réponses
4. **Le forcer à avouer l'absence d'information** plutôt qu'inventer

---

## Résumé du flux

```
query: "Quels sont les taux de TVA ?"
context_chunks: [chunk1, chunk2, chunk3, chunk4, chunk5]
    │
    ▼ generate_with_context()
    │
    ├── construction du contexte :
    │   "[Document 1]: ...texte chunk1...
    │    [Document 2]: ...texte chunk2..."
    │
    ├── construction du prompt :
    │   "Tu es un assistant... CONTEXTE: ... QUESTION: Quels sont les taux de TVA ?"
    │
    ▼ generate() → POST http://localhost:11434/api/generate
    │
    ▼ Ollama → Llama 3.1:8b → réponse texte
    │
    ▼ {"query": "...", "answer": "Les taux de TVA sont...", ...}
```
