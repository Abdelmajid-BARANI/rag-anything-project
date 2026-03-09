# Guide de Démarrage Rapide - RAG Benchmark

## 🚀 Démarrage en 5 Minutes

### 1. Installation d'Ollama et du Modèle

```powershell
# Télécharger Ollama depuis https://ollama.ai
# Ou utiliser winget
winget install Ollama.Ollama

# Télécharger le modèle Llama 3.1:8b
ollama pull llama3.1:8b
```

### 2. Installation des Dépendances Python

```powershell
# Créer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Ingestion des Documents

```powershell
# Charger et indexer les PDFs du dossier "données rag"
python ingest_documents.py
```

**Temps estimé**: 2-5 minutes (selon le nombre de documents)

### 4. Démarrer l'API

```powershell
# Démarrer le serveur API
python api.py
```

L'API sera disponible sur: **http://localhost:8000**

### 5. Tester avec Postman

#### Option A: Importer la Collection
1. Ouvrir Postman
2. Cliquer sur "Import"
3. Sélectionner `postman_collection.json`
4. Lancer les requêtes

#### Option B: Test Manuel

**Health Check:**
```
GET http://localhost:8000/health
```

**Poser une Question:**
```
POST http://localhost:8000/query
Content-Type: application/json

{
  "query": "Quelles sont les règles de TVA?",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 512
}
```

## 📊 Exemples de Questions

- "Quelles sont les obligations de facturation?"
- "Que dit l'article 289 du CGI?"
- "Comment déclarer la TVA?"
- "Quelles sont les sanctions en cas de non-respect?"

## 🔧 Résolution de Problèmes Courants

### Erreur: "Impossible de se connecter à Ollama"
```powershell
# Vérifier qu'Ollama est en cours d'exécution
ollama list

# Si le modèle n'est pas présent
ollama pull llama3.1:8b
```

### Erreur: "Vector store vide"
```powershell
# Exécuter d'abord l'ingestion
python ingest_documents.py
```

### Erreur: "Module introuvable"
```powershell
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

## 📈 Vérifier les Résultats

### Voir les Statistiques
```
GET http://localhost:8000/stats
```

### Documentation Interactive
Accéder à: **http://localhost:8000/docs**

## ✅ Checklist de Démarrage

- [ ] Ollama installé
- [ ] Modèle llama3.1:8b téléchargé
- [ ] Environnement virtuel Python créé
- [ ] Dépendances installées
- [ ] Documents ingérés (vector store créé)
- [ ] API démarrée
- [ ] Test avec Postman réussi

## 🎯 Prochaines Actions

1. **Tester plusieurs questions** pour évaluer la qualité
2. **Ajuster les paramètres** (top_k, temperature) selon les résultats
3. **Analyser les métriques** via `/stats`
4. **Créer un jeu de test** avec questions et réponses attendues
5. **Comparer avec d'autres modèles** (prochaine phase)

## 📞 Support

Si vous rencontrez des problèmes :
1. Vérifier les logs dans `logs/rag_benchmark.log`
2. Consulter le README.md complet
3. Vérifier la configuration dans `config.yaml`
