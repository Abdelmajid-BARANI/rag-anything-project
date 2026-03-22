"""
Pipeline RAG-Anything — 100% local via Ollama

Fonctionnalités:
  - Ingestion de documents (PDF, Office, images) via MinerU / Docling / PaddleOCR
  - Construction d'un Knowledge Graph multimodal (texte, images, tableaux, équations)
  - Requêtes texte (aquery), VLM-enhanced, multimodales (aquery_with_multimodal)
  - Insertion directe de content-list pré-parsée
  - 100% local : LLM + Vision + Embeddings via Ollama
"""
import asyncio
import json
import os
import re
import sys
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger

# Ajouter le dossier Scripts du venv au PATH pour que subprocess trouve "mineru"
_venv_scripts = Path(sys.executable).parent
if str(_venv_scripts) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_venv_scripts) + os.pathsep + os.environ.get("PATH", "")

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.prompt import PROMPTS
import raganything.utils as _rag_utils

# ─────────────────────────────────────────────
# PATCH: Prompts français stricts pour éviter les hallucinations
# ─────────────────────────────────────────────
PROMPTS['rag_response'] = """---Rôle---
Tu es un assistant juridique et fiscal qui répond aux questions en utilisant EXCLUSIVEMENT les informations du contexte fourni.

---Instructions STRICTES---
1. LIS ATTENTIVEMENT tout le contexte ci-dessous avant de répondre.
2. EXTRAIS les informations SPÉCIFIQUES demandées (dates, numéros d'articles, noms, valeurs exactes).
3. Si la question demande une date ou version, CHERCHE les mentions comme "Version en vigueur depuis...", "Modifié par...", ou des dates précises.
4. NE DIS JAMAIS "consultez le document" ou "l'information est mentionnée dans..." — EXTRAIS l'information directement.
5. Si l'information N'EST PAS dans le contexte, réponds UNIQUEMENT : "Je n'ai pas trouvé cette information dans les documents." et ARRÊTE-TOI. Ne spécule pas.
6. NE JAMAIS inventer, reformuler, ou utiliser tes connaissances générales. Cite les passages EXACTS du contexte.
7. Pour chaque affirmation, cite entre guillemets «» le passage exact du contexte qui la supporte.
8. Si tu ne peux pas citer un passage exact pour une affirmation, NE L'INCLUS PAS dans ta réponse.
9. Réponds en français de manière concise et directe.
10. Format de réponse: {response_type}

---Contexte---
{context_data}

{user_prompt}
---Réponse (avec citations exactes du contexte)---"""

PROMPTS['naive_rag_response'] = """---Instructions STRICTES---
Réponds à la question en utilisant UNIQUEMENT les informations du **Contexte** ci-dessous.
Pour chaque affirmation, cite entre guillemets «» le passage exact du contexte qui la supporte.
Si tu ne peux pas citer un passage exact, NE FAIS PAS cette affirmation.
Si l'information n'est pas dans le contexte, dis "Je n'ai pas trouvé cette information."
NE JAMAIS inventer ou utiliser des connaissances externes.
Réponds en français.
Format de réponse: {response_type}

---Contexte---
{content_data}

{user_prompt}
---Réponse (avec citations exactes)---"""

PROMPTS['fail_response'] = "Je n'ai pas trouvé cette information dans les documents fournis. Veuillez reformuler votre question ou vérifier que le document pertinent a bien été ingéré."

# PATCH: Extraction des mots-clés en français pour une meilleure correspondance
# Le prompt par défaut génère des mots-clés en anglais, ce qui cause de mauvaises
# correspondances dans les embeddings de documents français.
PROMPTS['keywords_extraction'] = """---Rôle---
Tu es un assistant qui extrait des mots-clés de recherche pour interroger un Knowledge Graph de documents fiscaux et juridiques français (CGI, BOI, annexes).

---Instructions---
1. high_level_keywords : 2-4 concepts généraux EXACTEMENT tels qu'ils apparaissent dans les textes juridiques fiscaux français.
   ATTENTION : utilise le vocabulaire juridique précis du domaine fiscal (ex: "audit de conformité", "opérateur de plateforme de dématérialisation partenaire", "piste d'audit fiable").
2. low_level_keywords : 3-6 entités précises, numéros d'articles, périodes, obligations mentionnées dans la question.
   Inclure les numéros d'articles CGI/BOI et les termes techniques exacts du droit fiscal français.
   INCLURE AUSSI les synonymes et variantes orthographiques pertinents.
3. Réponds UNIQUEMENT avec un JSON valide.
4. NE JAMAIS inventer de fausses références d'articles ou de lois.
5. TOUJOURS utiliser le vocabulaire du droit fiscal et de la facturation électronique française.

---Exemple---
Question: "Quels formats pour les factures électroniques ?"
Réponse: {{"high_level_keywords": ["factures électroniques", "formats structurés", "interopérabilité", "dématérialisation"], "low_level_keywords": ["UBL", "CII", "Factur-X", "PDF/A3", "article 41 septies C", "portail public de facturation", "format structuré"]}}

---Question à traiter---
{query}

---Réponse JSON---"""

load_dotenv()

# ── Patch separate_content ──────────────────────────────────────────────────
# La fonction originale classe tout ce qui n'est pas "text" comme multimodal,
# y compris list/header/footer/page_number qui ne contiennent ni image, ni
# tableau, ni équation. Cela génère des appels LLM inutiles et des erreurs.
# On ne garde comme multimodal que les vrais types riches.
_REAL_MULTIMODAL_TYPES = {"image", "table", "equation"}
_original_separate_content = _rag_utils.separate_content

def _filtered_separate_content(content_list):
    """Filtre les éléments structurels (header/footer/page_number) pour
    éviter des appels LLM inutiles sur du contenu non analysable, tout en
    conservant leur contenu textuel.

    Gère spécifiquement les éléments de type "list" en aplatissant leurs
    list_items en un bloc de texte unique, évitant ainsi la perte du contenu
    (bug original : conversion en type "text" sans champ "text" = contenu perdu).
    """
    filtered = []
    for item in content_list:
        item_type = item.get("type", "text")
        # Les vrais multimodaux restent tels quels
        if item_type in _REAL_MULTIMODAL_TYPES:
            filtered.append(item)
        # Les listes sont aplaties en texte (CORRECTION BUG : list_items → text)
        elif item_type == "list":
            list_items = item.get("list_items", [])
            if isinstance(list_items, list):
                combined = "\n".join(str(x).strip() for x in list_items if x)
            else:
                combined = str(list_items)
            if combined.strip():
                filtered.append({
                    "type": "text",
                    "text": combined,
                    "page_idx": item.get("page_idx", 0),
                })
        # Tout le reste est forcé en type "text" avec conservation du contenu
        else:
            new_item = dict(item)
            new_item["type"] = "text"
            # S'assurer que le champ "text" existe pour les types non-standard
            if "text" not in new_item:
                for field in ("content", "html", "raw"):
                    if field in new_item:
                        new_item["text"] = new_item[field]
                        break
            filtered.append(new_item)
    return _original_separate_content(filtered)

_rag_utils.separate_content = _filtered_separate_content
# Patch aussi dans le module raganything.raganything qui l'a déjà importé
try:
    import raganything.raganything as _rag_module
    _rag_module.separate_content = _filtered_separate_content
except Exception:
    pass
# Patch dans le module processor qui l'a importé directement
try:
    import raganything.processor as _processor_module
    _processor_module.separate_content = _filtered_separate_content
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────

# Miroir HuggingFace pour éviter les timeouts au premier téléchargement des modèles MinerU
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ─────────────────────────────────────────────
# Configuration — toutes les valeurs sont
# surchargeables via variables d'environnement
# ─────────────────────────────────────────────
OLLAMA_HOST     = os.getenv("OLLAMA_HOST",          "http://localhost:11434")
OLLAMA_BASE_URL = f"{OLLAMA_HOST}/v1"               # endpoint compatible OpenAI
LLM_MODEL       = os.getenv("OLLAMA_LLM_MODEL",     "mistral-nemo:12b")  # réponses (meilleur FR + anti-hallucination)
EXTRACT_MODEL   = os.getenv("OLLAMA_EXTRACT_MODEL", "qwen2.5:7b")       # extraction entités (rapide + structuré)
VISION_MODEL    = os.getenv("OLLAMA_VISION_MODEL",  "llava:7b")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL",   "nomic-embed-text")
EMBED_DIM       = int(os.getenv("OLLAMA_EMBED_DIM", "768"))  # nomic-embed-text = 768d

PARSER         = os.getenv("PARSER",        "docling")  # docling | mineru | paddleocr
PARSE_METHOD   = os.getenv("PARSE_METHOD",  "auto")     # auto | ocr | txt
DEVICE         = os.getenv("MINERU_DEVICE", "cuda")     # cuda | cpu | cuda:0

OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  "./output")
DATA_DIR    = "./donnees_rag"
WORKING_DIR = "./data/rag_anything_storage"
FAKE_API_KEY = "ollama"  # Ollama n'exige pas de vraie clé


# ─────────────────────────────────────────────
# 1a. LLM réponses — Llama 3.1:8b via Ollama
#     Utilisé pour la génération des réponses finales (query)
# ─────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """Tu es un assistant juridique et fiscal expert qui répond aux questions en utilisant UNIQUEMENT le contexte fourni.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec les informations présentes dans le contexte ci-dessous.
2. Pour chaque affirmation, CITE le passage exact du contexte entre guillemets «».
3. Si tu ne peux pas citer un passage exact pour une affirmation, NE FAIS PAS cette affirmation.
4. Si l'information n'est pas dans le contexte, dis \"Je n'ai pas trouvé cette information dans les documents.\"
5. NE JAMAIS inventer, supposer ou utiliser des connaissances externes.
6. Cite les références précises (numéros d'articles, dates, textes de loi) quand c'est pertinent.
7. Réponds en français de manière concise et directe."""

def make_llm_func():
    """Fonction LLM texte via API native Ollama /api/chat.

    Utilise l'API native Ollama au lieu de l'API compatible OpenAI
    pour pouvoir passer num_ctx=16384 et garantir que le modèle
    voit l'intégralité du contexte RAG.
    """
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("hashing_kv", None)
        kwargs.pop("response_format", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"=== LLM PROMPT ({sum(len(m['content']) for m in messages)} chars, {len(messages)} msgs) ===")

        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 16384,
                "temperature": 0.1,
            },
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json=payload,
                    timeout=600,
                )
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error(f"Erreur LLM Ollama: {e}")
            return "Je n'ai pas trouvé cette information dans les documents."
    return llm_func


# ─────────────────────────────────────────────
# 1b. LLM extraction — Llama 3.2:3b via Ollama
#     Utilisé uniquement pour extraire entités/relations (ingestion)
#     Plus rapide et suffisant pour cette tâche structurée
# ─────────────────────────────────────────────
def make_extract_func():
    """Fonction LLM dédiée à l'extraction d'entités/relations via API native Ollama.

    Utilise /api/chat avec num_ctx=8192 pour garantir que les chunks
    de texte sont entièrement visibles pendant l'extraction.
    """
    async def extract_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("hashing_kv", None)
        kwargs.pop("response_format", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": EXTRACT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "temperature": 0.0,
            },
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json=payload,
                    timeout=600,
                )
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error(f"Erreur extraction Ollama: {e}")
            return ""
    return extract_func


# ─────────────────────────────────────────────
# 2. LLM Vision — LLaVA via Ollama
#    Supporte 3 formats (doc officielle RAG-Anything) :
#      a) messages  → VLM enhanced query (multi-turn avec images)
#      b) image_data → image base64 seule
#      c) texte seul → redirigé vers le LLM texte
# ─────────────────────────────────────────────
def make_vision_func():
    """Fonction Vision compatible LightRAG, utilisant LLaVA via Ollama."""
    async def vision_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        kwargs.pop("hashing_kv", None)

        # (a) Format messages OpenAI complet — VLM enhanced query
        if messages:
            return await openai_complete_if_cache(
                VISION_MODEL, "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=FAKE_API_KEY,
                base_url=OLLAMA_BASE_URL,
                **kwargs,
            )

        # (b) Image base64 — Ollama /api/generate accepte base64 directement
        elif image_data:
            payload = {
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {"temperature": 0.1},
            }
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{OLLAMA_HOST}/api/generate",
                        json=payload,
                        timeout=120,
                    )
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except requests.RequestException as e:
                logger.error(f"Erreur vision Ollama: {e}")
                return ""

        # (c) Texte seul → LLM texte
        else:
            return await make_llm_func()(prompt, system_prompt, history_messages, **kwargs)

    return vision_func


# ─────────────────────────────────────────────
# 3. Embeddings — nomic-embed-text via Ollama
# ─────────────────────────────────────────────
def make_embedding_func():
    """Fonction d'embeddings compatible LightRAG, utilisant nomic-embed-text via Ollama."""
    
    async def embed(texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            try:
                # Exécuter la requête sync dans un thread pour ne pas bloquer l'event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda t=text: requests.post(
                        f"{OLLAMA_HOST}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": t},
                        timeout=60,
                    )
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.RequestException as e:
                logger.error(f"Erreur embedding Ollama: {e}")
                embeddings.append([0.0] * EMBED_DIM)
        return np.array(embeddings)

    return EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        func=embed,
    )


# ─────────────────────────────────────────────
# 4. Instance RAGAnything
# ─────────────────────────────────────────────
def get_rag_instance(working_dir: str = WORKING_DIR) -> RAGAnything:
    """
    Crée une instance RAGAnything configurée pour Ollama.

    Paramètres contrôlables via variables d'environnement :
      PARSE_METHOD   : auto | ocr | txt  (défaut: txt — le plus rapide)
      PARSER         : mineru | docling | paddleocr  (défaut: mineru)
      MINERU_DEVICE  : cpu | cuda | cuda:0  (défaut: cpu)
    """
    # Données texte + tableaux uniquement — vision désactivée
    enable_multimodal = PARSE_METHOD != "txt"

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=PARSER,
        parse_method=PARSE_METHOD,
        enable_image_processing=False,          # vision désactivée (données texte/tableaux)
        enable_table_processing=enable_multimodal,
        enable_equation_processing=enable_multimodal,
    )

    # Paramètres LightRAG optimisés pour Ollama sur CPU
    lightrag_kwargs = {
        # Timeouts (en secondes) — Ollama CPU est lent
        "default_llm_timeout": 600,         # 10 min par appel LLM (worker = 2x = 20 min)
        "default_embedding_timeout": 120,   # 2 min par batch d'embeddings

        # Chunks de taille moyenne avec fort chevauchement pour ne rien perdre
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 300,   # doublé: évite de couper les infos aux frontières

        # Extraction d'entités — 2 passes pour meilleur rappel sur les docs juridiques
        "entity_extract_max_gleaning": 2,
        "max_extract_input_tokens": 6000,

        # Concurrence — 1 seul LLM en parallèle (Ollama local ne peut pas en faire plus)
        "llm_model_max_async": 1,
        "embedding_func_max_async": 2,
        "max_parallel_insert": 1,
    }

    # llm_model_func = make_extract_func() → qwen2.5:7b
    #   LightRAG utilise ce LLM pour extract_entities() pendant l'ingestion.
    #   qwen2.5:7b est rapide et précis pour cette tâche structurée.
    #
    # La génération des réponses finales (query) utilise mistral-nemo:12b,
    # passé via QueryParam(model_func=make_llm_func()) dans query().
    return RAGAnything(
        config=config,
        llm_model_func=make_extract_func(),
        vision_model_func=make_vision_func(),
        embedding_func=make_embedding_func(),
        lightrag_kwargs=lightrag_kwargs,
    )


# ─────────────────────────────────────────────
# 5. Ingestion d'un document unique
# ─────────────────────────────────────────────
async def ingest_document(
    file_path: str,
    output_dir: str = OUTPUT_DIR,
    working_dir: str = WORKING_DIR,
    parse_method: Optional[str] = None,
    lang: str = "fr",
    device: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    formula: bool = True,
    table: bool = True,
    backend: str = "pipeline",
    display_stats: bool = True,
    doc_id: Optional[str] = None,
) -> RAGAnything:
    """
    Ingère un seul document avec le pipeline multimodal.

    Args:
        file_path   : Chemin vers le document (PDF, DOCX, PPTX, image…)
        output_dir  : Répertoire de sortie MinerU (images extraites, etc.)
        working_dir : Répertoire de stockage LightRAG (Knowledge Graph)
        parse_method: auto | ocr | txt  (surcharge PARSE_METHOD env)
        lang        : Langue du document pour l'OCR (ex: "fr", "en", "ch")
        device      : Périphérique d'inférence (cpu | cuda | cuda:0)
        start_page  : Page de début (0-based, PDF uniquement)
        end_page    : Page de fin   (0-based, PDF uniquement)
        formula     : Activer le parsing des formules mathématiques
        table       : Activer le parsing des tableaux
        backend     : Backend MinerU (pipeline | hybrid-auto-engine | vlm-auto-engine …)
        display_stats: Afficher les statistiques de contenu
        doc_id      : Identifiant personnalisé du document (auto-généré si None)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingestion du document: {file_path}")

    rag = get_rag_instance(working_dir)

    kwargs: Dict[str, Any] = {
        "display_stats": display_stats,
        "lang": lang,
        "formula": formula,
        "table": table,
        "backend": backend,
    }
    if device:
        kwargs["device"] = device
    if start_page is not None:
        kwargs["start_page"] = start_page
    if end_page is not None:
        kwargs["end_page"] = end_page
    if doc_id:
        kwargs["doc_id"] = doc_id

    await rag.process_document_complete(
        file_path=file_path,
        output_dir=output_dir,
        parse_method=parse_method or PARSE_METHOD,
        **kwargs,
    )

    logger.success(f"✅ Document ingéré : {Path(file_path).name}")
    return rag


# ─────────────────────────────────────────────
# 6. Ingestion d'un dossier complet
# ─────────────────────────────────────────────
async def ingest_all_documents(
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    working_dir: str = WORKING_DIR,
    file_extensions: Optional[List[str]] = None,
    recursive: bool = False,
    max_workers: int = 1,
) -> RAGAnything:
    """
    Ingère tous les documents d'un répertoire.

    Args:
        data_dir       : Répertoire source contenant les documents
        output_dir     : Répertoire de sortie MinerU
        working_dir    : Répertoire de stockage LightRAG
        file_extensions: Extensions à traiter (défaut: [".pdf"])
        recursive      : Recherche récursive dans les sous-dossiers
        max_workers    : Workers parallèles (1 recommandé en local pour Ollama)
    """
    if file_extensions is None:
        file_extensions = [".pdf"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingestion du dossier: {data_dir}")
    logger.info(f"Parser: {PARSER}, Méthode: {PARSE_METHOD}, Extensions: {file_extensions}")

    rag = get_rag_instance(working_dir)

    await rag.process_folder_complete(
        folder_path=data_dir,
        output_dir=output_dir,
        parse_method=PARSE_METHOD,
        file_extensions=file_extensions,
        recursive=recursive,
        max_workers=max_workers,
    )

    logger.success("✅ Ingestion du dossier terminée.")
    return rag


# ─────────────────────────────────────────────
# 7. Insertion directe d'une content-list pré-parsée
# ─────────────────────────────────────────────
async def insert_content_list(
    content_list: List[Dict[str, Any]],
    file_path: str,
    working_dir: str = WORKING_DIR,
    doc_id: Optional[str] = None,
    display_stats: bool = True,
) -> RAGAnything:
    """
    Insère directement une liste de contenus pré-parsés sans passer par MinerU.

    Utile pour :
      - Réutiliser un parsing déjà effectué
      - Injecter du contenu depuis un parser externe
      - Générer du contenu programmatiquement

    Format des éléments de content_list :
      Texte    : {"type": "text",     "text": "...",         "page_idx": 0}
      Image    : {"type": "image",    "img_path": "/abs/...", "image_caption": [...], "page_idx": 1}
      Tableau  : {"type": "table",    "table_body": "...",   "table_caption": [...], "page_idx": 2}
      Équation : {"type": "equation", "latex": "...",        "text": "...",           "page_idx": 3}

    Note : img_path doit être un chemin absolu.
    """
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Insertion de {len(content_list)} éléments depuis '{file_path}'")

    rag = get_rag_instance(working_dir)

    await rag.insert_content_list(
        content_list=content_list,
        file_path=file_path,
        doc_id=doc_id,
        display_stats=display_stats,
    )

    logger.success(f"✅ Content-list insérée ({len(content_list)} éléments).")
    return rag


# ─────────────────────────────────────────────
# 8. Requête texte (avec option VLM-enhanced)
# ─────────────────────────────────────────────
async def query(
    question: str,
    mode: str = "hybrid",
    vlm_enhanced: Optional[bool] = None,
    working_dir: str = WORKING_DIR,
) -> str:
    """
    Requête texte sur le Knowledge Graph RAG-Anything.

    Args:
        question     : Question à poser
        mode         : naive | local | global | hybrid
                         naive  → RAG classique (chunks vectoriels)
                         local  → graphe local (entités proches)
                         global → graphe global (thèmes transversaux)
                         hybrid → local + global (recommandé)
        vlm_enhanced : True  → force l'analyse visuelle des images du contexte
                       False → désactive le VLM même si disponible
                       None  → automatique (activé si vision_model_func fourni)
        working_dir  : Répertoire de stockage LightRAG
    """
    logger.info(f"Requête (mode={mode}, vlm_enhanced={vlm_enhanced}): {question[:60]}...")
    logger.debug(f"LLM réponse: {LLM_MODEL} | LLM extraction: {EXTRACT_MODEL}")

    rag = get_rag_instance(working_dir)
    await rag._ensure_lightrag_initialized()

    # Budget tokens large grâce à num_ctx=16384 via API native Ollama.
    # ~10000 tokens de contexte RAG + ~1500 tokens prompt/réponse = ~11500 < 16384 ✓
    if vlm_enhanced is None:
        vlm_enhanced = False

    kwargs: Dict[str, Any] = {
        "top_k": 25,                    # voisins graphe élargis pour meilleur rappel
        "chunk_top_k": 15,              # plus de chunks pour ne rien manquer
        "max_entity_tokens": 2000,      # entités larges pour capturer toutes les infos
        "max_relation_tokens": 1000,    # relations larges
        "max_total_tokens": 12000,      # contexte large (mistral-nemo supporte 128K)
        "model_func": make_llm_func(),
        "vlm_enhanced": vlm_enhanced,
    }

    result = await rag.aquery(question, mode=mode, **kwargs)

    _not_found = "Je n'ai pas trouvé"
    if not result or result.strip().startswith(_not_found):
        logger.warning(f"Mode '{mode}' n'a pas trouvé de réponse — fallback recherche textuelle.")
        result = await _keyword_fallback(question, working_dir)

    result = result or "Je n'ai pas trouvé cette information dans les documents."
    logger.success(f"Réponse générée ({len(result)} caractères)")
    return result


# ─────────────────────────────────────────────
# Fallback : recherche textuelle directe sur les chunks
# Contourne la dépendance au LLM pour l'extraction de mots-clés
# (utile quand le LLM extrait de mauvais mots-clés en mode hybrid/local)
# ─────────────────────────────────────────────
_FR_STOPWORDS = {
    "sur", "quelle", "quels", "quelles", "quel", "pour", "une", "un",
    "la", "le", "les", "de", "du", "des", "et", "en", "dans", "par",
    "est", "qui", "que", "avec", "sans", "mais", "donc", "comme", "plus",
    "cette", "ces", "tout", "tous", "toute", "toutes", "aussi", "très",
    "lors", "selon", "entre", "après", "avant", "depuis", "pendant",
}

def _extract_all_texts_from_parse_cache(working_dir: str) -> List[str]:
    """
    Extrait TOUT le texte du parse_cache MinerU, y compris les éléments
    de type 'list' (list_items) qui n'ont pas été indexés dans LightRAG
    à cause du bug de parsing initial.

    Retourne une liste de blocs de texte (un par item content_list).
    """
    cache_path = Path(working_dir) / "kv_store_parse_cache.json"
    if not cache_path.exists():
        return []

    with open(cache_path, "r", encoding="utf-8") as f:
        parse_cache = json.load(f)

    texts: List[str] = []
    for entry in parse_cache.values():
        for item in entry.get("content_list", []):
            item_type = item.get("type", "")

            if item_type == "list":
                # Aplatir les items de liste en texte
                list_items = item.get("list_items", [])
                if isinstance(list_items, list):
                    combined = "\n".join(str(x).strip() for x in list_items if x)
                else:
                    combined = str(list_items)
                if combined.strip():
                    texts.append(combined.strip())

            elif item_type == "text":
                text = item.get("text", "").strip()
                if text:
                    texts.append(text)

    return texts


async def _keyword_fallback(question: str, working_dir: str) -> str:
    """
    Recherche textuelle BM25-like sur TOUTES les sources disponibles :
    - kv_store_text_chunks.json  (chunks indexés dans LightRAG)
    - kv_store_parse_cache.json  (contenu MinerU complet, incl. list_items manquants)

    Extrait des extraits ciblés autour des mots-clés et les passe au LLM.
    """
    # ── 1. Mots-clés de la question ───────────────────────────────────────────
    words = re.findall(r"[a-zA-ZÀ-ÿ]{4,}", question.lower())
    keywords = [w for w in words if w not in _FR_STOPWORDS]
    if not keywords:
        return ""
    logger.debug(f"Fallback mots-clés: {keywords}")

    # ── 2. Collecter tous les blocs de texte disponibles ──────────────────────
    all_texts: List[str] = []

    # Source A : chunks indexés (textes complets)
    chunks_path = Path(working_dir) / "kv_store_text_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_texts.extend(c.get("content", "") for c in chunks.values() if c.get("content"))

    # Source B : parse_cache complet (inclut list_items non indexés)
    all_texts.extend(_extract_all_texts_from_parse_cache(working_dir))

    if not all_texts:
        return ""

    # ── 3. Scorer par correspondance (distinct keywords matched, puis fréquence) ─
    # Utiliser d'abord le nombre de mots-clés DISTINCTS trouvés (comme IDF :
    # les termes rares ont le même poids que les termes fréquents) puis la
    # fréquence totale pour départager les égalités.
    scored: List[tuple] = []
    for text in all_texts:
        text_lower = text.lower()
        distinct = sum(1 for kw in keywords if kw in text_lower)
        total    = sum(text_lower.count(kw) for kw in keywords)
        if distinct > 0:
            scored.append((distinct, total, text))

    if not scored:
        logger.warning("Fallback : aucun texte correspondant aux mots-clés.")
        return ""

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    logger.info(f"Fallback : {len(scored)} blocs scorés, distinct max={scored[0][0]}")

    # ── 4. Extraire les paragraphes/phrases pertinents ────────────────────────
    snippets: List[str] = []
    seen: set = set()

    for _, _, text in scored[:8]:
        # Découper en paragraphes/phrases
        paras = [p.strip() for p in re.split(r"\n{2,}|\n(?=\d+[°\.]|\$\d)", text) if p.strip()]
        for para in paras:
            para_lower = para.lower()
            if any(kw in para_lower for kw in keywords) and para not in seen and len(para) > 20:
                snippets.append(para)
                seen.add(para)
            if len(snippets) >= 15:
                break
        if len(snippets) >= 15:
            break

    if not snippets:
        snippets = [text[:600] for _, _, text in scored[:4]]

    context = "\n\n".join(snippets)
    logger.info(f"Fallback : {len(snippets)} extraits, {len(context)} caractères")

    # ── 5. Appel LLM avec prompt générique ───────────────────────────────────
    prompt = (
        f"Réponds à la question en utilisant uniquement les extraits de documents ci-dessous.\n"
        f"Réponds en français. Si l'information est absente des extraits, "
        f"dis uniquement : Je n'ai pas trouvé cette information dans les documents.\n\n"
        f"---Extraits---\n{context}\n\n"
        f"---Question---\n{question}\n\n"
        f"---Réponse---"
    )

    try:
        llm = make_llm_func()
        return await llm(prompt, system_prompt=None)
    except Exception as e:
        logger.error(f"Erreur LLM fallback: {e}")
        return ""


# ─────────────────────────────────────────────
# 9. Requête multimodale (avec contenu spécifique)
# ─────────────────────────────────────────────
async def query_multimodal(
    question: str,
    multimodal_content: List[Dict[str, Any]],
    mode: str = "hybrid",
    working_dir: str = WORKING_DIR,
) -> str:
    """
    Requête enrichie avec du contenu multimodal spécifique (tableau, équation, image…).

    Args:
        question          : Question à poser
        multimodal_content: Liste de contenus multimodaux à inclure dans la requête.
          Exemples :
            [{"type": "table",    "table_data": "csv...", "table_caption": "Perf"}]
            [{"type": "equation", "latex": "E=mc^2",      "equation_caption": "Énergie"}]
        mode              : naive | local | global | hybrid
        working_dir       : Répertoire de stockage LightRAG
    """
    logger.info(f"Requête multimodale (mode={mode}): {question[:60]}...")

    rag = get_rag_instance(working_dir)
    await rag._ensure_lightrag_initialized()
    result = await rag.aquery_with_multimodal(
        question,
        multimodal_content=multimodal_content,
        mode=mode,
    )
    logger.success(f"Réponse multimodale générée ({len(result)} caractères)")
    return result


# ─────────────────────────────────────────────
# 10. Vérifications
# ─────────────────────────────────────────────
def check_ollama_models() -> Dict[str, Any]:
    """Vérifie que les 3 modèles Ollama nécessaires sont disponibles (vision désactivée)."""
    required = {
        "llm (réponses)": LLM_MODEL,
        "extract (entités)": EXTRACT_MODEL,
        "embedding": EMBED_MODEL,
    }
    status: Dict[str, Any] = {}

    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        available = [m["name"] for m in response.json().get("models", [])]

        for role, model in required.items():
            base = model.split(":")[0]
            found = any(base in m for m in available)
            status[role] = {"model": model, "available": found}

    except requests.RequestException as e:
        logger.error(f"Impossible de contacter Ollama ({OLLAMA_HOST}): {e}")
        for role, model in required.items():
            status[role] = {"model": model, "available": False, "error": str(e)}

    return status


def is_rag_anything_ready() -> bool:
    """Retourne True si l'index LightRAG existe et contient des données."""
    p = Path(WORKING_DIR)
    return p.exists() and any(p.iterdir())


# ─────────────────────────────────────────────
# Point d'entrée — ingestion standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG-Anything Pipeline — Ingestion multimodale (Ollama)")
    print("=" * 60)
    print(f"  Parser       : {PARSER}")
    print(f"  Méthode      : {PARSE_METHOD}")
    print(f"  LLM          : {LLM_MODEL}")
    print(f"  Vision       : désactivée (données texte/tableaux)")
    print(f"  Embeddings   : {EMBED_MODEL} ({EMBED_DIM}d)")
    print(f"  Working dir  : {WORKING_DIR}")

    print("\nVérification des modèles Ollama...")
    status = check_ollama_models()
    missing = []
    for role, info in status.items():
        icon = "✅" if info.get("available") else "❌"
        print(f"  {icon} {role}: {info['model']}")
        if not info.get("available"):
            missing.append(info['model'])

    if missing:
        print("\n❌ Modèles manquants. Installez-les avec :")
        for m in missing:
            print(f"     ollama pull {m}")
        exit(1)

    print("\nDémarrage de l'ingestion...")
    asyncio.run(ingest_all_documents())

