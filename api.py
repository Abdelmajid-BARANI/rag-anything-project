"""
API FastAPI — RAG-Anything (LightRAG + MinerU, 100% local via Ollama)

Endpoints :
  GET  /health                -> etat du systeme
  POST /query                 -> requete texte (+ option VLM-enhanced)
  POST /query/multimodal      -> requete avec contenu multimodal (tableau, equation...)
  POST /ingest/document       -> ingerer un seul fichier
  POST /ingest/folder         -> ingerer un dossier complet
  POST /ingest/content-list   -> insertion directe d'une content-list pre-parsee
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Ajouter src au path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ingestion.rag_anything_pipeline import (
    query as rag_query,
    query_multimodal as rag_query_multimodal,
    ingest_document as rag_ingest_document,
    ingest_all_documents as rag_ingest_folder,
    insert_content_list as rag_insert_content_list,
    check_ollama_models,
    is_rag_anything_ready,
    WORKING_DIR,
    OUTPUT_DIR,
    DATA_DIR,
    LLM_MODEL,
    VISION_MODEL,
    EMBED_MODEL,
    PARSER,
    PARSE_METHOD,
)
from utils import setup_logging

setup_logging(log_level="INFO")


# ═══════════════════════════════════════════════
# Schemas Pydantic
# ═══════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str = Field(..., description="Question a poser au systeme RAG")
    mode: str = Field(
        default="hybrid",
        description="Mode : naive | local | global | hybrid",
    )
    vlm_enhanced: Optional[bool] = Field(
        default=None,
        description=(
            "True  -> force l'analyse visuelle des images dans le contexte, "
            "False -> desactive le VLM, "
            "None  -> automatique"
        ),
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    mode: str
    vlm_enhanced: Optional[bool]
    duration_seconds: float


class MultimodalQueryRequest(BaseModel):
    query: str = Field(..., description="Question a poser")
    multimodal_content: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "Liste de contenus multimodaux. "
            "Exemples : "
            "[{'type': 'table', 'table_data': 'csv...', 'table_caption': 'Titre'}] "
            "[{'type': 'equation', 'latex': 'E=mc^2', 'equation_caption': 'Energie'}]"
        ),
    )
    mode: str = Field(default="hybrid", description="naive | local | global | hybrid")


class IngestDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Chemin absolu vers le fichier a ingerer")
    parse_method: Optional[str] = Field(
        default=None, description="auto | ocr | txt (surcharge PARSE_METHOD)"
    )
    lang: str = Field(default="fr", description="Langue pour l'OCR (fr, en, ch...)")
    device: Optional[str] = Field(default=None, description="cpu | cuda | cuda:0")
    start_page: Optional[int] = Field(default=None, description="Page de debut (0-based)")
    end_page: Optional[int] = Field(default=None, description="Page de fin (0-based)")
    formula: bool = Field(default=True, description="Activer le parsing des formules")
    table: bool = Field(default=True, description="Activer le parsing des tableaux")
    backend: str = Field(default="pipeline", description="Backend MinerU")
    doc_id: Optional[str] = Field(default=None, description="Identifiant personnalise")


class IngestFolderRequest(BaseModel):
    folder_path: str = Field(
        default=DATA_DIR, description="Repertoire source contenant les documents"
    )
    file_extensions: List[str] = Field(
        default=[".pdf"], description="Extensions de fichiers a traiter"
    )
    recursive: bool = Field(default=False, description="Recherche recursive")
    max_workers: int = Field(
        default=1, description="Workers paralleles (1 recommande avec Ollama local)"
    )


class ContentListItem(BaseModel):
    type: str = Field(..., description="text | image | table | equation | <custom>")
    page_idx: int = Field(default=0, description="Numero de page (0-based)")
    text: Optional[str] = None
    img_path: Optional[str] = Field(default=None, description="Chemin absolu vers l'image")
    image_caption: Optional[List[str]] = None
    image_footnote: Optional[List[str]] = None
    table_body: Optional[str] = None
    table_caption: Optional[List[str]] = None
    table_footnote: Optional[List[str]] = None
    latex: Optional[str] = None

    model_config = {"extra": "allow"}


class InsertContentListRequest(BaseModel):
    content_list: List[ContentListItem] = Field(
        ..., description="Liste de contenus pre-parses a inserer"
    )
    file_path: str = Field(
        ..., description="Nom de fichier de reference (pour la citation)"
    )
    doc_id: Optional[str] = Field(
        default=None, description="Identifiant personnalise (auto-genere si None)"
    )
    display_stats: bool = Field(default=True, description="Afficher les statistiques")


class IngestResponse(BaseModel):
    status: str
    message: str
    duration_seconds: float


# ═══════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Demarrage de l'API RAG-Anything...")
    logger.info(f"  Parser    : {PARSER}  |  Methode : {PARSE_METHOD}")
    logger.info(f"  LLM       : {LLM_MODEL}")
    logger.info(f"  Vision    : {VISION_MODEL}")
    logger.info(f"  Embeddings: {EMBED_MODEL}")

    models = check_ollama_models()
    for role, info in models.items():
        icon = "OK" if info.get("available") else "MANQUANT"
        logger.info(f"  Ollama [{icon}] {role}: {info['model']}")

    if not is_rag_anything_ready():
        logger.warning(
            "Index RAG-Anything introuvable - "
            "lancez l'ingestion : python src/ingestion/rag_anything_pipeline.py"
        )
    else:
        logger.success("Index RAG-Anything pret.")

    yield
    logger.info("Arret de l'API RAG-Anything")


# ═══════════════════════════════════════════════
# Application
# ═══════════════════════════════════════════════

app = FastAPI(
    title="RAG-Anything API",
    description=(
        "API multimodale basee sur LightRAG + MinerU + Ollama (100% local). "
        "Supporte les requetes texte, VLM-enhanced et multimodales (tableaux, equations, images)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_MODES = {"naive", "local", "global", "hybrid"}


# ═══════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════

@app.get("/health", summary="Etat du systeme")
def health():
    """
    Verifie l'etat du systeme :
    - Disponibilite des modeles Ollama
    - Presence de l'index RAG-Anything
    - Configuration courante
    """
    ready = is_rag_anything_ready()
    models = check_ollama_models()
    return {
        "status": "ok",
        "index_ready": ready,
        "config": {
            "parser": PARSER,
            "parse_method": PARSE_METHOD,
            "llm": LLM_MODEL,
            "vision": VISION_MODEL,
            "embedding": EMBED_MODEL,
            "working_dir": WORKING_DIR,
        },
        "models": {
            role: {"model": info["model"], "available": info.get("available")}
            for role, info in models.items()
        },
    }


@app.post("/query", response_model=QueryResponse, summary="Requete texte")
async def query_endpoint(request: QueryRequest):
    """
    Pose une question au Knowledge Graph RAG-Anything.

    Modes disponibles :
    - naive  : RAG classique (chunks vectoriels uniquement)
    - local  : graphe local (entites proches de la question)
    - global : graphe global (themes transversaux)
    - hybrid : local + global (recommande)

    VLM-enhanced :
    Quand active, le systeme recupere automatiquement les images du contexte
    et les envoie a LLaVA pour une analyse visuelle approfondie.
    """
    if not is_rag_anything_ready():
        raise HTTPException(503, "Index RAG-Anything non disponible. Lancez l'ingestion d'abord.")
    if request.mode not in VALID_MODES:
        raise HTTPException(400, f"Mode invalide '{request.mode}'. Valeurs : {VALID_MODES}")

    start = time.time()
    try:
        answer = await rag_query(
            request.query,
            mode=request.mode,
            vlm_enhanced=request.vlm_enhanced,
        )
    except Exception as e:
        logger.error(f"Erreur requete: {e}")
        raise HTTPException(500, str(e))

    return QueryResponse(
        query=request.query,
        answer=answer,
        mode=request.mode,
        vlm_enhanced=request.vlm_enhanced,
        duration_seconds=round(time.time() - start, 2),
    )


@app.post("/query/multimodal", response_model=QueryResponse, summary="Requete multimodale")
async def query_multimodal_endpoint(request: MultimodalQueryRequest):
    """
    Requete enrichie avec du contenu multimodal specifique.

    Permet d'inclure dans la requete des elements comme :
    - Un tableau de donnees
    - Une equation mathematique (LaTeX)
    - Une image (chemin absolu)

    Exemple de multimodal_content :
    [
      {"type": "table", "table_data": "Methode,Score\\nRAG-Anything,95%", "table_caption": "Resultats"},
      {"type": "equation", "latex": "E=mc^2", "equation_caption": "Energie"}
    ]
    """
    if not is_rag_anything_ready():
        raise HTTPException(503, "Index RAG-Anything non disponible. Lancez l'ingestion d'abord.")
    if request.mode not in VALID_MODES:
        raise HTTPException(400, f"Mode invalide '{request.mode}'. Valeurs : {VALID_MODES}")

    start = time.time()
    try:
        answer = await rag_query_multimodal(
            request.query,
            multimodal_content=request.multimodal_content,
            mode=request.mode,
        )
    except Exception as e:
        logger.error(f"Erreur requete multimodale: {e}")
        raise HTTPException(500, str(e))

    return QueryResponse(
        query=request.query,
        answer=answer,
        mode=request.mode,
        vlm_enhanced=None,
        duration_seconds=round(time.time() - start, 2),
    )


@app.post("/ingest/document", response_model=IngestResponse, summary="Ingerer un document")
async def ingest_document_endpoint(
    request: IngestDocumentRequest, background_tasks: BackgroundTasks
):
    """
    Ingere un seul document dans le Knowledge Graph.

    Supporte : PDF, DOCX, PPTX, XLSX, images (JPG/PNG/BMP/TIFF...)

    L'ingestion est lancee en arriere-plan.
    Consultez /health pour verifier quand l'index est pret.
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(404, f"Fichier non trouve : {request.file_path}")

    start = time.time()

    async def _run():
        try:
            await rag_ingest_document(
                file_path=request.file_path,
                parse_method=request.parse_method,
                lang=request.lang,
                device=request.device,
                start_page=request.start_page,
                end_page=request.end_page,
                formula=request.formula,
                table=request.table,
                backend=request.backend,
                doc_id=request.doc_id,
            )
        except Exception as e:
            logger.error(f"Erreur ingestion document: {e}")

    background_tasks.add_task(_run)

    return IngestResponse(
        status="accepted",
        message=f"Ingestion de '{os.path.basename(request.file_path)}' lancee en arriere-plan.",
        duration_seconds=round(time.time() - start, 3),
    )


@app.post("/ingest/folder", response_model=IngestResponse, summary="Ingerer un dossier")
async def ingest_folder_endpoint(
    request: IngestFolderRequest, background_tasks: BackgroundTasks
):
    """
    Ingere tous les documents d'un dossier dans le Knowledge Graph.

    L'ingestion est lancee en arriere-plan.
    """
    if not os.path.isdir(request.folder_path):
        raise HTTPException(404, f"Dossier non trouve : {request.folder_path}")

    start = time.time()

    async def _run():
        try:
            await rag_ingest_folder(
                data_dir=request.folder_path,
                file_extensions=request.file_extensions,
                recursive=request.recursive,
                max_workers=request.max_workers,
            )
        except Exception as e:
            logger.error(f"Erreur ingestion dossier: {e}")

    background_tasks.add_task(_run)

    return IngestResponse(
        status="accepted",
        message=f"Ingestion du dossier '{request.folder_path}' lancee en arriere-plan.",
        duration_seconds=round(time.time() - start, 3),
    )


@app.post(
    "/ingest/content-list",
    response_model=IngestResponse,
    summary="Insertion directe d'une content-list",
)
async def insert_content_list_endpoint(
    request: InsertContentListRequest, background_tasks: BackgroundTasks
):
    """
    Insere directement une liste de contenus pre-parses SANS passer par MinerU.

    Utile pour :
    - Reutiliser des resultats de parsing deja effectues
    - Injecter du contenu depuis un parser externe
    - Generer du contenu programmatiquement

    Format de chaque element :
      {"type": "text",     "text": "...",          "page_idx": 0}
      {"type": "image",    "img_path": "/abs/...", "image_caption": [...], "page_idx": 1}
      {"type": "table",    "table_body": "...",    "table_caption": [...], "page_idx": 2}
      {"type": "equation", "latex": "...",         "text": "...",          "page_idx": 3}

    Note : img_path doit etre un chemin ABSOLU.
    """
    if not request.content_list:
        raise HTTPException(400, "La content-list est vide.")

    start = time.time()
    content_dicts = [item.model_dump(exclude_none=True) for item in request.content_list]

    async def _run():
        try:
            await rag_insert_content_list(
                content_list=content_dicts,
                file_path=request.file_path,
                doc_id=request.doc_id,
                display_stats=request.display_stats,
            )
        except Exception as e:
            logger.error(f"Erreur insertion content-list: {e}")

    background_tasks.add_task(_run)

    return IngestResponse(
        status="accepted",
        message=f"Insertion de {len(request.content_list)} elements depuis '{request.file_path}' lancee.",
        duration_seconds=round(time.time() - start, 3),
    )


# ═══════════════════════════════════════════════
# Point d'entree
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
