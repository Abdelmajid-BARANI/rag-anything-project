"""
Init file for ingestion module (RAG-Anything pipeline)
"""
from .rag_anything_pipeline import (
    get_rag_instance,
    ingest_document,
    ingest_all_documents,
    insert_content_list,
    query,
    query_multimodal,
    check_ollama_models,
    is_rag_anything_ready,
    # Constantes utiles
    WORKING_DIR,
    OUTPUT_DIR,
    DATA_DIR,
    LLM_MODEL,
    VISION_MODEL,
    EMBED_MODEL,
    PARSER,
    PARSE_METHOD,
)

__all__ = [
    "get_rag_instance",
    "ingest_document",
    "ingest_all_documents",
    "insert_content_list",
    "query",
    "query_multimodal",
    "check_ollama_models",
    "is_rag_anything_ready",
    "WORKING_DIR",
    "OUTPUT_DIR",
    "DATA_DIR",
    "LLM_MODEL",
    "VISION_MODEL",
    "EMBED_MODEL",
    "PARSER",
    "PARSE_METHOD",
]
