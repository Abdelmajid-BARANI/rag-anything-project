"""
Script de correction : ré-insère le contenu manquant des éléments "list"
qui ont été perdus lors de l'ingestion initiale.

PROBLÈME : La fonction _filtered_separate_content convertissait les items
de type "list" en type "text" SANS copier le contenu list_items → text.
Résultat : tout le texte des listes (ex: périodes d'audit) était perdu.

CE SCRIPT :
1. Lit kv_store_parse_cache.json (le parsing MinerU correct et complet)
2. Extrait TOUS les éléments de type "list" avec leur contenu list_items
3. Les insère dans LightRAG via insert_content_list
4. Vide les entrées de cache LLM erronées

Usage:
    python fix_missing_content.py

Prérequis: Ollama doit être lancé avec llama3.1:8b et nomic-embed-text
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Ajouter le répertoire src au PATH pour importer le pipeline
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.rag_anything_pipeline import insert_content_list, WORKING_DIR

PARSE_CACHE_PATH = Path(WORKING_DIR) / "kv_store_parse_cache.json"
LLM_CACHE_PATH   = Path(WORKING_DIR) / "kv_store_llm_response_cache.json"
DOC_STATUS_PATH  = Path(WORKING_DIR) / "kv_store_doc_status.json"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_llm_cache():
    """
    Vide le cache LLM pour forcer la re-génération des réponses.
    Les mauvaises réponses mises en cache bloquaient les corrections.
    """
    if not LLM_CACHE_PATH.exists():
        print("  Cache LLM introuvable, rien à vider.")
        return

    cache = load_json(LLM_CACHE_PATH)
    original_count = len(cache)

    # Supprimer uniquement les entrées de type "query" (réponses finales)
    # et "keywords" (mots-clés extraits) qui peuvent être incorrects.
    # Les entrées "extract" (extraction d'entités) sont conservées car elles
    # coûtent beaucoup de temps à recalculer.
    keys_to_delete = [
        k for k, v in cache.items()
        if v.get("cache_type") in ("query", "keywords")
    ]

    for k in keys_to_delete:
        del cache[k]

    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"  Cache LLM nettoyé : {len(keys_to_delete)} entrées supprimées "
          f"({original_count - len(keys_to_delete)} conservées).")


def extract_list_items_from_cache() -> dict[str, list[dict]]:
    """
    Lit le parse_cache et extrait tous les éléments de type "list"
    regroupés par entrée de cache (= par document parsé).

    Retourne un dict {cache_key: [content_items_text, ...]}
    """
    if not PARSE_CACHE_PATH.exists():
        print(f"  Parse cache introuvable : {PARSE_CACHE_PATH}")
        return {}

    cache = load_json(PARSE_CACHE_PATH)
    result = {}

    for key, entry in cache.items():
        content_list = entry.get("content_list", [])
        if not content_list:
            continue

        list_texts = []
        for item in content_list:
            if item.get("type") != "list":
                continue

            list_items = item.get("list_items", [])
            if not list_items:
                continue

            # Construire le texte de la liste
            if isinstance(list_items, list):
                combined = "\n".join(str(x).strip() for x in list_items if x)
            else:
                combined = str(list_items)

            if combined.strip():
                list_texts.append({
                    "type": "text",
                    "text": combined,
                    "page_idx": item.get("page_idx", 0),
                })

        if list_texts:
            result[key] = list_texts

    return result


def identify_document(cache_key: str, list_items: list[dict], doc_status: dict) -> str:
    """
    Tente d'identifier le nom de fichier associé à une entrée de cache
    en cherchant des correspondances de texte avec les résumés de doc_status.
    Retourne "document_<key[:8]>.pdf" si la correspondance échoue.
    """
    if not list_items:
        return f"document_{cache_key[:8]}.pdf"

    # Texte du premier item de liste pour la correspondance
    sample_text = list_items[0].get("text", "")[:200].lower()

    for doc_id, doc_info in doc_status.items():
        file_path = doc_info.get("file_path", "")
        summary = doc_info.get("content_summary", "").lower()
        # Correspondance floue : si des mots du résumé sont dans le texte
        if file_path and any(word in sample_text for word in summary.split()[:10] if len(word) > 4):
            return file_path

    return f"document_{cache_key[:8]}.pdf"


async def main():
    print("=" * 60)
    print("CORRECTION : Ré-insertion du contenu manquant des listes")
    print("=" * 60)

    # Étape 1 : Vider le cache LLM (réponses et mots-clés erronés)
    print("\n[1/3] Nettoyage du cache LLM...")
    clear_llm_cache()

    # Étape 2 : Extraire le contenu des listes depuis le parse cache
    print("\n[2/3] Extraction des éléments 'list' depuis le parse cache...")
    list_content_by_key = extract_list_items_from_cache()

    if not list_content_by_key:
        print("  Aucun élément 'list' trouvé dans le parse cache.")
        print("  Soit les documents n'ont pas de listes, soit le cache est vide.")
        return

    total_items = sum(len(v) for v in list_content_by_key.values())
    print(f"  Trouvé {len(list_content_by_key)} document(s) avec "
          f"{total_items} élément(s) de liste à ré-indexer.")

    # Étape 3 : Insérer le contenu manquant
    print("\n[3/3] Insertion dans LightRAG...")
    doc_status = load_json(DOC_STATUS_PATH) if DOC_STATUS_PATH.exists() else {}

    for i, (cache_key, items) in enumerate(list_content_by_key.items(), 1):
        file_path = identify_document(cache_key, items, doc_status)
        print(f"\n  Document {i}/{len(list_content_by_key)}: {file_path}")
        print(f"  → {len(items)} bloc(s) de liste à insérer")

        # Afficher un aperçu du contenu
        for j, item in enumerate(items[:2]):
            preview = item["text"][:120].replace("\n", " ")
            print(f"    [{j+1}] {preview}...")

        try:
            await insert_content_list(
                content_list=items,
                file_path=f"fix_lists_{file_path}",
                display_stats=False,
            )
            print(f"  ✅ Inséré avec succès")
        except Exception as e:
            print(f"  ❌ Erreur : {e}")

    print("\n" + "=" * 60)
    print("CORRECTION TERMINÉE")
    print("=" * 60)
    print("\nÉtapes suivantes :")
    print("1. Relancez l'API : python api.py")
    print("2. Testez la requête : POST /query")
    print('   {"query": "Sur quelle période porte l\'audit pour une première immatriculation ?", "mode": "hybrid"}')
    print("\nRéponse attendue :")
    print("  1° Pour l'obtention du numéro d'immatriculation : les SIX MOIS précédant la date d'engagement de l'audit")
    print("  2° Pour le renouvellement : les TROIS ANNÉES précédant la date d'engagement de l'audit")


if __name__ == "__main__":
    asyncio.run(main())
