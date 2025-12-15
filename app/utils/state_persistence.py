"""
State Persistence Module
========================

Gestion de la sauvegarde/lecture progressive de l'etat du pipeline via JSONL.
Permet la reprise apres crash sans perte de donnees.

Format JSONL (JSON Lines):
- Une entree JSON par ligne
- Append-only pour crash-safety
- Tolerant aux lignes corrompues

Architecture:
    filtering_progress.jsonl -> Etat du filtrage LLM (relevant/skipped)
    import_progress.jsonl    -> Etat de l'import Zotero

Usage:
    from app.utils.state_persistence import (
        append_filtering_state,
        get_processed_indices,
        rebuild_preview_from_state
    )
"""
import json
import os
import fcntl
import logging
from datetime import datetime
from typing import List, Dict, Set, Optional, Any

logger = logging.getLogger(__name__)

# Noms des fichiers de progression
FILTERING_STATE_FILE = "filtering_progress.jsonl"
IMPORT_STATE_FILE = "import_progress.jsonl"


# =============================================================================
# FILTERING STATE (filtrage LLM)
# =============================================================================

def append_filtering_state(session_folder: str, entry: Dict[str, Any]) -> None:
    """
    Append une entree au fichier filtering_progress.jsonl (thread-safe).

    Utilise file locking (fcntl) pour garantir l'integrite en cas d'acces concurrent.

    Args:
        session_folder: Chemin absolu vers le dossier de session
        entry: Dictionnaire avec les champs:
            - index: int (position dans la liste originale)
            - relevant: bool (True/False/None si erreur)
            - relevance_score: Optional[int]
            - relevance_reason: Optional[str]
            - zotero_data: Optional[Dict]
            - web_source: Optional[str]
            - reason: Optional[str] (si skipped)
            - error: Optional[str] (si erreur)

    Example:
        >>> append_filtering_state("/uploads/pop_abc123_Project", {
        ...     "index": 0,
        ...     "relevant": True,
        ...     "relevance_score": 95,
        ...     "zotero_data": {"title": "..."},
        ...     "web_source": "pdf"
        ... })
    """
    path = os.path.join(session_folder, FILTERING_STATE_FILE)
    entry["ts"] = datetime.utcnow().isoformat() + "Z"

    try:
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        logger.debug(f"Saved filtering state for index {entry.get('index')}")
    except Exception as e:
        logger.error(f"Failed to append filtering state: {e}")
        raise


def read_filtering_state(session_folder: str) -> List[Dict[str, Any]]:
    """
    Lit toutes les entrees du fichier filtering_progress.jsonl.

    Tolerant aux lignes corrompues (skip avec warning).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        Liste de dictionnaires representant chaque entree.
        Retourne liste vide si fichier inexistant.

    Example:
        >>> entries = read_filtering_state("/uploads/pop_abc123_Project")
        >>> len(entries)
        45
    """
    path = os.path.join(session_folder, FILTERING_STATE_FILE)

    if not os.path.exists(path):
        return []

    entries = []
    line_num = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping corrupted line {line_num} in {FILTERING_STATE_FILE}: {e}"
                    )
                    continue
    except Exception as e:
        logger.error(f"Failed to read filtering state: {e}")
        return entries  # Return what we have so far

    return entries


def get_processed_indices(session_folder: str) -> Set[int]:
    """
    Retourne les indices deja traites (pour reprise).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        Set des indices de citations deja traitees.

    Example:
        >>> processed = get_processed_indices("/uploads/pop_abc123_Project")
        >>> 5 in processed
        True
        >>> 100 in processed
        False
    """
    entries = read_filtering_state(session_folder)
    return {e["index"] for e in entries if "index" in e}


def get_filtering_stats(session_folder: str) -> Dict[str, int]:
    """
    Calcule les statistiques de filtrage depuis l'etat sauvegarde.

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        Dictionnaire avec:
            - total: Nombre total d'entrees
            - relevant: Nombre de citations pertinentes
            - skipped: Nombre de citations ignorees
            - errors: Nombre d'erreurs
    """
    entries = read_filtering_state(session_folder)

    relevant = sum(1 for e in entries if e.get("relevant") is True)
    errors = sum(1 for e in entries if e.get("error") is not None)
    skipped = len(entries) - relevant - errors

    return {
        "total": len(entries),
        "relevant": relevant,
        "skipped": skipped,
        "errors": errors
    }


def rebuild_preview_from_state(
    session_folder: str,
    citations: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reconstruit preview.json depuis filtering_progress.jsonl.

    Utile apres un crash pour recuperer l'etat sans re-traiter.

    Args:
        session_folder: Chemin absolu vers le dossier de session
        citations: Liste complete des citations originales (pour enrichissement)

    Returns:
        Dictionnaire compatible avec le format preview.json:
            {
                "relevant": [...],
                "skipped": [...]
            }

    Example:
        >>> citations = parse_pop_json("publishorperish.json")
        >>> preview = rebuild_preview_from_state(session_folder, citations)
        >>> len(preview["relevant"])
        20
    """
    entries = read_filtering_state(session_folder)
    preview: Dict[str, List[Dict[str, Any]]] = {"relevant": [], "skipped": []}

    for entry in entries:
        idx = entry.get("index")
        if idx is None:
            continue

        # Recuperer la citation originale si disponible
        citation = citations[idx] if idx < len(citations) else {}

        if entry.get("relevant") is True:
            preview["relevant"].append({
                "index": idx,
                "citation": citation,
                "zotero_data": entry.get("zotero_data"),
                "relevance_score": entry.get("relevance_score"),
                "relevance_reason": entry.get("relevance_reason"),
                "web_source": entry.get("web_source")
            })
        else:
            # Skipped ou Error
            reason = entry.get("reason") or entry.get("error") or "Skipped"
            preview["skipped"].append({
                "index": idx,
                "citation": citation,
                "reason": reason
            })

    return preview


def clear_filtering_state(session_folder: str) -> bool:
    """
    Supprime le fichier d'etat de filtrage (pour reset).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        True si supprime, False si n'existait pas
    """
    path = os.path.join(session_folder, FILTERING_STATE_FILE)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Cleared filtering state: {path}")
        return True
    return False


# =============================================================================
# IMPORT STATE (import Zotero)
# =============================================================================

def append_import_state(session_folder: str, entry: Dict[str, Any]) -> None:
    """
    Append une entree au fichier import_progress.jsonl (thread-safe).

    Args:
        session_folder: Chemin absolu vers le dossier de session
        entry: Dictionnaire avec les champs:
            - index: int (position dans la liste relevant)
            - imported: bool (True si succes)
            - zotero_key: Optional[str] (cle Zotero si cree/modifie)
            - action: Optional[str] ("created", "updated", "skipped")
            - error: Optional[str] (si erreur)

    Example:
        >>> append_import_state("/uploads/pop_abc123_Project", {
        ...     "index": 0,
        ...     "imported": True,
        ...     "zotero_key": "ABC123XY",
        ...     "action": "created"
        ... })
    """
    path = os.path.join(session_folder, IMPORT_STATE_FILE)
    entry["ts"] = datetime.utcnow().isoformat() + "Z"

    try:
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        logger.debug(f"Saved import state for index {entry.get('index')}")
    except Exception as e:
        logger.error(f"Failed to append import state: {e}")
        raise


def read_import_state(session_folder: str) -> List[Dict[str, Any]]:
    """
    Lit toutes les entrees du fichier import_progress.jsonl.

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        Liste de dictionnaires representant chaque entree.
    """
    path = os.path.join(session_folder, IMPORT_STATE_FILE)

    if not os.path.exists(path):
        return []

    entries = []
    line_num = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping corrupted line {line_num} in {IMPORT_STATE_FILE}: {e}"
                    )
                    continue
    except Exception as e:
        logger.error(f"Failed to read import state: {e}")

    return entries


def get_imported_indices(session_folder: str) -> Set[int]:
    """
    Retourne les indices deja importes avec succes (pour reprise).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        Set des indices de citations deja importees dans Zotero.
    """
    entries = read_import_state(session_folder)
    return {
        e["index"] for e in entries
        if "index" in e and e.get("imported") is True
    }


def get_import_stats(session_folder: str) -> Dict[str, int]:
    """
    Calcule les statistiques d'import depuis l'etat sauvegarde.

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        Dictionnaire avec:
            - total: Nombre total d'entrees
            - created: Nombre de citations creees
            - updated: Nombre de citations mises a jour
            - errors: Nombre d'erreurs
    """
    entries = read_import_state(session_folder)

    created = sum(1 for e in entries if e.get("action") == "created")
    updated = sum(1 for e in entries if e.get("action") == "updated")
    errors = sum(1 for e in entries if e.get("error") is not None)

    return {
        "total": len(entries),
        "created": created,
        "updated": updated,
        "errors": errors
    }


def clear_import_state(session_folder: str) -> bool:
    """
    Supprime le fichier d'etat d'import (pour reset).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        True si supprime, False si n'existait pas
    """
    path = os.path.join(session_folder, IMPORT_STATE_FILE)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Cleared import state: {path}")
        return True
    return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def has_filtering_progress(session_folder: str) -> bool:
    """
    Verifie si un filtrage partiel existe (pour detection reprise).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        True si au moins une entree de filtrage existe
    """
    path = os.path.join(session_folder, FILTERING_STATE_FILE)
    return os.path.exists(path) and os.path.getsize(path) > 0


def has_import_progress(session_folder: str) -> bool:
    """
    Verifie si un import partiel existe (pour detection reprise).

    Args:
        session_folder: Chemin absolu vers le dossier de session

    Returns:
        True si au moins une entree d'import existe
    """
    path = os.path.join(session_folder, IMPORT_STATE_FILE)
    return os.path.exists(path) and os.path.getsize(path) > 0


def get_session_progress_summary(
    session_folder: str,
    total_citations: int
) -> Dict[str, Any]:
    """
    Retourne un resume complet de la progression d'une session.

    Args:
        session_folder: Chemin absolu vers le dossier de session
        total_citations: Nombre total de citations dans le JSON original

    Returns:
        Dictionnaire avec:
            - filtering: {progress, total, relevant, skipped, can_resume}
            - import: {progress, total, created, updated, can_resume}
    """
    filtering_stats = get_filtering_stats(session_folder)
    import_stats = get_import_stats(session_folder)

    return {
        "filtering": {
            "progress": filtering_stats["total"],
            "total": total_citations,
            "relevant": filtering_stats["relevant"],
            "skipped": filtering_stats["skipped"],
            "errors": filtering_stats["errors"],
            "can_resume": filtering_stats["total"] < total_citations
        },
        "import": {
            "progress": import_stats["total"],
            "total": filtering_stats["relevant"],  # Only relevant are importable
            "created": import_stats["created"],
            "updated": import_stats["updated"],
            "errors": import_stats["errors"],
            "can_resume": import_stats["total"] < filtering_stats["relevant"]
        }
    }
