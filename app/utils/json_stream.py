"""
Résumés des fichiers JSON de chunks (listes), en mémoire bornée et en cache.

Un fichier ``output_chunks_with_embeddings.json`` peut peser plusieurs Go
(livre de 1 800 pages : 1,8 Go, vecteurs de 3 072 nombres). ``json.load``
le transforme en objets Python plusieurs fois plus gros : dans un conteneur
limité à 8 Go, le processus web était tué (OOM) dès l'ouverture de la
session. La lecture en flux vient de ``scripts/rad_json_stream.py`` : la
mémoire reste de l'ordre d'un chunk.

Les résumés (nombre d'éléments, champs d'espace d'embeddings) sont gardés en
cache par (chemin, taille, date de modification). Au-delà d'un seuil, le
calcul part dans un fil d'arrière-plan et l'appelant reçoit ``None`` tant
qu'il n'est pas fini : la page d'une session s'ouvre sans attendre.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from scripts.rad_json_stream import count_json_array, iter_json_array
except ImportError:  # scripts/ itself on sys.path
    from rad_json_stream import count_json_array, iter_json_array

logger = logging.getLogger(__name__)

SYNC_MAX_BYTES = 256 * 1024 * 1024
"""Au-delà, le résumé est calculé en arrière-plan (``cached_summary``)."""

_CACHE_MAX = 128
_CACHE: "OrderedDict[Tuple[str, int, int, Tuple[str, ...]], Dict[str, Any]]" = OrderedDict()
_IN_FLIGHT: set = set()
_LOCK = threading.Lock()


def summarize_json_array(path: str, fields: Sequence[str] = ()) -> Dict[str, Any]:
    """
    Résumé d'une liste JSON de chunks : nombre d'éléments et premiers ``fields`` trouvés.

    Args:
        path: Fichier JSON.
        fields: Champs à relever sur le premier élément (dict) qui en porte au moins un.

    Returns:
        ``{"count": n, "fields": {...}}``.

    Raises:
        ValueError: Pas une liste JSON valide.
    """
    count = 0
    found: Dict[str, Any] = {}
    for item in iter_json_array(path):
        count += 1
        if fields and not found and isinstance(item, dict):
            found = {name: item[name] for name in fields if name in item}
    return {"count": count, "fields": found}


def _cache_key(path: str, fields: Sequence[str]) -> Optional[Tuple[str, int, int, Tuple[str, ...]]]:
    """Clé de cache (chemin absolu, taille, date de modification, champs), ou ``None`` si le fichier manque."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), stat.st_size, stat.st_mtime_ns, tuple(fields))


def _remember(key: Tuple[str, int, int, Tuple[str, ...]], value: Dict[str, Any]) -> None:
    """Range un résumé dans le cache (les plus anciens sortent au-delà de ``_CACHE_MAX``)."""
    with _LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def _background_summary(path: str, fields: Sequence[str], key: Tuple[str, int, int, Tuple[str, ...]]) -> None:
    """Calcule un résumé dans un fil d'arrière-plan et le met en cache (une erreur est journalisée)."""
    try:
        _remember(key, summarize_json_array(path, fields))
    except Exception as exc:  # fichier en cours d'écriture, invalide : réessayé à la prochaine demande
        logger.warning(f"Résumé JSON impossible pour {os.path.basename(path)} : {type(exc).__name__}")
    finally:
        with _LOCK:
            _IN_FLIGHT.discard(key)


def cached_summary(path: str, fields: Sequence[str] = (), sync_max_bytes: int = SYNC_MAX_BYTES
                   ) -> Optional[Dict[str, Any]]:
    """
    Résumé d'une liste JSON, depuis le cache, calculé sur place, ou lancé en arrière-plan.

    À appeler hors de la boucle d'événements (``asyncio.to_thread``) : un petit
    fichier est lu sur place.

    Args:
        path: Fichier JSON.
        fields: Champs à relever (voir ``summarize_json_array``).
        sync_max_bytes: Taille au-delà de laquelle le calcul part en arrière-plan.

    Returns:
        ``{"count", "fields"}``, ou ``None`` (calcul en cours, fichier absent ou illisible).
    """
    key = _cache_key(path, fields)
    if key is None:
        return None
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    if key[1] <= sync_max_bytes:
        try:
            value = summarize_json_array(path, fields)
        except (OSError, ValueError):
            return None
        _remember(key, value)
        return value
    with _LOCK:
        if key in _IN_FLIGHT:
            return None
        _IN_FLIGHT.add(key)
    threading.Thread(target=_background_summary, args=(path, tuple(fields), key), daemon=True,
                     name="json-summary").start()
    return None


__all__ = ["count_json_array", "cached_summary", "iter_json_array", "summarize_json_array"]
