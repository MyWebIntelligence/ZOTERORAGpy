"""Collections Albert (4e cible vectorielle) : helpers purs — stdlib seulement.

Ce module ne fait **aucun appel réseau** et n'importe ni httpx ni le client :
les routes (``/upload_db``) et Celery peuvent l'importer pour résoudre
l'entrée d'une session (``resolve_albert_input``) sans charger le client.
L'envoi lui-même (``insert_to_albert``, ``_AlbertDedupAdapter``) vit dans
``scripts/rad_vectordb.py``.

Contrats (lot 5 du sprint Albert) :

* **métadonnées en liste blanche** : ``ALBERT_METADATA_FIELDS`` (10 champs au
  plus, ``content_id``, ``content_hash``, ``chunk_index`` et
  ``DEDUP_META_FIELDS`` imposés, jamais ``path``), validée par
  ``effective_metadata_fields`` **avant** tout appel réseau ;
* **valeurs scalaires conformes à l'API** (D16) : chaîne de 1 à 255
  caractères, entier ou flottant de valeur absolue au plus 1e16, booléen ;
  ``albert_scalar`` écarte ``''``, ``None``, NaN et l'infini, convertit un
  flottant entier en ``int``, un nombre trop grand en chaîne, joint les
  listes et tronque à 255 caractères ;
* **idempotence toujours active** : ``content_id_for`` donne un identifiant
  adressé par contenu stable d'un run à l'autre, même sans les champs de la
  dédup (``DEDUP_ENABLED=0``, où l'``id`` du chunk porte un ``doc_id``
  aléatoire). Limite : un nouveau chunking recodé produit d'autres textes,
  donc d'autres identifiants ;
* **un document Albert par document source** : ``document_name_for`` donne
  ``ragpy:{itemKey}:{basename(filename)}`` (repli sur le titre), stable d'un
  run à l'autre ;
* **manifeste** (``albert_manifest.jsonl``) : une ligne par tranche envoyée
  avec succès (``manifest_row``), écrite en append puis flush par le
  connecteur ; un rollback ajoute une ligne d'événement
  (``manifest_rollback_row``) pour que le relevé d'audit ne liste pas comme
  stockés des chunks supprimés ; ``read_manifest`` ne renvoie que les lignes
  de tranche, sauf demande explicite ;
* **collections privées seulement** : ``select_collection`` applique la
  correspondance exacte du nom côté client (D14 : les doublons de nom sont
  acceptés par le serveur) et refuse plusieurs correspondances en listant
  leurs ids. La visibilité doit être **confirmée** ``private`` : absente ou
  inconnue, l'envoi est refusé (``is_private`` échoue fermé).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from scripts import rad_dedup
except ImportError:  # contexte CLI : scripts/ est sur sys.path
    import rad_dedup

from .config import (
    DEFAULT_METADATA_FIELDS,
    FIELD_ENV_NAMES,
    FORBIDDEN_METADATA_FIELDS,
    validate_metadata_fields,
)

MANIFEST_FILENAME = "albert_manifest.jsonl"
"""Nom du manifeste d'envoi, écrit à côté du fichier d'entrée."""

CHUNKS_PER_SLICE = 64
"""Chunks au plus par ``POST /v1/documents/{id}/chunks`` (D16 : 65 → 422)."""

SCALAR_MAX_CHARS = 255
"""Longueur maximale d'une valeur de métadonnée chaîne (D16 : 256 → 422)."""

SCALAR_MAX_ABS = 1e16
"""Valeur absolue maximale d'une métadonnée numérique (au-delà : chaîne)."""

NAME_MAX_CHARS = 255
"""Longueur maximale d'un nom de document Albert produit par RAGpy."""

DOCUMENT_NAME_PREFIX = "ragpy:"
"""Préfixe des noms de documents créés par RAGpy."""

UNTITLED_LABEL = "sans-titre"
"""Libellé d'un document sans ``itemKey``, sans fichier et sans titre."""

VECTOR_KEYS = ("embedding", "sparse_embedding")
"""Clés retirées avant l'envoi : les embeddings sont calculés côté serveur."""

INPUT_CANDIDATES = (
    "output_chunks_with_embeddings_sparse.json",
    "output_chunks_with_embeddings.json",
    "output_chunks.json",
)
"""Fichiers d'entrée acceptés dans une session, par ordre de préférence."""

__all__ = [
    "CHUNKS_PER_SLICE",
    "DEFAULT_METADATA_FIELDS",
    "DOCUMENT_NAME_PREFIX",
    "INPUT_CANDIDATES",
    "MANIFEST_FILENAME",
    "NAME_MAX_CHARS",
    "SCALAR_MAX_ABS",
    "SCALAR_MAX_CHARS",
    "VECTOR_KEYS",
    "AlbertTargetError",
    "albert_scalar",
    "chunk_text",
    "content_id_for",
    "document_name_for",
    "effective_metadata_fields",
    "field_value",
    "group_by_document",
    "is_private",
    "iter_slices",
    "manifest_path_for",
    "manifest_rollback_row",
    "manifest_row",
    "manifest_row_age_s",
    "read_manifest",
    "resolve_albert_input",
    "sanitize_metadata",
    "select_collection",
    "strip_vectors",
    "visibility_of",
]


class AlbertTargetError(ValueError):
    """Cible Albert ambiguë ou refusée (noms en double, collection non privée…).

    Attributes:
        ids: identifiants Albert en cause (collections ou documents).
    """

    def __init__(self, message: str, *, ids: Optional[Iterable[Any]] = None) -> None:
        """Construit l'erreur.

        Args:
            message: message français, repris tel quel.
            ids: identifiants à citer (collections ou documents en double).
        """
        self.ids: List[Any] = list(ids or [])
        super().__init__(message)


# ---------------------------------------------------------------------------
# Liste blanche des métadonnées
# ---------------------------------------------------------------------------


def _split_names(value: Any) -> Tuple[str, ...]:
    """Découpe une liste de noms séparés par des virgules (sans vide ni doublon)."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else list(value)
    out: List[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def effective_metadata_fields(env: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """Liste blanche effective des métadonnées envoyées aux collections Albert.

    Lit ``ALBERT_METADATA_FIELDS`` (vide ou absente : ``DEFAULT_METADATA_FIELDS``)
    et la valide avec ``DEDUP_META_FIELDS`` (``rad_dedup.DedupConfig``) : une
    surcharge incomplète (sans ``content_id``, ``content_hash``,
    ``chunk_index`` ou un champ de ``DEDUP_META_FIELDS``), de plus de 10
    champs ou contenant ``path`` est refusée, avant tout appel réseau.

    Args:
        env: variables à lire (``os.environ`` par défaut).

    Returns:
        Le tuple des champs, dans l'ordre configuré.

    Raises:
        ValueError: liste refusée (message français listant les causes).
    """
    env = os.environ if env is None else env
    fields = _split_names(env.get(FIELD_ENV_NAMES["metadata_fields"])) or DEFAULT_METADATA_FIELDS
    dedup_fields = rad_dedup.DedupConfig.from_env(dict(env)).meta_fields
    return validate_metadata_fields(fields, dedup_fields)


def _is_blank(value: Any) -> bool:
    """Vrai pour ``None``, une chaîne vide ou blanche, et un flottant non fini."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _truncate(text: str) -> str:
    """Tronque une chaîne à ``SCALAR_MAX_CHARS`` caractères."""
    return text[:SCALAR_MAX_CHARS]


def albert_scalar(value: Any) -> Optional[Union[str, int, float, bool]]:
    """Convertit une valeur en métadonnée scalaire acceptée par Albert.

    Règles (D16) : ``None``, ``''`` (ou blanc), NaN et l'infini sont écartés
    (``None`` renvoyé) ; un booléen est gardé ; un flottant entier devient un
    ``int`` ; un nombre de valeur absolue supérieure à 1e16 devient une
    chaîne ; une liste est jointe par ``', '`` ; un dictionnaire est sérialisé
    en JSON ; toute chaîne est nettoyée de ses blancs périphériques puis
    tronquée à 255 caractères.

    Args:
        value: valeur brute du chunk.

    Returns:
        La valeur conforme, ou ``None`` si elle doit être omise.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= SCALAR_MAX_ABS else _truncate(str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            number = int(value)
            return number if abs(number) <= SCALAR_MAX_ABS else _truncate(str(number))
        return value if abs(value) <= SCALAR_MAX_ABS else _truncate(repr(value))
    if isinstance(value, str):
        text = value.strip()
        return _truncate(text) if text else None
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(item).strip() for item in value if not _is_blank(item)]
        joined = ", ".join(part for part in parts if part)
        return _truncate(joined) if joined else None
    if isinstance(value, Mapping):
        if not value:
            return None
        return _truncate(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str))
    text = str(value).strip()
    return _truncate(text) if text else None


# ---------------------------------------------------------------------------
# Champs d'un chunk
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")


def _first(chunk: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Première valeur non vide parmi ``names`` (``None`` sinon)."""
    for name in names:
        value = chunk.get(name)
        if not _is_blank(value):
            return value
    return None


def _basename(value: Any) -> str:
    """Nom de fichier seul (séparateurs ``/`` et ``\\``), jamais un chemin local."""
    if _is_blank(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    return os.path.basename(text.rstrip("/"))


def _year_of(value: Any) -> Optional[int]:
    """Année (entier) lue dans une date libre, ou ``None``."""
    if _is_blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = int(value)
        return number if 1000 <= number <= 2099 else None
    match = _YEAR_RE.search(str(value))
    return int(match.group(1)) if match else None


def chunk_text(chunk: Mapping[str, Any]) -> str:
    """Texte à envoyer d'un chunk (``text``, sinon ``chunk_text``, sinon ``''``)."""
    for name in ("text", "chunk_text"):
        value = chunk.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def content_id_for(chunk: Mapping[str, Any]) -> str:
    """Identifiant adressé par contenu d'un chunk, stable d'un run à l'autre.

    Ordre :

    1. ``chunk['id']`` s'il est déjà adressé par contenu (forme
       ``{hash[:16]}_{chunk_index}``, cohérente avec ``content_hash``) ;
    2. sinon ``rad_dedup.content_id(chunk['content_hash'], chunk_index)`` si
       le hash existe ;
    3. sinon ``rad_dedup.content_id(rad_dedup.content_hash(texte), chunk_index)``
       (cas ``DEDUP_ENABLED=0`` : l'``id`` vaut ``{doc_id aléatoire}_{idx}``).

    Limite documentée : un nouveau chunking recodé produit d'autres textes,
    donc d'autres identifiants.

    Args:
        chunk: chunk du fichier d'entrée.

    Returns:
        L'identifiant ``{hash[:16]}_{chunk_index}``.
    """
    index = chunk.get("chunk_index")
    chash = chunk.get("content_hash")
    current = chunk.get("id")
    if isinstance(current, str) and re.fullmatch(r"[0-9a-f]{16}_\S+", current):
        if not chash or current == rad_dedup.content_id(str(chash), index):
            return current
    if isinstance(chash, str) and chash.strip():
        return rad_dedup.content_id(chash.strip(), index)
    return rad_dedup.content_id(rad_dedup.content_hash(chunk_text(chunk)), index)


def field_value(chunk: Mapping[str, Any], name: str) -> Any:
    """Valeur brute d'un champ de la liste blanche pour un chunk.

    Correspondances : ``content_id`` ← ``content_id_for`` ; ``content_hash``
    ← le champ du chunk, sinon le hash du texte ; ``item_key`` ← ``item_key``
    ou ``itemKey`` ; ``doi`` ← ``doi``, sinon ``url`` (même clé) ; ``year`` ←
    ``year``, sinon l'année lue dans ``date`` ; ``filename`` ← nom de fichier
    seul ; ``path`` n'est jamais lu. Tout autre champ est lu tel quel.

    Args:
        chunk: chunk du fichier d'entrée.
        name: nom du champ de la liste blanche.

    Returns:
        La valeur brute (``None`` si absente).
    """
    if name.lower() in FORBIDDEN_METADATA_FIELDS:
        return None
    if name == "content_id":
        return content_id_for(chunk)
    if name == "content_hash":
        chash = chunk.get("content_hash")
        if isinstance(chash, str) and chash.strip():
            return chash.strip()
        return rad_dedup.content_hash(chunk_text(chunk))
    if name == "item_key":
        return _first(chunk, ("item_key", "itemKey"))
    if name == "doi":
        return _first(chunk, ("doi", "url"))
    if name == "year":
        year = chunk.get("year")
        return year if not _is_blank(year) else _year_of(chunk.get("date"))
    if name == "filename":
        return _basename(chunk.get("filename")) or None
    return chunk.get(name)


def sanitize_metadata(
    chunk: Mapping[str, Any],
    fields: Optional[Sequence[str]] = None,
    *,
    content_id: Optional[str] = None,
) -> Dict[str, Union[str, int, float, bool]]:
    """Métadonnées d'un chunk prêtes pour ``add_chunks`` (liste blanche, scalaires).

    ``path`` n'est jamais envoyé (même s'il figure dans ``fields``) ; ``doi``
    se replie sur ``url`` sous la même clé ; les valeurs passent par
    ``albert_scalar`` (valeurs vides omises).

    Args:
        chunk: chunk du fichier d'entrée.
        fields: liste blanche (défaut : ``effective_metadata_fields()``).
        content_id: identifiant déjà calculé (défaut : ``content_id_for``).

    Returns:
        Le dictionnaire de métadonnées (10 entrées au plus avec une liste
        blanche validée).
    """
    names = tuple(fields) if fields is not None else effective_metadata_fields()
    out: Dict[str, Union[str, int, float, bool]] = {}
    for name in names:
        if not isinstance(name, str) or not name or name.lower() in FORBIDDEN_METADATA_FIELDS:
            continue
        if name in out:
            continue
        raw = content_id if (name == "content_id" and content_id) else field_value(chunk, name)
        value = albert_scalar(raw)
        if value is not None:
            out[name] = value
    return out


def _clean_label(value: Any) -> str:
    """Libellé sur une ligne (blancs réduits), ``''`` si vide."""
    if _is_blank(value):
        return ""
    return " ".join(str(value).split())


def document_name_for(chunk: Mapping[str, Any]) -> str:
    """Nom du document Albert d'un chunk : ``ragpy:{itemKey}:{basename(filename)}``.

    Sans nom de fichier, le titre le remplace ; sans ``itemKey``, le premier
    segment est vide ; sans aucun des trois, ``sans-titre``. Au-delà de 255
    caractères, le nom est tronqué et suffixé d'un hash (``~{sha256[:16]}``)
    pour rester unique. Le ``doc_id`` aléatoire n'y entre jamais : le nom est
    stable d'un run à l'autre.

    Args:
        chunk: chunk du fichier d'entrée.

    Returns:
        Le nom du document.
    """
    item = _clean_label(_first(chunk, ("itemKey", "item_key")))
    label = _clean_label(_basename(chunk.get("filename"))) or _clean_label(chunk.get("title"))
    if not item and not label:
        label = UNTITLED_LABEL
    name = f"{DOCUMENT_NAME_PREFIX}{item}:{label}"
    if len(name) > NAME_MAX_CHARS:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        name = name[: NAME_MAX_CHARS - len(digest) - 1] + "~" + digest
    return name


def strip_vectors(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """Copie du chunk sans ``embedding`` ni ``sparse_embedding`` (aucun vecteur envoyé)."""
    return {key: value for key, value in chunk.items() if key not in VECTOR_KEYS}


def group_by_document(chunks: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Regroupe des chunks par ``document_name_for``, dans l'ordre de première apparition."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for chunk in chunks:
        groups.setdefault(document_name_for(chunk), []).append(chunk)
    return groups


def iter_slices(items: Sequence[Any], size: int = CHUNKS_PER_SLICE) -> Iterator[List[Any]]:
    """Tranches successives de ``size`` éléments au plus (plafonné à 64)."""
    step = max(1, min(int(size), CHUNKS_PER_SLICE))
    for start in range(0, len(items), step):
        yield list(items[start:start + step])


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def visibility_of(collection: Any) -> Optional[str]:
    """Visibilité déclarée d'une collection (minuscules), ou ``None`` si absente ou vide.

    Args:
        collection: collection telle que renvoyée par l'API.

    Returns:
        ``"private"``, ``"public"``… ou ``None`` quand la réponse ne la porte pas.
    """
    if not isinstance(collection, Mapping):
        return None
    visibility = collection.get("visibility")
    if _is_blank(visibility):
        return None
    return str(visibility).strip().lower()


def is_private(collection: Any) -> bool:
    """Vrai seulement si la visibilité est explicitement ``private`` (échec fermé).

    Une visibilité absente, vide ou inattendue compte comme **non** privée :
    RAGpy n'envoie jamais vers une collection dont la visibilité n'a pas été
    confirmée.
    """
    return visibility_of(collection) == "private"


def select_collection(listing: Iterable[Any], name: str) -> Optional[Dict[str, Any]]:
    """Choisit la collection privée de nom **exactement** ``name`` (D14).

    Seules comptent les collections dont la visibilité ``private`` est
    confirmée (``is_private``).

    Args:
        listing: collections listées (toutes pages).
        name: nom demandé.

    Returns:
        La collection unique correspondante, ou ``None`` si aucune.

    Raises:
        AlbertTargetError: plusieurs collections privées de ce nom (ids
            listés, aucun choix implicite), ou seulement des collections de ce
            nom non privées ou de visibilité inconnue (envoi refusé).
    """
    matches = [dict(c) for c in listing if isinstance(c, Mapping) and c.get("name") == name]
    private = [c for c in matches if is_private(c)]
    if len(private) > 1:
        ids = [c.get("id") for c in private]
        raise AlbertTargetError(
            f"{len(private)} collections Albert privées portent le nom « {name} » "
            f"(ids : {', '.join(str(i) for i in ids)}) : aucun choix implicite ; "
            "préciser --albert-collection-id.",
            ids=ids,
        )
    if private:
        return private[0]
    if matches:
        ids = [c.get("id") for c in matches]
        id_list = ", ".join(str(i) for i in ids)
        if all(visibility_of(c) is None for c in matches):
            raise AlbertTargetError(
                f"La collection Albert « {name} » (ids : {id_list}) : visibilité inconnue, "
                "envoi refusé (RAGpy n'envoie que vers des collections dont la visibilité "
                "« private » est confirmée).",
                ids=ids,
            )
        raise AlbertTargetError(
            f"La collection Albert « {name} » n'est pas privée "
            f"(ids : {id_list}) : RAGpy n'envoie que vers des "
            "collections privées.",
            ids=ids,
        )
    return None


# ---------------------------------------------------------------------------
# Manifeste
# ---------------------------------------------------------------------------


def manifest_path_for(directory: str) -> str:
    """Chemin du manifeste (``albert_manifest.jsonl``) dans ``directory``."""
    return os.path.join(directory, MANIFEST_FILENAME)


def manifest_row(
    *,
    collection_id: Any,
    document_id: Any,
    document_name: str,
    slice_index: int,
    content_ids: Sequence[str],
    chunk_ids: Optional[Sequence[Any]] = None,
    collection_name: Optional[str] = None,
    document_created: bool = False,
    reconciled: bool = False,
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Ligne du manifeste pour une tranche envoyée avec succès.

    D16 : l'API renvoie les ids des chunks créés ; ils sont recopiés dans
    ``chunk_ids`` (``None`` s'ils sont inconnus, par exemple après un
    rapprochement). Document, tranche et nombre sont toujours présents.

    Args:
        collection_id: collection cible.
        document_id: document Albert.
        document_name: nom du document (``document_name_for``).
        slice_index: rang de la tranche dans le document (à partir de 1).
        content_ids: identifiants adressés par contenu des chunks envoyés.
        chunk_ids: ids Albert renvoyés par l'envoi (D16), ou ``None``.
        collection_name: nom de la collection, si connu.
        document_created: vrai si le document a été créé pendant ce run.
        reconciled: vrai si la tranche a été confirmée par relecture après
            une issue incertaine.
        ts: horodatage ISO 8601 (défaut : maintenant, UTC).

    Returns:
        Le dictionnaire à sérialiser en une ligne JSON.
    """
    row: Dict[str, Any] = {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "collection_id": int(collection_id),
        "collection_name": collection_name,
        "document_id": int(document_id),
        "document_name": document_name,
        "document_created": bool(document_created),
        "slice": int(slice_index),
        "count": len(content_ids),
        "content_ids": list(content_ids),
        "chunk_ids": [int(i) for i in chunk_ids] if chunk_ids else None,
    }
    if reconciled:
        row["reconciled"] = True
    return row


def manifest_rollback_row(
    *,
    collection_id: Any,
    document_id: Any,
    document_name: str,
    chunks_removed: int,
    collection_name: Optional[str] = None,
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Ligne d'événement du manifeste : document supprimé par rollback.

    Elle suit les lignes de tranche du même document écrites plus tôt dans le
    run : le relevé d'audit montre ainsi que ces chunks ne sont plus stockés
    par la DINUM. Elle ne porte pas de ``content_ids`` et ``read_manifest``
    l'écarte par défaut.

    Args:
        collection_id: collection cible.
        document_id: document Albert supprimé.
        document_name: nom du document (``document_name_for``).
        chunks_removed: nombre de chunks envoyés puis retirés par la suppression.
        collection_name: nom de la collection, si connu.
        ts: horodatage ISO 8601 (défaut : maintenant, UTC).

    Returns:
        Le dictionnaire à sérialiser en une ligne JSON.
    """
    return {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "event": "rollback",
        "collection_id": int(collection_id),
        "collection_name": collection_name,
        "document_id": int(document_id),
        "document_name": document_name,
        "chunks_removed": int(chunks_removed),
    }


def manifest_row_age_s(row: Mapping[str, Any], now: Optional[datetime] = None) -> Optional[float]:
    """Âge (secondes) d'une ligne de manifeste d'après son ``ts``.

    Un horodatage sans fuseau est lu en UTC ; un horodatage futur donne un âge
    négatif.

    Args:
        row: ligne lue par ``read_manifest``.
        now: instant de référence (défaut : maintenant, UTC).

    Returns:
        L'âge en secondes, ou ``None`` si ``ts`` est absent ou illisible.
    """
    raw = row.get("ts") if isinstance(row, Mapping) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        stamp = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    reference = now if now is not None else datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return (reference - stamp).total_seconds()


def read_manifest(path: str, *, include_events: bool = False) -> List[Dict[str, Any]]:
    """Lit un manifeste JSONL (lignes illisibles ignorées ; fichier absent : ``[]``).

    Args:
        path: chemin du manifeste.
        include_events: vrai pour garder aussi les lignes d'événement
            (``event``, par exemple un rollback) ; par défaut, seules les
            lignes de tranche sont renvoyées.

    Returns:
        Les lignes, dans l'ordre du fichier.
    """
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                if "event" in row and not include_events:
                    continue
                rows.append(row)
    except OSError:
        return []
    return rows


# ---------------------------------------------------------------------------
# Entrée d'une session
# ---------------------------------------------------------------------------


def resolve_albert_input(session_dir: Any) -> Optional[str]:
    """Fichier d'entrée d'un envoi Albert dans une session.

    Ordre : ``output_chunks_with_embeddings_sparse.json``, puis
    ``output_chunks_with_embeddings.json``, puis ``output_chunks.json`` (les
    embeddings sont calculés côté serveur : aucun vecteur n'est requis).
    Réutilisé par la route ``/upload_db`` et par Celery.

    Args:
        session_dir: dossier de la session.

    Returns:
        Le chemin du premier fichier présent, ou ``None``.
    """
    if not session_dir:
        return None
    directory = os.fspath(session_dir)
    if not os.path.isdir(directory):
        return None
    for name in INPUT_CANDIDATES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None
