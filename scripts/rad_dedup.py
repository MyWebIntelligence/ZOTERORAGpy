"""scripts/rad_dedup.py — Déduplication des chunks à l'insertion vectorielle.

Module **fondation**, STDLIB UNIQUEMENT (aucune dépendance base vectorielle), de
sorte que le *write-path* (``rad_chunk.py``) et le *read-path* (les connecteurs de
``rad_vectordb.py``) partagent **exactement** la même normalisation et le même hash.

Importé par :
  - ``scripts/rad_chunk.py``   — calcule ``content_hash`` sur le texte BRUT pré-recodage
  - ``scripts/rad_vectordb.py`` — ``dedup_filter`` + adapters concrets par base
  - ``tests/test_dedup.py``

Cascade « du moins cher au plus cher » (chaque tier ne voit que les survivants du
précédent) :

    Tier 0  content_hash = sha256(normalize(texte BRUT pré-recodage))  ~µs, calculé dans rad_chunk
    Tier 1  seen-set mémoire des content_hash du run                   gratuit, couvre l'eventual consistency
    Tier 2  existence serveur-side batchée (1 RTT / lot)               fetch-by-id (Pinecone) / filtre payload (Weaviate/Qdrant)
    Tier 3  confirmation similarité d'embedding (OFF par défaut)       gated métadonnées + Jaccard + cap survivants

Invariant porteur : le hash et la clé d'existence **ne touchent JAMAIS** ``doc_id``
(champ aléatoire = cause racine du bug). ``doc_id`` reste au journal pour la
traçabilité seulement. Un refus exige **toujours** une corroboration métadonnée ;
``chunk_index`` est un corroborateur **souple** en Tier 2 (loggé, non bloquant) et
une **garde dure** en Tier 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger("rad_dedup")

# ---------------------------------------------------------------------------
# Tier 0 — normalisation conservatrice (FR-safe) + hash + clé adressée contenu
# ---------------------------------------------------------------------------

# Retire TOUS les commentaires HTML : <!-- Page N -->, <!-- Part N/M -->,
# <!-- OCR ÉCHOUÉ -->. Indispensable pour que l'OCR d'un livre entier et celui du
# même livre découpé (renumérotation globale des marqueurs Page) hashent identique.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_MIN_CHARS = 64


def normalize_text_for_hash(text: Any) -> str:
    """Normalisation **unique** partagée write-path/read-path (byte-pinned).

    ``NFC`` → retrait des commentaires HTML → collapse des blancs en un seul
    U+0020 → ``strip()``. **PAS** d'accent-strip, **PAS** de ``NFKC``, **PAS** de
    case-fold : collapser ces dimensions écraserait du texte français distinct
    (``côté`` vs ``cote``, ligatures, etc.) et provoquerait des faux positifs.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _HTML_COMMENT_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def content_hash(text: Any) -> str:
    """SHA-256 hex du texte normalisé. Déterministe, métrique-indépendant."""
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def content_id(chash: str, chunk_index: Any) -> str:
    """Identifiant **adressé par contenu** : ``"{hash[:16]}_{chunk_index}"``.

    Rend l'``upsert`` idempotent (Pinecone) et le ``uuid5`` stable (Weaviate/Qdrant)
    à travers les runs malgré le ``doc_id`` aléatoire.
    """
    return f"{chash[:16]}_{chunk_index}"


def is_dedup_eligible(normalized: str, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """Un chunk dont le texte normalisé est trop court (vide, « Références »,
    bruit OCR, placeholder de salvage) est **inéligible** : il ne participe jamais
    au matching (évite les faux positifs inter-documents que le hash seul ne sait
    pas écarter) et reste toujours conservé."""
    return len(normalized) >= max(0, min_chars)


def compute_dedup_fields(
    raw_text: Any, chunk_index: Any, min_chars: int = DEFAULT_MIN_CHARS
) -> Tuple[str, bool, str]:
    """Helper write-path : renvoie ``(content_hash, dedup_eligible, content_id)``
    en ne normalisant **qu'une fois** le texte brut pré-recodage.
    """
    norm = normalize_text_for_hash(raw_text)
    chash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    eligible = is_dedup_eligible(norm, min_chars)
    return chash, eligible, content_id(chash, chunk_index)


def backfill_chunk_dedup_fields(
    chunks: List[Dict[str, Any]], min_chars: int = DEFAULT_MIN_CHARS, text_key: str = "text"
) -> int:
    """Lot 5 (back-fill) — dote des chunks EXISTANTS (sans ``content_hash``) des
    champs ``content_hash`` / ``dedup_eligible`` et d'un ``id`` adressé par contenu,
    en hachant le texte **stocké** (``chunk[text_key]``). Mute en place, renvoie le
    nombre de chunks modifiés.

    CAVEAT décisif : le texte stocké est le texte **recodé** (post-GPT), pas le brut
    pré-recodage. Le hash back-fillé est donc *self-consistent* pour des **ré-uploads
    du même JSON** (idempotence + dédup cross-session lors d'un rebuild), mais ne
    matchera **pas** un ré-ingest frais (qui, lui, hache le brut). Le back-fill protège
    l'existant ; la protection pleine vient d'un ré-run du pipeline avec
    ``DEDUP_ENABLED=1``. Les chunks portant déjà un ``content_hash`` sont laissés tels
    quels (write-path raw préservé)."""
    modified = 0
    for c in chunks:
        if c.get("content_hash"):
            continue  # déjà hashé sur le brut (write-path) → ne pas écraser
        norm = normalize_text_for_hash(c.get(text_key, ""))
        chash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        eligible = is_dedup_eligible(norm, min_chars)
        c["content_hash"] = chash
        c["dedup_eligible"] = eligible
        if eligible:
            c["id"] = content_id(chash, c.get("chunk_index", 0))
        modified += 1
    return modified


# ---------------------------------------------------------------------------
# Configuration (variables DEDUP_*)
# ---------------------------------------------------------------------------


def _env_bool(value: Any, default: bool) -> bool:
    """Interprète une valeur d'environnement comme booléen (``1/true/yes/on``,
    insensible à la casse) ; renvoie ``default`` si elle vaut ``None``."""
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(value: Any, default: int) -> int:
    """Convertit une valeur d'environnement en ``int`` ; renvoie ``default`` si
    elle est absente ou invalide."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _env_float(value: Any, default: float) -> float:
    """Convertit une valeur d'environnement en ``float`` ; renvoie ``default`` si
    elle est absente ou invalide."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass
class DedupConfig:
    """Toute la configuration ``DEDUP_*``. Défauts = feature OFF, no-op."""

    enabled: bool = False
    semantic: bool = False
    sim_threshold: float = 0.97
    meta_fields: Tuple[str, ...] = ("title",)
    min_chars: int = DEFAULT_MIN_CHARS
    text_jaccard: float = 0.92
    query_batch: int = 100
    match_max: int = 3
    semantic_max_survivors: int = 500
    journal_dir: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "DedupConfig":
        """Construit la configuration depuis les variables ``DEDUP_*`` de ``env``
        (``os.environ`` par défaut). ``DEDUP_META_FIELDS`` est une liste séparée par
        des virgules (``title`` si vide) ; toute valeur absente ou invalide retombe sur
        le défaut du champ."""
        env = os.environ if env is None else env
        raw_meta = env.get("DEDUP_META_FIELDS", "title")
        meta_fields = tuple(s.strip() for s in raw_meta.split(",") if s.strip()) or ("title",)
        journal_dir = env.get("DEDUP_JOURNAL_DIR") or None
        return cls(
            enabled=_env_bool(env.get("DEDUP_ENABLED"), False),
            semantic=_env_bool(env.get("DEDUP_SEMANTIC"), False),
            sim_threshold=_env_float(env.get("DEDUP_SIM_THRESHOLD"), 0.97),
            meta_fields=meta_fields,
            min_chars=_env_int(env.get("DEDUP_MIN_CHARS"), DEFAULT_MIN_CHARS),
            text_jaccard=_env_float(env.get("DEDUP_TEXT_JACCARD"), 0.92),
            query_batch=_env_int(env.get("DEDUP_QUERY_BATCH"), 100),
            match_max=_env_int(env.get("DEDUP_MATCH_MAX"), 3),
            semantic_max_survivors=_env_int(env.get("DEDUP_SEMANTIC_MAX_SURVIVORS"), 500),
            journal_dir=journal_dir,
        )


# ---------------------------------------------------------------------------
# Adapter (Protocol) — implémenté concrètement dans rad_vectordb.py par base
# ---------------------------------------------------------------------------


@runtime_checkable
class DedupAdapter(Protocol):
    """Contrat d'accès base, dépendant de la base mais pas de ce module.

    Attributs :
        db_name : "pinecone" | "weaviate" | "qdrant" | "albert"
        metric  : "cosine" | "dotproduct" | "distance" | None (sens du seuil Tier 3 ;
                  None pour Albert, qui n'a pas de Tier 3)

    Méthodes :
        existing(chunks_batch) -> dict[content_hash, list[meta_dict]]
            Existence serveur-side batchée. Clé de retour = ``content_hash`` du
            chunk. ``meta_dict`` porte au minimum les ``DEDUP_META_FIELDS`` + ``id``
            + ``content_hash`` + ``chunk_index``. Pinecone résout par
            ``index.fetch(content_id)`` ; Weaviate/Qdrant par UUID adressé par
            contenu (``fetch_objects`` / ``retrieve``) ; Albert répond en mémoire,
            sans appel HTTP, depuis l'index construit par ``preload()``, et
            réhydrate les ``DEDUP_META_FIELDS`` assainis à l'envoi (titre tronqué
            à 255 caractères…) pour que la corroboration reste possible. La
            corroboration lisant ``chunk.get(champ)``, le connecteur Albert
            recopie d'abord sur ses copies des chunks la valeur des champs
            dérivés de sa liste blanche (``item_key``, ``year``, ``doi``,
            ``filename``).
            ``dedup_filter`` avale les exceptions de ``existing`` (lot conservé).
        nearest(embedding, meta_filter) -> list[dict] (OPTIONNEL, Tier 3)
            Chaque dict : ``{"score": float, "meta": dict, "text": str}``. Absente
            chez Albert : le Tier 3 est alors sauté.
        preload(...) (OPTIONNEL, hors de ce module)
            Albert seulement : liste les documents et chunks de la collection.
            Le connecteur l'appelle **avant** ``dedup_filter`` (qui ne l'appelle
            jamais) ; un échec fait échouer l'envoi bruyamment au lieu d'être
            avalé comme une erreur de ``existing``.
    """

    db_name: str
    metric: Optional[str]

    def existing(self, chunks_batch: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Existence serveur-side d'un lot : renvoie ``{content_hash: [meta_dict, ...]}``
        pour les chunks du lot déjà présents dans la base."""
        ...


# ---------------------------------------------------------------------------
# Helpers de décision
# ---------------------------------------------------------------------------


def _norm_meta(value: Any) -> str:
    """Normalise une valeur de métadonnée pour comparaison (``str`` + ``strip`` +
    minuscules) ; ``None`` ou ``""`` donnent ``""``."""
    return str(value).strip().lower() if value not in (None, "") else ""


def _meta_matches(chunk: Dict[str, Any], existing_meta: Dict[str, Any], meta_fields) -> bool:
    """Concordance métadonnée **obligatoire** avant tout refus. Si un champ
    requis est absent d'un côté, on **ne peut pas** corroborer → pas de refus."""
    for f in meta_fields:
        cv = _norm_meta(chunk.get(f))
        ev = _norm_meta(existing_meta.get(f))
        if not cv or not ev:
            return False
        if cv != ev:
            return False
    return True


def _matched_fields(chunk: Dict[str, Any], existing_meta: Dict[str, Any], meta_fields) -> List[str]:
    """Liste des champs de ``meta_fields`` non vides côté chunk et égaux (après
    ``_norm_meta``) entre le chunk et la métadonnée existante (journal)."""
    out = []
    for f in meta_fields:
        cv = _norm_meta(chunk.get(f))
        if cv and cv == _norm_meta(existing_meta.get(f)):
            out.append(f)
    return out


def _chunk_index_match(chunk: Dict[str, Any], existing_meta: Dict[str, Any]) -> Optional[bool]:
    """Compare les ``chunk_index`` (en ``int``) du chunk et de l'existant ; renvoie
    ``None`` si l'un des deux est absent ou non convertible."""
    try:
        return int(chunk.get("chunk_index")) == int(existing_meta.get("chunk_index"))
    except (TypeError, ValueError):
        return None


def _word_ngrams(text: str, n: int = 3):
    """Ensemble des n-grammes de mots du texte normalisé (``normalize_text_for_hash``
    + minuscules). Texte plus court que ``n`` mots → un seul « n-gramme » avec tous
    les mots ; texte vide → ensemble vide."""
    words = normalize_text_for_hash(text).lower().split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_word_ngram(a: str, b: str, n: int = 3) -> float:
    """Jaccard sur les n-grammes de mots — corroboration texte du Tier 3."""
    sa, sb = _word_ngrams(a, n), _word_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _length_ratio_ok(a: str, b: str, tol: float = 0.10) -> bool:
    """Vrai si les longueurs de ``a`` et ``b`` diffèrent d'au plus ``tol`` (ratio
    min/max >= ``1 - tol``) ; deux textes vides sont acceptés, un seul vide non."""
    la, lb = len(a or ""), len(b or "")
    if la == 0 and lb == 0:
        return True
    if la == 0 or lb == 0:
        return False
    return (min(la, lb) / max(la, lb)) >= (1.0 - tol)


def _metric_to_similarity(score: Any, metric: Optional[str]) -> Optional[float]:
    """Convertit un score de la base en similarité comparable au seuil.

    Weaviate ``near_vector`` renvoie une **distance** → similarité = 1 - distance.
    Cosinus/dotproduct (normalisé) : le score **est** déjà une similarité.
    """
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if metric == "distance":
        return 1.0 - s
    return s


def _utc_now_iso() -> str:
    """Horodatage courant UTC au format ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Journal (JSONL append, crash-safe) + summary
# ---------------------------------------------------------------------------

JOURNAL_FILENAME = "dedup_journal.jsonl"
SUMMARY_FILENAME = "dedup_summary.json"


def journal_path_for(journal_dir: str) -> str:
    """Chemin du journal JSONL (``dedup_journal.jsonl``) dans ``journal_dir``."""
    return os.path.join(journal_dir, JOURNAL_FILENAME)


def _build_rejection(
    chunk: Dict[str, Any],
    reason: str,
    matched: List[Dict[str, Any]],
    adapter: Any,
    config: DedupConfig,
    target_desc: str,
    score: Optional[float] = None,
    threshold: Optional[float] = None,
    jaccard: Optional[float] = None,
) -> Dict[str, Any]:
    """Construit l'entrée de journal d'un chunk refusé : cible, raison, chunk
    refusé (``doc_id`` pour traçabilité seulement) et au plus ``config.match_max``
    existants correspondants (le reste est compté dans
    ``matched_existing_truncated_count``). ``score`` / ``metric`` / ``text_jaccard``
    ne sont renseignés que pour une raison ``semantic_match``."""
    metric = getattr(adapter, "metric", None)
    is_semantic = reason == "semantic_match"
    matched_list = []
    for m in matched[: config.match_max]:
        matched_list.append(
            {
                "existing_id": m.get("id") or m.get("existing_id"),
                "content_hash": m.get("content_hash"),
                "score": score if is_semantic else None,
                "metric": metric if is_semantic else None,
                "title": m.get("title"),
                "authors": m.get("authors"),
                "chunk_index": m.get("chunk_index"),
                "metadata_match": _matched_fields(chunk, m, config.meta_fields),
                "chunk_index_match": _chunk_index_match(chunk, m),
                "text_jaccard": jaccard if is_semantic else None,
            }
        )
    truncated = max(0, len(matched) - config.match_max)
    return {
        "timestamp": _utc_now_iso(),
        "target_db": getattr(adapter, "db_name", None),
        "target": target_desc,
        "reason": reason,
        "refused": {
            "id": chunk.get("id"),
            "content_hash": chunk.get("content_hash"),
            # doc_id : traçabilité SEULEMENT (jamais une clé de match).
            "doc_id": chunk.get("doc_id"),
            "chunk_index": chunk.get("chunk_index"),
            "title": chunk.get("title"),
            "authors": chunk.get("authors"),
            "text_preview": (chunk.get("text") or "")[:200],
        },
        "matched_existing": matched_list,
        "matched_existing_truncated_count": truncated,
        "threshold_used": threshold,
    }


def _write_journal(
    rejections: List[Dict[str, Any]], journal_dir: str, kept_count: int
) -> Optional[str]:
    """Append JSONL (crash-safe, greppable) + summary overwrite. Best-effort :
    une erreur d'écriture ne fait jamais échouer l'insertion."""
    try:
        os.makedirs(journal_dir, exist_ok=True)
        path = journal_path_for(journal_dir)
        with open(path, "a", encoding="utf-8") as f:
            for r in rejections:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_reason: Dict[str, int] = {}
        for r in rejections:
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        summary = {
            "skipped_total": len(rejections),
            "by_reason": by_reason,
            "kept": kept_count,
            "journal": path,
        }
        with open(os.path.join(journal_dir, SUMMARY_FILENAME), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:  # pragma: no cover - advisory
        logger.warning("dedup journal write failed in %s: %s", journal_dir, exc)
        return None


# ---------------------------------------------------------------------------
# Tier 3 — confirmation sémantique (optionnelle)
# ---------------------------------------------------------------------------


def _tier3_match(
    chunk: Dict[str, Any], adapter: Any, config: DedupConfig
) -> Optional[Dict[str, Any]]:
    """Renvoie ``{"score","meta","jaccard"}`` si un voisin confirme le doublon,
    sinon ``None``. Ne refuse **jamais** sur le score seul : exige score≥seuil
    **ET** métadonnées **ET** ``chunk_index`` (garde dure) **ET** ratio de longueur
    **ET** Jaccard texte."""
    embedding = chunk.get("embedding")
    if not embedding:
        return None
    meta_filter = {"title": chunk.get("title"), "chunk_index": chunk.get("chunk_index")}
    try:
        candidates = adapter.nearest(embedding, meta_filter) or []
    except Exception as exc:  # pragma: no cover - advisory
        logger.warning("dedup Tier3 nearest() failed: %s", exc)
        return None
    cand_text = chunk.get("text") or ""
    metric = getattr(adapter, "metric", None)
    for cand in candidates:
        sim = _metric_to_similarity(cand.get("score"), metric)
        if sim is None or sim < config.sim_threshold:
            continue
        meta = cand.get("meta") or {}
        if not _meta_matches(chunk, meta, config.meta_fields):
            continue
        if _chunk_index_match(chunk, meta) is False:  # garde DURE en Tier 3
            continue
        existing_text = cand.get("text") or ""
        if not _length_ratio_ok(cand_text, existing_text, 0.10):
            continue
        jac = jaccard_word_ngram(cand_text, existing_text, n=3)
        if jac < config.text_jaccard:
            continue
        return {"score": sim, "meta": meta, "jaccard": jac}
    return None


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------


def dedup_filter(
    all_chunks: List[Dict[str, Any]],
    adapter: Any,
    config: DedupConfig,
    journal_dir: Optional[str] = None,
    target_desc: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Orchestre Tiers 1→3, écrit le journal, renvoie ``(kept, rejections)``.

    **Si ``config.enabled`` est faux → no-op vrai** : renvoie ``(all_chunks, [])``
    immédiatement, l'adapter n'est jamais touché, aucun journal écrit.

    Un chunk **participe** à la dédup ssi il porte un ``content_hash`` **et**
    ``dedup_eligible`` (les chunks legacy/inéligibles sont toujours conservés et
    jamais journalisés). L'ordre d'entrée est préservé dans ``kept``.
    """
    if not config.enabled:
        return list(all_chunks), []

    journal_dir = journal_dir or config.journal_dir
    rejections: List[Dict[str, Any]] = []
    rejected_ids = set()  # id() des objets chunk refusés (préserve l'ordre)

    # --- Tier 1 : seen-set mémoire (in-batch) ---------------------------------
    seen: Dict[str, Dict[str, Any]] = {}
    tier1_survivors: List[Dict[str, Any]] = []
    for chunk in all_chunks:
        chash = chunk.get("content_hash")
        eligible = chunk.get("dedup_eligible")
        if eligible is None:
            eligible = bool(chash)  # éligibilité inconnue → dérivée de la présence du hash
        if not chash or not eligible:
            continue  # toujours conservé (filtre final préservant l'ordre)
        if chash in seen:
            rejections.append(
                _build_rejection(chunk, "in_batch_duplicate", [seen[chash]], adapter, config, target_desc)
            )
            rejected_ids.add(id(chunk))
            continue
        seen[chash] = chunk
        tier1_survivors.append(chunk)

    # --- Tier 2 : existence serveur-side batchée ------------------------------
    qb = config.query_batch if config.query_batch > 0 else 100
    tier2_survivors: List[Dict[str, Any]] = []
    for i in range(0, len(tier1_survivors), qb):
        batch = tier1_survivors[i : i + qb]
        try:
            existing_map = adapter.existing(batch) or {}
        except Exception as exc:  # advisory : ne JAMAIS perdre un chunk unique
            logger.warning("dedup Tier2 existence check failed (%s); batch kept", exc)
            existing_map = {}
        for chunk in batch:
            matches = existing_map.get(chunk["content_hash"]) or []
            corroborated = [m for m in matches if _meta_matches(chunk, m, config.meta_fields)]
            if corroborated:
                rejections.append(
                    _build_rejection(chunk, "existing_hash_match", corroborated, adapter, config, target_desc)
                )
                rejected_ids.add(id(chunk))
            else:
                tier2_survivors.append(chunk)

    # --- Tier 3 : confirmation sémantique (optionnelle, capée) ----------------
    do_semantic = config.semantic and callable(getattr(adapter, "nearest", None))
    if do_semantic and len(tier2_survivors) > config.semantic_max_survivors:
        logger.warning(
            "dedup Tier3 sauté : %d survivants > cap DEDUP_SEMANTIC_MAX_SURVIVORS=%d",
            len(tier2_survivors),
            config.semantic_max_survivors,
        )
        do_semantic = False
    if do_semantic:
        for chunk in tier2_survivors:
            verdict = _tier3_match(chunk, adapter, config)
            if verdict is not None:
                rejections.append(
                    _build_rejection(
                        chunk,
                        "semantic_match",
                        [verdict["meta"]],
                        adapter,
                        config,
                        target_desc,
                        score=verdict["score"],
                        threshold=config.sim_threshold,
                        jaccard=verdict["jaccard"],
                    )
                )
                rejected_ids.add(id(chunk))

    kept = [c for c in all_chunks if id(c) not in rejected_ids]

    if rejections and journal_dir:
        _write_journal(rejections, journal_dir, len(kept))

    return kept, rejections
