"""Configuration de l'intégration Albert (DINUM) — stdlib seulement.

Tout est **désactivé par défaut** (``ALBERT_ENABLED=0``) : sans activation,
aucun appel, aucune route, aucune interface ni clé JSON Albert, et le
comportement du pipeline reste identique à l'octet.

Contenu :

- ``AlbertConfig`` : dataclass figée de tous les drapeaux du sprint, lue par
  ``AlbertConfig.from_env()`` au moment de l'appel (bibliothèques, routes) ou
  à l'import (scripts CLI, en constantes patchables). ``from_env`` ne lit
  **jamais** ``ALBERT_API_KEY`` : la clé est un identifiant utilisateur,
  transmis explicitement au client.
- ``normalise_base_url`` / ``root_url`` : URL de base serveur uniquement,
  https imposé, ``/v1`` garanti, hôte en liste blanche
  (``albert.api.etalab.gouv.fr``), réécriture de ``api.albert.etalab.gouv.fr``
  avec un WARNING, identifiants dans l'URL refusés.
- ``ENV_REGISTRY`` : (nom, défaut, sens) de chaque variable Albert lue par
  le code, source de la documentation et de ``.env.example``. ``ALBERT_LIVE``
  n'y figure pas : c'est un interrupteur de shell réservé aux tests live, qui
  ne doit jamais apparaître dans un fichier d'environnement.
- ``validate_metadata_fields`` : règles de la liste blanche de métadonnées
  des collections Albert.

Parseurs repris de ``rad_dedup`` (``_env_bool`` / ``_env_int`` /
``_env_float``), avec une différence : la chaîne vide vaut « non défini »
(le formulaire d'administration écrit ``NOM=`` pour un champ vide).
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, fields as dataclass_fields
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
"""URL de base officielle (``/v1`` compris)."""

ALLOWED_HOSTS = ("albert.api.etalab.gouv.fr",)
"""Hôtes acceptés sans ``ALBERT_ALLOW_CUSTOM_HOST``."""

REWRITTEN_HOSTS = MappingProxyType({"api.albert.etalab.gouv.fr": "albert.api.etalab.gouv.fr"})
"""Hôtes équivalents réécrits (avec WARNING) vers l'hôte officiel."""

OCR_MODES = ("auto", "chat", "ocr")
REASONING_EFFORTS = ("low", "medium", "high")
LIMITER_BACKENDS = ("local", "redis")
TOKEN_ESTIMATORS = ("chars", "tiktoken")

EMBED_BATCH_MAX = 64
"""Plafond serveur d'un lot d'embeddings (D12 : 65 donne un 413)."""

METADATA_MAX_FIELDS = 10
"""Nombre maximal de métadonnées scalaires par chunk (D16)."""

REQUIRED_METADATA_FIELDS = ("content_id", "content_hash", "chunk_index")
"""Champs imposés dans ``ALBERT_METADATA_FIELDS`` (idempotence, dédup)."""

FORBIDDEN_METADATA_FIELDS = ("path",)
"""Champs jamais envoyés à une collection Albert."""

DEFAULT_METADATA_FIELDS: Tuple[str, ...] = (
    "content_id",
    "content_hash",
    "chunk_index",
    "total_chunks",
    "title",
    "authors",
    "year",
    "doi",
    "item_key",
    "filename",
)
"""Liste blanche par défaut des métadonnées envoyées aux collections."""


# ---------------------------------------------------------------------------
# Parseurs (motif rad_dedup ; '' = non défini)
# ---------------------------------------------------------------------------


def _is_unset(value: Any) -> bool:
    """Vrai si la valeur est absente (``None``) ou vide après ``strip``."""
    return value is None or str(value).strip() == ""


def _env_bool(value: Any, default: bool) -> bool:
    """Interprète une valeur d'environnement comme booléen (``1/true/yes/on``,
    insensible à la casse) ; renvoie ``default`` si elle est absente ou vide."""
    if _is_unset(value):
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(value: Any, default: int) -> int:
    """Convertit une valeur d'environnement en ``int`` ; renvoie ``default`` si
    elle est absente, vide ou invalide."""
    if _is_unset(value):
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _env_float(value: Any, default: float) -> float:
    """Convertit une valeur d'environnement en ``float`` fini ; renvoie
    ``default`` si elle est absente, vide, invalide, infinie ou NaN."""
    if _is_unset(value):
        return default
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _env_str(value: Any, default: Optional[str]) -> Optional[str]:
    """Renvoie la valeur sans blancs périphériques, ou ``default`` si elle est
    absente ou vide."""
    if _is_unset(value):
        return default
    return str(value).strip()


def _env_choice(value: Any, default: str, choices: Iterable[str]) -> str:
    """Renvoie la valeur en minuscules si elle figure dans ``choices``, sinon
    ``default``."""
    text = _env_str(value, None)
    if text is None:
        return default
    text = text.lower()
    return text if text in tuple(choices) else default


def _at_least(value: Any, minimum: float, default: Any) -> Any:
    """Renvoie ``value`` si elle vaut au moins ``minimum``, sinon ``default``."""
    return value if value >= minimum else default


def _within(value: float, low: float, high: float, default: float) -> float:
    """Renvoie ``value`` si ``low <= value <= high``, sinon ``default``."""
    return value if low <= value <= high else default


def _split_names(value: Any) -> Tuple[str, ...]:
    """Découpe une liste de noms (chaîne séparée par des virgules ou itérable)
    en tuple sans vide ni doublon, dans l'ordre d'apparition."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else list(value)
    out = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


# ---------------------------------------------------------------------------
# URL de base
# ---------------------------------------------------------------------------


def _normalise_base_url(value: Optional[str], *, allow_custom_host: bool, log: bool) -> str:
    """Implémentation de ``normalise_base_url`` ; ``log`` active le WARNING de
    réécriture (désactivé par ``from_env`` quand Albert est OFF)."""
    if _is_unset(value):
        return DEFAULT_BASE_URL
    text = str(value).strip()
    if text.strip("/") == "":
        return DEFAULT_BASE_URL
    if any(ch.isspace() or ord(ch) < 32 or ch == "\\" for ch in text):
        raise ValueError("ALBERT_BASE_URL invalide : caractères interdits (blanc, contrôle ou antislash).")
    if "://" not in text:
        text = "https://" + text
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    if scheme not in ("https", "http"):
        raise ValueError("ALBERT_BASE_URL invalide : seul le schéma https est accepté.")
    if "@" in parts.netloc:
        raise ValueError(
            "ALBERT_BASE_URL invalide : identifiants dans l'URL refusés "
            "(la clé se configure dans Paramètres > Mes Identifiants)."
        )
    if parts.query or parts.fragment:
        raise ValueError("ALBERT_BASE_URL invalide : paramètres de requête et fragment interdits.")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("ALBERT_BASE_URL invalide : hôte absent.")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("ALBERT_BASE_URL invalide : port incorrect.") from None

    changes = []
    if scheme == "http":
        changes.append("https imposé")
    if host in REWRITTEN_HOSTS:
        host = REWRITTEN_HOSTS[host]
        changes.append("hôte api.albert.etalab.gouv.fr réécrit en albert.api.etalab.gouv.fr")
    official = host in ALLOWED_HOSTS
    if not official and not allow_custom_host:
        raise ValueError(
            f"ALBERT_BASE_URL refusée : l'hôte {host!r} n'est pas dans la liste blanche "
            f"({', '.join(ALLOWED_HOSTS)}) ; ALBERT_ALLOW_CUSTOM_HOST=1 autorise un "
            "OpenGateLLM auto-hébergé."
        )
    if official:
        if port not in (None, 443):
            raise ValueError("ALBERT_BASE_URL refusée : port non standard sur l'hôte officiel.")
        port = None

    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    if path == "":
        path = "/v1"
    elif not path.endswith("/v1"):
        cut = path.find("/v1/")
        if cut >= 0:
            path = path[: cut + 3]
            changes.append("suffixe de chemin après /v1 retiré")
        elif official:
            raise ValueError("ALBERT_BASE_URL refusée : le chemin doit être /v1 sur l'hôte officiel.")
        else:
            path = path + "/v1"
    if official and path != "/v1":
        raise ValueError("ALBERT_BASE_URL refusée : le chemin doit être /v1 sur l'hôte officiel.")

    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    normalised = urlunsplit(("https", netloc, path, "", ""))
    if changes and log:
        logger.warning(
            "URL de base Albert normalisée : %s -> %s (%s).",
            urlunsplit((scheme, parts.netloc, parts.path, "", "")),
            normalised,
            " ; ".join(changes),
        )
    return normalised


def normalise_base_url(value: Optional[str], *, allow_custom_host: bool = False) -> str:
    """Valide et normalise l'URL de base Albert (configuration serveur).

    Règles : vide ou ``None`` → ``DEFAULT_BASE_URL`` ; blancs et ``/`` final
    retirés ; schéma absent ou ``http`` → ``https`` ; ``/v1`` conservé ou
    ajouté (un suffixe d'endpoint après ``/v1`` est retiré) ; hôte
    ``api.albert.etalab.gouv.fr`` réécrit avec un WARNING ; identifiants
    (``user:pass@``), requête, fragment et schémas autres que http(s) refusés ;
    hôte hors liste blanche refusé sauf ``allow_custom_host``.

    Args:
        value: Valeur brute (``ALBERT_BASE_URL``).
        allow_custom_host: Autorise un hôte hors liste blanche (OpenGateLLM
            auto-hébergé) ; https reste imposé.

    Returns:
        L'URL normalisée, par exemple ``https://albert.api.etalab.gouv.fr/v1``.

    Raises:
        ValueError: URL refusée (le message ne recopie jamais d'identifiants).
    """
    return _normalise_base_url(value, allow_custom_host=allow_custom_host, log=True)


def root_url(base_url: str) -> str:
    """Renvoie la racine du service (sans ``/v1``), pour ``/health`` et ``/metrics``.

    Args:
        base_url: URL de base normalisée (vide → défaut).

    Returns:
        Par exemple ``https://albert.api.etalab.gouv.fr``.
    """
    text = (str(base_url).strip() if base_url else "") or DEFAULT_BASE_URL
    text = text.rstrip("/")
    if text.endswith("/v1"):
        text = text[: -len("/v1")]
    return text.rstrip("/")


# ---------------------------------------------------------------------------
# Liste blanche des métadonnées
# ---------------------------------------------------------------------------


def validate_metadata_fields(fields: Any, dedup_meta_fields: Any) -> Tuple[str, ...]:
    """Valide la liste blanche des métadonnées envoyées aux collections Albert.

    Refuse une liste contenant ``path``, de plus de 10 champs, ou à laquelle
    il manque ``content_id``, ``content_hash``, ``chunk_index`` ou l'un des
    ``DEDUP_META_FIELDS``. Les doublons et les noms vides sont ignorés.

    Args:
        fields: Noms (chaîne séparée par des virgules ou itérable).
        dedup_meta_fields: ``DEDUP_META_FIELDS`` effectifs (même format ;
            ``None`` = aucun).

    Returns:
        Le tuple des noms retenus, dans l'ordre donné.

    Raises:
        ValueError: Liste refusée (message français listant les causes).
    """
    names = _split_names(fields)
    dedup = _split_names(dedup_meta_fields)
    forbidden = [name for name in names if name.lower() in FORBIDDEN_METADATA_FIELDS]
    if forbidden:
        raise ValueError(
            "ALBERT_METADATA_FIELDS refusée : champ interdit "
            f"({', '.join(forbidden)}) ; le chemin local n'est jamais envoyé."
        )
    if len(names) > METADATA_MAX_FIELDS:
        raise ValueError(
            f"ALBERT_METADATA_FIELDS refusée : {len(names)} champs, "
            f"{METADATA_MAX_FIELDS} au plus (limite de l'API Albert)."
        )
    missing = [name for name in REQUIRED_METADATA_FIELDS + dedup if name not in names]
    if missing:
        raise ValueError(
            "ALBERT_METADATA_FIELDS refusée : champs obligatoires manquants "
            f"({', '.join(dict.fromkeys(missing))}) ; content_id, content_hash, "
            "chunk_index et DEDUP_META_FIELDS sont imposés."
        )
    return names


# ---------------------------------------------------------------------------
# Configuration figée
# ---------------------------------------------------------------------------

_REWRITE_WARNED: set = set()


@dataclass(frozen=True)
class AlbertConfig:
    """Tous les drapeaux ``ALBERT_*`` du sprint, figés. Défauts = Albert OFF.

    Les défauts de débit et de concurrence viennent des mesures D1, D5, D12,
    D19 et D22 (``tests/fixtures/albert/decisions.json``). La clé API n'est
    **pas** un champ : elle n'est jamais lue ici.
    """

    enabled: bool = False
    base_url: str = DEFAULT_BASE_URL
    allow_custom_host: bool = False
    ocr_enabled: bool = False
    ocr_mode: str = "auto"
    ocr_chat_model: str = "lightonocr-2-1b"
    ocr_doc_model: str = "mistral-ocr-2512"
    ocr_dpi: int = 200
    ocr_max_side: int = 1540
    ocr_max_tokens: int = 4096
    ocr_temperature: float = 0.2
    ocr_top_p: float = 0.9
    ocr_max_pages: int = 0
    ocr_max_failed_ratio: float = 0.5
    ocr_part_mb: float = 15.0
    ocr_part_pages: int = 100
    ocr_concurrency: int = 1
    ocr_skip_recode: bool = False
    embed_model: str = "bge-m3"
    embed_batch: int = 64
    embed_max_missing_ratio: float = 0.0
    embed_l2_normalize: bool = True
    reasoning_effort: str = "medium"
    reasoning_headroom: int = 2048
    model_fallback: bool = True
    busy_retries: int = 2
    recode_rpm: int = 45
    notes_rpm: int = 9
    ocr_rpm: int = 45
    embed_rpm: int = 450
    chat_tpm: int = 115000
    process_share: float = 1.0
    limiter_backend: str = "local"
    limiter_redis_url: Optional[str] = None
    recode_concurrency: int = 2
    notes_concurrency: int = 2
    embed_concurrency: int = 4
    push_concurrency: int = 1
    max_retries: int = 4
    retry_backoff: float = 2.0
    retry_max_backoff: float = 60.0
    retry_after_max: float = 120.0
    timeout_chat: float = 120.0
    timeout_notes: float = 300.0
    timeout_ocr_page: float = 120.0
    timeout_ocr_doc: float = 300.0
    timeout_embed: float = 60.0
    timeout_collections: float = 120.0
    subprocess_timeout: int = 21600
    metadata_fields: Tuple[str, ...] = DEFAULT_METADATA_FIELDS
    usage_log: bool = True
    preflight: bool = True
    token_estimator: str = "chars"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AlbertConfig":
        """Construit la configuration depuis les variables d'environnement.

        Lit uniquement les noms de ``FIELD_ENV_NAMES`` (jamais
        ``ALBERT_API_KEY``). Une valeur absente, vide, invalide ou hors bornes
        retombe sur le défaut du champ ; ``ALBERT_EMBED_BATCH`` est plafonné à
        64. Albert OFF, une ``ALBERT_BASE_URL`` refusée est remplacée par le
        défaut sans journal ; Albert ON, elle lève ``ValueError`` (refus au
        démarrage) et la réécriture d'hôte est journalisée une fois.

        Args:
            env: Variables à lire (``os.environ`` par défaut).

        Returns:
            L'instance figée.

        Raises:
            ValueError: Albert ON avec une ``ALBERT_BASE_URL`` refusée.
        """
        env = os.environ if env is None else env
        d = cls()

        def get(field_name: str) -> Any:
            """Lit la variable associée au champ ``field_name``."""
            return env.get(FIELD_ENV_NAMES[field_name])

        enabled = _env_bool(get("enabled"), d.enabled)
        allow_custom_host = _env_bool(get("allow_custom_host"), d.allow_custom_host)
        raw_base = get("base_url")
        if enabled:
            key = (str(raw_base).strip() if raw_base is not None else "", allow_custom_host)
            first = key not in _REWRITE_WARNED
            base_url = _normalise_base_url(raw_base, allow_custom_host=allow_custom_host, log=first)
            _REWRITE_WARNED.add(key)
        else:
            try:
                base_url = _normalise_base_url(raw_base, allow_custom_host=allow_custom_host, log=False)
            except ValueError:
                base_url = DEFAULT_BASE_URL

        embed_batch = _at_least(_env_int(get("embed_batch"), d.embed_batch), 1, d.embed_batch)
        metadata_fields = _split_names(get("metadata_fields")) or d.metadata_fields

        return cls(
            enabled=enabled,
            base_url=base_url,
            allow_custom_host=allow_custom_host,
            ocr_enabled=_env_bool(get("ocr_enabled"), d.ocr_enabled),
            ocr_mode=_env_choice(get("ocr_mode"), d.ocr_mode, OCR_MODES),
            ocr_chat_model=_env_str(get("ocr_chat_model"), d.ocr_chat_model),
            ocr_doc_model=_env_str(get("ocr_doc_model"), d.ocr_doc_model),
            ocr_dpi=_at_least(_env_int(get("ocr_dpi"), d.ocr_dpi), 1, d.ocr_dpi),
            ocr_max_side=_at_least(_env_int(get("ocr_max_side"), d.ocr_max_side), 1, d.ocr_max_side),
            ocr_max_tokens=_at_least(_env_int(get("ocr_max_tokens"), d.ocr_max_tokens), 1, d.ocr_max_tokens),
            ocr_temperature=_within(
                _env_float(get("ocr_temperature"), d.ocr_temperature), 0.0, 2.0, d.ocr_temperature
            ),
            ocr_top_p=_within(_env_float(get("ocr_top_p"), d.ocr_top_p), 0.0, 1.0, d.ocr_top_p),
            ocr_max_pages=_at_least(_env_int(get("ocr_max_pages"), d.ocr_max_pages), 0, d.ocr_max_pages),
            ocr_max_failed_ratio=_within(
                _env_float(get("ocr_max_failed_ratio"), d.ocr_max_failed_ratio),
                0.0,
                1.0,
                d.ocr_max_failed_ratio,
            ),
            ocr_part_mb=_at_least(_env_float(get("ocr_part_mb"), d.ocr_part_mb), 0.001, d.ocr_part_mb),
            ocr_part_pages=_at_least(_env_int(get("ocr_part_pages"), d.ocr_part_pages), 1, d.ocr_part_pages),
            ocr_concurrency=_at_least(_env_int(get("ocr_concurrency"), d.ocr_concurrency), 1, d.ocr_concurrency),
            ocr_skip_recode=_env_bool(get("ocr_skip_recode"), d.ocr_skip_recode),
            embed_model=_env_str(get("embed_model"), d.embed_model),
            embed_batch=min(embed_batch, EMBED_BATCH_MAX),
            embed_max_missing_ratio=_within(
                _env_float(get("embed_max_missing_ratio"), d.embed_max_missing_ratio),
                0.0,
                1.0,
                d.embed_max_missing_ratio,
            ),
            embed_l2_normalize=_env_bool(get("embed_l2_normalize"), d.embed_l2_normalize),
            reasoning_effort=_env_choice(get("reasoning_effort"), d.reasoning_effort, REASONING_EFFORTS),
            reasoning_headroom=_at_least(
                _env_int(get("reasoning_headroom"), d.reasoning_headroom), 0, d.reasoning_headroom
            ),
            model_fallback=_env_bool(get("model_fallback"), d.model_fallback),
            busy_retries=_at_least(_env_int(get("busy_retries"), d.busy_retries), 0, d.busy_retries),
            recode_rpm=_at_least(_env_int(get("recode_rpm"), d.recode_rpm), 1, d.recode_rpm),
            notes_rpm=_at_least(_env_int(get("notes_rpm"), d.notes_rpm), 1, d.notes_rpm),
            ocr_rpm=_at_least(_env_int(get("ocr_rpm"), d.ocr_rpm), 1, d.ocr_rpm),
            embed_rpm=_at_least(_env_int(get("embed_rpm"), d.embed_rpm), 1, d.embed_rpm),
            chat_tpm=_at_least(_env_int(get("chat_tpm"), d.chat_tpm), 1, d.chat_tpm),
            process_share=_within(
                _env_float(get("process_share"), d.process_share), 1e-9, 1.0, d.process_share
            ),
            limiter_backend=_env_choice(get("limiter_backend"), d.limiter_backend, LIMITER_BACKENDS),
            limiter_redis_url=_env_str(get("limiter_redis_url"), d.limiter_redis_url),
            recode_concurrency=_at_least(
                _env_int(get("recode_concurrency"), d.recode_concurrency), 1, d.recode_concurrency
            ),
            notes_concurrency=_at_least(
                _env_int(get("notes_concurrency"), d.notes_concurrency), 1, d.notes_concurrency
            ),
            embed_concurrency=_at_least(
                _env_int(get("embed_concurrency"), d.embed_concurrency), 1, d.embed_concurrency
            ),
            push_concurrency=_at_least(
                _env_int(get("push_concurrency"), d.push_concurrency), 1, d.push_concurrency
            ),
            max_retries=_at_least(_env_int(get("max_retries"), d.max_retries), 0, d.max_retries),
            retry_backoff=_at_least(_env_float(get("retry_backoff"), d.retry_backoff), 0.0, d.retry_backoff),
            retry_max_backoff=_at_least(
                _env_float(get("retry_max_backoff"), d.retry_max_backoff), 0.0, d.retry_max_backoff
            ),
            retry_after_max=_at_least(
                _env_float(get("retry_after_max"), d.retry_after_max), 0.0, d.retry_after_max
            ),
            timeout_chat=_positive_timeout(get("timeout_chat"), d.timeout_chat),
            timeout_notes=_positive_timeout(get("timeout_notes"), d.timeout_notes),
            timeout_ocr_page=_positive_timeout(get("timeout_ocr_page"), d.timeout_ocr_page),
            timeout_ocr_doc=_positive_timeout(get("timeout_ocr_doc"), d.timeout_ocr_doc),
            timeout_embed=_positive_timeout(get("timeout_embed"), d.timeout_embed),
            timeout_collections=_positive_timeout(get("timeout_collections"), d.timeout_collections),
            subprocess_timeout=_at_least(
                _env_int(get("subprocess_timeout"), d.subprocess_timeout), 1, d.subprocess_timeout
            ),
            metadata_fields=metadata_fields,
            usage_log=_env_bool(get("usage_log"), d.usage_log),
            preflight=_env_bool(get("preflight"), d.preflight),
            token_estimator=_env_choice(get("token_estimator"), d.token_estimator, TOKEN_ESTIMATORS),
        )


def _positive_timeout(value: Any, default: float) -> float:
    """Délai HTTP strictement positif ; ``default`` sinon."""
    number = _env_float(value, default)
    return number if number > 0 else default


# ---------------------------------------------------------------------------
# Registre des variables
# ---------------------------------------------------------------------------

FIELD_ENV_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "enabled": "ALBERT_ENABLED",
        "base_url": "ALBERT_BASE_URL",
        "allow_custom_host": "ALBERT_ALLOW_CUSTOM_HOST",
        "ocr_enabled": "OCR_ENABLE_ALBERT",
        "ocr_mode": "ALBERT_OCR_MODE",
        "ocr_chat_model": "ALBERT_OCR_CHAT_MODEL",
        "ocr_doc_model": "ALBERT_OCR_DOC_MODEL",
        "ocr_dpi": "ALBERT_OCR_DPI",
        "ocr_max_side": "ALBERT_OCR_MAX_SIDE",
        "ocr_max_tokens": "ALBERT_OCR_MAX_TOKENS",
        "ocr_temperature": "ALBERT_OCR_TEMPERATURE",
        "ocr_top_p": "ALBERT_OCR_TOP_P",
        "ocr_max_pages": "ALBERT_OCR_MAX_PAGES",
        "ocr_max_failed_ratio": "ALBERT_OCR_MAX_FAILED_RATIO",
        "ocr_part_mb": "ALBERT_OCR_PART_MB",
        "ocr_part_pages": "ALBERT_OCR_PART_PAGES",
        "ocr_concurrency": "ALBERT_OCR_CONCURRENCY",
        "ocr_skip_recode": "ALBERT_OCR_SKIP_RECODE",
        "embed_model": "ALBERT_EMBED_MODEL",
        "embed_batch": "ALBERT_EMBED_BATCH",
        "embed_max_missing_ratio": "ALBERT_EMBED_MAX_MISSING_RATIO",
        "embed_l2_normalize": "ALBERT_EMBED_L2_NORMALIZE",
        "reasoning_effort": "ALBERT_REASONING_EFFORT",
        "reasoning_headroom": "ALBERT_REASONING_HEADROOM",
        "model_fallback": "ALBERT_MODEL_FALLBACK",
        "busy_retries": "ALBERT_BUSY_RETRIES",
        "recode_rpm": "ALBERT_RECODE_RPM",
        "notes_rpm": "ALBERT_NOTES_RPM",
        "ocr_rpm": "ALBERT_OCR_RPM",
        "embed_rpm": "ALBERT_EMBED_RPM",
        "chat_tpm": "ALBERT_CHAT_TPM",
        "process_share": "ALBERT_PROCESS_SHARE",
        "limiter_backend": "ALBERT_LIMITER_BACKEND",
        "limiter_redis_url": "ALBERT_LIMITER_REDIS_URL",
        "recode_concurrency": "ALBERT_RECODE_CONCURRENCY",
        "notes_concurrency": "ALBERT_NOTES_CONCURRENCY",
        "embed_concurrency": "ALBERT_EMBED_CONCURRENCY",
        "push_concurrency": "ALBERT_PUSH_CONCURRENCY",
        "max_retries": "ALBERT_MAX_RETRIES",
        "retry_backoff": "ALBERT_RETRY_BACKOFF",
        "retry_max_backoff": "ALBERT_RETRY_MAX_BACKOFF",
        "retry_after_max": "ALBERT_RETRY_AFTER_MAX",
        "timeout_chat": "ALBERT_TIMEOUT_CHAT",
        "timeout_notes": "ALBERT_TIMEOUT_NOTES",
        "timeout_ocr_page": "ALBERT_TIMEOUT_OCR_PAGE",
        "timeout_ocr_doc": "ALBERT_TIMEOUT_OCR_DOC",
        "timeout_embed": "ALBERT_TIMEOUT_EMBED",
        "timeout_collections": "ALBERT_TIMEOUT_COLLECTIONS",
        "subprocess_timeout": "ALBERT_SUBPROCESS_TIMEOUT",
        "metadata_fields": "ALBERT_METADATA_FIELDS",
        "usage_log": "ALBERT_USAGE_LOG",
        "preflight": "ALBERT_PREFLIGHT",
        "token_estimator": "ALBERT_TOKEN_ESTIMATOR",
    }
)
"""Champ d'``AlbertConfig`` → variable d'environnement lue par ``from_env``."""

ENV_REGISTRY: Tuple[Tuple[str, str, str], ...] = (
    ("ALBERT_ENABLED", "0",
     "Interrupteur maître. 0 = comportement strictement identique : ni appel, ni route, ni interface, ni clé JSON Albert."),
    ("ALBERT_API_KEY", "",
     "Clé Bearer Albert (identifiant albert_api_key ; repli .env réservé aux admins). Jamais lue par la bibliothèque, masquée dans /get_credentials."),
    ("ALBERT_BASE_URL", DEFAULT_BASE_URL,
     "URL de base de l'API, configuration serveur uniquement : https, /v1 garanti, hôte en liste blanche ; vide = défaut."),
    ("ALBERT_ALLOW_CUSTOM_HOST", "0",
     "1 = autorise un OpenGateLLM auto-hébergé hors liste blanche (https reste imposé)."),
    ("OCR_ENABLE_ALBERT", "0",
     "1 = maillon OCR Albert avant Mistral (exige ALBERT_ENABLED=1 et une clé)."),
    ("ALBERT_OCR_MODE", "auto",
     "auto = /v1/ocr si l'accès est établi, sinon LightOnOCR ; chat = LightOnOCR seul ; ocr = /v1/ocr seul."),
    ("ALBERT_OCR_CHAT_MODEL", "lightonocr-2-1b",
     "Modèle OCR par chat sur pages rastérisées (id épinglé)."),
    ("ALBERT_OCR_DOC_MODEL", "mistral-ocr-2512",
     "Modèle de /v1/ocr, accès restreint (id épinglé)."),
    ("ALBERT_OCR_DPI", "200",
     "Résolution de rastérisation des pages envoyées à LightOnOCR."),
    ("ALBERT_OCR_MAX_SIDE", "1540",
     "Plus grand côté d'une page rastérisée, en pixels."),
    ("ALBERT_OCR_MAX_TOKENS", "4096",
     "max_tokens d'une requête OCR par chat."),
    ("ALBERT_OCR_TEMPERATURE", "0.2",
     "Température d'une requête OCR par chat."),
    ("ALBERT_OCR_TOP_P", "0.9",
     "top_p d'une requête OCR par chat."),
    ("ALBERT_OCR_MAX_PAGES", "0",
     "Plafond de pages par document (0 = aucun ; au-delà, résultat partiel)."),
    ("ALBERT_OCR_MAX_FAILED_RATIO", "0.5",
     "Part de pages en échec au-delà de laquelle le maillon Albert est abandonné."),
    ("ALBERT_OCR_PART_MB", "15",
     "Taille maximale (Mo) d'une part envoyée en base64 à /v1/ocr."),
    ("ALBERT_OCR_PART_PAGES", "100",
     "Nombre maximal de pages d'une part envoyée à /v1/ocr."),
    ("ALBERT_OCR_CONCURRENCY", "1",
     "Requêtes OCR Albert simultanées par processus."),
    ("ALBERT_OCR_SKIP_RECODE", "0",
     "1 = texte albert_lightonocr exclu du recodage (0 = recodé, défaut prudent)."),
    ("EMBEDDING_PROVIDER", "openai",
     "Fournisseur d'embeddings : openai (text-embedding-3-large, 3072 d) ou albert (bge-m3, 1024 d, exige ALBERT_ENABLED=1)."),
    ("ALBERT_EMBED_MODEL", "bge-m3",
     "Modèle d'embeddings Albert (id épinglé, jamais de repli)."),
    ("ALBERT_EMBED_BATCH", "64",
     "Textes par requête d'embeddings (plafonné à 64)."),
    ("ALBERT_EMBED_MAX_MISSING_RATIO", "0.0",
     "Part d'embeddings manquants tolérée avant l'échec de la phase (0.0 = aucun manque)."),
    ("ALBERT_EMBED_L2_NORMALIZE", "1",
     "1 = vecteurs normalisés L2 (norme enregistrée dans l'espace vectoriel)."),
    ("ALBERT_REASONING_EFFORT", "medium",
     "Effort de raisonnement des modèles gpt-oss (low, medium, high)."),
    ("ALBERT_REASONING_HEADROOM", "2048",
     "Tokens ajoutés à max_tokens pour le raisonnement des modèles gpt-oss."),
    ("ALBERT_MODEL_FALLBACK", "1",
     "1 = repli sur le modèle suivant du rôle après des 503 répétés (jamais sur 404, jamais pour les embeddings)."),
    ("ALBERT_BUSY_RETRIES", "2",
     "Essais courts sur 503 avant le repli de rôle."),
    ("ALBERT_RECODE_RPM", "45",
     "Requêtes par minute du rôle recode (limiteur proactif, environ 90 % du quota)."),
    ("ALBERT_NOTES_RPM", "9",
     "Requêtes par minute du rôle notes."),
    ("ALBERT_OCR_RPM", "45",
     "Requêtes par minute du rôle OCR."),
    ("ALBERT_EMBED_RPM", "450",
     "Requêtes par minute du rôle embed (envois de chunks compris)."),
    ("ALBERT_CHAT_TPM", "115000",
     "Tokens d'entrée par minute, estimés, pour les rôles de chat."),
    ("ALBERT_PROCESS_SHARE", "1.0",
     "Part des quotas attribuée à ce processus (0 < part <= 1)."),
    ("ALBERT_LIMITER_BACKEND", "local",
     "local = limiteur par processus ; redis = partagé (recommandé avec Celery ou plusieurs sessions)."),
    ("ALBERT_LIMITER_REDIS_URL", "",
     "URL Redis du limiteur ; vide = CELERY_BROKER_URL, sinon redis://localhost:6379/0."),
    ("ALBERT_RECODE_CONCURRENCY", "2",
     "Recodages Albert simultanés par processus."),
    ("ALBERT_NOTES_CONCURRENCY", "2",
     "Appels Albert simultanés pour les notes et les fiches, par processus."),
    ("ALBERT_EMBED_CONCURRENCY", "4",
     "Requêtes d'embeddings Albert simultanées."),
    ("ALBERT_PUSH_CONCURRENCY", "1",
     "Envois de chunks simultanés vers une collection Albert."),
    ("ALBERT_MAX_RETRIES", "4",
     "Réessais au plus sur erreur transitoire."),
    ("ALBERT_RETRY_BACKOFF", "2.0",
     "Base (secondes) du backoff exponentiel."),
    ("ALBERT_RETRY_MAX_BACKOFF", "60",
     "Plafond (secondes) du backoff."),
    ("ALBERT_RETRY_AFTER_MAX", "120",
     "Retry-After au-delà duquel un 429 signifie un quota épuisé (abandon)."),
    ("ALBERT_TIMEOUT_CHAT", "120",
     "Délai HTTP (secondes) d'une requête de chat (recodage, filtre de citations)."),
    ("ALBERT_TIMEOUT_NOTES", "300",
     "Délai HTTP (secondes) d'une requête de note ou de fiche."),
    ("ALBERT_TIMEOUT_OCR_PAGE", "120",
     "Délai HTTP (secondes) de l'OCR d'une page par chat."),
    ("ALBERT_TIMEOUT_OCR_DOC", "300",
     "Délai HTTP (secondes) d'une part envoyée à /v1/ocr."),
    ("ALBERT_TIMEOUT_EMBED", "60",
     "Délai HTTP (secondes) d'une requête d'embeddings."),
    ("ALBERT_TIMEOUT_COLLECTIONS", "120",
     "Délai HTTP (secondes) des appels aux collections et documents."),
    ("ALBERT_SUBPROCESS_TIMEOUT", "21600",
     "Délai (secondes) des sous-processus quand la requête sélectionne Albert."),
    ("ALBERT_METADATA_FIELDS", ",".join(DEFAULT_METADATA_FIELDS),
     "Métadonnées envoyées aux collections : 10 au plus, avec content_id, content_hash, chunk_index et DEDUP_META_FIELDS ; jamais path."),
    ("DEDUP_SIM_THRESHOLD_BGE_M3", "",
     "Seuil Tier 3 de la dédup pour bge-m3 ; vide = Tier 3 sauté pour bge-m3."),
    ("ALBERT_USAGE_LOG", "1",
     "1 = ledger albert_usage.jsonl écrit quand Albert a été appelé."),
    ("ALBERT_PREFLIGHT", "1",
     "1 = contrôle du compte et des modèles (/v1/me, /v1/models) avant un traitement Albert."),
    ("ALBERT_TOKEN_ESTIMATOR", "chars",
     "Estimation des tokens d'entrée du limiteur (chars = longueur / 3)."),
)
"""(nom, défaut, sens) de chaque variable Albert lue par le code (hors ``ALBERT_LIVE``)."""


def _check_registry() -> None:
    """Vérifie à l'import que chaque champ a une variable enregistrée."""
    names = {name for name, _default, _meaning in ENV_REGISTRY}
    field_names = {f.name for f in dataclass_fields(AlbertConfig)}
    if set(FIELD_ENV_NAMES) != field_names or not set(FIELD_ENV_NAMES.values()) <= names:
        raise RuntimeError("rad_albert.config : ENV_REGISTRY et AlbertConfig désynchronisés.")


_check_registry()
