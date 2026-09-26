"""Socle de l'intégration Albert (DINUM) pour RAGpy.

Importable en ``scripts.rad_albert`` (application, Celery, tests) et en
``rad_albert`` (scripts CLI lancés avec ``scripts/`` sur ``sys.path``).

Ce ``__init__`` reste **léger** : il ne réexporte que ``config`` et
``errors``, deux modules de la bibliothèque standard. Les modules qui
dépendent de httpx (client, limiteur, OCR, collections…) s'importent
explicitement ; ainsi, Albert désactivé, ni httpx ni openai ne sont chargés
par ce paquet.
"""

from .config import (
    ALLOWED_HOSTS,
    DEFAULT_BASE_URL,
    DEFAULT_METADATA_FIELDS,
    ENV_REGISTRY,
    FIELD_ENV_NAMES,
    METADATA_MAX_FIELDS,
    REQUIRED_METADATA_FIELDS,
    AlbertConfig,
    normalise_base_url,
    root_url,
    validate_metadata_fields,
)
from .errors import (
    AUTH_REASONS,
    MESSAGES_FR,
    PERMANENT_REASONS,
    QUOTA_REASON,
    AlbertAuthError,
    AlbertDisabledError,
    AlbertError,
    AlbertModelBusy,
    AlbertPermanentError,
    AlbertQuotaExhausted,
    AlbertTransientError,
    AlbertTruncatedError,
    AlbertUncertainWriteError,
    classify_http_error,
    is_ocr_access_denied,
    redact,
)

__all__ = [
    # config
    "ALLOWED_HOSTS",
    "DEFAULT_BASE_URL",
    "DEFAULT_METADATA_FIELDS",
    "ENV_REGISTRY",
    "FIELD_ENV_NAMES",
    "METADATA_MAX_FIELDS",
    "REQUIRED_METADATA_FIELDS",
    "AlbertConfig",
    "normalise_base_url",
    "root_url",
    "validate_metadata_fields",
    # errors
    "AUTH_REASONS",
    "MESSAGES_FR",
    "PERMANENT_REASONS",
    "QUOTA_REASON",
    "AlbertAuthError",
    "AlbertDisabledError",
    "AlbertError",
    "AlbertModelBusy",
    "AlbertPermanentError",
    "AlbertQuotaExhausted",
    "AlbertTransientError",
    "AlbertTruncatedError",
    "AlbertUncertainWriteError",
    "classify_http_error",
    "is_ocr_access_denied",
    "redact",
]
