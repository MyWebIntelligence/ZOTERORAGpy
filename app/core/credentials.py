"""
Credential Management Module
============================

This module handles the secure encryption, decryption, and management of user API keys.
It uses Fernet symmetric encryption (from the `cryptography` library) to store sensitive
credentials in the database. The encryption key is derived from the application's
`JWT_SECRET_KEY` to ensure security.

Key Features:
- Derivation of Fernet key from `JWT_SECRET_KEY`.
- Encryption and decryption of credential dictionaries.
- Masking of credentials for safe UI display.
- Role-based credential access (ADMIN can fallback to .env, non-admin cannot).
- Subprocess environment builder for secure credential injection (non-admin
  subprocesses lose the server secrets of SERVER_SECRET_ENV_VARS and receive
  RAGPY_DOTENV_DENY so they cannot reload the stripped credential *_API_KEY
  secrets from .env).

Security Model:
    - ADMIN users: Personal credentials first, then fallback to .env
    - NON-ADMIN users: Personal credentials ONLY, no .env access
"""
import os
import json
import base64
import functools
import hashlib
import logging
from typing import Dict, Optional, Any, List
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.config import settings
from app.models.user import User
from scripts.rad_settings.access import refresh_environ
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class CredentialMissingError(Exception):
    """
    Exception raised when a required credential is missing for a user.

    This exception is used to provide clear, user-friendly error messages
    when a non-admin user attempts to use a feature requiring credentials
    they haven't configured.

    Attributes:
        credential_key: The missing credential key (e.g., "openai_api_key").
        message: User-friendly error message.
        is_admin: Whether the user is an admin (affects error message).
    """

    def __init__(self, credential_key: str, message: str = None, is_admin: bool = False):
        self.credential_key = credential_key
        self.is_admin = is_admin
        if message is None:
            message = get_credential_error_message(credential_key)
        super().__init__(message)


# Derive a Fernet-compatible key from JWT_SECRET_KEY
def _derive_fernet_key(secret: str) -> bytes:
    """
    Derive a 32-byte Fernet key from a secret (SHA256, then urlsafe base64).

    Args:
        secret: ``JWT_SECRET_KEY`` or ``JWT_SECRET_KEY_PREVIOUS``.

    Returns:
        The base64-encoded Fernet key.
    """
    hashed = hashlib.sha256(secret.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(hashed)


def _get_encryption_key() -> bytes:
    """
    Derive a 32-byte key from JWT_SECRET_KEY for Fernet encryption.
    Uses SHA256 hash and base64 encoding.
    """
    return _derive_fernet_key(settings.JWT_SECRET_KEY)


def get_fernet() -> MultiFernet:
    """
    Get the Fernet instance of the stored credentials.

    Encrypts with the key derived from ``JWT_SECRET_KEY``; decrypts with it
    and, during a rotation, with the key derived from
    ``JWT_SECRET_KEY_PREVIOUS`` (audit A09: rotating the JWT secret never
    makes the stored credentials unreadable; ``rotate`` re-encrypts them).

    Returns:
        A ``MultiFernet`` (same ``encrypt``/``decrypt`` API as ``Fernet``).
    """
    keys = [Fernet(_get_encryption_key())]
    previous = getattr(settings, "JWT_SECRET_KEY_PREVIOUS", None)
    if previous and previous != settings.JWT_SECRET_KEY:
        keys.append(Fernet(_derive_fernet_key(previous)))
    return MultiFernet(keys)


# List of all credential keys that can be stored per-user
CREDENTIAL_KEYS = [
    # OpenAI
    "openai_api_key",
    # OpenRouter
    "openrouter_api_key",
    "openrouter_model",
    # Mistral
    "mistral_api_key",
    "mistral_model",
    "mistral_url",
    # Albert (DINUM). Always registered, even when Albert is disabled, so that
    # the non-admin purge of build_subprocess_env covers ALBERT_API_KEY. Hidden
    # from UI/JSON listings when disabled: see visible_credential_keys().
    # ALBERT_BASE_URL is server configuration, never a user credential.
    "albert_api_key",
    # Servers added by the « configuration unifiée » sprint (lot L9): personal
    # keys, listed only when the administrator declares the server address in
    # block 1 of the .env (see visible_credential_keys()).
    "anthropic_api_key",
    "google_api_key",
    "deepseek_api_key",
    "qwen_api_key",
    "glm_api_key",
    # Pinecone
    "pinecone_api_key",
    "pinecone_env",
    # Weaviate
    "weaviate_api_key",
    "weaviate_url",
    # Qdrant
    "qdrant_api_key",
    "qdrant_url",
    # Zotero
    "zotero_api_key",
    "zotero_user_id",
    "zotero_group_id",
]

# Mapping from credential key to environment variable name
CREDENTIAL_ENV_MAPPING = {
    "openai_api_key": "OPENAI_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "openrouter_model": "OPENROUTER_DEFAULT_MODEL",
    "mistral_api_key": "MISTRAL_API_KEY",
    "mistral_model": "MISTRAL_OCR_MODEL",
    "mistral_url": "MISTRAL_API_BASE_URL",
    "albert_api_key": "ALBERT_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "qwen_api_key": "QWEN_API_KEY",
    "glm_api_key": "GLM_API_KEY",
    "pinecone_api_key": "PINECONE_API_KEY",
    "pinecone_env": "PINECONE_ENV",
    "weaviate_api_key": "WEAVIATE_API_KEY",
    "weaviate_url": "WEAVIATE_URL",
    "qdrant_api_key": "QDRANT_API_KEY",
    "qdrant_url": "QDRANT_URL",
    "zotero_api_key": "ZOTERO_API_KEY",
    "zotero_user_id": "ZOTERO_USER_ID",
    "zotero_group_id": "ZOTERO_GROUP_ID",
}

# User-friendly error messages for missing credentials (French)
CREDENTIAL_ERROR_MESSAGES = {
    "openai_api_key": "Clé API OpenAI requise. Configurez-la dans Paramètres > Mes Identifiants.",
    "openrouter_api_key": "Clé API OpenRouter requise pour ce modèle. Configurez-la dans Paramètres > Mes Identifiants.",
    "openrouter_model": "Modèle OpenRouter requis. Configurez-le dans Paramètres > Mes Identifiants.",
    "mistral_api_key": "Clé API Mistral requise pour l'OCR des PDFs. Configurez-la dans Paramètres > Mes Identifiants.",
    "mistral_model": "Modèle Mistral OCR requis.",
    "mistral_url": "URL API Mistral requise.",
    "albert_api_key": "Clé API Albert (DINUM) requise. Configurez-la dans Paramètres > Mes Identifiants.",
    "anthropic_api_key": "Clé API Anthropic requise pour ce serveur. Configurez-la dans Paramètres > Mes Identifiants.",
    "google_api_key": "Clé API Google Gemini requise pour ce serveur. Configurez-la dans Paramètres > Mes Identifiants.",
    "deepseek_api_key": "Clé API DeepSeek requise pour ce serveur. Configurez-la dans Paramètres > Mes Identifiants.",
    "qwen_api_key": "Clé API Qwen (Alibaba Cloud) requise pour ce serveur. Configurez-la dans Paramètres > Mes Identifiants.",
    "glm_api_key": "Clé API GLM (Z.ai) requise pour ce serveur. Configurez-la dans Paramètres > Mes Identifiants.",
    "pinecone_api_key": "Clé API Pinecone requise. Configurez-la dans Paramètres > Mes Identifiants.",
    "pinecone_env": "Environnement Pinecone requis (ex: us-east-1).",
    "weaviate_api_key": "Clé API Weaviate requise. Configurez-la dans Paramètres > Mes Identifiants.",
    "weaviate_url": "URL Weaviate requise (ex: https://xxx.weaviate.network).",
    "qdrant_api_key": "Clé API Qdrant requise. Configurez-la dans Paramètres > Mes Identifiants.",
    "qdrant_url": "URL Qdrant requise (ex: https://xxx.qdrant.io).",
    "zotero_api_key": "Clé API Zotero requise pour la synchronisation. Configurez-la dans Paramètres > Mes Identifiants.",
    "zotero_user_id": "ID utilisateur Zotero requis. Configurez-le dans Paramètres > Mes Identifiants.",
    "zotero_group_id": "ID groupe Zotero requis (si bibliothèque de groupe). Configurez-le dans Paramètres > Mes Identifiants.",
}


# Credential shown in UI/JSON listings only when Albert is enabled (see albert_enabled()).
_ALBERT_CREDENTIAL_KEY = "albert_api_key"

# Personal keys of the servers added by the « configuration unifiée » sprint
# (lot L9): credential key -> address variable of block 1. Each one is shown in
# UI/JSON listings, and accepted by PUT /users/me/credentials, only when the
# administrator declares that address (see server_credential_declared()).
SERVER_CREDENTIAL_BASE_URLS = {
    "anthropic_api_key": "ANTHROPIC_API_BASE_URL",
    "google_api_key": "GOOGLE_API_BASE_URL",
    "deepseek_api_key": "DEEPSEEK_API_BASE_URL",
    "qwen_api_key": "QWEN_API_BASE_URL",
    "glm_api_key": "GLM_API_BASE_URL",
}

# Master switch of the Albert integration (server configuration).
_ALBERT_SWITCH_ENV_VAR = "ALBERT_ENABLED"

# Environment variable set for NON-ADMIN subprocesses by build_subprocess_env:
# sorted, comma-separated (no spaces) names of the *_API_KEY variables that were
# stripped and not re-injected, which the subprocess must not reload from .env
# (read by scripts/rad_env.load_dotenv_guarded).
DOTENV_DENY_ENV_VAR = "RAGPY_DOTENV_DENY"

# Server secrets that the pipeline scripts never need. NON-ADMIN subprocesses
# do not inherit them (JWT_SECRET_KEY also derives the Fernet key that decrypts
# every stored user credential). They are only stripped from the inherited env:
# RAGPY_DOTENV_DENY keeps its contract (mapped *_API_KEY names only).
SERVER_SECRET_ENV_VARS = (
    # Secrets serveur, et clé du serveur local (sprint « configuration unifiée »),
    # réservée aux administrateurs. Les clés Anthropic, Google, DeepSeek, Qwen et
    # GLM sont des identifiants personnels depuis le lot L9. Ordre alphabétique.
    "FLOWER_PASSWORD", "JWT_SECRET_KEY", "JWT_SECRET_KEY_PREVIOUS", "LOCAL_API_KEY", "RESEND_API_KEY",
)


@functools.lru_cache(maxsize=16)
def _parse_albert_switch(raw: str) -> bool:
    """
    Interpret a raw ``ALBERT_ENABLED`` value with the parser of ``AlbertConfig``.

    Only the switch is handed to ``AlbertConfig.from_env``, so an invalid
    ``ALBERT_BASE_URL`` never raises here and nothing is logged. The stdlib
    module ``rad_albert.config`` is imported with the double-import pattern of
    the pipeline scripts; when it cannot be imported, Albert is considered OFF.
    The result only depends on ``raw``, hence the cache.

    Args:
        raw: The raw value of ``ALBERT_ENABLED`` (never empty here).

    Returns:
        True when the parser of ``AlbertConfig`` reads the value as enabled.
    """
    try:
        from scripts.rad_albert.config import AlbertConfig
    except ImportError:
        try:
            from rad_albert.config import AlbertConfig
        except ImportError:
            return False
    return bool(AlbertConfig.from_env({_ALBERT_SWITCH_ENV_VAR: raw}).enabled)


def albert_enabled() -> bool:
    """
    Tell whether the server enables Albert (master switch ``ALBERT_ENABLED``).

    Single switch shared by the credential listings, the settings form, the
    ``/api/albert/*`` routes and the page templates. The value is interpreted
    by the parser of ``AlbertConfig.from_env`` (``scripts/rad_albert/config.py``:
    ``1``, ``true``, ``yes`` or ``on``, case-insensitive, surrounding blanks
    ignored), so the web layer, the pipeline scripts and ``EmbeddingConfig``
    always agree. The environment is read at call time; an absent or empty
    value means OFF without importing anything.

    Returns:
        True when Albert is enabled.
    """
    raw = os.environ.get(_ALBERT_SWITCH_ENV_VAR)
    if raw is None or not raw.strip():
        return False
    return _parse_albert_switch(raw)


def server_credential_declared(credential_key: str) -> bool:
    """
    Tell whether the server of a lot L9 personal key is declared in block 1.

    Args:
        credential_key: A key of ``SERVER_CREDENTIAL_BASE_URLS`` (any other key
            answers True: it is not gated by a server address).

    Returns:
        True when the address variable of that server is set (environment read
        at call time).
    """
    var = SERVER_CREDENTIAL_BASE_URLS.get(credential_key)
    return var is None or bool((os.environ.get(var) or "").strip())


def visible_credential_keys() -> List[str]:
    """
    List the credential keys that may be exposed in UI forms and JSON responses.

    ``albert_api_key`` stays registered in ``CREDENTIAL_KEYS`` (so storage and
    the non-admin purge always cover it), but it is hidden from listings unless
    the server enables Albert (``albert_enabled()``). Likewise the personal
    keys of the servers of ``SERVER_CREDENTIAL_BASE_URLS`` are listed only when
    their address is declared (``server_credential_declared()``). The
    environment is read at call time.

    Returns:
        A new list following the order of ``CREDENTIAL_KEYS``.
    """
    enabled = albert_enabled()
    return [
        key for key in CREDENTIAL_KEYS
        if (enabled or key != _ALBERT_CREDENTIAL_KEY) and server_credential_declared(key)
    ]


def get_credential_error_message(credential_key: str) -> str:
    """
    Get a user-friendly error message for a missing credential.

    Args:
        credential_key: The credential key (e.g., "openai_api_key").

    Returns:
        Localized error message in French.
    """
    return CREDENTIAL_ERROR_MESSAGES.get(
        credential_key,
        f"Le credential '{credential_key}' est requis. Configurez-le dans Paramètres > Mes Identifiants."
    )


def encrypt_credentials(credentials: Dict[str, str]) -> str:
    """
    Encrypt a dictionary of credentials to a string for database storage.

    Args:
        credentials: Dict of credential key -> value

    Returns:
        Encrypted string (base64 encoded)
    """
    # Filter to only valid keys and non-empty values
    filtered = {
        k: v for k, v in credentials.items()
        if k in CREDENTIAL_KEYS and v
    }

    if not filtered:
        return ""

    json_str = json.dumps(filtered, ensure_ascii=False)
    fernet = get_fernet()
    encrypted = fernet.encrypt(json_str.encode('utf-8'))
    return encrypted.decode('utf-8')


def decrypt_credentials(encrypted_str: str) -> Dict[str, str]:
    """
    Decrypt a stored credential string back to a dictionary.

    Args:
        encrypted_str: Encrypted string from database

    Returns:
        Dict of credential key -> value, or empty dict on error
    """
    if not encrypted_str:
        return {}

    try:
        fernet = get_fernet()
        decrypted = fernet.decrypt(encrypted_str.encode('utf-8'))
        return json.loads(decrypted.decode('utf-8'))
    except (InvalidToken, json.JSONDecodeError, Exception):
        return {}


def mask_credential(value: str, show_chars: int = 4) -> str:
    """
    Mask a credential value for display, showing only last N characters.

    Args:
        value: The credential value to mask
        show_chars: Number of characters to show at the end

    Returns:
        Masked string like "••••••••abcd"
    """
    if not value:
        return ""

    if len(value) <= show_chars:
        return "•" * len(value)

    return "•" * (len(value) - show_chars) + value[-show_chars:]


def get_user_credentials(user: User) -> Dict[str, str]:
    """
    Get decrypted credentials for a user.

    Args:
        user: User model instance

    Returns:
        Dict of credential key -> value
    """
    if not user or not hasattr(user, 'api_credentials') or not user.api_credentials:
        return {}

    return decrypt_credentials(user.api_credentials)


def get_credential_or_env(
    user: User,
    credential_key: str,
    env_key: str = None,
    raise_if_missing: bool = False
) -> Optional[str]:
    """
    Get a credential value from user settings, with role-based fallback to environment.

    Security Model:
        - ADMIN users: Personal credentials first, then fallback to .env
        - NON-ADMIN users: Personal credentials ONLY, no .env access

    Args:
        user: User model instance.
        credential_key: Key in user credentials (e.g., "openai_api_key").
        env_key: Environment variable name (e.g., "OPENAI_API_KEY"),
                 defaults to CREDENTIAL_ENV_MAPPING or uppercase of credential_key.
        raise_if_missing: If True, raise CredentialMissingError instead of returning None.

    Returns:
        Credential value or None.

    Raises:
        CredentialMissingError: If raise_if_missing=True and credential not found.
    """
    # First try user credentials (all users)
    user_creds = get_user_credentials(user)
    if user_creds.get(credential_key):
        logger.debug(f"Credential '{credential_key}' found in user credentials")
        return user_creds[credential_key]

    # Only ADMIN users can fall back to environment variables
    if user and hasattr(user, 'is_admin') and user.is_admin:
        if env_key is None:
            env_key = CREDENTIAL_ENV_MAPPING.get(credential_key, credential_key.upper())
        env_value = os.getenv(env_key)
        if env_value:
            logger.debug(f"Admin user fallback to .env for '{credential_key}'")
            return env_value

    # Non-admin users cannot access .env - log warning
    if user and not getattr(user, 'is_admin', False):
        logger.debug(f"Non-admin user denied .env fallback for '{credential_key}'")

    # Credential not found
    if raise_if_missing:
        is_admin = user.is_admin if user and hasattr(user, 'is_admin') else False
        raise CredentialMissingError(
            credential_key=credential_key,
            is_admin=is_admin
        )

    return None


def get_masked_credentials(user: User) -> Dict[str, Any]:
    """
    Get credentials with values masked for safe display.

    Args:
        user: User model instance

    Returns:
        Dict with 'has_value' boolean and 'masked' string for each credential
    """
    credentials = get_user_credentials(user)

    result = {}
    for key in CREDENTIAL_KEYS:
        value = credentials.get(key, "")
        result[key] = {
            "has_value": bool(value),
            "masked": mask_credential(value) if value else ""
        }

    return result


def update_user_credentials(user: User, updates: Dict[str, str], db: Session) -> None:
    """
    Update specific credentials for a user, preserving existing ones.

    Args:
        user: User model instance
        updates: Dict of credential key -> new value (empty string to delete)
        db: Database session
    """
    # Get existing credentials
    current = get_user_credentials(user)

    # Apply updates
    for key, value in updates.items():
        if key not in CREDENTIAL_KEYS:
            continue

        if value:  # Set or update
            current[key] = value
        elif key in current:  # Delete if empty
            del current[key]

    # Encrypt and save
    user.api_credentials = encrypt_credentials(current) if current else None
    db.commit()


def _normalise_mistral_base_url(value: str) -> str:
    """
    Strip any path suffix from a stored Mistral base URL.

    The codebase appends specific paths (`/v1/files`, `/v1/ocr`) to
    `MISTRAL_API_BASE_URL`, so the env value must be the bare host. Some user
    credential records have historically stored the OCR endpoint
    (`https://api.mistral.ai/v1/ocr`) here, leading to malformed requests
    such as `https://api.mistral.ai/v1/ocr/v1/files` that 404.

    Returns the base URL trimmed to scheme+host (e.g. `https://api.mistral.ai`).
    Logs a warning when normalisation actually changes the input.
    """
    if not value:
        return value
    cleaned = value.strip().rstrip("/")
    # Match any URL whose path is non-empty and re-emit only scheme+netloc.
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(cleaned)
        if parsed.scheme and parsed.netloc and parsed.path:
            normalised = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
            if normalised != cleaned:
                logger.warning(
                    "Mistral base URL credential normalised: %r → %r "
                    "(le suffixe de chemin a été retiré pour éviter les 404).",
                    value, normalised,
                )
                return normalised
    except Exception as exc:
        logger.debug("Mistral URL normalisation skipped: %s", exc)
    return cleaned


def build_subprocess_env(
    user: User,
    required_keys: List[str] = None
) -> Dict[str, str]:
    """
    Build environment dictionary for subprocess with user credentials.

    This function creates a copy of the current environment and injects
    the user's credentials, ensuring proper isolation from system-wide
    .env variables for non-admin users.

    Security Model:
        - ADMIN users: Keep existing .env credentials, overlay with personal credentials.
        - NON-ADMIN users: REMOVE all credential env vars and the server secrets
          of ``SERVER_SECRET_ENV_VARS`` (never needed by the pipeline scripts),
          inject only personal credentials, and set ``RAGPY_DOTENV_DENY`` to the
          sorted, comma-separated (no spaces) names of the ``*_API_KEY``
          variables of ``CREDENTIAL_ENV_MAPPING`` that were stripped and not
          re-injected, so the subprocess cannot reload them from ``.env``
          (see ``scripts/rad_env.load_dotenv_guarded``). The variable may be an
          empty string when every such key was re-injected. It is never set for
          ADMIN users. The server secrets are only stripped from the inherited
          environment: they are not part of the deny-list.

    Args:
        user: User model instance.
        required_keys: Optional list of required credential keys. If any are missing,
                      raises CredentialMissingError.

    Returns:
        Dictionary of environment variables safe for subprocess execution.

    Raises:
        CredentialMissingError: If a required credential is missing.

    Example:
        >>> env = build_subprocess_env(user, required_keys=["openai_api_key"])
        >>> process = await asyncio.create_subprocess_exec(*cmd, env=env)
    """
    # Start with a copy of the current environment, resynchronised with the
    # .env first (an edit made since startup reaches the script).
    refresh_environ()
    env = os.environ.copy()

    # Get user's personal credentials
    user_creds = get_user_credentials(user)
    is_admin = user and hasattr(user, 'is_admin') and user.is_admin

    # For NON-ADMIN users: CLEAR all credential env vars first
    # This prevents leakage of .env credentials to subprocess
    if not is_admin:
        logger.info(f"Building subprocess env for non-admin user - clearing .env credentials")
        for env_key in CREDENTIAL_ENV_MAPPING.values():
            env.pop(env_key, None)
        # Server secrets (JWT signing key, e-mail API key, Flower password)
        # are never needed by the pipeline scripts.
        for env_key in SERVER_SECRET_ENV_VARS:
            env.pop(env_key, None)
    else:
        logger.debug(f"Building subprocess env for admin user - keeping .env fallbacks")
        # The dotenv deny-list only concerns non-admin subprocesses.
        env.pop(DOTENV_DENY_ENV_VAR, None)

    # Inject user's personal credentials (overrides .env for admins, sets for non-admins)
    injected_env_keys = set()
    for cred_key, env_key in CREDENTIAL_ENV_MAPPING.items():
        value = user_creds.get(cred_key)
        if value:
            # Defensive normalisation for known-broken historical values.
            # Some users saved `mistral_url` as `https://api.mistral.ai/v1/ocr`
            # (the OCR endpoint path leaked into the base URL setting), which
            # then produces requests to `.../v1/ocr/v1/files` and fails with
            # 404. Strip any `/v1/...` suffix so the base URL is the host.
            if cred_key == "mistral_url":
                value = _normalise_mistral_base_url(value)
            env[env_key] = value
            injected_env_keys.add(env_key)
            logger.debug(f"Injected user credential '{cred_key}' as '{env_key}'")

    # For NON-ADMIN users: forbid the subprocess from reloading the stripped
    # secrets from .env (names only, never values).
    if not is_admin:
        denied = sorted(
            env_key for env_key in CREDENTIAL_ENV_MAPPING.values()
            if env_key.endswith("_API_KEY") and env_key not in injected_env_keys
        )
        env[DOTENV_DENY_ENV_VAR] = ",".join(denied)

    # Validate required credentials
    if required_keys:
        missing = []
        for cred_key in required_keys:
            env_key = CREDENTIAL_ENV_MAPPING.get(cred_key, cred_key.upper())
            if not env.get(env_key):
                missing.append(cred_key)

        if missing:
            # Raise error for the first missing credential
            raise CredentialMissingError(
                credential_key=missing[0],
                is_admin=is_admin
            )

    return env
