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
- Subprocess environment builder for secure credential injection.

Security Model:
    - ADMIN users: Personal credentials first, then fallback to .env
    - NON-ADMIN users: Personal credentials ONLY, no .env access
"""
import os
import json
import base64
import hashlib
import logging
from typing import Dict, Optional, Any, List
from cryptography.fernet import Fernet, InvalidToken

from app.config import settings
from app.models.user import User
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
def _get_encryption_key() -> bytes:
    """
    Derive a 32-byte key from JWT_SECRET_KEY for Fernet encryption.
    Uses SHA256 hash and base64 encoding.
    """
    key_bytes = settings.JWT_SECRET_KEY.encode('utf-8')
    # SHA256 produces 32 bytes, which we base64 encode for Fernet
    hashed = hashlib.sha256(key_bytes).digest()
    return base64.urlsafe_b64encode(hashed)


def get_fernet() -> Fernet:
    """Get Fernet instance with derived key."""
    return Fernet(_get_encryption_key())


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
        - NON-ADMIN users: REMOVE all credential env vars, inject only personal credentials.

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
    # Start with a copy of current environment
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
    else:
        logger.debug(f"Building subprocess env for admin user - keeping .env fallbacks")

    # Inject user's personal credentials (overrides .env for admins, sets for non-admins)
    for cred_key, env_key in CREDENTIAL_ENV_MAPPING.items():
        value = user_creds.get(cred_key)
        if value:
            env[env_key] = value
            logger.debug(f"Injected user credential '{cred_key}' as '{env_key}'")

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
