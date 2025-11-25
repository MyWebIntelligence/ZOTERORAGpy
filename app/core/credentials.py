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
- Retrieval of credentials with environment variable fallback.
"""
import json
import base64
import hashlib
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet, InvalidToken

from app.config import settings
from app.models.user import User
from sqlalchemy.orm import Session


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


def get_credential_or_env(user: User, credential_key: str, env_key: str = None) -> Optional[str]:
    """
    Get a credential value from user settings, falling back to environment.

    Args:
        user: User model instance
        credential_key: Key in user credentials (e.g., "openai_api_key")
        env_key: Environment variable name (e.g., "OPENAI_API_KEY"), defaults to uppercase of credential_key

    Returns:
        Credential value or None
    """
    import os

    # First try user credentials
    user_creds = get_user_credentials(user)
    if user_creds.get(credential_key):
        return user_creds[credential_key]

    # Fall back to environment
    if env_key is None:
        env_key = credential_key.upper()

    return os.getenv(env_key)


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
