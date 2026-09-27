"""
Configuration Module
====================

This module defines the configuration settings for the RAGpy application.
It uses `pydantic-settings` (implicitly via class structure, though here it seems to be a plain class using os.getenv)
or standard environment variable loading to manage configuration.

It handles:
- Base paths (data, uploads, logs).
- Database connection strings.
- Security settings (JWT, CORS).
- External service credentials (OpenAI, Mistral, Pinecone, etc.).
- Email configuration (Resend).

Deployment safety (audit A09, 2026-09-27): ``RAGPY_ENV=production`` makes
``enforce_secure_settings`` refuse to start with the public placeholder JWT
secret (or a secret shorter than 32 characters); in any other environment
the same problems are logged loudly at startup. ``JWT_SECRET_KEY_PREVIOUS``
keeps the credentials encrypted under the former secret readable during a
rotation (``scripts/rotate_credentials_key.py`` re-encrypts them).
"""
import logging
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
LOGS_DIR = BASE_DIR / "logs"

# Créer les répertoires si nécessaire
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# Public placeholder of the repository: never a valid secret in production.
DEFAULT_JWT_SECRET_KEY = "change-this-secret-key-in-production-min-32-chars"
MIN_JWT_SECRET_LENGTH = 32
PRODUCTION_ENV = "production"


def jwt_secret_from_env(env=None) -> str:
    """
    ``JWT_SECRET_KEY`` of ``env`` (``os.environ`` by default), the placeholder when absent or empty.

    An empty value (``JWT_SECRET_KEY=`` copied from ``.env.example``) never
    becomes an empty signing key: it falls back to the placeholder, which
    ``enforce_secure_settings`` reports (and refuses in production).

    Args:
        env: Mapping of environment variables.

    Returns:
        The secret to use.
    """
    source = os.environ if env is None else env
    return source.get("JWT_SECRET_KEY") or DEFAULT_JWT_SECRET_KEY


class Settings:
    """
    Application Settings
    --------------------
    
    This class acts as a centralized configuration holder.
    It reads from environment variables and provides default values.
    """

    # Application
    APP_NAME: str = "MyDoc Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{DATA_DIR}/ragpy.db"
    )

    # Deployment environment: "production" refuses insecure defaults at startup
    RAGPY_ENV: str = (os.getenv("RAGPY_ENV") or "development").strip().lower()

    # JWT Authentication
    JWT_SECRET_KEY: str = jwt_secret_from_env()
    # Former secret during a rotation: stored credentials encrypted under it stay readable
    JWT_SECRET_KEY_PREVIOUS: Optional[str] = os.getenv("JWT_SECRET_KEY_PREVIOUS") or None
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
    )

    # Security
    BCRYPT_ROUNDS: int = 12
    MAX_LOGIN_ATTEMPTS: int = 10
    LOCKOUT_DURATION_MINUTES: int = 15

    # CORS (explicit origins; "*" disables credentialed cross-origin requests)
    CORS_ORIGINS: list = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
        if origin.strip()
    ]

    # Email (Resend)
    RESEND_API_KEY: Optional[str] = os.getenv("RESEND_API_KEY")
    RESEND_FROM_EMAIL: str = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

    # Application URL (pour les liens dans les emails)
    APP_URL: str = os.getenv("APP_URL", "http://localhost:8000")

    # Token expiration
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = int(
        os.getenv("EMAIL_VERIFICATION_EXPIRE_HOURS", "24")
    )
    PASSWORD_RESET_EXPIRE_HOURS: int = int(
        os.getenv("PASSWORD_RESET_EXPIRE_HOURS", "1")
    )

    # API Keys (existantes)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    WEAVIATE_API_KEY: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # User Registration Settings
    # Sandbox mode: when True, new users are blocked until admin approval
    USERS_SANDBOX: bool = os.getenv("USERS_SANDBOX", "FALSE").upper() == "TRUE"


settings = Settings()


def insecure_settings_problems(config: "Settings" = None) -> List[str]:
    """
    List the insecure deployment settings of ``config`` (audit A09).

    Args:
        config: Settings to check (the module ``settings`` by default).

    Returns:
        French messages, one per problem (empty when the settings are safe):
        placeholder or short ``JWT_SECRET_KEY``, wildcard CORS origin.
    """
    config = config or settings
    problems = []
    secret = config.JWT_SECRET_KEY or ""
    if secret == DEFAULT_JWT_SECRET_KEY:
        problems.append(
            "JWT_SECRET_KEY vaut la valeur de remplacement publique du dépôt : n'importe qui peut forger "
            "un jeton et les identifiants chiffrés des utilisateurs sont lisibles. Définir un secret "
            "aléatoire (python -c \"import secrets; print(secrets.token_urlsafe(48))\") puis rechiffrer "
            "les identifiants (JWT_SECRET_KEY_PREVIOUS + scripts/rotate_credentials_key.py)."
        )
    elif len(secret) < MIN_JWT_SECRET_LENGTH:
        problems.append(f"JWT_SECRET_KEY trop court ({len(secret)} caractères, minimum {MIN_JWT_SECRET_LENGTH}).")
    if "*" in (config.CORS_ORIGINS or []):
        problems.append("CORS_ORIGINS contient « * » : requêtes cross-origin sans identifiants seulement.")
    return problems


def enforce_secure_settings(config: "Settings" = None) -> List[str]:
    """
    Refuse to start in production with insecure settings; warn elsewhere.

    ``RAGPY_ENV=production``: a placeholder or short ``JWT_SECRET_KEY``
    raises (the wildcard CORS origin is only logged, since the middleware
    then disables credentials). Other environments log every problem at
    ERROR level so that it shows in the startup logs.

    Args:
        config: Settings to check (the module ``settings`` by default).

    Returns:
        The problems found (logged).

    Raises:
        RuntimeError: In production, when the JWT secret is unsafe.
    """
    config = config or settings
    problems = insecure_settings_problems(config)
    for problem in problems:
        logger.error(f"Configuration non sûre : {problem}")
    secret = config.JWT_SECRET_KEY or ""
    unsafe_secret = secret == DEFAULT_JWT_SECRET_KEY or len(secret) < MIN_JWT_SECRET_LENGTH
    if config.RAGPY_ENV == PRODUCTION_ENV and unsafe_secret:
        raise RuntimeError(
            "Démarrage refusé (RAGPY_ENV=production) : JWT_SECRET_KEY absent, valeur de remplacement "
            "ou trop court. " + " ".join(problems)
        )
    return problems
