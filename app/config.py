"""
Configuration centralisée pour RAGpy
"""
import os
from pathlib import Path
from typing import Optional
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


class Settings:
    """Configuration de l'application"""

    # Application
    APP_NAME: str = "MyDoc Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{DATA_DIR}/ragpy.db"
    )

    # JWT Authentication
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY",
        "change-this-secret-key-in-production-min-32-chars"
    )
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

    # CORS
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")

    # Email (optionnel)
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")

    # API Keys (existantes)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    WEAVIATE_API_KEY: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")


settings = Settings()
