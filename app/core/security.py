"""
Security utilities for password hashing and JWT token management
"""
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

# Configuration du contexte de hachage bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie si un mot de passe en clair correspond au hash stocké.

    Args:
        plain_password: Mot de passe en clair
        hashed_password: Hash bcrypt stocké

    Returns:
        True si le mot de passe correspond, False sinon
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Génère un hash bcrypt pour un mot de passe.

    Args:
        password: Mot de passe en clair

    Returns:
        Hash bcrypt du mot de passe
    """
    return pwd_context.hash(password)


def create_access_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[dict] = None
) -> str:
    """
    Crée un token d'accès JWT.

    Args:
        subject: Identifiant du sujet (généralement user_id)
        expires_delta: Durée de validité (défaut: JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        additional_claims: Claims supplémentaires à inclure

    Returns:
        Token JWT encodé
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }

    if additional_claims:
        to_encode.update(additional_claims)

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Crée un token de rafraîchissement JWT.

    Args:
        subject: Identifiant du sujet (généralement user_id)
        expires_delta: Durée de validité (défaut: JWT_REFRESH_TOKEN_EXPIRE_DAYS)

    Returns:
        Token JWT encodé
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Décode et valide un token JWT.

    Args:
        token: Token JWT à décoder

    Returns:
        Payload du token si valide, None sinon
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def generate_verification_token() -> str:
    """
    Génère un token sécurisé pour la vérification d'email ou reset de mot de passe.

    Returns:
        Token aléatoire de 32 caractères hexadécimaux
    """
    return secrets.token_hex(32)


def generate_reset_token() -> str:
    """
    Génère un token sécurisé pour le reset de mot de passe.

    Returns:
        Token aléatoire URL-safe
    """
    return secrets.token_urlsafe(32)


class TokenData:
    """Structure pour les données extraites d'un token"""

    def __init__(
        self,
        user_id: Optional[int] = None,
        token_type: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ):
        self.user_id = user_id
        self.token_type = token_type
        self.expires_at = expires_at


def extract_token_data(token: str) -> Optional[TokenData]:
    """
    Extrait les données structurées d'un token JWT.

    Args:
        token: Token JWT à analyser

    Returns:
        TokenData si valide, None sinon
    """
    payload = decode_token(token)
    if payload is None:
        return None

    try:
        user_id = int(payload.get("sub"))
        token_type = payload.get("type")
        exp = payload.get("exp")
        expires_at = datetime.fromtimestamp(exp) if exp else None

        return TokenData(
            user_id=user_id,
            token_type=token_type,
            expires_at=expires_at
        )
    except (ValueError, TypeError):
        return None
