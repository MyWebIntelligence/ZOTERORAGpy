"""
Security Utilities Module
=========================

This module provides essential security functions for the application, including:
- Password hashing and verification using `bcrypt`.
- JWT (JSON Web Token) creation, decoding, and validation for authentication.
- Secure token generation for email verification and password resets.

It relies on the `python-jose` library for JWT operations and `bcrypt` for password hashing.
"""
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import bcrypt
from jose import JWTError, jwt

from app.config import settings


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies if a plain text password matches a stored bcrypt hash.

    Args:
        plain_password: The plain text password to verify.
        hashed_password: The stored bcrypt hash.

    Returns:
        True if the password matches, False otherwise.
    """
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def get_password_hash(password: str) -> str:
    """
    Generates a bcrypt hash for a given password.

    Args:
        password: The plain text password to hash.

    Returns:
        The bcrypt hash of the password as a string.
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def create_access_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[dict] = None
) -> str:
    """
    Creates a new JWT access token.

    Args:
        subject: The subject of the token (typically the user ID).
        expires_delta: The lifespan of the token. Defaults to the value of
                       `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` from settings.
        additional_claims: Optional dictionary of additional claims to include
                           in the token payload.

    Returns:
        The encoded JWT access token as a string.
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
    Creates a new JWT refresh token.

    Args:
        subject: The subject of the token (typically the user ID).
        expires_delta: The lifespan of the token. Defaults to the value of
                       `JWT_REFRESH_TOKEN_EXPIRE_DAYS` from settings.

    Returns:
        The encoded JWT refresh token as a string.
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
    Decodes and validates a JWT token.

    Args:
        token: The JWT token to decode.

    Returns:
        The token's payload as a dictionary if it is valid, otherwise None.
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
    Generates a secure token for email verification or password reset.

    Returns:
        A random 32-character hexadecimal token.
    """
    return secrets.token_hex(32)


def generate_reset_token() -> str:
    """
    Generates a secure token for password reset.

    Returns:
        A URL-safe random token.
    """
    return secrets.token_urlsafe(32)


class TokenData:
    """A data class for holding structured data extracted from a JWT."""

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
    Extracts structured data from a JWT token.

    Args:
        token: The JWT token to analyze.

    Returns:
        A `TokenData` object containing the user ID, token type, and expiration
        if the token is valid, otherwise None.
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
