"""
Authentication Middleware
=========================

This module defines FastAPI dependencies and utilities for handling user authentication
and authorization. It implements JWT-based security using `HTTPBearer`.

Key Dependencies:
- `get_current_user`: Enforces authentication and returns the current user.
- `get_optional_user`: Returns the current user if authenticated, else None.
- `get_current_active_user`: Ensures the user is authenticated, active, and verified.
- `require_admin`: Restricts access to administrators.
- `require_roles`: Restricts access based on user roles.
"""
from typing import List, Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.user import User
from app.core.security import decode_token

# Security scheme pour le header Authorization
security = HTTPBearer(auto_error=False)


def get_token_from_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Extracts a JWT token from the request.

    It checks for the token in the following order:
    1. The `Authorization` header (as a "Bearer" token).
    2. The `access_token` cookie.

    Args:
        request: The incoming FastAPI `Request` object.
        credentials: The `Authorization` header credentials, if present.

    Returns:
        The extracted token string if found, otherwise None.
    """
    # D'abord vérifier le header Authorization
    if credentials and credentials.credentials:
        return credentials.credentials

    # Ensuite vérifier le cookie
    token = request.cookies.get("access_token")
    if token:
        return token

    return None


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    FastAPI dependency to get the currently authenticated user.

    This function enforces authentication. It retrieves the token from the
    request, decodes it, and fetches the corresponding user from the database.

    Args:
        request: The incoming FastAPI `Request` object.
        db: The database session dependency.
        credentials: The `Authorization` header credentials.

    Returns:
        The authenticated `User` object.

    Raises:
        HTTPException: If the user is not authenticated (401), the token is
                       invalid, or the user is not found.
    """
    token = get_token_from_request(request, credentials)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Non authentifié",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Décoder le token
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Vérifier le type de token
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Type de token invalide",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Récupérer l'utilisateur
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


async def get_optional_user(
    request: Request,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    FastAPI dependency to get the current user if authenticated, or None otherwise.

    This function does not raise an exception if the user is not authenticated,
    making it suitable for endpoints that have optional authentication.

    Args:
        request: The incoming FastAPI `Request` object.
        db: The database session dependency.
        credentials: The `Authorization` header credentials.

    Returns:
        The `User` object if authenticated, otherwise None.
    """
    token = get_token_from_request(request, credentials)

    if not token:
        return None

    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    try:
        user_id = int(payload.get("sub"))
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except (ValueError, TypeError):
        return None


async def get_current_active_user(
    user: User = Depends(get_current_user)
) -> User:
    """
    FastAPI dependency to get an active and verified user.

    This function builds on `get_current_user` by adding checks to ensure
    the user's account is active, not locked, and has been verified.

    Args:
        user: The user object from `get_current_user`.

    Returns:
        The `User` object if the user is active and verified.

    Raises:
        HTTPException: If the account is inactive (403), locked (423), or
                       not verified (403).
    """
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte désactivé"
        )

    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Compte temporairement verrouillé"
        )

    # Vérifier que l'email est confirmé
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Veuillez vérifier votre adresse email pour accéder à cette fonctionnalité. "
                   "Vérifiez votre boîte de réception ou demandez un nouvel email de vérification.",
            headers={"X-Email-Verification-Required": "true"}
        )

    return user


async def require_admin(
    user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency pour vérifier que l'utilisateur est administrateur.

    Lève une exception 403 si l'utilisateur n'est pas admin.

    Usage:
        @app.get("/admin-only")
        async def admin_route(user: User = Depends(require_admin)):
            ...
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs"
        )

    return user


def require_roles(*roles: str):
    """
    Factory de dependency pour vérifier les rôles.

    Usage:
        @app.get("/editor-only")
        async def editor_route(user: User = Depends(require_roles("EDITOR", "ADMIN"))):
            ...
    """
    async def dependency(user: User = Depends(get_current_active_user)) -> User:
        user_roles = user.roles or []
        if not any(role in user_roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rôle requis: {', '.join(roles)}"
            )
        return user

    return Depends(dependency)


def require_verified(user: User = Depends(get_current_active_user)) -> User:
    """
    Dependency pour vérifier que l'email est vérifié.

    Usage:
        @app.get("/verified-only")
        async def verified_route(user: User = Depends(require_verified)):
            ...
    """
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Veuillez vérifier votre email avant de continuer"
        )

    return user
