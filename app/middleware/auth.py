"""
Authentication middleware and dependencies
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
    Extrait le token JWT de la requête.

    Cherche dans l'ordre:
    1. Header Authorization: Bearer <token>
    2. Cookie access_token
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
    Dependency pour obtenir l'utilisateur actuellement connecté.

    Lève une exception 401 si non authentifié.

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            ...
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
    Dependency pour obtenir l'utilisateur si connecté, None sinon.

    N'élève pas d'exception si non authentifié.

    Usage:
        @app.get("/public")
        async def public_route(user: Optional[User] = Depends(get_optional_user)):
            if user:
                # Utilisateur connecté
            else:
                # Visiteur anonyme
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
    Dependency pour obtenir un utilisateur actif.

    Lève une exception 403 si le compte est désactivé ou verrouillé.

    Usage:
        @app.get("/active-only")
        async def active_route(user: User = Depends(get_current_active_user)):
            ...
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
