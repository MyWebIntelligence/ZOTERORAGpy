"""
Authentication routes: register, login, logout, password reset
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database.session import get_db
from app.models.user import User
from app.models.audit import AuditLog, AuditAction, create_audit_log
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    TokenResponse,
    PasswordResetRequest,
    PasswordReset,
    PasswordChange,
    TokenRefresh
)
from app.schemas.user import UserResponse
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_verification_token,
    generate_reset_token
)
from app.middleware.auth import get_current_user, get_optional_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_client_ip(request: Request) -> str:
    """Extrait l'adresse IP du client"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Inscription d'un nouvel utilisateur.

    Si aucun administrateur n'existe dans le système, le nouvel utilisateur
    devient automatiquement administrateur et est vérifié immédiatement.
    """
    # Vérifier si l'email existe déjà
    existing_user = db.query(User).filter(User.email == user_data.email.lower()).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Un compte avec cet email existe déjà"
        )

    # Vérifier s'il existe au moins un administrateur
    # Si aucun admin n'existe, le nouvel utilisateur devient admin
    admin_exists = db.query(User).filter(
        User.roles.contains(["ADMIN"])
    ).first() is not None

    # Fallback: vérifier aussi avec une requête plus robuste (JSON array check)
    if admin_exists is False:
        # Double check avec une approche différente pour les bases qui ne supportent pas contains
        all_users = db.query(User).all()
        admin_exists = any(u.is_admin for u in all_users)

    should_be_admin = not admin_exists

    # Créer l'utilisateur
    new_user = User(
        email=user_data.email.lower(),
        hashed_password=get_password_hash(user_data.password),
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        organization=user_data.organization,
        title=user_data.title,
        roles=["USER", "ADMIN"] if should_be_admin else ["USER"],
        is_active=True,
        is_verified=should_be_admin,  # Auto-vérifié si devient admin
        verification_token=None if should_be_admin else generate_verification_token()
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.REGISTER,
        user_id=new_user.id,
        resource_type="user",
        resource_id=new_user.id,
        details={
            "email": new_user.email,
            "auto_promoted_admin": should_be_admin,
            "reason": "no_admin_existed" if should_be_admin else None
        },
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        first_name=new_user.first_name,
        last_name=new_user.last_name,
        full_name=new_user.full_name,
        organization=new_user.organization,
        title=new_user.title,
        roles=new_user.roles or [],
        is_active=new_user.is_active,
        is_verified=new_user.is_verified,
        is_admin=new_user.is_admin,
        created_at=new_user.created_at,
        last_login=new_user.last_login
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    user_data: UserLogin,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Connexion d'un utilisateur.

    Retourne un token JWT pour l'authentification des requêtes suivantes.
    """
    # Rechercher l'utilisateur
    user = db.query(User).filter(User.email == user_data.email.lower()).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # Vérifier si le compte est verrouillé
    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Compte temporairement verrouillé. Réessayez après {user.locked_until}"
        )

    # Vérifier le mot de passe
    if not verify_password(user_data.password, user.hashed_password):
        # Incrémenter les tentatives échouées
        user.failed_login_attempts += 1

        # Verrouiller si trop de tentatives
        if user.failed_login_attempts >= settings.MAX_LOGIN_ATTEMPTS:
            user.locked_until = datetime.utcnow() + timedelta(
                minutes=settings.LOCKOUT_DURATION_MINUTES
            )

        db.commit()

        # Log d'audit
        create_audit_log(
            db=db,
            action=AuditAction.LOGIN_FAILED,
            user_id=user.id,
            details={"attempts": user.failed_login_attempts},
            ip_address=get_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            success=False,
            error_message="Invalid password"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # Vérifier si le compte est actif
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Votre compte a été désactivé. Contactez un administrateur."
        )

    # Réinitialiser les tentatives échouées
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.utcnow()
    db.commit()

    # Créer les tokens
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    if user_data.remember_me:
        access_token_expires = timedelta(days=7)

    access_token = create_access_token(
        subject=user.id,
        expires_delta=access_token_expires,
        additional_claims={
            "email": user.email,
            "roles": user.roles or []
        }
    )
    refresh_token = create_refresh_token(subject=user.id)

    # Stocker le token dans un cookie HTTP-only
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax",
        max_age=int(access_token_expires.total_seconds())
    )

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.LOGIN,
        user_id=user.id,
        details={"remember_me": user_data.remember_me},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds())
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Déconnexion de l'utilisateur.

    Supprime le cookie d'authentification.
    """
    # Supprimer le cookie
    response.delete_cookie(key="access_token")

    # Log d'audit si utilisateur connecté
    if current_user:
        create_audit_log(
            db=db,
            action=AuditAction.LOGOUT,
            user_id=current_user.id,
            ip_address=get_client_ip(request),
            user_agent=request.headers.get("User-Agent")
        )

    return {"message": "Déconnexion réussie"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Rafraîchissement du token d'accès.
    """
    payload = decode_token(token_data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de rafraîchissement invalide"
        )

    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé ou désactivé"
        )

    # Créer un nouveau token d'accès
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.id,
        expires_delta=access_token_expires,
        additional_claims={
            "email": user.email,
            "roles": user.roles or []
        }
    )

    # Mettre à jour le cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax",
        max_age=int(access_token_expires.total_seconds())
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds())
    )


@router.post("/forgot-password")
async def forgot_password(
    data: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Demande de réinitialisation de mot de passe.

    Envoie un email avec un lien de reset (si le serveur SMTP est configuré).
    """
    user = db.query(User).filter(User.email == data.email.lower()).first()

    # Toujours retourner succès pour éviter l'énumération d'emails
    if not user:
        return {"message": "Si un compte existe avec cet email, un lien de réinitialisation a été envoyé."}

    # Générer un token de reset
    reset_token = generate_reset_token()
    user.reset_token = reset_token
    user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
    db.commit()

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PASSWORD_RESET_REQUEST,
        user_id=user.id,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    # TODO: Envoyer l'email avec le lien de reset
    # Pour l'instant, on log le token (en dev uniquement)
    if settings.DEBUG:
        print(f"Password reset token for {user.email}: {reset_token}")

    return {"message": "Si un compte existe avec cet email, un lien de réinitialisation a été envoyé."}


@router.post("/reset-password")
async def reset_password(
    data: PasswordReset,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Réinitialisation du mot de passe avec un token.
    """
    user = db.query(User).filter(User.reset_token == data.token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token de réinitialisation invalide"
        )

    # Vérifier l'expiration
    if user.reset_token_expires and user.reset_token_expires < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le token de réinitialisation a expiré"
        )

    # Mettre à jour le mot de passe
    user.hashed_password = get_password_hash(data.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    user.password_changed_at = datetime.utcnow()
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PASSWORD_RESET,
        user_id=user.id,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return {"message": "Mot de passe réinitialisé avec succès"}


@router.post("/change-password")
async def change_password(
    data: PasswordChange,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Changement de mot de passe pour un utilisateur connecté.
    """
    # Vérifier le mot de passe actuel
    if not verify_password(data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mot de passe actuel incorrect"
        )

    # Mettre à jour le mot de passe
    current_user.hashed_password = get_password_hash(data.new_password)
    current_user.password_changed_at = datetime.utcnow()
    db.commit()

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PASSWORD_CHANGE,
        user_id=current_user.id,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return {"message": "Mot de passe modifié avec succès"}


@router.get("/verify-email/{token}")
async def verify_email(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Vérification de l'email avec un token.
    """
    user = db.query(User).filter(User.verification_token == token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token de vérification invalide"
        )

    user.is_verified = True
    user.verification_token = None
    db.commit()

    return RedirectResponse(url="/login?verified=true")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Retourne les informations de l'utilisateur connecté.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        full_name=current_user.full_name,
        organization=current_user.organization,
        title=current_user.title,
        roles=current_user.roles or [],
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )
