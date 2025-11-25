"""
User Profile Routes
===================

This module handles user profile management. It allows users to view and update
their own profile information and manage their personal API credentials.

Key Features:
- Profile Management: View and update personal details (name, organization, etc.).
- Account Deletion: Allow users to delete their own accounts (with safeguards).
- Credential Management: Manage personal API keys (masked for security).
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.user import User
from app.models.audit import AuditAction, create_audit_log
from app.schemas.user import UserResponse, UserUpdate, UserCredentialsResponse, UserCredentialsUpdate, CredentialValue
from app.middleware.auth import get_current_active_user
from app.core.credentials import (
    get_masked_credentials,
    update_user_credentials,
    CREDENTIAL_KEYS
)

router = APIRouter(prefix="/users", tags=["Users"])


def get_client_ip(request: Request) -> str:
    """Extrait l'adresse IP du client"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("/me", response_model=UserResponse)
async def get_my_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    Retourne le profil de l'utilisateur connecté.
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


@router.put("/me", response_model=UserResponse)
async def update_my_profile(
    user_data: UserUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Met à jour le profil de l'utilisateur connecté.
    """
    # Mettre à jour les champs modifiables
    if user_data.first_name is not None:
        current_user.first_name = user_data.first_name
    if user_data.last_name is not None:
        current_user.last_name = user_data.last_name
    if user_data.organization is not None:
        current_user.organization = user_data.organization
    if user_data.title is not None:
        current_user.title = user_data.title

    db.commit()
    db.refresh(current_user)

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.USER_UPDATE,
        user_id=current_user.id,
        resource_type="user",
        resource_id=current_user.id,
        details=user_data.model_dump(exclude_unset=True),
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

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


@router.delete("/me")
async def delete_my_account(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime le compte de l'utilisateur connecté.

    Note: Un admin ne peut pas supprimer son propre compte s'il est le seul admin.
    """
    if current_user.is_admin:
        # Vérifier s'il y a d'autres admins
        # Note: On utilise is_admin property car JSON contains peut etre peu fiable sur SQLite
        other_users = db.query(User).filter(
            User.id != current_user.id,
            User.is_active == True
        ).all()

        other_admins = [u for u in other_users if u.is_admin]

        if len(other_admins) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Impossible de supprimer le dernier administrateur"
            )

    # Log d'audit avant suppression
    create_audit_log(
        db=db,
        action=AuditAction.USER_DELETE,
        user_id=current_user.id,
        resource_type="user",
        resource_id=current_user.id,
        details={"email": current_user.email, "self_delete": True},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    # Supprimer l'utilisateur
    db.delete(current_user)
    db.commit()

    return {"message": "Compte supprimé avec succès"}


# --- Credentials Management ---

@router.get("/me/credentials", response_model=UserCredentialsResponse)
async def get_my_credentials(
    current_user: User = Depends(get_current_active_user)
):
    """
    Retourne les credentials de l'utilisateur connecte (valeurs masquees).
    """
    masked = get_masked_credentials(current_user)

    # Build response with all credential keys
    response_data = {}
    for key in CREDENTIAL_KEYS:
        if key in masked:
            response_data[key] = CredentialValue(**masked[key])
        else:
            response_data[key] = CredentialValue(has_value=False, masked="")

    return UserCredentialsResponse(**response_data)


@router.put("/me/credentials")
async def update_my_credentials(
    credentials: UserCredentialsUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Met a jour les credentials de l'utilisateur connecte.

    Seuls les champs fournis sont mis a jour.
    Pour supprimer un credential, envoyer une chaine vide.
    """
    # Convert Pydantic model to dict, excluding None values
    updates = credentials.model_dump(exclude_unset=True)

    if not updates:
        return {"message": "Aucune modification"}

    # Update credentials
    update_user_credentials(current_user, updates, db)

    # Log d'audit (sans les valeurs sensibles)
    create_audit_log(
        db=db,
        action=AuditAction.USER_UPDATE,
        user_id=current_user.id,
        resource_type="user_credentials",
        resource_id=current_user.id,
        details={"updated_keys": list(updates.keys())},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return {"message": "Credentials mis a jour avec succes", "updated": list(updates.keys())}
