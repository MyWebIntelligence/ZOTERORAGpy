"""
User profile routes
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.user import User
from app.models.audit import AuditAction, create_audit_log
from app.schemas.user import UserResponse, UserUpdate
from app.middleware.auth import get_current_active_user

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
        admin_count = db.query(User).filter(
            User.roles.contains(["ADMIN"]),
            User.id != current_user.id,
            User.is_active == True
        ).count()

        if admin_count == 0:
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
