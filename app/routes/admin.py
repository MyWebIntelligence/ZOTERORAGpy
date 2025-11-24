"""
Administration routes for user and system management
"""
import os
import shutil
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_
from pydantic import BaseModel

from app.database.session import get_db
from app.models.user import User
from app.models.project import Project
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.audit import AuditAction, create_audit_log
from app.schemas.user import (
    UserResponse,
    UserAdminUpdate,
    UserListResponse,
    AdminStats
)
from app.core.security import get_password_hash, generate_reset_token
from app.middleware.auth import require_admin
from app.services.email_service import email_service
from app.config import settings

# Directory configuration
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAGPY_DIR = os.path.dirname(APP_DIR)
UPLOAD_DIR = os.path.join(RAGPY_DIR, "uploads")

router = APIRouter(prefix="/api/admin", tags=["Administration"])


def get_client_ip(request: Request) -> str:
    """Extrait l'adresse IP du client"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Retourne les statistiques du tableau de bord administrateur.
    """
    # Compter les utilisateurs
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    verified_users = db.query(User).filter(User.is_verified == True).count()

    # Compter les admins (JSON contains)
    admin_users = db.query(User).filter(
        User.roles.contains(["ADMIN"])
    ).count()

    # Compter les projets
    total_projects = db.query(Project).count()

    # Connexions récentes (7 derniers jours)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_logins = db.query(User).filter(
        User.last_login >= week_ago
    ).count()

    return AdminStats(
        total_users=total_users,
        active_users=active_users,
        admin_users=admin_users,
        verified_users=verified_users,
        total_projects=total_projects,
        recent_logins=recent_logins
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, max_length=100),
    is_active: Optional[bool] = None,
    is_admin: Optional[bool] = None,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Liste tous les utilisateurs avec pagination et filtres.
    """
    query = db.query(User)

    # Filtres
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                User.email.ilike(search_term),
                User.first_name.ilike(search_term),
                User.last_name.ilike(search_term),
                User.organization.ilike(search_term)
            )
        )

    if is_active is not None:
        query = query.filter(User.is_active == is_active)

    if is_admin is not None:
        if is_admin:
            query = query.filter(User.roles.contains(["ADMIN"]))
        else:
            # Pas admin = ne contient pas ADMIN (approximation)
            query = query.filter(~User.roles.contains(["ADMIN"]))

    # Compter le total
    total = query.count()

    # Pagination
    offset = (page - 1) * per_page
    users = query.order_by(User.created_at.desc()).offset(offset).limit(per_page).all()

    # Calculer le nombre de pages
    pages = (total + per_page - 1) // per_page

    # Convertir en réponse
    user_responses = [
        UserResponse(
            id=u.id,
            email=u.email,
            first_name=u.first_name,
            last_name=u.last_name,
            full_name=u.full_name,
            organization=u.organization,
            title=u.title,
            roles=u.roles or [],
            is_active=u.is_active,
            is_verified=u.is_verified,
            is_admin=u.is_admin,
            created_at=u.created_at,
            last_login=u.last_login
        )
        for u in users
    ]

    return UserListResponse(
        users=user_responses,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Retourne les détails d'un utilisateur.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=user.full_name,
        organization=user.organization,
        title=user.title,
        roles=user.roles or [],
        is_active=user.is_active,
        is_verified=user.is_verified,
        is_admin=user.is_admin,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserAdminUpdate,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Met à jour un utilisateur (admin).
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    # Empêcher de se retirer son propre rôle admin
    if user.id == admin.id and user_data.roles is not None:
        if "ADMIN" not in user_data.roles and user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vous ne pouvez pas vous retirer le rôle administrateur"
            )

    # Mettre à jour les champs
    if user_data.first_name is not None:
        user.first_name = user_data.first_name
    if user_data.last_name is not None:
        user.last_name = user_data.last_name
    if user_data.organization is not None:
        user.organization = user_data.organization
    if user_data.title is not None:
        user.title = user_data.title
    if user_data.roles is not None:
        user.roles = user_data.roles
    if user_data.is_active is not None:
        user.is_active = user_data.is_active
    if user_data.is_verified is not None:
        user.is_verified = user_data.is_verified

    db.commit()
    db.refresh(user)

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.USER_UPDATE,
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        details=user_data.model_dump(exclude_unset=True),
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=user.full_name,
        organization=user.organization,
        title=user.title,
        roles=user.roles or [],
        is_active=user.is_active,
        is_verified=user.is_verified,
        is_admin=user.is_admin,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Supprime un utilisateur.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    # Empêcher de se supprimer soi-même
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas supprimer votre propre compte via l'admin"
        )

    # Vérifier si c'est le dernier admin
    if user.is_admin:
        # Note: On utilise is_admin property car JSON contains peut etre peu fiable sur SQLite
        other_users = db.query(User).filter(
            User.id != user.id,
            User.is_active == True
        ).all()

        other_admins = [u for u in other_users if u.is_admin]

        if len(other_admins) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Impossible de supprimer le dernier administrateur"
            )

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.USER_DELETE,
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        details={"email": user.email, "deleted_by": admin.email},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    # Supprimer l'utilisateur
    db.delete(user)
    db.commit()

    return {"message": "Utilisateur supprimé avec succès"}


@router.post("/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Active/désactive un utilisateur.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    # Empêcher de se désactiver soi-même
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas vous désactiver vous-même"
        )

    # Toggle le statut
    user.is_active = not user.is_active

    # Réinitialiser le verrouillage si on réactive
    if user.is_active:
        user.failed_login_attempts = 0
        user.locked_until = None

    db.commit()

    # Log d'audit
    action = AuditAction.USER_UNBLOCK if user.is_active else AuditAction.USER_BLOCK
    create_audit_log(
        db=db,
        action=action,
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        details={"new_status": user.is_active},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    status_text = "activé" if user.is_active else "désactivé"
    return {"message": f"Utilisateur {status_text} avec succès", "is_active": user.is_active}


@router.post("/users/{user_id}/toggle-admin")
async def toggle_user_admin(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Ajoute/retire le rôle administrateur à un utilisateur.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    # Empêcher de se retirer son propre rôle admin
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas modifier votre propre rôle administrateur"
        )

    # Toggle le rôle admin
    if user.is_admin:
        # Vérifier qu'il reste au moins un autre admin actif
        # Note: On utilise is_admin property car JSON contains peut etre peu fiable sur SQLite
        other_users = db.query(User).filter(
            User.id != user.id,
            User.is_active == True
        ).all()

        other_admins = [u for u in other_users if u.is_admin]

        if len(other_admins) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Impossible de retirer le dernier administrateur"
            )

        user.remove_role("ADMIN")
    else:
        user.add_role("ADMIN")

    db.commit()

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.USER_ROLE_CHANGE,
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        details={"is_admin": user.is_admin, "roles": user.roles},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    status_text = "promu administrateur" if user.is_admin else "rétrogradé utilisateur"
    return {"message": f"Utilisateur {status_text}", "is_admin": user.is_admin}


@router.post("/users/{user_id}/reset-password")
async def admin_reset_password(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Force la réinitialisation du mot de passe d'un utilisateur.

    Génère un token de reset et retourne l'URL (en dev) ou envoie un email (en prod).
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    # Générer un token de reset (expire en 1 heure)
    reset_token = generate_reset_token()
    user.reset_token = reset_token
    user.reset_token_expires = datetime.utcnow() + timedelta(
        hours=settings.PASSWORD_RESET_EXPIRE_HOURS
    )
    db.commit()

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PASSWORD_RESET_REQUEST,
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        details={"triggered_by_admin": True, "admin_email": admin.email},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    # Envoyer l'email de reset (initié par admin)
    email_sent = await email_service.send_password_reset_email(
        to_email=user.email,
        first_name=user.first_name or "",
        reset_token=reset_token,
        initiated_by_admin=True
    )

    # Log en dev si email pas configuré
    if settings.DEBUG and not email_service.is_available:
        print(f"Admin-triggered password reset token for {user.email}: {reset_token}")
        return {
            "message": "Token de réinitialisation généré (email non configuré)",
            "reset_token": reset_token,
            "expires_in_hours": settings.PASSWORD_RESET_EXPIRE_HOURS
        }

    return {
        "message": f"Email de réinitialisation envoyé à {user.email}",
        "email_sent": email_sent,
        "expires_in_hours": settings.PASSWORD_RESET_EXPIRE_HOURS
    }


@router.post("/users/{user_id}/verify")
async def admin_verify_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Vérifie manuellement l'email d'un utilisateur.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    if user.is_verified:
        return {"message": "L'utilisateur est déjà vérifié"}

    user.is_verified = True
    user.verification_token = None
    db.commit()

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.USER_UPDATE,
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        details={"action": "manual_verification"},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return {"message": "Utilisateur vérifié avec succès"}


# --- Pipeline Sessions Admin ---

class AdminSessionResponse(BaseModel):
    """Session response with user and project info for admin"""
    id: int
    session_folder: str
    original_filename: Optional[str]
    status: str
    source_type: Optional[str]
    row_count: Optional[int]
    chunk_count: Optional[int]
    vector_db_type: Optional[str]
    index_name: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    # Project info
    project_id: int
    project_name: str
    # User info
    owner_id: int
    owner_email: str
    owner_name: Optional[str]

    class Config:
        from_attributes = True


class AdminSessionListResponse(BaseModel):
    sessions: List[AdminSessionResponse]
    total: int
    page: int
    per_page: int
    pages: int


@router.get("/sessions", response_model=AdminSessionListResponse)
async def list_all_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Liste toutes les sessions de pipeline (admin seulement).
    Inclut les informations du projet et de l'utilisateur.
    """
    query = db.query(PipelineSession).join(
        Project, PipelineSession.project_id == Project.id
    ).join(
        User, Project.owner_id == User.id
    )

    # Apply status filter
    if status_filter:
        try:
            status_enum = SessionStatus(status_filter)
            query = query.filter(PipelineSession.status == status_enum)
        except ValueError:
            pass  # Invalid status, ignore filter

    # Count total
    total = query.count()

    # Pagination
    offset = (page - 1) * per_page
    sessions = query.order_by(PipelineSession.created_at.desc()).offset(offset).limit(per_page).all()

    # Build response with project and user info
    session_responses = []
    for session in sessions:
        project = db.query(Project).filter(Project.id == session.project_id).first()
        owner = db.query(User).filter(User.id == project.owner_id).first() if project else None

        session_responses.append(AdminSessionResponse(
            id=session.id,
            session_folder=session.session_folder,
            original_filename=session.original_filename,
            status=session.status.value if session.status else "unknown",
            source_type=session.source_type,
            row_count=session.row_count,
            chunk_count=session.chunk_count,
            vector_db_type=session.vector_db_type,
            index_name=session.index_name,
            created_at=session.created_at,
            updated_at=session.updated_at,
            project_id=session.project_id,
            project_name=project.name if project else "Projet supprime",
            owner_id=owner.id if owner else 0,
            owner_email=owner.email if owner else "Inconnu",
            owner_name=owner.full_name if owner else None
        ))

    pages = (total + per_page - 1) // per_page

    return AdminSessionListResponse(
        sessions=session_responses,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages
    )


@router.delete("/sessions/{session_id}")
async def admin_delete_session(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """
    Supprime une session de pipeline (admin seulement).
    Supprime egalement les fichiers associes.
    """
    session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvee"
        )

    # Get project for audit
    project = db.query(Project).filter(Project.id == session.project_id).first()

    # Delete files
    session_path = os.path.join(UPLOAD_DIR, session.session_folder)
    if os.path.exists(session_path):
        shutil.rmtree(session_path)

    # Update project if this was the active session
    if project and project.session_folder == session.session_folder:
        project.session_folder = None

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PROJECT_DELETE,
        user_id=admin.id,
        resource_type="pipeline_session",
        resource_id=session.id,
        details={
            "session_folder": session.session_folder,
            "project_id": session.project_id,
            "project_name": project.name if project else "Unknown"
        },
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    db.delete(session)
    db.commit()

    return JSONResponse({"message": "Session supprimee avec succes"})
