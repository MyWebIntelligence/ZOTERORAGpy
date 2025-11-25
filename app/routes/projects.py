"""
Project Management Routes
=========================

This module handles the CRUD operations for projects and their members. It enforces
access control, allowing users to manage their own projects and collaborate on others.

Key Features:
- Project CRUD: Create, read, update, and delete projects.
- Membership Management: Add and remove project members with specific roles.
- Access Control: Verify user permissions for project access and modification.
- Dashboard: List projects owned by or shared with the current user.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.database.session import get_db
from app.models.user import User
from app.models.project import Project, ProjectMember, ProjectRole
from app.models.audit import AuditAction, create_audit_log
from app.middleware.auth import get_current_active_user

router = APIRouter(prefix="/projects", tags=["Projects"])


# Schemas
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Mon projet de recherche",
                "description": "Analyse de corpus académique"
            }
        }


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    session_folder: Optional[str] = Field(None, max_length=255)
    vector_db_type: Optional[str] = Field(None, max_length=50)
    index_name: Optional[str] = Field(None, max_length=255)


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    owner_id: int
    owner_name: Optional[str] = None
    session_folder: Optional[str]
    vector_db_type: Optional[str]
    index_name: Optional[str]
    is_archived: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    user_role: Optional[str] = None

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    owned: List[ProjectResponse]
    collaborations: List[ProjectResponse]
    favorites: List[ProjectResponse]


class MemberCreate(BaseModel):
    email: str
    role: str = ProjectRole.VIEWER.value


class MemberResponse(BaseModel):
    id: int
    user_id: int
    email: str
    full_name: str
    role: str
    accepted_at: Optional[datetime]

    class Config:
        from_attributes = True


def get_client_ip(request: Request) -> str:
    """Extrait l'adresse IP du client"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("", response_model=ProjectListResponse)
async def list_my_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Liste les projets de l'utilisateur connecté.

    Retourne:
    - owned: Projets dont l'utilisateur est propriétaire
    - collaborations: Projets où l'utilisateur est collaborateur
    - favorites: Projets favoris (à implémenter)
    """
    # Projets possédés
    owned_projects = db.query(Project).filter(
        Project.owner_id == current_user.id,
        Project.is_archived == False
    ).order_by(Project.updated_at.desc()).all()

    # Projets en collaboration
    collaboration_memberships = db.query(ProjectMember).filter(
        ProjectMember.user_id == current_user.id
    ).all()

    collaboration_project_ids = [m.project_id for m in collaboration_memberships]
    collaboration_projects = db.query(Project).filter(
        Project.id.in_(collaboration_project_ids),
        Project.is_archived == False
    ).all() if collaboration_project_ids else []

    # Convertir en réponse
    def to_response(project: Project, role: str = None) -> ProjectResponse:
        owner = db.query(User).filter(User.id == project.owner_id).first()
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            owner_id=project.owner_id,
            owner_name=owner.full_name if owner else None,
            session_folder=project.session_folder,
            vector_db_type=project.vector_db_type,
            index_name=project.index_name,
            is_archived=project.is_archived,
            created_at=project.created_at,
            updated_at=project.updated_at,
            user_role=role or ProjectRole.OWNER.value
        )

    owned_responses = [to_response(p, ProjectRole.OWNER.value) for p in owned_projects]

    collab_responses = []
    for p in collaboration_projects:
        membership = next((m for m in collaboration_memberships if m.project_id == p.id), None)
        role = membership.role if membership else ProjectRole.VIEWER.value
        collab_responses.append(to_response(p, role))

    return ProjectListResponse(
        owned=owned_responses,
        collaborations=collab_responses,
        favorites=[]  # À implémenter
    )


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Crée un nouveau projet.
    """
    project = Project(
        name=project_data.name,
        description=project_data.description,
        owner_id=current_user.id
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PROJECT_CREATE,
        user_id=current_user.id,
        resource_type="project",
        resource_id=project.id,
        details={"name": project.name},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        owner_id=project.owner_id,
        owner_name=current_user.full_name,
        session_folder=project.session_folder,
        vector_db_type=project.vector_db_type,
        index_name=project.index_name,
        is_archived=project.is_archived,
        created_at=project.created_at,
        updated_at=project.updated_at,
        user_role=ProjectRole.OWNER.value
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retourne les détails d'un projet.
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Vérifier l'accès
    user_role = project.get_user_role(current_user.id)
    if not user_role and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès non autorisé à ce projet"
        )

    owner = db.query(User).filter(User.id == project.owner_id).first()

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        owner_id=project.owner_id,
        owner_name=owner.full_name if owner else None,
        session_folder=project.session_folder,
        vector_db_type=project.vector_db_type,
        index_name=project.index_name,
        is_archived=project.is_archived,
        created_at=project.created_at,
        updated_at=project.updated_at,
        user_role=user_role or "admin"
    )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Met à jour un projet.
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Vérifier les droits d'édition
    if not project.can_edit(current_user.id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas les droits pour modifier ce projet"
        )

    # Mettre à jour
    if project_data.name is not None:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    if project_data.session_folder is not None:
        project.session_folder = project_data.session_folder
    if project_data.vector_db_type is not None:
        project.vector_db_type = project_data.vector_db_type
    if project_data.index_name is not None:
        project.index_name = project_data.index_name

    db.commit()
    db.refresh(project)

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PROJECT_UPDATE,
        user_id=current_user.id,
        resource_type="project",
        resource_id=project.id,
        details=project_data.model_dump(exclude_unset=True),
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    owner = db.query(User).filter(User.id == project.owner_id).first()

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        owner_id=project.owner_id,
        owner_name=owner.full_name if owner else None,
        session_folder=project.session_folder,
        vector_db_type=project.vector_db_type,
        index_name=project.index_name,
        is_archived=project.is_archived,
        created_at=project.created_at,
        updated_at=project.updated_at,
        user_role=project.get_user_role(current_user.id)
    )


@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime un projet.
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Seul le propriétaire ou un admin peut supprimer
    if project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul le propriétaire peut supprimer ce projet"
        )

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PROJECT_DELETE,
        user_id=current_user.id,
        resource_type="project",
        resource_id=project.id,
        details={"name": project.name},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    db.delete(project)
    db.commit()

    return {"message": "Projet supprimé avec succès"}


@router.get("/{project_id}/members", response_model=List[MemberResponse])
async def list_project_members(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Liste les membres d'un projet.
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Vérifier l'accès
    if not project.has_access(current_user.id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès non autorisé"
        )

    members = []

    # Ajouter le propriétaire
    owner = db.query(User).filter(User.id == project.owner_id).first()
    if owner:
        members.append(MemberResponse(
            id=0,
            user_id=owner.id,
            email=owner.email,
            full_name=owner.full_name,
            role=ProjectRole.OWNER.value,
            accepted_at=project.created_at
        ))

    # Ajouter les membres
    for pm in project.members:
        user = db.query(User).filter(User.id == pm.user_id).first()
        if user:
            members.append(MemberResponse(
                id=pm.id,
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=pm.role,
                accepted_at=pm.accepted_at
            ))

    return members


@router.post("/{project_id}/members", response_model=MemberResponse)
async def add_project_member(
    project_id: int,
    member_data: MemberCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Ajoute un membre à un projet.
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Seul le propriétaire peut ajouter des membres
    if project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul le propriétaire peut ajouter des membres"
        )

    # Trouver l'utilisateur à ajouter
    user_to_add = db.query(User).filter(User.email == member_data.email.lower()).first()

    if not user_to_add:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé avec cet email"
        )

    # Vérifier qu'il n'est pas déjà membre
    if user_to_add.id == project.owner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'utilisateur est déjà propriétaire du projet"
        )

    existing = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == user_to_add.id
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'utilisateur est déjà membre du projet"
        )

    # Ajouter le membre
    member = ProjectMember(
        project_id=project_id,
        user_id=user_to_add.id,
        role=member_data.role,
        invited_by=current_user.id,
        accepted_at=datetime.utcnow()  # Auto-accepté pour simplifier
    )

    db.add(member)
    db.commit()
    db.refresh(member)

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PROJECT_MEMBER_ADD,
        user_id=current_user.id,
        resource_type="project",
        resource_id=project.id,
        details={"added_user_id": user_to_add.id, "role": member_data.role},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    return MemberResponse(
        id=member.id,
        user_id=user_to_add.id,
        email=user_to_add.email,
        full_name=user_to_add.full_name,
        role=member.role,
        accepted_at=member.accepted_at
    )


@router.delete("/{project_id}/members/{user_id}")
async def remove_project_member(
    project_id: int,
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retire un membre d'un projet.
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Seul le propriétaire peut retirer des membres (ou le membre lui-même)
    if project.owner_id != current_user.id and current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas les droits pour retirer ce membre"
        )

    member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == user_id
    ).first()

    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Membre non trouvé"
        )

    # Log d'audit
    create_audit_log(
        db=db,
        action=AuditAction.PROJECT_MEMBER_REMOVE,
        user_id=current_user.id,
        resource_type="project",
        resource_id=project.id,
        details={"removed_user_id": user_id},
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("User-Agent")
    )

    db.delete(member)
    db.commit()

    return {"message": "Membre retiré avec succès"}
