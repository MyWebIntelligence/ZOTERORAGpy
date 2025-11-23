"""
Project model for organizing user work
"""
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.base import Base, TimestampMixin

import enum


class ProjectRole(str, enum.Enum):
    """Rôles possibles dans un projet"""
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"


class Project(Base, TimestampMixin):
    """
    Modèle de projet pour organiser le travail des utilisateurs.

    Chaque projet correspond à une session de traitement et possède
    son propre dossier dans uploads/.
    """
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Propriétaire du projet
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Dossier de session (ex: "uuid_MaBiblio")
    session_folder = Column(String(255), nullable=True)

    # Configuration du projet
    vector_db_type = Column(String(50), nullable=True)  # pinecone, weaviate, qdrant
    index_name = Column(String(255), nullable=True)

    # Statut
    is_archived = Column(Boolean, default=False, nullable=False)

    # Relations
    owner = relationship("User", back_populates="owned_projects")
    members = relationship(
        "ProjectMember",
        back_populates="project",
        cascade="all, delete-orphan"
    )
    sessions = relationship(
        "PipelineSession",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="desc(PipelineSession.created_at)"
    )

    def __repr__(self):
        return f"<Project {self.name}>"

    @property
    def is_owner(self, user_id: int) -> bool:
        """Vérifie si un utilisateur est le propriétaire"""
        return self.owner_id == user_id

    def get_user_role(self, user_id: int) -> Optional[str]:
        """Retourne le rôle d'un utilisateur dans le projet"""
        if self.owner_id == user_id:
            return ProjectRole.OWNER.value

        for member in self.members:
            if member.user_id == user_id:
                return member.role

        return None

    def has_access(self, user_id: int) -> bool:
        """Vérifie si un utilisateur a accès au projet"""
        return self.get_user_role(user_id) is not None

    def can_edit(self, user_id: int) -> bool:
        """Vérifie si un utilisateur peut modifier le projet"""
        role = self.get_user_role(user_id)
        return role in [ProjectRole.OWNER.value, ProjectRole.COLLABORATOR.value]

    def to_dict(self) -> dict:
        """Convertit le projet en dictionnaire"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "session_folder": self.session_folder,
            "vector_db_type": self.vector_db_type,
            "index_name": self.index_name,
            "is_archived": self.is_archived,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ProjectMember(Base, TimestampMixin):
    """
    Association entre utilisateurs et projets pour les collaborations.
    """
    __tablename__ = "project_members"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(50), default=ProjectRole.VIEWER.value, nullable=False)

    # Invitation
    invited_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    accepted_at = Column(DateTime(timezone=True), nullable=True)

    # Relations
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships", foreign_keys=[user_id])

    def __repr__(self):
        return f"<ProjectMember project={self.project_id} user={self.user_id} role={self.role}>"

    def to_dict(self) -> dict:
        """Convertit l'association en dictionnaire"""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "role": self.role,
            "invited_by": self.invited_by,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
