"""
User model for authentication and authorization
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """
    Modèle utilisateur pour l'authentification et l'autorisation.

    Le premier utilisateur créé devient automatiquement administrateur.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    # Informations personnelles
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    organization = Column(String(255), nullable=True)
    title = Column(String(255), nullable=True)  # Titre/fonction

    # Rôles et permissions (stockés en JSON: ["USER", "ADMIN"])
    roles = Column(JSON, default=["USER"], nullable=False)

    # Statuts
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Sécurité
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)

    # Token de vérification/reset
    verification_token = Column(String(255), nullable=True)
    reset_token = Column(String(255), nullable=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)

    # API Credentials (encrypted JSON)
    # Stores user's personal API keys for various services
    api_credentials = Column(Text, nullable=True)

    # Relations
    owned_projects = relationship(
        "Project",
        back_populates="owner",
        cascade="all, delete-orphan"
    )
    project_memberships = relationship(
        "ProjectMember",
        back_populates="user",
        cascade="all, delete-orphan",
        primaryjoin="User.id == ProjectMember.user_id"
    )
    audit_logs = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.email}>"

    @property
    def full_name(self) -> str:
        """Retourne le nom complet de l'utilisateur"""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or self.email

    @property
    def is_admin(self) -> bool:
        """Vérifie si l'utilisateur a le rôle ADMIN"""
        return "ADMIN" in (self.roles or [])

    @property
    def is_locked(self) -> bool:
        """Vérifie si le compte est temporairement verrouillé"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until.replace(tzinfo=None)

    def has_role(self, role: str) -> bool:
        """Vérifie si l'utilisateur a un rôle spécifique"""
        return role in (self.roles or [])

    def add_role(self, role: str) -> None:
        """Ajoute un rôle à l'utilisateur"""
        if self.roles is None:
            self.roles = []
        if role not in self.roles:
            self.roles = self.roles + [role]

    def remove_role(self, role: str) -> None:
        """Retire un rôle de l'utilisateur"""
        if self.roles and role in self.roles:
            self.roles = [r for r in self.roles if r != role]

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convertit l'utilisateur en dictionnaire.

        Args:
            include_sensitive: Si True, inclut les champs sensibles
        """
        data = {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "organization": self.organization,
            "title": self.title,
            "roles": self.roles or [],
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

        if include_sensitive:
            data.update({
                "failed_login_attempts": self.failed_login_attempts,
                "is_locked": self.is_locked,
                "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            })

        return data
