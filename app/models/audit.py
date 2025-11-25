"""
Audit Log Model
===============

This module defines the `AuditLog` model and associated utilities for tracking
user actions within the application. It provides a comprehensive history of
who did what, when, and on which resource.

Key Components:
- `AuditAction`: Constants defining all trackable action types.
- `AuditLog`: The SQLAlchemy model for storing audit records.
- `create_audit_log`: Utility function to easily create new audit entries.
"""
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.base import Base


class AuditAction:
    """Constantes pour les types d'actions auditées"""
    # Authentication
    LOGIN = "LOGIN"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    REGISTER = "REGISTER"
    PASSWORD_RESET_REQUEST = "PASSWORD_RESET_REQUEST"
    PASSWORD_RESET = "PASSWORD_RESET"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"

    # User management
    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"
    USER_BLOCK = "USER_BLOCK"
    USER_UNBLOCK = "USER_UNBLOCK"
    USER_ROLE_CHANGE = "USER_ROLE_CHANGE"

    # Projects
    PROJECT_CREATE = "PROJECT_CREATE"
    PROJECT_UPDATE = "PROJECT_UPDATE"
    PROJECT_DELETE = "PROJECT_DELETE"
    PROJECT_MEMBER_ADD = "PROJECT_MEMBER_ADD"
    PROJECT_MEMBER_REMOVE = "PROJECT_MEMBER_REMOVE"

    # Pipeline operations
    UPLOAD_FILE = "UPLOAD_FILE"
    PROCESS_DATAFRAME = "PROCESS_DATAFRAME"
    GENERATE_CHUNKS = "GENERATE_CHUNKS"
    GENERATE_EMBEDDINGS = "GENERATE_EMBEDDINGS"
    UPLOAD_TO_VECTORDB = "UPLOAD_TO_VECTORDB"


class AuditLog(Base):
    """
    Journal d'audit pour tracer les actions des utilisateurs.

    Permet de suivre qui a fait quoi, quand, et sur quelle ressource.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Utilisateur qui a effectué l'action (nullable pour les actions système)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Type d'action (voir AuditAction)
    action = Column(String(100), nullable=False, index=True)

    # Ressource concernée
    resource_type = Column(String(50), nullable=True)  # user, project, file
    resource_id = Column(Integer, nullable=True)

    # Détails supplémentaires (JSON)
    details = Column(JSON, nullable=True)

    # Informations de requête
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    user_agent = Column(Text, nullable=True)

    # Résultat de l'action
    success = Column(Integer, default=1, nullable=False)  # 1 = success, 0 = failure
    error_message = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )

    # Relations
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} by user={self.user_id}>"

    def to_dict(self) -> dict:
        """Convertit le log en dictionnaire"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "success": bool(self.success),
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def create_audit_log(
    db,
    action: str,
    user_id: int = None,
    resource_type: str = None,
    resource_id: int = None,
    details: dict = None,
    ip_address: str = None,
    user_agent: str = None,
    success: bool = True,
    error_message: str = None
) -> AuditLog:
    """
    Crée une entrée dans le journal d'audit.

    Args:
        db: Session de base de données
        action: Type d'action (utiliser AuditAction)
        user_id: ID de l'utilisateur (optionnel)
        resource_type: Type de ressource concernée
        resource_id: ID de la ressource concernée
        details: Détails supplémentaires (dict)
        ip_address: Adresse IP du client
        user_agent: User-Agent du navigateur
        success: Succès de l'action
        error_message: Message d'erreur si échec

    Returns:
        AuditLog créé
    """
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        success=1 if success else 0,
        error_message=error_message
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
