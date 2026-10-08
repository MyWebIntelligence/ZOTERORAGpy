"""
Data Models Package
===================

This package contains the SQLAlchemy data models for the RAGpy application.
It exports the key models for easy access.

Models:
- `User`: User authentication and profile data.
- `Project`: Project management and organization.
- `ProjectMember`: Many-to-many relationship for project collaboration.
- `AuditLog`: System audit logging for security and tracking.
- `BackgroundTask`: Long-running background task tracking.
- `PipelineSession`, `SessionOwner`: Pipeline sessions and the uploader of sessions outside any project.
- `AlbertCorpus`, `AlbertCorpusSource`: Albert collections registered in RAGpy and their local document catalogue.
"""
from app.models.user import User
from app.models.project import Project, ProjectMember
from app.models.audit import AuditLog
from app.models.background_task import BackgroundTask, TaskStatus, TaskType
from app.models.pipeline_session import PipelineSession, SessionOwner
from app.models.albert_corpus import AlbertCorpus, AlbertCorpusSource
from app.models.user_setting import UserSetting

__all__ = [
    "User",
    "Project",
    "ProjectMember",
    "AuditLog",
    "BackgroundTask",
    "TaskStatus",
    "TaskType",
    "PipelineSession",
    "SessionOwner",
    "AlbertCorpus",
    "AlbertCorpusSource",
]
