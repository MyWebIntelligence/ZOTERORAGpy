"""
Background Task Model
=====================

This module defines the `BackgroundTask` model for tracking long-running
tasks that persist even when the client disconnects. It enables users to
resume monitoring task progress after closing and reopening the browser.

Key Features:
- Stores task status, progress, and logs in the database.
- Tracks the subprocess PID for cancellation support.
- Links tasks to users and optionally to projects.
- Supports multiple task types (extraction, chunking, embedding, etc.).
"""
import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text
)
from sqlalchemy.orm import relationship

from app.database.base import Base, TimestampMixin


class TaskStatus(str, enum.Enum):
    """Status values for background tasks."""
    PENDING = "pending"       # Task created, not yet started
    RUNNING = "running"       # Task is currently executing
    COMPLETED = "completed"   # Task finished successfully
    FAILED = "failed"         # Task failed with an error
    CANCELLED = "cancelled"   # Task was cancelled by user


class TaskType(str, enum.Enum):
    """Types of background tasks supported by the system."""
    PROCESS_DATAFRAME = "process_dataframe"      # PDF extraction + OCR
    INITIAL_CHUNKING = "initial_chunking"        # Text chunking + GPT recodage
    DENSE_EMBEDDING = "dense_embedding"          # OpenAI embeddings
    SPARSE_EMBEDDING = "sparse_embedding"        # spaCy sparse features
    CLUSTERING = "clustering"                    # UMAP + HDBSCAN clustering
    ZOTERO_NOTES = "zotero_notes"               # Zotero note generation
    FILTER_CITATIONS = "filter_citations"        # LLM citation filtering
    IMPORT_CITATIONS = "import_citations"        # Zotero import + PDF download


class BackgroundTask(Base, TimestampMixin):
    """
    Model for tracking long-running background tasks.

    This model persists task state to the database, allowing tasks to
    continue running even if the client disconnects. Users can resume
    monitoring task progress at any time.

    Attributes:
        task_type: The type of task being executed.
        status: Current status of the task.
        user_id: ID of the user who initiated the task.
        session_folder: Path to the session folder containing task files.
        project_id: Optional ID of the associated project.
        progress_current: Current progress count.
        progress_total: Total items to process.
        progress_percent: Calculated progress percentage (0-100).
        progress_message: Human-readable progress message.
        logs: Last N lines of task output (for debugging).
        started_at: Timestamp when task started executing.
        completed_at: Timestamp when task finished.
        error_message: Error description if task failed.
        pid: Process ID of the subprocess (for cancellation).
        result_file: Path to the output file when completed.
    """
    __tablename__ = "background_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(Enum(TaskType), nullable=False, index=True)
    status = Column(
        Enum(TaskStatus),
        default=TaskStatus.PENDING,
        nullable=False,
        index=True
    )

    # User and session association
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    session_folder = Column(String(500), nullable=False)
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Progress tracking
    progress_current = Column(Integer, default=0, nullable=False)
    progress_total = Column(Integer, default=0, nullable=False)
    progress_percent = Column(Float, default=0.0, nullable=False)
    progress_message = Column(String(500), default="", nullable=False)

    # Task output logs (last 50 lines)
    logs = Column(Text, default="", nullable=False)

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)

    # Process management
    pid = Column(Integer, nullable=True)

    # Result
    result_file = Column(String(500), nullable=True)

    # Relationships
    user = relationship("User", back_populates="background_tasks")
    project = relationship("Project", back_populates="background_tasks")

    def __repr__(self) -> str:
        return f"<BackgroundTask {self.id} ({self.task_type.value}: {self.status.value})>"

    def to_dict(self) -> dict:
        """
        Convert the task to a dictionary for API responses.

        Returns:
            Dictionary containing all task information.
        """
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "user_id": self.user_id,
            "session_folder": self.session_folder,
            "project_id": self.project_id,
            "progress": {
                "current": self.progress_current,
                "total": self.progress_total,
                "percent": self.progress_percent,
                "message": self.progress_message
            },
            "logs": self.logs.split('\n') if self.logs else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "result_file": self.result_file
        }

    @property
    def is_active(self) -> bool:
        """Check if the task is still active (pending or running)."""
        return self.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

    @property
    def is_finished(self) -> bool:
        """Check if the task has finished (completed, failed, or cancelled)."""
        return self.status in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """
        Calculate the task duration in seconds.

        Returns:
            Duration in seconds, or None if task hasn't started.
        """
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
