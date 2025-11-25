"""
Pipeline Session Model
======================

This module defines the `PipelineSession` model, which tracks the state and progress
of document processing pipelines. Each session represents a distinct execution run
associated with a project.

Key Components:
- `SessionStatus`: Enumeration of all possible pipeline states (e.g., EXTRACTING, CHUNKING).
- `PipelineSession`: The SQLAlchemy model tracking status, counts, and errors.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship

from app.database.base import Base


class SessionStatus(str, Enum):
    """Pipeline session status"""
    CREATED = "created"           # Just created, upload done
    EXTRACTING = "extracting"     # Running OCR/extraction
    EXTRACTED = "extracted"       # CSV generated
    CHUNKING = "chunking"         # Running chunking
    CHUNKED = "chunked"           # Chunks generated
    EMBEDDING = "embedding"       # Generating embeddings
    EMBEDDED = "embedded"         # Embeddings done
    UPLOADING = "uploading"       # Uploading to vector DB
    COMPLETED = "completed"       # Fully processed
    ERROR = "error"               # Error occurred


class PipelineSession(Base):
    """
    Tracks pipeline sessions (processing runs) for a project.
    Each session corresponds to a directory in uploads/.
    """
    __tablename__ = "pipeline_sessions"

    id = Column(Integer, primary_key=True, index=True)

    # Link to project (required)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    # Session identification
    session_folder = Column(String(255), unique=True, nullable=False, index=True)
    original_filename = Column(String(255), nullable=True)

    # Status tracking
    status = Column(SQLEnum(SessionStatus), default=SessionStatus.CREATED, nullable=False)

    # Processing info
    source_type = Column(String(50), nullable=True)  # 'zip', 'csv'
    row_count = Column(Integer, nullable=True)       # Number of items processed
    chunk_count = Column(Integer, nullable=True)     # Number of chunks generated

    # Vector DB info (when uploaded)
    vector_db_type = Column(String(50), nullable=True)   # 'pinecone', 'weaviate', 'qdrant'
    index_name = Column(String(255), nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="sessions")

    def __repr__(self):
        return f"<PipelineSession(id={self.id}, project_id={self.project_id}, folder={self.session_folder}, status={self.status})>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "session_folder": self.session_folder,
            "original_filename": self.original_filename,
            "status": self.status.value if self.status else None,
            "source_type": self.source_type,
            "row_count": self.row_count,
            "chunk_count": self.chunk_count,
            "vector_db_type": self.vector_db_type,
            "index_name": self.index_name,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
