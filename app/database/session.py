"""
Database Session Management
===========================

This module configures the SQLAlchemy engine and session factory.
It provides the `get_db` dependency for FastAPI to manage database transactions
within the request lifecycle.

Key Components:
- `engine`: The configured SQLAlchemy engine.
- `SessionLocal`: Factory for creating new database sessions.
- `get_db`: Generator function for dependency injection.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.config import settings

# Créer le moteur SQLAlchemy
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG,
    pool_pre_ping=True
)

# Factory de sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Provides a database session to a FastAPI dependency.

    This dependency injection function creates a new SQLAlchemy session for each
    request, yields it to the endpoint, and ensures that the session is
    properly closed after the request is finished.

    Usage in FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
