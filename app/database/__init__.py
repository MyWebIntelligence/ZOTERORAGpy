"""
Database Package
================

This package manages the database layer of the RAGpy application.
It exposes the SQLAlchemy engine, session factory, and base model class.

Components:
- `Base`: The declarative base for all SQLAlchemy models.
- `engine`: The SQLAlchemy engine instance connected to the database.
- `SessionLocal`: The session factory for creating new database sessions.
- `get_db`: A dependency for FastAPI to provide a database session per request.
"""
from app.database.base import Base
from app.database.session import engine, SessionLocal, get_db

__all__ = ["Base", "engine", "SessionLocal", "get_db"]
