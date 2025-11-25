"""
Database Initialization Module
==============================

This module is responsible for initializing the database schema and performing
necessary migrations on application startup.

It handles:
- Creating all tables defined in SQLAlchemy models.
- Running manual migrations for schema changes (e.g., adding columns).
- Verifying database connectivity.
"""
import logging
from sqlalchemy import inspect, text
from app.database.base import Base
from app.database.session import engine

logger = logging.getLogger(__name__)


def run_migrations():
    """
    Executes manual database migrations for columns added after initial creation.

    This function is necessary because SQLAlchemy's `create_all` does not
    modify existing tables. It inspects the database and applies missing
    columns, such as adding the `api_credentials` column to the `users` table.
    """
    inspector = inspect(engine)

    # Migration: Ajouter api_credentials a la table users
    if "users" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("users")]
        if "api_credentials" not in columns:
            logger.info("Migration: Adding api_credentials column to users table")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN api_credentials TEXT"))
                conn.commit()
            logger.info("Migration completed: api_credentials column added")


def init_database():
    """
    Initializes the database by creating all necessary tables.

    This function should be called on application startup. It imports all
    SQLAlchemy models to ensure they are registered with the `Base.metadata`
    and then creates all tables. It also triggers `run_migrations` to handle
    any manual schema adjustments.
    """
    # Import des modèles pour qu'ils soient enregistrés dans Base.metadata
    from app.models import user, project, audit, pipeline_session  # noqa: F401

    logger.info("Initializing database...")

    # Créer toutes les tables
    Base.metadata.create_all(bind=engine)

    # Executer les migrations pour les colonnes ajoutees
    run_migrations()

    logger.info("Database initialized successfully")


def check_database_connection() -> bool:
    """
    Verifies that the database connection is functional.

    Returns:
        True if the connection is successful, False otherwise.
    """
    try:
        inspector = inspect(engine)
        inspector.get_table_names()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_table_names() -> list:
    """
    Retrieves a list of all existing table names in the database.

    Returns:
        A list of table names as strings.
    """
    inspector = inspect(engine)
    return inspector.get_table_names()
