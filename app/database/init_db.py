"""
Database initialization and migration utilities
"""
import logging
from sqlalchemy import inspect, text
from app.database.base import Base
from app.database.session import engine

logger = logging.getLogger(__name__)


def run_migrations():
    """
    Execute des migrations manuelles pour les colonnes ajoutees apres la creation initiale.
    SQLAlchemy create_all ne modifie pas les tables existantes.
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
    Initialise la base de données en créant toutes les tables.
    À appeler au démarrage de l'application.
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
    """Vérifie que la connexion à la base de données fonctionne"""
    try:
        inspector = inspect(engine)
        inspector.get_table_names()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_table_names() -> list:
    """Retourne la liste des tables existantes"""
    inspector = inspect(engine)
    return inspector.get_table_names()
