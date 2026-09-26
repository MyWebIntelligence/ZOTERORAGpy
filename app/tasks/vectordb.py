"""
Celery Task: Vector Database Upload
===================================

This module contains Celery tasks for uploading embeddings to vector databases.
It wraps the rad_vectordb.py script functionality for asynchronous execution
via Celery workers.

Task:
    upload_to_vectordb_task: Upload embeddings to Pinecone, Weaviate, or Qdrant

Features:
    - Support for multiple vector databases
    - Automatic retry on network failures
    - Batch processing for large datasets
    - Progress reporting via task state updates
"""
import os
import sys
import logging
from datetime import datetime
from celery import Task

from app.celery_app import celery_app

# Add scripts directory to path for imports
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '../../scripts')
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

logger = logging.getLogger(__name__)


class VectorDBTask(Task):
    """
    Base task class with automatic retry for vector DB tasks.

    Includes handling for network timeouts and API errors.
    """
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}
    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True


@celery_app.task(
    base=VectorDBTask,
    bind=True,
    name='vectordb.upload',
    queue='vectordb'
)
def upload_to_vectordb_task(
    self,
    input_file: str,
    session_id: int,
    db_choice: str,
    pinecone_index_name: str = None,
    pinecone_namespace: str = None,
    weaviate_class_name: str = None,
    weaviate_tenant_name: str = None,
    qdrant_collection_name: str = None
) -> dict:
    """
    Upload embeddings to a vector database.

    This task processes the output_chunks_with_embeddings_sparse.json file
    and uploads vectors to the specified database.

    Args:
        self: Celery task instance (bound)
        input_file: Path to embeddings JSON file
        session_id: Database session ID for status tracking
        db_choice: Target database ('pinecone', 'weaviate', 'qdrant')
        pinecone_index_name: Pinecone index name (if db_choice='pinecone')
        pinecone_namespace: Pinecone namespace (optional)
        weaviate_class_name: Weaviate class name (if db_choice='weaviate')
        weaviate_tenant_name: Weaviate tenant name (optional)
        qdrant_collection_name: Qdrant collection name (if db_choice='qdrant')

    Returns:
        dict: {
            "status": "success",
            "inserted_count": int,
            "database": str,
            "duration_seconds": float
        }

    Raises:
        Exception: Re-raised after max retries exhausted
    """
    start_time = datetime.utcnow()

    try:
        # Update session status
        _update_session_status(session_id, 'UPLOADING', vector_db=db_choice)

        # Report initial progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'percent': 0,
                'item': f'Connecting to {db_choice}...',
                'status': f'Starting upload to {db_choice}'
            }
        )

        logger.info(f"Starting vectordb upload task: {input_file} -> {db_choice}")

        # Progress callback
        def progress_callback(current: int, total: int, batch_info: str = ''):
            """Update Celery task state with progress."""
            percent = int((current / total) * 100) if total > 0 else 0
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'item': batch_info,
                    'status': f'Uploading batch {current}/{total}'
                }
            )

        # Execute upload based on database choice
        if db_choice == 'pinecone':
            result = _upload_to_pinecone(
                input_file,
                pinecone_index_name,
                pinecone_namespace,
                progress_callback
            )
        elif db_choice == 'weaviate':
            result = _upload_to_weaviate(
                input_file,
                weaviate_class_name,
                weaviate_tenant_name,
                progress_callback
            )
        elif db_choice == 'qdrant':
            result = _upload_to_qdrant(
                input_file,
                qdrant_collection_name,
                progress_callback
            )
        else:
            raise ValueError(f"Unknown database: {db_choice}")

        duration = (datetime.utcnow() - start_time).total_seconds()

        # Depuis le Lot 0.a les 3 connecteurs renvoient un dict homogène.
        # Repli défensif si un connecteur renvoyait encore un int.
        if isinstance(result, dict):
            inserted_count = result.get('inserted_count', 0)
            db_status = result.get('status', 'success')
            skipped_count = result.get('skipped_count', 0)
            journal_path = result.get('journal_path')
            db_message = result.get('message', '')
        else:
            inserted_count = result or 0
            db_status = 'success' if inserted_count else 'error'
            skipped_count = 0
            journal_path = None
            db_message = ''

        ok = db_status in ('success', 'success_partial_data')

        # Refléter le VRAI statut du connecteur : un dict d'erreur (ex. fichier
        # introuvable) ne lève pas d'exception → ne JAMAIS marquer COMPLETED à tort.
        _update_session_status(
            session_id,
            'COMPLETED' if ok else 'ERROR',
            vector_db=db_choice,
            index_name=pinecone_index_name or weaviate_class_name or qdrant_collection_name,
            error_message=None if ok else (db_message or 'Vector DB upload error'),
        )

        # Update metrics
        _update_vectordb_metrics(inserted_count, db_choice)

        logger.info(
            f"VectorDB upload {db_status}: {inserted_count} vectors "
            f"(skipped dedup: {skipped_count}) to {db_choice} in {duration:.1f}s"
        )

        response = {
            "status": "success" if ok else "error",
            "db_status": db_status,
            "inserted_count": inserted_count,
            "database": db_choice,
            "duration_seconds": duration,
        }
        if skipped_count:
            response["skipped_count"] = skipped_count
        if journal_path:
            response["journal_path"] = journal_path
        return response

    except Exception as e:
        logger.error(f"VectorDB upload task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise


def _upload_to_pinecone(
    input_file: str,
    index_name: str,
    namespace: str,
    progress_callback
) -> dict:
    """
    Upload embeddings to Pinecone.

    Args:
        input_file: Path to embeddings JSON file
        index_name: Pinecone index name
        namespace: Pinecone namespace (optional)
        progress_callback: Callback for progress updates

    Returns:
        dict with inserted_count
    """
    try:
        from scripts.rad_vectordb import insert_to_pinecone
    except ImportError as e:
        logger.error(f"Failed to import rad_vectordb: {e}")
        raise

    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY not configured")

    # NB (Lot 0.b) : aucun des connecteurs rad_vectordb n'accepte `progress_callback`
    # (signature `insert_to_pinecone(embeddings_json_file, index_name, pinecone_api_key,
    # namespace)`). Le passer levait un TypeError → le passage est supprimé ici.
    result = insert_to_pinecone(
        embeddings_json_file=input_file,
        index_name=index_name,
        namespace=namespace,
        pinecone_api_key=api_key
    )

    return result


def _upload_to_weaviate(
    input_file: str,
    class_name: str,
    tenant_name: str,
    progress_callback
) -> dict:
    """
    Upload embeddings to Weaviate.

    Args:
        input_file: Path to embeddings JSON file
        class_name: Weaviate class name
        tenant_name: Weaviate tenant name (optional)
        progress_callback: Callback for progress updates

    Returns:
        dict with inserted_count
    """
    try:
        from scripts.rad_vectordb import insert_to_weaviate_hybrid
    except ImportError as e:
        logger.error(f"Failed to import rad_vectordb: {e}")
        raise

    # Lot 0.b : `url` et `api_key` sont REQUIS (positionnels) par le connecteur et
    # étaient totalement omis ici → ValueError immédiat. Le worker Celery lit les
    # credentials depuis l'environnement du process (comme le chemin Pinecone).
    url = os.getenv('WEAVIATE_URL')
    api_key = os.getenv('WEAVIATE_API_KEY')
    if not url or not api_key:
        raise ValueError("WEAVIATE_URL et WEAVIATE_API_KEY doivent être configurés")

    result = insert_to_weaviate_hybrid(
        embeddings_json_file=input_file,
        url=url,
        api_key=api_key,
        class_name=class_name,
        tenant_name=tenant_name
    )

    return result


def _upload_to_qdrant(
    input_file: str,
    collection_name: str,
    progress_callback
) -> dict:
    """
    Upload embeddings to Qdrant.

    Args:
        input_file: Path to embeddings JSON file
        collection_name: Qdrant collection name
        progress_callback: Callback for progress updates

    Returns:
        dict with inserted_count
    """
    try:
        from scripts.rad_vectordb import insert_to_qdrant
    except ImportError as e:
        logger.error(f"Failed to import rad_vectordb: {e}")
        raise

    # Lot 0.b : `qdrant_url` était omis (ValueError immédiat) et `progress_callback`
    # non supporté (TypeError). Credentials lus depuis l'environnement du worker.
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')  # None toléré (instance locale non sécurisée)
    if not qdrant_url:
        raise ValueError("QDRANT_URL doit être configuré")

    result = insert_to_qdrant(
        embeddings_json_file=input_file,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )

    return result


def _update_session_status(
    session_id: int,
    status: str,
    vector_db: str = None,
    index_name: str = None,
    error_message: str = None
) -> None:
    """
    Update pipeline session status in database.

    Args:
        session_id: Database session ID
        status: New status value
        vector_db: Vector database type
        index_name: Vector database index/collection name
        error_message: Error message if status is ERROR
    """
    try:
        from app.database.session import SessionLocal
        from app.models.pipeline_session import PipelineSession, SessionStatus

        status_map = {
            'UPLOADING': SessionStatus.UPLOADING,
            'COMPLETED': SessionStatus.COMPLETED,
            'ERROR': SessionStatus.ERROR,
        }

        db = SessionLocal()
        try:
            session = db.query(PipelineSession).filter(
                PipelineSession.id == session_id
            ).first()

            if session:
                session.status = status_map.get(status, SessionStatus.ERROR)
                session.updated_at = datetime.utcnow()

                if vector_db:
                    session.vector_db_type = vector_db

                if index_name:
                    session.index_name = index_name

                if status == 'COMPLETED':
                    session.completed_at = datetime.utcnow()

                if error_message:
                    session.error_message = error_message[:1000]

                db.commit()
        finally:
            db.close()

    except Exception as e:
        logger.warning(f"Failed to update session status: {e}")


def _update_vectordb_metrics(inserted_count: int, database: str) -> None:
    """
    Update Prometheus metrics for vector database uploads.

    Args:
        inserted_count: Number of vectors inserted
        database: Database type ('pinecone', 'weaviate', 'qdrant')
    """
    try:
        from app.utils.metrics import vectordb_upsert_total, METRICS_ENABLED

        if METRICS_ENABLED:
            vectordb_upsert_total.labels(database=database).inc(inserted_count)
    except Exception as e:
        logger.debug(f"Failed to update metrics: {e}")
