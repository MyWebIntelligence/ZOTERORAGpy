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
        inserted_count = result.get('inserted_count', 0)

        # Update session to COMPLETED
        _update_session_status(
            session_id,
            'COMPLETED',
            vector_db=db_choice,
            index_name=pinecone_index_name or weaviate_class_name or qdrant_collection_name
        )

        # Update metrics
        _update_vectordb_metrics(inserted_count, db_choice)

        logger.info(
            f"VectorDB upload completed: {inserted_count} vectors "
            f"to {db_choice} in {duration:.1f}s"
        )

        return {
            "status": "success",
            "inserted_count": inserted_count,
            "database": db_choice,
            "duration_seconds": duration
        }

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

    result = insert_to_pinecone(
        embeddings_json_file=input_file,
        index_name=index_name,
        namespace=namespace,
        pinecone_api_key=api_key,
        progress_callback=progress_callback
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

    result = insert_to_weaviate_hybrid(
        embeddings_json_file=input_file,
        class_name=class_name,
        tenant_name=tenant_name,
        progress_callback=progress_callback
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

    result = insert_to_qdrant(
        embeddings_json_file=input_file,
        collection_name=collection_name,
        progress_callback=progress_callback
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
