"""
Celery Tasks: Embedding Generation
==================================

This module contains Celery tasks for generating dense and sparse embeddings.
It wraps the rad_chunk.py script functionality for asynchronous execution
via Celery workers.

Tasks:
    dense_embedding_task: Generate dense embeddings using OpenAI
    sparse_embedding_task: Generate sparse embeddings using spaCy

Features:
    - Automatic retry on API rate limits
    - Progress reporting via task state updates
    - Session status tracking in database
    - Prometheus metrics integration
"""
import os
import sys
import json
import logging
from datetime import datetime
from celery import Task

from app.celery_app import celery_app

# Add scripts directory to path for imports
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '../../scripts')
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

logger = logging.getLogger(__name__)


class EmbeddingTask(Task):
    """
    Base task class with automatic retry for embedding tasks.

    Includes special handling for API rate limits with longer backoff.
    """
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 5, 'countdown': 60}
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes for rate limit recovery
    retry_jitter = True


@celery_app.task(
    base=EmbeddingTask,
    bind=True,
    name='embeddings.dense_embedding',
    queue='embeddings'
)
def dense_embedding_task(
    self,
    input_file: str,
    output_dir: str,
    session_id: int
) -> dict:
    """
    Generate dense embeddings using OpenAI text-embedding-3-large.

    This task processes the output_chunks.json file and generates
    dense vector embeddings for each chunk.

    Args:
        self: Celery task instance (bound)
        input_file: Path to input JSON file (output_chunks.json)
        output_dir: Directory for output JSON file
        session_id: Database session ID for status tracking

    Returns:
        dict: {
            "status": "success",
            "chunk_count": int,
            "output": str,
            "duration_seconds": float
        }

    Raises:
        Exception: Re-raised after max retries exhausted
    """
    start_time = datetime.utcnow()
    output_file = os.path.join(output_dir, 'output_chunks_with_embeddings.json')

    try:
        # Update session status
        _update_session_status(session_id, 'EMBEDDING')

        # Report initial progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'percent': 0,
                'item': 'Initializing embedding generation...',
                'status': 'Starting dense embedding generation'
            }
        )

        logger.info(f"Starting dense embedding task: {input_file}")

        # Import and execute
        try:
            from scripts.rad_chunk import run_dense_phase
        except ImportError as e:
            logger.error(f"Failed to import rad_chunk: {e}")
            raise

        # Progress callback
        def progress_callback(current: int, total: int, chunk_id: str):
            """Update Celery task state with progress."""
            percent = int((current / total) * 100) if total > 0 else 0
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'item': f'Chunk {chunk_id}' if chunk_id else '',
                    'status': f'Generating embeddings {current}/{total}'
                }
            )

        # Execute dense embedding generation
        run_dense_phase(
            input_file=input_file,
            output_dir=output_dir,
            progress_callback=progress_callback
        )

        # Count chunks
        chunk_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk_count = len(chunks) if isinstance(chunks, list) else 0

        duration = (datetime.utcnow() - start_time).total_seconds()

        # Update metrics
        _update_embedding_metrics(chunk_count, 'dense')

        logger.info(f"Dense embedding completed: {chunk_count} chunks in {duration:.1f}s")

        return {
            "status": "success",
            "chunk_count": chunk_count,
            "output": output_file,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Dense embedding task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise


@celery_app.task(
    base=EmbeddingTask,
    bind=True,
    name='embeddings.sparse_embedding',
    queue='embeddings'
)
def sparse_embedding_task(
    self,
    input_file: str,
    output_dir: str,
    session_id: int
) -> dict:
    """
    Generate sparse embeddings using spaCy NLP features.

    This task processes the output_chunks_with_embeddings.json file and
    generates sparse vector embeddings based on TF and linguistic features.

    Args:
        self: Celery task instance (bound)
        input_file: Path to input JSON (output_chunks_with_embeddings.json)
        output_dir: Directory for output JSON file
        session_id: Database session ID for status tracking

    Returns:
        dict: {
            "status": "success",
            "chunk_count": int,
            "output": str,
            "duration_seconds": float
        }

    Raises:
        Exception: Re-raised after max retries exhausted
    """
    start_time = datetime.utcnow()
    output_file = os.path.join(
        output_dir,
        'output_chunks_with_embeddings_sparse.json'
    )

    try:
        # Update session status
        _update_session_status(session_id, 'EMBEDDING')

        # Report initial progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'percent': 0,
                'item': 'Initializing sparse embedding...',
                'status': 'Starting sparse embedding generation'
            }
        )

        logger.info(f"Starting sparse embedding task: {input_file}")

        # Import and execute
        try:
            from scripts.rad_chunk import run_sparse_phase
        except ImportError as e:
            logger.error(f"Failed to import rad_chunk: {e}")
            raise

        # Progress callback
        def progress_callback(current: int, total: int, chunk_id: str):
            """Update Celery task state with progress."""
            percent = int((current / total) * 100) if total > 0 else 0
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'item': f'Chunk {chunk_id}' if chunk_id else '',
                    'status': f'Generating sparse embeddings {current}/{total}'
                }
            )

        # Execute sparse embedding generation
        run_sparse_phase(
            input_file=input_file,
            output_dir=output_dir,
            progress_callback=progress_callback
        )

        # Count chunks
        chunk_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk_count = len(chunks) if isinstance(chunks, list) else 0

        duration = (datetime.utcnow() - start_time).total_seconds()

        # Update session to EMBEDDED
        _update_session_status(session_id, 'EMBEDDED')

        # Update metrics
        _update_embedding_metrics(chunk_count, 'sparse')

        logger.info(f"Sparse embedding completed: {chunk_count} chunks in {duration:.1f}s")

        return {
            "status": "success",
            "chunk_count": chunk_count,
            "output": output_file,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Sparse embedding task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise


def _update_session_status(
    session_id: int,
    status: str,
    error_message: str = None
) -> None:
    """
    Update pipeline session status in database.

    Args:
        session_id: Database session ID
        status: New status value
        error_message: Error message if status is ERROR
    """
    try:
        from app.database.session import SessionLocal
        from app.models.pipeline_session import PipelineSession, SessionStatus

        status_map = {
            'EMBEDDING': SessionStatus.EMBEDDING,
            'EMBEDDED': SessionStatus.EMBEDDED,
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

                if error_message:
                    session.error_message = error_message[:1000]

                db.commit()
        finally:
            db.close()

    except Exception as e:
        logger.warning(f"Failed to update session status: {e}")


def _update_embedding_metrics(chunk_count: int, embedding_type: str) -> None:
    """
    Update Prometheus metrics for embedding generation.

    Args:
        chunk_count: Number of embeddings generated
        embedding_type: Type of embedding ('dense' or 'sparse')
    """
    try:
        from app.utils.metrics import embeddings_generated_total, METRICS_ENABLED

        if METRICS_ENABLED:
            model = 'text-embedding-3-large' if embedding_type == 'dense' else 'spacy'
            embeddings_generated_total.labels(
                model=model,
                type=embedding_type
            ).inc(chunk_count)
    except Exception as e:
        logger.debug(f"Failed to update metrics: {e}")
