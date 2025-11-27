"""
Celery Task: Text Chunking
==========================

This module contains Celery tasks for chunking extracted text and
performing GPT recoding. It wraps the rad_chunk.py script functionality
for asynchronous execution via Celery workers.

Task:
    initial_chunking_task: Chunk CSV text and optionally recode with GPT

Features:
    - Automatic retry on transient failures
    - Progress reporting via task state updates
    - Session status tracking in database
    - Model selection for GPT recoding
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


class ChunkingTask(Task):
    """
    Base task class with automatic retry for chunking tasks.

    Attributes:
        autoretry_for: Tuple of exception types to retry on
        retry_kwargs: Retry configuration
        retry_backoff: Enable exponential backoff
    """
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 30}
    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True


@celery_app.task(
    base=ChunkingTask,
    bind=True,
    name='chunking.initial_chunking',
    queue='chunking'
)
def initial_chunking_task(
    self,
    input_csv: str,
    output_dir: str,
    session_id: int,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Chunk CSV text content and optionally recode with GPT.

    This task processes the output.csv from extraction, splits text into
    chunks using RecursiveCharacterTextSplitter, and optionally recodes
    using GPT for improved quality.

    Args:
        self: Celery task instance (bound)
        input_csv: Path to input CSV file (output.csv)
        output_dir: Directory for output JSON file
        session_id: Database session ID for status tracking
        model: LLM model for recoding (default: gpt-4o-mini)

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
    output_file = os.path.join(output_dir, 'output_chunks.json')

    try:
        # Update session status
        _update_session_status(session_id, 'CHUNKING')

        # Report initial progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'percent': 0,
                'item': 'Initializing chunking...',
                'status': 'Starting text chunking'
            }
        )

        logger.info(f"Starting chunking task: {input_csv} with model {model}")

        # Import and execute chunking
        try:
            from scripts.rad_chunk import run_initial_phase
        except ImportError as e:
            logger.error(f"Failed to import rad_chunk: {e}")
            raise

        # Progress callback
        def progress_callback(current: int, total: int, doc_title: str):
            """Update Celery task state with progress."""
            percent = int((current / total) * 100) if total > 0 else 0
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'item': doc_title[:100] if doc_title else '',
                    'status': f'Chunking document {current}/{total}'
                }
            )

        # Execute chunking
        run_initial_phase(
            input_csv=input_csv,
            output_dir=output_dir,
            model=model,
            progress_callback=progress_callback
        )

        # Count chunks
        chunk_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk_count = len(chunks) if isinstance(chunks, list) else 0

        duration = (datetime.utcnow() - start_time).total_seconds()

        # Update session status
        _update_session_status(session_id, 'CHUNKED', chunk_count=chunk_count)

        # Update metrics
        _update_chunking_metrics(chunk_count)

        logger.info(f"Chunking completed: {chunk_count} chunks in {duration:.1f}s")

        return {
            "status": "success",
            "chunk_count": chunk_count,
            "output": output_file,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Chunking task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise


def _update_session_status(
    session_id: int,
    status: str,
    chunk_count: int = None,
    error_message: str = None
) -> None:
    """
    Update pipeline session status in database.

    Args:
        session_id: Database session ID
        status: New status value
        chunk_count: Number of chunks generated (optional)
        error_message: Error message if status is ERROR (optional)
    """
    try:
        from app.database.session import SessionLocal
        from app.models.pipeline_session import PipelineSession, SessionStatus

        status_map = {
            'CHUNKING': SessionStatus.CHUNKING,
            'CHUNKED': SessionStatus.CHUNKED,
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

                if chunk_count is not None:
                    session.chunk_count = chunk_count

                if error_message:
                    session.error_message = error_message[:1000]

                db.commit()
        finally:
            db.close()

    except Exception as e:
        logger.warning(f"Failed to update session status: {e}")


def _update_chunking_metrics(chunk_count: int) -> None:
    """
    Update Prometheus metrics for chunking.

    Args:
        chunk_count: Number of chunks generated
    """
    try:
        from app.utils.metrics import chunks_generated_total, METRICS_ENABLED

        if METRICS_ENABLED:
            chunks_generated_total.labels(phase='initial').inc(chunk_count)
    except Exception as e:
        logger.debug(f"Failed to update metrics: {e}")
