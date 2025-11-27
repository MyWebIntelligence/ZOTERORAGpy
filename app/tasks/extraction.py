"""
Celery Task: PDF Extraction
===========================

This module contains Celery tasks for extracting text from PDF documents
using OCR. It wraps the rad_dataframe.py script functionality for
asynchronous execution via Celery workers.

Task:
    process_dataframe_task: Process Zotero JSON + PDFs to generate CSV

Features:
    - Automatic retry on transient failures
    - Progress reporting via task state updates
    - Session status tracking in database
    - Prometheus metrics integration
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


class ExtractionTask(Task):
    """
    Base task class with automatic retry for extraction tasks.

    Attributes:
        autoretry_for: Tuple of exception types to retry on
        retry_kwargs: Retry configuration (max_retries, countdown)
        retry_backoff: Enable exponential backoff
    """
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 30}
    retry_backoff = True
    retry_backoff_max = 300  # Max 5 minutes between retries
    retry_jitter = True      # Add randomness to prevent thundering herd


@celery_app.task(
    base=ExtractionTask,
    bind=True,
    name='extraction.process_dataframe',
    queue='extraction'
)
def process_dataframe_task(
    self,
    json_path: str,
    base_dir: str,
    output_path: str,
    session_id: int
) -> dict:
    """
    Process Zotero JSON export and PDFs to generate CSV with extracted text.

    This task wraps the rad_dataframe.py functionality for async execution.
    It updates task state with progress information for real-time monitoring.

    Args:
        self: Celery task instance (bound)
        json_path: Path to Zotero JSON export file
        base_dir: Base directory for resolving PDF paths
        output_path: Output CSV file path
        session_id: Database session ID for status tracking

    Returns:
        dict: {
            "status": "success",
            "row_count": int,
            "output": str,
            "duration_seconds": float
        }

    Raises:
        Exception: Re-raised after max retries exhausted
    """
    start_time = datetime.utcnow()

    try:
        # Update session status to EXTRACTING
        _update_session_status(session_id, 'EXTRACTING')

        # Report initial progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'percent': 0,
                'item': 'Initializing extraction...',
                'status': 'Starting PDF extraction'
            }
        )

        logger.info(f"Starting extraction task: {json_path} -> {output_path}")

        # Import rad_dataframe module
        try:
            from scripts.rad_dataframe import load_zotero_to_dataframe_incremental
        except ImportError as e:
            logger.error(f"Failed to import rad_dataframe: {e}")
            raise

        # Progress callback for real-time updates
        def progress_callback(current: int, total: int, item_title: str):
            """Update Celery task state with progress."""
            percent = int((current / total) * 100) if total > 0 else 0
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'item': item_title[:100] if item_title else '',
                    'status': f'Processing document {current}/{total}'
                }
            )

        # Execute extraction
        df = load_zotero_to_dataframe_incremental(
            json_path=json_path,
            base_dir=base_dir,
            output_csv=output_path,
            progress_callback=progress_callback
        )

        row_count = len(df) if df is not None else 0
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Update session status to EXTRACTED
        _update_session_status(session_id, 'EXTRACTED', row_count=row_count)

        # Update Prometheus metrics
        _update_extraction_metrics(row_count)

        logger.info(f"Extraction completed: {row_count} documents in {duration:.1f}s")

        return {
            "status": "success",
            "row_count": row_count,
            "output": output_path,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Extraction task failed: {e}", exc_info=True)

        # Update session status to ERROR
        _update_session_status(session_id, 'ERROR', error_message=str(e))

        # Re-raise for Celery retry mechanism
        raise


def _update_session_status(
    session_id: int,
    status: str,
    row_count: int = None,
    error_message: str = None
) -> None:
    """
    Update pipeline session status in database.

    Args:
        session_id: Database session ID
        status: New status value (EXTRACTING, EXTRACTED, ERROR)
        row_count: Number of rows processed (optional)
        error_message: Error message if status is ERROR (optional)
    """
    try:
        from app.database.session import SessionLocal
        from app.models.pipeline_session import PipelineSession, SessionStatus

        # Map string status to enum
        status_map = {
            'EXTRACTING': SessionStatus.EXTRACTING,
            'EXTRACTED': SessionStatus.EXTRACTED,
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

                if row_count is not None:
                    session.row_count = row_count

                if error_message:
                    session.error_message = error_message[:1000]  # Truncate

                if status == 'EXTRACTED':
                    session.completed_at = datetime.utcnow()

                db.commit()
                logger.debug(f"Session {session_id} status updated to {status}")
        finally:
            db.close()

    except Exception as e:
        logger.warning(f"Failed to update session status: {e}")


def _update_extraction_metrics(row_count: int) -> None:
    """
    Update Prometheus metrics for extraction.

    Args:
        row_count: Number of documents extracted
    """
    try:
        from app.utils.metrics import pdf_processed_total, METRICS_ENABLED

        if METRICS_ENABLED:
            pdf_processed_total.labels(provider='celery').inc(row_count)
    except Exception as e:
        logger.debug(f"Failed to update metrics: {e}")
