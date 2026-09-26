"""
Celery Task: PDF Extraction
===========================

This module contains the Celery task that extracts text from the documents of
a Zotero export (OCR). Like the HTTP route ``POST /process_dataframe``, it
runs ``scripts/rad_dataframe.py`` in a subprocess (same argv, same timeout)
through ``app.tasks.runner``, with the environment of the submitting user
(``build_subprocess_env``, user loaded from ``user_id``). No credential is
read from the worker environment for a non-admin user, and none travels
through the broker.

Task:
    process_dataframe_task: Process Zotero JSON + PDFs to generate CSV

Features:
    - Retry limited to infrastructure errors (``runner.INFRASTRUCTURE_ERRORS``);
      a non-zero exit, a timeout or a missing credential is never retried
    - Progress reporting from the script's ``PROGRESS|...`` lines
    - Session status tracking in database
    - Prometheus metrics integration
"""
import os
import logging
from datetime import datetime
from celery import Task
from celery.exceptions import Ignore

from app.celery_app import celery_app
from app.tasks import runner

logger = logging.getLogger(__name__)


class ExtractionTask(Task):
    """
    Base task class for extraction tasks.

    Attributes:
        autoretry_for: Infrastructure errors only (database, process spawn)
        retry_kwargs: Retry configuration (max_retries, countdown)
        retry_backoff: Enable exponential backoff
    """
    autoretry_for = runner.INFRASTRUCTURE_ERRORS
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
    session_id: int,
    user_id: int = None
) -> dict:
    """
    Process Zotero JSON export and PDFs to generate CSV with extracted text.

    Runs ``rad_dataframe.py --json --dir --output`` with the credentials of
    ``user_id`` (Mistral, else OpenAI, as the HTTP route). It updates task
    state with progress information for real-time monitoring.

    Args:
        self: Celery task instance (bound)
        json_path: Path to Zotero JSON export file
        base_dir: Base directory for resolving PDF paths
        output_path: Output CSV file path
        session_id: Database session ID for status tracking
        user_id: ID of the user who submitted the task (required)

    Returns:
        dict: {
            "status": "success",
            "row_count": int,
            "output": str,
            "duration_seconds": float
        }

    Raises:
        Exception: Re-raised after max retries exhausted (infrastructure
            errors) or immediately (script failure, missing credential)
    """
    start_time = datetime.utcnow()

    try:
        # Update session status to EXTRACTING
        _update_session_status(session_id, 'EXTRACTING')

        # Report initial progress
        runner.report_state(self, {
            'current': 0,
            'total': 100,
            'percent': 0,
            'item': 'Initializing extraction...',
            'status': 'Starting PDF extraction'
        })

        logger.info(f"Starting extraction task: {json_path} -> {output_path}")

        user = runner.load_user(user_id)
        env = runner.build_task_env(user, runner.STAGE_EXTRACTION)
        cmd = runner.extraction_argv(json_path, base_dir, output_path)

        result = runner.run_script(
            cmd,
            env,
            on_progress=runner.make_progress_reporter(self, 'Processing document'),
            timeout=runner.EXTRACTION_TIMEOUT
        )
        runner.check_script_result(result, cmd, env)

        if not os.path.exists(output_path):
            raise runner.ScriptFailedError("Output CSV not found after script execution.")

        row_count = _count_csv_rows(output_path)
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

    except runner.ScriptRevokedError as e:
        # Revoked while the script ran: the process group is already killed;
        # keep the REVOKED state set by the worker.
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise Ignore()

    except Exception as e:
        logger.error(f"Extraction task failed: {e}", exc_info=True)

        # Update session status to ERROR
        _update_session_status(session_id, 'ERROR', error_message=str(e))

        # Re-raise (retried only for infrastructure errors)
        raise


def _count_csv_rows(csv_path: str) -> int:
    """
    Count the data rows of the CSV written by rad_dataframe.py.

    Reads the file like the HTTP route (escapechar first, then plain).

    Args:
        csv_path: Path to the output CSV

    Returns:
        Number of rows (0 if the file cannot be parsed)
    """
    try:
        import pandas as pd

        try:
            df = pd.read_csv(csv_path, escapechar='\\', dtype=str, keep_default_na=False)
        except pd.errors.ParserError:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        return len(df)
    except Exception as e:
        logger.warning(f"Could not count rows of {csv_path}: {e}")
        return 0


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
        from app.models.pipeline_session import PipelineSession, SessionStatus

        # Map string status to enum
        status_map = {
            'EXTRACTING': SessionStatus.EXTRACTING,
            'EXTRACTED': SessionStatus.EXTRACTED,
            'ERROR': SessionStatus.ERROR,
        }

        db = runner.SessionLocal()
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
