"""
Celery Task: Text Chunking
==========================

This module contains the Celery task that chunks extracted text and performs
GPT recoding. Like the HTTP route ``POST /initial_text_chunking``, it runs
``scripts/rad_chunk.py --phase initial`` in a subprocess (same argv, same
timeout) through ``app.tasks.runner``, with the environment of the
submitting user (``build_subprocess_env``, user loaded from ``user_id``).

Task:
    initial_chunking_task: Chunk CSV text and optionally recode with GPT

Features:
    - Retry limited to infrastructure errors (``runner.INFRASTRUCTURE_ERRORS``);
      a non-zero exit, a timeout or a missing credential is never retried
    - Progress reporting from the script's ``PROGRESS|...`` lines
    - Session status tracking in database
    - Model selection for GPT recoding (``provider/model`` → OpenRouter key)
"""
import os
import json
import logging
from datetime import datetime
from celery import Task
from celery.exceptions import Ignore

from app.celery_app import celery_app
from app.tasks import runner

logger = logging.getLogger(__name__)


class ChunkingTask(Task):
    """
    Base task class for chunking tasks.

    Attributes:
        autoretry_for: Infrastructure errors only (database, process spawn)
        retry_kwargs: Retry configuration
        retry_backoff: Enable exponential backoff
    """
    autoretry_for = runner.INFRASTRUCTURE_ERRORS
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
    model: str = "gpt-4o-mini",
    user_id: int = None
) -> dict:
    """
    Chunk CSV text content and optionally recode with GPT.

    Runs ``rad_chunk.py --phase initial --model <model>`` with the
    credentials of ``user_id`` (single resolver, as the HTTP route: Albert
    key for ``albert/<id>`` while Albert is enabled, OpenRouter key for a
    ``provider/model`` slug, OpenAI key otherwise).

    Args:
        self: Celery task instance (bound)
        input_csv: Path to input CSV file (output.csv)
        output_dir: Directory for output JSON file
        session_id: Database session ID for status tracking
        model: LLM model for recoding (default: gpt-4o-mini)
        user_id: ID of the user who submitted the task (required)

    Returns:
        dict: {
            "status": "success",
            "chunk_count": int,
            "output": str,
            "duration_seconds": float
        }

    Raises:
        Exception: Re-raised after max retries exhausted (infrastructure
            errors) or immediately (script failure, missing credential)
    """
    start_time = datetime.utcnow()
    output_file = os.path.join(output_dir, 'output_chunks.json')
    model = model or runner.DEFAULT_CHUNKING_MODEL

    try:
        # Update session status
        _update_session_status(session_id, 'CHUNKING')

        # Report initial progress
        runner.report_state(self, {
            'current': 0,
            'total': 100,
            'percent': 0,
            'item': 'Initializing chunking...',
            'status': 'Starting text chunking'
        })

        logger.info(f"Starting chunking task: {input_csv} with model {model}")

        user = runner.load_user(user_id)
        env = runner.build_task_env(user, runner.STAGE_CHUNKING, model=model)
        cmd = runner.chunking_argv(input_csv, output_dir, model)

        result = runner.run_script(
            cmd,
            env,
            on_progress=runner.make_progress_reporter(self, 'Chunking document'),
            # albert/<id> model: Albert timeout (route rule)
            timeout=runner.task_timeout(runner.CHUNKING_TIMEOUT, runner.chunking_selects_albert(model))
        )
        runner.check_script_result(result, cmd, env)

        if not os.path.exists(output_file):
            raise runner.ScriptFailedError("Chunking completed but output file not found.")

        chunk_count = _count_json_items(output_file)
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

    except runner.ScriptRevokedError as e:
        # Revoked while the script ran: keep the REVOKED state set by the worker.
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise Ignore()

    except Exception as e:
        logger.error(f"Chunking task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise


def _count_json_items(json_path: str) -> int:
    """
    Count the chunks of a pipeline JSON file.

    Args:
        json_path: Path to a JSON list of chunks

    Returns:
        Number of items (0 if the file is not a readable JSON list)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception as e:
        logger.warning(f"Could not count chunks of {json_path}: {e}")
        return 0


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
        from app.models.pipeline_session import PipelineSession, SessionStatus

        status_map = {
            'CHUNKING': SessionStatus.CHUNKING,
            'CHUNKED': SessionStatus.CHUNKED,
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
