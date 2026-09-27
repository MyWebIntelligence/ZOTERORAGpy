"""
Celery Tasks: Embedding Generation
==================================

This module contains the Celery tasks that generate dense and sparse
embeddings. Like the HTTP routes ``POST /dense_embedding_generation`` and
``POST /sparse_embedding_generation``, they run ``scripts/rad_chunk.py
--phase dense|sparse`` in a subprocess (same argv, same timeout) through
``app.tasks.runner``, with the environment of the submitting user
(``build_subprocess_env``, user loaded from ``user_id``).

Tasks:
    dense_embedding_task: Generate dense embeddings using OpenAI
    sparse_embedding_task: Generate sparse embeddings using spaCy

Features:
    - Retry limited to infrastructure errors (``runner.INFRASTRUCTURE_ERRORS``);
      a non-zero exit (rate limits included: the script has its own retries),
      a timeout or a missing credential is never retried
    - Progress reporting from the script's ``PROGRESS|...`` lines
    - Session status tracking in database
    - Prometheus metrics integration
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


class EmbeddingTask(Task):
    """
    Base task class for embedding tasks.

    Attributes:
        autoretry_for: Infrastructure errors only (database, process spawn)
        retry_kwargs: Retry configuration
        retry_backoff: Enable exponential backoff
    """
    autoretry_for = runner.INFRASTRUCTURE_ERRORS
    retry_kwargs = {'max_retries': 5, 'countdown': 60}
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes between retries
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
    session_id: int,
    user_id: int = None,
    embedding_provider: str = None
) -> dict:
    """
    Generate dense embeddings using OpenAI text-embedding-3-large.

    Runs ``rad_chunk.py --phase dense`` on output_chunks.json with the
    OpenAI key of ``user_id`` (required, as the HTTP route). With
    ``embedding_provider='albert'`` (resolved by the submitting route while
    Albert is enabled), the Albert key is required instead, and the
    provider is written to ``EMBEDDING_PROVIDER`` (Albert timeout).

    Args:
        self: Celery task instance (bound)
        input_file: Path to input JSON file (output_chunks.json)
        output_dir: Directory for output JSON file
        session_id: Database session ID for status tracking
        user_id: ID of the user who submitted the task (required)
        embedding_provider: Resolved provider (``openai``/``albert``), or
            None for the historical OpenAI path (environment untouched)

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
    output_file = os.path.join(output_dir, 'output_chunks_with_embeddings.json')

    try:
        # Update session status
        _update_session_status(session_id, 'EMBEDDING')

        # Report initial progress
        runner.report_state(self, {
            'current': 0,
            'total': 100,
            'percent': 0,
            'item': 'Initializing embedding generation...',
            'status': 'Starting dense embedding generation'
        })

        logger.info(f"Starting dense embedding task: {input_file}")

        user = runner.load_user(user_id)
        env = runner.build_task_env(user, runner.STAGE_DENSE, embedding_provider=embedding_provider)
        cmd = runner.dense_argv(input_file, output_dir)

        result = runner.run_script(
            cmd,
            env,
            session_dir=output_dir,  # session lock (audit A12)
            on_progress=runner.make_progress_reporter(self, 'Generating embeddings'),
            timeout=runner.task_timeout(runner.DENSE_TIMEOUT, embedding_provider == runner.ALBERT_DB_CHOICE)
        )
        runner.check_script_result(result, cmd, env)

        if not os.path.exists(output_file):
            raise runner.ScriptFailedError("Output file not created")

        chunk_count = _count_json_items(output_file)
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

    except runner.ScriptRevokedError as e:
        # Revoked while the script ran: keep the REVOKED state set by the worker.
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise Ignore()

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
    session_id: int,
    user_id: int = None
) -> dict:
    """
    Generate sparse embeddings using spaCy NLP features.

    Runs ``rad_chunk.py --phase sparse`` on
    output_chunks_with_embeddings.json. No external credential is required,
    but the environment is still isolated for ``user_id``.

    Args:
        self: Celery task instance (bound)
        input_file: Path to input JSON (output_chunks_with_embeddings.json)
        output_dir: Directory for output JSON file
        session_id: Database session ID for status tracking
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
            errors) or immediately (script failure)
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
        runner.report_state(self, {
            'current': 0,
            'total': 100,
            'percent': 0,
            'item': 'Initializing sparse embedding...',
            'status': 'Starting sparse embedding generation'
        })

        logger.info(f"Starting sparse embedding task: {input_file}")

        user = runner.load_user(user_id)
        env = runner.build_task_env(user, runner.STAGE_SPARSE)
        cmd = runner.sparse_argv(input_file, output_dir)

        result = runner.run_script(
            cmd,
            env,
            session_dir=output_dir,  # session lock (audit A12)
            on_progress=runner.make_progress_reporter(self, 'Generating sparse embeddings'),
            timeout=runner.SPARSE_TIMEOUT
        )
        runner.check_script_result(result, cmd, env)

        if not os.path.exists(output_file):
            raise runner.ScriptFailedError("Output file not created")

        chunk_count = _count_json_items(output_file)
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

    except runner.ScriptRevokedError as e:
        # Revoked while the script ran: keep the REVOKED state set by the worker.
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise Ignore()

    except Exception as e:
        logger.error(f"Sparse embedding task failed: {e}", exc_info=True)
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
        from app.models.pipeline_session import PipelineSession, SessionStatus

        status_map = {
            'EMBEDDING': SessionStatus.EMBEDDING,
            'EMBEDDED': SessionStatus.EMBEDDED,
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
