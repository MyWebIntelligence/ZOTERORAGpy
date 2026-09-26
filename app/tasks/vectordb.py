"""
Celery Task: Vector Database Upload
===================================

This module contains the Celery task that uploads embeddings to a vector
database. Like the HTTP route ``POST /upload_db``, it runs
``scripts/rad_vectordb.py --input --db ...`` in a subprocess (same argv,
``--class`` included, same timeout) through ``app.tasks.runner``, with the
environment of the submitting user (``build_subprocess_env``, user loaded
from ``user_id``). The worker never reads a vector database credential
from its own environment on behalf of a user, and no credential travels
through the broker.

Task:
    upload_to_vectordb_task: Upload embeddings to Pinecone, Weaviate, or Qdrant

Features:
    - Support for multiple vector databases
    - Retry limited to infrastructure errors (``runner.INFRASTRUCTURE_ERRORS``);
      a non-zero exit (connector error), a timeout or a missing credential is
      never retried
    - Result parsed from the script's ``=== Result ===`` block with the same
      anchored patterns as the HTTP route
"""
import re
import logging
from datetime import datetime
from celery import Task
from celery.exceptions import Ignore

from app.celery_app import celery_app
from app.tasks import runner

logger = logging.getLogger(__name__)

# "Status: <status>" line of the rad_vectordb.py "=== Result ===" block.
_RESULT_STATUS_RE = re.compile(r'^Status:\s*(\S+)', re.MULTILINE)


class VectorDBTask(Task):
    """
    Base task class for vector DB tasks.

    Attributes:
        autoretry_for: Infrastructure errors only (database, process spawn)
        retry_kwargs: Retry configuration
        retry_backoff: Enable exponential backoff
    """
    autoretry_for = runner.INFRASTRUCTURE_ERRORS
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
    qdrant_collection_name: str = None,
    user_id: int = None
) -> dict:
    """
    Upload embeddings to a vector database.

    Runs ``rad_vectordb.py`` on output_chunks_with_embeddings_sparse.json
    with the vector database credentials of ``user_id`` (as the HTTP route).

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
        user_id: ID of the user who submitted the task (required)

    Returns:
        dict: {
            "status": "success",
            "db_status": str,
            "inserted_count": int,
            "database": str,
            "duration_seconds": float,
            "skipped_count": int (only when > 0),
            "journal_path": str (only when a dedup journal exists)
        }

    Raises:
        Exception: Re-raised after max retries exhausted (infrastructure
            errors) or immediately (connector error, missing credential)
    """
    start_time = datetime.utcnow()

    try:
        # Update session status
        _update_session_status(session_id, 'UPLOADING', vector_db=db_choice)

        # Report initial progress
        runner.report_state(self, {
            'current': 0,
            'total': 100,
            'percent': 0,
            'item': f'Connecting to {db_choice}...',
            'status': f'Starting upload to {db_choice}'
        })

        logger.info(f"Starting vectordb upload task: {input_file} -> {db_choice}")

        if db_choice not in runner.VECTORDB_CHOICES:
            raise ValueError(f"Unknown database: {db_choice}")

        user = runner.load_user(user_id)
        env = runner.build_task_env(user, runner.STAGE_VECTORDB, db_choice=db_choice)
        cmd = runner.vectordb_argv(
            input_file,
            db_choice,
            pinecone_index_name=pinecone_index_name,
            pinecone_namespace=pinecone_namespace,
            weaviate_class_name=weaviate_class_name,
            weaviate_tenant_name=weaviate_tenant_name,
            qdrant_collection_name=qdrant_collection_name,
        )

        result = runner.run_script(
            cmd,
            env,
            on_progress=runner.make_progress_reporter(self, 'Uploading batch'),
            timeout=runner.VECTORDB_TIMEOUT
        )
        # rad_vectordb.py exits 1 on a connector error (status error or
        # partial_error): its diagnostics are on stdout.
        runner.check_script_result(result, cmd, env, prefer_stdout=True)

        parsed = runner.parse_vectordb_stdout(result.stdout)
        inserted_count = parsed["inserted_count"] or 0
        skipped_count = parsed["skipped_count"] or 0
        journal_path = parsed["journal_path"]
        status_match = _RESULT_STATUS_RE.search(result.stdout or "")
        db_status = status_match.group(1) if status_match else 'success'

        duration = (datetime.utcnow() - start_time).total_seconds()

        _update_session_status(
            session_id,
            'COMPLETED',
            vector_db=db_choice,
            index_name=pinecone_index_name or weaviate_class_name or qdrant_collection_name,
        )

        # Update metrics
        _update_vectordb_metrics(inserted_count, db_choice)

        logger.info(
            f"VectorDB upload {db_status}: {inserted_count} vectors "
            f"(skipped dedup: {skipped_count}) to {db_choice} in {duration:.1f}s"
        )

        response = {
            "status": "success",
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

    except runner.ScriptRevokedError as e:
        # Revoked while the script ran: keep the REVOKED state set by the worker.
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise Ignore()

    except Exception as e:
        logger.error(f"VectorDB upload task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise


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
        from app.models.pipeline_session import PipelineSession, SessionStatus

        status_map = {
            'UPLOADING': SessionStatus.UPLOADING,
            'COMPLETED': SessionStatus.COMPLETED,
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
