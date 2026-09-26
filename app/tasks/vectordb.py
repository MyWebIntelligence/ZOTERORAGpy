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
    - Albert collections (``db_choice='albert'``, queued only while Albert is
      enabled): never retried (a redelivered message is refused), retention
      acknowledgement required, manifest copied out of ``uploads/``
"""
import os
import re
import logging
from datetime import datetime
from celery import Task
from celery.exceptions import Ignore

from app.celery_app import celery_app
from app.core.config import UPLOAD_DIR
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
    user_id: int = None,
    albert_collection_id: int = None,
    albert_collection_name: str = None,
    albert_create_collection: bool = False,
    albert_gdpr_ack: bool = False
) -> dict:
    """
    Upload embeddings to a vector database.

    Runs ``rad_vectordb.py`` on output_chunks_with_embeddings_sparse.json
    with the vector database credentials of ``user_id`` (as the HTTP route).
    ``db_choice='albert'`` runs ``_upload_to_albert`` instead (never retried).

    Args:
        self: Celery task instance (bound)
        input_file: Path to embeddings JSON file
        session_id: Database session ID for status tracking
        db_choice: Target database ('pinecone', 'weaviate', 'qdrant', 'albert')
        pinecone_index_name: Pinecone index name (if db_choice='pinecone')
        pinecone_namespace: Pinecone namespace (optional)
        weaviate_class_name: Weaviate class name (if db_choice='weaviate')
        weaviate_tenant_name: Weaviate tenant name (optional)
        qdrant_collection_name: Qdrant collection name (if db_choice='qdrant')
        user_id: ID of the user who submitted the task (required)
        albert_collection_id: Albert collection id (if db_choice='albert')
        albert_collection_name: Exact Albert collection name (if db_choice='albert')
        albert_create_collection: Create the private collection if the name is unknown
        albert_gdpr_ack: Retention acknowledgement given at submission (required for albert)

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

    if db_choice == runner.ALBERT_DB_CHOICE:
        return _upload_to_albert(
            self,
            input_file=input_file,
            session_id=session_id,
            user_id=user_id,
            collection_id=albert_collection_id,
            collection_name=albert_collection_name,
            create_collection=albert_create_collection,
            gdpr_ack=albert_gdpr_ack,
            start_time=start_time,
        )

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


def _albert_session_label(input_file: str) -> str:
    """
    Session folder of an upload input, relative to uploads/ (as the HTTP route).

    Args:
        input_file: Absolute path of the upload input file.

    Returns:
        The relative session folder, or the folder name when the input is
        outside the upload directory.
    """
    folder = os.path.dirname(os.path.abspath(input_file))
    try:
        relative = os.path.relpath(folder, UPLOAD_DIR)
    except ValueError:
        relative = None
    if not relative or relative.startswith(os.pardir):
        return os.path.basename(folder)
    return relative


def _is_redelivered(task) -> bool:
    """
    Tell whether the broker delivered the current message of ``task`` again.

    With ``task_acks_late`` and ``task_reject_on_worker_lost``, a message is
    requeued after a worker loss, and Redis also restores an unacknowledged
    message after its visibility timeout; kombu then marks it
    ``redelivered`` in ``delivery_info``.

    Args:
        task: The bound Celery task.

    Returns:
        True when ``task.request.delivery_info['redelivered']`` is set
        (False outside a worker, e.g. a direct ``task.run`` call).
    """
    info = getattr(getattr(task, "request", None), "delivery_info", None)
    if not isinstance(info, dict):
        return False
    return bool(info.get("redelivered"))


def _upload_to_albert(
    task,
    *,
    input_file: str,
    session_id: int,
    user_id: int,
    collection_id: int,
    collection_name: str,
    create_collection: bool,
    gdpr_ack: bool,
    start_time: datetime
) -> dict:
    """
    Albert branch of ``upload_to_vectordb_task``: never retried.

    Runs ``rad_vectordb.py --db albert`` (flags of the HTTP route, retention
    acknowledgement required) with the Albert key of ``user_id`` and the
    Albert timeout. Every failure is deterministic for Celery: an
    infrastructure error (database, process spawn) is re-raised as a
    ``TaskError`` so that ``autoretry_for`` never replays an upload, and a
    message the broker delivers again (``_is_redelivered``: worker lost,
    visibility timeout) fails with a ``TaskError`` before anything is sent.
    The manifest written next to the input is copied out of ``uploads/`` in
    a ``finally`` block, even after a failure.

    Args:
        task: The bound Celery task.
        input_file: Upload input (sparse, dense or plain chunks file).
        session_id: Database session ID for status tracking.
        user_id: ID of the user who submitted the task.
        collection_id: Albert collection id, or None.
        collection_name: Exact Albert collection name, or None.
        create_collection: Create the private collection if the name is unknown.
        gdpr_ack: Retention acknowledgement given at submission.
        start_time: Start time of the task (duration).

    Returns:
        dict: the keys of the other targets (``status``, ``db_status``,
        ``inserted_count``, ``database``, ``duration_seconds``, optional
        ``skipped_count`` / ``journal_path``) plus ``existing_count`` and,
        when written, ``manifest_path``.

    Raises:
        runner.TaskError: Missing acknowledgement or target, script failure
            (``ScriptFailedError``), timeout, infrastructure error.
        CredentialMissingError: No Albert key for the user.
        Ignore: The task was revoked while the script ran.
    """
    db_choice = runner.ALBERT_DB_CHOICE
    manifest_source = None
    try:
        if _is_redelivered(task):
            # Albert writes are not idempotent under concurrency: a message
            # delivered again (worker lost, visibility timeout) is never
            # replayed; the manifest of the first delivery is still archived.
            manifest_source = runner.albert_manifest_source(input_file)
            raise runner.TaskError(
                "Envoi Albert remis une seconde fois par le broker (worker perdu ou délai de "
                "visibilité dépassé) : non rejoué automatiquement ; vérifier la collection puis relancer l'envoi."
            )

        _update_session_status(session_id, 'UPLOADING', vector_db=db_choice)
        runner.report_state(task, {
            'current': 0,
            'total': 100,
            'percent': 0,
            'item': 'Connecting to albert...',
            'status': 'Starting upload to albert'
        })

        if not gdpr_ack:
            raise runner.TaskError("Acquittement de rétention Albert manquant : envoi refusé.")
        if collection_id is None and not (collection_name or "").strip():
            raise runner.TaskError("Collection Albert requise : identifiant ou nom exact.")

        logger.info(f"Starting Albert upload task: {input_file}")

        user = runner.load_user(user_id)
        env = runner.build_task_env(user, runner.STAGE_VECTORDB, db_choice=db_choice)
        cmd = runner.vectordb_argv(
            input_file,
            db_choice,
            albert_collection_id=collection_id,
            albert_collection_name=collection_name,
            albert_create_collection=bool(create_collection),
            albert_ack_retention=True,
        )
        manifest_source = runner.albert_manifest_source(input_file)

        result = runner.run_script(
            cmd,
            env,
            on_progress=runner.make_progress_reporter(task, 'Uploading batch'),
            timeout=runner.task_timeout(runner.VECTORDB_TIMEOUT, True)
        )
        runner.check_script_result(result, cmd, env, prefer_stdout=True)

        parsed = runner.parse_albert_stdout(result.stdout)
        inserted_count = parsed["inserted_count"] or 0
        skipped_count = parsed["skipped_count"] or 0
        status_match = _RESULT_STATUS_RE.search(result.stdout or "")
        db_status = status_match.group(1) if status_match else 'success'
        duration = (datetime.utcnow() - start_time).total_seconds()

        _update_session_status(
            session_id,
            'COMPLETED',
            vector_db=db_choice,
            index_name=collection_name or (str(collection_id) if collection_id is not None else None),
        )
        _update_vectordb_metrics(inserted_count, db_choice)

        logger.info(
            f"Albert upload {db_status}: {inserted_count} chunks "
            f"(existing: {parsed['existing_count'] or 0}, skipped dedup: {skipped_count}) in {duration:.1f}s"
        )

        response = {
            "status": "success",
            "db_status": db_status,
            "inserted_count": inserted_count,
            "database": db_choice,
            "duration_seconds": duration,
            "existing_count": parsed["existing_count"] or 0,
        }
        if skipped_count:
            response["skipped_count"] = skipped_count
        if parsed["journal_path"]:
            response["journal_path"] = parsed["journal_path"]
        if parsed["manifest_path"]:
            response["manifest_path"] = parsed["manifest_path"]
        return response

    except runner.ScriptRevokedError as e:
        # Revoked while the script ran: keep the REVOKED state set by the worker.
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise Ignore()

    except runner.TaskInfrastructureError as e:
        # Never replayed automatically on the Albert path.
        logger.error(f"Albert upload task stopped (infrastructure, not retried): {e}")
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise runner.TaskError(f"Envoi Albert interrompu, non relancé automatiquement : {e}") from e

    except Exception as e:
        logger.error(f"Albert upload task failed: {e}", exc_info=True)
        _update_session_status(session_id, 'ERROR', error_message=str(e))
        raise

    finally:
        if manifest_source:
            runner.archive_albert_manifest(manifest_source, user_id, _albert_session_label(input_file))


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
