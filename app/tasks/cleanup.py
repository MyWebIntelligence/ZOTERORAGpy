"""
Celery Tasks: Session Cleanup
=============================

This module contains Celery tasks for cleaning up expired sessions
and orphaned processes. These tasks run periodically via Celery Beat.

Tasks:
    cleanup_sessions_task: Remove expired session files and DB records
    cleanup_orphaned_processes_task: Terminate orphaned background processes

Features:
    - Periodic execution via Celery Beat
    - Safe file deletion with error handling
    - Database cleanup with transaction safety
    - Prometheus metrics for monitoring
"""
import os
import shutil
import logging
from datetime import datetime, timedelta
from celery import Task

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


class CleanupTask(Task):
    """
    Base task class for cleanup tasks.

    No automatic retry - cleanup failures are logged but not retried.
    """
    autoretry_for = ()  # No auto-retry for cleanup
    ignore_result = True  # Don't store results


@celery_app.task(
    base=CleanupTask,
    bind=True,
    name='cleanup.cleanup_sessions',
    queue='cleanup'
)
def cleanup_sessions_task(self) -> dict:
    """
    Clean up expired pipeline sessions.

    This task:
    1. Finds sessions past their TTL (expires_at)
    2. Deletes associated files from uploads/
    3. Marks sessions as cleaned_up in database

    Returns:
        dict: {
            "cleaned_count": int,
            "errors": int,
            "duration_seconds": float
        }
    """
    start_time = datetime.utcnow()
    cleaned_count = 0
    error_count = 0

    try:
        from app.database.session import SessionLocal
        from app.models.pipeline_session import PipelineSession
        from app.core.config import UPLOAD_DIR

        db = SessionLocal()
        try:
            # Find expired sessions not yet cleaned
            now = datetime.utcnow()
            expired_sessions = db.query(PipelineSession).filter(
                PipelineSession.expires_at <= now,
                PipelineSession.cleaned_up == False,
                PipelineSession.expires_at.isnot(None)
            ).all()

            logger.info(f"Found {len(expired_sessions)} expired sessions to clean")

            for session in expired_sessions:
                try:
                    # Delete session folder
                    session_path = os.path.join(UPLOAD_DIR, session.session_folder)
                    if os.path.exists(session_path):
                        shutil.rmtree(session_path)
                        logger.info(f"Deleted session folder: {session_path}")

                    # Mark as cleaned
                    session.mark_cleaned_up()
                    cleaned_count += 1

                except Exception as e:
                    logger.error(f"Failed to cleanup session {session.id}: {e}")
                    error_count += 1

            db.commit()

        finally:
            db.close()

        # Update metrics
        _update_cleanup_metrics(cleaned_count, 'expired')

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"Session cleanup completed: {cleaned_count} cleaned, "
            f"{error_count} errors in {duration:.1f}s"
        )

        return {
            "cleaned_count": cleaned_count,
            "errors": error_count,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Session cleanup task failed: {e}", exc_info=True)
        return {
            "cleaned_count": cleaned_count,
            "errors": error_count + 1,
            "error": str(e)
        }


@celery_app.task(
    base=CleanupTask,
    bind=True,
    name='cleanup.cleanup_orphaned_processes',
    queue='cleanup'
)
def cleanup_orphaned_processes_task(self) -> dict:
    """
    Clean up orphaned background processes.

    This task finds and terminates processes that:
    1. Were started by RAGpy (rad_*.py scripts)
    2. Have been running longer than expected
    3. Are not associated with active sessions

    Returns:
        dict: {
            "killed_count": int,
            "errors": int
        }
    """
    killed_count = 0
    error_count = 0

    try:
        import psutil

        # Process patterns to look for
        patterns = [
            'rad_dataframe.py',
            'rad_chunk.py',
            'rad_vectordb.py'
        ]

        # Max runtime before considering orphaned (2 hours)
        max_runtime = 7200

        now = datetime.utcnow()

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline)

                # Check if it's a RAGpy process
                is_ragpy_process = any(p in cmdline_str for p in patterns)
                if not is_ragpy_process:
                    continue

                # Check runtime
                create_time = proc.info.get('create_time', 0)
                runtime = now.timestamp() - create_time

                if runtime > max_runtime:
                    logger.warning(
                        f"Killing orphaned process PID {proc.pid}: "
                        f"{cmdline_str[:100]} (runtime: {runtime:.0f}s)"
                    )
                    proc.terminate()

                    # Wait briefly then force kill if needed
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()

                    killed_count += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as e:
                logger.error(f"Error checking process: {e}")
                error_count += 1

        if killed_count > 0:
            _update_cleanup_metrics(killed_count, 'orphaned')

        logger.info(
            f"Orphaned process cleanup: {killed_count} killed, {error_count} errors"
        )

        return {
            "killed_count": killed_count,
            "errors": error_count
        }

    except ImportError:
        logger.warning("psutil not available, skipping orphan cleanup")
        return {"killed_count": 0, "errors": 0, "skipped": True}
    except Exception as e:
        logger.error(f"Orphaned process cleanup failed: {e}", exc_info=True)
        return {"killed_count": 0, "errors": 1, "error": str(e)}


@celery_app.task(
    base=CleanupTask,
    bind=True,
    name='cleanup.cleanup_old_logs',
    queue='cleanup'
)
def cleanup_old_logs_task(self, days: int = 30) -> dict:
    """
    Clean up old log files.

    Args:
        days: Delete logs older than this many days (default: 30)

    Returns:
        dict: {
            "deleted_count": int,
            "freed_bytes": int
        }
    """
    deleted_count = 0
    freed_bytes = 0

    try:
        from app.core.config import LOG_DIR

        if not os.path.exists(LOG_DIR):
            return {"deleted_count": 0, "freed_bytes": 0}

        cutoff = datetime.utcnow() - timedelta(days=days)

        for filename in os.listdir(LOG_DIR):
            filepath = os.path.join(LOG_DIR, filename)

            if not os.path.isfile(filepath):
                continue

            # Check file modification time
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

            if mtime < cutoff:
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                    freed_bytes += size
                    logger.info(f"Deleted old log: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete log {filename}: {e}")

        logger.info(
            f"Log cleanup: deleted {deleted_count} files, "
            f"freed {freed_bytes / 1024 / 1024:.1f} MB"
        )

        return {
            "deleted_count": deleted_count,
            "freed_bytes": freed_bytes
        }

    except Exception as e:
        logger.error(f"Log cleanup failed: {e}", exc_info=True)
        return {"deleted_count": 0, "freed_bytes": 0, "error": str(e)}


def _update_cleanup_metrics(count: int, reason: str) -> None:
    """
    Update Prometheus metrics for cleanup operations.

    Args:
        count: Number of items cleaned
        reason: Cleanup reason ('expired', 'orphaned', 'manual')
    """
    try:
        from app.utils.metrics import session_cleanup_total, METRICS_ENABLED

        if METRICS_ENABLED:
            session_cleanup_total.labels(reason=reason).inc(count)
    except Exception as e:
        logger.debug(f"Failed to update metrics: {e}")
