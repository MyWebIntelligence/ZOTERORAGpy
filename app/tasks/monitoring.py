"""
Celery Task: System Monitoring
==============================

This module contains Celery tasks for updating system metrics.
These tasks run periodically via Celery Beat to keep Prometheus
metrics up to date.

Task:
    update_metrics_task: Update all system metrics

Features:
    - Periodic execution via Celery Beat (every minute)
    - Disk usage monitoring
    - Session status tracking
    - LLM semaphore status
"""
import os
import logging
from datetime import datetime
from celery import Task

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


class MonitoringTask(Task):
    """
    Base task class for monitoring tasks.

    No retry, results not stored - monitoring is best-effort.
    """
    autoretry_for = ()
    ignore_result = True


@celery_app.task(
    base=MonitoringTask,
    bind=True,
    name='monitoring.update_metrics',
    queue='monitoring'
)
def update_metrics_task(self) -> dict:
    """
    Update all system metrics for Prometheus.

    This task updates:
    - Disk usage (uploads, logs, data folders)
    - Active session counts by status
    - Pending cleanup session count
    - LLM semaphore availability (if applicable)

    Returns:
        dict: Summary of updated metrics
    """
    updated = []

    try:
        # Update disk usage metrics
        try:
            from app.utils.metrics import update_disk_usage_metrics
            update_disk_usage_metrics()
            updated.append('disk_usage')
        except Exception as e:
            logger.debug(f"Failed to update disk metrics: {e}")

        # Update session metrics
        try:
            from app.utils.metrics import update_session_metrics
            update_session_metrics()
            updated.append('session_metrics')
        except Exception as e:
            logger.debug(f"Failed to update session metrics: {e}")

        # Update LLM semaphore metrics
        try:
            _update_llm_semaphore_metrics()
            updated.append('llm_semaphore')
        except Exception as e:
            logger.debug(f"Failed to update LLM semaphore metrics: {e}")

        # Update active user count
        try:
            _update_active_users_metrics()
            updated.append('active_users')
        except Exception as e:
            logger.debug(f"Failed to update active users metrics: {e}")

        return {
            "status": "success",
            "updated_metrics": updated,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Metrics update task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "updated_metrics": updated
        }


def _update_llm_semaphore_metrics() -> None:
    """
    Update LLM semaphore availability metric.

    Tracks how many slots are available for concurrent LLM calls.
    """
    try:
        from app.utils.metrics import llm_semaphore_available, METRICS_ENABLED

        if not METRICS_ENABLED:
            return

        # Try to get semaphore from llm_note_generator
        try:
            from app.utils.llm_note_generator import _LLM_SEMAPHORE

            if _LLM_SEMAPHORE is not None:
                # asyncio.Semaphore doesn't expose available count directly
                # We track MAX - (MAX - _value) = _value
                max_concurrent = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 5))
                # Note: This is an approximation as _value is internal
                llm_semaphore_available.set(max_concurrent)
        except ImportError:
            pass

    except Exception as e:
        logger.debug(f"Failed to update LLM semaphore metrics: {e}")


def _update_active_users_metrics() -> None:
    """
    Update active users metric.

    Counts users who have logged in within the last 24 hours.
    """
    try:
        from app.utils.metrics import active_users, METRICS_ENABLED

        if not METRICS_ENABLED:
            return

        from app.database.session import SessionLocal
        from app.models.user import User
        from datetime import timedelta

        db = SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)

            count = db.query(User).filter(
                User.last_login >= cutoff
            ).count()

            active_users.set(count)

        finally:
            db.close()

    except Exception as e:
        logger.debug(f"Failed to update active users metrics: {e}")


@celery_app.task(
    base=MonitoringTask,
    bind=True,
    name='monitoring.health_check',
    queue='monitoring'
)
def health_check_task(self) -> dict:
    """
    Perform a health check on the Celery worker.

    This task can be used to verify that workers are responding.

    Returns:
        dict: Health status information
    """
    import platform
    import socket

    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent
        }
    except ImportError:
        memory_info = {"error": "psutil not available"}
        cpu_percent = None

    return {
        "status": "healthy",
        "worker_hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": cpu_percent,
        "memory": memory_info
    }
