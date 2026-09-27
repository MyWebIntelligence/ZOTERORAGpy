"""
Celery Application Configuration
================================

This module configures the Celery distributed task queue for RAGpy.
It handles asynchronous execution of long-running tasks like:
- PDF extraction and OCR
- Text chunking
- Embedding generation
- Vector database uploads
- Session cleanup

Architecture:
    [User Request] -> [FastAPI] -> [Celery Queue] -> [Worker Pool]
                                        |
                                   [Redis Broker]
                                        |
                                   [Result Backend]

Environment Variables:
    CELERY_BROKER_URL: Redis broker URL (default: redis://localhost:6379/0)
    CELERY_RESULT_BACKEND: Redis result backend URL (default: redis://localhost:6379/0)
    ENABLE_CELERY: Feature flag to enable/disable Celery (default: false)

Queues and schedule (audit A04, 2026-09-27):
    Every task declares its queue in its decorator; ``CELERY_QUEUES`` lists
    them all in ``task_queues``, so a worker started without ``-Q`` consumes
    every one of them (``default`` included). ``task_routes`` and the Beat
    schedule use the explicit task names (``chunking.initial_chunking``,
    ``cleanup.cleanup_orphaned_processes``...), checked against the registry
    by ``tests/test_celery_tasks.py``.

    The expired-session cleanup belongs to the web application (APScheduler,
    ``app/core/scheduler.py``, ``CLEANUP_ENABLED``): Beat does not schedule
    it, and ``cleanup.cleanup_sessions`` (on demand) runs the same service.
    Beat schedules the worker-side jobs only: orphaned script processes of
    the worker container and the metrics refresh.

Usage:
    # Start worker (consumes every queue of CELERY_QUEUES)
    celery -A app.celery_app worker --loglevel=info --concurrency=4

    # Start beat scheduler
    celery -A app.celery_app beat --loglevel=info

    # Start Flower monitoring
    celery -A app.celery_app flower --port=5555
"""
import os
import logging
from celery import Celery
from kombu import Queue

logger = logging.getLogger(__name__)

# Configuration broker et backend
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Feature flag for dual mode (subprocess vs Celery)
CELERY_ENABLED = os.getenv('ENABLE_CELERY', 'false').lower() in ('true', '1', 'yes')

# Every queue a task of app/tasks/ publishes to (decorator ``queue=``), plus the
# default one: a worker started without -Q consumes all of them.
CELERY_DEFAULT_QUEUE = 'default'
CELERY_QUEUES = (
    CELERY_DEFAULT_QUEUE,
    'extraction',
    'chunking',
    'embeddings',
    'vectordb',
    'cleanup',
    'monitoring',
)

# Create Celery instance
celery_app = Celery(
    'ragpy',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.extraction',
        'app.tasks.chunking',
        'app.tasks.embeddings',
        'app.tasks.vectordb',
        'app.tasks.cleanup',
        'app.tasks.monitoring'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Retry policy - acknowledge tasks only after completion
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Timeouts (PDF extraction can take a while)
    task_soft_time_limit=3600,   # 1h soft limit (sends SIGTERM)
    task_time_limit=7200,        # 2h hard limit (sends SIGKILL)

    # Result expiration (clean up results after 24h)
    result_expires=86400,

    # Concurrency control
    worker_prefetch_multiplier=1,        # Fetch one task at a time
    worker_max_tasks_per_child=100,      # Restart worker after 100 tasks (memory cleanup)

    # Monitoring (for Flower)
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Queues consumed by a worker started without -Q (all of them)
    task_queues=tuple(Queue(name) for name in CELERY_QUEUES),

    # Task routing by explicit task name (same queues as the decorators)
    task_routes={
        'extraction.*': {'queue': 'extraction'},
        'chunking.*': {'queue': 'chunking'},
        'embeddings.*': {'queue': 'embeddings'},
        'vectordb.*': {'queue': 'vectordb'},
        'cleanup.*': {'queue': 'cleanup'},
        'monitoring.*': {'queue': 'monitoring'},
    },

    # Default queue for unspecified tasks
    task_default_queue=CELERY_DEFAULT_QUEUE,

    # Task track started state (for progress monitoring)
    task_track_started=True,
)

# Beat schedule for periodic tasks (explicit task names of the registry).
# The expired-session cleanup is scheduled by the web application
# (APScheduler), never here: one owner for that job.
celery_app.conf.beat_schedule = {
    # Update system metrics every minute
    'update-system-metrics': {
        'task': 'monitoring.update_metrics',
        'schedule': 60.0,  # 1 minute
        'options': {'queue': 'monitoring'}
    },
    # Cleanup orphaned processes of the worker container every hour
    'cleanup-orphaned-processes': {
        'task': 'cleanup.cleanup_orphaned_processes',
        'schedule': 3600.0,  # 1 hour
        'options': {'queue': 'cleanup'}
    },
}


def is_celery_available() -> bool:
    """
    Check if Celery broker (Redis) is available.

    Returns:
        True if broker is reachable, False otherwise.
    """
    if not CELERY_ENABLED:
        return False

    try:
        # Try to ping the broker
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Celery broker not available: {e}")
        return False


def get_task_status(task_id: str) -> dict:
    """
    Get the status of a Celery task.

    Args:
        task_id: The Celery task ID

    Returns:
        Dictionary with task state and metadata
    """
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task queued, waiting for worker',
            'progress': 0
        }
    elif task.state == 'STARTED':
        response = {
            'state': task.state,
            'status': 'Task started',
            'progress': 0
        }
    elif task.state == 'PROGRESS':
        info = task.info or {}
        response = {
            'state': task.state,
            'current': info.get('current', 0),
            'total': info.get('total', 1),
            'percent': info.get('percent', 0),
            'item': info.get('item', ''),
            'status': info.get('status', 'Processing...')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result,
            'progress': 100
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'error': str(task.info) if task.info else 'Unknown error',
            'traceback': task.traceback if hasattr(task, 'traceback') else None
        }
    elif task.state == 'REVOKED':
        response = {
            'state': task.state,
            'status': 'Task was cancelled'
        }
    else:
        response = {
            'state': task.state,
            'status': f'Unknown state: {task.state}'
        }

    return response


def revoke_task(task_id: str, terminate: bool = False) -> dict:
    """
    Revoke (cancel) a Celery task.

    Args:
        task_id: The Celery task ID to revoke
        terminate: If True, terminate the task immediately (SIGTERM)

    Returns:
        Dictionary with revocation status
    """
    try:
        celery_app.control.revoke(task_id, terminate=terminate, signal='SIGTERM')
        return {
            'success': True,
            'message': f'Task {task_id} revoked' + (' and terminated' if terminate else '')
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Log configuration on import
logger.info(f"Celery configured - Broker: {CELERY_BROKER_URL}, Enabled: {CELERY_ENABLED}")
