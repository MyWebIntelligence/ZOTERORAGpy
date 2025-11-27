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

Usage:
    # Start worker
    celery -A app.celery_app worker --loglevel=info --concurrency=4

    # Start beat scheduler
    celery -A app.celery_app beat --loglevel=info

    # Start Flower monitoring
    celery -A app.celery_app flower --port=5555
"""
import os
import logging
from celery import Celery

logger = logging.getLogger(__name__)

# Configuration broker et backend
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Feature flag for dual mode (subprocess vs Celery)
CELERY_ENABLED = os.getenv('ENABLE_CELERY', 'false').lower() in ('true', '1', 'yes')

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

    # Task routing (optional - for future scaling)
    task_routes={
        'app.tasks.extraction.*': {'queue': 'extraction'},
        'app.tasks.chunking.*': {'queue': 'chunking'},
        'app.tasks.embeddings.*': {'queue': 'embeddings'},
        'app.tasks.vectordb.*': {'queue': 'vectordb'},
        'app.tasks.cleanup.*': {'queue': 'cleanup'},
        'app.tasks.monitoring.*': {'queue': 'monitoring'},
    },

    # Default queue for unspecified tasks
    task_default_queue='default',

    # Task track started state (for progress monitoring)
    task_track_started=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Cleanup expired sessions every 6 hours
    'cleanup-expired-sessions': {
        'task': 'app.tasks.cleanup.cleanup_sessions_task',
        'schedule': 21600.0,  # 6 hours in seconds
        'options': {'queue': 'cleanup'}
    },
    # Update system metrics every minute
    'update-system-metrics': {
        'task': 'app.tasks.monitoring.update_metrics_task',
        'schedule': 60.0,  # 1 minute
        'options': {'queue': 'monitoring'}
    },
    # Cleanup orphaned processes every hour
    'cleanup-orphaned-processes': {
        'task': 'app.tasks.cleanup.cleanup_orphaned_processes_task',
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
