"""
Prometheus Metrics Module
=========================

This module defines custom Prometheus metrics for monitoring RAGpy performance
and operational health. Metrics are exposed via the /metrics endpoint.

Metric Categories:
- Counters: Track cumulative values (PDFs processed, chunks generated, errors)
- Histograms: Track latency distributions (extraction time, embedding time)
- Gauges: Track current state (active sessions, disk usage)

Usage:
    from app.utils.metrics import pdf_processed_total, track_duration

    # Increment counter
    pdf_processed_total.labels(provider='mistral').inc()

    # Track duration with decorator
    @track_duration(pdf_extraction_duration, provider='mistral')
    def extract_pdf():
        ...

Environment Variables:
    ENABLE_METRICS: Set to 'true' to enable metrics (default: true)
"""

import os
import time
from functools import wraps
from typing import Optional, Callable, Any

from prometheus_client import Counter, Histogram, Gauge, Info

# Check if metrics are enabled
METRICS_ENABLED = os.getenv('ENABLE_METRICS', 'true').lower() in ('true', '1', 'yes')

# =============================================================================
# COUNTERS - Cumulative metrics that only increase
# =============================================================================

pdf_processed_total = Counter(
    'ragpy_pdf_processed_total',
    'Total number of PDFs processed',
    ['provider']  # mistral, openai, legacy
)

chunks_generated_total = Counter(
    'ragpy_chunks_generated_total',
    'Total number of chunks generated',
    ['phase']  # initial, dense, sparse
)

embeddings_generated_total = Counter(
    'ragpy_embeddings_generated_total',
    'Total number of embeddings generated',
    ['model', 'type']  # text-embedding-3-large/dense, spacy/sparse
)

vectordb_upsert_total = Counter(
    'ragpy_vectordb_upsert_total',
    'Total vectors upserted to vector databases',
    ['database']  # pinecone, weaviate, qdrant
)

api_errors_total = Counter(
    'ragpy_api_errors_total',
    'Total API errors encountered',
    ['provider', 'error_type']  # openai/rate_limit, mistral/timeout, etc.
)

api_requests_total = Counter(
    'ragpy_api_requests_total',
    'Total API requests made',
    ['provider', 'endpoint']  # openai/embeddings, mistral/ocr, etc.
)

session_cleanup_total = Counter(
    'ragpy_session_cleanup_total',
    'Total sessions cleaned up',
    ['reason']  # expired, orphaned, manual
)

auth_events_total = Counter(
    'ragpy_auth_events_total',
    'Total authentication events',
    ['event_type']  # login_success, login_failed, logout, token_refresh
)

# =============================================================================
# HISTOGRAMS - Track distributions (latency, sizes)
# =============================================================================

pdf_extraction_duration = Histogram(
    'ragpy_pdf_extraction_duration_seconds',
    'PDF text extraction duration in seconds',
    ['provider'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]  # 1s to 10min
)

chunking_duration = Histogram(
    'ragpy_chunking_duration_seconds',
    'Document chunking duration in seconds',
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

embedding_batch_duration = Histogram(
    'ragpy_embedding_batch_duration_seconds',
    'Embedding batch generation duration in seconds',
    ['model'],
    buckets=[0.5, 1, 2, 5, 10, 20, 60, 120]
)

vectordb_upsert_duration = Histogram(
    'ragpy_vectordb_upsert_duration_seconds',
    'Vector database upsert duration in seconds',
    ['database'],
    buckets=[0.5, 1, 2, 5, 10, 30, 60]
)

request_latency = Histogram(
    'ragpy_request_latency_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

chunk_size_tokens = Histogram(
    'ragpy_chunk_size_tokens',
    'Chunk size in tokens',
    buckets=[100, 250, 500, 750, 1000, 1500, 2000, 3000]
)

# =============================================================================
# GAUGES - Track current state values
# =============================================================================

active_sessions = Gauge(
    'ragpy_active_sessions',
    'Number of active processing sessions',
    ['status']  # extracting, chunking, embedding, uploading
)

active_users = Gauge(
    'ragpy_active_users',
    'Number of active users (logged in last 24h)'
)

disk_usage_bytes = Gauge(
    'ragpy_disk_usage_bytes',
    'Disk usage in bytes',
    ['folder']  # uploads, logs, data
)

pending_cleanup_sessions = Gauge(
    'ragpy_pending_cleanup_sessions',
    'Number of sessions pending cleanup'
)

llm_semaphore_available = Gauge(
    'ragpy_llm_semaphore_available',
    'Number of available LLM semaphore slots'
)

# =============================================================================
# INFO - Static information about the application
# =============================================================================

app_info = Info(
    'ragpy_app',
    'RAGpy application information'
)

# Set app info on module load
app_info.info({
    'version': '1.0.0',
    'python_version': os.popen('python --version 2>&1').read().strip().replace('Python ', ''),
})


# =============================================================================
# UTILITY FUNCTIONS AND DECORATORS
# =============================================================================

def track_duration(histogram: Histogram, **labels):
    """
    Decorator to track function execution duration using a Prometheus histogram.

    Args:
        histogram: The Prometheus Histogram to record to.
        **labels: Static labels to apply to the metric.

    Example:
        @track_duration(pdf_extraction_duration, provider='mistral')
        def extract_pdf(pdf_path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not METRICS_ENABLED:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Merge static labels with any dynamic ones from kwargs
                final_labels = labels.copy()

                # Extract known label keys from function kwargs
                for label_key in ['provider', 'model', 'database', 'method', 'endpoint']:
                    if label_key in kwargs and label_key not in final_labels:
                        final_labels[label_key] = str(kwargs[label_key])

                if final_labels:
                    histogram.labels(**final_labels).observe(duration)
                else:
                    histogram.observe(duration)
        return wrapper
    return decorator


def track_duration_async(histogram: Histogram, **labels):
    """
    Async version of track_duration decorator.

    Args:
        histogram: The Prometheus Histogram to record to.
        **labels: Static labels to apply to the metric.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not METRICS_ENABLED:
                return await func(*args, **kwargs)

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                final_labels = labels.copy()

                for label_key in ['provider', 'model', 'database', 'method', 'endpoint']:
                    if label_key in kwargs and label_key not in final_labels:
                        final_labels[label_key] = str(kwargs[label_key])

                if final_labels:
                    histogram.labels(**final_labels).observe(duration)
                else:
                    histogram.observe(duration)
        return wrapper
    return decorator


def increment_counter(counter: Counter, value: int = 1, **labels):
    """
    Safely increment a counter with labels.

    Args:
        counter: The Prometheus Counter to increment.
        value: Amount to increment by (default: 1).
        **labels: Labels to apply.
    """
    if not METRICS_ENABLED:
        return

    if labels:
        counter.labels(**labels).inc(value)
    else:
        counter.inc(value)


def set_gauge(gauge: Gauge, value: float, **labels):
    """
    Safely set a gauge value with labels.

    Args:
        gauge: The Prometheus Gauge to set.
        value: Value to set.
        **labels: Labels to apply.
    """
    if not METRICS_ENABLED:
        return

    if labels:
        gauge.labels(**labels).set(value)
    else:
        gauge.set(value)


def observe_histogram(histogram: Histogram, value: float, **labels):
    """
    Safely observe a value in a histogram.

    Args:
        histogram: The Prometheus Histogram to observe.
        value: Value to observe.
        **labels: Labels to apply.
    """
    if not METRICS_ENABLED:
        return

    if labels:
        histogram.labels(**labels).observe(value)
    else:
        histogram.observe(value)


# =============================================================================
# DISK USAGE UPDATE FUNCTION
# =============================================================================

def update_disk_usage_metrics():
    """
    Update disk usage gauges for monitored folders.
    Call this periodically (e.g., every minute via scheduler).
    """
    if not METRICS_ENABLED:
        return

    import os
    from app.core.config import UPLOAD_DIR, LOG_DIR

    # Define folders to monitor
    folders = {
        'uploads': UPLOAD_DIR,
        'logs': LOG_DIR,
        'data': os.path.join(os.path.dirname(UPLOAD_DIR), 'data')
    }

    for folder_name, folder_path in folders.items():
        if os.path.exists(folder_path):
            total_size = 0
            try:
                for dirpath, dirnames, filenames in os.walk(folder_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            pass
                disk_usage_bytes.labels(folder=folder_name).set(total_size)
            except Exception:
                pass


def update_session_metrics():
    """
    Update session-related gauges from database.
    Call this periodically.
    """
    if not METRICS_ENABLED:
        return

    try:
        from app.database.session import SessionLocal
        from app.models.pipeline_session import PipelineSession, SessionStatus
        from datetime import datetime

        db = SessionLocal()
        try:
            # Count active sessions by status
            active_statuses = [
                SessionStatus.EXTRACTING,
                SessionStatus.CHUNKING,
                SessionStatus.EMBEDDING,
                SessionStatus.UPLOADING
            ]

            for status in active_statuses:
                count = db.query(PipelineSession).filter(
                    PipelineSession.status == status
                ).count()
                active_sessions.labels(status=status.value).set(count)

            # Count pending cleanup
            now = datetime.utcnow()
            pending = db.query(PipelineSession).filter(
                PipelineSession.expires_at <= now,
                PipelineSession.cleaned_up == False,
                PipelineSession.expires_at.isnot(None)
            ).count()
            pending_cleanup_sessions.set(pending)

        finally:
            db.close()
    except Exception:
        pass  # Don't fail if DB is unavailable
