"""
Celery Tasks Package
====================

This package contains Celery task definitions for asynchronous processing
in the RAGpy pipeline. Tasks are organized by processing stage:

Modules:
    extraction: PDF text extraction and OCR tasks
    chunking: Text chunking and GPT recoding tasks
    embeddings: Dense and sparse embedding generation tasks
    vectordb: Vector database upload tasks
    cleanup: Session and file cleanup tasks
    monitoring: System metrics update tasks

Usage:
    from app.tasks.extraction import process_dataframe_task

    # Queue task for async execution
    task = process_dataframe_task.delay(
        json_path="/path/to/export.json",
        base_dir="/path/to/pdfs",
        output_path="/path/to/output.csv",
        session_id=123
    )

    # Check task status
    from app.celery_app import get_task_status
    status = get_task_status(task.id)
"""
from app.tasks.extraction import process_dataframe_task
from app.tasks.chunking import initial_chunking_task
from app.tasks.embeddings import dense_embedding_task, sparse_embedding_task
from app.tasks.vectordb import upload_to_vectordb_task
from app.tasks.cleanup import cleanup_sessions_task, cleanup_orphaned_processes_task
from app.tasks.monitoring import update_metrics_task

__all__ = [
    'process_dataframe_task',
    'initial_chunking_task',
    'dense_embedding_task',
    'sparse_embedding_task',
    'upload_to_vectordb_task',
    'cleanup_sessions_task',
    'cleanup_orphaned_processes_task',
    'update_metrics_task',
]
