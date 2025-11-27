"""
Celery Task Routes
==================

This module provides FastAPI endpoints for managing Celery tasks.
It supports a dual-mode architecture where processing can be done via
either subprocess (legacy) or Celery (production).

Features:
    - Task submission endpoints for all pipeline stages
    - Task status polling endpoint
    - Task cancellation endpoint
    - Automatic fallback to subprocess when Celery unavailable

Environment Variables:
    ENABLE_CELERY: Set to 'true' to enable Celery mode (default: false)

Endpoints:
    POST /api/celery/process_dataframe - Submit extraction task
    POST /api/celery/initial_chunking - Submit chunking task
    POST /api/celery/dense_embedding - Submit dense embedding task
    POST /api/celery/sparse_embedding - Submit sparse embedding task
    POST /api/celery/upload_vectordb - Submit vector DB upload task
    GET /api/celery/task/{task_id}/status - Get task status
    POST /api/celery/task/{task_id}/cancel - Cancel task
"""
import os
import logging
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from app.celery_app import (
    CELERY_ENABLED,
    is_celery_available,
    get_task_status,
    revoke_task
)
from app.core.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/celery", tags=["celery"])


def _check_celery_available() -> None:
    """
    Check if Celery is enabled and available.

    Raises:
        HTTPException: If Celery is not available
    """
    if not CELERY_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Celery is not enabled",
                "message": "Set ENABLE_CELERY=true in .env to enable Celery mode"
            }
        )

    if not is_celery_available():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Celery broker unavailable",
                "message": "Redis broker is not reachable. Check if Redis is running."
            }
        )


@router.post("/process_dataframe")
async def submit_extraction_task(
    path: str = Form(...),
    session_id: int = Form(...)
):
    """
    Submit PDF extraction task to Celery queue.

    This endpoint queues an extraction task for processing Zotero JSON
    and associated PDFs. Returns immediately with a task ID for polling.

    Args:
        path: Relative path to session directory under uploads/
        session_id: Database session ID for tracking

    Returns:
        JSONResponse: {
            "task_id": str,
            "status": "queued",
            "message": str
        }
    """
    _check_celery_available()

    # Resolve paths
    absolute_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))

    if not os.path.isdir(absolute_path):
        raise HTTPException(
            status_code=400,
            detail=f"Session directory not found: {path}"
        )

    # Find JSON file
    excluded_prefixes = ('output_', 'output.', 'generated_')
    try:
        json_files = [
            f for f in os.listdir(absolute_path)
            if f.lower().endswith('.json') and not f.startswith(excluded_prefixes)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list directory: {e}")

    if not json_files:
        raise HTTPException(
            status_code=400,
            detail="No Zotero JSON file found (excluding pipeline-generated files)"
        )

    json_path = os.path.join(absolute_path, json_files[0])
    output_path = os.path.join(absolute_path, 'output.csv')

    # Submit task
    from app.tasks.extraction import process_dataframe_task

    task = process_dataframe_task.delay(
        json_path=json_path,
        base_dir=absolute_path,
        output_path=output_path,
        session_id=session_id
    )

    logger.info(f"Submitted extraction task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": "Extraction task queued for processing"
    })


@router.post("/initial_chunking")
async def submit_chunking_task(
    path: str = Form(...),
    session_id: int = Form(...),
    model: str = Form("gpt-4o-mini")
):
    """
    Submit text chunking task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID
        model: LLM model for GPT recoding (default: gpt-4o-mini)

    Returns:
        JSONResponse with task_id
    """
    _check_celery_available()

    absolute_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_csv = os.path.join(absolute_path, 'output.csv')

    if not os.path.exists(input_csv):
        raise HTTPException(
            status_code=400,
            detail="output.csv not found. Complete extraction step first."
        )

    from app.tasks.chunking import initial_chunking_task

    task = initial_chunking_task.delay(
        input_csv=input_csv,
        output_dir=absolute_path,
        session_id=session_id,
        model=model
    )

    logger.info(f"Submitted chunking task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": f"Chunking task queued (model: {model})"
    })


@router.post("/dense_embedding")
async def submit_dense_embedding_task(
    path: str = Form(...),
    session_id: int = Form(...)
):
    """
    Submit dense embedding generation task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID

    Returns:
        JSONResponse with task_id
    """
    _check_celery_available()

    absolute_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_file = os.path.join(absolute_path, 'output_chunks.json')

    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=400,
            detail="output_chunks.json not found. Complete chunking step first."
        )

    from app.tasks.embeddings import dense_embedding_task

    task = dense_embedding_task.delay(
        input_file=input_file,
        output_dir=absolute_path,
        session_id=session_id
    )

    logger.info(f"Submitted dense embedding task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": "Dense embedding task queued"
    })


@router.post("/sparse_embedding")
async def submit_sparse_embedding_task(
    path: str = Form(...),
    session_id: int = Form(...)
):
    """
    Submit sparse embedding generation task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID

    Returns:
        JSONResponse with task_id
    """
    _check_celery_available()

    absolute_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_file = os.path.join(absolute_path, 'output_chunks_with_embeddings.json')

    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=400,
            detail="output_chunks_with_embeddings.json not found. Complete dense embedding first."
        )

    from app.tasks.embeddings import sparse_embedding_task

    task = sparse_embedding_task.delay(
        input_file=input_file,
        output_dir=absolute_path,
        session_id=session_id
    )

    logger.info(f"Submitted sparse embedding task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": "Sparse embedding task queued"
    })


@router.post("/upload_vectordb")
async def submit_vectordb_upload_task(
    path: str = Form(...),
    session_id: int = Form(...),
    db_choice: str = Form(...),
    pinecone_index_name: Optional[str] = Form(None),
    pinecone_namespace: Optional[str] = Form(None),
    weaviate_class_name: Optional[str] = Form(None),
    weaviate_tenant_name: Optional[str] = Form(None),
    qdrant_collection_name: Optional[str] = Form(None)
):
    """
    Submit vector database upload task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID
        db_choice: Target database ('pinecone', 'weaviate', 'qdrant')
        pinecone_index_name: Pinecone index name (if applicable)
        pinecone_namespace: Pinecone namespace (optional)
        weaviate_class_name: Weaviate class name (if applicable)
        weaviate_tenant_name: Weaviate tenant name (optional)
        qdrant_collection_name: Qdrant collection name (if applicable)

    Returns:
        JSONResponse with task_id
    """
    _check_celery_available()

    absolute_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_file = os.path.join(absolute_path, 'output_chunks_with_embeddings_sparse.json')

    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=400,
            detail="Embeddings file not found. Complete embedding generation first."
        )

    # Validate db_choice
    if db_choice not in ('pinecone', 'weaviate', 'qdrant'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid db_choice: {db_choice}. Must be pinecone, weaviate, or qdrant."
        )

    from app.tasks.vectordb import upload_to_vectordb_task

    task = upload_to_vectordb_task.delay(
        input_file=input_file,
        session_id=session_id,
        db_choice=db_choice,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        weaviate_class_name=weaviate_class_name,
        weaviate_tenant_name=weaviate_tenant_name,
        qdrant_collection_name=qdrant_collection_name
    )

    logger.info(f"Submitted vectordb upload task {task.id} to {db_choice}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": f"Upload task queued for {db_choice}"
    })


@router.get("/task/{task_id}/status")
async def get_celery_task_status(task_id: str):
    """
    Get the status of a Celery task.

    This endpoint provides real-time status updates for queued tasks.
    Poll this endpoint to track task progress.

    Args:
        task_id: The Celery task ID returned when task was submitted

    Returns:
        JSONResponse: {
            "state": str,  # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED
            "current": int,  # Current progress (if PROGRESS)
            "total": int,  # Total items (if PROGRESS)
            "percent": int,  # Progress percentage (if PROGRESS)
            "item": str,  # Current item being processed (if PROGRESS)
            "status": str,  # Human-readable status message
            "result": dict,  # Task result (if SUCCESS)
            "error": str  # Error message (if FAILURE)
        }
    """
    status = get_task_status(task_id)
    return JSONResponse(status)


@router.post("/task/{task_id}/cancel")
async def cancel_celery_task(
    task_id: str,
    terminate: bool = Query(False, description="Force terminate running task")
):
    """
    Cancel a Celery task.

    Args:
        task_id: The Celery task ID to cancel
        terminate: If True, forcefully terminate running task (SIGTERM)

    Returns:
        JSONResponse: {
            "success": bool,
            "message": str
        }
    """
    result = revoke_task(task_id, terminate=terminate)

    if result.get('success'):
        logger.info(f"Task {task_id} cancelled (terminate={terminate})")
        return JSONResponse(result)
    else:
        raise HTTPException(status_code=500, detail=result.get('error'))


@router.get("/status")
async def get_celery_status():
    """
    Get overall Celery system status.

    Returns:
        JSONResponse: {
            "enabled": bool,
            "available": bool,
            "broker_url": str (masked)
        }
    """
    from app.celery_app import CELERY_BROKER_URL

    # Mask sensitive parts of broker URL
    masked_url = CELERY_BROKER_URL
    if '@' in masked_url:
        # Hide password in URL
        parts = masked_url.split('@')
        masked_url = parts[0].split(':')[0] + ':***@' + parts[1]

    return JSONResponse({
        "enabled": CELERY_ENABLED,
        "available": is_celery_available() if CELERY_ENABLED else False,
        "broker_url": masked_url
    })


@router.get("/workers")
async def get_celery_workers():
    """
    Get information about active Celery workers.

    Returns:
        JSONResponse with worker information
    """
    _check_celery_available()

    try:
        from app.celery_app import celery_app

        # Get active workers
        inspector = celery_app.control.inspect(timeout=2)
        active = inspector.active() or {}
        stats = inspector.stats() or {}

        workers = []
        for worker_name, worker_stats in stats.items():
            workers.append({
                "name": worker_name,
                "active_tasks": len(active.get(worker_name, [])),
                "pool": worker_stats.get('pool', {}).get('implementation', 'unknown'),
                "concurrency": worker_stats.get('pool', {}).get('max-concurrency', 0),
                "processed": worker_stats.get('total', {})
            })

        return JSONResponse({
            "workers": workers,
            "total_workers": len(workers)
        })

    except Exception as e:
        logger.error(f"Failed to get worker info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
