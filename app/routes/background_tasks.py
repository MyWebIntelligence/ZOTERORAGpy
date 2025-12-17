"""
Background Tasks Routes
=======================

This module provides API endpoints for managing and monitoring background tasks.
Tasks persist in the database, allowing users to disconnect and reconnect while
tasks continue running on the server.

Key Features:
- List active and recent tasks for the current user.
- Get detailed status of a specific task.
- Stream real-time progress updates via Server-Sent Events (SSE).
- Cancel running tasks.
- Clean up stale tasks (admin only).

Endpoints:
- GET /api/tasks/active: List all active (pending/running) tasks.
- GET /api/tasks/recent: List recent tasks (all statuses).
- GET /api/tasks/{task_id}: Get task status.
- GET /api/tasks/{task_id}/stream: SSE stream for real-time progress.
- POST /api/tasks/{task_id}/cancel: Cancel a running task.
- POST /api/tasks/cleanup: Clean up stale tasks (admin only).
"""
import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.middleware.auth import get_current_active_user
from app.models.user import User
from app.models.background_task import BackgroundTask, TaskStatus, TaskType
from app.services.background_task_manager import background_task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["Background Tasks"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class TaskProgress(BaseModel):
    """Progress information for a task."""
    current: int
    total: int
    percent: float
    message: str


class TaskResponse(BaseModel):
    """Response schema for a single task."""
    id: int
    task_type: str
    status: str
    user_id: int
    session_folder: str
    project_id: Optional[int]
    progress: TaskProgress
    logs: List[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    result_file: Optional[str]

    class Config:
        from_attributes = True


class TaskListResponse(BaseModel):
    """Response schema for a list of tasks."""
    tasks: List[TaskResponse]
    count: int


class CancelResponse(BaseModel):
    """Response schema for task cancellation."""
    success: bool
    message: str


class CleanupResponse(BaseModel):
    """Response schema for stale task cleanup."""
    cleaned_count: int
    message: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/active", response_model=TaskListResponse)
async def get_active_tasks(
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all active (pending/running) tasks for the current user.

    Args:
        project_id: Optional project ID to filter tasks.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        List of active tasks with their status and progress.
    """
    tasks = await background_task_manager.get_active_tasks(
        user_id=current_user.id,
        db=db,
        project_id=project_id
    )
    return TaskListResponse(tasks=tasks, count=len(tasks))


@router.get("/recent", response_model=TaskListResponse)
async def get_recent_tasks(
    limit: int = Query(10, ge=1, le=50, description="Maximum tasks to return"),
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get recent tasks for the current user (all statuses).

    Args:
        limit: Maximum number of tasks to return (1-50).
        project_id: Optional project ID to filter tasks.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        List of recent tasks with their status and progress.
    """
    tasks = await background_task_manager.get_recent_tasks(
        user_id=current_user.id,
        db=db,
        limit=limit,
        project_id=project_id
    )
    return TaskListResponse(tasks=tasks, count=len(tasks))


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the current status of a specific task.

    Args:
        task_id: ID of the task to query.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        Task status including progress, logs, and timing information.

    Raises:
        HTTPException 404: Task not found.
        HTTPException 403: User does not own this task.
    """
    task = await background_task_manager.get_task(task_id, db)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    # Verify ownership (unless admin)
    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this task"
        )

    return task.to_dict()


@router.get("/{task_id}/stream")
async def stream_task_progress(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Stream task progress updates via Server-Sent Events (SSE).

    This endpoint allows clients to receive real-time progress updates.
    The connection can be re-established at any time to resume monitoring.

    Args:
        task_id: ID of the task to monitor.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        SSE stream with progress updates, heartbeats, and completion events.

    Raises:
        HTTPException 404: Task not found.
        HTTPException 403: User does not own this task.
    """
    # Verify task exists and user has access
    task = await background_task_manager.get_task(task_id, db)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this task"
        )

    async def generate_events():
        """
        Generator that yields SSE events for task progress.

        Polls the database every 2 seconds and sends updates when progress changes.
        Sends heartbeats to keep the connection alive.
        """
        last_progress = -1
        last_logs_count = 0
        poll_interval = 2  # seconds

        while True:
            try:
                # Get fresh task status from DB
                task_status = await background_task_manager.get_task_status(task_id, db)

                if not task_status:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Task not found'})}\n\n"
                    break

                current_progress = task_status["progress"]["current"]
                current_logs = task_status.get("logs", [])
                current_logs_count = len(current_logs)

                # Send progress update if changed
                if current_progress != last_progress:
                    last_progress = current_progress
                    yield f"data: {json.dumps({'type': 'progress', **task_status['progress']})}\n\n"

                # Send new log lines
                if current_logs_count > last_logs_count:
                    new_logs = current_logs[last_logs_count:]
                    for log_line in new_logs:
                        yield f"data: {json.dumps({'type': 'log', 'message': log_line})}\n\n"
                    last_logs_count = current_logs_count

                # Check for task completion
                task_status_value = task_status["status"]
                if task_status_value in ["completed", "failed", "cancelled"]:
                    final_event = {
                        "type": task_status_value,
                        "message": task_status.get("error_message") or "Task finished",
                        "result_file": task_status.get("result_file")
                    }
                    yield f"data: {json.dumps(final_event)}\n\n"
                    break

                # Heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.exception(f"Error in SSE stream for task {task_id}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/{task_id}/cancel", response_model=CancelResponse)
async def cancel_task(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Cancel a running or pending task.

    Attempts to gracefully stop the task by:
    1. Cancelling the asyncio task.
    2. Sending SIGTERM to the subprocess (if PID is known).

    Args:
        task_id: ID of the task to cancel.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        Success status and message.

    Raises:
        HTTPException 404: Task not found.
        HTTPException 403: User does not own this task.
        HTTPException 400: Task cannot be cancelled (not running).
    """
    task = await background_task_manager.get_task(task_id, db)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this task"
        )

    success = await background_task_manager.cancel_task(task_id, db)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task cannot be cancelled (may not be running)"
        )

    logger.info(f"Task {task_id} cancelled by user {current_user.id}")

    return CancelResponse(
        success=True,
        message=f"Task {task_id} has been cancelled"
    )


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_stale_tasks(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Clean up stale tasks (admin only).

    Finds tasks marked as RUNNING but with no active process and marks
    them as FAILED. This handles cases where tasks crashed without
    updating their status.

    Args:
        current_user: Authenticated admin user.
        db: Database session.

    Returns:
        Count of cleaned up tasks.

    Raises:
        HTTPException 403: User is not an admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    cleaned_count = await background_task_manager.cleanup_stale_tasks(db)

    logger.info(f"Admin {current_user.id} cleaned up {cleaned_count} stale tasks")

    return CleanupResponse(
        cleaned_count=cleaned_count,
        message=f"Cleaned up {cleaned_count} stale task(s)"
    )
