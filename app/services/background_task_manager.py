"""
Background Task Manager
=======================

This module provides a service for managing long-running background tasks
with database persistence. Tasks continue running even if the client
disconnects, and users can resume monitoring progress at any time.

Key Features:
- Database persistence for task state, progress, and logs.
- Asyncio-based task execution that survives client disconnection.
- Progress updates written to DB for SSE streaming from any connection.
- PID tracking for task cancellation support.
- Integration with existing ProcessManager for subprocess tracking.

Usage:
    from app.services.background_task_manager import background_task_manager

    # Create and start a task
    task = await background_task_manager.create_task(db, TaskType.FILTER_CITATIONS, user.id, session_folder)
    await background_task_manager.start_task(task.id, my_coroutine(), db)

    # From another connection, check status
    status = await background_task_manager.get_task_status(task.id, db)

    # Cancel if needed
    await background_task_manager.cancel_task(task.id, db)
"""
import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.models.background_task import BackgroundTask, TaskStatus, TaskType
from app.services.process_manager import process_manager

logger = logging.getLogger(__name__)

# Maximum lines of logs to keep in DB
MAX_LOG_LINES = 100


class BackgroundTaskManager:
    """
    Manages background tasks with database persistence.

    This class provides methods to create, start, update, and cancel
    background tasks. Tasks are stored in the database, allowing users
    to disconnect and reconnect without losing progress information.
    """

    def __init__(self):
        """Initialize the background task manager."""
        self._running_tasks: Dict[int, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        logger.info("BackgroundTaskManager initialized")

    async def create_task(
        self,
        db: Session,
        task_type: TaskType,
        user_id: int,
        session_folder: str,
        project_id: Optional[int] = None,
        total: int = 0,
        message: str = "Initializing..."
    ) -> BackgroundTask:
        """
        Create a new background task in the database.

        Args:
            db: Database session.
            task_type: Type of task to create.
            user_id: ID of the user initiating the task.
            session_folder: Path to the session folder.
            project_id: Optional project ID for project-scoped tasks.
            total: Total number of items to process (for progress).
            message: Initial progress message.

        Returns:
            The created BackgroundTask instance.
        """
        task = BackgroundTask(
            task_type=task_type,
            user_id=user_id,
            session_folder=session_folder,
            project_id=project_id,
            progress_total=total,
            progress_message=message,
            status=TaskStatus.PENDING
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        logger.info(f"Created background task {task.id} ({task_type.value}) for user {user_id}")
        return task

    async def start_task(
        self,
        task_id: int,
        coroutine: Coroutine,
        db: Session
    ) -> None:
        """
        Start executing a background task.

        The task runs in the background and continues even if the
        client disconnects. Progress is persisted to the database.

        Args:
            task_id: ID of the task to start.
            coroutine: Async coroutine to execute.
            db: Database session.

        Raises:
            ValueError: If task not found.
        """
        task = db.query(BackgroundTask).filter_by(id=task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if task.status != TaskStatus.PENDING:
            raise ValueError(f"Task {task_id} is not pending (status: {task.status.value})")

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        db.commit()

        # Launch in background (detached from current request)
        async with self._lock:
            asyncio_task = asyncio.create_task(
                self._run_and_track(task_id, coroutine)
            )
            self._running_tasks[task_id] = asyncio_task

        logger.info(f"Started background task {task_id}")

    async def _run_and_track(self, task_id: int, coroutine: Coroutine) -> None:
        """
        Execute the coroutine and update task status when done.

        Args:
            task_id: ID of the task being executed.
            coroutine: The coroutine to execute.
        """
        db = SessionLocal()
        try:
            await coroutine
            await self._update_status(task_id, TaskStatus.COMPLETED, db=db)
        except asyncio.CancelledError:
            await self._update_status(task_id, TaskStatus.CANCELLED, db=db)
            logger.info(f"Task {task_id} was cancelled")
        except Exception as e:
            error_msg = str(e)
            logger.exception(f"Task {task_id} failed: {error_msg}")
            await self._update_status(task_id, TaskStatus.FAILED, error_msg, db=db)
        finally:
            db.close()
            async with self._lock:
                self._running_tasks.pop(task_id, None)

    async def _update_status(
        self,
        task_id: int,
        status: TaskStatus,
        error_message: Optional[str] = None,
        db: Optional[Session] = None
    ) -> None:
        """
        Update task status in the database.

        Args:
            task_id: ID of the task to update.
            status: New status value.
            error_message: Error message if failed.
            db: Optional database session (creates one if not provided).
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            task = db.query(BackgroundTask).filter_by(id=task_id).first()
            if task:
                task.status = status
                task.completed_at = datetime.utcnow()
                if error_message:
                    task.error_message = error_message
                db.commit()
                logger.info(f"Task {task_id} status updated to {status.value}")
        finally:
            if close_db:
                db.close()

    async def update_progress(
        self,
        task_id: int,
        current: int,
        total: Optional[int] = None,
        message: str = "",
        log_line: Optional[str] = None,
        result_file: Optional[str] = None
    ) -> None:
        """
        Update task progress in the database.

        This method is called by task implementations to report progress.
        The progress is persisted to DB so any client can read it.

        Args:
            task_id: ID of the task to update.
            current: Current progress count.
            total: Optional total count (updates if provided).
            message: Progress message to display.
            log_line: Optional log line to append.
            result_file: Optional result file path (set when complete).
        """
        db = SessionLocal()
        try:
            task = db.query(BackgroundTask).filter_by(id=task_id).first()
            if not task:
                return

            task.progress_current = current
            if total is not None:
                task.progress_total = total

            if task.progress_total > 0:
                task.progress_percent = round(
                    (current / task.progress_total) * 100, 2
                )
            else:
                task.progress_percent = 0.0

            task.progress_message = message

            if log_line:
                # Keep last MAX_LOG_LINES lines
                lines = task.logs.split('\n') if task.logs else []
                lines = lines[-(MAX_LOG_LINES - 1):]
                lines.append(log_line)
                task.logs = '\n'.join(lines)

            if result_file:
                task.result_file = result_file

            db.commit()
        finally:
            db.close()

    async def set_pid(self, task_id: int, pid: int) -> None:
        """
        Set the subprocess PID for a task (for cancellation support).

        Args:
            task_id: ID of the task.
            pid: Process ID to track.
        """
        db = SessionLocal()
        try:
            task = db.query(BackgroundTask).filter_by(id=task_id).first()
            if task:
                task.pid = pid
                db.commit()
                logger.debug(f"Task {task_id} PID set to {pid}")
        finally:
            db.close()

    async def get_task(self, task_id: int, db: Session) -> Optional[BackgroundTask]:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task.
            db: Database session.

        Returns:
            BackgroundTask instance or None.
        """
        return db.query(BackgroundTask).filter_by(id=task_id).first()

    async def get_task_status(self, task_id: int, db: Session) -> Optional[Dict]:
        """
        Get task status as a dictionary.

        Args:
            task_id: ID of the task.
            db: Database session.

        Returns:
            Dictionary with task status or None if not found.
        """
        task = db.query(BackgroundTask).filter_by(id=task_id).first()
        if not task:
            return None
        return task.to_dict()

    async def cancel_task(self, task_id: int, db: Session) -> bool:
        """
        Cancel a running task.

        Attempts to gracefully stop the task by:
        1. Cancelling the asyncio task
        2. Sending SIGTERM to the subprocess (if PID known)

        Args:
            task_id: ID of the task to cancel.
            db: Database session.

        Returns:
            True if task was cancelled, False otherwise.
        """
        task = db.query(BackgroundTask).filter_by(id=task_id).first()
        if not task:
            return False

        if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            logger.warning(f"Cannot cancel task {task_id}: status is {task.status.value}")
            return False

        # Cancel asyncio task if running
        async with self._lock:
            asyncio_task = self._running_tasks.get(task_id)
            if asyncio_task:
                asyncio_task.cancel()

        # Kill subprocess if PID known
        if task.pid:
            try:
                os.kill(task.pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to PID {task.pid} for task {task_id}")
            except ProcessLookupError:
                pass  # Process already dead
            except OSError as e:
                logger.warning(f"Failed to kill PID {task.pid}: {e}")

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Cancelled task {task_id}")
        return True

    async def get_active_tasks(
        self,
        user_id: int,
        db: Session,
        project_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get all active tasks for a user.

        Args:
            user_id: ID of the user.
            db: Database session.
            project_id: Optional project ID to filter by.

        Returns:
            List of task status dictionaries.
        """
        query = db.query(BackgroundTask).filter(
            BackgroundTask.user_id == user_id,
            BackgroundTask.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
        )

        if project_id is not None:
            query = query.filter(BackgroundTask.project_id == project_id)

        tasks = query.order_by(BackgroundTask.created_at.desc()).all()
        return [task.to_dict() for task in tasks]

    async def get_recent_tasks(
        self,
        user_id: int,
        db: Session,
        limit: int = 10,
        project_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get recent tasks for a user (active and finished).

        Args:
            user_id: ID of the user.
            db: Database session.
            limit: Maximum number of tasks to return.
            project_id: Optional project ID to filter by.

        Returns:
            List of task status dictionaries.
        """
        query = db.query(BackgroundTask).filter(
            BackgroundTask.user_id == user_id
        )

        if project_id is not None:
            query = query.filter(BackgroundTask.project_id == project_id)

        tasks = query.order_by(
            BackgroundTask.created_at.desc()
        ).limit(limit).all()

        return [task.to_dict() for task in tasks]

    async def cleanup_stale_tasks(self, db: Session) -> int:
        """
        Clean up stale tasks (running tasks with dead processes).

        This should be called periodically to handle tasks that crashed
        without updating their status.

        Args:
            db: Database session.

        Returns:
            Number of tasks cleaned up.
        """
        stale_tasks = db.query(BackgroundTask).filter(
            BackgroundTask.status == TaskStatus.RUNNING
        ).all()

        cleaned = 0
        for task in stale_tasks:
            # Check if asyncio task is still running
            is_asyncio_running = task.id in self._running_tasks

            # Check if subprocess is still running
            is_subprocess_running = False
            if task.pid:
                try:
                    os.kill(task.pid, 0)
                    is_subprocess_running = True
                except OSError:
                    pass

            # If neither is running, mark as failed
            if not is_asyncio_running and not is_subprocess_running:
                task.status = TaskStatus.FAILED
                task.error_message = "Task terminated unexpectedly (stale task cleanup)"
                task.completed_at = datetime.utcnow()
                cleaned += 1
                logger.warning(f"Cleaned up stale task {task.id}")

        if cleaned > 0:
            db.commit()

        return cleaned

    def is_task_running(self, task_id: int) -> bool:
        """
        Check if a task is currently running in this process.

        Args:
            task_id: ID of the task.

        Returns:
            True if task is running, False otherwise.
        """
        return task_id in self._running_tasks


# Singleton instance
background_task_manager = BackgroundTaskManager()
