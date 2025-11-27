"""
Background Scheduler Module
===========================

This module provides a centralized scheduler for background tasks using APScheduler.
It handles periodic cleanup of expired sessions and other maintenance tasks.

Key Features:
- Non-blocking background task execution
- Configurable cleanup intervals via environment variables
- Graceful startup and shutdown
- Thread-safe singleton pattern

Environment Variables:
- CLEANUP_INTERVAL_HOURS: Hours between cleanup runs (default: 6)
- CLEANUP_ENABLED: Enable/disable automatic cleanup (default: true)
"""

import os
import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

logger = logging.getLogger(__name__)

# Configuration from environment
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))
CLEANUP_ENABLED = os.getenv("CLEANUP_ENABLED", "true").lower() in ("true", "1", "yes")


class SessionCleanupScheduler:
    """
    Singleton scheduler for session cleanup tasks.

    Usage:
        from app.core.scheduler import cleanup_scheduler

        # Start the scheduler (typically in FastAPI lifespan)
        cleanup_scheduler.start()

        # Stop the scheduler
        cleanup_scheduler.stop()

        # Manually trigger cleanup
        cleanup_scheduler.run_cleanup_now()
    """

    _instance: Optional['SessionCleanupScheduler'] = None

    def __new__(cls) -> 'SessionCleanupScheduler':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._scheduler: Optional[BackgroundScheduler] = None
        self._initialized = True
        self._running = False
        logger.info("SessionCleanupScheduler initialized")

    def _job_listener(self, event):
        """Log job execution events."""
        if event.exception:
            logger.error(f"Scheduled job failed: {event.exception}")
        else:
            logger.debug(f"Scheduled job completed successfully")

    def _cleanup_job(self):
        """
        The actual cleanup job that runs periodically.
        Imports are done inside to avoid circular imports.
        """
        from app.services.session_cleanup import cleanup_expired_sessions

        logger.info("Running scheduled session cleanup...")
        try:
            result = cleanup_expired_sessions()
            logger.info(f"Cleanup completed: {result.get('message', 'Done')}")
        except Exception as e:
            logger.error(f"Cleanup job failed: {e}")

    def start(self) -> bool:
        """
        Start the background scheduler.

        Returns:
            True if scheduler started, False if already running or disabled.
        """
        if not CLEANUP_ENABLED:
            logger.info("Automatic cleanup is disabled (CLEANUP_ENABLED=false)")
            return False

        if self._running:
            logger.warning("Scheduler is already running")
            return False

        try:
            self._scheduler = BackgroundScheduler(
                job_defaults={
                    'coalesce': True,  # Combine missed runs
                    'max_instances': 1,  # Only one instance at a time
                    'misfire_grace_time': 3600  # 1 hour grace for missed jobs
                }
            )

            # Add listener for job events
            self._scheduler.add_listener(
                self._job_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
            )

            # Schedule the cleanup job
            self._scheduler.add_job(
                self._cleanup_job,
                trigger=IntervalTrigger(hours=CLEANUP_INTERVAL_HOURS),
                id='session_cleanup',
                name='Expired Session Cleanup',
                replace_existing=True
            )

            self._scheduler.start()
            self._running = True
            logger.info(
                f"Scheduler started. Cleanup will run every {CLEANUP_INTERVAL_HOURS} hours"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop the background scheduler gracefully.

        Returns:
            True if stopped successfully, False otherwise.
        """
        if not self._running or not self._scheduler:
            logger.info("Scheduler is not running")
            return False

        try:
            self._scheduler.shutdown(wait=True)
            self._running = False
            self._scheduler = None
            logger.info("Scheduler stopped gracefully")
            return True
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            return False

    def run_cleanup_now(self) -> bool:
        """
        Trigger an immediate cleanup run (in addition to scheduled runs).

        Returns:
            True if job was triggered, False otherwise.
        """
        if not self._scheduler:
            # Run directly without scheduler
            logger.info("Running cleanup directly (scheduler not active)")
            self._cleanup_job()
            return True

        try:
            job = self._scheduler.get_job('session_cleanup')
            if job:
                job.modify(next_run_time=None)  # Run immediately
                self._scheduler.add_job(
                    self._cleanup_job,
                    id='session_cleanup_immediate',
                    replace_existing=True
                )
                logger.info("Immediate cleanup job triggered")
                return True
            else:
                # No job scheduled, run directly
                self._cleanup_job()
                return True
        except Exception as e:
            logger.error(f"Failed to trigger immediate cleanup: {e}")
            return False

    def get_status(self) -> dict:
        """
        Get current scheduler status.

        Returns:
            Dict with scheduler status information.
        """
        status = {
            "enabled": CLEANUP_ENABLED,
            "running": self._running,
            "interval_hours": CLEANUP_INTERVAL_HOURS
        }

        if self._scheduler and self._running:
            job = self._scheduler.get_job('session_cleanup')
            if job:
                status["next_run"] = job.next_run_time.isoformat() if job.next_run_time else None

        return status


# Singleton instance
cleanup_scheduler = SessionCleanupScheduler()
