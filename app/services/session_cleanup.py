"""
Session Cleanup Service
=======================

This module provides functionality for cleaning up expired pipeline sessions.
It handles deletion of session files from the uploads directory and updates
the database to mark sessions as cleaned.

Key Features:
- Automatic cleanup of expired sessions based on TTL
- Safe file deletion with error handling
- Database transaction management
- Configurable cleanup parameters via environment variables

Environment Variables:
- SESSION_TTL_HOURS: Default session lifetime in hours (default: 24)
- UPLOADS_DIR: Path to uploads directory (default: ./uploads)
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.upload_safety import remove_session_files
from app.models.pipeline_session import PipelineSession, SessionOwner, SessionStatus
from app.database.session import SessionLocal

logger = logging.getLogger(__name__)

# Default uploads directory (relative to project root)
DEFAULT_UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")


def get_uploads_dir() -> str:
    """
    Get the uploads directory path from environment or default.

    Returns:
        Absolute path to the uploads directory.
    """
    return os.getenv("UPLOADS_DIR", DEFAULT_UPLOADS_DIR)


def get_expired_sessions(db: Session) -> List[PipelineSession]:
    """
    Query all sessions that are expired and not yet cleaned up.

    Args:
        db: SQLAlchemy database session.

    Returns:
        List of expired PipelineSession objects.
    """
    now = datetime.utcnow()
    return db.query(PipelineSession).filter(
        PipelineSession.expires_at <= now,
        PipelineSession.cleaned_up == False,
        PipelineSession.expires_at.isnot(None)
    ).all()


def delete_session_files(session_folder: str) -> Tuple[bool, str]:
    """
    Delete all files associated with a session folder.

    This includes the session directory and any associated archive files
    (ZIP, tar.gz) that were uploaded. Every removal is confined to the
    uploads directory (``app.core.upload_safety.remove_session_files``).

    Args:
        session_folder: The session folder name (e.g., 'uuid_filename' or 'uuid_filename/subdir').

    Returns:
        Tuple of (success: bool, message: str).
    """
    uploads_dir = get_uploads_dir()

    try:
        # Folder, empty parent and uploaded archive, confined to the uploads
        # directory (a stored folder with an unsafe name removes nothing).
        removed = remove_session_files(session_folder, uploads_dir)
        deleted_items = removed["deleted"]
        total_size = removed["total_size"]
        file_count = removed["file_count"]

        if not deleted_items:
            return True, f"Session folder already deleted: {session_folder}"

        size_mb = total_size / (1024 * 1024)
        message = f"Deleted {file_count} files ({size_mb:.2f} MB) from {session_folder}"
        logger.info(message)
        return True, message

    except PermissionError as e:
        message = f"Permission denied deleting {session_folder}: {e}"
        logger.error(message)
        return False, message
    except OSError as e:
        message = f"Error deleting {session_folder}: {e}"
        logger.error(message)
        return False, message


def cleanup_session(session: PipelineSession, db: Session) -> Dict:
    """
    Clean up a single session: delete files and mark as cleaned.

    Args:
        session: The PipelineSession to clean up.
        db: SQLAlchemy database session.

    Returns:
        Dict with cleanup result details.
    """
    result = {
        "session_id": session.id,
        "session_folder": session.session_folder,
        "status": session.status.value if session.status else None,
        "files_deleted": False,
        "marked_cleaned": False,
        "error": None
    }

    # Delete files
    success, message = delete_session_files(session.session_folder)
    result["files_deleted"] = success
    result["delete_message"] = message

    if success:
        # Mark as cleaned up in database
        try:
            session.mark_cleaned_up()
            db.commit()
            result["marked_cleaned"] = True
            logger.info(f"Session {session.id} ({session.session_folder}) marked as cleaned up")
        except Exception as e:
            db.rollback()
            result["error"] = f"Database error: {e}"
            logger.error(f"Failed to mark session {session.id} as cleaned: {e}")
    else:
        result["error"] = message

    return result


def cleanup_expired_sessions() -> Dict:
    """
    Main cleanup function: find and clean all expired sessions.

    This is designed to be called by the scheduler or manually via admin endpoint.

    Returns:
        Dict with summary of cleanup operation.
    """
    logger.info("Starting expired sessions cleanup...")
    start_time = datetime.utcnow()

    results = {
        "started_at": start_time.isoformat(),
        "sessions_found": 0,
        "sessions_cleaned": 0,
        "sessions_failed": 0,
        "total_freed_mb": 0.0,
        "details": []
    }

    db = SessionLocal()
    try:
        # Get all expired sessions
        expired_sessions = get_expired_sessions(db)
        results["sessions_found"] = len(expired_sessions)

        if not expired_sessions:
            logger.info("No expired sessions found")
            results["message"] = "No expired sessions to clean"
            return results

        logger.info(f"Found {len(expired_sessions)} expired session(s) to clean")

        # Clean each session
        for session in expired_sessions:
            cleanup_result = cleanup_session(session, db)
            results["details"].append(cleanup_result)

            if cleanup_result["files_deleted"] and cleanup_result["marked_cleaned"]:
                results["sessions_cleaned"] += 1
            else:
                results["sessions_failed"] += 1

        results["completed_at"] = datetime.utcnow().isoformat()
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        results["elapsed_seconds"] = elapsed
        results["message"] = f"Cleaned {results['sessions_cleaned']}/{results['sessions_found']} sessions in {elapsed:.2f}s"

        logger.info(results["message"])

    except Exception as e:
        logger.error(f"Cleanup operation failed: {e}")
        results["error"] = str(e)
        db.rollback()
    finally:
        db.close()

    return results


_JOB_GROUPS = ("pipeline", "notes", "clustering")


def _last_activity(folder_path: str) -> float:
    """
    Most recent modification time of a folder and everything under it (0 when absent).

    Args:
        folder_path: Absolute folder path.

    Returns:
        A POSIX timestamp.
    """
    try:
        latest = os.path.getmtime(folder_path)
    except OSError:
        return 0.0
    for root, dirs, files in os.walk(folder_path):
        for name in dirs + files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return latest


def _stale_owner_upload(row: SessionOwner, uploads_dir: str) -> bool:
    """
    Tell whether an upload made outside any project may be removed by the orphan cleanup.

    Stale: uploaded more than ``SESSION_TTL_HOURS`` ago (default 24) AND
    untouched for as long (newest modification under its folder): a session
    still worked on between two stages is kept. Running jobs are checked at
    deletion time, under their lock (``_lock_folder_jobs``).

    Args:
        row: The ``SessionOwner`` row.
        uploads_dir: The uploads directory.

    Returns:
        True when the folder may be removed.
    """
    from datetime import timedelta
    from app.models.pipeline_session import DEFAULT_SESSION_TTL_HOURS

    try:
        ttl_hours = int(os.getenv("SESSION_TTL_HOURS", DEFAULT_SESSION_TTL_HOURS))
    except ValueError:
        ttl_hours = DEFAULT_SESSION_TTL_HOURS
    cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
    if row.created_at is None or row.created_at > cutoff:
        return False
    last = _last_activity(os.path.join(uploads_dir, row.session_folder))
    return last < (time.time() - ttl_hours * 3600)


def _lock_folder_jobs(keys) -> Optional[list]:
    """
    Take the job locks (every group) of the given session keys, or none of them.

    Held while a folder is deleted, so a stage cannot start on it in between
    (audit A12). Released by the caller.

    Args:
        keys: Canonical session folders under the folder to delete.

    Returns:
        The tickets to release, or None when a job holds one of the locks.
    """
    from app.services import job_control

    tickets = []
    for key in sorted(keys):
        for group in _JOB_GROUPS:
            ticket, busy = job_control.acquire_job(key, group, None, admission=False)
            if busy is not None:
                for held in tickets:
                    held.release()
                return None
            tickets.append(ticket)
    return tickets


def cleanup_orphaned_folders() -> Dict:
    """
    Clean up upload folders that exist on disk but have no database record.

    This handles cases where sessions were deleted from DB but files remain,
    or where uploads failed before creating a database record. A folder is
    known when a ``PipelineSession`` row, or the ``SessionOwner`` row of a
    recent upload made outside any project, names it or a folder under it
    (audit A02): a ZIP recorded on its single root folder is never taken for
    an orphan. An upload outside any project uploaded and untouched for more
    than ``SESSION_TTL_HOURS`` is removed with its row. Every deletion holds
    the job locks of the folder: a folder with a running job is skipped.

    Returns:
        Dict with cleanup results.
    """
    logger.info("Scanning for orphaned upload folders...")

    uploads_dir = get_uploads_dir()
    results = {
        "orphaned_folders": 0,
        "deleted": 0,
        "failed": 0,
        "details": []
    }

    if not os.path.exists(uploads_dir):
        results["message"] = f"Uploads directory does not exist: {uploads_dir}"
        return results

    db = SessionLocal()
    try:
        # Top-level folders known to the database: project sessions and recent
        # uploads made outside any project (SessionOwner). A ZIP whose single
        # root folder was recorded ("uid_name/Root") keeps its top-level folder
        # "uid_name". An upload outside any project older than SESSION_TTL_HOURS
        # and not locked by a running job is stale: removed with its row.
        recorded = [row.session_folder for row in db.query(PipelineSession.session_folder).all()]
        stale_owners = {}
        for row in db.query(SessionOwner).all():
            top = row.session_folder.split("/")[0] if row.session_folder else ""
            if _stale_owner_upload(row, uploads_dir):
                stale_owners.setdefault(top, []).append(row)
            else:
                recorded.append(row.session_folder)
        db_folders = set(
            folder.split("/")[0] for folder in recorded if folder
        )

        # Get all folders on disk
        disk_folders = set(
            f for f in os.listdir(uploads_dir)
            if os.path.isdir(os.path.join(uploads_dir, f))
        )

        # Find orphaned folders (on disk but not in DB)
        orphaned = disk_folders - db_folders
        results["orphaned_folders"] = len(orphaned)

        for folder in orphaned:
            keys = {folder} | {row.session_folder for row in stale_owners.get(folder, [])}
            tickets = _lock_folder_jobs(keys)
            if tickets is None:
                results["details"].append({
                    "folder": folder,
                    "deleted": False,
                    "message": "Skipped: a job is running on this session"
                })
                results["skipped_busy"] = results.get("skipped_busy", 0) + 1
                continue
            try:
                success, message = delete_session_files(folder)
            finally:
                for ticket in tickets:
                    ticket.release()
            results["details"].append({
                "folder": folder,
                "deleted": success,
                "message": message
            })
            if success:
                results["deleted"] += 1
                for row in stale_owners.get(folder, []):
                    db.delete(row)
            else:
                results["failed"] += 1
        db.commit()

        results["message"] = f"Found {len(orphaned)} orphaned folders, deleted {results['deleted']}"
        logger.info(results["message"])

    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")
        results["error"] = str(e)
    finally:
        db.close()

    return results


def get_cleanup_stats() -> Dict:
    """
    Get statistics about sessions eligible for cleanup.

    Returns:
        Dict with cleanup statistics.
    """
    db = SessionLocal()
    try:
        now = datetime.utcnow()

        # Count sessions by status
        total_sessions = db.query(PipelineSession).count()
        expired_count = db.query(PipelineSession).filter(
            PipelineSession.expires_at <= now,
            PipelineSession.cleaned_up == False,
            PipelineSession.expires_at.isnot(None)
        ).count()
        cleaned_count = db.query(PipelineSession).filter(
            PipelineSession.cleaned_up == True
        ).count()
        no_expiry_count = db.query(PipelineSession).filter(
            PipelineSession.expires_at.is_(None)
        ).count()

        # Calculate disk usage
        uploads_dir = get_uploads_dir()
        total_size = 0
        folder_count = 0

        if os.path.exists(uploads_dir):
            for folder in os.listdir(uploads_dir):
                folder_path = os.path.join(uploads_dir, folder)
                if os.path.isdir(folder_path):
                    folder_count += 1
                    for dirpath, dirnames, filenames in os.walk(folder_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:
                                total_size += os.path.getsize(filepath)
                            except OSError:
                                pass

        return {
            "total_sessions": total_sessions,
            "expired_pending_cleanup": expired_count,
            "already_cleaned": cleaned_count,
            "no_expiry_set": no_expiry_count,
            "disk_folders": folder_count,
            "disk_usage_mb": total_size / (1024 * 1024),
            "uploads_dir": uploads_dir,
            "current_time": now.isoformat()
        }

    finally:
        db.close()
