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
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from sqlalchemy.orm import Session

from app.models.pipeline_session import PipelineSession, SessionStatus
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

    Args:
        session_folder: The session folder name (e.g., 'uuid_filename').

    Returns:
        Tuple of (success: bool, message: str).
    """
    uploads_dir = get_uploads_dir()
    session_path = os.path.join(uploads_dir, session_folder)

    if not os.path.exists(session_path):
        return True, f"Session folder already deleted: {session_folder}"

    try:
        # Get folder size before deletion (for logging)
        total_size = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(session_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
                file_count += 1

        # Delete the folder and all contents
        shutil.rmtree(session_path)

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


def cleanup_orphaned_folders() -> Dict:
    """
    Clean up upload folders that exist on disk but have no database record.

    This handles cases where sessions were deleted from DB but files remain,
    or where uploads failed before creating a database record.

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
        # Get all session folders from database
        db_folders = set(
            s.session_folder for s in db.query(PipelineSession.session_folder).all()
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
            success, message = delete_session_files(folder)
            results["details"].append({
                "folder": folder,
                "deleted": success,
                "message": message
            })
            if success:
                results["deleted"] += 1
            else:
                results["failed"] += 1

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
