import logging
import os
import sys
import csv

"""
Main Application Module
=======================

This module initializes the FastAPI application, configures logging, sets up
database connections, and registers all API routers. It serves as the entry
point for the RAGpy application.

Key Features:
- FastAPI application initialization with lifespan management.
- Logging configuration (console and file handlers).
- Database initialization on startup.
- Registration of all application routers (Auth, Users, Admin, Projects, Pipeline, etc.).
- Health check endpoints for monitoring.
- Static file mounting and template configuration.
"""

# Increase CSV field size limit
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

# Import Zotero utilities (keep if used by other modules or future use, though not used in main directly anymore)
from app.utils import zotero_client, llm_note_generator, zotero_parser

# Import authentication and database modules
from app.database.init_db import init_database
from app.database.session import get_db

# Import routers
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.admin import router as admin_router
from app.routes.projects import router as projects_router
from app.routes.pages import router as pages_router
from app.routes.pipeline import router as pipeline_router
# New routers
from app.routes.ingestion import router as ingestion_router
from app.routes.processing import router as processing_router
from app.routes.settings import router as settings_router

from app.middleware.auth import get_optional_user, get_current_active_user
from app.core.credentials import get_credential_or_env, get_user_credentials
from app.core.config import APP_DIR, RAGPY_DIR, LOG_DIR, UPLOAD_DIR, STATIC_DIR, TEMPLATES_DIR

# --- Logging Configuration ---
# Ensure LOG_DIR exists (config.py does it, but safe to keep or rely on config)
os.makedirs(LOG_DIR, exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# File Handler
app_log_file = os.path.join(LOG_DIR, "app.log")
file_handler = RotatingFileHandler(app_log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
# Get root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

logger.info(f"APP_DIR initialized to: {APP_DIR}")
logger.info(f"RAGPY_DIR initialized to: {RAGPY_DIR}")
logger.info(f"LOG_DIR initialized to: {LOG_DIR}")
logger.info(f"UPLOAD_DIR initialized to: {UPLOAD_DIR}")
logger.info(f"Application log file: {app_log_file}")

# --- Database Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.

    This async context manager is used by FastAPI to handle tasks that need to
    be executed when the application starts and before it shuts down.

    - On startup: It initializes the database by calling `init_database()`.
    - On shutdown: It logs a shutdown message.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Startup: Initialize database
    logger.info("Initializing database...")
    init_database()
    logger.info("Database initialized successfully")
    yield
    # Shutdown: cleanup if needed
    logger.info("Application shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MyDoc Intelligence",
    description="RAGpy - Pipeline de traitement de documents académiques",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include routers
# HTML pages (must be before API routers with same paths if any, though usually they are distinct)
app.include_router(pages_router) 
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(admin_router)
app.include_router(projects_router)
app.include_router(pipeline_router)
# Include new routers
app.include_router(ingestion_router)
app.include_router(processing_router)
app.include_router(settings_router)

# Health check endpoint pour Docker healthcheck
@app.get("/health")
async def health_check():
    """
    Provides a health check endpoint for monitoring and Docker health checks.

    This endpoint can be used by external services to verify that the application
    is running and responsive. It returns a simple JSON response indicating a
    healthy status.

    Returns:
        dict: A dictionary with a "status" key set to "healthy".
    """
    return {"status": "healthy"}


@app.get("/health/detailed")
async def health_detailed():
    """
    Health check détaillé avec métriques système.

    Retourne des informations sur:
    - CPU et mémoire
    - Espace disque
    - Connectivité base de données
    - Sessions actives
    - Configuration workers

    Returns:
        JSONResponse: Métriques système avec status code 200 (healthy) ou 503 (degraded)
    """
    import psutil
    import sqlite3
    from datetime import datetime, timezone
    from fastapi.responses import JSONResponse
    from sqlalchemy import or_

    # CPU et RAM
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()

    # Database connectivity
    db_status = "unknown"
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ragpy.db")
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Disk space
    try:
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / 1e9
        disk_percent = disk.percent
    except Exception:
        disk_free_gb = 0
        disk_percent = 100

    # Active sessions count (sessions in processing states)
    active_sessions = 0
    try:
        from app.models.pipeline_session import PipelineSession, SessionStatus
        db = next(get_db())
        # Count sessions in active processing states
        active_statuses = [
            SessionStatus.EXTRACTING,
            SessionStatus.CHUNKING,
            SessionStatus.EMBEDDING,
            SessionStatus.UPLOADING
        ]
        active_sessions = db.query(PipelineSession).filter(
            or_(*[PipelineSession.status == s for s in active_statuses])
        ).count()
        db.close()
    except Exception:
        pass

    # Build health data
    is_healthy = db_status == "healthy" and cpu_percent < 90 and mem.percent < 90

    health_data = {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "cpu_percent": round(cpu_percent, 1),
            "memory_percent": round(mem.percent, 1),
            "memory_available_gb": round(mem.available / 1e9, 2),
            "disk_free_gb": round(disk_free_gb, 2),
            "disk_percent": round(disk_percent, 1)
        },
        "database": {
            "status": db_status,
            "active_sessions": active_sessions
        },
        "workers": {
            "uvicorn_workers": int(os.getenv('UVICORN_WORKERS', 1)),
            "max_workers_configured": int(os.getenv('DEFAULT_MAX_WORKERS', os.cpu_count() or 4)),
            "max_concurrent_llm": int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 5))
        }
    }

    status_code = 200 if is_healthy else 503
    return JSONResponse(content=health_data, status_code=status_code)


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """
    Serves the main homepage of the application.

    This endpoint determines whether a user is logged in by checking for an
    access token in the request cookies.

    - If the user is authenticated, it renders the main projects page (`user/projects.html`).
    - If the user is not authenticated, it displays the login page (`auth/login.html`).

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        TemplateResponse: An HTML response rendering either the projects page
                          or the login page, based on authentication status.
    """
    # Get optional user from cookie
    db = next(get_db())
    try:
        token = request.cookies.get("access_token")
        current_user = None

        if token:
            from app.core.security import decode_token
            from app.models.user import User

            payload = decode_token(token)
            if payload and payload.get("type") == "access":
                try:
                    user_id = int(payload.get("sub"))
                    current_user = db.query(User).filter(User.id == user_id).first()
                except (ValueError, TypeError):
                    pass

        # Return appropriate template based on login status
        return templates.TemplateResponse(
            "user/projects.html" if current_user else "auth/login.html",
            {
                "request": request,
                "current_user": current_user,
                "show_sidebar": False
            }
        )
    finally:
        db.close()
