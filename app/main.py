import logging
import os
import sys
import csv

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
    """Application lifespan handler for startup/shutdown events"""
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

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Main application page - shows projects if logged in, or login prompt"""
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
