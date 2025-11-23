"""
API routes for RAGpy
"""
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.admin import router as admin_router
from app.routes.projects import router as projects_router

__all__ = ["auth_router", "users_router", "admin_router", "projects_router"]
