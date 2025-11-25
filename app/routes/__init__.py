"""
API Routes Package
==================

This package contains the API route definitions for the RAGpy application.
It aggregates routers from various modules to be included in the main FastAPI application.

Key Routers:
- `auth_router`: Authentication endpoints (login, register, etc.).
- `users_router`: User management endpoints.
- `admin_router`: Administration endpoints.
- `projects_router`: Project management endpoints.
"""
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.admin import router as admin_router
from app.routes.projects import router as projects_router

__all__ = ["auth_router", "users_router", "admin_router", "projects_router"]
