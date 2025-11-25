"""
Middleware Package
==================

This package contains middleware and dependency injection functions for FastAPI.
It primarily handles authentication and authorization logic.

Key Components:
- `get_current_user`: Dependency to retrieve the authenticated user.
- `get_current_active_user`: Dependency to retrieve an active, verified user.
- `require_admin`: Dependency to enforce admin privileges.
- `require_roles`: Factory for role-based access control.
"""
from app.middleware.auth import (
    get_current_user,
    get_current_active_user,
    get_optional_user,
    require_admin,
    require_roles
)

__all__ = [
    "get_current_user",
    "get_current_active_user",
    "get_optional_user",
    "require_admin",
    "require_roles"
]
