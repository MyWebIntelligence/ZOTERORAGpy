"""
Schemas Package
===============

This package contains Pydantic schemas for request/response validation and data
serialization. These schemas define the structure of data exchanged between the
client and server, ensuring type safety and validation.

Key Schemas:
- `auth`: Authentication and authorization schemas (login, registration, tokens).
- `user`: User management schemas (profiles, updates, credentials).
"""
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    TokenResponse,
    PasswordReset,
    PasswordResetRequest
)
from app.schemas.user import (
    UserResponse,
    UserUpdate,
    UserListResponse
)

__all__ = [
    # Auth
    "UserRegister",
    "UserLogin",
    "TokenResponse",
    "PasswordReset",
    "PasswordResetRequest",
    # User
    "UserResponse",
    "UserUpdate",
    "UserListResponse"
]
