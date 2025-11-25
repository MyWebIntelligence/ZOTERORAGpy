"""
Services Package
================

This package contains business logic and service layers that abstract complex operations
from the API routes. Services are designed to be reusable and testable components.

Key Services:
- `EmailService`: Handles email notifications and transactional emails (Resend).
- `ProcessManager`: Manages background subprocesses with session isolation.
"""
from .email_service import email_service, EmailService

__all__ = ["email_service", "EmailService"]
