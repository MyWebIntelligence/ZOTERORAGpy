"""
Core Configuration Constants
============================

This module defines constant paths and directories used throughout the application.
It calculates absolute paths based on the file's location to ensure portability.

Defined Paths:
- `APP_DIR`: The main application directory (`app/`).
- `RAGPY_DIR`: The project root directory.
- `LOG_DIR`: Directory for application logs.
- `UPLOAD_DIR`: Directory for user uploads and session data.
- `STATIC_DIR`: Directory for static assets (CSS, JS, images).
- `TEMPLATES_DIR`: Directory for Jinja2 templates.
"""
import os

# --- Path Definitions ---
# app/core/config.py -> app/core -> app
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAGPY_DIR = os.path.dirname(APP_DIR)
LOG_DIR = os.path.join(RAGPY_DIR, "logs")
UPLOAD_DIR = os.path.join(RAGPY_DIR, "uploads")
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
