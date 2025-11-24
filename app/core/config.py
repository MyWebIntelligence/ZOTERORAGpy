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
