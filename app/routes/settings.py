"""
Settings Routes
===============

This module manages application-level settings and credentials. It allows users
(typically admins or authorized users) to view and update API keys and configuration
variables stored in the `.env` file.

Key Features:
- Credential Management: Retrieve and save API keys (OpenAI, Pinecone, etc.).
- Environment Configuration: Interface for modifying the `.env` file safely.
"""
import os
import logging
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from app.core.config import RAGPY_DIR

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/get_credentials")
async def get_credentials():
    """
    Get credentials from ragpy/.env for the settings form.
    """
    env_path = os.path.join(RAGPY_DIR, ".env")
    env_path = os.path.abspath(env_path)
    
    logger.info(f"Attempting to read .env file for get_credentials at: {env_path}")
    if not os.path.exists(env_path):
        logger.error(f".env file not found at: {env_path}")
        credential_keys_on_missing = [
            "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
            "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
            "PINECONE_API_KEY", "PINECONE_ENV",
            "WEAVIATE_API_KEY", "WEAVIATE_URL", "QDRANT_API_KEY", "QDRANT_URL",
            "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID"
        ]
        empty_credentials = {k: "" for k in credential_keys_on_missing}
        logger.info("Returning empty credentials as .env file was not found.")
        return JSONResponse(status_code=200, content=empty_credentials)

    # Read existing .env
    env_vars = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        logger.info(f"Read {len(env_vars)} environment variables from {env_path}")
    except Exception as e:
        logger.error(f"Error reading .env file: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to read credentials: {str(e)}"})

    # Return just the credentials keys that we need for the form
    credential_keys = [
        "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
        "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
        "PINECONE_API_KEY", "PINECONE_ENV",
        "WEAVIATE_API_KEY", "WEAVIATE_URL",
        "QDRANT_API_KEY", "QDRANT_URL",
        "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID"
    ]
    
    credentials = {k: env_vars.get(k, "") for k in credential_keys}
    
    return credentials

@router.post("/save_credentials")
async def save_credentials(
    data: dict = Body(...)
):
    """
    Save credentials for OpenAI, OpenRouter, Mistral, Pinecone, Weaviate, Qdrant to ragpy/.env.
    """
    env_path = os.path.join(RAGPY_DIR, ".env")
    env_path = os.path.abspath(env_path)
    logger.info(f"Attempting to save credentials to .env file at: {env_path}")

    # Read existing .env to preserve other keys
    env_vars = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip()
        except Exception as e:
            logger.error(f"Error reading existing .env file: {str(e)}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"Failed to read existing credentials: {str(e)}"})

    # Update with new values
    # Only update keys that are present in the request data and are valid credential keys
    valid_keys = [
        "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
        "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
        "PINECONE_API_KEY", "PINECONE_ENV",
        "WEAVIATE_API_KEY", "WEAVIATE_URL",
        "QDRANT_API_KEY", "QDRANT_URL",
        "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID"
    ]

    updated_count = 0
    for key in valid_keys:
        if key in data:
            env_vars[key] = str(data[key]).strip()
            updated_count += 1
    
    logger.info(f"Updating {updated_count} credential keys.")

    # Write back to .env
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
        logger.info(f"Successfully saved credentials to {env_path}")
        return JSONResponse({"status": "success", "message": "Credentials saved successfully."})
    except Exception as e:
        logger.error(f"Error writing to .env file: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to save credentials: {str(e)}"})
