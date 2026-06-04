"""
Settings Routes
===============

This module manages application-level settings and credentials. It allows ADMIN users
only to view and update API keys and configuration variables.

Key Features:
- Credential Management: Retrieve and save API keys (OpenAI, Pinecone, etc.) - ADMIN ONLY.
- Environment Configuration: Interface for modifying the `.env` file safely - ADMIN ONLY.
- Vector DB Discovery: List available indexes from Pinecone, Weaviate, Qdrant - AUTHENTICATED.

Security Note:
    All endpoints in this module require authentication.
    These credential endpoints are restricted to the ADMIN role.
    Regular users should use their personal credentials stored in the database
    via the /users/me/credentials endpoint.

Storage Note:
    The pipeline reads credentials from the per-user database store first, with
    `.env` only as an admin fallback (see app/core/credentials.py). To keep the
    admin settings form authoritative, /save_credentials writes BOTH the `.env`
    file and the admin's personal database credentials, and /get_credentials
    returns the effective value (database first, then `.env`).
"""
import os
import logging
from fastapi import APIRouter, Body, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import RAGPY_DIR
from app.core.credentials import (
    get_user_credentials,
    update_user_credentials,
    CREDENTIAL_ENV_MAPPING,
)
from app.database.session import get_db
from app.middleware.auth import require_admin, get_current_active_user
from app.models.user import User

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/get_credentials")
async def get_credentials(
    admin_user: User = Depends(require_admin)
):
    """
    Get the admin's *effective* credentials for the settings form.

    Returns the values the pipeline actually uses: the admin's personal
    credentials (stored encrypted in the database) take precedence, falling back
    to the application's ``.env`` file when a personal value is absent. This
    mirrors the precedence enforced by ``get_credential_or_env`` /
    ``build_subprocess_env`` so the form never shows a value that differs from
    what runs.

    **ADMIN ONLY**: This endpoint exposes sensitive API keys.
    Regular users should use /users/me/credentials for their personal credentials.

    Args:
        admin_user: The authenticated admin user (injected by require_admin).

    Returns:
        JSONResponse: Dictionary of credential key-value pairs.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If user is not an admin.
    """
    logger.info(f"Admin user {admin_user.email} accessing .env credentials")
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

    # Overlay the admin's personal credentials (database) so the form reflects
    # the *effective* value used by the pipeline (database takes precedence over
    # .env, matching get_credential_or_env / build_subprocess_env).
    user_creds = get_user_credentials(admin_user)
    for cred_key, env_key in CREDENTIAL_ENV_MAPPING.items():
        value = user_creds.get(cred_key)
        if value:
            credentials[env_key] = value

    return credentials

@router.post("/save_credentials")
async def save_credentials(
    data: dict = Body(...),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Save credentials for OpenAI, OpenRouter, Mistral, Pinecone, Weaviate, Qdrant.

    Writes to BOTH stores so the saved value is the one the pipeline uses:
      1. The application's ``.env`` file (deployment-wide fallback).
      2. The admin's personal credentials in the database — the store read first
         by ``build_subprocess_env`` / ``get_credential_or_env``. Without this
         mirror, a stale personal credential would silently shadow the ``.env``
         value for admins, so the UI key would never take effect.

    **ADMIN ONLY**. Regular users use PUT /users/me/credentials.

    Args:
        data: Dictionary of credential key-value pairs to save.
        admin_user: The authenticated admin user (injected by require_admin).
        db: Database session (injected) used to persist personal credentials.

    Returns:
        JSONResponse: Status message indicating success or failure.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If user is not an admin.
    """
    logger.info(f"Admin user {admin_user.email} saving .env credentials")
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
    except Exception as e:
        logger.error(f"Error writing to .env file: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to save credentials: {str(e)}"})

    # Mirror the same values into the admin's personal credentials (database).
    # This is the store the pipeline reads first, so it is what actually takes
    # effect for the admin's own runs. An empty value clears the personal
    # credential (handled by update_user_credentials), falling back to .env.
    env_to_cred = {env_key: cred_key for cred_key, env_key in CREDENTIAL_ENV_MAPPING.items()}
    db_updates = {
        env_to_cred[key]: str(data[key]).strip()
        for key in valid_keys
        if key in data and key in env_to_cred
    }
    if db_updates:
        try:
            update_user_credentials(admin_user, db_updates, db)
            logger.info(
                f"Mirrored {len(db_updates)} credential(s) to personal store for {admin_user.email}"
            )
        except Exception as e:
            logger.error(f"Error mirroring credentials to database: {str(e)}", exc_info=True)
            return JSONResponse(status_code=500, content={
                "error": f"Credentials saved to .env but failed to persist personal credentials: {str(e)}"
            })

    return JSONResponse({"status": "success", "message": "Credentials saved successfully."})


@router.get("/api/pinecone/indexes")
async def list_pinecone_indexes(
    api_key: str = Query(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all available Pinecone indexes using the provided or configured API key.

    **AUTHENTICATED**: Requires a valid authenticated user.

    With Pinecone v3+, each index has its own host URL. This endpoint returns
    the list of indexes with their metadata so the frontend can populate a dropdown.

    Args:
        api_key: Optional API key. If not provided, uses user's credentials or PINECONE_API_KEY from env.
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON with list of indexes containing name, dimension, metric, host, and stats.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If account is inactive or not verified.
    """
    # Import credential helpers
    from app.core.credentials import (
        get_credential_or_env,
        get_credential_error_message,
        CredentialMissingError
    )

    # Get API key: parameter > user credentials > environment (admin only)
    pinecone_api_key = api_key or get_credential_or_env(current_user, "pinecone_api_key")

    if not pinecone_api_key:
        logger.warning(f"User {current_user.email} missing Pinecone credentials")
        return JSONResponse(
            status_code=403,
            content={
                "error": get_credential_error_message("pinecone_api_key"),
                "credential_required": "pinecone_api_key",
                "configure_url": "/settings/credentials"
            }
        )

    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=pinecone_api_key)
        indexes_list = list(pc.list_indexes())

        indexes_data = []
        for idx in indexes_list:
            index_info = {
                "name": idx.name,
                "dimension": idx.dimension,
                "metric": idx.metric,
                "host": idx.host,
                "status": getattr(idx.status, "state", "unknown") if hasattr(idx, "status") else "ready"
            }

            # Try to get vector count
            try:
                index = pc.Index(idx.name)
                stats = index.describe_index_stats()
                index_info["vector_count"] = stats.total_vector_count
                index_info["namespaces"] = list(stats.namespaces.keys()) if stats.namespaces else []
            except Exception as stats_error:
                logger.warning(f"Could not get stats for index {idx.name}: {stats_error}")
                index_info["vector_count"] = None
                index_info["namespaces"] = []

            indexes_data.append(index_info)

        logger.info(f"Found {len(indexes_data)} Pinecone indexes")
        return JSONResponse({
            "success": True,
            "indexes": indexes_data,
            "count": len(indexes_data)
        })

    except ImportError:
        logger.error("Pinecone library not installed")
        return JSONResponse(
            status_code=500,
            content={"error": "Pinecone library not installed on server."}
        )
    except Exception as e:
        logger.error(f"Error listing Pinecone indexes: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list Pinecone indexes: {str(e)}"}
        )


@router.get("/api/weaviate/collections")
async def list_weaviate_collections(
    api_key: str = Query(None),
    url: str = Query(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all available Weaviate collections/classes.

    **AUTHENTICATED**: Requires a valid authenticated user.

    Args:
        api_key: Optional API key. If not provided, uses user's credentials or WEAVIATE_API_KEY from env.
        url: Optional Weaviate URL. If not provided, uses user's credentials or WEAVIATE_URL from env.
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON with list of collections.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If account is inactive or not verified.
    """
    from app.core.credentials import get_credential_or_env, get_credential_error_message

    weaviate_api_key = api_key or get_credential_or_env(current_user, "weaviate_api_key")
    weaviate_url = url or get_credential_or_env(current_user, "weaviate_url")

    if not weaviate_url:
        logger.warning(f"User {current_user.email} missing Weaviate URL")
        return JSONResponse(
            status_code=403,
            content={
                "error": get_credential_error_message("weaviate_url"),
                "credential_required": "weaviate_url",
                "configure_url": "/settings/credentials"
            }
        )

    try:
        import weaviate
        from weaviate.classes.init import Auth

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key) if weaviate_api_key else None
        )

        collections = []
        for collection in client.collections.list_all():
            collections.append({
                "name": collection,
                "description": ""
            })

        client.close()

        logger.info(f"Found {len(collections)} Weaviate collections")
        return JSONResponse({
            "success": True,
            "collections": collections,
            "count": len(collections)
        })

    except ImportError:
        return JSONResponse(status_code=500, content={"error": "Weaviate library not installed."})
    except Exception as e:
        logger.error(f"Error listing Weaviate collections: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to list collections: {str(e)}"})


@router.get("/api/qdrant/collections")
async def list_qdrant_collections(
    api_key: str = Query(None),
    url: str = Query(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all available Qdrant collections.

    **AUTHENTICATED**: Requires a valid authenticated user.

    Args:
        api_key: Optional API key. If not provided, uses user's credentials or QDRANT_API_KEY from env.
        url: Optional Qdrant URL. If not provided, uses user's credentials or QDRANT_URL from env.
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON with list of collections.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If account is inactive or not verified.
    """
    from app.core.credentials import get_credential_or_env, get_credential_error_message

    qdrant_api_key = api_key or get_credential_or_env(current_user, "qdrant_api_key")
    qdrant_url = url or get_credential_or_env(current_user, "qdrant_url")

    if not qdrant_url:
        logger.warning(f"User {current_user.email} missing Qdrant URL")
        return JSONResponse(
            status_code=403,
            content={
                "error": get_credential_error_message("qdrant_url"),
                "credential_required": "qdrant_url",
                "configure_url": "/settings/credentials"
            }
        )

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None
        )

        collections_response = client.get_collections()
        collections = []
        for col in collections_response.collections:
            col_info = client.get_collection(col.name)
            collections.append({
                "name": col.name,
                "vector_count": col_info.points_count,
                "dimension": col_info.config.params.vectors.size if hasattr(col_info.config.params.vectors, 'size') else None
            })

        logger.info(f"Found {len(collections)} Qdrant collections")
        return JSONResponse({
            "success": True,
            "collections": collections,
            "count": len(collections)
        })

    except ImportError:
        return JSONResponse(status_code=500, content={"error": "Qdrant library not installed."})
    except Exception as e:
        logger.error(f"Error listing Qdrant collections: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to list collections: {str(e)}"})
