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

Input Note:
    /save_credentials refuses (400) any value containing a control or line
    separator character, so a form field can never inject extra ``.env`` lines.
    The vector DB listing routes take no key nor URL from the query string:
    they only use the caller's stored credentials (``.env`` fallback for admins).

Albert (DINUM) Note:
    ``ALBERT_API_KEY`` joins the admin form only when Albert is enabled
    (``albert_enabled()``, see ``_admin_form_env_keys``). It is never returned
    in clear: /get_credentials sends ``••••`` plus its last 4 characters, and
    /save_credentials ignores a value starting with that mask (an empty string
    clears the key). ``GET /api/albert/status`` and ``GET /api/albert/models``
    are served by ``_AlbertGatedRoute``: while Albert is disabled they answer
    exactly like an unknown route (404 ``{"detail": "Not Found"}`` for every
    method, no trailing-slash redirect) and they never appear in the OpenAPI
    schema. Each call makes a single attempt with a short timeout.
    Text placed after a form feed (``\\f``) in an endpoint docstring is left
    out of the OpenAPI description, which keeps it identical to the historical
    one while Albert is disabled.
"""
import os
import time
import logging
import dataclasses
import unicodedata
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from sqlalchemy.orm import Session
from starlette.routing import Match
from starlette.types import Scope

from app.core.config import RAGPY_DIR
from app.core.credentials import (
    get_user_credentials,
    update_user_credentials,
    albert_enabled,
    CREDENTIAL_ENV_MAPPING,
)
from app.database.session import get_db
from app.middleware.auth import require_admin, get_current_active_user
from app.models.user import User

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Environment variables of the admin credentials form, in the historical order
# (formerly three identical literal lists in get_credentials / save_credentials).
_ADMIN_FORM_ENV_KEYS = (
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
    "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
    "PINECONE_API_KEY", "PINECONE_ENV",
    "WEAVIATE_API_KEY", "WEAVIATE_URL",
    "QDRANT_API_KEY", "QDRANT_URL",
    "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID",
)
# Albert key: in the form only when Albert is enabled, placed after Mistral.
ALBERT_ENV_KEY = "ALBERT_API_KEY"
_ALBERT_FORM_ANCHOR = "MISTRAL_API_BASE_URL"
# Server-side mask of the Albert key: "••••" followed by its last 4 characters.
ALBERT_KEY_MASK_PREFIX = "\u2022\u2022\u2022\u2022"
_ALBERT_KEY_VISIBLE_CHARS = 4

ALBERT_CREDENTIAL_KEY = "albert_api_key"
_CONFIGURE_URL = "/settings/credentials"
# /v1/me: an account expiring within this many days triggers a warning.
_ALBERT_EXPIRY_WARNING_DAYS = 30
_SECONDS_PER_DAY = 86400
# /api/albert/*: HTTP timeout (seconds) of the single upstream attempt, so a
# "Tester la connexion" click never holds a worker thread for long.
_ALBERT_WEB_TIMEOUT = 15.0

# Unicode categories refused in a /save_credentials value: control characters
# (C0, DEL, C1 including NEL) and the line / paragraph separators.
_FORBIDDEN_VALUE_CATEGORIES = ("Cc", "Zl", "Zp")
_INVALID_VALUE_MESSAGE = (
    "Valeur refusée : caractères de contrôle ou retours à la ligne interdits."
)


def _admin_form_env_keys() -> List[str]:
    """
    Return the environment variables handled by the admin credentials form.

    The historical 15 names, in their historical order; ``ALBERT_API_KEY`` is
    inserted after ``MISTRAL_API_BASE_URL`` only when Albert is enabled
    (``albert_enabled()``), so the form, the JSON of /get_credentials and the
    keys accepted by /save_credentials are unchanged while Albert is OFF.

    Returns:
        A new list of environment variable names.
    """
    keys = list(_ADMIN_FORM_ENV_KEYS)
    if albert_enabled():
        keys.insert(keys.index(_ALBERT_FORM_ANCHOR) + 1, ALBERT_ENV_KEY)
    return keys


def _has_forbidden_chars(value: str) -> bool:
    """
    Tell whether a form value contains a control or line separator character.

    Such a character (``\\r``, ``\\n``, NUL, NEL, U+2028…) would split the
    ``NAME=value`` line written to ``.env`` and inject extra variables.

    Args:
        value: The submitted value, already stripped.

    Returns:
        True when at least one character belongs to a refused Unicode category.
    """
    return any(unicodedata.category(ch) in _FORBIDDEN_VALUE_CATEGORIES for ch in value)


def _mask_albert_key(value: Optional[str]) -> str:
    """
    Mask an Albert key for the admin form (the key is never sent in clear).

    Args:
        value: The stored key (may be empty or None).

    Returns:
        ``''`` for an empty value, otherwise ``••••`` followed by the last 4
        characters (only ``••••`` when the value has 4 characters or fewer).
    """
    if not value:
        return ""
    if len(value) <= _ALBERT_KEY_VISIBLE_CHARS:
        return ALBERT_KEY_MASK_PREFIX
    return ALBERT_KEY_MASK_PREFIX + value[-_ALBERT_KEY_VISIBLE_CHARS:]


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
    \f
    Only the keys of ``_admin_form_env_keys()`` are returned, database overlay
    included (an Albert key stored in the database stays hidden while Albert is
    disabled). ``ALBERT_API_KEY`` is always masked server-side (``••••`` plus
    its last 4 characters, or ``''``). The text after the form feed is left out
    of the OpenAPI description.
    """
    logger.info(f"Admin user {admin_user.email} accessing .env credentials")
    env_path = os.path.join(RAGPY_DIR, ".env")
    env_path = os.path.abspath(env_path)
    form_keys = _admin_form_env_keys()
    
    logger.info(f"Attempting to read .env file for get_credentials at: {env_path}")
    if not os.path.exists(env_path):
        logger.error(f".env file not found at: {env_path}")
        credential_keys_on_missing = form_keys
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
    credential_keys = form_keys
    
    credentials = {k: env_vars.get(k, "") for k in credential_keys}

    # Overlay the admin's personal credentials (database) so the form reflects
    # the *effective* value used by the pipeline (database takes precedence over
    # .env, matching get_credential_or_env / build_subprocess_env). Only keys of
    # the form are overlaid: a database Albert key stays hidden while Albert is OFF.
    user_creds = get_user_credentials(admin_user)
    for cred_key, env_key in CREDENTIAL_ENV_MAPPING.items():
        if env_key not in credentials:
            continue
        value = user_creds.get(cred_key)
        if value:
            credentials[env_key] = value

    # The Albert key is never returned in clear.
    if ALBERT_ENV_KEY in credentials:
        credentials[ALBERT_ENV_KEY] = _mask_albert_key(credentials[ALBERT_ENV_KEY])

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
    \f
    Only the keys of ``_admin_form_env_keys()`` are accepted (``ALBERT_API_KEY``
    only when Albert is enabled). An ``ALBERT_API_KEY`` value starting with the
    server-side mask ``••••`` is the masked key echoed back by the form: it is
    ignored, so the stored key is kept. An empty value clears the key.

    A value containing a control or line separator character (``\\r``,
    ``\\n``…) is refused with a 400 ``{error, invalid_keys}`` response (names
    only, never values) before anything is written. The text after the form
    feed is left out of the OpenAPI description.
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
    valid_keys = _admin_form_env_keys()

    # Accepted values, in form order. The masked Albert key echoed by the form
    # is skipped so that it never overwrites the stored key. A value carrying a
    # control or line separator character is refused before anything is written.
    accepted = {}
    invalid_keys = []
    for key in valid_keys:
        if key in data:
            value = str(data[key]).strip()
            if _has_forbidden_chars(value):
                invalid_keys.append(key)
                continue
            if key == ALBERT_ENV_KEY and value.startswith(ALBERT_KEY_MASK_PREFIX):
                continue
            accepted[key] = value

    if invalid_keys:
        logger.warning(f"Refused credential values with control characters: {', '.join(invalid_keys)}")
        return JSONResponse(status_code=400, content={
            "error": _INVALID_VALUE_MESSAGE,
            "invalid_keys": invalid_keys,
        })

    updated_count = 0
    for key, value in accepted.items():
        env_vars[key] = value
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
        env_to_cred[key]: value
        for key, value in accepted.items()
        if key in env_to_cred
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
    current_user: User = Depends(get_current_active_user)
):
    """
    List all available Pinecone indexes using the caller's configured API key.

    **AUTHENTICATED**: Requires a valid authenticated user.

    With Pinecone v3+, each index has its own host URL. This endpoint returns
    the list of indexes with their metadata so the frontend can populate a dropdown.

    The key comes only from the caller's stored credentials (``.env`` fallback
    for admins, see ``get_credential_or_env``); it is never read from the query
    string, so it never appears in URLs nor access logs.

    Args:
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

    # Get API key: user credentials > environment (admin only)
    pinecone_api_key = get_credential_or_env(current_user, "pinecone_api_key")

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
    current_user: User = Depends(get_current_active_user)
):
    """
    List all available Weaviate collections/classes.

    **AUTHENTICATED**: Requires a valid authenticated user.

    The cluster URL and the API key come only from the caller's stored
    credentials (``.env`` fallback for admins, see ``get_credential_or_env``).
    Neither is read from the query string, so a crafted link can never send a
    stored key to a caller-chosen host.

    Args:
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON with list of collections.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If account is inactive or not verified.
    """
    from app.core.credentials import get_credential_or_env, get_credential_error_message

    weaviate_api_key = get_credential_or_env(current_user, "weaviate_api_key")
    weaviate_url = get_credential_or_env(current_user, "weaviate_url")

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
    current_user: User = Depends(get_current_active_user)
):
    """
    List all available Qdrant collections.

    **AUTHENTICATED**: Requires a valid authenticated user.

    The instance URL and the API key come only from the caller's stored
    credentials (``.env`` fallback for admins, see ``get_credential_or_env``).
    Neither is read from the query string, so a crafted link can never send a
    stored key to a caller-chosen host.

    Args:
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON with list of collections.

    Raises:
        HTTPException 401: If not authenticated.
        HTTPException 403: If account is inactive or not verified.
    """
    from app.core.credentials import get_credential_or_env, get_credential_error_message

    qdrant_api_key = get_credential_or_env(current_user, "qdrant_api_key")
    qdrant_url = get_credential_or_env(current_user, "qdrant_url")

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


# ---------------------------------------------------------------------------
# Albert (DINUM): connection status and model catalogue
# ---------------------------------------------------------------------------
class _AlbertGatedRoute(APIRoute):
    """
    Route that the router does not see while Albert is disabled.

    ``matches`` answers ``Match.NONE`` when ``albert_enabled()`` is false, so a
    disabled ``/api/albert/*`` path behaves exactly like an unknown path for
    every HTTP method (404 ``{"detail": "Not Found"}``, never a 405) and the
    trailing-slash redirect never reveals it. While Albert is enabled the
    route behaves like any ``APIRoute`` (405 for another method, redirect of
    a trailing slash). The switch is only read for a request whose path
    matches the route.
    """

    def matches(self, scope: Scope) -> Tuple[Match, Scope]:
        """
        Match like ``APIRoute`` while Albert is enabled, never otherwise.

        Args:
            scope: The ASGI scope of the request.

        Returns:
            ``(Match.NONE, {})`` while Albert is disabled, otherwise the result
            of ``APIRoute.matches``.
        """
        match, child_scope = super().matches(scope)
        if match is not Match.NONE and not albert_enabled():
            return Match.NONE, {}
        return match, child_scope


# Routes of /api/albert/*: gated by _AlbertGatedRoute, hidden from OpenAPI.
# Included into ``router`` at the end of this module (route class preserved).
albert_router = APIRouter(route_class=_AlbertGatedRoute)


def _require_albert_enabled() -> None:
    """
    Dependency answering 404 ``{"detail": "Not Found"}`` while Albert is disabled.

    Defence in depth for the ``/api/albert/*`` routes: ``_AlbertGatedRoute``
    already hides them from the router while Albert is disabled. Declared
    before the authentication dependency, so an unauthenticated client gets
    the same 404 as for an unknown route.

    Raises:
        HTTPException 404: If ``albert_enabled()`` is false.
    """
    if not albert_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


def _albert_web_config(cfg: Any) -> Any:
    """
    Copy the server configuration with the short timeout of the web routes.

    Args:
        cfg: Server configuration (``AlbertConfig``).

    Returns:
        A copy whose ``timeout_collections`` (used by ``me()`` and ``models()``)
        is at most ``_ALBERT_WEB_TIMEOUT`` seconds.
    """
    timeout = min(float(cfg.timeout_collections), _ALBERT_WEB_TIMEOUT)
    return dataclasses.replace(cfg, timeout_collections=timeout)


def _albert_client_factory(cfg: Any, api_key: str) -> Any:
    """
    Build the Albert HTTP client for one web request (tests swap ``AlbertClient`` for a fake transport).

    The client makes a single attempt (``RetryPolicy.single()``: no retry, no
    ``Retry-After`` wait), uses no proactive limiter and has a short timeout
    (``_ALBERT_WEB_TIMEOUT``), so a request never holds a worker thread of the
    threadpool for long. The client module (httpx) is imported lazily, so it
    is never loaded while Albert is disabled.

    Args:
        cfg: Server configuration (``AlbertConfig``).
        api_key: The caller's Albert key (never read from the environment here).

    Returns:
        An ``AlbertClient`` instance.
    """
    from scripts.rad_albert.client import AlbertClient
    from scripts.rad_albert.retry import RetryPolicy

    web_cfg = _albert_web_config(cfg)
    return AlbertClient(
        web_cfg,
        api_key,
        policy=RetryPolicy.from_config(web_cfg).single(),
        use_limiter=False,
    )


def _albert_key_or_403(current_user: User) -> Tuple[Optional[str], Optional[JSONResponse]]:
    """
    Resolve the caller's Albert key with the role-based credential policy.

    Personal key first; the ``.env`` fallback applies to admins only
    (``get_credential_or_env``), so a non-admin without a personal key gets a
    403 even when the server has a key.

    Args:
        current_user: The authenticated user.

    Returns:
        ``(key, None)`` when a key is available, otherwise ``(None, response)``
        with the 403 ``{error, credential_required, configure_url}`` response.
    """
    from app.core.credentials import get_credential_or_env, get_credential_error_message

    api_key = get_credential_or_env(current_user, ALBERT_CREDENTIAL_KEY)
    if api_key:
        return api_key, None
    logger.warning(f"User {current_user.id} missing Albert credentials")
    return None, JSONResponse(
        status_code=403,
        content={
            "error": get_credential_error_message(ALBERT_CREDENTIAL_KEY),
            "credential_required": ALBERT_CREDENTIAL_KEY,
            "configure_url": _CONFIGURE_URL,
        },
    )


def _albert_call(method_name: str, cfg: Any, api_key: str) -> Any:
    """
    Run one read-only Albert call (``me`` or ``models``); executed in a worker thread.

    Args:
        method_name: Name of the ``AlbertClient`` method to call.
        cfg: Server configuration (``AlbertConfig``).
        api_key: The caller's Albert key.

    Returns:
        The value returned by the client method.

    Raises:
        AlbertError: Upstream failure, classified by the client.
    """
    client = _albert_client_factory(cfg, api_key)
    try:
        return getattr(client, method_name)()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


async def _run_albert_call(
    method_name: str, api_key: str, current_user: User
) -> Tuple[Any, Optional[JSONResponse]]:
    """
    Call Albert off the event loop and turn failures into JSON error responses.

    Mapping: account errors (upstream 401, expired account, exhausted budget)
    give 400 with ``credential_invalid``; quota exhaustion, 429, 5xx, network
    and other upstream errors give 502; an invalid server configuration or a
    missing client gives 500. Response bodies are built from fixed French
    messages and never carry the key.

    Args:
        method_name: Name of the ``AlbertClient`` method to call.
        api_key: The caller's Albert key.
        current_user: The authenticated user (logged by id only).

    Returns:
        ``(result, None)`` on success, otherwise ``(None, error_response)``.
    """
    from scripts.rad_albert.config import AlbertConfig
    from scripts.rad_albert.errors import (
        MESSAGES_FR,
        AlbertAuthError,
        AlbertError,
        AlbertQuotaExhausted,
        AlbertTransientError,
        redact,
    )

    try:
        cfg = AlbertConfig.from_env()
    except ValueError as exc:
        logger.error(f"Invalid Albert server configuration: {redact(str(exc), [api_key])}")
        return None, JSONResponse(status_code=500, content={
            "error": "Configuration Albert du serveur invalide (ALBERT_BASE_URL).",
        })

    try:
        return await run_in_threadpool(_albert_call, method_name, cfg, api_key), None
    except AlbertQuotaExhausted as exc:
        logger.warning(f"Albert {method_name} for user {current_user.id}: quota exhausted (HTTP {exc.status})")
        return None, JSONResponse(status_code=502, content={
            "error": MESSAGES_FR["quota_exhausted"],
            "reason": "quota_exhausted",
            "upstream_status": exc.status,
        })
    except AlbertAuthError as exc:
        logger.warning(f"Albert {method_name} for user {current_user.id}: {exc.reason} (HTTP {exc.status})")
        return None, JSONResponse(status_code=400, content={
            "error": MESSAGES_FR.get(exc.reason, MESSAGES_FR["invalid_key"]),
            "credential_invalid": ALBERT_CREDENTIAL_KEY,
            "reason": exc.reason,
            "configure_url": _CONFIGURE_URL,
        })
    except AlbertTransientError as exc:
        logger.warning(f"Albert {method_name} for user {current_user.id}: transient failure (HTTP {exc.status})")
        return None, JSONResponse(status_code=502, content={
            "error": MESSAGES_FR["transient"],
            "reason": "transient",
            "upstream_status": exc.status,
        })
    except AlbertError as exc:
        reason = getattr(exc, "reason", None)
        logger.warning(
            f"Albert {method_name} for user {current_user.id}: {type(exc).__name__} "
            f"{reason or ''} (HTTP {exc.status})"
        )
        return None, JSONResponse(status_code=502, content={
            "error": MESSAGES_FR.get(reason or "", MESSAGES_FR["error"]),
            "reason": reason,
            "upstream_status": exc.status,
        })
    except ImportError as exc:
        logger.error(f"Albert client unavailable: {type(exc).__name__}")
        return None, JSONResponse(status_code=500, content={
            "error": "Client Albert indisponible sur ce serveur.",
        })
    except Exception as exc:
        logger.error(
            f"Unexpected error during Albert {method_name}: {type(exc).__name__}: "
            f"{redact(str(exc), [api_key])}"
        )
        return None, JSONResponse(status_code=500, content={
            "error": "Erreur inattendue lors de l'appel à Albert.",
        })


def _field(obj: Any, name: str) -> Any:
    """
    Read a field from a mapping or an attribute from an object.

    Args:
        obj: A dict-like response entry or an object.
        name: Field name.

    Returns:
        The value, or None when absent.
    """
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _is_number(value: Any) -> bool:
    """Return True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _albert_account_summary(me: Any, now: Optional[float] = None) -> dict:
    """
    Keep the non-identifying account fields of ``GET /v1/me``.

    The email, name and identifiers of the account are never copied.

    Args:
        me: The ``/v1/me`` response (dict or object).
        now: Current Unix time (defaults to ``time.time()``).

    Returns:
        ``expires`` (Unix seconds or None), ``expires_in_days``, ``expired``,
        ``expires_soon`` (fewer than 30 days left), ``budget``, ``permissions``
        and ``limits`` (``router_id``, ``type``, ``value`` only).
    """
    now = time.time() if now is None else now
    expires = _field(me, "expires")
    expires = expires if _is_number(expires) else None
    days_left = int((expires - now) // _SECONDS_PER_DAY) if expires is not None else None
    budget = _field(me, "budget")
    permissions = _field(me, "permissions") or []
    limits = _field(me, "limits") or []
    return {
        "expires": expires,
        "expires_in_days": days_left,
        "expired": expires is not None and expires <= now,
        "expires_soon": expires is not None and now < expires and days_left < _ALBERT_EXPIRY_WARNING_DAYS,
        "budget": budget if (budget is None or _is_number(budget)) else None,
        "permissions": [p for p in permissions if isinstance(p, str)] if isinstance(permissions, list) else [],
        "limits": [
            {
                "router_id": _field(item, "router_id"),
                "type": _field(item, "type"),
                "value": _field(item, "value"),
            }
            for item in (limits if isinstance(limits, list) else [])
        ],
    }


def _albert_models_summary(listing: Any) -> List[dict]:
    """
    Reduce a ``GET /v1/models`` listing to the fields the UI needs.

    Args:
        listing: The client result: a list of models, or a dict with ``data``.

    Returns:
        A list of ``{id, type, aliases, max_context_length}`` dicts.
    """
    entries = listing.get("data") if isinstance(listing, Mapping) else listing
    models = []
    for model in entries or []:
        model_id = _field(model, "id")
        if not isinstance(model_id, str) or not model_id:
            continue
        aliases = _field(model, "aliases") or []
        context = _field(model, "max_context_length")
        models.append({
            "id": model_id,
            "type": _field(model, "type"),
            "aliases": [a for a in aliases if isinstance(a, str)] if isinstance(aliases, list) else [],
            "max_context_length": context if _is_number(context) else None,
        })
    return models


@albert_router.get("/api/albert/status", include_in_schema=False)
async def albert_status(
    _albert_on: None = Depends(_require_albert_enabled),
    current_user: User = Depends(get_current_active_user),
):
    """
    Test the caller's Albert key with ``GET /v1/me`` ("Tester la connexion").

    **AUTHENTICATED**, and only while Albert is enabled (404 otherwise). The
    key comes from the caller's credentials (``.env`` fallback for admins
    only); it is never accepted as a query parameter nor returned. A single
    upstream attempt is made (see ``_albert_client_factory``).

    Args:
        _albert_on: Albert gate (404 while ``albert_enabled()`` is false).
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON ``{success, account, [warning]}`` where ``account`` holds the
        expiry, budget, permissions and limits of the account (never its email
        or name).

    Raises:
        HTTPException 404: If Albert is disabled.
        HTTPException 401: If not authenticated.
    """
    api_key, error = _albert_key_or_403(current_user)
    if error is not None:
        return error

    me, error = await _run_albert_call("me", api_key, current_user)
    if error is not None:
        return error

    account = _albert_account_summary(me)
    payload = {"success": True, "account": account}
    if account["expired"]:
        payload["warning"] = (
            "Compte Albert expiré : écrire à albert.api@numerique.gouv.fr pour le renouveler."
        )
    elif account["expires_soon"]:
        payload["warning"] = (
            f"Le compte Albert expire dans {account['expires_in_days']} jour(s) : "
            "écrire à albert.api@numerique.gouv.fr pour le renouveler."
        )
    return JSONResponse(payload)


@albert_router.get("/api/albert/models", include_in_schema=False)
async def list_albert_models(
    _albert_on: None = Depends(_require_albert_enabled),
    current_user: User = Depends(get_current_active_user),
):
    """
    List the Albert models visible with the caller's key (``GET /v1/models``).

    **AUTHENTICATED**, and only while Albert is enabled (404 otherwise). Same
    key policy, single attempt and error mapping as ``/api/albert/status``.

    Args:
        _albert_on: Albert gate (404 while ``albert_enabled()`` is false).
        current_user: The authenticated user (injected by get_current_active_user).

    Returns:
        JSON ``{success, models, count}``; each model has ``id``, ``type``,
        ``aliases`` and ``max_context_length``.

    Raises:
        HTTPException 404: If Albert is disabled.
        HTTPException 401: If not authenticated.
    """
    api_key, error = _albert_key_or_403(current_user)
    if error is not None:
        return error

    listing, error = await _run_albert_call("models", api_key, current_user)
    if error is not None:
        return error

    models = _albert_models_summary(listing)
    return JSONResponse({"success": True, "models": models, "count": len(models)})


# The /api/albert/* routes join the module router last (after every historical
# route, so the order of the historical routes is unchanged); include_router
# keeps their _AlbertGatedRoute class.
router.include_router(albert_router)
