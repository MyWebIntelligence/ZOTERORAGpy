"""
Settings Routes
===============

This module manages application-level settings and credentials. It allows users
(typically admins or authorized users) to view and update API keys and configuration
variables stored in the `.env` file.

Key Features:
- Credential Management: Retrieve and save API keys (OpenAI, Pinecone, etc.).
- Environment Configuration: Interface for modifying the `.env` file safely.
- Vector DB Discovery: List available indexes from Pinecone, Weaviate, Qdrant.
"""
import os
import logging
from fastapi import APIRouter, Body, Query
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


@router.get("/api/pinecone/indexes")
async def list_pinecone_indexes(api_key: str = Query(None)):
    """
    List all available Pinecone indexes using the provided or configured API key.

    With Pinecone v3+, each index has its own host URL. This endpoint returns
    the list of indexes with their metadata so the frontend can populate a dropdown.

    Args:
        api_key: Optional API key. If not provided, uses PINECONE_API_KEY from env.

    Returns:
        JSON with list of indexes containing name, dimension, metric, host, and stats.
    """
    # Get API key from parameter or environment
    pinecone_api_key = api_key or os.getenv("PINECONE_API_KEY", "")

    if not pinecone_api_key:
        logger.warning("Pinecone API key not configured")
        return JSONResponse(
            status_code=400,
            content={"error": "Pinecone API key not configured. Please add it in Settings."}
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
async def list_weaviate_collections(api_key: str = Query(None), url: str = Query(None)):
    """
    List all available Weaviate collections/classes.

    Args:
        api_key: Optional API key. If not provided, uses WEAVIATE_API_KEY from env.
        url: Optional Weaviate URL. If not provided, uses WEAVIATE_URL from env.

    Returns:
        JSON with list of collections.
    """
    weaviate_api_key = api_key or os.getenv("WEAVIATE_API_KEY", "")
    weaviate_url = url or os.getenv("WEAVIATE_URL", "")

    if not weaviate_url:
        return JSONResponse(
            status_code=400,
            content={"error": "Weaviate URL not configured."}
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
async def list_qdrant_collections(api_key: str = Query(None), url: str = Query(None)):
    """
    List all available Qdrant collections.

    Args:
        api_key: Optional API key. If not provided, uses QDRANT_API_KEY from env.
        url: Optional Qdrant URL. If not provided, uses QDRANT_URL from env.

    Returns:
        JSON with list of collections.
    """
    qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY", "")
    qdrant_url = url or os.getenv("QDRANT_URL", "")

    if not qdrant_url:
        return JSONResponse(
            status_code=400,
            content={"error": "Qdrant URL not configured."}
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
