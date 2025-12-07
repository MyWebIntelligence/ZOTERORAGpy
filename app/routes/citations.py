"""
Citation Import Routes
======================

This module handles the import of citations from Publish or Perish JSON files to Zotero,
with LLM-based relevance filtering and user preview/validation workflow.

Key Features:
- Upload Publish or Perish JSON files with configuration
- LLM-based citation filtering with real-time SSE progress
- Preview interface with relevant/skipped citations
- Import selected citations to Zotero (create or update)
- Progress tracking via PipelineSession model

Workflow:
1. Upload JSON + config → Create session
2. Filter citations via LLM → Save preview.json
3. User validates selection → Import to Zotero
4. Return final stats (created/updated/errors)
"""
import os
import json
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_DIR
from app.database.session import get_db
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.project import Project
from app.middleware.auth import get_current_active_user
from app.models.user import User
from app.utils.publishorperish_parser import parse_pop_json
from app.utils.citation_fetcher import fetch_citation_content
from app.utils.citation_filter import filter_citation_with_llm
from app.utils.llm_note_generator import get_llm_semaphore
from app.utils.zotero_client import (
    get_or_create_collection,
    create_or_update_item,
    ZoteroAPIError
)

# Setup logger
logger = logging.getLogger(__name__)

# Main router for project-related endpoints
router = APIRouter(prefix="/api/projects", tags=["Citations"])

# Separate router for session-related endpoints (without /projects prefix)
session_router = APIRouter(prefix="/api", tags=["Citation Sessions"])


@router.post("/{project_id}/upload_pop_json")
async def upload_pop_json(
    project_id: int,
    json_file: UploadFile = File(...),
    collection_name: str = Form(...),
    collection_description: str = Form(""),
    model: str = Form("gpt-4o-mini"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload Publish or Perish JSON file and create citation import session.

    Args:
        project_id: ID of the target project
        json_file: Uploaded JSON file from Publish or Perish export
        collection_name: Target Zotero collection name (created if doesn't exist)
        collection_description: Description for the bibliography context
        model: LLM model to use for filtering (gpt-4o-mini, google/gemini-2.5-flash, etc.)
        db: Database session
        current_user: Authenticated user

    Returns:
        JSONResponse with:
            - session_id: Pipeline session ID
            - session_folder: Relative path to session folder
            - total_citations: Number of citations in the file
            - requires_confirmation: True if citation count > 100

    Raises:
        HTTPException 404: Project not found
        HTTPException 403: User doesn't have access to project
        HTTPException 400: Invalid JSON file format
    """
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check user has access (owner or member)
    if project.owner_id != current_user.id:
        # TODO: Check project members table when implemented
        # For now, only owner can import citations
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to import citations for this project"
        )

    # Generate unique session folder
    unique_id = str(uuid.uuid4().hex)[:8]
    session_folder_name = f"pop_{unique_id}_{project.name[:20]}"
    session_folder = os.path.join(UPLOAD_DIR, session_folder_name)
    os.makedirs(session_folder, exist_ok=True)
    logger.info(f"Created session folder: {session_folder}")

    # Save uploaded JSON file
    json_path = os.path.join(session_folder, "publishorperish.json")
    try:
        with open(json_path, "wb") as f:
            content = await json_file.read()
            f.write(content)
        logger.info(f"Saved uploaded JSON to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Parse and count citations
    try:
        citations = parse_pop_json(json_path)
        total_citations = len(citations)
        logger.info(f"Parsed {total_citations} citations from uploaded JSON")
    except ValueError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid Publish or Perish JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON file: {str(e)}")

    # Save configuration for later use
    config = {
        "collection_name": collection_name,
        "collection_description": collection_description,
        "model": model,
        "project_id": project_id,
        "project_name": project.name,
        "project_description": project.description or ""
    }
    config_path = os.path.join(session_folder, "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved configuration to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

    # Create PipelineSession
    try:
        relative_session_folder = os.path.relpath(session_folder, UPLOAD_DIR)
        pipeline_session = PipelineSession(
            project_id=project_id,
            session_folder=relative_session_folder,
            original_filename=json_file.filename,
            source_type="pop_json",
            status=SessionStatus.CREATED,
            row_count=total_citations  # Total citations to process
        )
        db.add(pipeline_session)
        db.commit()
        db.refresh(pipeline_session)
        session_id = pipeline_session.id
        logger.info(f"Created PipelineSession {session_id} for citation import (project {project_id})")
    except Exception as e:
        logger.error(f"Failed to create PipelineSession: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

    # Check if confirmation required (> 100 citations)
    requires_confirmation = total_citations > 100

    return JSONResponse({
        "session_id": session_id,
        "session_folder": relative_session_folder,
        "total_citations": total_citations,
        "requires_confirmation": requires_confirmation
    })


@router.post("/{project_id}/filter_citations_sse")
async def filter_citations_sse(
    project_id: int,
    session_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Filter citations using LLM with Server-Sent Events for real-time progress.

    This endpoint performs the LLM filtering step with web content fetching.
    It streams progress in real-time and saves results to preview.json for user validation.

    Args:
        project_id: ID of the project
        session_id: Pipeline session ID
        db: Database session
        current_user: Authenticated user

    Returns:
        StreamingResponse with SSE events:
            - init: {"type": "init", "total": N}
            - progress: {"type": "progress", "current": i, "status": "relevant"|"skipped", "title": "..."}
            - complete: {"type": "complete", "relevant": X, "skipped": Y}
            - error: {"type": "error", "message": "..."}

    Raises:
        HTTPException 404: Session or project not found
        HTTPException 403: User doesn't have access
    """
    async def event_generator():
        try:
            # Get session from database
            pipeline_session = db.query(PipelineSession).filter(
                PipelineSession.id == session_id,
                PipelineSession.project_id == project_id
            ).first()

            if not pipeline_session:
                yield f'data: {{"type": "error", "message": "Session not found"}}\n\n'
                return

            # Verify project access
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project or project.owner_id != current_user.id:
                yield f'data: {{"type": "error", "message": "Access denied"}}\n\n'
                return

            # Build absolute paths
            session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
            json_path = os.path.join(session_folder, "publishorperish.json")
            config_path = os.path.join(session_folder, "config.json")
            preview_path = os.path.join(session_folder, "preview.json")

            # Check files exist
            if not os.path.exists(json_path):
                yield f'data: {{"type": "error", "message": "Citations JSON not found"}}\n\n'
                return
            if not os.path.exists(config_path):
                yield f'data: {{"type": "error", "message": "Configuration not found"}}\n\n'
                return

            # Load citations
            try:
                citations = parse_pop_json(json_path)
                logger.info(f"Loaded {len(citations)} citations for filtering")
            except Exception as e:
                logger.error(f"Failed to parse citations: {e}")
                yield f'data: {{"type": "error", "message": "Failed to parse citations: {str(e)}"}}\n\n'
                return

            # Load config
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                yield f'data: {{"type": "error", "message": "Failed to load configuration: {str(e)}"}}\n\n'
                return

            # Update session status
            pipeline_session.status = SessionStatus.FILTERING_CITATIONS
            db.commit()
            logger.info(f"Session {session_id} status → FILTERING_CITATIONS")

            # Send init event
            total = len(citations)
            yield f'data: {{"type": "init", "total": {total}}}\n\n'

            # Prepare results storage
            preview_results = {
                "relevant": [],
                "skipped": []
            }

            # Get global LLM semaphore for concurrency control
            semaphore = get_llm_semaphore()

            # Process each citation
            for idx, citation in enumerate(citations):
                # Use mode='json' to ensure HttpUrl and other types are serializable
                citation_dict = citation.model_dump(mode='json')

                # Fetch web content
                try:
                    content, source = await fetch_citation_content(
                        article_url=str(citation.article_url) if citation.article_url else None,
                        fulltext_url=str(citation.fulltext_url) if citation.fulltext_url else None,
                        timeout=30,
                        max_chars=10000
                    )
                    logger.debug(f"Fetched content for citation {idx}: source={source}, length={len(content)}")
                except Exception as e:
                    logger.warning(f"Failed to fetch content for citation {idx}: {e}")
                    content = ""
                    source = "none"

                # Filter with LLM (with semaphore for concurrency control)
                try:
                    async with semaphore:
                        remaining_slots = semaphore._value
                        logger.debug(f"Acquired LLM slot for citation {idx} ({remaining_slots} remaining)")

                        filter_result = await filter_citation_with_llm(
                            citation=citation_dict,
                            web_content=content,
                            web_content_source=source,
                            project_name=config["project_name"],
                            project_description=config["project_description"],
                            collection_name=config["collection_name"],
                            collection_description=config["collection_description"],
                            model=config["model"]
                        )

                    # Categorize result
                    if isinstance(filter_result, dict):
                        # Citation is relevant - extract zotero_item from filter result
                        preview_results["relevant"].append({
                            "index": idx,
                            "citation": citation_dict,
                            "zotero_data": filter_result.get("zotero_item", filter_result),
                            "relevance_score": filter_result.get("relevance_score"),
                            "relevance_reason": filter_result.get("relevance_reason"),
                            "web_source": source
                        })
                        status = "relevant"
                        logger.info(f"Citation {idx} marked as RELEVANT")
                    else:
                        # Citation skipped (filter_result == "NA")
                        preview_results["skipped"].append({
                            "index": idx,
                            "citation": citation_dict,
                            "reason": "Not relevant (LLM decision)"
                        })
                        status = "skipped"
                        logger.info(f"Citation {idx} marked as SKIPPED")

                except Exception as e:
                    logger.error(f"LLM filtering failed for citation {idx}: {e}")
                    # Skip citation on error
                    preview_results["skipped"].append({
                        "index": idx,
                        "citation": citation_dict,
                        "reason": f"Error during filtering: {str(e)}"
                    })
                    status = "error"

                # Yield progress event
                title_escaped = citation.title.replace('"', '\\"').replace('\n', ' ')[:50]
                yield f'data: {{"type": "progress", "current": {idx + 1}, "status": "{status}", "title": "{title_escaped}"}}\n\n'

            # Save preview results
            try:
                with open(preview_path, "w", encoding="utf-8") as f:
                    json.dump(preview_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved preview results to: {preview_path}")
            except Exception as e:
                logger.error(f"Failed to save preview: {e}")
                yield f'data: {{"type": "error", "message": "Failed to save preview results: {str(e)}"}}\n\n'
                return

            # Update session status back to CREATED (ready for import)
            pipeline_session.status = SessionStatus.CREATED
            db.commit()
            logger.info(f"Session {session_id} status → CREATED (filtering complete)")

            # Send completion event
            relevant_count = len(preview_results["relevant"])
            skipped_count = len(preview_results["skipped"])
            yield f'data: {{"type": "complete", "relevant": {relevant_count}, "skipped": {skipped_count}}}\n\n'

        except Exception as e:
            logger.error(f"Unexpected error in filter_citations_sse: {e}", exc_info=True)
            yield f'data: {{"type": "error", "message": "Unexpected error: {str(e)}"}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{project_id}/import_citations_sse")
async def import_citations_sse(
    project_id: int,
    session_id: int = Form(...),
    selected_indices: str = Form(...),  # JSON array "[0,1,2,...]"
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Import selected citations to Zotero with Server-Sent Events for progress tracking.

    This endpoint performs the actual import to Zotero, creating or updating items
    as needed. It uses the triple deduplication strategy (DOI → URL → Title).

    Args:
        project_id: ID of the project
        session_id: Pipeline session ID
        selected_indices: JSON array of selected citation indices from preview
        db: Database session
        current_user: Authenticated user

    Returns:
        StreamingResponse with SSE events:
            - init: {"type": "init", "total": N}
            - progress: {"type": "progress", "current": i, "action": "created"|"updated", "title": "..."}
            - error: {"type": "error", "index": i, "message": "..."}
            - complete: {"type": "complete", "created": X, "updated": Y, "errors": Z}

    Raises:
        HTTPException 404: Session or project not found
        HTTPException 403: User doesn't have access
    """
    async def event_generator():
        try:
            # Get session from database
            pipeline_session = db.query(PipelineSession).filter(
                PipelineSession.id == session_id,
                PipelineSession.project_id == project_id
            ).first()

            if not pipeline_session:
                yield f'data: {{"type": "error", "message": "Session not found"}}\n\n'
                return

            # Verify project access
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project or project.owner_id != current_user.id:
                yield f'data: {{"type": "error", "message": "Access denied"}}\n\n'
                return

            # Build absolute paths
            session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
            preview_path = os.path.join(session_folder, "preview.json")
            config_path = os.path.join(session_folder, "config.json")

            # Check files exist
            if not os.path.exists(preview_path):
                yield f'data: {{"type": "error", "message": "Preview results not found. Please run filtering first."}}\n\n'
                return
            if not os.path.exists(config_path):
                yield f'data: {{"type": "error", "message": "Configuration not found"}}\n\n'
                return

            # Load preview results
            try:
                with open(preview_path, "r", encoding="utf-8") as f:
                    preview = json.load(f)
                logger.info(f"Loaded preview with {len(preview['relevant'])} relevant citations")
            except Exception as e:
                logger.error(f"Failed to load preview: {e}")
                yield f'data: {{"type": "error", "message": "Failed to load preview results: {str(e)}"}}\n\n'
                return

            # Load config
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                yield f'data: {{"type": "error", "message": "Failed to load configuration: {str(e)}"}}\n\n'
                return

            # Parse selected indices
            try:
                selected = json.loads(selected_indices)
                if not isinstance(selected, list):
                    raise ValueError("selected_indices must be a JSON array")
            except Exception as e:
                logger.error(f"Invalid selected_indices format: {e}")
                yield f'data: {{"type": "error", "message": "Invalid selected indices format: {str(e)}"}}\n\n'
                return

            # Filter selected items
            try:
                to_import = [preview["relevant"][i] for i in selected]
                logger.info(f"Selected {len(to_import)} citations to import")
            except IndexError as e:
                logger.error(f"Invalid index in selection: {e}")
                yield f'data: {{"type": "error", "message": "Invalid citation index in selection"}}\n\n'
                return

            # Get Zotero credentials from environment
            zotero_api_key = os.getenv("ZOTERO_API_KEY")
            library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "users")
            library_id = os.getenv("ZOTERO_USER_ID")

            if not zotero_api_key or not library_id:
                yield f'data: {{"type": "error", "message": "Zotero credentials not configured"}}\n\n'
                return

            # Update session status
            pipeline_session.status = SessionStatus.IMPORTING_CITATIONS
            db.commit()
            logger.info(f"Session {session_id} status → IMPORTING_CITATIONS")

            # Get or create collection
            try:
                coll_result = get_or_create_collection(
                    library_type=library_type,
                    library_id=library_id,
                    collection_name=config["collection_name"],
                    description=config["collection_description"],
                    api_key=zotero_api_key
                )
                collection_key = coll_result["key"]
                logger.info(f"Using Zotero collection: {collection_key} (created={coll_result['created']})")
            except ZoteroAPIError as e:
                logger.error(f"Failed to get/create collection: {e}")
                yield f'data: {{"type": "error", "message": "Failed to access Zotero collection: {str(e)}"}}\n\n'
                return

            # Send init event
            total = len(to_import)
            yield f'data: {{"type": "init", "total": {total}}}\n\n'

            # Counters
            created = 0
            updated = 0
            errors = 0
            error_details = []

            # Import each citation
            for idx, item in enumerate(to_import):
                try:
                    # Prepare Zotero item data
                    zotero_data = item["zotero_data"].copy()
                    zotero_data["collections"] = [collection_key]

                    # Create or update item
                    result = create_or_update_item(
                        library_type=library_type,
                        library_id=library_id,
                        item_data=zotero_data,
                        api_key=zotero_api_key
                    )

                    if result["success"]:
                        if result["action"] == "created":
                            created += 1
                        else:
                            updated += 1

                        # Yield progress event
                        title_escaped = item["citation"]["title"].replace('"', '\\"').replace('\n', ' ')[:50]
                        yield f'data: {{"type": "progress", "current": {idx + 1}, "action": "{result["action"]}", "title": "{title_escaped}"}}\n\n'
                        logger.info(f"Citation {idx} {result['action']}: {result['item_key']}")
                    else:
                        errors += 1
                        error_msg = result.get("message", "Unknown error")
                        error_details.append({"index": idx, "message": error_msg})
                        yield f'data: {{"type": "error", "index": {idx}, "message": "{error_msg}"}}\n\n'
                        logger.error(f"Failed to import citation {idx}: {error_msg}")

                except Exception as e:
                    errors += 1
                    error_msg = str(e).replace('"', '\\"')
                    error_details.append({"index": idx, "message": error_msg})
                    yield f'data: {{"type": "error", "index": {idx}, "message": "{error_msg}"}}\n\n'
                    logger.error(f"Exception importing citation {idx}: {e}", exc_info=True)

            # Save error log if any
            if error_details:
                error_log_path = os.path.join(session_folder, "import_errors.json")
                try:
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        json.dump(error_details, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved error log to: {error_log_path}")
                except Exception as e:
                    logger.warning(f"Failed to save error log: {e}")

            # Update session with final stats
            pipeline_session.status = SessionStatus.COMPLETED
            pipeline_session.row_count = total  # Total processed
            pipeline_session.chunk_count = created + updated  # Successfully imported
            db.commit()
            logger.info(f"Session {session_id} status → COMPLETED (created={created}, updated={updated}, errors={errors})")

            # Send completion event
            yield f'data: {{"type": "complete", "created": {created}, "updated": {updated}, "errors": {errors}}}\n\n'

        except Exception as e:
            logger.error(f"Unexpected error in import_citations_sse: {e}", exc_info=True)
            yield f'data: {{"type": "error", "message": "Unexpected error: {str(e)}"}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@session_router.get("/sessions/{session_id}/preview")
async def get_preview_results(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get preview results (relevant and skipped citations) for user validation.

    This endpoint retrieves the preview.json file created during the filtering step,
    allowing the frontend to display the results for user review.

    Args:
        session_id: Pipeline session ID
        db: Database session
        current_user: Authenticated user

    Returns:
        JSONResponse with:
            {
                "relevant": [
                    {
                        "index": int,
                        "citation": {...},
                        "zotero_data": {...},
                        "web_source": "pdf"|"html"|"none"
                    },
                    ...
                ],
                "skipped": [
                    {
                        "index": int,
                        "citation": {...},
                        "reason": str
                    },
                    ...
                ]
            }

    Raises:
        HTTPException 404: Session not found or preview not available
        HTTPException 403: User doesn't have access to this session
    """
    # Get session from database
    pipeline_session = db.query(PipelineSession).filter(
        PipelineSession.id == session_id
    ).first()

    if not pipeline_session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify project access
    project = db.query(Project).filter(Project.id == pipeline_session.project_id).first()
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Build path to preview file
    session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
    preview_path = os.path.join(session_folder, "preview.json")

    # Check if preview exists
    if not os.path.exists(preview_path):
        raise HTTPException(
            status_code=404,
            detail="Preview not available. Please run the filtering step first."
        )

    # Load and return preview results
    try:
        with open(preview_path, "r", encoding="utf-8") as f:
            preview_results = json.load(f)
        logger.info(f"Loaded preview for session {session_id}: {len(preview_results['relevant'])} relevant, {len(preview_results['skipped'])} skipped")
        return JSONResponse(preview_results)
    except Exception as e:
        logger.error(f"Failed to load preview results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load preview results: {str(e)}")
