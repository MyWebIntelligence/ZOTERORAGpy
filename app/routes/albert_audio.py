"""
Albert Audio Ingestion Routes
=============================

Import of audio recordings (interviews, seminars, meetings) transcribed by
Albert (DINUM, ``whisper-large-v3``), sprint Albert R2 (extension E1).

* ``POST /upload_audio``: one or more audio files (mp3, wav, m4a, ogg, opus,
  flac, aac, webm, mp4, mkv) stored under server-generated names in a new
  session folder (``<session>/audio/``), recorded like the other uploads
  (project session or ``SessionOwner``);
* ``POST /process_audio_sse``: runs ``scripts/rad_audio.py`` on that folder
  (Server-Sent Events, same lifecycle as the other pipeline stages) and
  produces ``output.csv`` (``texteocr_provider=albert_whisper``), which the
  chunking stage then takes as usual (no recoding of the transcription).

Both routes answer 404 (like an unknown route, hidden from OpenAPI) unless
``ALBERT_ENABLED=1`` and ``ALBERT_AUDIO_ENABLED=1``. The Albert key comes from
the caller's credentials (``.env`` fallback for admins only); the inference
keys of other providers never reach the script under ``albert_only``.
"""
import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import Depends, File, Form, HTTPException, UploadFile
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

import app.routes.processing as processing_routes
from app.core.albert_policy import check_configuration, policy_error_body, restrict_subprocess_env
from app.core.config import RAGPY_DIR
from app.core.credentials import CredentialMissingError, albert_enabled, build_subprocess_env
from app.core.upload_safety import (
    UnsafePathError,
    UploadTooLargeError,
    confined_path,
    filename_extension,
    max_unzipped_bytes,
    max_upload_bytes,
    new_session_folder_name,
    safe_filename_stem,
    safe_remove_tree,
    save_upload,
)
from app.database.session import get_db
from app.middleware.auth import get_current_active_user
from app.models.pipeline_session import SessionStatus
from app.models.user import User
from app.routes.settings import ALBERT_CREDENTIAL_KEY, _AlbertGatedRoute
from app.services import job_control

logger = logging.getLogger(__name__)

router = APIRouter(route_class=_AlbertGatedRoute)

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".webm", ".mp4", ".mkv")
"""Extensions accepted at upload (non mp3/wav files are converted by ffmpeg in the script)."""
AUDIO_SUBDIR = "audio"
MAX_AUDIO_FILES = 200
_ABORT_RE = re.compile(r"^Albert abort: kind=(\w+) reason=(\w+)(?: credential_required=(\w+))?\s*$")
_FILE_RE = re.compile(r"Fichier\s+(\d+)\s*/\s*(\d+)")
_SEGMENT_RE = re.compile(r"Segment\s+(\d+)\s*/\s*(\d+)")


def _require_audio_enabled() -> None:
    """
    Dependency answering 404 unless Albert and its audio capability are enabled.

    Raises:
        HTTPException 404: ``ALBERT_ENABLED=0`` or ``ALBERT_AUDIO_ENABLED=0``.
    """
    if not albert_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        from scripts.rad_albert.config import AlbertConfig

        enabled = AlbertConfig.from_env().audio_enabled
    except ValueError:
        enabled = False
    if not enabled:
        raise HTTPException(status_code=404, detail="Not Found")


def parse_audio_logs(line: str) -> Optional[Dict[str, Any]]:
    """
    Progress events of ``rad_audio.py`` (``Fichier i/n``, ``Segment j/m``, abort line).

    Args:
        line: One output line of the script.

    Returns:
        An SSE event dict, or None for an uninteresting line.
    """
    text = line.strip()
    abort = _ABORT_RE.match(text)
    if abort:
        event: Dict[str, Any] = {"type": "error", "message": f"Transcription Albert arrêtée ({abort.group(2)})."}
        if abort.group(3):
            event["credential_required"] = abort.group(3)
        return event
    match = _FILE_RE.search(text)
    if match:
        current, total = int(match.group(1)), int(match.group(2))
        return {"type": "progress", "current": current, "total": total, "message": text[:300]}
    match = _SEGMENT_RE.search(text)
    if match:
        return {"type": "log", "message": text[:300]}
    return None


@router.post("/upload_audio", include_in_schema=False)
async def upload_audio(
    files: List[UploadFile] = File(...),
    project_id: Optional[int] = Form(None),
    _audio_on: None = Depends(_require_audio_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Upload audio recordings into a new session folder (``<session>/audio/``).

    A given ``project_id`` is checked before anything is written (same rule
    as ``/upload_csv``). Files are stored under server-generated names
    (``NN_<sanitised stem><ext>``); a refused or failed upload removes the
    session folder. ``UPLOAD_MAX_MB`` applies to each file (413).

    Returns:
        JSON ``{path, tree, message, project_id, session_id}``.
    """
    from app.routes.ingestion import (
        UPLOAD_OWNER_ERROR_MESSAGE,
        _record_upload_owner,
        _too_large,
        _upload_project_or_refusal,
    )

    def _remove_upload(folder: str) -> None:
        """Remove the new session folder, confined to the upload root of this request."""
        try:
            safe_remove_tree(folder, upload_dir)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning(f"Audio upload folder not removed: {type(exc).__name__}")

    project, refusal = _upload_project_or_refusal(db, project_id, current_user)
    if refusal is not None:
        return refusal
    if not files or len(files) > MAX_AUDIO_FILES:
        return JSONResponse(status_code=400, content={"error": f"De 1 à {MAX_AUDIO_FILES} fichiers audio par envoi."})
    for upload in files:
        if filename_extension(upload.filename) not in AUDIO_EXTENSIONS:
            return JSONResponse(status_code=400, content={
                "error": "Format audio non pris en charge : " + ", ".join(AUDIO_EXTENSIONS) + ".",
            })
    upload_dir = processing_routes.UPLOAD_DIR
    dst_dir_name = new_session_folder_name(files[0].filename)
    try:
        dst_dir = confined_path(upload_dir, dst_dir_name)
        audio_dir = confined_path(dst_dir, AUDIO_SUBDIR)
    except UnsafePathError:  # pragma: no cover - generated names never leave UPLOAD_DIR
        return JSONResponse(status_code=400, content={"error": "Nom de fichier invalide."})
    os.makedirs(audio_dir, exist_ok=True)
    stored = []
    total_cap = max_unzipped_bytes()
    total = 0
    try:
        for index, upload in enumerate(files, start=1):
            extension = filename_extension(upload.filename)
            name = f"{index:02d}_{safe_filename_stem(upload.filename, default='audio')}{extension}"
            target = confined_path(audio_dir, name)
            total += await asyncio.to_thread(save_upload, upload.file, target, max_upload_bytes())
            if total_cap is not None and total > total_cap:
                raise UploadTooLargeError(
                    f"Envoi audio de plus de {total_cap // (1024 * 1024)} Mo au total (UPLOAD_MAX_UNZIPPED_MB)."
                )
            stored.append(f"{AUDIO_SUBDIR}/{name}")
    except UploadTooLargeError as exc:
        _remove_upload(dst_dir)
        return _too_large(exc)
    except Exception as exc:
        logger.error(f"Audio upload failed: {type(exc).__name__}: {exc}")
        _remove_upload(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Échec de l'enregistrement des fichiers audio."})
    relative = os.path.relpath(dst_dir, upload_dir)
    recorded, session_id = _record_upload_owner(
        db, project, current_user, relative, files[0].filename, "audio", SessionStatus.CREATED,
        row_count=len(stored),
    )
    if not recorded:
        _remove_upload(dst_dir)
        return JSONResponse(status_code=500, content={"error": UPLOAD_OWNER_ERROR_MESSAGE})
    return JSONResponse({
        "path": relative,
        "tree": stored,
        "message": f"{len(stored)} enregistrement(s) audio importé(s) : lancer la transcription Albert.",
        "project_id": project_id,
        "session_id": session_id,
    })


@router.post("/process_audio_sse", include_in_schema=False)
async def process_audio_sse(
    path: str = Form(...),
    language: str = Form(""),
    prompt: str = Form(""),
    _audio_on: None = Depends(_require_audio_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Transcribe the audio files of a session with Albert (Server-Sent Events).

    Checks, in order: session write right (``_pipeline_session_refusal``),
    stage lock (409 busy event), inference policy, the caller's Albert key
    (403 event ``credential_required``), audio folder present. The script
    runs under ``ALBERT_SUBPROCESS_TIMEOUT``; a client disconnect does not
    stop it (same rule as the other stages).

    Args:
        path: Session folder (relative to uploads/).
        language: ISO 639-1 language, ``auto`` for detection, empty = server default.
        prompt: Optional vocabulary hint (proper nouns, acronyms), 500 characters at most.

    Returns:
        ``text/event-stream`` response.
    """
    refusal = processing_routes._pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return processing_routes._session_refusal_sse(refusal)
    ticket, busy = processing_routes._acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return processing_routes._session_refusal_sse(busy)
    try:
        response = await _process_audio_impl(path, language, prompt, current_user, ticket)
    except BaseException:
        ticket.release()
        raise
    return processing_routes._release_after_stream(response, ticket)


async def _process_audio_impl(path: str, language: str, prompt: str, user: User, ticket: Any) -> StreamingResponse:
    """Body of ``/process_audio_sse`` once the session is checked and locked."""
    from app.utils.sse_helpers import run_subprocess_with_sse

    sse_error = processing_routes._sse_error_response
    try:
        check_configuration()
    except ValueError as exc:
        return sse_error(dict(policy_error_body(exc), type="error", message=str(exc)))
    session_dir = os.path.abspath(os.path.join(processing_routes.UPLOAD_DIR, path))
    audio_dir = os.path.join(session_dir, AUDIO_SUBDIR)
    if not os.path.isdir(audio_dir):
        return sse_error({"type": "error", "message": "Aucun dossier audio dans cette session : importer des enregistrements d'abord."})
    language = (language or "").strip().lower()
    if language and not re.fullmatch(r"auto|[a-z]{2,3}", language):
        return sse_error({"type": "error", "message": "Langue invalide (code ISO 639-1 ou « auto »)."})
    prompt = (prompt or "").strip()
    if len(prompt) > 500 or any(ord(ch) < 32 and ch not in "\t" for ch in prompt):
        return sse_error({"type": "error", "message": "Amorce invalide (500 caractères au plus, sans retour à la ligne)."})
    try:
        env = restrict_subprocess_env(build_subprocess_env(user, required_keys=[ALBERT_CREDENTIAL_KEY]))
    except CredentialMissingError as exc:
        return sse_error(processing_routes._credential_error_payload(exc))
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_audio.py")
    output_csv = os.path.join(session_dir, "output.csv")
    cmd = ["python3", "-u", script_path, "--input", audio_dir, "--output", output_csv]
    if language:
        cmd.extend(["--language", language])
    if prompt:
        cmd.append(f"--prompt={prompt}")
    timeout = processing_routes._subprocess_timeout(1800, True)
    logger.info(f"Albert audio transcription for user {user.id} on session '{path}'")

    async def events():
        """Relay the script events."""
        async for event in run_subprocess_with_sse(
            cmd, parse_audio_logs, session_folder=processing_routes._session_key(path), timeout=timeout,
            env=env, job_ticket=ticket,
        ):
            yield event

    return StreamingResponse(events(), media_type="text/event-stream")
