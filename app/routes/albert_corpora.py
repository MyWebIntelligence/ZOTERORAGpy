"""
Albert Corpora Routes
=====================

Management of the Albert (DINUM) corpora known to RAGpy (sprint Albert R2) and
of the account usage. The corpora registry (``app/services/albert_access.py``)
is filled after each upload to an Albert collection (step 4, ``db_choice=albert``)
or by binding an existing private collection.

The search and sourced answers (``/api/albert/search``, ``/api/albert/rag``,
page ``/albert/rag``) were removed on 2026-10-03: RAGpy prepares the corpora,
the questions are asked by other tools.

Every route is gated like ``/api/albert/*`` in ``settings.py``
(``_AlbertGatedRoute``): while ``ALBERT_ENABLED=0`` it answers exactly like an
unknown route (404) and it never appears in the OpenAPI schema.

Routes:

* ``GET /api/albert/capabilities``: switches, policy and defaults (no remote call);
* ``GET /api/albert/corpora``: corpora visible to the caller (local registry);
* ``POST /api/albert/corpora/bind``: attach an existing private collection,
  checked remotely with the caller's key;
* ``DELETE /api/albert/corpora/{id}``: detach a corpus from RAGpy (the remote
  collection is kept);
* ``GET /api/albert/corpora/{id}/documents``: remote documents joined with the
  local catalogue;
* ``DELETE /api/albert/corpora/{id}/documents/{document_id}?confirm=true``:
  erase one remote document (membership checked, audited);
* ``GET /api/albert/usage?days=N``: daily usage of the caller's account.

Security: a corpus id from the browser is resolved through the local registry
and the caller's rights (``app/services/albert_access.py``); the remote call
then uses the caller's own key (personal key, ``.env`` fallback for admins
only). No key or URL is ever accepted from the request.
"""
import dataclasses
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.middleware.auth import get_current_active_user
from app.models.user import User
from app.routes.settings import (
    ALBERT_CREDENTIAL_KEY,
    _AlbertGatedRoute,
    _albert_key_or_403,
    _require_albert_enabled,
)
from app.services import albert_access
from app.services.albert_access import CorpusAction

logger = logging.getLogger(__name__)

router = APIRouter(route_class=_AlbertGatedRoute)

CONFIGURE_URL = "/settings/credentials"
MAX_USAGE_DAYS = 90
DOCUMENTS_PER_PAGE = 50
DOCUMENTS_MAX_PER_PAGE = 100
CONFIRM_VALUES = ("1", "true", "yes", "on")
DOCUMENT_DELETE_AUDIT_ACTION = "ALBERT_DOCUMENT_DELETE"
CORPUS_DETACH_AUDIT_ACTION = "ALBERT_CORPUS_DETACH"
CORPUS_BIND_AUDIT_ACTION = "ALBERT_CORPUS_BIND"
WEB_TIMEOUT = 15.0
"""HTTP timeout of the management calls (corpora, documents, usage)."""


# ---------------------------------------------------------------------------
# Gates and configuration
# ---------------------------------------------------------------------------
def _config() -> Any:
    """
    Current server configuration (``AlbertConfig.from_env()``).

    Returns:
        The configuration.

    Raises:
        HTTPException 500: Refused ``ALBERT_BASE_URL``.
    """
    from scripts.rad_albert.config import AlbertConfig

    try:
        return AlbertConfig.from_env()
    except ValueError:
        raise HTTPException(status_code=500, detail="Configuration Albert du serveur invalide (ALBERT_BASE_URL).")


def key_fingerprint(api_key: str) -> str:
    """
    Non-reversible fingerprint of a key (same as ``AlbertClient.key_fingerprint``).

    Args:
        api_key: The Albert key.

    Returns:
        ``sha256(key)[:12]``.
    """
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:12]


def _client(cfg: Any, api_key: str) -> Any:
    """
    Albert client of one management request (tests swap ``AlbertClient`` for a fake transport).

    A single attempt with a short timeout, no process-wide limiter.

    Args:
        cfg: Server configuration.
        api_key: The caller's key.

    Returns:
        An ``AlbertClient``.
    """
    from scripts.rad_albert.client import AlbertClient
    from scripts.rad_albert.retry import RetryPolicy

    web_cfg = dataclasses.replace(cfg, timeout_collections=min(float(cfg.timeout_collections), WEB_TIMEOUT))
    return AlbertClient(web_cfg, api_key, policy=RetryPolicy.from_config(web_cfg).single(), use_limiter=False)


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    """JSON error ``{error, …}``."""
    body = {"error": message}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


def error_payload(exc: BaseException) -> Tuple[int, Dict[str, Any]]:
    """
    Map an exception of an Albert management call to an HTTP status and a JSON body.

    Messages are fixed French texts (never the key).

    Args:
        exc: The exception.

    Returns:
        ``(status, body)``.
    """
    from scripts.rad_albert.errors import (
        MESSAGES_FR,
        AlbertAuthError,
        AlbertError,
        AlbertQuotaExhausted,
        AlbertTransientError,
    )
    from scripts.rad_albert.policy import AlbertPolicyError

    if isinstance(exc, AlbertPolicyError):
        return 400, {"error": str(exc), "reason": getattr(exc, "reason", "bad_request")}
    if isinstance(exc, AlbertQuotaExhausted):
        return 429, {"error": str(exc) if "Limiteur" in str(exc) else MESSAGES_FR["quota_exhausted"],
                     "reason": "quota_exhausted"}
    if isinstance(exc, AlbertAuthError):
        return 400, {"error": MESSAGES_FR.get(exc.reason, MESSAGES_FR["invalid_key"]), "reason": exc.reason,
                     "credential_invalid": ALBERT_CREDENTIAL_KEY, "configure_url": CONFIGURE_URL}
    if isinstance(exc, AlbertTransientError):
        return 502, {"error": MESSAGES_FR["transient"], "reason": "transient", "upstream_status": exc.status}
    if isinstance(exc, AlbertError):
        reason = getattr(exc, "reason", None)
        return 502, {"error": MESSAGES_FR.get(reason or "", MESSAGES_FR["error"]), "reason": reason,
                     "upstream_status": exc.status}
    return 500, {"error": "Erreur inattendue lors de l'appel à Albert.", "reason": "internal"}


async def _read_json(request: Request) -> Dict[str, Any]:
    """
    JSON body of a request (an object), or an empty dict.

    Raises:
        HTTPException 400: Body that is not a JSON object.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Corps JSON attendu.")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Corps JSON attendu (objet).")
    return body


def _http_error(exc: HTTPException) -> JSONResponse:
    """JSON ``{error}`` body for an ``HTTPException`` raised while preparing a request."""
    return _error(exc.status_code, str(exc.detail))


# ---------------------------------------------------------------------------
# Capabilities and corpora
# ---------------------------------------------------------------------------
@router.get("/api/albert/capabilities", include_in_schema=False)
async def albert_capabilities(
    _albert_on: None = Depends(_require_albert_enabled),
    current_user: User = Depends(get_current_active_user),
):
    """
    Switches, inference policy and defaults of the Albert integration (no remote call).

    Returns:
        JSON ``{policy, audio, ocr, embedding_provider}``.
    """
    from scripts.rad_albert.policy import describe

    cfg = _config()
    return JSONResponse({
        "success": True,
        "policy": describe(cfg),
        "audio": cfg.audio_enabled,
        "ocr": cfg.ocr_enabled,
        "embedding_provider": (os.environ.get("EMBEDDING_PROVIDER") or "openai").strip().lower() or "openai",
    })


@router.get("/api/albert/corpora", include_in_schema=False)
async def list_corpora(
    _albert_on: None = Depends(_require_albert_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Corpora visible to the caller (own corpora and those of their projects), on the configured host.

    Returns:
        JSON ``{success, corpora}``; each corpus has ``documents`` (catalogue
        size), ``chunks`` (catalogued chunks) and ``can_write`` / ``is_owner``.
    """
    cfg = _config()
    corpora = albert_access.list_corpora(db, current_user, base_url=cfg.base_url)
    items = []
    for corpus in corpora:
        data = corpus.to_dict()
        data["documents"] = len(corpus.sources)
        data["chunks"] = sum(s.chunk_count or 0 for s in corpus.sources)
        data["is_owner"] = corpus.owner_user_id == current_user.id
        data["can_write"] = albert_access.corpus_refusal(db, corpus, current_user, CorpusAction.WRITE) is None
        items.append(data)
    return JSONResponse({"success": True, "corpora": items})


@router.post("/api/albert/corpora/bind", include_in_schema=False)
async def bind_corpus(
    request: Request,
    _albert_on: None = Depends(_require_albert_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Attach an existing private Albert collection to RAGpy (body ``{collection_id, project_id?}``).

    The collection is read with the caller's key and must be private; the
    caller becomes the owner of the corpus. Audited (``ALBERT_CORPUS_BIND``).

    Returns:
        JSON ``{success, corpus}`` (201), or the existing corpus (200).
    """
    from app.models.audit import create_audit_log
    from scripts.rad_albert.collections import is_private
    from scripts.rad_albert.errors import AlbertError

    try:
        payload = await _read_json(request)
    except HTTPException as exc:
        return _http_error(exc)
    try:
        collection_id = int(payload.get("collection_id"))
        if collection_id <= 0 or isinstance(payload.get("collection_id"), bool):
            raise ValueError
    except (TypeError, ValueError):
        return _error(400, "Identifiant de collection Albert invalide (entier positif attendu).")
    project_id = payload.get("project_id")
    if project_id not in (None, ""):
        try:
            project_id = int(project_id)
        except (TypeError, ValueError):
            return _error(400, "Identifiant de projet invalide.")
    else:
        project_id = None
    api_key, error = _albert_key_or_403(current_user)
    if error is not None:
        return error
    cfg = _config()
    existing = albert_access.find_corpus(db, current_user.id, cfg.base_url, collection_id)
    if existing is not None and existing.status != "deleted":
        return JSONResponse({"success": True, "corpus": existing.to_dict(), "existing": True})

    def fetch() -> Any:
        """Read the collection with the caller's key."""
        client = _client(cfg, api_key)
        try:
            return client.get_collection(collection_id)
        finally:
            client.close()

    try:
        collection = await run_in_threadpool(fetch)
    except AlbertError as exc:
        status, body = error_payload(exc)
        if getattr(exc, "reason", None) == "not_found":
            return _error(404, "Collection Albert introuvable ou inaccessible avec votre clé.")
        return JSONResponse(status_code=status, content=body)
    if not isinstance(collection, dict) or not is_private(collection):
        return _error(400, "Seules les collections privées peuvent être rattachées à RAGpy.")
    try:
        corpus = albert_access.bind_corpus(
            db, user=current_user, collection=collection, base_url=cfg.base_url,
            key_fingerprint=key_fingerprint(api_key), project_id=project_id,
        )
    except ValueError as exc:
        return _error(403, str(exc))
    try:
        create_audit_log(db, action=CORPUS_BIND_AUDIT_ACTION, user_id=current_user.id,
                         resource_type="albert_collection", resource_id=collection_id,
                         details={"collection_id": collection_id, "corpus_id": corpus.id, "project_id": project_id})
    except Exception as exc:
        logger.error(f"Albert corpus bind audit not written: {type(exc).__name__}")
        db.rollback()
    return JSONResponse(status_code=201, content={"success": True, "corpus": corpus.to_dict()})


@router.delete("/api/albert/corpora/{corpus_id}", include_in_schema=False)
async def detach_corpus(
    corpus_id: str,
    _albert_on: None = Depends(_require_albert_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Detach a corpus from RAGpy (local registry only: the Albert collection is kept).

    Returns:
        JSON ``{success, detached}``.
    """
    from app.models.audit import create_audit_log

    corpus, refusal = albert_access.get_corpus(db, current_user, corpus_id, CorpusAction.ADMIN)
    if refusal is not None:
        return _error(*refusal)
    details = {"corpus_id": corpus.id, "collection_id": corpus.collection_id}
    albert_access.detach_corpus(db, corpus)
    try:
        create_audit_log(db, action=CORPUS_DETACH_AUDIT_ACTION, user_id=current_user.id,
                         resource_type="albert_collection", resource_id=details["collection_id"], details=details)
    except Exception as exc:
        logger.error(f"Albert corpus detach audit not written: {type(exc).__name__}")
        db.rollback()
    return JSONResponse({"success": True, "detached": details["corpus_id"]})


@router.get("/api/albert/corpora/{corpus_id}/documents", include_in_schema=False)
async def list_corpus_documents(
    corpus_id: str,
    page: str = "1",
    per_page: str = str(DOCUMENTS_PER_PAGE),
    _albert_on: None = Depends(_require_albert_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Remote documents of a corpus (caller's key) joined with the local catalogue.

    Returns:
        JSON ``{success, corpus, documents, total, page, per_page, pages, status}``.
    """
    from scripts.rad_albert.errors import AlbertError

    corpus, refusal = albert_access.get_corpus(db, current_user, corpus_id, CorpusAction.READ)
    if refusal is not None:
        return _error(*refusal)
    if not (page.isdigit() and per_page.isdigit()) or int(page) < 1 or not 1 <= int(per_page) <= DOCUMENTS_MAX_PER_PAGE:
        return _error(400, f"Pagination invalide : page >= 1 et 1 <= per_page <= {DOCUMENTS_MAX_PER_PAGE}.")
    api_key, error = _albert_key_or_403(current_user)
    if error is not None:
        return error
    cfg = _config()

    def fetch() -> Any:
        """List the documents of the collection with the caller's key."""
        client = _client(cfg, api_key)
        try:
            return client.list_documents(corpus.collection_id)
        finally:
            client.close()

    try:
        remote = await run_in_threadpool(fetch)
    except AlbertError as exc:
        if getattr(exc, "reason", None) == "not_found":
            if corpus.owner_user_id == current_user.id:
                albert_access.mark_status(db, corpus, "unreachable")
            return _error(404, "Collection Albert introuvable ou inaccessible avec votre clé.")
        status, body = error_payload(exc)
        return JSONResponse(status_code=status, content=body)
    if corpus.status != "active" and corpus.owner_user_id == current_user.id:
        albert_access.mark_status(db, corpus, "active", checked=True)
    catalogue = {s.document_id: s for s in corpus.sources}
    documents = []
    for doc in remote or []:
        try:
            did = int(doc.get("id"))
        except (TypeError, ValueError):
            continue
        source = catalogue.get(did)
        entry = {"id": did, "name": doc.get("name"), "chunks": doc.get("chunks"), "created": doc.get("created"),
                 "in_catalogue": source is not None}
        if source is not None:
            entry.update({k: v for k, v in source.to_dict().items() if k not in ("document_id", "document_name")})
        documents.append(entry)
    documents.sort(key=lambda d: (str(d.get("title") or d.get("name") or "").lower(), d["id"]))
    total = len(documents)
    size = int(per_page)
    start = (int(page) - 1) * size
    return JSONResponse({
        "success": True, "corpus": corpus.to_dict(), "documents": documents[start:start + size],
        "total": total, "page": int(page), "per_page": size, "pages": (total + size - 1) // size,
    })


@router.delete("/api/albert/corpora/{corpus_id}/documents/{document_id}", include_in_schema=False)
async def delete_corpus_document(
    corpus_id: str,
    document_id: str,
    confirm: str = "",
    _albert_on: None = Depends(_require_albert_enabled),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Erase one document of a corpus on Albert (irreversible, ``?confirm=true``, audited).

    The document must belong to the corpus collection (checked remotely
    before the deletion); the audit entry is written before the call and
    completed with the outcome.

    Returns:
        JSON ``{success, deleted}``.
    """
    from app.models.audit import create_audit_log
    from scripts.rad_albert.errors import AlbertError

    corpus, refusal = albert_access.get_corpus(db, current_user, corpus_id, CorpusAction.WRITE)
    if refusal is not None:
        return _error(*refusal)
    if not document_id.isdigit() or int(document_id) <= 0:
        return _error(400, "Identifiant de document Albert invalide.")
    did = int(document_id)
    if str(confirm or "").strip().lower() not in CONFIRM_VALUES:
        return _error(400, "Suppression définitive du document sur Albert : confirmer avec confirm=true.",
                      confirm_required=True)
    api_key, error = _albert_key_or_403(current_user)
    if error is not None:
        return error
    cfg = _config()

    def check() -> Any:
        """Read the document with the caller's key (membership check)."""
        client = _client(cfg, api_key)
        try:
            return client.get_document(did)
        finally:
            client.close()

    try:
        document = await run_in_threadpool(check)
    except AlbertError as exc:
        if getattr(exc, "reason", None) == "not_found":
            return _error(404, "Document Albert introuvable ou inaccessible avec votre clé.")
        status, body = error_payload(exc)
        return JSONResponse(status_code=status, content=body)
    try:
        belongs = int((document or {}).get("collection_id")) == corpus.collection_id
    except (TypeError, ValueError):
        belongs = False
    if not belongs:
        return _error(404, "Ce document n'appartient pas à la collection du corpus.")
    try:
        entry = create_audit_log(
            db, action=DOCUMENT_DELETE_AUDIT_ACTION, user_id=current_user.id, resource_type="albert_document",
            resource_id=did, details={"corpus_id": corpus.id, "collection_id": corpus.collection_id,
                                      "document_id": did, "outcome": "requested"},
            success=False, error_message="Suppression demandée, résultat en attente de la réponse d'Albert.",
        )
    except Exception as exc:
        logger.error(f"Albert document deletion refused, audit log unavailable: {type(exc).__name__}")
        db.rollback()
        return _error(500, "Journal d'audit indisponible : suppression du document Albert refusée.")

    def delete() -> None:
        """Delete the document with the caller's key."""
        client = _client(cfg, api_key)
        try:
            client.delete_document(did)
        finally:
            client.close()

    outcome_error = None
    try:
        await run_in_threadpool(delete)
    except AlbertError as exc:
        outcome_error = exc
    try:
        entry.success = 1 if outcome_error is None else 0
        entry.error_message = None if outcome_error is None else "Suppression refusée ou en échec côté Albert."
        entry.details = {"corpus_id": corpus.id, "collection_id": corpus.collection_id, "document_id": did,
                         "outcome": "deleted" if outcome_error is None else "failed"}
        db.commit()
    except Exception as exc:
        logger.error(f"Albert document deletion outcome not recorded: {type(exc).__name__}")
        db.rollback()
    if outcome_error is not None:
        status, body = error_payload(outcome_error)
        return JSONResponse(status_code=status, content=body)
    try:
        albert_access.remove_source(db, corpus, did)
        db.commit()
    except Exception as exc:
        logger.error(f"Albert corpus catalogue not updated after deletion: {type(exc).__name__}")
        db.rollback()
    logger.info(f"Albert document {did} of corpus {corpus.id} deleted by user {current_user.id}")
    return JSONResponse({"success": True, "deleted": did})


def summarise_usage(buckets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Daily buckets and totals of ``/v1/usage`` (missing fields stay unknown, never zero).

    Args:
        buckets: ``usage.bucket`` entries.

    Returns:
        ``{days: [{date_utc, start_time, requests, prompt_tokens, completion_tokens, total_tokens, cost, kWh,
        kgCO2eq}], totals: {...}}``.
    """
    days = []
    totals: Dict[str, Any] = {k: None for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens",
                                                "cost", "kWh", "kgCO2eq")}
    for bucket in buckets:
        start = bucket.get("start_time")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            continue
        impacts = bucket.get("impacts") if isinstance(bucket.get("impacts"), dict) else {}
        row = {
            "start_time": int(start),
            "date_utc": datetime.fromtimestamp(int(start), tz=timezone.utc).date().isoformat(),
            "requests": bucket.get("requests"),
            "prompt_tokens": bucket.get("prompt_tokens"),
            "completion_tokens": bucket.get("completion_tokens"),
            "total_tokens": bucket.get("total_tokens"),
            "cost": bucket.get("cost"),
            "kWh": impacts.get("kWh"),
            "kgCO2eq": impacts.get("kgCO2eq"),
        }
        for name in totals:
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[name] = (totals[name] or 0) + value
        days.append(row)
    days.sort(key=lambda r: r["start_time"])
    return {"days": days, "totals": totals}


@router.get("/api/albert/usage", include_in_schema=False)
async def albert_usage(
    days: str = "30",
    _albert_on: None = Depends(_require_albert_enabled),
    current_user: User = Depends(get_current_active_user),
):
    """
    Daily usage of the caller's Albert account (``/v1/usage``, UTC buckets).

    The account usage includes every use of the key, not only RAGpy; ``cost``
    keeps the unit of the service (no currency is assumed).

    Args:
        days: Number of days (1 to 90).

    Returns:
        JSON ``{success, days, totals, start_time, end_time, timezone_note}``.
    """
    from scripts.rad_albert.errors import AlbertError

    if not days.isdigit() or not 1 <= int(days) <= MAX_USAGE_DAYS:
        return _error(400, f"Période invalide : 1 à {MAX_USAGE_DAYS} jours.")
    api_key, error = _albert_key_or_403(current_user)
    if error is not None:
        return error
    cfg = _config()
    end = int(time.time())
    start = end - int(days) * 86400

    def fetch() -> List[Dict[str, Any]]:
        """Read the usage buckets with the caller's key."""
        client = _client(cfg, api_key)
        try:
            return client.usage(start, end)
        finally:
            client.close()

    try:
        buckets = await run_in_threadpool(fetch)
    except AlbertError as exc:
        status, body = error_payload(exc)
        return JSONResponse(status_code=status, content=body)
    summary = summarise_usage(buckets)
    return JSONResponse({
        "success": True, "start_time": start, "end_time": end, **summary,
        "timezone_note": "Seaux journaliers en UTC (affichage en heure de Paris par l'interface).",
        "cost_unit": "unité interne d'Albert",
    })
