"""
Albert Corpus Access
====================

Single service deciding who may search, list or modify an Albert corpus
(sprint Albert R2, lot L4), and keeping the local registry up to date.

Rules (the remote Albert account decides again with the caller's own key):

* ``READ`` (search, answer, list documents): the corpus owner, every member
  of its project (owner, collaborator, viewer), administrators;
* ``WRITE`` (delete a remote document, change the project): the corpus owner,
  the project owner and collaborators (never a viewer), administrators;
* ``ADMIN`` (detach the corpus from RAGpy): the corpus owner and administrators.

A deleted corpus is refused for every action except listing by its owner.
A corpus shared through a project never lends the owner's Albert key: each
member calls Albert with their own key, so a private collection of another
account answers 404 and is reported as unavailable.

The registry is filled after each upload to an Albert collection from the
upload manifest (partial uploads included) and from the chunks file
(bibliographic fields), or by binding an existing private collection checked
remotely. No API key is ever stored or logged.
"""
import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.albert_corpus import (
    CORPUS_ORIGIN_BIND,
    CORPUS_ORIGIN_UPLOAD,
    CORPUS_STATUS_ACTIVE,
    CORPUS_STATUS_DELETED,
    CORPUS_STATUS_ORPHANED,
    AlbertCorpus,
    AlbertCorpusSource,
)
from app.models.pipeline_session import PipelineSession
from app.models.project import Project, ProjectMember, ProjectRole

logger = logging.getLogger(__name__)


class CorpusAction(str, Enum):
    """What a request does with an Albert corpus."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


CORPUS_NOT_FOUND_MESSAGE = "Corpus Albert introuvable."
CORPUS_DENIED_MESSAGE = "Accès non autorisé à ce corpus Albert."
CORPUS_READ_ONLY_MESSAGE = (
    "Accès en lecture seule à ce corpus : seuls son propriétaire et les collaborateurs du projet "
    "peuvent supprimer des documents."
)
CORPUS_OWNER_ONLY_MESSAGE = "Seul le propriétaire du corpus (ou un administrateur) peut le détacher de RAGpy."
CORPUS_DELETED_MESSAGE = "Ce corpus Albert a été supprimé."

_EDIT_ROLES = (ProjectRole.OWNER.value, ProjectRole.COLLABORATOR.value)
_TEXT_LIMITS = {"document_name": 512, "source_key": 255, "title": 512, "authors": 512, "year": 32,
                "doi": 255, "url": 1024, "filename": 512, "session_folder": 255}


def _project_role(db: Session, project_id: Optional[int], user_id: int) -> Optional[str]:
    """Role of ``user_id`` in the project, or None (no project, unknown project, no access)."""
    if project_id is None:
        return None
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        return None
    return project.get_user_role(user_id)


def corpus_refusal(db: Session, corpus: Optional[AlbertCorpus], user: Any, action: CorpusAction) -> Optional[Tuple[int, str]]:
    """
    Decide whether ``user`` may perform ``action`` on ``corpus``.

    Args:
        db: Database session.
        corpus: The corpus (None when not found).
        user: The authenticated user.
        action: Requested action.

    Returns:
        None when allowed, otherwise ``(status_code, message)`` (404 for an
        unknown corpus or a corpus the user cannot see, 403 otherwise).
    """
    if corpus is None:
        return 404, CORPUS_NOT_FOUND_MESSAGE
    is_admin = bool(getattr(user, "is_admin", False))
    if corpus.status == CORPUS_STATUS_ORPHANED and not is_admin:
        # Owner account deleted: the (reusable) owner id never grants access.
        return 404, CORPUS_NOT_FOUND_MESSAGE
    is_owner = corpus.owner_user_id == getattr(user, "id", None)
    role = None if (is_owner or is_admin) else _project_role(db, corpus.project_id, user.id)
    visible = is_owner or is_admin or role is not None
    if not visible:
        return 404, CORPUS_NOT_FOUND_MESSAGE
    if corpus.status == CORPUS_STATUS_DELETED:
        return 410, CORPUS_DELETED_MESSAGE
    if action == CorpusAction.READ:
        return None
    if action == CorpusAction.WRITE:
        if is_owner or is_admin or role in _EDIT_ROLES:
            return None
        return 403, CORPUS_READ_ONLY_MESSAGE
    if is_owner or is_admin:
        return None
    return 403, CORPUS_OWNER_ONLY_MESSAGE


def get_corpus(db: Session, user: Any, corpus_id: Any, action: CorpusAction) -> Tuple[Optional[AlbertCorpus], Optional[Tuple[int, str]]]:
    """
    Load a corpus and check the access of ``user``.

    Args:
        db: Database session.
        user: The authenticated user.
        corpus_id: Corpus id (any value; invalid ids give 404).
        action: Requested action.

    Returns:
        ``(corpus, None)`` when allowed, otherwise ``(None, (status, message))``.
    """
    try:
        cid = int(corpus_id)
    except (TypeError, ValueError):
        return None, (404, CORPUS_NOT_FOUND_MESSAGE)
    corpus = db.query(AlbertCorpus).filter(AlbertCorpus.id == cid).first()
    refusal = corpus_refusal(db, corpus, user, action)
    if refusal is not None:
        return None, refusal
    return corpus, None


def accessible_project_ids(db: Session, user: Any) -> List[int]:
    """
    Projects whose corpora the user may read (owned or member of).

    Args:
        db: Database session.
        user: The authenticated user.

    Returns:
        Project ids.
    """
    owned = [row[0] for row in db.query(Project.id).filter(Project.owner_id == user.id).all()]
    member = [row[0] for row in db.query(ProjectMember.project_id).filter(ProjectMember.user_id == user.id).all()]
    return sorted(set(owned) | set(member))


def list_corpora(db: Session, user: Any, *, base_url: Optional[str] = None,
                 include_deleted: bool = False) -> List[AlbertCorpus]:
    """
    Corpora visible to ``user`` (own corpora and those of their projects).

    Administrators see their own corpora and those of their projects too (a
    full inventory is available through the admin listing of the database);
    this keeps the search page of an administrator usable.

    Args:
        db: Database session.
        user: The authenticated user.
        base_url: Keep only the corpora of this Albert host (current server configuration).
        include_deleted: Also list deleted corpora (owner only).

    Returns:
        Corpora ordered by name then id.
    """
    projects = accessible_project_ids(db, user)
    clauses = [AlbertCorpus.owner_user_id == user.id]
    if projects:
        clauses.append(AlbertCorpus.project_id.in_(projects))
    query = db.query(AlbertCorpus).filter(or_(*clauses))
    if base_url:
        query = query.filter(AlbertCorpus.base_url == base_url)
    rows = query.all()
    out = []
    for corpus in rows:
        if corpus.status == CORPUS_STATUS_ORPHANED:
            continue
        if corpus.status == CORPUS_STATUS_DELETED and not (include_deleted and corpus.owner_user_id == user.id):
            continue
        out.append(corpus)
    out.sort(key=lambda c: (str(c.collection_name or "").lower(), c.id))
    return out


def find_corpus(db: Session, owner_user_id: int, base_url: str, collection_id: int) -> Optional[AlbertCorpus]:
    """
    Corpus of an owner for an Albert collection (any status).

    Args:
        db: Database session.
        owner_user_id: RAGpy owner.
        base_url: Albert base URL.
        collection_id: Albert collection id.

    Returns:
        The corpus, or None.
    """
    return db.query(AlbertCorpus).filter(
        AlbertCorpus.owner_user_id == owner_user_id,
        AlbertCorpus.base_url == base_url,
        AlbertCorpus.collection_id == int(collection_id),
    ).first()


def register_corpus(
    db: Session,
    *,
    owner_user_id: int,
    collection_id: int,
    base_url: str,
    collection_name: Optional[str] = None,
    key_fingerprint: Optional[str] = None,
    project_id: Optional[int] = None,
    origin: str = CORPUS_ORIGIN_UPLOAD,
    gdpr_ack: bool = False,
    checked: bool = False,
) -> AlbertCorpus:
    """
    Create or refresh the corpus of an Albert collection (no commit).

    An existing corpus (same owner, host and collection) is reactivated and
    updated; a project already set is never removed by a later upload made
    outside the project.

    Args:
        db: Database session.
        owner_user_id: RAGpy owner.
        collection_id: Albert collection id.
        base_url: Albert base URL.
        collection_name: Collection name.
        key_fingerprint: Non-reversible fingerprint of the key used.
        project_id: Project sharing the corpus (None = personal).
        origin: ``upload`` or ``bind``.
        gdpr_ack: True when the retention acknowledgement was given.
        checked: True when the collection was just checked remotely.

    Returns:
        The corpus (flushed, with an id).
    """
    now = datetime.utcnow()
    corpus = find_corpus(db, owner_user_id, base_url, collection_id)
    if corpus is None:
        corpus = AlbertCorpus(
            owner_user_id=owner_user_id, collection_id=int(collection_id), base_url=base_url,
            origin=origin, visibility="private", status=CORPUS_STATUS_ACTIVE,
        )
        db.add(corpus)
    corpus.status = CORPUS_STATUS_ACTIVE
    if collection_name:
        corpus.collection_name = str(collection_name)[:255]
    if key_fingerprint:
        corpus.key_fingerprint = str(key_fingerprint)[:32]
    if project_id is not None:
        corpus.project_id = int(project_id)
    if gdpr_ack and corpus.gdpr_ack_at is None:
        corpus.gdpr_ack_at = now
    if checked:
        corpus.last_checked_at = now
    db.flush()
    return corpus


def _clip(name: str, value: Any) -> Optional[str]:
    """Text value cut to the column size (None for blanks)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text[: _TEXT_LIMITS.get(name, 255)]


def upsert_sources(db: Session, corpus: AlbertCorpus, documents: Iterable[Mapping[str, Any]]) -> int:
    """
    Add or update catalogue entries of a corpus (no commit).

    Args:
        db: Database session.
        corpus: The corpus.
        documents: Dicts with ``document_id`` and catalogue fields
            (``chunk_count`` replaces the stored count when given).

    Returns:
        Number of entries written.
    """
    existing = {s.document_id: s for s in corpus.sources}
    written = 0
    for doc in documents:
        try:
            document_id = int(doc.get("document_id"))
        except (TypeError, ValueError):
            continue
        source = existing.get(document_id)
        if source is None:
            source = AlbertCorpusSource(corpus_id=corpus.id, document_id=document_id)
            corpus.sources.append(source)
            existing[document_id] = source
        for name in ("document_name", "source_key", "title", "authors", "year", "doi", "url", "filename",
                     "session_folder"):
            value = _clip(name, doc.get(name))
            if value is not None:
                setattr(source, name, value)
        if doc.get("chunk_count") is not None:
            try:
                source.chunk_count = int(doc["chunk_count"])
            except (TypeError, ValueError):
                pass
        written += 1
    db.flush()
    return written


def remove_source(db: Session, corpus: AlbertCorpus, document_id: int) -> bool:
    """
    Remove a document from the local catalogue (no commit).

    Args:
        db: Database session.
        corpus: The corpus.
        document_id: Albert document id.

    Returns:
        True when an entry was removed.
    """
    for source in list(corpus.sources):
        if source.document_id == int(document_id):
            corpus.sources.remove(source)
            db.flush()
            return True
    return False


def project_for_session(db: Session, session_folder: Optional[str]) -> Optional[int]:
    """
    Project of a session folder (``PipelineSession`` row), or None.

    Args:
        db: Database session.
        session_folder: Canonical session folder (relative to uploads/).

    Returns:
        The project id, or None for a session outside any project.
    """
    if not session_folder:
        return None
    row = db.query(PipelineSession).filter(PipelineSession.session_folder == session_folder).first()
    return row.project_id if row is not None else None


def document_catalogue_from_chunks(chunks_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Bibliographic fields per Albert document name, read from a chunks file.

    The document name is computed exactly as by the upload
    (``rad_albert.collections.document_name_for``); the random ``doc_id`` is
    never used. Local paths are never kept (file name only).

    Args:
        chunks_path: The chunks JSON sent to Albert (None or unreadable: empty result).

    Returns:
        ``{document_name: {source_key, title, authors, year, doi, url, filename}}``.
    """
    if not chunks_path or not os.path.isfile(chunks_path):
        return {}
    try:
        from scripts.rad_albert.collections import _basename, _year_of, document_name_for
    except ImportError:  # scripts/ itself on sys.path
        from rad_albert.collections import _basename, _year_of, document_name_for
    try:
        with open(chunks_path, encoding="utf-8") as handle:
            chunks = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning(f"Albert corpus catalogue: chunks file unreadable ({type(exc).__name__})")
        return {}
    if isinstance(chunks, Mapping):
        chunks = chunks.get("chunks") or []
    catalogue: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks if isinstance(chunks, list) else []:
        if not isinstance(chunk, Mapping):
            continue
        name = document_name_for(chunk)
        if name in catalogue:
            continue
        item_key = chunk.get("itemKey") or chunk.get("item_key")
        year = chunk.get("year") or _year_of(chunk.get("date"))
        doi = chunk.get("doi")
        url = chunk.get("url")
        catalogue[name] = {
            "source_key": item_key or chunk.get("title"),
            "title": chunk.get("title"),
            "authors": chunk.get("authors"),
            "year": year,
            "doi": doi,
            "url": url,
            "filename": _basename(chunk.get("filename")) or None,
        }
    return catalogue


def manifest_offset(path: Optional[str]) -> int:
    """
    Current size of a manifest (0 when absent), taken just before an upload run.

    The session manifest is append-only across runs and users: only the
    lines written after this offset belong to the run being registered.

    Args:
        path: Manifest path (``albert_manifest.jsonl``).

    Returns:
        The size in bytes.
    """
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def read_manifest_since(path: Optional[str], offset: int = 0) -> List[Dict[str, Any]]:
    """
    Manifest lines (slices and events) written after ``offset`` bytes.

    A file shorter than ``offset`` (rewritten meanwhile) is read from its start.
    Unreadable lines are ignored.

    Args:
        path: Manifest path.
        offset: Byte offset taken before the run (``manifest_offset``).

    Returns:
        The decoded lines, in file order.
    """
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            handle.seek(offset if 0 <= offset <= size else 0)
            data = handle.read()
    except OSError:
        return []
    for raw in data.decode("utf-8", "replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def register_manifest_run(
    db: Session,
    *,
    user: Any,
    manifest_path: Optional[str],
    offset: int,
    base_url: str,
    key_fingerprint: Optional[str],
    session_folder: Optional[str],
    chunks_path: Optional[str] = None,
) -> List[AlbertCorpus]:
    """
    Register the corpora written by ONE upload run (lines after ``offset``), then commit.

    Used by ``/upload_db`` and by the Celery upload task, success or failure:
    the live slices of the run (documents rolled back excluded) are recorded
    under the uploader, whose key wrote them.

    Args:
        db: Database session.
        user: The uploader.
        manifest_path: Session manifest.
        offset: Manifest size before the run.
        base_url: Albert base URL.
        key_fingerprint: Fingerprint of the uploader's key.
        session_folder: Canonical session folder.
        chunks_path: Chunks file sent (bibliographic fields).

    Returns:
        The registered corpora (empty when the run wrote no slice).
    """
    live = live_manifest_slices(read_manifest_since(manifest_path, offset))
    if not live:
        return []
    return register_upload(
        db, user=user, manifest_rows=live, base_url=base_url, key_fingerprint=key_fingerprint,
        session_folder=session_folder, chunks_path=chunks_path,
    )


def live_manifest_slices(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Successful slices of a manifest whose document was not rolled back afterwards.

    A rollback event (``event`` line) removes the slices written before it for
    the same document; slices written after it (a later run) are kept.

    Args:
        rows: Manifest lines read with events (``read_manifest(..., include_events=True)``).

    Returns:
        The live slice rows, in file order.
    """
    by_document: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    order: List[Tuple[Any, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = (row.get("collection_id"), row.get("document_id"))
        if "event" in row:
            if str(row.get("event")).lower().startswith("rollback"):
                by_document[key] = []
            continue
        if key not in by_document:
            by_document[key] = []
            order.append(key)
        elif key not in order:
            order.append(key)
        by_document[key].append(dict(row))
    out: List[Dict[str, Any]] = []
    for key in order:
        out.extend(by_document.get(key, []))
    return out


def documents_from_manifest(rows: Iterable[Mapping[str, Any]], catalogue: Mapping[str, Mapping[str, Any]],
                            session_folder: Optional[str]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Group manifest rows by collection then document.

    Args:
        rows: Successful slices (``read_manifest`` without events).
        catalogue: Fields per document name (``document_catalogue_from_chunks``).
        session_folder: Session of the upload (traceability).

    Returns:
        ``{collection_id: {document_id: {document_id, document_name, chunk_count, …}}}``.
    """
    grouped: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        try:
            collection_id = int(row.get("collection_id"))
            document_id = int(row.get("document_id"))
        except (TypeError, ValueError):
            continue
        documents = grouped.setdefault(collection_id, {})
        entry = documents.get(document_id)
        if entry is None:
            name = row.get("document_name")
            entry = {"document_id": document_id, "document_name": name, "chunk_count": 0,
                     "session_folder": session_folder, "_collection_name": row.get("collection_name")}
            entry.update(dict(catalogue.get(name) or {}))
            documents[document_id] = entry
        try:
            entry["chunk_count"] += int(row.get("count") or 0)
        except (TypeError, ValueError):
            pass
    return grouped


def register_upload(
    db: Session,
    *,
    user: Any,
    manifest_rows: Iterable[Mapping[str, Any]],
    base_url: str,
    key_fingerprint: Optional[str],
    session_folder: Optional[str],
    chunks_path: Optional[str] = None,
    gdpr_ack: bool = True,
) -> List[AlbertCorpus]:
    """
    Register the corpora and documents of an Albert upload, then commit.

    Called after ``rad_vectordb.py --db albert`` (success or failure: the
    successful slices of a partial upload are registered too, so the remote
    data can always be found and erased). The project of the session, when
    there is one, shares the corpus.

    Args:
        db: Database session.
        user: The uploader.
        manifest_rows: Successful slices of the run.
        base_url: Albert base URL of the upload.
        key_fingerprint: Fingerprint of the uploader's key.
        session_folder: Canonical session folder.
        chunks_path: Chunks file sent (bibliographic fields).
        gdpr_ack: Retention acknowledgement given.

    Returns:
        The registered corpora (empty when the manifest has no slice).
    """
    rows = list(manifest_rows)
    if not rows:
        return []
    catalogue = document_catalogue_from_chunks(chunks_path)
    grouped = documents_from_manifest(rows, catalogue, session_folder)
    project_id = project_for_session(db, session_folder)
    corpora = []
    try:
        for collection_id, documents in grouped.items():
            name = next((d.get("_collection_name") for d in documents.values() if d.get("_collection_name")), None)
            corpus = register_corpus(
                db, owner_user_id=user.id, collection_id=collection_id, base_url=base_url,
                collection_name=name, key_fingerprint=key_fingerprint, project_id=project_id,
                origin=CORPUS_ORIGIN_UPLOAD, gdpr_ack=gdpr_ack, checked=True,
            )
            known = {s.document_id: s.chunk_count or 0 for s in corpus.sources}
            for doc in documents.values():
                doc.pop("_collection_name", None)
                previous = known.get(doc["document_id"])
                if previous is not None and previous > doc["chunk_count"]:
                    doc["chunk_count"] = previous
            upsert_sources(db, corpus, documents.values())
            corpora.append(corpus)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return corpora


def bind_corpus(
    db: Session,
    *,
    user: Any,
    collection: Mapping[str, Any],
    base_url: str,
    key_fingerprint: Optional[str],
    project_id: Optional[int] = None,
) -> AlbertCorpus:
    """
    Register an existing private collection already checked remotely, then commit.

    Args:
        db: Database session.
        user: The caller (becomes the owner).
        collection: Remote collection (``id``, ``name``, ``visibility``).
        base_url: Albert base URL.
        key_fingerprint: Fingerprint of the caller's key.
        project_id: Project to share with (the caller must be able to edit it).

    Returns:
        The corpus.

    Raises:
        ValueError: Collection not private, or project not editable by the caller.
    """
    if str(collection.get("visibility") or "").lower() != "private":
        raise ValueError("Seules les collections privées peuvent être rattachées à RAGpy.")
    if project_id is not None:
        role = _project_role(db, project_id, user.id)
        if role not in _EDIT_ROLES and not getattr(user, "is_admin", False):
            raise ValueError("Projet introuvable ou non modifiable par cet utilisateur.")
    try:
        corpus = register_corpus(
            db, owner_user_id=user.id, collection_id=int(collection["id"]), base_url=base_url,
            collection_name=collection.get("name"), key_fingerprint=key_fingerprint,
            project_id=project_id, origin=CORPUS_ORIGIN_BIND, checked=True,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return corpus


def mark_status(db: Session, corpus: AlbertCorpus, status: str, *, checked: bool = False) -> None:
    """
    Change the status of a corpus and commit (never raises).

    Args:
        db: Database session.
        corpus: The corpus.
        status: New status.
        checked: Record a successful remote check now.
    """
    try:
        corpus.status = status
        if checked:
            corpus.last_checked_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        logger.error(f"Albert corpus {corpus.id}: status not recorded ({type(exc).__name__})")
        try:
            db.rollback()
        except Exception:
            pass


def detach_corpus(db: Session, corpus: AlbertCorpus) -> None:
    """
    Remove a corpus and its catalogue from RAGpy (the remote collection is kept), then commit.

    Args:
        db: Database session.
        corpus: The corpus.
    """
    try:
        db.delete(corpus)
        db.commit()
    except Exception:
        db.rollback()
        raise


def orphan_user_corpora(db: Session, user_id: int) -> int:
    """
    Mark the corpora of a deleted user account ``orphaned`` (no commit).

    SQLite reuses the highest ids and its foreign keys are not enforced: the
    owner id of these rows could later designate another account. Orphaned
    corpora are invisible to everyone but administrators; their inventory is
    kept in the ``USER_DELETE`` audit entry.

    Args:
        db: Database session.
        user_id: Deleted user.

    Returns:
        Number of corpora marked.
    """
    rows = db.query(AlbertCorpus).filter(
        AlbertCorpus.owner_user_id == user_id,
        AlbertCorpus.status.notin_([CORPUS_STATUS_DELETED, CORPUS_STATUS_ORPHANED]),
    ).all()
    for corpus in rows:
        corpus.status = CORPUS_STATUS_ORPHANED
        corpus.project_id = None
    db.flush()
    return len(rows)


def detach_project_corpora(db: Session, project_id: int) -> int:
    """
    Unshare the corpora of a deleted project (``project_id`` set to NULL, no commit).

    Without this, a later project reusing the same id would inherit read and
    write rights on them (SQLite foreign keys are not enforced).

    Args:
        db: Database session.
        project_id: Deleted project.

    Returns:
        Number of corpora updated.
    """
    rows = db.query(AlbertCorpus).filter(AlbertCorpus.project_id == project_id).all()
    for corpus in rows:
        corpus.project_id = None
    db.flush()
    return len(rows)


def mark_collection_deleted(db: Session, base_url: str, collection_id: int) -> int:
    """
    Mark every corpus of a deleted Albert collection ``deleted``, then commit (never raises).

    Args:
        db: Database session.
        base_url: Albert base URL.
        collection_id: Deleted collection.

    Returns:
        Number of corpora marked.
    """
    try:
        rows = db.query(AlbertCorpus).filter(
            AlbertCorpus.base_url == base_url, AlbertCorpus.collection_id == int(collection_id)
        ).all()
        for corpus in rows:
            corpus.status = CORPUS_STATUS_DELETED
        db.commit()
        return len(rows)
    except Exception as exc:
        logger.error(f"Albert corpus registry not updated after the collection deletion: {type(exc).__name__}")
        try:
            db.rollback()
        except Exception:
            pass
        return 0
