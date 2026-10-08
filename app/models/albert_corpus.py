"""
Albert Corpus Models
====================

Local registry of the Albert (DINUM) collections that RAGpy users may search
(sprint Albert R2, lot L4).

An Albert collection id sent by the browser is never an authorization: the
search and answer routes only accept a corpus registered here, then check the
remote access again with the caller's own Albert key. A corpus is created
automatically after an upload to an Albert collection (``/upload_db`` with
``db_choice=albert``, partial uploads included, so the remote data can always
be found and erased), or explicitly by binding an existing private collection
the caller's key can read.

Key Components:
- `AlbertCorpus`: one Albert collection known to RAGpy (owner, optional
  project, host, collection id, non-reversible key fingerprint, status).
- `AlbertCorpusSource`: local catalogue of the documents of a corpus (Albert
  document id, stable source key, bibliographic fields, chunk count), kept
  outside ``uploads/`` so a session cleanup never loses it.

No API key is ever stored in these tables.
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database.base import Base

CORPUS_STATUS_ACTIVE = "active"
CORPUS_STATUS_UNREACHABLE = "unreachable"
CORPUS_STATUS_DELETED = "deleted"
CORPUS_STATUS_ORPHANED = "orphaned"
"""Owner account deleted with ``?albert_data=keep``: invisible to everyone but administrators
(SQLite reuses user ids, so the dangling owner id must never grant access)."""
CORPUS_STATUSES = (CORPUS_STATUS_ACTIVE, CORPUS_STATUS_UNREACHABLE, CORPUS_STATUS_DELETED, CORPUS_STATUS_ORPHANED)

CORPUS_ORIGIN_UPLOAD = "upload"
CORPUS_ORIGIN_BIND = "bind"


class AlbertCorpus(Base):
    """
    An Albert collection registered in RAGpy.

    Attributes:
        owner_user_id: RAGpy user who uploaded or bound the collection.
        project_id: Optional project sharing the corpus (members may search it
            with their own Albert key; writes stay with owner/collaborators).
        collection_id: Albert collection id.
        collection_name: Collection name when the corpus was registered.
        base_url: Albert base URL (``…/v1``) of the collection.
        key_fingerprint: ``sha256[:12]`` of the key used at registration
            (never the key itself), to detect a key change.
        visibility: ``private`` (RAGpy never writes to another visibility).
        status: ``active``, ``unreachable`` (remote check failed), ``deleted``
            (collection deleted from RAGpy) or ``orphaned`` (owner account deleted).
        origin: ``upload`` (registered after an upload) or ``bind``.
        gdpr_ack_at: Retention acknowledgement time (uploads).
        last_checked_at: Last successful remote check.
    """
    __tablename__ = "albert_corpora"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "base_url", "collection_id", name="uq_albert_corpus_owner_collection"),
        Index("ix_albert_corpora_project", "project_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    collection_id = Column(Integer, nullable=False, index=True)
    collection_name = Column(String(255), nullable=True)
    base_url = Column(String(255), nullable=False)
    key_fingerprint = Column(String(32), nullable=True)
    visibility = Column(String(16), nullable=False, default="private")
    status = Column(String(32), nullable=False, default=CORPUS_STATUS_ACTIVE)
    origin = Column(String(16), nullable=False, default=CORPUS_ORIGIN_UPLOAD)
    gdpr_ack_at = Column(DateTime, nullable=True)
    last_checked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    sources = relationship(
        "AlbertCorpusSource",
        back_populates="corpus",
        cascade="all, delete-orphan",
        order_by="AlbertCorpusSource.id",
    )

    def __repr__(self) -> str:
        return f"<AlbertCorpus id={self.id} collection={self.collection_id} owner={self.owner_user_id}>"

    def to_dict(self) -> dict:
        """
        Serialise the corpus for API responses (never a key, never a local path).

        Returns:
            The public fields of the corpus.
        """
        return {
            "id": self.id,
            "owner_user_id": self.owner_user_id,
            "project_id": self.project_id,
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "visibility": self.visibility,
            "status": self.status,
            "origin": self.origin,
            "gdpr_ack_at": self.gdpr_ack_at.isoformat() if self.gdpr_ack_at else None,
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AlbertCorpusSource(Base):
    """
    A document of an Albert corpus, in the local catalogue.

    Attributes:
        corpus_id: Parent corpus.
        document_id: Albert document id.
        document_name: Albert document name (``ragpy:{itemKey}:{filename}``).
        source_key: Stable source key (item key, else title), never the random ``doc_id``.
        title, authors, year, doi, url, filename: Bibliographic fields (when known).
        chunk_count: Chunks sent for this document (sum of the manifest slices).
        session_folder: Session the document was uploaded from (traceability only).
    """
    __tablename__ = "albert_corpus_sources"
    __table_args__ = (
        UniqueConstraint("corpus_id", "document_id", name="uq_albert_corpus_source_document"),
    )

    id = Column(Integer, primary_key=True, index=True)
    corpus_id = Column(Integer, ForeignKey("albert_corpora.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(Integer, nullable=False)
    document_name = Column(String(512), nullable=True)
    source_key = Column(String(255), nullable=True)
    title = Column(String(512), nullable=True)
    authors = Column(String(512), nullable=True)
    year = Column(String(32), nullable=True)
    doi = Column(String(255), nullable=True)
    url = Column(String(1024), nullable=True)
    filename = Column(String(512), nullable=True)
    chunk_count = Column(Integer, nullable=True)
    session_folder = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    corpus = relationship("AlbertCorpus", back_populates="sources")

    def __repr__(self) -> str:
        return f"<AlbertCorpusSource corpus={self.corpus_id} document={self.document_id}>"

    def to_dict(self) -> dict:
        """
        Serialise the source for API responses.

        Returns:
            The catalogue fields of the document.
        """
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "source_key": self.source_key,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "filename": self.filename,
            "chunk_count": self.chunk_count,
        }
