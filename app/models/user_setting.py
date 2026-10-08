"""
User Setting Model
==================

Personal choices of a user (sprint « configuration unifiée », lot L9): the
server and model of each service (``LLM_RECODE_SERVER``, ``LLM_NOTES_MODEL``,
``OCR_SERVER``…). An empty or missing value inherits the server value of the
``.env``. API keys are never stored here (they stay in the encrypted
``users.api_credentials``): these values are not secrets, and keeping them out
of the credentials registry spares them the non-admin isolation of
``build_subprocess_env`` (which would strip the server values).
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from app.database.base import Base


class UserSetting(Base):
    """One personal setting (``name`` = variable of the registry, ``value`` = choice)."""

    __tablename__ = "user_settings"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_settings_user_name"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    value = Column(String(500), nullable=False, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        """Short representation (the value is not a secret)."""
        return f"<UserSetting user={self.user_id} {self.name}>"
