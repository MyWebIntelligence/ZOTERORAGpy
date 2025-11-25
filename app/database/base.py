"""
SQLAlchemy Base Model
=====================

This module defines the `Base` class for all SQLAlchemy models and provides
common mixins for model definitions.

It includes:
- `Base`: The declarative base class.
- `TimestampMixin`: A mixin to add `created_at` and `updated_at` columns to models.
"""
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """A mixin that adds automatic timestamping to SQLAlchemy models.

    This class provides two columns:
    - `created_at`: Automatically set to the current time when a record is created.
    - `updated_at`: Automatically set to the current time whenever a record is
      created or updated.
    """
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
