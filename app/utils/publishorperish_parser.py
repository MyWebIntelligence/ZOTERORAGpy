"""
Publish or Perish Parser Module
=================================

This module handles parsing and validation of Publish or Perish JSON export files.
It uses Pydantic models to ensure data integrity and proper type validation.

Author: RAGpy Team
Date: 2025-12-05
Updated: 2025-12-06 - Added UTF-8 BOM support and auto-generated UIDs
"""

import json
import hashlib
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, field_validator, model_validator
from pathlib import Path


class PopCitation(BaseModel):
    """
    Pydantic model for a Publish or Perish citation.

    Validates citation data from PoP JSON exports with strict type checking
    and reasonable defaults for optional fields.

    Attributes:
        uid (str): Unique identifier for the citation
        title (str): Article/paper title
        source (Optional[str]): Publication source (journal, conference, etc.)
        publisher (Optional[str]): Publisher name
        article_url (Optional[HttpUrl]): URL to article page (HTML)
        fulltext_url (Optional[HttpUrl]): URL to full-text PDF
        abstract (Optional[str]): Article abstract
        year (Optional[int]): Publication year (validated range: 1900-2100)
        cites (Optional[int]): Number of citations
        authors (List[str]): List of author names (required, can be empty)
        doi (Optional[str]): Digital Object Identifier
        type (Optional[str]): Document type (e.g., "PDF", "HTML")

    Examples:
        >>> citation = PopCitation(
        ...     uid="GS:12345",
        ...     title="Deep Learning Advances",
        ...     authors=["Smith, J.", "Doe, A."],
        ...     year=2024
        ... )
        >>> print(citation.title)
        Deep Learning Advances
    """

    uid: Optional[str] = None  # Optional, auto-generated if missing
    title: str
    source: Optional[str] = None
    publisher: Optional[str] = None
    article_url: Optional[HttpUrl] = None
    fulltext_url: Optional[HttpUrl] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    cites: Optional[int] = None
    authors: List[str]  # Required field but can be empty list
    doi: Optional[str] = None
    type: Optional[str] = None

    @field_validator('year', mode='before')
    @classmethod
    def validate_year_range(cls, v) -> Optional[int]:
        """
        Validate publication year is within reasonable range.

        Converts 0 or invalid years to None (PoP uses 0 for unknown years).

        Args:
            v: Year value to validate

        Returns:
            Validated year or None

        Raises:
            ValueError: If year is outside valid range (1900-2100) and not 0
        """
        if v is None or v == 0:
            return None
        if isinstance(v, int) and (v < 1900 or v > 2100):
            # Log warning but don't fail - treat as unknown year
            return None
        return v

    @field_validator('authors', mode='before')
    @classmethod
    def ensure_authors_list(cls, v):
        """
        Ensure authors field is always a list.

        Handles cases where PoP exports may have:
        - None value
        - Single string author
        - Already correct list format

        Args:
            v: Authors value from JSON

        Returns:
            List of author strings
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator('cites')
    @classmethod
    def validate_cites(cls, v: Optional[int]) -> Optional[int]:
        """
        Validate citation count is non-negative.

        Args:
            v: Citation count to validate

        Returns:
            Validated citation count or None

        Raises:
            ValueError: If citation count is negative
        """
        if v is not None and v < 0:
            raise ValueError(f'Citation count cannot be negative: {v}')
        return v

    @model_validator(mode='after')
    def validate_has_minimal_data(self):
        """
        Ensure citation has at least title, and auto-generate uid if missing.

        Auto-generates a uid from title+authors hash if not provided.

        Returns:
            Self if validation passes

        Raises:
            ValueError: If title is missing/empty
        """
        if not self.title or not self.title.strip():
            raise ValueError('Citation must have a non-empty title')

        # Auto-generate uid if missing
        if not self.uid or not self.uid.strip():
            # Create deterministic uid from title + authors
            uid_source = f"{self.title}:{','.join(self.authors or [])}"
            self.uid = f"AUTO:{hashlib.md5(uid_source.encode()).hexdigest()[:12]}"

        return self


def parse_pop_json(json_path: str) -> List[PopCitation]:
    """
    Parse and validate a Publish or Perish JSON export file.

    This function reads a JSON file containing citation data from Publish or Perish,
    validates each citation against the PopCitation model, and returns a list of
    validated citation objects.

    Args:
        json_path: Path to the Publish or Perish JSON export file

    Returns:
        List of validated PopCitation objects

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If JSON is invalid or citations fail validation
        json.JSONDecodeError: If file contains invalid JSON syntax

    Examples:
        >>> citations = parse_pop_json("data/mybiblio.json")
        >>> print(f"Loaded {len(citations)} citations")
        Loaded 42 citations

        >>> for citation in citations:
        ...     print(f"{citation.title} ({citation.year})")
    """
    # Check file exists
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Read and parse JSON (utf-8-sig handles BOM if present)
    try:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in file {json_path}: {str(e)}")

    # Validate structure
    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON array of citations, got {type(data).__name__}. "
            f"Ensure the file contains a top-level array."
        )

    if len(data) == 0:
        raise ValueError(f"JSON file is empty (no citations found)")

    # Parse each citation
    citations = []
    errors = []

    for idx, item in enumerate(data):
        try:
            citation = PopCitation(**item)
            citations.append(citation)
        except Exception as e:
            errors.append(f"Citation {idx} validation failed: {str(e)}")

    # Report errors if any
    if errors:
        error_summary = "\n".join(errors[:5])  # Show first 5 errors
        if len(errors) > 5:
            error_summary += f"\n... and {len(errors) - 5} more errors"
        raise ValueError(
            f"Failed to parse {len(errors)}/{len(data)} citations:\n{error_summary}"
        )

    return citations


def get_citation_summary(citations: List[PopCitation]) -> dict:
    """
    Generate summary statistics for a list of citations.

    Args:
        citations: List of PopCitation objects

    Returns:
        Dictionary with summary statistics including:
        - total: Total number of citations
        - with_doi: Citations with DOI
        - with_abstract: Citations with abstract
        - with_fulltext_url: Citations with PDF URL
        - year_range: Tuple of (min_year, max_year)
        - avg_cites: Average citation count

    Examples:
        >>> summary = get_citation_summary(citations)
        >>> print(f"Total: {summary['total']}, With DOI: {summary['with_doi']}")
    """
    if not citations:
        return {
            'total': 0,
            'with_doi': 0,
            'with_abstract': 0,
            'with_fulltext_url': 0,
            'year_range': (None, None),
            'avg_cites': 0
        }

    years = [c.year for c in citations if c.year is not None]
    cites = [c.cites for c in citations if c.cites is not None]

    return {
        'total': len(citations),
        'with_doi': sum(1 for c in citations if c.doi),
        'with_abstract': sum(1 for c in citations if c.abstract),
        'with_fulltext_url': sum(1 for c in citations if c.fulltext_url),
        'year_range': (min(years) if years else None, max(years) if years else None),
        'avg_cites': sum(cites) / len(cites) if cites else 0
    }
