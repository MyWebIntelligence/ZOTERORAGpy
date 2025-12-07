"""
Tests for Publish or Perish Parser Module
==========================================

Test suite for publishorperish_parser.py covering:
- Valid citation parsing
- Field validation (year range, authors list, etc.)
- Error handling
- Edge cases

Author: RAGpy Team
Date: 2025-12-05
"""

import pytest
import json
import tempfile
from pathlib import Path
from pydantic import ValidationError

from app.utils.publishorperish_parser import (
    PopCitation,
    parse_pop_json,
    get_citation_summary
)


class TestPopCitationModel:
    """Test suite for PopCitation Pydantic model validation."""

    def test_valid_minimal_citation(self):
        """Test creation of citation with minimal required fields."""
        citation = PopCitation(
            uid="GS:12345",
            title="Test Article",
            authors=["Smith, J."]
        )
        assert citation.uid == "GS:12345"
        assert citation.title == "Test Article"
        assert citation.authors == ["Smith, J."]
        assert citation.year is None
        assert citation.doi is None

    def test_valid_complete_citation(self):
        """Test creation of citation with all fields populated."""
        citation = PopCitation(
            uid="GS:67890",
            title="Deep Learning Advances in NLP",
            source="Nature Machine Intelligence",
            publisher="Springer",
            article_url="https://example.com/article",
            fulltext_url="https://example.com/article.pdf",
            abstract="This is an abstract...",
            year=2024,
            cites=42,
            authors=["Smith, J.", "Doe, A.", "Johnson, B."],
            doi="10.1038/s12345-024-00001-x",
            type="PDF"
        )
        assert citation.title == "Deep Learning Advances in NLP"
        assert citation.year == 2024
        assert citation.cites == 42
        assert len(citation.authors) == 3

    def test_year_validation_valid_range(self):
        """Test year validation accepts valid years."""
        # Boundary values
        citation_min = PopCitation(uid="1", title="Old Paper", authors=[], year=1900)
        citation_max = PopCitation(uid="2", title="Future Paper", authors=[], year=2100)
        citation_current = PopCitation(uid="3", title="Current Paper", authors=[], year=2024)

        assert citation_min.year == 1900
        assert citation_max.year == 2100
        assert citation_current.year == 2024

    def test_year_validation_rejects_too_old(self):
        """Test year validation rejects years before 1900."""
        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="1", title="Ancient Paper", authors=[], year=1899)
        assert "out of valid range" in str(exc_info.value).lower()

    def test_year_validation_rejects_too_future(self):
        """Test year validation rejects years after 2100."""
        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="1", title="Far Future Paper", authors=[], year=2101)
        assert "out of valid range" in str(exc_info.value).lower()

    def test_authors_field_empty_list(self):
        """Test authors field accepts empty list."""
        citation = PopCitation(uid="1", title="No Author Paper", authors=[])
        assert citation.authors == []

    def test_authors_field_single_string_conversion(self):
        """Test single string author is converted to list."""
        citation = PopCitation(uid="1", title="Single Author", authors="Smith, J.")
        assert citation.authors == ["Smith, J."]

    def test_authors_field_none_conversion(self):
        """Test None value for authors is converted to empty list."""
        citation = PopCitation(uid="1", title="Test", authors=None)
        assert citation.authors == []

    def test_cites_validation_positive(self):
        """Test citation count accepts positive values."""
        citation = PopCitation(uid="1", title="Popular Paper", authors=[], cites=100)
        assert citation.cites == 100

    def test_cites_validation_zero(self):
        """Test citation count accepts zero."""
        citation = PopCitation(uid="1", title="New Paper", authors=[], cites=0)
        assert citation.cites == 0

    def test_cites_validation_rejects_negative(self):
        """Test citation count rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="1", title="Test", authors=[], cites=-5)
        assert "cannot be negative" in str(exc_info.value).lower()

    def test_empty_title_rejected(self):
        """Test empty or whitespace-only title is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="1", title="", authors=[])
        assert "non-empty title" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="1", title="   ", authors=[])
        assert "non-empty title" in str(exc_info.value).lower()

    def test_empty_uid_rejected(self):
        """Test empty or whitespace-only uid is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="", title="Test", authors=[])
        assert "non-empty uid" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            PopCitation(uid="   ", title="Test", authors=[])
        assert "non-empty uid" in str(exc_info.value).lower()

    def test_url_validation(self):
        """Test URL fields accept valid URLs."""
        citation = PopCitation(
            uid="1",
            title="Test",
            authors=[],
            article_url="https://example.com/article",
            fulltext_url="https://arxiv.org/pdf/2024.12345.pdf"
        )
        assert str(citation.article_url) == "https://example.com/article"
        assert "arxiv.org" in str(citation.fulltext_url)


class TestParsePopJson:
    """Test suite for parse_pop_json function."""

    def test_parse_valid_json_array(self, tmp_path):
        """Test parsing valid JSON array with multiple citations."""
        # Create test JSON file
        test_data = [
            {
                "uid": "GS:1",
                "title": "First Paper",
                "authors": ["Smith, J."],
                "year": 2020
            },
            {
                "uid": "GS:2",
                "title": "Second Paper",
                "authors": ["Doe, A.", "Johnson, B."],
                "year": 2021,
                "doi": "10.1234/test"
            }
        ]

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(test_data), encoding='utf-8')

        # Parse
        citations = parse_pop_json(str(json_file))

        # Assertions
        assert len(citations) == 2
        assert citations[0].title == "First Paper"
        assert citations[0].year == 2020
        assert citations[1].doi == "10.1234/test"
        assert len(citations[1].authors) == 2

    def test_parse_empty_array_raises_error(self, tmp_path):
        """Test parsing empty JSON array raises ValueError."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]", encoding='utf-8')

        with pytest.raises(ValueError) as exc_info:
            parse_pop_json(str(json_file))
        assert "empty" in str(exc_info.value).lower()

    def test_parse_invalid_json_structure(self, tmp_path):
        """Test parsing non-array JSON raises ValueError."""
        # Object instead of array
        json_file = tmp_path / "object.json"
        json_file.write_text('{"uid": "1", "title": "Test"}', encoding='utf-8')

        with pytest.raises(ValueError) as exc_info:
            parse_pop_json(str(json_file))
        assert "array" in str(exc_info.value).lower()

    def test_parse_invalid_json_syntax(self, tmp_path):
        """Test parsing malformed JSON raises ValueError."""
        json_file = tmp_path / "malformed.json"
        json_file.write_text('[{"uid": "1", "title": }]', encoding='utf-8')

        with pytest.raises(ValueError) as exc_info:
            parse_pop_json(str(json_file))
        assert "json" in str(exc_info.value).lower()

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_pop_json("/nonexistent/path/file.json")

    def test_parse_citation_validation_errors(self, tmp_path):
        """Test parsing handles citation validation errors gracefully."""
        # Create JSON with invalid citation (missing required field)
        test_data = [
            {
                "uid": "GS:1",
                "title": "Valid Paper",
                "authors": []
            },
            {
                "uid": "GS:2",
                # Missing title (required)
                "authors": []
            }
        ]

        json_file = tmp_path / "invalid.json"
        json_file.write_text(json.dumps(test_data), encoding='utf-8')

        with pytest.raises(ValueError) as exc_info:
            parse_pop_json(str(json_file))
        # Should report which citation failed
        assert "Citation 1" in str(exc_info.value)

    def test_parse_multiple_validation_errors(self, tmp_path):
        """Test parsing reports multiple validation errors."""
        test_data = [
            {"uid": "1", "title": "Valid", "authors": []},
            {"uid": "2", "title": "", "authors": []},  # Invalid: empty title
            {"uid": "3", "title": "Test", "authors": [], "year": 1800},  # Invalid: year
            {"uid": "4", "title": "Good", "authors": []},
            {"uid": "5", "title": "Test", "authors": [], "cites": -10}  # Invalid: negative cites
        ]

        json_file = tmp_path / "multi_errors.json"
        json_file.write_text(json.dumps(test_data), encoding='utf-8')

        with pytest.raises(ValueError) as exc_info:
            parse_pop_json(str(json_file))
        error_msg = str(exc_info.value)
        # Should show first 5 errors and count total
        assert "3/5" in error_msg or "3 citations" in error_msg.lower()


class TestGetCitationSummary:
    """Test suite for get_citation_summary function."""

    def test_summary_empty_list(self):
        """Test summary of empty citation list."""
        summary = get_citation_summary([])
        assert summary['total'] == 0
        assert summary['with_doi'] == 0
        assert summary['year_range'] == (None, None)
        assert summary['avg_cites'] == 0

    def test_summary_basic_stats(self):
        """Test summary calculates basic statistics correctly."""
        citations = [
            PopCitation(uid="1", title="A", authors=[], year=2020, cites=10, doi="10.1/a"),
            PopCitation(uid="2", title="B", authors=[], year=2022, cites=20),
            PopCitation(uid="3", title="C", authors=[], year=2021, cites=15, doi="10.1/c"),
        ]

        summary = get_citation_summary(citations)

        assert summary['total'] == 3
        assert summary['with_doi'] == 2
        assert summary['year_range'] == (2020, 2022)
        assert summary['avg_cites'] == pytest.approx(15.0)

    def test_summary_with_optional_fields(self):
        """Test summary handles optional fields correctly."""
        citations = [
            PopCitation(
                uid="1",
                title="A",
                authors=[],
                abstract="Abstract A",
                fulltext_url="https://example.com/a.pdf"
            ),
            PopCitation(
                uid="2",
                title="B",
                authors=[],
                abstract="Abstract B"
            ),
            PopCitation(
                uid="3",
                title="C",
                authors=[]
            ),
        ]

        summary = get_citation_summary(citations)

        assert summary['with_abstract'] == 2
        assert summary['with_fulltext_url'] == 1

    def test_summary_with_none_values(self):
        """Test summary handles None values in optional fields."""
        citations = [
            PopCitation(uid="1", title="A", authors=[], year=None, cites=None),
            PopCitation(uid="2", title="B", authors=[], year=2020, cites=None),
            PopCitation(uid="3", title="C", authors=[], year=None, cites=5),
        ]

        summary = get_citation_summary(citations)

        assert summary['total'] == 3
        assert summary['year_range'] == (2020, 2020)  # Only one year present
        assert summary['avg_cites'] == 5.0  # Only one cite count present


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_realistic_pop_export(self, tmp_path):
        """Test parsing realistic Publish or Perish export."""
        realistic_data = [
            {
                "uid": "GS:scholar:12345",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
                "year": 2017,
                "source": "Advances in Neural Information Processing Systems",
                "cites": 50000,
                "abstract": "The dominant sequence transduction models...",
                "article_url": "https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
                "fulltext_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "type": "PDF"
            },
            {
                "uid": "GS:scholar:67890",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Devlin, J.", "Chang, M.-W.", "Lee, K.", "Toutanova, K."],
                "year": 2019,
                "source": "Proceedings of NAACL-HLT",
                "cites": 40000,
                "doi": "10.18653/v1/N19-1423",
                "article_url": "https://aclanthology.org/N19-1423/",
                "type": "HTML"
            }
        ]

        json_file = tmp_path / "realistic.json"
        json_file.write_text(json.dumps(realistic_data, indent=2), encoding='utf-8')

        # Parse
        citations = parse_pop_json(str(json_file))

        # Verify
        assert len(citations) == 2
        assert citations[0].title == "Attention Is All You Need"
        assert citations[0].cites == 50000
        assert len(citations[0].authors) == 3

        # Summary
        summary = get_citation_summary(citations)
        assert summary['total'] == 2
        assert summary['with_doi'] == 1
        assert summary['avg_cites'] == 45000.0

    def test_incomplete_pop_data(self, tmp_path):
        """Test handling of incomplete PoP exports (missing optional fields)."""
        incomplete_data = [
            {
                "uid": "GS:1",
                "title": "Minimal Citation",
                "authors": None,  # Will be converted to []
                # Missing: year, source, abstract, urls, etc.
            }
        ]

        json_file = tmp_path / "incomplete.json"
        json_file.write_text(json.dumps(incomplete_data), encoding='utf-8')

        # Should parse successfully
        citations = parse_pop_json(str(json_file))

        assert len(citations) == 1
        assert citations[0].title == "Minimal Citation"
        assert citations[0].authors == []
        assert citations[0].year is None
        assert citations[0].abstract is None
