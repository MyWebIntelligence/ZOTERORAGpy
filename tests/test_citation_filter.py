"""
Tests for Citation Filter Module
==================================

Test suite for citation_filter.py covering:
- LLM response parsing (JSON, markdown, "NA")
- ItemType validation and mapping
- Author name parsing
- Filter result validation
- Complete filtering workflow with mocks

Author: RAGpy Team
Date: 2025-12-05
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from pydantic import ValidationError

from app.utils.citation_filter import (
    filter_citation_with_llm,
    infer_item_type_from_source,
    parse_author_name,
    parse_authors_list,
    _parse_llm_response,
    _validate_filter_result,
    _build_filter_prompt,
    ZoteroCreator,
    ZoteroItemData,
    CitationFilterResult,
    VALID_ZOTERO_TYPES,
    ITEMTYPE_MAPPINGS
)


class TestZoteroCreator:
    """Test suite for ZoteroCreator Pydantic model."""

    def test_valid_creator(self):
        """Test creation of valid creator."""
        creator = ZoteroCreator(
            creatorType="author",
            firstName="John",
            lastName="Smith"
        )
        assert creator.firstName == "John"
        assert creator.lastName == "Smith"
        assert creator.creatorType == "author"

    def test_creator_strips_whitespace(self):
        """Test creator strips whitespace from names."""
        creator = ZoteroCreator(
            firstName="  Alice  ",
            lastName="  Doe  "
        )
        assert creator.firstName == "Alice"
        assert creator.lastName == "Doe"

    def test_empty_name_rejected(self):
        """Test empty names are rejected."""
        with pytest.raises(ValidationError):
            ZoteroCreator(firstName="", lastName="Smith")

        with pytest.raises(ValidationError):
            ZoteroCreator(firstName="John", lastName="")


class TestZoteroItemData:
    """Test suite for ZoteroItemData Pydantic model."""

    def test_valid_item(self):
        """Test creation of valid Zotero item."""
        item = ZoteroItemData(
            itemType="journalArticle",
            title="Test Article",
            creators=[
                ZoteroCreator(firstName="John", lastName="Smith")
            ],
            date="2024",
            publicationTitle="Nature"
        )
        assert item.itemType == "journalArticle"
        assert item.title == "Test Article"
        assert len(item.creators) == 1

    def test_itemtype_validation_valid(self):
        """Test valid itemTypes are accepted."""
        for valid_type in ["journalArticle", "conferencePaper", "preprint", "book"]:
            item = ZoteroItemData(
                itemType=valid_type,
                title="Test",
                creators=[ZoteroCreator(firstName="A", lastName="B")]
            )
            assert item.itemType == valid_type

    def test_itemtype_mapping(self):
        """Test invalid itemTypes are mapped to valid ones."""
        # "article" should map to "journalArticle"
        item = ZoteroItemData(
            itemType="article",
            title="Test",
            creators=[ZoteroCreator(firstName="A", lastName="B")]
        )
        assert item.itemType == "journalArticle"

        # "conference" should map to "conferencePaper"
        item = ZoteroItemData(
            itemType="conference",
            title="Test",
            creators=[ZoteroCreator(firstName="A", lastName="B")]
        )
        assert item.itemType == "conferencePaper"

    def test_unknown_itemtype_defaults(self):
        """Test unknown itemTypes default to journalArticle."""
        item = ZoteroItemData(
            itemType="unknownType",
            title="Test",
            creators=[ZoteroCreator(firstName="A", lastName="B")]
        )
        assert item.itemType == "journalArticle"


class TestCitationFilterResult:
    """Test suite for CitationFilterResult model."""

    def test_valid_result(self):
        """Test complete valid filter result."""
        result = CitationFilterResult(
            relevance_score=85,
            relevance_reason="Highly relevant to research topic",
            zotero_item=ZoteroItemData(
                itemType="journalArticle",
                title="Test Article",
                creators=[ZoteroCreator(firstName="J", lastName="Smith")],
                date="2024"
            )
        )
        assert result.relevance_score == 85
        assert "relevant" in result.relevance_reason

    def test_score_validation_bounds(self):
        """Test relevance score validation (0-100)."""
        # Valid scores
        for score in [0, 50, 100]:
            result = CitationFilterResult(
                relevance_score=score,
                relevance_reason="Test",
                zotero_item=ZoteroItemData(
                    itemType="journalArticle",
                    title="Test",
                    creators=[ZoteroCreator(firstName="A", lastName="B")]
                )
            )
            assert result.relevance_score == score

        # Invalid scores
        for score in [-1, 101, 200]:
            with pytest.raises(ValidationError):
                CitationFilterResult(
                    relevance_score=score,
                    relevance_reason="Test",
                    zotero_item=ZoteroItemData(
                        itemType="journalArticle",
                        title="Test",
                        creators=[ZoteroCreator(firstName="A", lastName="B")]
                    )
                )


class TestParseLLMResponse:
    """Test suite for LLM response parsing."""

    def test_parse_na_response(self):
        """Test parsing 'NA' response."""
        assert _parse_llm_response("NA") == "NA"
        assert _parse_llm_response("na") == "NA"
        assert _parse_llm_response("  NA  ") == "NA"

    def test_parse_clean_json(self):
        """Test parsing clean JSON object."""
        json_response = '''
        {
            "relevance_score": 90,
            "relevance_reason": "Directly relevant",
            "zotero_item": {
                "itemType": "journalArticle",
                "title": "Test",
                "creators": [],
                "date": "2024"
            }
        }
        '''
        result = _parse_llm_response(json_response)
        assert isinstance(result, dict)
        assert result["relevance_score"] == 90

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        markdown_response = '''
        ```json
        {
            "relevance_score": 75,
            "relevance_reason": "Somewhat relevant",
            "zotero_item": {
                "itemType": "conferencePaper",
                "title": "Test",
                "creators": [],
                "date": "2023"
            }
        }
        ```
        '''
        result = _parse_llm_response(markdown_response)
        assert isinstance(result, dict)
        assert result["relevance_score"] == 75

    def test_parse_json_without_language_tag(self):
        """Test parsing JSON in markdown without language tag."""
        response = '''
        ```
        {"relevance_score": 60, "relevance_reason": "Test", "zotero_item": {"itemType": "journalArticle", "title": "T", "creators": [], "date": "2024"}}
        ```
        '''
        result = _parse_llm_response(response)
        assert isinstance(result, dict)
        assert result["relevance_score"] == 60

    def test_parse_embedded_json(self):
        """Test extracting JSON from text with surrounding content."""
        response = '''
        Here is my analysis:
        {"relevance_score": 80, "relevance_reason": "Very relevant", "zotero_item": {"itemType": "book", "title": "Test", "creators": [], "date": "2024"}}
        This citation is relevant.
        '''
        result = _parse_llm_response(response)
        assert isinstance(result, dict)
        assert result["relevance_score"] == 80

    def test_parse_invalid_json_raises_error(self):
        """Test invalid JSON raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_llm_response('{"invalid": json}')
        assert "Invalid JSON" in str(exc_info.value)

    def test_parse_no_json_raises_error(self):
        """Test response with no JSON raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_llm_response("This is just plain text without JSON")
        assert "No valid JSON" in str(exc_info.value)


class TestValidateFilterResult:
    """Test suite for filter result validation."""

    def test_validate_complete_result(self):
        """Test validation of complete filter result."""
        data = {
            "relevance_score": 85,
            "relevance_reason": "Highly relevant to topic",
            "zotero_item": {
                "itemType": "journalArticle",
                "title": "Machine Learning Advances",
                "creators": [
                    {"creatorType": "author", "firstName": "Alice", "lastName": "Smith"}
                ],
                "date": "2024",
                "publicationTitle": "Nature AI",
                "DOI": "10.1234/test",
                "language": "en",
                "tags": [{"tag": "ML"}, {"tag": "AI"}]
            }
        }

        result = _validate_filter_result(data)
        assert isinstance(result, CitationFilterResult)
        assert result.relevance_score == 85
        assert result.zotero_item.itemType == "journalArticle"

    def test_validate_minimal_result(self):
        """Test validation with minimal required fields."""
        data = {
            "relevance_score": 50,
            "relevance_reason": "Marginally relevant",
            "zotero_item": {
                "itemType": "webpage",
                "title": "Blog Post",
                "creators": [
                    {"firstName": "B", "lastName": "Doe", "creatorType": "author"}
                ]
            }
        }

        result = _validate_filter_result(data)
        assert result.relevance_score == 50
        assert result.zotero_item.itemType == "webpage"

    def test_validate_missing_fields_raises_error(self):
        """Test missing required fields raise ValueError."""
        # Missing relevance_reason
        with pytest.raises(ValueError):
            _validate_filter_result({
                "relevance_score": 70,
                "zotero_item": {"itemType": "book", "title": "T", "creators": []}
            })

        # Missing zotero_item
        with pytest.raises(ValueError):
            _validate_filter_result({
                "relevance_score": 70,
                "relevance_reason": "Test"
            })


class TestInferItemType:
    """Test suite for itemType inference from source."""

    def test_infer_journal(self):
        """Test inference of journal articles."""
        assert infer_item_type_from_source("Nature Machine Intelligence") == "journalArticle"
        assert infer_item_type_from_source("Journal of AI Research") == "journalArticle"
        assert infer_item_type_from_source("IEEE Transactions on Neural Networks") == "journalArticle"

    def test_infer_conference(self):
        """Test inference of conference papers."""
        assert infer_item_type_from_source("Proceedings of ACL 2024") == "conferencePaper"
        assert infer_item_type_from_source("NeurIPS Conference") == "conferencePaper"
        assert infer_item_type_from_source("ICML Workshop on ML") == "conferencePaper"

    def test_infer_preprint(self):
        """Test inference of preprints."""
        assert infer_item_type_from_source("arXiv preprint arXiv:2024.12345") == "preprint"
        assert infer_item_type_from_source("bioRxiv") == "preprint"
        assert infer_item_type_from_source("HAL archives") == "preprint"

    def test_infer_book(self):
        """Test inference of books."""
        assert infer_item_type_from_source("Springer Book Series") == "book"
        assert infer_item_type_from_source("Book Chapter in AI") == "book"

    def test_infer_report(self):
        """Test inference of reports."""
        assert infer_item_type_from_source("Technical Report TR-2024-01") == "report"
        assert infer_item_type_from_source("Working Paper Series") == "report"

    def test_infer_thesis(self):
        """Test inference of thesis."""
        assert infer_item_type_from_source("PhD Thesis, MIT") == "thesis"
        assert infer_item_type_from_source("Doctoral Dissertation") == "thesis"

    def test_infer_default(self):
        """Test default inference when unclear."""
        assert infer_item_type_from_source("Unknown Source") == "journalArticle"
        assert infer_item_type_from_source(None) == "journalArticle"
        assert infer_item_type_from_source("") == "journalArticle"


class TestParseAuthorName:
    """Test suite for author name parsing."""

    def test_parse_lastname_firstname_format(self):
        """Test 'LastName, FirstName' format."""
        creator = parse_author_name("Smith, John")
        assert creator.firstName == "John"
        assert creator.lastName == "Smith"

        creator = parse_author_name("Doe, A.")
        assert creator.firstName == "A."
        assert creator.lastName == "Doe"

    def test_parse_firstname_lastname_format(self):
        """Test 'FirstName LastName' format."""
        creator = parse_author_name("Alice Johnson")
        assert creator.firstName == "Alice"
        assert creator.lastName == "Johnson"

        creator = parse_author_name("H Lin")
        assert creator.firstName == "H"
        assert creator.lastName == "Lin"

    def test_parse_multi_word_firstname(self):
        """Test multi-word first names."""
        creator = parse_author_name("Jean-Pierre Dupont")
        assert creator.firstName == "Jean-Pierre"
        assert creator.lastName == "Dupont"

        creator = parse_author_name("Maria del Carmen Garcia")
        assert creator.firstName == "Maria del Carmen"
        assert creator.lastName == "Garcia"

    def test_parse_single_name(self):
        """Test single name (use as lastName with placeholder firstName)."""
        creator = parse_author_name("Einstein")
        assert creator.firstName == "-"  # Placeholder to satisfy validation
        assert creator.lastName == "Einstein"

    def test_parse_with_whitespace(self):
        """Test parsing handles extra whitespace."""
        creator = parse_author_name("  Smith , John  ")
        assert creator.firstName == "John"
        assert creator.lastName == "Smith"


class TestParseAuthorsList:
    """Test suite for authors list parsing."""

    def test_parse_multiple_authors(self):
        """Test parsing list of multiple authors."""
        authors = ["Smith, J.", "Doe, A.", "Johnson, B."]
        creators = parse_authors_list(authors)

        assert len(creators) == 3
        assert creators[0]["lastName"] == "Smith"
        assert creators[1]["lastName"] == "Doe"
        assert creators[2]["lastName"] == "Johnson"

    def test_parse_empty_list(self):
        """Test parsing empty list returns Unknown."""
        creators = parse_authors_list([])
        assert len(creators) == 1
        assert creators[0]["lastName"] == "Unknown"

    def test_parse_list_with_empty_strings(self):
        """Test list with empty strings are skipped."""
        authors = ["Smith, J.", "", "Doe, A.", "  "]
        creators = parse_authors_list(authors)

        # Should have 2 valid creators
        assert len(creators) == 2
        assert creators[0]["lastName"] == "Smith"
        assert creators[1]["lastName"] == "Doe"


class TestBuildFilterPrompt:
    """Test suite for prompt building."""

    @patch('app.utils.citation_filter._load_filter_prompt_template')
    def test_build_prompt_replaces_placeholders(self, mock_load):
        """Test prompt building replaces all placeholders."""
        mock_load.return_value = """
        Project: {PROJECT_NAME}
        Description: {PROJECT_DESCRIPTION}
        Collection: {COLLECTION_NAME}
        Title: {CITATION_TITLE}
        Authors: {CITATION_AUTHORS}
        Content: {WEB_CONTENT}
        """

        citation = {
            "uid": "GS:123",
            "title": "Test Article",
            "authors": ["Smith, J."],
            "year": 2024,
            "source": "Nature",
            "doi": "10.1234/test",
            "abstract": "Abstract text",
            "cites": 100
        }

        prompt = _build_filter_prompt(
            citation=citation,
            web_content="Full text content",
            web_content_source="pdf",
            project_name="My Research",
            project_description="AI Ethics",
            collection_name="Main Papers",
            collection_description="Core collection"
        )

        assert "My Research" in prompt
        assert "AI Ethics" in prompt
        assert "Main Papers" in prompt
        assert "Test Article" in prompt
        assert "Smith, J." in prompt
        assert "Full text content" in prompt


class TestFilterCitationWithLLM:
    """Test suite for complete filtering workflow."""

    @pytest.mark.asyncio
    @patch('app.utils.citation_filter._call_llm_api')
    @patch('app.utils.citation_filter.get_llm_semaphore')
    async def test_filter_returns_relevant_result(self, mock_semaphore, mock_llm):
        """Test filtering returns relevant citation with full data."""
        # Mock semaphore
        mock_sem = AsyncMock()
        mock_sem._value = 5
        mock_semaphore.return_value = mock_sem

        # Mock LLM response
        llm_response = json.dumps({
            "relevance_score": 90,
            "relevance_reason": "Directly addresses project topic",
            "zotero_item": {
                "itemType": "journalArticle",
                "title": "Machine Learning Review",
                "creators": [
                    {"creatorType": "author", "firstName": "Alice", "lastName": "Smith"}
                ],
                "date": "2024",
                "publicationTitle": "AI Journal",
                "DOI": "10.1234/test",
                "language": "en",
                "tags": [{"tag": "ML"}]
            }
        })
        mock_llm.return_value = llm_response

        citation = {
            "uid": "GS:1",
            "title": "ML Review",
            "authors": ["Smith, A."],
            "year": 2024
        }

        result = await filter_citation_with_llm(
            citation=citation,
            web_content="Content text",
            web_content_source="pdf",
            project_name="ML Research",
            project_description="ML review",
            collection_name="Papers",
            collection_description="Main papers"
        )

        assert isinstance(result, dict)
        assert result["relevance_score"] == 90
        assert result["zotero_item"]["itemType"] == "journalArticle"

    @pytest.mark.asyncio
    @patch('app.utils.citation_filter._call_llm_api')
    @patch('app.utils.citation_filter.get_llm_semaphore')
    async def test_filter_returns_na(self, mock_semaphore, mock_llm):
        """Test filtering returns NA for irrelevant citation."""
        # Mock semaphore
        mock_sem = AsyncMock()
        mock_sem._value = 5
        mock_semaphore.return_value = mock_sem

        # Mock LLM response
        mock_llm.return_value = "NA"

        citation = {
            "uid": "GS:2",
            "title": "Irrelevant Topic",
            "authors": ["Doe, J."],
            "year": 2023
        }

        result = await filter_citation_with_llm(
            citation=citation,
            web_content="",
            web_content_source="none",
            project_name="ML Research",
            project_description="ML",
            collection_name="Papers",
            collection_description="Papers"
        )

        assert result == "NA"

    @pytest.mark.asyncio
    @patch('app.utils.citation_filter._call_llm_api')
    @patch('app.utils.citation_filter.get_llm_semaphore')
    async def test_filter_retries_on_failure(self, mock_semaphore, mock_llm):
        """Test filtering retries on API failure."""
        # Mock semaphore
        mock_sem = AsyncMock()
        mock_sem._value = 5
        mock_semaphore.return_value = mock_sem

        # First call fails, second succeeds
        mock_llm.side_effect = [
            ValueError("API error"),
            "NA"
        ]

        citation = {"uid": "GS:3", "title": "Test", "authors": []}

        result = await filter_citation_with_llm(
            citation=citation,
            web_content="",
            web_content_source="none",
            project_name="Test",
            project_description="Test",
            collection_name="Test",
            collection_description="Test",
            max_retries=1
        )

        # Should succeed after retry
        assert result == "NA"
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    @patch('app.utils.citation_filter._call_llm_api')
    @patch('app.utils.citation_filter.get_llm_semaphore')
    async def test_filter_fails_after_retries(self, mock_semaphore, mock_llm):
        """Test filtering raises error after all retries."""
        # Mock semaphore
        mock_sem = AsyncMock()
        mock_sem._value = 5
        mock_semaphore.return_value = mock_sem

        # All calls fail
        mock_llm.side_effect = ValueError("Persistent error")

        citation = {"uid": "GS:4", "title": "Test", "authors": []}

        with pytest.raises(ValueError) as exc_info:
            await filter_citation_with_llm(
                citation=citation,
                web_content="",
                web_content_source="none",
                project_name="Test",
                project_description="Test",
                collection_name="Test",
                collection_description="Test",
                max_retries=1
            )

        assert "failed" in str(exc_info.value).lower()
