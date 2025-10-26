"""
Unit tests for LLM note generator.

These tests use mocking to avoid real LLM API calls.
Run with: pytest tests/test_llm_note_generator.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.utils import llm_note_generator


class TestDetectLanguage:
    """Test language detection."""

    def test_explicit_french(self):
        """Test explicit French language."""
        metadata = {"language": "fr"}
        lang = llm_note_generator._detect_language(metadata)
        assert lang == "fr"

    def test_explicit_english(self):
        """Test explicit English language."""
        metadata = {"language": "en-US"}
        lang = llm_note_generator._detect_language(metadata)
        assert lang == "en"

    def test_default_language(self):
        """Test default to French when no language."""
        metadata = {}
        lang = llm_note_generator._detect_language(metadata)
        assert lang == "fr"

    def test_unsupported_language(self):
        """Test unsupported language defaults to French."""
        metadata = {"language": "zh"}
        lang = llm_note_generator._detect_language(metadata)
        assert lang == "fr"


class TestBuildPrompt:
    """Test prompt building."""

    def test_complete_metadata(self):
        """Test prompt with complete metadata."""
        metadata = {
            "title": "Test Article",
            "authors": "Smith, J.",
            "date": "2024",
            "abstract": "This is an abstract",
            "doi": "10.1234/test",
            "url": "https://example.com"
        }

        prompt = llm_note_generator._build_prompt(
            metadata,
            "Full text content",
            "en"
        )

        assert "Test Article" in prompt
        assert "Smith, J." in prompt
        assert "2024" in prompt
        assert "This is an abstract" in prompt
        assert "Full text content" in prompt
        assert "English" in prompt

    def test_minimal_metadata(self):
        """Test prompt with minimal metadata."""
        metadata = {"title": "Minimal"}

        prompt = llm_note_generator._build_prompt(
            metadata,
            "Text",
            "fr"
        )

        assert "Minimal" in prompt
        assert "français" in prompt


class TestSentinelFunctions:
    """Test sentinel-related functions."""

    def test_sentinel_in_html_positive(self):
        """Test finding sentinel in HTML."""
        html = "<!-- ragpy-note-id:test-123 --><p>Content</p>"
        assert llm_note_generator.sentinel_in_html(html) is True

    def test_sentinel_in_html_negative(self):
        """Test not finding sentinel in HTML."""
        html = "<p>Regular content</p>"
        assert llm_note_generator.sentinel_in_html(html) is False

    def test_extract_sentinel(self):
        """Test extracting sentinel from HTML."""
        html = "<!-- ragpy-note-id:abc-123-def --><p>Content</p>"
        sentinel = llm_note_generator.extract_sentinel_from_html(html)
        assert sentinel == "ragpy-note-id:abc-123-def"

    def test_extract_sentinel_none(self):
        """Test extracting sentinel when none present."""
        html = "<p>No sentinel</p>"
        sentinel = llm_note_generator.extract_sentinel_from_html(html)
        assert sentinel is None


class TestFallbackTemplate:
    """Test fallback template generation."""

    def test_french_template(self):
        """Test French template generation."""
        metadata = {
            "title": "Test Article",
            "authors": "Smith, J.",
            "date": "2024",
            "abstract": "This is the abstract"
        }

        html = llm_note_generator._fallback_template(metadata, "fr")

        assert "Test Article" in html
        assert "Smith, J." in html
        assert "2024" in html
        assert "Fiche de lecture" in html
        assert "Problématique" in html
        assert "à compléter" in html

    def test_english_template(self):
        """Test English template generation."""
        metadata = {
            "title": "Test Article",
            "authors": "Doe, J.",
            "date": "2024"
        }

        html = llm_note_generator._fallback_template(metadata, "en")

        assert "Reading Note" in html
        assert "Research Question" in html
        assert "to be completed" in html


class TestBuildNoteHtml:
    """Test main note building function."""

    def test_template_mode(self):
        """Test building note with template (no LLM)."""
        metadata = {
            "title": "Test",
            "authors": "Author",
            "date": "2024",
            "language": "fr"
        }

        sentinel, html = llm_note_generator.build_note_html(
            metadata,
            text_content="Test content",
            use_llm=False
        )

        # Check sentinel format
        assert sentinel.startswith("ragpy-note-id:")

        # Check HTML contains sentinel and content
        assert sentinel in html
        assert "Test" in html
        assert "<!-- ragpy-note-id:" in html

    @patch('app.utils.llm_note_generator.openai_client')
    def test_llm_mode_openai(self, mock_client):
        """Test building note with OpenAI LLM."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "<p><strong>Ref:</strong> Test Article</p><p>Content</p>"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        metadata = {
            "title": "Test Article",
            "authors": "Smith",
            "date": "2024",
            "language": "en"
        }

        sentinel, html = llm_note_generator.build_note_html(
            metadata,
            text_content="Full text",
            model="gpt-4o-mini",
            use_llm=True
        )

        # Check LLM was called
        mock_client.chat.completions.create.assert_called_once()

        # Check output
        assert sentinel.startswith("ragpy-note-id:")
        assert "Test Article" in html
        assert sentinel in html

    @patch('app.utils.llm_note_generator.openrouter_client')
    def test_llm_mode_openrouter(self, mock_client):
        """Test building note with OpenRouter LLM."""
        # Mock OpenRouter response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "<p>Generated content</p>"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        metadata = {
            "title": "Test",
            "language": "fr"
        }

        sentinel, html = llm_note_generator.build_note_html(
            metadata,
            text_content="Text",
            model="openai/gemini-2.5-flash",  # OpenRouter format
            use_llm=True
        )

        # Check LLM was called
        mock_client.chat.completions.create.assert_called_once()

        # Check output
        assert sentinel.startswith("ragpy-note-id:")
        assert "Generated content" in html

    def test_no_content_fallback(self):
        """Test fallback when no content available."""
        metadata = {
            "title": "Test",
            "language": "fr"
        }

        sentinel, html = llm_note_generator.build_note_html(
            metadata,
            text_content=None,  # No text
            use_llm=True
        )

        # Should use template fallback
        assert sentinel.startswith("ragpy-note-id:")
        assert "Test" in html
        assert "Fiche de lecture" in html


class TestGenerateWithLlm:
    """Test LLM generation function."""

    @patch('app.utils.llm_note_generator.openai_client')
    def test_openai_generation(self, mock_client):
        """Test generation with OpenAI."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        result = llm_note_generator._generate_with_llm(
            prompt="Test prompt",
            model="gpt-4o-mini"
        )

        assert result == "Generated text"
        mock_client.chat.completions.create.assert_called_once()

    @patch('app.utils.llm_note_generator.openrouter_client')
    def test_openrouter_generation(self, mock_client):
        """Test generation with OpenRouter."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "OpenRouter generated"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        result = llm_note_generator._generate_with_llm(
            prompt="Test prompt",
            model="anthropic/claude-3-5-haiku"  # OpenRouter format
        )

        assert result == "OpenRouter generated"
        mock_client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
