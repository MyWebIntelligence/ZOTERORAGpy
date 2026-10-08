"""
Unit tests for LLM note generator.

These tests use mocking to avoid real LLM API calls.
Run with: pytest tests/test_llm_note_generator.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.utils import llm_note_generator

# Fake keys passed explicitly: the generator never falls back to .env here.
FAKE_OPENAI_KEY = "fake-openai-key"
FAKE_OPENROUTER_KEY = "fake-openrouter-key"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _chat_response(content):
    """Build a chat.completions response object whose first choice carries ``content``."""
    message = Mock()
    message.content = content
    choice = Mock()
    choice.message = message
    response = Mock()
    response.choices = [choice]
    return response


@pytest.fixture
def llm_clients(tmp_path):
    """Patch the ``OpenAI`` class used by the generator with two fake clients.

    The client built with the OpenRouter ``base_url`` is ``clients["openrouter"]``,
    any other is ``clients["openai"]``. ``clients["built_with"]`` records, per
    client, whether it was built with the expected fake key (a boolean, never
    the key itself). ``.env`` discovery points at an absent file, so the host
    ``.env`` is never read.
    """
    clients = {"openai": MagicMock(name="openai_client"),
               "openrouter": MagicMock(name="openrouter_client"),
               "built_with": {}}

    def _factory(*args, **kwargs):
        """Return the fake client matching the requested endpoint."""
        if kwargs.get("base_url") == OPENROUTER_BASE_URL:
            clients["built_with"]["openrouter"] = kwargs.get("api_key") == FAKE_OPENROUTER_KEY
            return clients["openrouter"]
        clients["built_with"]["openai"] = kwargs.get("api_key") == FAKE_OPENAI_KEY
        return clients["openai"]

    absent_dotenv = str(tmp_path / "absent.env")
    with patch("app.utils.llm_note_generator.OpenAI", side_effect=_factory), \
            patch.object(llm_note_generator, "find_dotenv", lambda *a, **k: absent_dotenv, create=True):
        yield clients


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

    def test_template_mode(self, llm_clients):
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
            use_llm=False,
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=FAKE_OPENROUTER_KEY
        )

        # use_llm=False: no LLM call at all
        llm_clients["openai"].chat.completions.create.assert_not_called()
        llm_clients["openrouter"].chat.completions.create.assert_not_called()

        # Check sentinel format
        assert sentinel.startswith("ragpy-note-id:")

        # Check HTML contains sentinel and content
        assert sentinel in html
        assert "Test" in html
        assert "<!-- ragpy-note-id:" in html

    def test_llm_mode_openai(self, llm_clients):
        """Test building note with OpenAI LLM (explicit keys, plain model name)."""
        openai_client = llm_clients["openai"]
        openai_client.chat.completions.create.return_value = _chat_response(
            "<p><strong>Ref:</strong> Test Article</p><p>Content</p>"
        )

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
            use_llm=True,
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=FAKE_OPENROUTER_KEY
        )

        # Check LLM was called, on the OpenAI client only, with the requested model
        openai_client.chat.completions.create.assert_called_once()
        assert openai_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"
        llm_clients["openrouter"].chat.completions.create.assert_not_called()
        assert llm_clients["built_with"]["openai"] is True

        # Check output
        assert sentinel.startswith("ragpy-note-id:")
        assert "Test Article" in html
        assert sentinel in html

    def test_llm_mode_openrouter(self, llm_clients):
        """Test building note with OpenRouter LLM (``provider/model`` name)."""
        openrouter_client = llm_clients["openrouter"]
        openrouter_client.chat.completions.create.return_value = _chat_response("<p>Generated content</p>")

        metadata = {
            "title": "Test",
            "language": "fr"
        }

        sentinel, html = llm_note_generator.build_note_html(
            metadata,
            text_content="Text",
            model="google/gemini-2.5-flash",  # OpenRouter format
            use_llm=True,
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=FAKE_OPENROUTER_KEY
        )

        # Check LLM was called, on the OpenRouter client only
        openrouter_client.chat.completions.create.assert_called_once()
        assert openrouter_client.chat.completions.create.call_args.kwargs["model"] == "google/gemini-2.5-flash"
        llm_clients["openai"].chat.completions.create.assert_not_called()
        assert llm_clients["built_with"]["openrouter"] is True

        # Check output
        assert sentinel.startswith("ragpy-note-id:")
        assert "Generated content" in html

    def test_llm_failure_falls_back_to_template(self, llm_clients):
        """An LLM error never breaks note building: the template is used instead."""
        llm_clients["openai"].chat.completions.create.side_effect = RuntimeError("boom")
        metadata = {"title": "Test", "language": "fr"}

        with patch("time.sleep"):
            sentinel, html = llm_note_generator.build_note_html(
                metadata,
                text_content="Text",
                model="gpt-4o-mini",
                use_llm=True,
                openai_api_key=FAKE_OPENAI_KEY,
                openrouter_api_key=FAKE_OPENROUTER_KEY
            )

        # One call plus one retry, then the template
        assert llm_clients["openai"].chat.completions.create.call_count == 2
        assert sentinel in html
        assert "Fiche de lecture" in html

    def test_no_content_fallback(self, llm_clients):
        """Test fallback when no content available."""
        metadata = {
            "title": "Test",
            "language": "fr"
        }

        sentinel, html = llm_note_generator.build_note_html(
            metadata,
            text_content=None,  # No text
            use_llm=True,
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=FAKE_OPENROUTER_KEY
        )

        # Nothing to summarise: the LLM is not called
        llm_clients["openai"].chat.completions.create.assert_not_called()

        # Should use template fallback
        assert sentinel.startswith("ragpy-note-id:")
        assert "Test" in html
        assert "Fiche de lecture" in html


class TestGenerateWithLlm:
    """Test LLM generation function."""

    def test_openai_generation(self, llm_clients):
        """Test generation with OpenAI."""
        openai_client = llm_clients["openai"]
        openai_client.chat.completions.create.return_value = _chat_response("Generated text")

        result = llm_note_generator._generate_with_llm(
            prompt="Test prompt",
            model="gpt-4o-mini",
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=FAKE_OPENROUTER_KEY
        )

        assert result == "Generated text"
        openai_client.chat.completions.create.assert_called_once()
        kwargs = openai_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"][-1] == {"role": "user", "content": "Test prompt"}
        llm_clients["openrouter"].chat.completions.create.assert_not_called()

    def test_openrouter_generation(self, llm_clients):
        """Test generation with OpenRouter."""
        openrouter_client = llm_clients["openrouter"]
        openrouter_client.chat.completions.create.return_value = _chat_response("OpenRouter generated")

        result = llm_note_generator._generate_with_llm(
            prompt="Test prompt",
            model="mistralai/mistral-small-3.1-24b-instruct",  # OpenRouter format
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=FAKE_OPENROUTER_KEY
        )

        assert result == "OpenRouter generated"
        openrouter_client.chat.completions.create.assert_called_once()
        kwargs = openrouter_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "mistralai/mistral-small-3.1-24b-instruct"
        llm_clients["openai"].chat.completions.create.assert_not_called()

    def test_openrouter_model_without_openrouter_key_uses_openai(self, llm_clients, monkeypatch):
        """A ``provider/model`` name without an OpenRouter key falls back to OpenAI gpt-4o-mini."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        openai_client = llm_clients["openai"]
        openai_client.chat.completions.create.return_value = _chat_response("Fallback text")

        result = llm_note_generator._generate_with_llm(
            prompt="Test prompt",
            model="google/gemini-2.5-flash",
            openai_api_key=FAKE_OPENAI_KEY,
            openrouter_api_key=""
        )

        assert result == "Fallback text"
        assert openai_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"
        llm_clients["openrouter"].chat.completions.create.assert_not_called()


def _finished_response(content, finish_reason):
    """Chat response whose first choice carries ``content`` and ``finish_reason``."""
    response = _chat_response(content)
    response.choices[0].finish_reason = finish_reason
    return response


class TestTruncatedAnswers:
    """A truncated answer (``finish_reason=length``) is never returned as a note.

    Reasoning models count their reasoning in ``max_tokens``: under a tight
    budget the visible answer can stop mid-sentence, or be empty.
    """

    MODEL = "google/gemini-3.8-flash"

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        """Skip the 2 s pause between attempts."""
        monkeypatch.setattr("time.sleep", lambda seconds: None)

    def _generate(self, mode="short"):
        """Call the generator on the OpenRouter path."""
        return llm_note_generator._generate_with_llm(
            prompt="Test prompt", model=self.MODEL, mode=mode,
            openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
        )

    def _budgets(self, client):
        """``max_tokens`` of every call, in order."""
        return [call.kwargs["max_tokens"] for call in client.chat.completions.create.call_args_list]

    def test_truncated_text_retried_with_larger_budget(self, llm_clients):
        """Cut text on the first call: one retry with a larger budget, its answer returned."""
        client = llm_clients["openrouter"]
        client.chat.completions.create.side_effect = [
            _finished_response("I. CADRAGE ET PROBL", "length"),
            _finished_response("Résumé complet.", "stop"),
        ]

        assert self._generate() == "Résumé complet."
        first, retry = self._budgets(client)
        assert first == llm_note_generator.NOTE_MODE_MAX_TOKENS["short"]
        assert retry == llm_note_generator._truncation_retry_budget(first)
        assert retry > first

    def test_empty_truncated_answer_retried_with_larger_budget(self, llm_clients):
        """Reasoning ate the whole budget (no content): the retry gets a larger budget."""
        client = llm_clients["openrouter"]
        client.chat.completions.create.side_effect = [
            _finished_response(None, "length"),
            _finished_response("Résumé complet.", "stop"),
        ]

        assert self._generate() == "Résumé complet."
        first, retry = self._budgets(client)
        assert retry > first

    def test_truncated_twice_raises(self, llm_clients):
        """Still cut after the retry: an error, never the truncated text."""
        client = llm_clients["openrouter"]
        client.chat.completions.create.side_effect = [
            _finished_response("I. CADRAGE", "length"),
            _finished_response("I. CADRAGE ET", "length"),
        ]

        with pytest.raises(llm_note_generator.LLMTruncatedError):
            self._generate()
        assert client.chat.completions.create.call_count == 2

    def test_truncated_error_is_a_value_error(self):
        """Callers that catch ``ValueError`` keep working."""
        assert issubclass(llm_note_generator.LLMTruncatedError, ValueError)

    @pytest.mark.parametrize("mode", ["short", "extended", "pedagogique", "evaluation", "book"])
    def test_retry_budget_bounds(self, mode):
        """The retry budget is larger than the first one, floored and capped."""
        first = llm_note_generator.NOTE_MODE_MAX_TOKENS[mode]
        retry = llm_note_generator._truncation_retry_budget(first)
        assert retry > first
        assert retry >= llm_note_generator.NOTE_TRUNCATION_RETRY_MIN_TOKENS
        assert retry <= llm_note_generator.NOTE_TRUNCATION_RETRY_MAX_TOKENS

    def test_complete_answer_single_call(self, llm_clients):
        """``finish_reason=stop``: one call, budget of the mode, unchanged behaviour."""
        client = llm_clients["openrouter"]
        client.chat.completions.create.return_value = _finished_response("Résumé.", "stop")

        assert self._generate() == "Résumé."
        assert self._budgets(client) == [llm_note_generator.NOTE_MODE_MAX_TOKENS["short"]]

    def test_short_abstract_truncated_twice_raises(self, llm_clients):
        """``build_abstract_text`` raises: no half summary reaches the Zotero abstract."""
        client = llm_clients["openrouter"]
        client.chat.completions.create.side_effect = [
            _finished_response("I. CADRAGE", "length"),
            _finished_response("I. CADRAGE ET", "length"),
        ]

        with pytest.raises(llm_note_generator.LLMTruncatedError):
            llm_note_generator.build_abstract_text(
                {"title": "Article", "language": "fr"}, text_content="Texte intégral.",
                model=self.MODEL, openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
