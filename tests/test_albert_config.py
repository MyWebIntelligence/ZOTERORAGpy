"""Tests du socle Albert : configuration (``rad_albert.config``) et taxonomie
des erreurs (``rad_albert.errors``).

Aucun appel réseau. Les assertions portent sur des booléens et des noms,
jamais sur une valeur de clé.
"""

from __future__ import annotations

import ast
import json
import logging
import pickle
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import rad_albert
from scripts.rad_albert import config as albert_config
from scripts.rad_albert import errors as albert_errors
from scripts.rad_albert.config import (
    DEFAULT_BASE_URL,
    DEFAULT_METADATA_FIELDS,
    ENV_REGISTRY,
    FIELD_ENV_NAMES,
    AlbertConfig,
    normalise_base_url,
    root_url,
    validate_metadata_fields,
)
from scripts.rad_albert.errors import (
    AUTH_REASONS,
    MESSAGES_FR,
    PERMANENT_REASONS,
    QUOTA_REASON,
    AlbertAuthError,
    AlbertDisabledError,
    AlbertError,
    AlbertModelBusy,
    AlbertPermanentError,
    AlbertQuotaExhausted,
    AlbertTransientError,
    AlbertTruncatedError,
    classify_http_error,
    redact,
)

try:
    from tests.albert_fakes import FAKE_ALBERT_KEY
except ImportError:  # module livré en parallèle ; même convention de clé factice
    FAKE_ALBERT_KEY = "fake-albert-key-0001"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "albert"
CONFIG_LOGGER = albert_config.logger.name


class RecordingEnv(Mapping):
    """Mapping d'environnement qui enregistre chaque nom consulté."""

    def __init__(self, data):
        """Initialise avec les variables ``data`` et un journal vide."""
        self._data = dict(data)
        self.accessed = set()

    def __getitem__(self, key):
        """Renvoie la variable ``key`` en notant l'accès."""
        self.accessed.add(key)
        return self._data[key]

    def __iter__(self):
        """Itère sur les noms (non utilisé par ``from_env``)."""
        return iter(self._data)

    def __len__(self):
        """Nombre de variables."""
        return len(self._data)


def _fixture(name):
    """Charge la fixture Albert ``name`` (sans extension)."""
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def _decisions():
    """Charge ``decisions.json`` (valeurs mesurées D1 à D22)."""
    with open(FIXTURES / "decisions.json", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# AlbertConfig
# ---------------------------------------------------------------------------


def test_defaults_off_never_reads_key():
    env = RecordingEnv({"ALBERT_API_KEY": FAKE_ALBERT_KEY})
    cfg = AlbertConfig.from_env(env)
    assert cfg.enabled is False
    assert cfg.ocr_enabled is False
    assert "ALBERT_API_KEY" not in env.accessed
    # Sprint « configuration unifiée » (lot L6) : la superposition des couples
    # EMBEDDING/AUDIO lit seulement LLM_DEFAULT_SERVER en mode historique.
    assert env.accessed == set(FIELD_ENV_NAMES.values()) | {"LLM_DEFAULT_SERVER"}
    assert all(FAKE_ALBERT_KEY not in repr(getattr(cfg, f.name)) for f in fields(cfg))
    assert (FAKE_ALBERT_KEY in repr(cfg)) is False
    assert not any("key" in f.name for f in fields(cfg))
    assert cfg == AlbertConfig() == AlbertConfig.from_env({})


def test_from_env_defaults_to_os_environ(monkeypatch):
    for name in FIELD_ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    assert AlbertConfig.from_env().enabled is False
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_RECODE_RPM", "12")
    cfg = AlbertConfig.from_env()
    assert cfg.enabled is True
    assert cfg.recode_rpm == 12


def test_empty_string_means_unset():
    env = {name: "" for name in FIELD_ENV_NAMES.values()}
    assert AlbertConfig.from_env(env) == AlbertConfig()
    env = {name: "   " for name in FIELD_ENV_NAMES.values()}
    assert AlbertConfig.from_env(env) == AlbertConfig()


def test_config_is_frozen():
    cfg = AlbertConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_defaults_consistent_with_decisions():
    decisions = _decisions()
    d1 = decisions["D1"]["value"]
    cfg = AlbertConfig()
    assert cfg.recode_rpm == d1["ALBERT_RECODE_RPM"] == 45
    assert cfg.notes_rpm == d1["ALBERT_NOTES_RPM"] == 9
    assert cfg.ocr_rpm == d1["ALBERT_OCR_RPM"] == 45
    assert cfg.embed_rpm == d1["ALBERT_EMBED_RPM"] == 450
    assert cfg.chat_tpm == d1["ALBERT_CHAT_TPM"] == 115000
    assert cfg.ocr_mode == decisions["D3"]["value"]["ALBERT_OCR_MODE"] == "auto"
    d4 = decisions["D4"]["value"]
    assert cfg.ocr_part_mb == d4["ALBERT_OCR_PART_MB"]
    assert cfg.ocr_part_pages == d4["ALBERT_OCR_PART_PAGES"]
    assert cfg.timeout_ocr_doc == d4["ALBERT_TIMEOUT_OCR_DOC"]
    d5 = decisions["D5"]["value"]
    assert cfg.ocr_concurrency == d5["ALBERT_OCR_CONCURRENCY"] == 1
    assert cfg.timeout_ocr_page == d5["ALBERT_TIMEOUT_OCR_PAGE"]
    d9 = decisions["D9"]["value"]
    assert cfg.reasoning_headroom == d9["ALBERT_REASONING_HEADROOM"]
    assert cfg.reasoning_effort == d9["ALBERT_REASONING_EFFORT"]
    d12 = decisions["D12"]["value"]
    assert cfg.embed_batch == d12["ALBERT_EMBED_BATCH"] == 64
    assert cfg.embed_l2_normalize is bool(d12["ALBERT_EMBED_L2_NORMALIZE"]) is True
    assert cfg.retry_after_max == decisions["D13"]["value"]["ALBERT_RETRY_AFTER_MAX"]
    assert cfg.push_concurrency == decisions["D19"]["value"]["ALBERT_PUSH_CONCURRENCY"] == 1
    assert cfg.ocr_skip_recode is bool(decisions["D21"]["value"]["ALBERT_OCR_SKIP_RECODE"]) is False
    d22 = decisions["D22"]["value"]
    assert cfg.recode_concurrency == d22["ALBERT_RECODE_CONCURRENCY"] == 2
    assert cfg.notes_concurrency == d22["ALBERT_NOTES_CONCURRENCY"] == 2
    assert cfg.embed_concurrency == d22["ALBERT_EMBED_CONCURRENCY"] == 4


def test_sprint_table_defaults():
    cfg = AlbertConfig()
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.allow_custom_host is False
    assert (cfg.ocr_chat_model, cfg.ocr_doc_model) == ("lightonocr-2-1b", "mistral-ocr-2512")
    assert (cfg.ocr_dpi, cfg.ocr_max_side, cfg.ocr_max_tokens) == (200, 1540, 4096)
    assert (cfg.ocr_temperature, cfg.ocr_top_p) == (0.2, 0.9)
    assert (cfg.ocr_max_pages, cfg.ocr_max_failed_ratio) == (0, 0.5)
    assert (cfg.embed_model, cfg.embed_max_missing_ratio) == ("bge-m3", 0.0)
    assert (cfg.model_fallback, cfg.busy_retries) == (True, 2)
    assert (cfg.process_share, cfg.limiter_backend, cfg.limiter_redis_url) == (1.0, "local", None)
    assert (cfg.max_retries, cfg.retry_backoff, cfg.retry_max_backoff) == (4, 2.0, 60.0)
    assert (cfg.timeout_chat, cfg.timeout_notes, cfg.timeout_embed, cfg.timeout_collections) == (
        120.0,
        300.0,
        60.0,
        120.0,
    )
    assert cfg.subprocess_timeout == 21600
    assert cfg.metadata_fields == DEFAULT_METADATA_FIELDS
    assert len(cfg.metadata_fields) == 10
    assert (cfg.usage_log, cfg.preflight, cfg.token_estimator) == (True, True, "chars")


@pytest.mark.parametrize(
    "raw, expected",
    [("1", True), ("true", True), ("YES", True), (" on ", True), ("0", False), ("no", False), ("maybe", False)],
)
def test_bool_parsing_follows_rad_dedup(raw, expected):
    assert AlbertConfig.from_env({"ALBERT_ENABLED": raw}).enabled is expected


def test_invalid_and_out_of_range_values_fall_back_to_defaults():
    d = AlbertConfig()
    cfg = AlbertConfig.from_env(
        {
            "ALBERT_RECODE_RPM": "abc",
            "ALBERT_NOTES_RPM": "0",
            "ALBERT_CHAT_TPM": "-5",
            "ALBERT_OCR_TEMPERATURE": "nan",
            "ALBERT_OCR_TOP_P": "1.5",
            "ALBERT_OCR_MAX_FAILED_RATIO": "inf",
            "ALBERT_PROCESS_SHARE": "0",
            "ALBERT_TIMEOUT_CHAT": "0",
            "ALBERT_MAX_RETRIES": "-1",
            "ALBERT_OCR_MODE": "bogus",
            "ALBERT_REASONING_EFFORT": "extreme",
            "ALBERT_LIMITER_BACKEND": "sqlite",
            "ALBERT_TOKEN_ESTIMATOR": "magic",
            "ALBERT_EMBED_BATCH": "0",
        }
    )
    assert cfg.recode_rpm == d.recode_rpm
    assert cfg.notes_rpm == d.notes_rpm
    assert cfg.chat_tpm == d.chat_tpm
    assert cfg.ocr_temperature == d.ocr_temperature
    assert cfg.ocr_top_p == d.ocr_top_p
    assert cfg.ocr_max_failed_ratio == d.ocr_max_failed_ratio
    assert cfg.process_share == d.process_share
    assert cfg.timeout_chat == d.timeout_chat
    assert cfg.max_retries == d.max_retries
    assert cfg.ocr_mode == d.ocr_mode
    assert cfg.reasoning_effort == d.reasoning_effort
    assert cfg.limiter_backend == d.limiter_backend
    assert cfg.token_estimator == d.token_estimator
    assert cfg.embed_batch == d.embed_batch


def test_valid_overrides_are_typed():
    cfg = AlbertConfig.from_env(
        {
            "ALBERT_ENABLED": "1",
            "OCR_ENABLE_ALBERT": "1",
            "ALBERT_OCR_MODE": "CHAT",
            "ALBERT_OCR_PART_MB": "7.5",
            "ALBERT_EMBED_BATCH": "500",
            "ALBERT_EMBED_L2_NORMALIZE": "0",
            "ALBERT_PROCESS_SHARE": "0.25",
            "ALBERT_LIMITER_BACKEND": "Redis",
            "ALBERT_LIMITER_REDIS_URL": " redis://cache:6379/2 ",
            "ALBERT_REASONING_EFFORT": "low",
            "ALBERT_TIMEOUT_NOTES": "45",
            "ALBERT_SUBPROCESS_TIMEOUT": "3600",
            "ALBERT_METADATA_FIELDS": "content_id, content_hash,chunk_index,,title,title",
        }
    )
    assert cfg.enabled is True and cfg.ocr_enabled is True
    assert cfg.ocr_mode == "chat"
    assert cfg.ocr_part_mb == 7.5
    assert cfg.embed_batch == 64
    assert cfg.embed_l2_normalize is False
    assert cfg.process_share == 0.25
    assert cfg.limiter_backend == "redis"
    assert cfg.limiter_redis_url == "redis://cache:6379/2"
    assert cfg.reasoning_effort == "low"
    assert cfg.timeout_notes == 45.0 and isinstance(cfg.timeout_notes, float)
    assert cfg.subprocess_timeout == 3600 and isinstance(cfg.subprocess_timeout, int)
    assert cfg.metadata_fields == ("content_id", "content_hash", "chunk_index", "title")


# ---------------------------------------------------------------------------
# URL de base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "/", "//"])
def test_normalise_empty_returns_default(raw):
    assert normalise_base_url(raw) == DEFAULT_BASE_URL


@pytest.mark.parametrize(
    "raw",
    [
        "https://albert.api.etalab.gouv.fr/v1",
        "https://albert.api.etalab.gouv.fr/v1/",
        "https://albert.api.etalab.gouv.fr",
        "https://albert.api.etalab.gouv.fr/",
        "  HTTPS://Albert.API.Etalab.Gouv.Fr/v1  ",
        "albert.api.etalab.gouv.fr/v1",
        "https://albert.api.etalab.gouv.fr:443/v1",
        "https://albert.api.etalab.gouv.fr.//v1",
    ],
)
def test_normalise_slash_root_and_case(raw):
    assert normalise_base_url(raw) == DEFAULT_BASE_URL


def test_normalise_http_forced_to_https(caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        assert normalise_base_url("http://albert.api.etalab.gouv.fr/v1") == DEFAULT_BASE_URL
    assert any("https" in r.getMessage() for r in caplog.records)


def test_normalise_api_albert_host_rewritten_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        out = normalise_base_url("https://api.albert.etalab.gouv.fr/v1")
    assert out == DEFAULT_BASE_URL
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "api.albert.etalab.gouv.fr" in warnings[0].getMessage()


def test_normalise_endpoint_suffix_trimmed_to_v1(caplog):
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        out = normalise_base_url("https://albert.api.etalab.gouv.fr/v1/chat/completions")
    assert out == DEFAULT_BASE_URL
    assert len(caplog.records) == 1


def test_normalise_already_normal_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger=CONFIG_LOGGER):
        normalise_base_url(DEFAULT_BASE_URL)
        normalise_base_url(DEFAULT_BASE_URL + "/")
    assert caplog.records == []


@pytest.mark.parametrize(
    "raw",
    [
        "https://evil.example.com/v1",
        "https://albert.api.etalab.gouv.fr.evil.example.com/v1",
        "https://etalab.gouv.fr/v1",
        "https://127.0.0.1/v1",
        "http://localhost:8000/v1",
        "https://albert.api.etalab.gouv.fr@evil.example.com/v1",
    ],
)
def test_normalise_foreign_host_rejected(raw):
    with pytest.raises(ValueError):
        normalise_base_url(raw)


def test_normalise_userinfo_refused_even_on_official_host():
    secret = "fake-password-0001"
    for raw in (
        f"https://user:{secret}@albert.api.etalab.gouv.fr/v1",
        f"https://{secret}@albert.api.etalab.gouv.fr/v1",
    ):
        with pytest.raises(ValueError) as exc_info:
            normalise_base_url(raw, allow_custom_host=True)
        assert (secret in str(exc_info.value)) is False


@pytest.mark.parametrize(
    "raw",
    [
        "ftp://albert.api.etalab.gouv.fr/v1",
        "file:///etc/passwd",
        "javascript://albert.api.etalab.gouv.fr/v1",
        "https://albert.api.etalab.gouv.fr/v1?api_key=fake-0001",
        "https://albert.api.etalab.gouv.fr/v1#frag",
        "https://albert.api.etalab.gouv.fr:8443/v1",
        "https://albert.api.etalab.gouv.fr/v2",
        "https://albert.api.etalab.gouv.fr/proxy/v1",
        "https://albert.api.etalab.gouv.fr/v1 x",
        "https://albert.api.etalab.gouv.fr:99999/v1",
    ],
)
def test_normalise_rejects_other_schemes_query_port_path(raw):
    with pytest.raises(ValueError):
        normalise_base_url(raw)


def test_normalise_custom_host_flag():
    assert normalise_base_url("https://llm.example.org/v1", allow_custom_host=True) == "https://llm.example.org/v1"
    assert normalise_base_url("http://llm.example.org", allow_custom_host=True) == "https://llm.example.org/v1"
    assert normalise_base_url("https://llm.example.org:8443/api/v1/", allow_custom_host=True) == (
        "https://llm.example.org:8443/api/v1"
    )
    assert normalise_base_url("https://llm.example.org/api", allow_custom_host=True) == (
        "https://llm.example.org/api/v1"
    )
    assert normalise_base_url("https://[::1]:8000/v1", allow_custom_host=True) == "https://[::1]:8000/v1"
    assert normalise_base_url("https://api.albert.etalab.gouv.fr/v1", allow_custom_host=True) == DEFAULT_BASE_URL
    with pytest.raises(ValueError):
        normalise_base_url("https://llm.example.org/v1")


def test_root_url_strips_v1():
    assert root_url(DEFAULT_BASE_URL) == "https://albert.api.etalab.gouv.fr"
    assert root_url(DEFAULT_BASE_URL + "/") == "https://albert.api.etalab.gouv.fr"
    assert root_url("https://llm.example.org:8443/api/v1") == "https://llm.example.org:8443/api"
    assert root_url("") == "https://albert.api.etalab.gouv.fr"


def test_from_env_base_url_rules(caplog):
    with caplog.at_level(logging.DEBUG, logger=CONFIG_LOGGER):
        off = AlbertConfig.from_env({"ALBERT_BASE_URL": "https://evil.example.com/v1"})
        off_rewrite = AlbertConfig.from_env({"ALBERT_BASE_URL": "https://api.albert.etalab.gouv.fr/v1"})
    assert off.base_url == DEFAULT_BASE_URL
    assert off_rewrite.base_url == DEFAULT_BASE_URL
    assert caplog.records == []

    with pytest.raises(ValueError):
        AlbertConfig.from_env({"ALBERT_ENABLED": "1", "ALBERT_BASE_URL": "https://evil.example.com/v1"})

    custom = AlbertConfig.from_env(
        {
            "ALBERT_ENABLED": "1",
            "ALBERT_ALLOW_CUSTOM_HOST": "1",
            "ALBERT_BASE_URL": "https://llm.example.org",
        }
    )
    assert custom.allow_custom_host is True
    assert custom.base_url == "https://llm.example.org/v1"

    raw = "https://api.albert.etalab.gouv.fr/v1/"
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        first = AlbertConfig.from_env({"ALBERT_ENABLED": "1", "ALBERT_BASE_URL": raw})
        second = AlbertConfig.from_env({"ALBERT_ENABLED": "1", "ALBERT_BASE_URL": raw})
    assert first.base_url == second.base_url == DEFAULT_BASE_URL
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) <= 1


# ---------------------------------------------------------------------------
# ENV_REGISTRY
# ---------------------------------------------------------------------------


def test_env_registry_covers_every_field():
    names = [name for name, _default, _meaning in ENV_REGISTRY]
    assert len(names) == len(set(names))
    assert {f.name for f in fields(AlbertConfig)} == set(FIELD_ENV_NAMES)
    assert set(FIELD_ENV_NAMES.values()) <= set(names)
    assert len(set(FIELD_ENV_NAMES.values())) == len(FIELD_ENV_NAMES)
    for extra in ("ALBERT_API_KEY", "EMBEDDING_PROVIDER", "DEDUP_SIM_THRESHOLD_BGE_M3", "OCR_ENABLE_ALBERT"):
        assert extra in names
    assert "ALBERT_LIVE" not in names
    allowed_extra = {"OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER", "DEDUP_SIM_THRESHOLD_BGE_M3"}
    assert all(name.startswith("ALBERT_") or name in allowed_extra for name in names)
    for entry in ENV_REGISTRY:
        assert len(entry) == 3
        assert all(isinstance(part, str) for part in entry)
        assert entry[2].strip() != ""


def test_env_registry_defaults_parse_to_config_defaults():
    env = {name: default for name, default, _meaning in ENV_REGISTRY}
    assert AlbertConfig.from_env(env) == AlbertConfig()


def test_env_registry_is_off_by_default():
    defaults = {name: default for name, default, _meaning in ENV_REGISTRY}
    assert defaults["ALBERT_ENABLED"] == "0"
    assert defaults["OCR_ENABLE_ALBERT"] == "0"
    assert defaults["ALBERT_API_KEY"] == ""
    assert defaults["ALBERT_ALLOW_CUSTOM_HOST"] == "0"
    assert defaults["EMBEDDING_PROVIDER"] == "openai"
    assert defaults["DEDUP_SIM_THRESHOLD_BGE_M3"] == ""
    assert defaults["ALBERT_BASE_URL"] == DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# validate_metadata_fields
# ---------------------------------------------------------------------------


def test_metadata_fields_validation():
    assert validate_metadata_fields(DEFAULT_METADATA_FIELDS, ("title",)) == DEFAULT_METADATA_FIELDS
    assert validate_metadata_fields(",".join(DEFAULT_METADATA_FIELDS), "title") == DEFAULT_METADATA_FIELDS
    assert validate_metadata_fields("content_id, content_hash, chunk_index, doi, doi", ()) == (
        "content_id",
        "content_hash",
        "chunk_index",
        "doi",
    )
    assert validate_metadata_fields(["content_id", "content_hash", "chunk_index"], None) == (
        "content_id",
        "content_hash",
        "chunk_index",
    )
    too_many = DEFAULT_METADATA_FIELDS + ("publisher",)
    with pytest.raises(ValueError, match="10"):
        validate_metadata_fields(too_many, ("title",))
    with pytest.raises(ValueError, match="path"):
        validate_metadata_fields(("content_id", "content_hash", "chunk_index", "path"), ())
    for required in ("content_id", "content_hash", "chunk_index"):
        partial = tuple(name for name in DEFAULT_METADATA_FIELDS if name != required)
        with pytest.raises(ValueError, match=required):
            validate_metadata_fields(partial, ("title",))
    with pytest.raises(ValueError, match="authors"):
        validate_metadata_fields(("content_id", "content_hash", "chunk_index", "title"), "title,authors")
    with pytest.raises(ValueError):
        validate_metadata_fields("", ())


# ---------------------------------------------------------------------------
# Taxonomie des erreurs
# ---------------------------------------------------------------------------


def _classify_fixture(name, **kwargs):
    """Classe la réponse enregistrée dans la fixture ``name``."""
    fx = _fixture(name)
    return classify_http_error(
        fx["status"],
        json.dumps(fx["body"]),
        fx.get("headers") or {},
        endpoint=fx["request"]["path"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "fixture_name, cls, reason",
    [
        ("P6_invalid_key", AlbertAuthError, "invalid_key"),
        ("P6_unknown_model", AlbertPermanentError, "not_found"),
        ("P6_wrong_model_type", AlbertPermanentError, "validation"),
        ("P12_batch_65", AlbertPermanentError, "payload_too_large"),
        ("P12_empty_string", AlbertPermanentError, "bad_request"),
        ("P14_limit_101", AlbertPermanentError, "validation"),
        ("P16_push_65", AlbertPermanentError, "validation"),
        ("P20_v1_health", AlbertPermanentError, "not_found"),
        ("P3_ocr_page0", AlbertPermanentError, "not_found"),
    ],
)
def test_classification_real_fixtures(fixture_name, cls, reason):
    fx = _fixture(fixture_name)
    err = _classify_fixture(fixture_name)
    assert type(err) is cls
    assert err.reason == reason
    assert err.status == fx["status"]
    assert err.endpoint == fx["request"]["path"]
    assert err.detail
    assert len(err.detail) <= albert_errors.DETAIL_MAX_CHARS


def test_p6_details_are_readable():
    assert _classify_fixture("P6_invalid_key").detail == "Invalid API key."
    assert _classify_fixture("P6_unknown_model").detail == "Model not found."
    assert "wrong type" in _classify_fixture("P6_wrong_model_type").detail


def test_validation_detail_never_echoes_input():
    err = _classify_fixture("P16_push_65")
    assert "body.chunks" in err.detail
    assert ("limite push_65" in err.detail) is False
    assert ("content_hash" in err.detail) is False
    assert ("limite push_65" in str(err)) is False


_FUTURE = format_datetime(datetime.now(timezone.utc) + timedelta(hours=2), usegmt=True)
_PAST = format_datetime(datetime.now(timezone.utc) - timedelta(hours=2), usegmt=True)


@pytest.mark.parametrize(
    "status, body, headers, kwargs, cls, reason",
    [
        (400, {"detail": "InsufficientBudgetException: budget exhausted"}, {}, {}, AlbertAuthError, "budget_exhausted"),
        (400, {"detail": "Insufficient budget."}, {}, {}, AlbertAuthError, "budget_exhausted"),
        (400, {"detail": "Key expiration exceeds maximum"}, {}, {}, AlbertPermanentError, "bad_request"),
        (401, {"detail": "Invalid authentication scheme."}, {}, {}, AlbertAuthError, "invalid_key"),
        (
            403,
            {"detail": "Your account has expired. Please contact support to renew your account."},
            {},
            {},
            AlbertAuthError,
            "account_expired",
        ),
        (403, {"detail": "Insufficient permissions."}, {}, {}, AlbertPermanentError, "no_access"),
        (404, {"detail": "Collection not found."}, {}, {}, AlbertPermanentError, "not_found"),
        (409, {"detail": "Key name already exists."}, {}, {}, AlbertPermanentError, "conflict"),
        (413, {"detail": "too large"}, {}, {}, AlbertPermanentError, "payload_too_large"),
        (422, {"detail": "Wrong model type"}, {}, {}, AlbertPermanentError, "validation"),
        (302, "", {"location": "https://elsewhere.example.org"}, {}, AlbertPermanentError, "bad_request"),
        (418, "teapot", {}, {}, AlbertPermanentError, "bad_request"),
        (405, {"detail": "Method Not Allowed"}, {}, {}, AlbertPermanentError, "bad_request"),
    ],
)
def test_classification_matrix_status_based(status, body, headers, kwargs, cls, reason):
    body_text = body if isinstance(body, str) else json.dumps(body)
    err = classify_http_error(status, body_text, headers, endpoint="/v1/x", **kwargs)
    assert type(err) is cls
    assert err.reason == reason
    assert err.status == status
    assert err.endpoint == "/v1/x"
    assert isinstance(err, AlbertError)


@pytest.mark.parametrize("status", [408, 500, 501, 502, 504, 599])
def test_classification_transient_statuses(status):
    err = classify_http_error(status, "upstream error", {}, endpoint="/v1/chat/completions")
    assert type(err) is AlbertTransientError
    assert err.retry_after is None
    assert err.status == status


def test_classification_503_is_model_busy():
    body = json.dumps({"detail": "Model is too busy, please try again later"})
    err = classify_http_error(503, body, {})
    assert type(err) is AlbertModelBusy
    assert isinstance(err, AlbertTransientError)
    err_plain = classify_http_error(503, "Service Unavailable", {})
    assert type(err_plain) is AlbertModelBusy


@pytest.mark.parametrize(
    "headers, kwargs, cls, expected_retry",
    [
        ({}, {}, AlbertTransientError, None),
        ({"retry-after": "30"}, {}, AlbertTransientError, 30.0),
        ({"Retry-After": "120"}, {}, AlbertTransientError, 120.0),
        ({"retry-after": "121"}, {}, AlbertQuotaExhausted, 121.0),
        ({"retry-after": "not-a-date"}, {}, AlbertTransientError, None),
        ({"retry-after": "-3"}, {}, AlbertTransientError, None),
        ({"retry-after": _PAST}, {}, AlbertTransientError, 0.0),
        ({}, {"retry_after": 12.5}, AlbertTransientError, 12.5),
        ({}, {"retry_after": 121.0}, AlbertQuotaExhausted, 121.0),
        ({"retry-after": "5"}, {"retry_after": 600.0}, AlbertQuotaExhausted, 600.0),
        ({}, {"retry_after": 50.0, "retry_after_max": 30.0}, AlbertQuotaExhausted, 50.0),
        ({}, {"retry_after": -1.0}, AlbertTransientError, 0.0),
    ],
)
def test_classification_429_retry_after(headers, kwargs, cls, expected_retry):
    err = classify_http_error(429, json.dumps({"detail": "Rate limit exceeded"}), headers, **kwargs)
    assert type(err) is cls
    assert err.retry_after == expected_retry
    assert err.status == 429


def test_classification_429_http_date_beyond_cap_is_quota():
    err = classify_http_error(429, "", {"retry-after": _FUTURE})
    assert type(err) is AlbertQuotaExhausted
    assert err.reason == QUOTA_REASON
    assert isinstance(err, AlbertAuthError)
    assert err.retry_after > 120.0
    assert err.credential_required == "albert_api_key"


def test_classification_accepts_httpx_like_headers():
    class CaseInsensitiveHeaders(Mapping):
        """En-têtes insensibles à la casse, comme ``httpx.Headers``."""

        def __init__(self, data):
            """Stocke les en-têtes en minuscules."""
            self._data = {k.lower(): v for k, v in data.items()}

        def __getitem__(self, key):
            """Lecture insensible à la casse."""
            return self._data[key.lower()]

        def __iter__(self):
            """Itère sur les noms en minuscules."""
            return iter(self._data)

        def __len__(self):
            """Nombre d'en-têtes."""
            return len(self._data)

    err = classify_http_error(429, "", CaseInsensitiveHeaders({"RETRY-AFTER": "7"}))
    assert err.retry_after == 7.0
    assert classify_http_error(429, "", None).retry_after is None


def test_error_hierarchy():
    assert issubclass(AlbertAuthError, AlbertError)
    assert issubclass(AlbertQuotaExhausted, AlbertAuthError)
    assert issubclass(AlbertPermanentError, AlbertError)
    assert issubclass(AlbertTransientError, AlbertError)
    assert issubclass(AlbertModelBusy, AlbertTransientError)
    assert issubclass(AlbertTruncatedError, AlbertError)
    assert issubclass(AlbertDisabledError, ValueError)
    assert not issubclass(AlbertDisabledError, AlbertError)
    assert not issubclass(AlbertPermanentError, AlbertTransientError)
    assert not issubclass(AlbertAuthError, AlbertTransientError)


def test_every_error_has_status_endpoint_detail():
    samples = [
        AlbertError(),
        AlbertAuthError(),
        AlbertQuotaExhausted(),
        AlbertPermanentError(),
        AlbertTransientError(),
        AlbertModelBusy(),
        AlbertTruncatedError(),
    ]
    for err in samples:
        assert err.status is None and err.endpoint is None and err.detail is None
        assert str(err)
    assert AlbertTransientError(retry_after=3).retry_after == 3.0
    assert AlbertTransientError().retry_after is None
    trunc = AlbertTruncatedError(status=200, endpoint="/v1/chat/completions", detail="finish_reason=length")
    assert (trunc.status, trunc.endpoint, trunc.detail) == (200, "/v1/chat/completions", "finish_reason=length")


def test_reasons_are_validated():
    assert AlbertAuthError().reason == "invalid_key"
    for reason in AUTH_REASONS:
        assert AlbertAuthError(reason=reason).reason == reason
    for reason in PERMANENT_REASONS:
        assert AlbertPermanentError(reason=reason).reason == reason
    with pytest.raises(ValueError):
        AlbertAuthError(reason="not_a_reason")
    with pytest.raises(ValueError):
        AlbertAuthError(reason=QUOTA_REASON)
    with pytest.raises(ValueError):
        AlbertPermanentError(reason="invalid_key")
    with pytest.raises(ValueError):
        AlbertQuotaExhausted(reason="invalid_key")
    assert AlbertQuotaExhausted().reason == QUOTA_REASON


def test_messages_fr_cover_all_reasons():
    for key in AUTH_REASONS + PERMANENT_REASONS + (QUOTA_REASON,):
        assert MESSAGES_FR[key].strip()
    for key in ("error", "transient", "model_busy", "truncated", "disabled"):
        assert MESSAGES_FR[key].strip()
    assert "albert.api@numerique.gouv.fr" in MESSAGES_FR["account_expired"]
    assert "albert.api@numerique.gouv.fr" in str(AlbertAuthError(reason="account_expired"))
    assert "/v1/models" in MESSAGES_FR["not_found"]
    assert str(AlbertAuthError(reason="budget_exhausted")).startswith(MESSAGES_FR["budget_exhausted"])


def test_detail_redacts_secrets_and_is_capped():
    body = json.dumps(
        {
            "detail": (
                f"Bad header Authorization: Bearer {FAKE_ALBERT_KEY} for user someone@example.org "
                f"api_key={FAKE_ALBERT_KEY} " + "x" * 500
            )
        }
    )
    err = classify_http_error(400, body, {}, endpoint="/v1/models")
    assert len(err.detail) <= albert_errors.DETAIL_MAX_CHARS
    assert (FAKE_ALBERT_KEY in err.detail) is False
    assert (FAKE_ALBERT_KEY in str(err)) is False
    assert ("someone@example.org" in err.detail) is False
    assert ("someone@example.org" in str(err)) is False
    assert "Bearer ***" in err.detail


def test_redact_with_explicit_secret_and_limit():
    text = f"prefix {FAKE_ALBERT_KEY} suffix"
    out = redact(text, secrets=[FAKE_ALBERT_KEY])
    assert (FAKE_ALBERT_KEY in out) is False
    assert out.startswith("prefix") and out.endswith("suffix")
    assert len(redact("a" * 1000)) == albert_errors.DETAIL_MAX_CHARS
    assert redact(None) == ""
    assert redact("a\n\t b") == "a b"
    long_message = AlbertPermanentError("m" * 1000)
    assert len(str(long_message)) == 1000


def test_errors_pickle_roundtrip():
    samples = [
        classify_http_error(401, json.dumps({"detail": "Invalid API key."}), {}, endpoint="/v1/models"),
        classify_http_error(429, "", {}, retry_after=500.0),
        classify_http_error(429, "", {"retry-after": "3"}),
        classify_http_error(503, "busy", {}),
        classify_http_error(422, json.dumps({"detail": "Wrong model type"}), {}),
        AlbertTruncatedError(status=200),
        AlbertDisabledError(model="albert/gpt-oss-120b"),
    ]
    for err in samples:
        clone = pickle.loads(pickle.dumps(err))
        assert type(clone) is type(err)
        assert str(clone) == str(err)
        for attr in ("status", "endpoint", "detail", "reason", "retry_after", "model"):
            assert getattr(clone, attr, None) == getattr(err, attr, None)


def test_disabled_error_is_value_error_with_french_message():
    err = AlbertDisabledError(model="albert/ministral-3-8b-instruct-2512")
    assert isinstance(err, ValueError)
    assert "ALBERT_ENABLED" in str(err)
    assert "albert/ministral-3-8b-instruct-2512" in str(err)
    assert err.model == "albert/ministral-3-8b-instruct-2512"
    assert str(AlbertDisabledError("personnalisé")) == "personnalisé"


# ---------------------------------------------------------------------------
# Paquet : import léger et réexports
# ---------------------------------------------------------------------------


def _run_python(code):
    """Exécute ``code`` dans un interpréteur neuf, à la racine du dépôt."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def test_package_import_is_light():
    code = (
        "import sys\n"
        "import scripts.rad_albert\n"
        "import scripts.rad_albert.config, scripts.rad_albert.errors\n"
        "print('httpx' in sys.modules, 'openai' in sys.modules, 'requests' in sys.modules)\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "False False False"


def test_cli_import_path_is_light():
    code = (
        "import sys\n"
        "sys.path.insert(0, 'scripts')\n"
        "import rad_albert\n"
        "from rad_albert import AlbertConfig, classify_http_error\n"
        "print('httpx' in sys.modules, 'openai' in sys.modules, AlbertConfig.from_env({}).enabled)\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "False False False"


def test_package_import_prints_nothing():
    proc = _run_python("import scripts.rad_albert")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_package_reexports_only_config_and_errors():
    init_path = PROJECT_ROOT / "scripts" / "rad_albert" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_modules.add((node.level, node.module))
        elif isinstance(node, ast.Import):
            imported_modules.add((0, ",".join(alias.name for alias in node.names)))
    assert imported_modules == {(1, "config"), (1, "errors")}
    for name in rad_albert.__all__:
        obj = getattr(rad_albert, name)
        assert hasattr(albert_config, name) or hasattr(albert_errors, name)
        assert obj is getattr(albert_config, name, None) or obj is getattr(albert_errors, name, None)
