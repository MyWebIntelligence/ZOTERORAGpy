"""Catalogue Albert : replis datés, modèles exclus, résolution alias → id, embeddings sans repli."""

import dataclasses
import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest

from scripts.rad_albert import catalog
from scripts.rad_albert import errors as albert_errors
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert import retry as albert_retry
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertPermanentError, AlbertUncertainWriteError
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert, load_fixture

PROJECT_ROOT = Path(__file__).resolve().parent
FIXTURES = Path("tests/fixtures/albert")
FIXED_TODAY = date(2026, 9, 26)
MESSAGES = [{"role": "user", "content": "Réponds seulement : OK."}]
SMALL_CHAIN_ROLES = ("recode", "citation", "book_structure", "long_context")
MINISTRAL = "ministral-3-8b-instruct-2512"
MISTRAL_SMALL = "mistral-small-3-2-24b-instruct-2506"
GEMMA = "gemma-4-31b-it"



def _p2_listing():
    """Réponse ``/v1/models`` réelle (fixture P2)."""
    return load_fixture("P2_models.json")["body"]


def _decisions():
    """Décisions mesurées (``decisions.json``)."""
    return json.loads((FIXTURES / "decisions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("role", SMALL_CHAIN_ROLES)
def test_fallback_chain_drops_mistral_small_after_2026_12_01(role):
    assert catalog.fallback_chain(role, today=date(2026, 11, 30)) == [MINISTRAL, MISTRAL_SMALL]
    assert catalog.fallback_chain(role, today=date(2026, 12, 1)) == [MINISTRAL, GEMMA]
    assert catalog.fallback_chain(role, today=date(2027, 6, 1)) == [MINISTRAL, GEMMA]


def test_excluded_never_default():
    day = date(2026, 1, 1)
    while day <= date(2027, 12, 31):
        for role in catalog.ROLES:
            chain = catalog.fallback_chain(role, today=day)
            assert not [m for m in chain if catalog.is_excluded(m)], (role, day)
        day += timedelta(days=15)
    for spec in catalog.ROLES.values():
        assert catalog.is_excluded(spec.primary) is False
    defaults = AlbertConfig(enabled=True)
    for name in (defaults.embed_model, defaults.ocr_chat_model, defaults.ocr_doc_model):
        assert catalog.is_excluded(name) is False
    for name in ("qwen3-coder-30b-a3b-instruct", "openweight-code", "deepseek-v4-flash",
                 "deepseek-v4-flash-0731", "qwen3-vl-embedding-8b", "mistral-medium-2508"):
        assert catalog.is_excluded(name) is True


def test_resolve_alias_to_id():
    listing = _p2_listing()
    assert catalog.resolve_model("openweight-small", listing) == MINISTRAL
    assert catalog.resolve_model("OPENWEIGHT-SMALL", listing) == MINISTRAL
    assert catalog.resolve_model("BAAI/bge-m3", listing) == "bge-m3"
    assert catalog.resolve_model(MINISTRAL, listing) == MINISTRAL
    with pytest.raises(catalog.ModelNotFoundError) as excinfo:
        catalog.resolve_model("mistral-ocr-2512", listing)
    assert excinfo.value.reason == "not_found"
    names = catalog.listing_name_map(listing)
    assert names["openweight-small"] == MINISTRAL
    assert names["openweight-large"] == "gpt-oss-120b"
    assert names["baai/bge-m3"] == "bge-m3"


def test_embeddings_have_no_fallback():
    assert "embed" in catalog.NO_FALLBACK_ROLES
    day = date(2026, 1, 1)
    while day <= date(2027, 12, 31):
        assert catalog.fallback_chain("embed", today=day) == ["bge-m3"]
        day += timedelta(days=30)


def test_catalog_consistent_with_p2_and_d2():
    listing = {entry["id"]: entry for entry in _p2_listing()["data"]}
    d2 = _decisions()["D2"]["value"]["models"]
    known = [spec for spec in list(catalog.MODELS.values()) + list(catalog._EXCLUDED_SPECS) if spec.id in listing]
    assert {spec.id for spec in known} >= set(d2)
    for spec in known:
        entry = listing[spec.id]
        assert spec.type == entry["type"], spec.id
        assert spec.context == entry["max_context_length"], spec.id
        assert set(spec.aliases) == set(entry["aliases"]), spec.id
        if spec.id in d2:
            assert spec.type == d2[spec.id]["type"], spec.id
            assert spec.context == d2[spec.id]["max_context_length"], spec.id
            assert set(spec.aliases) == set(d2[spec.id]["aliases"]), spec.id
