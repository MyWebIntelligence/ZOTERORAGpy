"""Tests live de l'OCR souverain Albert (lot 4) : LightOnOCR, ``/v1/ocr`` et chaîne complète.

Marqueur ``albert_live`` : sautés sauf si ``ALBERT_LIVE=1`` est exporté dans le
shell (``ALBERT_LIVE`` ne doit jamais figurer dans un fichier ``.env``). Appels
d'inférence seulement (aucune collection créée). La clé vient de l'environnement
du shell, sinon de ``ALBERT_API_KEY`` dans le ``.env`` racine ; elle n'est jamais
affichée.

* LightOnOCR (``lightonocr-2-1b``) sur un PDF de 2 pages généré (``PAGE_ONE_7Q``,
  ``PAGE_TWO_7Q``) : marqueurs ``<!-- Page 1 -->`` et ``<!-- Page 2 -->``. Un 404
  sur ce modèle (fin de phase de test prévue le 2026-10-01) arrête le test avec
  un message d'escalade.
* ``/v1/ocr`` avec ``pages: [0]`` : sauté quand ``decisions.json`` (D3) indique
  l'absence d'accès ; dans ce cas, un test complémentaire vérifie que la réponse
  réelle est reconnue par ``errors.is_ocr_access_denied``.
* Chaîne complète ``extract_text_with_ocr`` avec le maillon Albert actif et
  Mistral, OpenAI et l'OCR local coupés : provider ``albert_*``, sans repli.

Commande : ``ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_ocr.py -m albert_live -v -p no:cacheprovider``
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import re
import threading
from pathlib import Path

import fitz  # type: ignore[import-not-found]
import pytest

from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertPermanentError, is_ocr_access_denied

pytestmark = pytest.mark.albert_live

RAGPY_ROOT = Path(__file__).resolve().parents[2]
DECISIONS_PATH = RAGPY_ROOT / "tests" / "fixtures" / "albert" / "decisions.json"
PAGE_ONE = "PAGE_ONE_7Q"
PAGE_TWO = "PAGE_TWO_7Q"
OCR_CHAT_MODEL = "lightonocr-2-1b"
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
ESCALATION = (
    "ESCALADE : lightonocr-2-1b répond 404 (modèle retiré ; fin de phase de test prévue le "
    "2026-10-01). Arrêt du lot 4 : choisir un autre modèle OCR avec l'utilisateur."
)


def _live_key():
    """Clé Albert du shell, sinon du ``.env`` racine (valeur jamais affichée)."""
    key = (os.environ.get("ALBERT_API_KEY") or "").strip()
    if not key:
        try:
            from dotenv import dotenv_values
        except ImportError:
            dotenv_values = None
        if dotenv_values is not None:
            key = (dotenv_values(RAGPY_ROOT / ".env").get("ALBERT_API_KEY") or "").strip()
    return key


def _d3():
    """Décision D3 mesurée au L0 (accès à ``/v1/ocr``)."""
    decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    return decisions["D3"]["value"]


def _v1_ocr_access_expected():
    """Vrai si D3 indique un accès à ``/v1/ocr`` sur le compte sondé."""
    d3 = _d3()
    return bool(d3.get("ocr_access")) and not d3.get("skip_live_ocr_doc_test")


def _two_page_pdf(directory):
    """PDF A4 de 2 pages : ``PAGE_ONE_7Q`` puis ``PAGE_TWO_7Q`` en grands caractères."""
    doc = fitz.open()
    for token in (PAGE_ONE, PAGE_TWO):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 160), token, fontsize=40)
        page.insert_text((60, 240), "Document de test RAGpy (sonde OCR).", fontsize=16)
    path = Path(directory) / "albert_live_ocr.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def _markers(text):
    """Numéros des marqueurs ``<!-- Page N -->`` d'un texte."""
    return [int(n) for n in PAGE_MARKER_RE.findall(text or "")]


@pytest.fixture(autouse=True)
def _fresh_albert_state():
    """Oublie les limiteurs partagés et le cache du preflight autour de chaque test."""
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture(scope="module")
def live_key():
    """Clé Albert réelle ; saute le module si elle est absente."""
    key = _live_key()
    if not key:
        pytest.skip("ALBERT_API_KEY absente du shell et du .env : tests live impossibles")
    return key


@pytest.fixture(scope="module")
def live_cfg():
    """Configuration Albert ON (OCR compris) lue depuis l'environnement."""
    env = dict(os.environ)
    env["ALBERT_ENABLED"] = "1"
    env["OCR_ENABLE_ALBERT"] = "1"
    return AlbertConfig.from_env(env)


@pytest.fixture
def live_client(live_cfg, live_key):
    """Client Albert réel (fermé en fin de test)."""
    client = AlbertClient(live_cfg, live_key)
    yield client
    client.close()


def test_live_lightonocr_two_pages(live_client, live_cfg, tmp_path):
    ocr = importlib.import_module("scripts.rad_albert.ocr")
    pdf = _two_page_pdf(tmp_path)
    try:
        outcome = ocr.ocr_pdf_lightonocr(
            pdf, live_client, cfg=live_cfg, semaphore=threading.BoundedSemaphore(1)
        )
    except AlbertPermanentError as exc:
        if exc.status == 404:
            pytest.fail(ESCALATION, pytrace=False)
        raise
    assert outcome.provider == "albert_lightonocr"
    assert _markers(outcome.text) == [1, 2]
    assert PAGE_ONE in outcome.text.split("<!-- Page 2 -->", 1)[0]
    assert PAGE_TWO in outcome.text.split("<!-- Page 2 -->", 1)[1]
    assert outcome.partial is False
    assert (outcome.pages_done, outcome.pages_total) == (2, 2)
    served = [record for record in live_client.ledger if record.endpoint == "/v1/chat/completions"]
    assert len(served) == 2
    assert all(record.model == OCR_CHAT_MODEL for record in served)


def test_live_v1_ocr_page_zero(live_client, tmp_path):
    if not _v1_ocr_access_expected():
        pytest.skip("D3 : pas d'accès à /v1/ocr sur ce compte (404 « Model mistral-ocr-2512 not found. »)")
    pdf_bytes = Path(_two_page_pdf(tmp_path)).read_bytes()
    data = live_client.ocr_document(pdf_bytes, pages=[0])
    pages = data["pages"]
    assert [page["index"] for page in pages] == [0]
    assert PAGE_ONE in pages[0]["markdown"]
    assert PAGE_TWO not in pages[0]["markdown"]


def test_live_v1_ocr_access_denied_recognised(live_client, tmp_path):
    if _v1_ocr_access_expected():
        pytest.skip("D3 : accès à /v1/ocr mesuré ; le refus d'accès ne peut pas être observé")
    pdf_bytes = Path(_two_page_pdf(tmp_path)).read_bytes()
    with pytest.raises(AlbertPermanentError) as excinfo:
        live_client.ocr_document(pdf_bytes, pages=[0])
    assert excinfo.value.status in (403, 404)
    assert is_ocr_access_denied(excinfo.value)


def test_live_full_chain_provider_albert(live_key, monkeypatch, tmp_path):
    rad = importlib.import_module("scripts.rad_dataframe")
    settings = dict(os.environ)
    settings.update({
        "ALBERT_ENABLED": "1",
        "OCR_ENABLE_ALBERT": "1",
        "ALBERT_OCR_MODE": "auto",
        "ALBERT_USAGE_LOG": "0",
    })
    for name in ("ALBERT_ENABLED", "OCR_ENABLE_ALBERT", "ALBERT_OCR_MODE", "ALBERT_USAGE_LOG"):
        monkeypatch.setenv(name, settings[name])
    # Constantes de script lues à l'import (décision 11), posées comme avec cet env.
    monkeypatch.setattr(rad, "ALBERT_CONFIG", AlbertConfig.from_env(settings))
    monkeypatch.setattr(rad, "_ALBERT_CONFIG_ERROR", None)
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", True)
    monkeypatch.setattr(rad, "ALBERT_API_KEY", live_key)
    monkeypatch.setattr(rad, "_ALBERT_ACCOUNT_DISABLED", None)
    monkeypatch.setattr(rad, "_ALBERT_V1_OCR_AVAILABLE", None)
    monkeypatch.setattr(rad, "_ALBERT_WARNED", set())
    monkeypatch.setattr(rad, "_ALBERT_OCR_LEDGER", None)
    monkeypatch.setattr(rad, "_ALBERT_OCR_TRANSPORT", None)
    monkeypatch.setattr(rad, "_ALBERT_OCR_MODEL_IDS", {})
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
    monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", False)
    monkeypatch.setattr(rad, "_local_ocr_available", lambda *a, **k: False)
    monkeypatch.setattr(rad, "OCR_MIN_CHARS_PER_PAGE", 1)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def no_mistral(*args, **kwargs):
        """Le maillon Mistral SaaS ne doit jamais être appelé ici."""
        raise AssertionError("Mistral appelé alors que le maillon Albert est actif")

    monkeypatch.setattr(rad, "_extract_text_with_mistral", no_mistral)

    result = rad.extract_text_with_ocr(_two_page_pdf(tmp_path), return_details=True)

    assert result.provider in ("albert_lightonocr", "albert_mistral_ocr")
    assert result.fallback_from is None
    assert _markers(result.text) == [1, 2]
    assert PAGE_ONE in result.text
    assert result.pages_total == 2
    assert rad._ALBERT_ACCOUNT_DISABLED is None
    if not _v1_ocr_access_expected():
        assert result.provider == "albert_lightonocr"
        assert rad._ALBERT_V1_OCR_AVAILABLE is not True
    key_leaked = live_key in result.text or live_key in (result.error or "")
    assert key_leaked is False


def test_live_decisions_d3_default_mode_auto():
    d3 = _d3()
    assert d3["ALBERT_OCR_MODE"] == "auto"
    assert dataclasses.replace(AlbertConfig(), enabled=True).ocr_mode == "auto"
