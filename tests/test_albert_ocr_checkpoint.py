"""Tests de la reprise page par page de l'OCR LightOnOCR (``scripts/rad_albert/ocr_checkpoint.py``).

Harnais de ``tests/test_albert_ocr.py`` (``AlbertChain``, ``PageRouter``, PDF de test aux
largeurs de pages distinctes) : aucun réseau, faux serveur Albert seulement.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os

import pytest

import scripts.rad_dataframe as rad
from scripts.rad_albert import ocr_checkpoint
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert
from tests.test_albert_ocr import (  # noqa: F401
    PageRouter,
    _build_pdf,
    _fresh_albert_state,
    _page_text,
    _sizes,
    chain,
)

CFG = AlbertConfig(enabled=True, ocr_enabled=True)


def _ocr():
    """Module ``scripts.rad_albert.ocr``."""
    return importlib.import_module("scripts.rad_albert.ocr")


def _client(fake):
    """Client sans limiteur ni attente."""
    return AlbertClient(CFG, FAKE_ALBERT_KEY, transport=fake.transport, sleep=lambda _s: None, use_limiter=False)


def _checkpoint(root, pdf, cfg=CFG):
    """Point de reprise du document pour LightOnOCR."""
    return ocr_checkpoint.PageCheckpoint(str(root), pdf, ocr_checkpoint.lightonocr_params(cfg, "lightonocr-2-1b"))


def _run(fake, pdf, checkpoint):
    """OCR LightOnOCR du PDF avec le point de reprise."""
    return _ocr().ocr_pdf_lightonocr(pdf, _client(fake), cfg=CFG, checkpoint=checkpoint)


def _router(fake, pages):
    """Routeur page par page (textes de plus de 500 caractères)."""
    router = PageRouter(fake, _sizes(pages))
    for number in range(1, pages + 1):
        router.texts[number] = _page_text(number)
    return router


def test_second_run_reuses_every_validated_page(tmp_path):
    pdf = _build_pdf(tmp_path, "livre", pages=3)
    fake = FakeAlbert()
    _router(fake, 3)
    first = _run(fake, pdf, _checkpoint(tmp_path / "out", pdf))
    assert len(fake.calls_to("POST", "/v1/chat/completions")) == 3
    fake.reset_calls()
    checkpoint = _checkpoint(tmp_path / "out", pdf)
    second = _run(fake, pdf, checkpoint)
    assert fake.calls_to("POST", "/v1/chat/completions") == []
    assert second.text == first.text and checkpoint.reused == 3
    assert second.pages_done == 3 and not second.partial


def test_truncated_and_failed_pages_are_never_checkpointed(tmp_path):
    pdf = _build_pdf(tmp_path, "livre", pages=3)
    fake = FakeAlbert()
    router = _router(fake, 3)
    router.plan(2, "length")
    router.plan(3, *([500] * 10))
    outcome = _run(fake, pdf, _checkpoint(tmp_path / "out", pdf))
    assert outcome.partial and outcome.pages_failed == (3,)
    before = len(router.requests)
    _run(fake, pdf, _checkpoint(tmp_path / "out", pdf))
    assert sorted(set(router.requests[before:])) == [2, 3]


def test_changed_parameters_or_document_do_not_reuse(tmp_path):
    pdf = _build_pdf(tmp_path, "livre", pages=2)
    fake = FakeAlbert()
    _router(fake, 2)
    _run(fake, pdf, _checkpoint(tmp_path / "out", pdf))
    other_cfg = dataclasses.replace(CFG, ocr_dpi=150)
    assert _checkpoint(tmp_path / "out", pdf, other_cfg).load(range(2)) == {}
    other_pdf = _build_pdf(tmp_path / "autre", "livre", pages=2, sizes=[(220, 300), (260, 300)])
    assert _checkpoint(tmp_path / "out", other_pdf).load(range(2)) == {}


def test_tampered_or_foreign_checkpoint_files_are_ignored(tmp_path):
    pdf = _build_pdf(tmp_path, "livre", pages=2)
    fake = FakeAlbert()
    _router(fake, 2)
    checkpoint = _checkpoint(tmp_path / "out", pdf)
    _run(fake, pdf, checkpoint)
    page_file = os.path.join(checkpoint.directory, "page_00000.json")
    data = json.loads(open(page_file, encoding="utf-8").read())
    data["doc"] = "0" * 16
    open(page_file, "w", encoding="utf-8").write(json.dumps(data))
    open(os.path.join(checkpoint.directory, "page_00001.json"), "w", encoding="utf-8").write("{illisible")
    assert _checkpoint(tmp_path / "out", pdf).load(range(2)) == {}


def test_checkpoint_directory_is_confined_and_disabled_by_config(tmp_path):
    pdf = _build_pdf(tmp_path, "livre", pages=1)
    checkpoint = _checkpoint(tmp_path / "out", pdf)
    root = os.path.realpath(tmp_path / "out")
    assert os.path.commonpath([checkpoint.directory, root]) == root
    assert ocr_checkpoint.DIRNAME in checkpoint.directory
    assert ocr_checkpoint.open_checkpoint(str(tmp_path), pdf, dataclasses.replace(CFG, ocr_checkpoint=False),
                                          "lightonocr-2-1b") is None
    assert ocr_checkpoint.open_checkpoint(None, pdf, CFG, "lightonocr-2-1b") is None


def test_rad_dataframe_chain_resumes_from_checkpoints(chain, tmp_path, monkeypatch):
    pdf = _build_pdf(tmp_path, "livre", pages=3)
    run = chain(mode="chat")
    _router(run.fake, 3)
    monkeypatch.setattr(rad, "_ALBERT_OCR_CHECKPOINT_ROOT", str(tmp_path / "session"))
    first = run.run(pdf)
    assert first.provider == "albert_lightonocr" and len(run.chat_calls()) == 3
    run.fake.reset_calls()
    second = run.run(pdf)
    assert run.chat_calls() == [] and second.text == first.text
    assert os.path.isdir(tmp_path / "session" / ocr_checkpoint.DIRNAME)


def test_rad_dataframe_without_root_writes_nothing(chain, tmp_path, monkeypatch):
    pdf = _build_pdf(tmp_path, "livre", pages=2)
    run = chain(mode="chat")
    _router(run.fake, 2)
    monkeypatch.setattr(rad, "_ALBERT_OCR_CHECKPOINT_ROOT", None)
    run.run(pdf)
    run.fake.reset_calls()
    run.run(pdf)
    assert len(run.chat_calls()) == 2
    assert not any(ocr_checkpoint.DIRNAME in str(p) for p in tmp_path.rglob("*"))


@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_image_attachment_is_ocrd_by_albert_chain(chain, tmp_path, extension):
    import fitz

    run = chain(mode="chat")
    run.fake.ocr_chat_reply = "Texte lu dans l'image. " * 40
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), False)
    pix.clear_with(255)
    image = tmp_path / f"scan{extension}"
    pix.save(str(image))
    result = rad._extract_text_from_image(str(image), retry=False)
    assert result.provider == "albert_lightonocr"
    assert "Texte lu dans l'image" in result.text
    assert not list(tmp_path.glob("ragpy_img_*.pdf"))
