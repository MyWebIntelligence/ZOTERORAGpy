"""OCR explicite, sans repli implicite (sprint « configuration unifiée », lot L5).

``OCR_SERVER`` + ``OCR_MODEL`` déclarés : un seul moteur, contrôlé avant le
premier document ; un échec du moteur fait échouer le document (jamais un autre
fournisseur), un document en échec n'est ni écrit ni marqué traité (la reprise
le refait), une erreur de compte arrête le lot. Sans ``OCR_MODEL`` : chaîne
historique (couverte par les goldens G1/G13).

Repli déclaré (``OCR_SERVER_FALLBACK`` + ``OCR_MODEL_FALLBACK``, demande du
2026-10-03) : autre serveur, contrôlé avant le premier document ; un document
en échec est repris par lui, une erreur de compte du moteur principal fait
passer la suite du lot par lui ; repli tracé, colonnes du moteur qui a servi.
"""

from __future__ import annotations

import dataclasses
import json
import os

import fitz
import pytest

import scripts.rad_dataframe as rad
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_settings.models import ServiceConfigError
from tests.test_albert_ocr import _zotero_item

MISTRAL = "https://api.mistral.ai/v1"
ALBERT = "https://albert.api.etalab.gouv.fr/v1"
LOCAL = "http://localhost:11434/v1"
LONG_TEXT = ("La méthode distributionnelle regroupe les entités qui se combinent de façon "
             "complémentaire ou similaire. ") * 12

GLOBALS = ("_OCR_EXPLICIT", "_OCR_CHOICE", "_OCR_FALLBACK", "_OCR_PRIMARY_DOWN", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL", "MISTRAL_API_KEY",
           "ALBERT_CONFIG", "OCR_ENABLE_ALBERT", "ALBERT_API_KEY", "_ALBERT_ACCOUNT_DISABLED", "PDF_EXTRACTION_WORKERS")


@pytest.fixture
def explicit(monkeypatch):
    """Configure l'OCR explicite pour un test ; l'état du module est restauré ensuite."""
    for name in GLOBALS:
        monkeypatch.setattr(rad, name, getattr(rad, name))

    def configure(server, model, **extra):
        env = {"OCR_SERVER": server, "OCR_MODEL": model, "MISTRAL_API_BASE_URL": MISTRAL,
               "LOCAL_API_BASE_URL": LOCAL, "OCR_SERVER_FALLBACK": "", "OCR_MODEL_FALLBACK": "", **extra}
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return rad.configure_explicit_ocr(env)

    return configure


def _no_other_provider(monkeypatch):
    """Tout autre moteur appelé = repli interdit : le test échoue."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("repli vers un autre moteur")

    for name in ("_extract_text_with_mistral", "_try_albert_ocr", "_extract_text_with_local",
                 "_extract_text_with_openai"):
        monkeypatch.setattr(rad, name, forbidden)


def _text_pdf(path, text=LONG_TEXT, pages=1):
    """PDF né numérique (couche texte)."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return str(path)


def _scan_pdf(path):
    """PDF « scanné » : une image, aucune couche texte."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_no_ocr_model_keeps_the_historical_chain(explicit, monkeypatch):
    """``OCR_MODEL`` absent : aucun mode explicite."""
    monkeypatch.delenv("OCR_MODEL", raising=False)
    assert rad.configure_explicit_ocr({}) is None and rad._OCR_EXPLICIT is False


def test_mistral_choice_sets_model_and_address(explicit):
    """Mistral : modèle et adresse (sans /v1 pour l'OCR historique) du couple déclaré."""
    choice = explicit(MISTRAL, "mistral-ocr-2512")
    assert choice.server.key == "mistral" and rad._OCR_EXPLICIT is True
    assert rad.MISTRAL_OCR_MODEL == "mistral-ocr-2512" and rad.MISTRAL_API_BASE_URL == "https://api.mistral.ai"
    assert rad._explicit_columns() == {"texteocr_server": MISTRAL, "texteocr_model": "mistral-ocr-2512"}


@pytest.mark.parametrize("model, mode, field", [("lightonocr-2-1b", "chat", "ocr_chat_model"),
                                                ("mistral-ocr-2512", "ocr", "ocr_doc_model")])
def test_albert_choice_selects_the_route_by_model_type(explicit, monkeypatch, model, mode, field):
    """Albert : /v1/ocr pour un modèle ``image-to-text``, page par page sinon ; jamais ``auto``."""
    monkeypatch.setattr(rad, "ALBERT_CONFIG", AlbertConfig(enabled=True))
    explicit(ALBERT, model, ALBERT_BASE_URL=ALBERT, ALBERT_ENABLED="1")
    assert rad.ALBERT_CONFIG.ocr_mode == mode and getattr(rad.ALBERT_CONFIG, field) == model
    assert rad.ALBERT_CONFIG.ocr_enabled is True


def test_albert_choice_requires_albert_enabled(explicit, monkeypatch):
    """Albert désactivé : refus explicite."""
    monkeypatch.setattr(rad, "ALBERT_CONFIG", AlbertConfig(enabled=False))
    with pytest.raises(rad.OCRPreflightError, match="ALBERT_ENABLED"):
        explicit(ALBERT, "lightonocr-2-1b", ALBERT_BASE_URL=ALBERT)


def test_server_without_ocr_is_refused(explicit):
    """OpenRouter n'est pas un serveur d'OCR : erreur de configuration."""
    with pytest.raises(ServiceConfigError):
        explicit("https://openrouter.ai/api/v1", "x", OPENROUTER_API_BASE_URL="https://openrouter.ai/api/v1")


# ---------------------------------------------------------------------------
# Extraction : un seul moteur, aucun repli
# ---------------------------------------------------------------------------
def test_mistral_failure_fails_the_document_without_fallback(explicit, monkeypatch, tmp_path):
    """Échec Mistral : ``OCRExtractionError``, aucun autre moteur appelé."""
    explicit(MISTRAL, "mistral-ocr-latest")
    _no_other_provider(monkeypatch)

    def failing(*_args, **_kwargs):
        raise rad.OCRExtractionError("Mistral en panne")

    monkeypatch.setattr(rad, "_extract_text_with_mistral", failing)
    monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("repli PyMuPDF")))
    with pytest.raises(rad.OCRExtractionError, match="Mistral en panne") as info:
        rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf"), return_details=True)
    assert not isinstance(info.value, rad.OCRAccountError)


def test_mistral_401_is_an_account_error(explicit, monkeypatch, tmp_path):
    """401 Mistral (clé ou plafond de dépense) : erreur de compte, le lot s'arrêtera."""
    explicit(MISTRAL, "mistral-ocr-latest")

    def refused(*_args, **_kwargs):
        raise rad._MistralAuthError("OCR Mistral refusé (401)")

    monkeypatch.setattr(rad, "_extract_text_with_mistral", refused)
    with pytest.raises(rad.OCRAccountError):
        rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf"))


def test_mistral_success_keeps_partial_flags(explicit, monkeypatch, tmp_path):
    """Succès Mistral : texte, fournisseur ``mistral`` et drapeaux partiels du moteur."""
    explicit(MISTRAL, "mistral-ocr-latest")
    monkeypatch.setattr(rad, "_extract_text_with_mistral",
                        lambda *a, **k: rad._MistralOcr("texte", True, 2, 3, "1/3 (pages 3-3)"))
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf"), return_details=True)
    assert (result.provider, result.partial, result.pages_done, result.pages_total) == ("mistral", True, 2, 3)


def test_albert_failure_and_account_error(explicit, monkeypatch, tmp_path):
    """Albert : échec du maillon = échec du document ; compte mémorisé = erreur de compte."""
    monkeypatch.setattr(rad, "ALBERT_CONFIG", AlbertConfig(enabled=True))
    explicit(ALBERT, "lightonocr-2-1b", ALBERT_BASE_URL=ALBERT, ALBERT_ENABLED="1")
    _no_other_provider(monkeypatch)

    def failing(*_args, **_kwargs):
        raise rad._AlbertOcrLinkError("LightOnOCR indisponible", "albert_lightonocr")

    monkeypatch.setattr(rad, "_try_albert_ocr", failing)
    pdf = _text_pdf(tmp_path / "a.pdf")
    with pytest.raises(rad.OCRExtractionError) as info:
        rad.extract_text_with_ocr(pdf)
    assert not isinstance(info.value, rad.OCRAccountError)
    monkeypatch.setattr(rad, "_ALBERT_ACCOUNT_DISABLED", "quota journalier épuisé")
    with pytest.raises(rad.OCRAccountError, match="quota"):
        rad.extract_text_with_ocr(pdf)


def test_pymupdf_reads_the_native_text_layer_without_tesseract(explicit, monkeypatch, tmp_path):
    """``local`` + ``pymupdf`` : couche texte native ; jamais d'OCR Tesseract caché."""
    explicit("local", "pymupdf")
    calls = []
    real = fitz.Page.get_text

    def spy(self, option="text", *args, **kwargs):
        calls.append(option)
        return real(self, option, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_text", spy)
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf", text="Peu de mots."), return_details=True)
    assert result.provider == "pymupdf" and "Peu de mots" in result.text
    assert "ocr" not in calls
    assert result.partial is True  # densité faible : signalé, jamais un autre moteur


def test_pymupdf_on_a_scan_fails_explicitly(explicit, tmp_path):
    """Scan sans couche texte : échec explicite (choisir un moteur d'OCR)."""
    explicit("local", "pymupdf")
    with pytest.raises(rad.OCRExtractionError, match="aucune couche texte"):
        rad.extract_text_with_ocr(_scan_pdf(tmp_path / "scan.pdf"))


def test_docling_failure_fails_the_document(explicit, monkeypatch, tmp_path):
    """``local`` + ``docling`` : échec du moteur = échec du document."""
    explicit("local", "docling")
    _no_other_provider(monkeypatch)

    def failing(*_args, **_kwargs):
        raise RuntimeError("docling a planté")

    monkeypatch.setattr(rad, "_extract_text_with_local", failing)
    with pytest.raises(rad.OCRExtractionError, match="docling a planté"):
        rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf"))


class _FakeOpenAI:
    """Client compatible OpenAI factice du serveur local (OCR page par page)."""

    calls = []
    fail_pages = set()

    def __init__(self, api_key=None, base_url=None):
        _FakeOpenAI.base_url = base_url
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        _FakeOpenAI.calls.append(kwargs)
        number = len(_FakeOpenAI.calls)
        if number in _FakeOpenAI.fail_pages:
            raise RuntimeError("page illisible")

        class _Choice:
            finish_reason = "stop"
            message = type("M", (), {"content": f"Texte de la page {number}. " + LONG_TEXT})()

        return type("R", (), {"choices": [_Choice()]})()


def test_local_server_ocr_page_by_page(explicit, monkeypatch, tmp_path):
    """Serveur local : une requête par page, marqueurs de page, adresse déclarée."""
    import openai

    explicit(LOCAL, "lightonocr-2-1b")
    _FakeOpenAI.calls, _FakeOpenAI.fail_pages = [], set()
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf", pages=3), return_details=True)
    assert result.provider == "local_chat" and result.partial is False
    assert result.text.count("<!-- Page ") == 3 and len(_FakeOpenAI.calls) == 3
    assert _FakeOpenAI.base_url == LOCAL and _FakeOpenAI.calls[0]["model"] == "lightonocr-2-1b"
    _FakeOpenAI.calls, _FakeOpenAI.fail_pages = [], {2}
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "b.pdf", pages=3), return_details=True)
    assert result.partial is True and "OCR ÉCHOUÉ (local_chat)" in result.text and result.pages_done == 2
    _FakeOpenAI.calls, _FakeOpenAI.fail_pages = [], {1, 2}
    with pytest.raises(rad.OCRExtractionError, match="2/3 pages en échec"):
        rad.extract_text_with_ocr(_text_pdf(tmp_path / "c.pdf", pages=3))


# ---------------------------------------------------------------------------
# Contrôle préalable
# ---------------------------------------------------------------------------
class _Resp:
    """Réponse ``requests`` factice."""

    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_preflight_mistral(explicit, monkeypatch):
    """Mistral : clé requise, 401 et modèle absent refusés, modèle listé accepté."""
    explicit(MISTRAL, "mistral-ocr-latest")
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
    with pytest.raises(rad.OCRPreflightError, match="clé Mistral absente"):
        rad.preflight_explicit_ocr()
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", "fake-mistral-key")
    seen = []
    monkeypatch.setattr(rad.requests, "get", lambda url, **kw: seen.append(url) or _Resp(401, {}))
    with pytest.raises(rad.OCRPreflightError, match="401"):
        rad.preflight_explicit_ocr()
    assert seen == ["https://api.mistral.ai/v1/models"]
    monkeypatch.setattr(rad.requests, "get", lambda url, **kw: _Resp(200, {"data": [{"id": "mistral-small"}]}))
    with pytest.raises(rad.OCRPreflightError, match="absent du catalogue"):
        rad.preflight_explicit_ocr()
    listing = {"data": [{"id": "mistral-ocr-2512", "aliases": ["mistral-ocr-latest"]}]}
    monkeypatch.setattr(rad.requests, "get", lambda url, **kw: _Resp(200, listing))
    rad.preflight_explicit_ocr()


def test_preflight_docling_and_local_server(explicit, monkeypatch):
    """Docling non installé : refus ; serveur local : modèle listé exigé ; réseau coupé : refus."""
    explicit("local", "docling")
    monkeypatch.setattr(rad, "_local_ocr_available", lambda: False)
    with pytest.raises(rad.OCRPreflightError, match="Docling non installé"):
        rad.preflight_explicit_ocr()
    explicit(LOCAL, "lightonocr-2-1b")
    monkeypatch.setattr(rad.requests, "get", lambda url, **kw: _Resp(200, {"data": [{"id": "lightonocr-2-1b"}]}))
    rad.preflight_explicit_ocr()

    def unreachable(url, **kw):
        raise ConnectionError("connexion refusée")

    monkeypatch.setattr(rad.requests, "get", unreachable)
    with pytest.raises(rad.OCRPreflightError, match="inaccessible"):
        rad.preflight_explicit_ocr()


# ---------------------------------------------------------------------------
# Chargeur : tout ou rien par document, reprise, arrêt sur erreur de compte
# ---------------------------------------------------------------------------
def _export(tmp_path, keys):
    """Export Zotero minimal (un PDF par élément) ; renvoie le chemin du JSON."""
    items = [_zotero_item(key, f"files/{key.lower()}.pdf", title=f"Titre {key}") for key in keys]
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _progress(output_csv):
    with open(rad.get_progress_file_path(output_csv), encoding="utf-8") as fh:
        return sorted(json.load(fh).get("processed_keys", []))


def _errors(output_csv):
    with open(rad.get_errors_file_path(output_csv), encoding="utf-8") as fh:
        return json.load(fh)["errors"]


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_document_is_neither_written_nor_marked_done(explicit, monkeypatch, tmp_path, workers):
    """Échec d'un document : ni ligne ni marque « traité » ; la relance le refait."""
    explicit("local", "pymupdf")
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", workers)
    _text_pdf(tmp_path / "files" / "a1.pdf")
    _scan_pdf(tmp_path / "files" / "b2.pdf")
    json_path, output_csv = _export(tmp_path, ["A1", "B2"]), str(tmp_path / "output.csv")
    df = rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert list(df["itemKey"]) == ["A1"]
    assert set(df.columns) >= {"texteocr_server", "texteocr_model"} and df["texteocr_model"].iloc[0] == "pymupdf"
    assert _progress(output_csv) == ["A1"]
    assert [e["itemKey"] for e in _errors(output_csv) if e["error_type"] == "OCR_FAILED"] == ["B2"]
    # Le scan reçoit enfin une couche texte : la relance ne traite que B2.
    _text_pdf(tmp_path / "files" / "b2.pdf")
    df = rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert sorted(df["itemKey"]) == ["A1", "B2"] and _progress(output_csv) == ["A1", "B2"]


@pytest.mark.parametrize("workers", [1, 2])
def test_account_error_stops_the_run(explicit, monkeypatch, tmp_path, workers):
    """Erreur de compte : le lot s'arrête, l'erreur est consignée, rien n'est marqué traité."""
    explicit(MISTRAL, "mistral-ocr-latest")
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", workers)

    def refused(*_args, **_kwargs):
        raise rad._MistralAuthError("OCR Mistral refusé (401) : plafond de dépense")

    monkeypatch.setattr(rad, "_extract_text_with_mistral", refused)
    for key in ("a1", "b2", "c3"):
        _text_pdf(tmp_path / "files" / f"{key}.pdf")
    json_path, output_csv = _export(tmp_path, ["A1", "B2", "C3"]), str(tmp_path / "output.csv")
    with pytest.raises(rad.OCRAccountError, match="plafond"):
        rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert any(e["error_type"] == "OCR_ACCOUNT_ERROR" for e in _errors(output_csv))
    progress_file = rad.get_progress_file_path(output_csv)
    assert not os.path.exists(progress_file) or _progress(output_csv) == []


def test_images_follow_the_engine(explicit):
    """Images acceptées si le moteur sait les lire, jamais avec PyMuPDF."""
    explicit("local", "pymupdf")
    assert ".png" not in rad.supported_attachment_extensions()
    explicit(MISTRAL, "mistral-ocr-latest")
    assert ".png" in rad.supported_attachment_extensions()


def test_dataclass_replace_keeps_other_albert_settings(explicit, monkeypatch):
    """La reconfiguration Albert ne touche que les champs de l'OCR."""
    base = dataclasses.replace(AlbertConfig(enabled=True), ocr_dpi=150)
    monkeypatch.setattr(rad, "ALBERT_CONFIG", base)
    explicit(ALBERT, "lightonocr-2-1b", ALBERT_BASE_URL=ALBERT, ALBERT_ENABLED="1")
    assert rad.ALBERT_CONFIG.ocr_dpi == 150 and rad.ALBERT_CONFIG.enabled is True


# ---------------------------------------------------------------------------
# Repli déclaré (OCR_SERVER_FALLBACK + OCR_MODEL_FALLBACK)
# ---------------------------------------------------------------------------
def _fallback_pymupdf(explicit):
    """Moteur principal Mistral, repli ``local`` + ``pymupdf``."""
    return explicit(MISTRAL, "mistral-ocr-latest", OCR_SERVER_FALLBACK="local", OCR_MODEL_FALLBACK="pymupdf")


def test_fallback_configuration_rules(explicit):
    """Repli : autre serveur exigé ; avec local, un autre moteur interne ; vide = aucun repli."""
    explicit(MISTRAL, "mistral-ocr-latest")
    assert rad._OCR_FALLBACK is None
    _fallback_pymupdf(explicit)
    assert (rad._OCR_FALLBACK.server_url, rad._OCR_FALLBACK.model) == ("local", "pymupdf")
    with pytest.raises(ServiceConfigError, match="autre serveur"):
        explicit(MISTRAL, "mistral-ocr-latest", OCR_SERVER_FALLBACK=MISTRAL, OCR_MODEL_FALLBACK="mistral-ocr-2512")
    explicit("local", "docling", OCR_SERVER_FALLBACK="local", OCR_MODEL_FALLBACK="pymupdf")
    with pytest.raises(ServiceConfigError, match="autre serveur"):
        explicit("local", "pymupdf", OCR_SERVER_FALLBACK="local", OCR_MODEL_FALLBACK="pymupdf")
    with pytest.raises(ServiceConfigError):  # modèle seul : aucun serveur par défaut capable d'OCR
        explicit(MISTRAL, "mistral-ocr-latest", OCR_MODEL_FALLBACK="pymupdf")


def test_document_failure_is_served_by_the_fallback(explicit, monkeypatch, tmp_path):
    """Échec du moteur principal : le repli sert le document, repli tracé, colonnes du repli."""
    _fallback_pymupdf(explicit)
    monkeypatch.setattr(rad, "_extract_text_with_mistral",
                        lambda *a, **k: (_ for _ in ()).throw(rad.OCRExtractionError("Mistral en panne")))
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf"), return_details=True)
    assert result.provider == "pymupdf" and result.fallback_from == "Mistral — mistral-ocr-latest"
    assert rad._explicit_columns(result) == {"texteocr_server": "local", "texteocr_model": "pymupdf"}
    assert "repli" in rad._albert_fallback_message(result)
    assert rad._OCR_PRIMARY_DOWN is None  # simple échec : le moteur principal reste essayé


def test_both_engines_failing_fails_the_document(explicit, monkeypatch, tmp_path):
    """Échec du principal et du repli : le document échoue, avec les deux causes."""
    _fallback_pymupdf(explicit)
    monkeypatch.setattr(rad, "_extract_text_with_mistral",
                        lambda *a, **k: (_ for _ in ()).throw(rad.OCRExtractionError("Mistral en panne")))
    with pytest.raises(rad.OCRExtractionError, match="Mistral en panne.*repli") as info:
        rad.extract_text_with_ocr(_scan_pdf(tmp_path / "scan.pdf"))
    assert not isinstance(info.value, rad.OCRAccountError)


def test_primary_account_error_switches_the_rest_of_the_run(explicit, monkeypatch, tmp_path):
    """Erreur de compte du principal : repli pour ce document et les suivants, principal plus appelé."""
    _fallback_pymupdf(explicit)
    calls = []

    def refused(*_args, **_kwargs):
        calls.append(1)
        raise rad._MistralAuthError("OCR Mistral refusé (401) : plafond de dépense")

    monkeypatch.setattr(rad, "_extract_text_with_mistral", refused)
    for name in ("a.pdf", "b.pdf"):
        result = rad.extract_text_with_ocr(_text_pdf(tmp_path / name), return_details=True)
        assert result.provider == "pymupdf" and result.fallback_from
    assert len(calls) == 1 and "plafond" in rad._OCR_PRIMARY_DOWN


def test_fallback_account_error_stops_the_run(explicit, monkeypatch, tmp_path):
    """Erreur de compte du repli : le lot s'arrête."""
    explicit("local", "pymupdf", OCR_SERVER_FALLBACK=MISTRAL, OCR_MODEL_FALLBACK="mistral-ocr-latest")
    monkeypatch.setattr(rad, "_extract_text_with_mistral",
                        lambda *a, **k: (_ for _ in ()).throw(rad._MistralAuthError("OCR Mistral refusé (401)")))
    with pytest.raises(rad.OCRAccountError):
        rad.extract_text_with_ocr(_scan_pdf(tmp_path / "scan.pdf"))


def test_preflight_checks_the_fallback_too(explicit, monkeypatch):
    """Repli inaccessible : rien ne démarre, le message nomme le repli."""
    explicit("local", "pymupdf", OCR_SERVER_FALLBACK="local", OCR_MODEL_FALLBACK="docling")
    monkeypatch.setattr(rad, "_local_ocr_available", lambda: False)
    with pytest.raises(rad.OCRPreflightError, match="OCR de repli"):
        rad.preflight_explicit_ocr()
    monkeypatch.setattr(rad, "_local_ocr_available", lambda: True)
    rad.preflight_explicit_ocr()


def test_loader_records_the_fallback(explicit, monkeypatch, tmp_path):
    """Chargeur : ligne écrite avec les colonnes du repli, entrée OCR_PROVIDER_FALLBACK."""
    _fallback_pymupdf(explicit)
    monkeypatch.setattr(rad, "_extract_text_with_mistral",
                        lambda *a, **k: (_ for _ in ()).throw(rad.OCRExtractionError("Mistral en panne")))
    _text_pdf(tmp_path / "files" / "a1.pdf")
    json_path, output_csv = _export(tmp_path, ["A1"]), str(tmp_path / "output.csv")
    df = rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert list(df["itemKey"]) == ["A1"] and df["texteocr_model"].iloc[0] == "pymupdf"
    assert df["texteocr_server"].iloc[0] == "local"
    entries = [e for e in _errors(output_csv) if e["error_type"] == "OCR_PROVIDER_FALLBACK"]
    assert len(entries) == 1 and entries[0]["fallback_from"] == "Mistral — mistral-ocr-latest"


# ---------------------------------------------------------------------------
# Routes : seule la clé du serveur déclaré est exigée
# ---------------------------------------------------------------------------
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: E402,F401
from tests.test_albert_routes import _fresh_ocr_session, _post, albert_env  # noqa: E402,F401


def test_route_ocr_requires_only_the_declared_server_key(albert_env, monkeypatch):
    """OCR sur Mistral : clé Mistral exigée, sans cascade vers OpenAI ou Albert."""
    env = albert_env
    for name, value in {"OCR_SERVER": MISTRAL, "OCR_MODEL": "mistral-ocr-latest", "MISTRAL_API_BASE_URL": MISTRAL}.items():
        monkeypatch.setenv(name, value)
    path = _fresh_ocr_session(env, "gsess-ocr-explicit-mistral")
    resp, errors = _post(env, "member_mistral", "/process_dataframe", {"path": path})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    facts = env.capture.take()[0][0]["env"]
    assert facts["MISTRAL_API_KEY"] == "member_mistral.mistral_api_key"
    path = _fresh_ocr_session(env, "gsess-ocr-explicit-nokey")
    resp, _ = _post(env, "member_openrouter", "/process_dataframe", {"path": path})
    assert resp.status_code == 403 and resp.json()["credential_required"] == "mistral_api_key"
    assert env.capture.take() == ([], [])


def test_route_ocr_local_engine_needs_no_key(albert_env, monkeypatch):
    """Moteur interne (``local`` + ``pymupdf``) : aucune clé exigée."""
    env = albert_env
    monkeypatch.setenv("OCR_SERVER", "local")
    monkeypatch.setenv("OCR_MODEL", "pymupdf")
    path = _fresh_ocr_session(env, "gsess-ocr-explicit-local")
    resp, errors = _post(env, "member_openrouter", "/process_dataframe", {"path": path})
    assert resp.status_code == 200 and errors == [], resp.text[:300]


def test_route_ocr_refuses_an_undeclared_server(albert_env, monkeypatch):
    """Serveur OCR non déclaré au bloc 1 : 400 avant tout lancement."""
    env = albert_env
    monkeypatch.setenv("OCR_SERVER", "https://ocr.example.org/v1")
    monkeypatch.setenv("OCR_MODEL", "x")
    path = _fresh_ocr_session(env, "gsess-ocr-explicit-unknown")
    resp, _ = _post(env, "admin", "/process_dataframe", {"path": path})
    assert resp.status_code == 400 and resp.json()["model_not_configured"] is True
    assert env.capture.take() == ([], [])


def test_route_ocr_requires_the_fallback_key_too(albert_env, monkeypatch):
    """Repli Mistral déclaré : sa clé est exigée aussi (moteur principal local, sans clé)."""
    env = albert_env
    for name, value in {"OCR_SERVER": "local", "OCR_MODEL": "pymupdf", "MISTRAL_API_BASE_URL": MISTRAL,
                        "OCR_SERVER_FALLBACK": MISTRAL, "OCR_MODEL_FALLBACK": "mistral-ocr-latest"}.items():
        monkeypatch.setenv(name, value)
    path = _fresh_ocr_session(env, "gsess-ocr-fallback-nokey")
    resp, _ = _post(env, "member_openrouter", "/process_dataframe", {"path": path})
    assert resp.status_code == 403 and resp.json()["credential_required"] == "mistral_api_key"
    assert env.capture.take() == ([], [])
    path = _fresh_ocr_session(env, "gsess-ocr-fallback-key")
    resp, errors = _post(env, "member_mistral", "/process_dataframe", {"path": path})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    monkeypatch.setenv("OCR_SERVER_FALLBACK", "local")
    monkeypatch.setenv("OCR_MODEL_FALLBACK", "pymupdf")
    path = _fresh_ocr_session(env, "gsess-ocr-fallback-same")
    resp, _ = _post(env, "admin", "/process_dataframe", {"path": path})
    assert resp.status_code == 400 and resp.json()["model_not_configured"] is True
