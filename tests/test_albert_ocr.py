"""Tests du lot 4 : OCR souverain Albert en tête de chaîne (``rad_dataframe`` et ``rad_albert/ocr.py``).

Trois familles, toutes hors ligne (``tests/albert_fakes.FakeAlbert`` derrière un
``httpx.MockTransport``, clé factice ``FAKE_ALBERT_KEY`` seulement) :

* **chaîne et identité à l'octet** : Albert désactivé, aucun appel Albert même
  avec une clé, message d'erreur final, journaux (G13) et résultats (G1)
  identiques aux références ; parseur ``pages[]`` de Mistral inchangé ;
  ``OCRResult`` jamais déballé par position ; Albert activé, le maillon passe
  avant Mistral et tout repli vers Mistral est tracé (``fallback_from`` et
  entrée ``OCR_PROVIDER_FALLBACK``) ; une erreur de compte n'est jamais
  retentée et est mémorisée pour le processus ;
* **``/v1/ocr``** (``albert_mistral_ocr``) : pages indexées à partir de 0,
  marqueurs ``<!-- Page N -->`` avec N = index + 1 continus d'une part à
  l'autre, absence d'accès (403, ou 404 « Model … not found » mesuré en D3)
  mémorisée sans seconde sonde, 413 suivi d'un seul redécoupage puis de
  LightOnOCR ;
* **LightOnOCR** (``albert_lightonocr``) : message image seule et paramètres
  §9.1, rastérisation d'une page A4 à 1 541 px au plus, marqueurs continus
  avec page en échec, ``finish_reason=length`` partiel, seuil d'échec, 404 en
  une seule requête, sémaphore libéré pendant le backoff, contrôle de densité
  propre à LightOnOCR, entrée ``OCR_PARTIAL`` ; pool à plusieurs fils
  (``ALBERT_OCR_CONCURRENCY=3``) : marqueurs dans l'ordre malgré des fins
  désordonnées, fenêtre de rendu bornée à 2 × fils, arrêt du pool sur erreur
  de compte (drapeau partagé, envois en file annulés), page transitoire sans
  trou de marqueurs.

Tout client Albert construit pendant un test est branché sur le faux par un
crochet sur ``AlbertClient.__init__`` (transport injecté, sommeil factice) : une
fuite vers le vrai transport serait refusée par la garde réseau du
``conftest.py`` racine. Les réponses LightOnOCR sont routées page par page : la
page est déduite de la largeur de l'image reçue (chaque page du PDF de test a
une largeur propre).
"""

from __future__ import annotations

import ast
import base64
import importlib
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path

import fitz  # type: ignore[import-not-found]
import httpx
import pytest

from scripts import rad_dataframe as rad
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from tests.albert_fakes import FAKE_ALBERT_KEY, NAMED_ERRORS, FakeAlbert

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SCRIPTS_DIR = REPO_ROOT / "tests" / "fixtures" / "albert" / "golden_off" / "scripts"

OCR_BASE_FIELDS = ("text", "provider", "partial", "pages_done", "pages_total", "error")
PROVIDER_CHAT = "albert_lightonocr"
PROVIDER_DOC = "albert_mistral_ocr"
OCR_CHAT_MODEL = "lightonocr-2-1b"
OCR_CHAT_ALIAS = "openweight-ocr"
OCR_DOC_MODEL = "mistral-ocr-2512"
FAKE_MISTRAL_KEY = "fake-mistral-0001"
CHAT_PATH = "/v1/chat/completions"
OCR_PATH = "/v1/ocr"

# Colonnes d'un enregistrement ``_process_single_zotero_item`` à la baseline (liste littérale).
BASELINE_RECORD_KEYS = [
    "itemKey", "type", "title", "abstract", "date", "url", "doi", "authors",
    "filename", "path", "attachment_title", "texteocr", "texteocr_provider",
    "texteocr_partial", "texteocr_pages_done", "texteocr_pages_total",
]

# Même motif que ``app/utils/book_note_generator.py::PAGE_MARKER_RE`` (vérifié par un test).
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
FAILED_PAGE_RE = re.compile(r"<!--\s*OCR ÉCHOUÉ \(albert_lightonocr\) : [^\n]*-->")

# Réglages communs du maillon Albert en test : pas de ledger écrit, backoff minuscule.
ALBERT_TEST_ENV = {
    "ALBERT_ENABLED": "1",
    "OCR_ENABLE_ALBERT": "1",
    "ALBERT_USAGE_LOG": "0",
    "ALBERT_RETRY_BACKOFF": "0.01",
    "ALBERT_RETRY_MAX_BACKOFF": "0.05",
}

# Statut → corps d'erreur du faux (textes des sondes).
_ERROR_BODIES = {
    400: NAMED_ERRORS["bad_request"][1],
    401: NAMED_ERRORS["invalid_key"][1],
    403: NAMED_ERRORS["no_access"][1],
    404: {"detail": f"Model {OCR_CHAT_MODEL} not found."},
    413: NAMED_ERRORS["payload_too_large"][1],
    422: NAMED_ERRORS["wrong_model_type"][1],
    500: NAMED_ERRORS["server_error"][1],
    502: NAMED_ERRORS["bad_gateway"][1],
}


# ---------------------------------------------------------------------------
# Outils : PDF, images, champs
# ---------------------------------------------------------------------------
def _sizes(pages):
    """Tailles (points) des pages de test : largeurs toutes distinctes (200, 240, 280…)."""
    return [(200 + 40 * index, 280) for index in range(pages)]


def _build_pdf(directory, name, pages=3, *, sizes=None):
    """PDF de test (une page par taille) ; renvoie son chemin."""
    sizes = list(sizes) if sizes is not None else _sizes(pages)
    doc = fitz.open()
    for number, (width, height) in enumerate(sizes, start=1):
        page = doc.new_page(width=width, height=height)
        page.insert_text((12, 40), f"Page {number}", fontsize=11)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def _page_text(number):
    """Texte LightOnOCR simulé d'une page (plus de 500 caractères : densité saine)."""
    return (f"Texte transcrit de la page {number}. " * 30).strip()


def _expected_size(width, height, dpi=200, max_side=1540):
    """Taille attendue (pixels) du rendu : ``zoom = min(dpi/72, max_side/max(w, h))``."""
    zoom = min(dpi / 72.0, max_side / float(max(width, height)))
    return width * zoom, height * zoom


def _b64_payload(value):
    """Octets d'une valeur base64 (URI ``data:`` acceptée)."""
    text = str(value)
    if text.startswith("data:"):
        text = text.split(",", 1)[1]
    return base64.b64decode(text)


def _png_size(value):
    """(largeur, hauteur) d'une image PNG donnée en base64 ou en URI ``data:``."""
    raw = _b64_payload(value)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "image LightOnOCR attendue en PNG"
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def _image_url(call):
    """URL de l'unique image d'une requête de chat LightOnOCR."""
    content = call.json["messages"][0]["content"]
    parts = [part for part in content if isinstance(part, dict) and part.get("type") == "image_url"]
    assert len(parts) == 1, content
    image = parts[0]["image_url"]
    return image["url"] if isinstance(image, dict) else image


def _document_page_count(call):
    """Nombre de pages du PDF envoyé à ``/v1/ocr`` (URI ``data:``)."""
    raw = _b64_payload(call.json["document"]["document_url"])
    with fitz.open(stream=raw, filetype="pdf") as doc:
        return doc.page_count


def _markers(text):
    """Numéros des marqueurs ``<!-- Page N -->`` d'un texte, dans l'ordre."""
    return [int(n) for n in PAGE_MARKER_RE.findall(text or "")]


def _held(semaphore):
    """Vrai si le sémaphore (capacité 1) est tenu au moment de l'appel."""
    if semaphore.acquire(blocking=False):
        semaphore.release()
        return False
    return True


class _NoSleepTime:
    """Remplaçant du module ``time`` de ``rad_dataframe`` : ``sleep`` noté, le reste délégué."""

    def __init__(self):
        """Mémorise le vrai module et une liste de sommeils vide."""
        import time as real_time

        self._time = real_time
        self.sleeps = []

    def sleep(self, seconds):
        """Note le sommeil demandé sans attendre."""
        self.sleeps.append(seconds)

    def __getattr__(self, name):
        """Délègue tout autre attribut au vrai module ``time``."""
        return getattr(self._time, name)


# ---------------------------------------------------------------------------
# Crochet client, chaîne isolée, espion Mistral
# ---------------------------------------------------------------------------
def _hook_albert_client(monkeypatch, sink):
    """Branche tout ``AlbertClient`` construit pendant le test sur ``sink.fake`` et ``sink.sleep``."""
    original = AlbertClient.__init__

    def patched_init(client, *args, **kwargs):
        """Injecte le transport du faux et le sommeil factice, puis construit le client."""
        if kwargs.get("transport") is None:
            kwargs["transport"] = sink.fake.transport
        kwargs.setdefault("sleep", sink.sleep)
        original(client, *args, **kwargs)
        sink.clients.append(client)

    monkeypatch.setattr(AlbertClient, "__init__", patched_init)


def _reset_albert_process_state(monkeypatch, cfg=None):
    """Remet à zéro l'état Albert du processus (mémos, avertissements uniques, ledger, transport).

    ``cfg`` remplace la configuration lue à l'import (``rad.ALBERT_CONFIG``) ;
    aucune erreur de configuration n'est simulée.
    """
    if cfg is not None:
        monkeypatch.setattr(rad, "ALBERT_CONFIG", cfg)
    monkeypatch.setattr(rad, "_ALBERT_CONFIG_ERROR", None)
    monkeypatch.setattr(rad, "_ALBERT_ACCOUNT_DISABLED", None)
    monkeypatch.setattr(rad, "_ALBERT_V1_OCR_AVAILABLE", None)
    monkeypatch.setattr(rad, "_ALBERT_WARNED", set())
    monkeypatch.setattr(rad, "_ALBERT_OCR_LEDGER", None)
    monkeypatch.setattr(rad, "_ALBERT_OCR_TRANSPORT", None)
    monkeypatch.setattr(rad, "_ALBERT_OCR_MODEL_IDS", {})


def _isolate_chain(monkeypatch):
    """Chaîne OCR sans Mistral, sans OpenAI, sans OCR local ; seuil de densité de la baseline."""
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
    monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", False)
    monkeypatch.setattr(rad, "OCR_ENABLE_LOCAL_FALLBACK", True)
    monkeypatch.setattr(rad, "_local_ocr_available", lambda *a, **k: False)
    monkeypatch.setattr(rad, "OCR_MIN_CHARS_PER_PAGE", 500)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)


class MistralSpy:
    """Remplaçant de ``_extract_text_with_mistral`` : note l'appel (et l'état du faux Albert)."""

    def __init__(self, fake=None, *, error=None):
        """Prépare l'espion ; ``error`` est levée à chaque appel si fournie."""
        self.fake = fake
        self.error = error
        self.calls = []

    def __call__(self, pdf_path, max_pages=None, **kwargs):
        """Renvoie un ``_MistralOcr`` complet (ou lève ``error``)."""
        self.calls.append({
            "pdf": os.path.basename(pdf_path),
            "albert_calls_before": len(self.fake.calls) if self.fake is not None else 0,
        })
        if self.error is not None:
            raise self.error
        pages = rad._pdf_page_count(pdf_path)
        text = "\n\n".join(f"<!-- Page {n} -->\nTexte Mistral page {n}" for n in range(1, pages + 1))
        return rad._MistralOcr(text=text, partial=False, pages_done=pages, pages_total=pages)


def _install_mistral(monkeypatch, fake=None, *, error=None):
    """Active le maillon Mistral (clé factice) avec un espion ; renvoie l'espion."""
    spy = MistralSpy(fake, error=error)
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", FAKE_MISTRAL_KEY)
    monkeypatch.setattr(rad, "_extract_text_with_mistral", spy)
    return spy


class AlbertChain:
    """Chaîne OCR de ``rad_dataframe`` avec le maillon Albert actif, servie par ``FakeAlbert``."""

    def __init__(self, monkeypatch, fake=None, *, mode="auto", env=None):
        """Active Albert (env et constantes de module), isole la chaîne et branche le faux.

        Args:
            monkeypatch: fixture pytest.
            fake: API simulée (défaut : ``FakeAlbert()``, sans accès à ``/v1/ocr``).
            mode: ``ALBERT_OCR_MODE`` (``auto``, ``chat`` ou ``ocr``).
            env: variables ``ALBERT_*`` supplémentaires.
        """
        self.monkeypatch = monkeypatch
        self.fake = fake if fake is not None else FakeAlbert()
        self.clients = []
        self.sleeps = []
        self.sleep_observer = None
        settings = dict(ALBERT_TEST_ENV)
        settings["ALBERT_OCR_MODE"] = mode
        settings.update(env or {})
        settings = {name: str(value) for name, value in settings.items()}
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        # Constantes de script lues à l'import (décision 11) : configuration,
        # interrupteur et clé sont posés comme le ferait l'import avec cet env.
        self.cfg = AlbertConfig.from_env(settings)
        _reset_albert_process_state(monkeypatch, self.cfg)
        monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", bool(self.cfg.enabled and self.cfg.ocr_enabled))
        monkeypatch.setattr(rad, "ALBERT_API_KEY", FAKE_ALBERT_KEY)
        monkeypatch.setattr(rad, "_ALBERT_OCR_TRANSPORT", self.fake.transport)
        _isolate_chain(monkeypatch)
        _hook_albert_client(monkeypatch, self)

    def sleep(self, seconds):
        """Sommeil factice des clients Albert : noté (et observé), sans attente."""
        self.sleeps.append(float(seconds))
        if self.sleep_observer is not None:
            self.sleep_observer(seconds)

    def use_mistral(self, *, error=None):
        """Active le maillon Mistral (espion) ; renvoie l'espion."""
        return _install_mistral(self.monkeypatch, self.fake, error=error)

    def router(self, sizes):
        """Routeur LightOnOCR page par page pour un PDF de pages ``sizes``."""
        return PageRouter(self.fake, sizes)

    def run(self, pdf_path, **kwargs):
        """``extract_text_with_ocr(return_details=True)`` sur ``pdf_path``."""
        return rad.extract_text_with_ocr(pdf_path, return_details=True, **kwargs)

    def chat_calls(self):
        """Requêtes LightOnOCR reçues par le faux."""
        return self.fake.calls_to("POST", CHAT_PATH)

    def v1_calls(self):
        """Requêtes ``/v1/ocr`` reçues par le faux."""
        return self.fake.calls_to("POST", OCR_PATH)


class PageRouter:
    """Réponses LightOnOCR page par page ; la page est déduite de la largeur de l'image reçue.

    Actions possibles par page (consommées dans l'ordre, ``"ok"`` ensuite) :
    ``"ok"`` (texte de la page), ``"empty"`` (contenu vide, ``stop``),
    ``"length"`` (texte, ``finish_reason=length``), ``"length_empty"``
    (contenu ``null``, ``length``), ou un statut HTTP d'erreur (``int``).
    """

    def __init__(self, fake, sizes, *, dpi=200, max_side=1540):
        """Remplace la route de chat du faux par ce routeur."""
        self.fake = fake
        self.widths = [_expected_size(w, h, dpi, max_side)[0] for w, h in sizes]
        self.plans = {}
        self.texts = {}
        self.requests = []
        self.observer = None
        self._lock = threading.Lock()
        self._local = threading.local()
        self._original = fake._route_post_v1_chat_completions
        fake._route_post_v1_chat_completions = self._route
        fake.ocr_chat_reply = self._reply_text

    def plan(self, page, *actions):
        """Programme les réponses successives de la page ``page`` (numérotée à partir de 1)."""
        self.plans.setdefault(page, []).extend(actions)
        return self

    def text(self, page):
        """Texte renvoyé pour la page ``page``."""
        return self.texts.get(page, _page_text(page))

    def page_of(self, call):
        """Numéro (à partir de 1) de la page dont l'image est jointe à ``call``."""
        width, _height = _png_size(_image_url(call))
        gaps = [abs(width - expected) for expected in self.widths]
        best = min(range(len(gaps)), key=gaps.__getitem__)
        assert gaps[best] <= 3, f"largeur d'image inattendue : {width} px (attendu : {self.widths})"
        return best + 1

    def _reply_text(self, image_url, index):
        """Texte de la réponse en cours (posé par ``_route`` dans le même thread)."""
        return self._local.text

    def _route(self, call):
        """Route de chat du faux : applique l'action programmée pour la page."""
        page = self.page_of(call)
        with self._lock:
            self.requests.append(page)
            queue = self.plans.get(page) or []
            action = queue.pop(0) if queue else "ok"
        if self.observer is not None:
            self.observer(page)
        if isinstance(action, int):
            return httpx.Response(action, json=_ERROR_BODIES.get(action, {"detail": "Error"}))
        self._local.text = "" if action in ("empty", "length_empty") else self.text(page)
        response = self._original(call)
        if action in ("length", "length_empty"):
            body = response.json()
            choice = body["choices"][0]
            choice["finish_reason"] = "length"
            if action == "length_empty":
                choice["message"]["content"] = None
            response = httpx.Response(200, json=body)
        return response


def _albert_off_with_key(monkeypatch):
    """Albert désactivé, clé factice présente dans l'env et dans la constante de module."""
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", False)
    monkeypatch.setattr(rad, "ALBERT_API_KEY", FAKE_ALBERT_KEY)


class _OffSink:
    """Collecteur du crochet client quand Albert est désactivé (aucun client attendu)."""

    def __init__(self):
        """Faux Albert et listes vides."""
        self.fake = FakeAlbert()
        self.clients = []
        self.sleeps = []

    def sleep(self, seconds):
        """Sommeil factice noté."""
        self.sleeps.append(seconds)


def _albert_log_records(caplog, start=0, level=logging.WARNING):
    """Enregistrements de niveau >= ``level`` (depuis ``start``) qui mentionnent Albert."""
    return [
        record for record in caplog.records[start:]
        if record.levelno >= level and "albert" in record.getMessage().lower()
    ]


def _zotero_item(key, relative_path, title="Document de test Albert"):
    """Élément Zotero minimal avec une pièce jointe PDF."""
    return {
        "key": key, "itemType": "journalArticle", "title": title,
        "abstractNote": "", "date": "2024", "url": "", "DOI": "",
        "creators": [{"lastName": "Durand", "firstName": "Anne"}],
        "attachments": [{"path": relative_path, "title": "PDF"}],
    }


@pytest.fixture(autouse=True)
def _fresh_albert_state():
    """Oublie les limiteurs partagés et le cache du preflight avant et après chaque test."""
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def chain(monkeypatch):
    """Fabrique de chaînes Albert actives : ``chain(fake=None, mode='auto', env=None)``."""

    def make(fake=None, *, mode="auto", env=None):
        """Construit une ``AlbertChain`` pour ce test."""
        return AlbertChain(monkeypatch, fake, mode=mode, env=env)

    return make


def _ocr_module():
    """Module ``scripts.rad_albert.ocr`` (importé à la demande)."""
    return importlib.import_module("scripts.rad_albert.ocr")


# ===========================================================================
# Chaîne et identité à l'octet (Albert OFF)
# ===========================================================================
@pytest.mark.parametrize(
    "case, enabled, key, env_on",
    [
        ("off_with_key", False, FAKE_ALBERT_KEY, False),
        ("off_constant_env_on", False, FAKE_ALBERT_KEY, True),
        ("on_without_key", True, None, True),
    ],
    ids=["off_with_key", "off_constant_env_on", "on_without_key"],
)
def test_off_no_albert_call_even_with_key(case, enabled, key, env_on, monkeypatch, tmp_path, network_guard):
    sink = _OffSink()
    _isolate_chain(monkeypatch)
    _reset_albert_process_state(monkeypatch)
    _hook_albert_client(monkeypatch, sink)
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    monkeypatch.setenv("ALBERT_ENABLED", "1" if env_on else "0")
    monkeypatch.setenv("OCR_ENABLE_ALBERT", "1" if env_on else "0")
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", enabled)
    monkeypatch.setattr(rad, "ALBERT_API_KEY", key)
    attempts = []
    real_try = rad._try_albert_ocr

    def try_spy(*args, **kwargs):
        """Espion du maillon Albert (ne doit jamais être appelé)."""
        attempts.append(args)
        return real_try(*args, **kwargs)

    monkeypatch.setattr(rad, "_try_albert_ocr", try_spy)
    mistral = _install_mistral(monkeypatch, sink.fake)

    result = rad.extract_text_with_ocr(_build_pdf(tmp_path, case, 2), return_details=True)

    assert result.provider == "mistral"
    if enabled:
        # Albert sélectionné sans clé : aucun appel, mais repli tracé (jamais silencieux).
        assert isinstance(result.fallback_from, str) and result.fallback_from.startswith("albert")
    else:
        assert getattr(result, "fallback_from", None) is None
    assert len(mistral.calls) == 1
    assert attempts == []
    assert sink.clients == []
    assert sink.fake.calls == []
    assert not any("etalab" in record.host for record in network_guard.records())
    assert rad._ALBERT_ACCOUNT_DISABLED is None
    assert rad._ALBERT_V1_OCR_AVAILABLE is None


def test_off_final_error_message_identical(monkeypatch, tmp_path):
    golden = json.loads((GOLDEN_SCRIPTS_DIR / "g1_all_providers_fail.json").read_text(encoding="utf-8"))
    expected = golden["exception"]
    sink = _OffSink()
    _isolate_chain(monkeypatch)
    _reset_albert_process_state(monkeypatch)
    _hook_albert_client(monkeypatch, sink)
    _albert_off_with_key(monkeypatch)
    _install_mistral(monkeypatch, sink.fake, error=rad.OCRExtractionError("Mistral KO (test)"))
    monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", lambda *a, **k: "")

    with pytest.raises(rad.OCRExtractionError) as excinfo:
        rad.extract_text_with_ocr(_build_pdf(tmp_path, "illisible", 2), return_details=True)

    assert type(excinfo.value).__name__ == expected["type"]
    assert str(excinfo.value) == expected["message"]
    assert sink.clients == [] and sink.fake.calls == []


def test_on_final_error_message_mentions_albert(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_MAX_RETRIES": "0"})
    sizes = _sizes(2)
    router = albert.router(sizes)
    for page in (1, 2):
        router.plan(page, 500)
    albert.use_mistral(error=rad.OCRExtractionError("Mistral KO (test)"))
    albert.monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", lambda *a, **k: "")
    golden = json.loads((GOLDEN_SCRIPTS_DIR / "g1_all_providers_fail.json").read_text(encoding="utf-8"))

    with pytest.raises(rad.OCRExtractionError) as excinfo:
        albert.run(_build_pdf(tmp_path, "illisible_on", sizes=sizes))

    message = str(excinfo.value)
    assert message != golden["exception"]["message"]
    assert "albert" in message.lower()
    assert FAKE_ALBERT_KEY not in message


def _assert_matches_golden(golden, name, obj):
    """Compare le rendu JSON de ``obj`` au golden ``name`` (lecture seule, jamais réécrit)."""
    path = GOLDEN_SCRIPTS_DIR / name
    expected = path.read_bytes()
    actual = golden._dump(obj).encode("utf-8")
    if actual != expected:
        golden._fail_with_diff(name, expected.decode("utf-8"), actual.decode("utf-8"))


def _golden_harness_env(monkeypatch, golden):
    """Environnement des goldens (``_golden_env``), puis clé Albert factice avec Albert OFF."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in golden.ALBERT_OFF_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", False)
    monkeypatch.setattr(rad, "ALBERT_API_KEY", FAKE_ALBERT_KEY)


def test_off_logs_match_golden(monkeypatch, tmp_path, caplog):
    golden = importlib.import_module("tests.test_albert_off_golden")
    _golden_harness_env(monkeypatch, golden)
    sink = _OffSink()
    _hook_albert_client(monkeypatch, sink)
    caplog.set_level(logging.INFO)

    success_root = tmp_path / "success"
    success_root.mkdir()
    _capture, logs = golden._run_http_scenario(monkeypatch, success_root, "pages_index_markdown", caplog)
    _assert_matches_golden(golden, "g13_ocr_logs_mistral_success.json", logs)

    http_root = tmp_path / "http"
    http_root.mkdir()
    http_logs = {}
    for name in golden.HTTP_SCENARIO_NAMES:
        _capture, logs = golden._run_http_scenario(monkeypatch, http_root, name, caplog)
        http_logs[name] = logs
    _assert_matches_golden(golden, "g13_ocr_logs_mistral_http.json", http_logs)

    split_root = tmp_path / "split"
    split_root.mkdir()
    harness = golden._OcrHarness(monkeypatch, split_root)
    golden._apply_force_split(monkeypatch)
    start = len(caplog.records)
    golden._scenario_mistral_split_salvage(harness)
    _assert_matches_golden(
        golden, "g13_ocr_logs_mistral_split_salvage.json", golden._log_records(caplog, split_root, start)
    )

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    harness = golden._OcrHarness(monkeypatch, legacy_root)
    start = len(caplog.records)
    golden._scenario_legacy_no_keys(harness)
    first = golden._log_records(caplog, legacy_root, start)
    harness.calls.clear()
    start = len(caplog.records)
    golden._scenario_legacy_sparse_openai_key(harness, monkeypatch)
    second = golden._log_records(caplog, legacy_root, start)
    _assert_matches_golden(golden, "g13_ocr_logs_legacy.json", {
        "legacy_no_keys": first,
        "legacy_sparse_openai_key_set": second,
    })

    assert sink.clients == [] and sink.fake.calls == []


def test_off_ocr_results_match_golden_with_key(monkeypatch, tmp_path):
    golden = importlib.import_module("tests.test_albert_off_golden")
    _golden_harness_env(monkeypatch, golden)
    sink = _OffSink()
    _hook_albert_client(monkeypatch, sink)

    root = tmp_path / "legacy"
    root.mkdir()
    _assert_matches_golden(
        golden, "g1_legacy_no_mistral_key.json", golden._scenario_legacy_no_keys(golden._OcrHarness(monkeypatch, root))
    )
    root = tmp_path / "fast"
    root.mkdir()
    capture = golden._scenario_mistral_fast_path(golden._OcrHarness(monkeypatch, root), monkeypatch)
    _assert_matches_golden(golden, "g1_mistral_fast_path.json", capture)
    root = tmp_path / "split"
    root.mkdir()
    harness = golden._OcrHarness(monkeypatch, root)
    golden._apply_force_split(monkeypatch)
    _assert_matches_golden(golden, "g1_mistral_split_salvage.json", golden._scenario_mistral_split_salvage(harness))

    assert sink.clients == [] and sink.fake.calls == []


_OFF_SUBPROCESS_CODE = r"""
import json, sys
sys.path.insert(0, ".")
import scripts.rad_dataframe as rad
rad.MISTRAL_API_KEY = None
rad.OCR_ENABLE_OPENAI_FALLBACK = False
rad._local_ocr_available = lambda *a, **k: False
result = rad.extract_text_with_ocr(sys.argv[1], return_details=True)
watched = ("scripts.rad_albert.client", "scripts.rad_albert.ocr", "scripts.rad_albert.limiter",
           "scripts.rad_albert.retry", "rad_albert.client", "rad_albert.ocr")
print(json.dumps({
    "provider": result.provider,
    "fallback_from": result.fallback_from,
    "enabled": rad.OCR_ENABLE_ALBERT,
    "key_set": rad.ALBERT_API_KEY is not None,
    "loaded": sorted(name for name in watched if name in sys.modules),
}))
"""


def _dense_pdf(directory, name, pages=2):
    """PDF dont chaque page porte plus de 50 mots (l'extracteur legacy n'appelle jamais Tesseract)."""
    doc = fitz.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        words = [f"terme{number}x{index}" for index in range(100)]
        for row, start in enumerate(range(0, len(words), 10)):
            page.insert_text((40, 60 + 16 * row), " ".join(words[start:start + 10]), fontsize=9)
    path = Path(directory) / f"{name}.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def test_off_never_imports_albert_client_or_ocr(tmp_path):
    import subprocess
    import sys

    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith("ALBERT_") and not any(tag in name for tag in ("KEY", "SECRET", "TOKEN", "PASSWORD"))
    }
    env.update({
        "ALBERT_ENABLED": "0",
        "OCR_ENABLE_ALBERT": "0",
        "ALBERT_API_KEY": FAKE_ALBERT_KEY,
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost,testserver",
    })
    completed = subprocess.run(
        [sys.executable, "-c", _OFF_SUBPROCESS_CODE, _dense_pdf(tmp_path, "article_off")],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
        stdin=subprocess.DEVNULL,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {
        "provider": "legacy",
        "fallback_from": None,
        "enabled": False,
        "key_set": True,
        "loaded": [],
    }


def test_off_record_columns_unchanged(monkeypatch, tmp_path):
    sink = _OffSink()
    _isolate_chain(monkeypatch)
    _hook_albert_client(monkeypatch, sink)
    _albert_off_with_key(monkeypatch)
    _install_mistral(monkeypatch, sink.fake)
    _build_pdf(tmp_path / "files", "article", 2)

    outcome = rad._process_single_zotero_item(_zotero_item("ITEMOFF1", "files/article.pdf"), str(tmp_path))

    assert [list(record) for record in outcome.records] == [BASELINE_RECORD_KEYS]
    assert outcome.records[0]["texteocr_provider"] == "mistral"
    assert outcome.errors == []
    assert sink.fake.calls == []


# ---------------------------------------------------------------------------
# Parseur pages[] de Mistral : extraction pure, octets inchangés
# ---------------------------------------------------------------------------
def _baseline_payload_to_markdown(response_payload):
    """Copie verbatim du parseur ``pages[]`` de la baseline (``_mistral_upload_and_ocr_once``, commit 3a37a43)."""
    markdown_text = ""
    if isinstance(response_payload, dict):
        pages = response_payload.get("pages")
        if isinstance(pages, list) and pages:
            page_blocks = []
            for fallback_idx, page in enumerate(pages, start=1):
                if not isinstance(page, dict):
                    continue
                raw_idx = page.get("index")
                if isinstance(raw_idx, int):
                    page_num = raw_idx + 1
                else:
                    page_num = fallback_idx
                page_md = ""
                for key in ("markdown", "text"):
                    value = page.get(key)
                    if isinstance(value, str) and value.strip():
                        page_md = value.strip()
                        break
                if page_md:
                    page_blocks.append(f"<!-- Page {page_num} -->\n{page_md}")
            if page_blocks:
                markdown_text = "\n\n".join(page_blocks)

        if not markdown_text:
            for key in ("markdown", "text"):
                candidate = response_payload.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    markdown_text = candidate.strip()
                    break

        if not markdown_text:
            outputs = response_payload.get("output")
            if isinstance(outputs, list):
                blocks = []
                for block in outputs:
                    if isinstance(block, dict):
                        for key in ("markdown", "text", "content"):
                            value = block.get(key)
                            if isinstance(value, str) and value.strip():
                                blocks.append(value.strip())
                                break
                markdown_text = "\n\n".join(blocks)

    return markdown_text.strip()


def _baseline_shift(text, page_offset):
    """Renumérotation de la baseline (``_extract_text_with_mistral``, parts découpées)."""
    if page_offset <= 0:
        return text

    def shift(match):
        """Décale le numéro d'un marqueur de page."""
        return f"<!-- Page {int(match.group(1)) + page_offset} -->"

    return re.sub(r"<!--\s*Page\s+(\d+)\s*-->", shift, text)


def _parser_payloads():
    """Charges ``/v1/ocr`` de test : celles des goldens HTTP, plus des formes limites."""
    golden = importlib.import_module("tests.test_albert_off_golden")
    payloads = []
    for spec in golden._http_scenario_specs().values():
        for answer in spec.get("ocr") or []:
            if isinstance(answer, BaseException) or getattr(answer, "status_code", 0) != 200:
                continue
            payloads.append(answer.json())
    payloads += [
        {"pages": [{"index": 4, "markdown": "Cinquième"}, {"index": 0, "markdown": "Première"}]},
        {"pages": [{"index": 0, "markdown": ""}, {"index": 1, "text": "  Seul texte  "}]},
        {"pages": [{"markdown": "Sans index A"}, {"markdown": "Sans index B"}]},
        {"pages": [{"index": 0, "markdown": "É accentué\n\n| a | b |\n|---|---|"}]},
        {"pages": [], "output": []},
        {"pages": None, "markdown": None, "text": 3},
        {"output": [{"text": "  a  "}, {"content": " b "}]},
        [],
        None,
        "texte brut",
    ]
    return payloads


def test_mistral_payload_parser_bytes_unchanged():
    parser = rad._ocr_pages_payload_to_markdown
    payloads = _parser_payloads()
    assert len(payloads) >= 20
    for payload in payloads:
        expected = _baseline_payload_to_markdown(payload)
        assert parser(payload) == expected, payload
        assert parser(payload, page_offset=0) == expected, payload
    for payload in payloads:
        pages = payload.get("pages") if isinstance(payload, dict) else None
        if not (isinstance(pages, list) and pages):
            continue
        base = _baseline_payload_to_markdown(payload)
        if not PAGE_MARKER_RE.search(base):
            continue
        for offset in (1, 7, 500):
            assert parser(payload, page_offset=offset) == _baseline_shift(base, offset), (payload, offset)
    # Le chemin Mistral appelle le parseur extrait (pas de copie en ligne).
    tree = ast.parse(Path(rad.__file__).read_text(encoding="utf-8"))
    once = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_mistral_upload_and_ocr_once"
    )
    called = {
        node.func.id for node in ast.walk(once)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_ocr_pages_payload_to_markdown" in called
    assert "page_blocks" not in ast.unparse(once)


def test_page_marker_regex_matches_book_note_generator():
    source = (REPO_ROOT / "app" / "utils" / "book_note_generator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    patterns = [
        node.value.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PAGE_MARKER_RE" for t in node.targets)
        and isinstance(node.value, ast.Call) and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
    ]
    assert patterns == [PAGE_MARKER_RE.pattern]


# ---------------------------------------------------------------------------
# OCRResult : nouveau dernier champ, jamais déballé par position
# ---------------------------------------------------------------------------
_OCR_PRODUCERS = {
    "extract_text_with_ocr",
    "extract_text_with_ocr_retry",
    "_finalize_ocr_result",
    "_extract_text_from_epub",
    "_extract_text_from_plain",
    "_try_albert_ocr",
    "OCRResult",
}
_OCR_VARIABLE_RE = re.compile(r"^ocr_(?:payload|result|outcome|details)s?$")


def _call_name(node):
    """Nom de la fonction appelée par ``node`` (``f(...)`` ou ``x.f(...)``), sinon ``None``."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


def _ocr_producers(trees):
    """Fonctions connues pour renvoyer un ``OCRResult`` (liste fixe, plus annotations de retour)."""
    names = set(_OCR_PRODUCERS)
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
                if "OCRResult" in ast.unparse(node.returns):
                    names.add(node.name)
    return names


def _scope_nodes(scope):
    """Nœuds d'une portée (module ou fonction), sans descendre dans les fonctions et classes imbriquées."""
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, nested):
            stack.extend(ast.iter_child_nodes(node))


def _positional_uses(tree, producers):
    """Usages positionnels d'un ``OCRResult`` : déballage, indice, tranche, étoile, itération."""
    findings = set()
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        nodes = list(_scope_nodes(scope))
        tracked = set()
        for node in nodes:
            if isinstance(node, ast.Name) and _OCR_VARIABLE_RE.match(node.id):
                tracked.add(node.id)
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and _call_name(node.value) in producers:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        tracked.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
                        findings.add(("déballage", node.lineno))

        def is_ocr(value):
            """Vrai si ``value`` désigne un résultat OCR (variable suivie ou appel producteur)."""
            return (isinstance(value, ast.Name) and value.id in tracked) or _call_name(value) in producers

        for node in nodes:
            if isinstance(node, ast.Subscript) and is_ocr(node.value):
                findings.add(("indice", node.lineno))
            elif isinstance(node, ast.Starred) and is_ocr(node.value):
                findings.add(("étoile", node.lineno))
            elif isinstance(node, (ast.For, ast.comprehension)) and is_ocr(node.iter):
                findings.add(("itération", getattr(node, "lineno", getattr(node.iter, "lineno", 0))))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if node.value is not None and is_ocr(node.value) and any(
                    isinstance(t, (ast.Tuple, ast.List)) for t in targets
                ):
                    findings.add(("déballage", node.lineno))
            elif _call_name(node) in ("tuple", "list") and node.args and is_ocr(node.args[0]):
                findings.add(("conversion", node.lineno))
    return findings


def test_ocrresult_never_unpacked_positionally():
    files = sorted(
        path for root in ("scripts", "app") for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    assert any(path.name == "rad_dataframe.py" for path in files)
    trees = {}
    for path in files:
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
    producers = _ocr_producers(trees.values())
    offenders = sorted(
        f"{path.relative_to(REPO_ROOT)}:{line} ({kind})"
        for path, tree in trees.items()
        for kind, line in _positional_uses(tree, producers)
    )
    assert offenders == []


def test_ocr_positional_detector_catches_unpacking():
    sample = ast.parse(
        "def f(p):\n"
        "    text, provider, *rest = extract_text_with_ocr(p, return_details=True)\n"
        "    ocr_result = extract_text_with_ocr_retry(p)\n"
        "    return ocr_result[0], ocr_result.provider\n"
    )
    kinds = sorted(kind for kind, _line in _positional_uses(sample, set(_OCR_PRODUCERS)))
    assert kinds == ["déballage", "indice"]


def test_ocrresult_fallback_from_is_last_optional_field():
    assert rad.OCRResult._fields == OCR_BASE_FIELDS + ("fallback_from",)
    assert rad.OCRResult._field_defaults["fallback_from"] is None
    result = rad.OCRResult(text="t", provider="legacy")
    assert result.fallback_from is None
    assert rad._finalize_ocr_result("t", "legacy", True).fallback_from is None


# ===========================================================================
# Chaîne Albert active
# ===========================================================================
def test_on_albert_runs_before_mistral(chain, tmp_path):
    albert = chain(mode="auto")
    sizes = _sizes(3)
    router = albert.router(sizes)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "article", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert result.fallback_from is None
    assert result.partial is False
    assert (result.pages_done, result.pages_total) == (3, 3)
    assert mistral.calls == []
    assert sorted(router.requests) == [1, 2, 3]
    assert len(albert.v1_calls()) <= 1
    assert _markers(result.text) == [1, 2, 3]
    for page in (1, 2, 3):
        assert router.text(page) in result.text
    assert rad.ALBERT_OCR_SEMAPHORE is not rad.MISTRAL_SEMAPHORE
    assert albert.clients, "le maillon Albert doit construire un AlbertClient"


def test_on_v1_access_uses_albert_mistral_ocr(chain, tmp_path):
    fake = FakeAlbert(ocr_access=True)
    fake.ocr_doc_reply = lambda index: f"Markdown document index {index}. " * 25
    albert = chain(fake, mode="auto")
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "rapport", 3))

    assert result.provider == PROVIDER_DOC
    assert result.fallback_from is None
    assert mistral.calls == []
    assert albert.chat_calls() == []
    assert rad._ALBERT_V1_OCR_AVAILABLE is True
    for call in albert.v1_calls():
        assert call.json["model"] == OCR_DOC_MODEL


def test_mode_chat_never_calls_v1_ocr(chain, tmp_path):
    albert = chain(FakeAlbert(ocr_access=True), mode="chat")
    sizes = _sizes(2)
    albert.router(sizes)

    result = albert.run(_build_pdf(tmp_path, "chat_seul", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert albert.v1_calls() == []
    assert len(albert.chat_calls()) == 2


def test_albert_fail_then_mistral_recorded_fallback(chain, tmp_path, monkeypatch):
    albert = chain(mode="chat", env={"ALBERT_MAX_RETRIES": "0"})
    sizes = _sizes(3)
    router = albert.router(sizes)
    for page in (1, 2, 3):
        router.plan(page, 500, 500)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "echec", sizes=sizes))

    assert result.provider == "mistral"
    assert isinstance(result.fallback_from, str) and result.fallback_from.startswith("albert")
    assert len(mistral.calls) == 1
    assert mistral.calls[0]["albert_calls_before"] >= 1, "Albert doit être essayé avant Mistral"
    assert rad._ALBERT_ACCOUNT_DISABLED is None

    monkeypatch.setattr(rad, "time", _NoSleepTime())
    _build_pdf(tmp_path / "files", "echec_zotero", sizes=sizes)
    outcome = rad._process_single_zotero_item(_zotero_item("ITEMFB01", "files/echec_zotero.pdf"), str(tmp_path))

    assert [record["texteocr_provider"] for record in outcome.records] == ["mistral"]
    fallbacks = [entry for entry in outcome.errors if entry.get("error_type") == "OCR_PROVIDER_FALLBACK"]
    assert len(fallbacks) == 1
    entry = fallbacks[0]
    assert entry.get("itemKey") == "ITEMFB01"
    assert any("albert" in str(value).lower() for value in entry.values())
    assert any("mistral" in str(value).lower() for value in entry.values())
    assert FAKE_ALBERT_KEY not in json.dumps(outcome.errors, ensure_ascii=False)


_ACCOUNT_ERRORS = [
    ("invalid_key", "invalid_key"),
    ("account_expired", "account_expired"),
    ("budget_exhausted", "budget_exhausted"),
    ("quota_exhausted", {"status": 429, "body": {"detail": "Too many requests."}, "retry_after": 600}),
]


@pytest.mark.parametrize("reason, entry", _ACCOUNT_ERRORS, ids=[r for r, _ in _ACCOUNT_ERRORS])
def test_auth_error_falls_through_without_retry(reason, entry, chain, tmp_path):
    albert = chain(mode="auto")
    sizes = _sizes(2)
    albert.router(sizes)
    albert.fake.inject("*", "*", entry)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, f"compte_{reason}", sizes=sizes))

    assert len(albert.fake.calls) == 1, [(c.method, c.path) for c in albert.fake.calls]
    assert albert.sleeps == []
    assert result.provider == "mistral"
    assert isinstance(result.fallback_from, str) and result.fallback_from.startswith("albert")
    assert len(mistral.calls) == 1
    assert rad._ALBERT_ACCOUNT_DISABLED
    assert FAKE_ALBERT_KEY not in str(rad._ALBERT_ACCOUNT_DISABLED)


def test_account_error_memoized_skips_next_documents(chain, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    albert = chain(mode="auto")
    sizes = _sizes(2)
    albert.router(sizes)
    albert.fake.inject("*", "*", "invalid_key")
    mistral = albert.use_mistral()

    first = albert.run(_build_pdf(tmp_path, "doc1", sizes=sizes))
    calls_after_first = len(albert.fake.calls)
    warnings_after_first = len(_albert_log_records(caplog))
    marker = len(caplog.records)
    second = albert.run(_build_pdf(tmp_path, "doc2", sizes=sizes))
    third = albert.run(_build_pdf(tmp_path, "doc3", sizes=sizes))

    assert calls_after_first == 1
    assert len(albert.fake.calls) == 1, "les documents suivants sautent le maillon sans appel"
    assert [r.provider for r in (first, second, third)] == ["mistral", "mistral", "mistral"]
    assert len(mistral.calls) == 3
    assert rad._ALBERT_ACCOUNT_DISABLED
    assert warnings_after_first >= 1
    assert _albert_log_records(caplog, marker) == [], "le WARNING de compte n'est émis qu'une fois"


def test_memoized_skip_is_traced_fallback(chain, tmp_path):
    albert = chain(mode="auto")
    sizes = _sizes(2)
    albert.router(sizes)
    albert.fake.inject("*", "*", "invalid_key")
    albert.use_mistral()

    albert.run(_build_pdf(tmp_path, "doc1", sizes=sizes))
    second = albert.run(_build_pdf(tmp_path, "doc2", sizes=sizes))

    assert second.provider == "mistral"
    assert isinstance(second.fallback_from, str) and second.fallback_from.startswith("albert"), (
        "un document traité par Mistral alors qu'Albert est sélectionné doit porter fallback_from"
    )


# ===========================================================================
# /v1/ocr (albert_mistral_ocr)
# ===========================================================================
def test_doc_ocr_pages_zero_based(chain, tmp_path):
    fake = FakeAlbert(ocr_access=True)
    fake.ocr_doc_reply = lambda index: f"Markdown index {index}. " * 30
    albert = chain(fake, mode="auto")

    result = albert.run(_build_pdf(tmp_path, "trois_pages", 3))

    assert result.provider == PROVIDER_DOC
    calls = albert.v1_calls()
    assert calls
    for call in calls:
        pages = call.json.get("pages")
        assert pages == list(range(_document_page_count(call))), pages
        assert call.json["document"]["type"] == "document_url"
        assert call.json["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert any(call.json["pages"] == [0, 1, 2] for call in calls)

    fake.reset_calls()
    capped = albert.run(_build_pdf(tmp_path, "plafond", 3), max_pages=2)
    doc_calls = [call for call in albert.v1_calls() if call.json.get("pages") is not None]
    assert doc_calls
    for call in doc_calls:
        assert min(call.json["pages"]) == 0
        assert max(call.json["pages"]) <= 1, call.json["pages"]
    assert _markers(capped.text) == [1, 2]

    # Gate du lot : la seule liste 1-based reste celle du chemin Mistral historique.
    assert "range(1, max_pages + 1)" not in Path(_ocr_module().__file__).read_text(encoding="utf-8")
    assert Path(rad.__file__).read_text(encoding="utf-8").count("range(1, max_pages + 1)") == 1


def test_doc_ocr_markers_index_plus_one(chain, tmp_path):
    fake = FakeAlbert(ocr_access=True)
    fake.ocr_doc_reply = lambda index: f"CONTENU_INDEX_{index} " + "mot " * 150
    albert = chain(fake, mode="auto")

    whole = albert.run(_build_pdf(tmp_path, "entier", 3))

    assert whole.provider == PROVIDER_DOC
    assert _markers(whole.text) == [1, 2, 3]
    for number in (1, 2, 3):
        block = whole.text.split(f"<!-- Page {number} -->", 1)[1]
        assert block.lstrip().startswith(f"CONTENU_INDEX_{number - 1}")

    split_albert = chain(fake, mode="auto", env={"ALBERT_OCR_PART_PAGES": "1"})
    fake.reset_calls()
    split = split_albert.run(_build_pdf(tmp_path, "decoupe", 3))

    assert split.provider == PROVIDER_DOC
    doc_calls = [call for call in split_albert.v1_calls() if _document_page_count(call) == 1]
    assert len(doc_calls) == 3, "un document de 3 pages découpé en parts d'une page"
    assert _markers(split.text) == [1, 2, 3], "marqueurs continus d'une part à l'autre"
    assert (split.pages_done, split.pages_total) == (3, 3)


@pytest.mark.parametrize("status", [403, 404])
def test_doc_ocr_403_memoized_no_second_probe(status, chain, tmp_path):
    if status == 403:
        fake = FakeAlbert(ocr_access=True)
        fake.inject("POST", OCR_PATH, "no_access")
    else:
        fake = FakeAlbert(ocr_access=False)  # 404 « Model mistral-ocr-2512 not found. » (D3)
    albert = chain(fake, mode="auto")
    sizes = _sizes(2)
    router = albert.router(sizes)
    mistral = albert.use_mistral()

    first = albert.run(_build_pdf(tmp_path, "premier", sizes=sizes))
    v1_after_first = len(albert.v1_calls())
    second = albert.run(_build_pdf(tmp_path, "second", sizes=sizes))

    if status == 403:
        assert v1_after_first == 1
    else:
        # D3 : au plus une tentative (ou la liste /v1/models du preflight) pour tout le processus.
        assert v1_after_first <= 1
    assert len(albert.v1_calls()) == v1_after_first, "pas de seconde sonde /v1/ocr"
    assert rad._ALBERT_V1_OCR_AVAILABLE is False
    assert [first.provider, second.provider] == [PROVIDER_CHAT, PROVIDER_CHAT]
    assert first.fallback_from is None and second.fallback_from is None
    assert sorted(router.requests) == [1, 1, 2, 2]
    assert mistral.calls == []
    assert rad._ALBERT_ACCOUNT_DISABLED is None


def test_413_resplit_then_lightonocr(chain, tmp_path):
    fake = FakeAlbert(ocr_access=True)
    fake.inject("POST", OCR_PATH, 413, 413)
    albert = chain(fake, mode="auto")
    sizes = _sizes(4)
    router = albert.router(sizes)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "trop_gros", sizes=sizes))

    calls = albert.v1_calls()
    assert len(calls) == 2, "un seul redécoupage plus fin après le premier 413"
    assert _document_page_count(calls[1]) < _document_page_count(calls[0])
    assert result.provider == PROVIDER_CHAT
    assert sorted(router.requests) == [1, 2, 3, 4]
    assert mistral.calls == []
    assert rad._ALBERT_V1_OCR_AVAILABLE is not False, "un 413 n'est pas une absence d'accès"
    assert fake.pending_injections() == 0


def test_doc_ocr_split_parts_are_unlinked(chain, tmp_path, monkeypatch):
    fake = FakeAlbert(ocr_access=True)
    fake.ocr_doc_reply = lambda index: "Partie OCR " * 60
    albert = chain(fake, mode="auto", env={"ALBERT_OCR_PART_PAGES": "1"})
    created = []
    real_split = rad._split_pdf_for_ocr

    def split_spy(*args, **kwargs):
        """Note les parts temporaires créées par le découpage."""
        parts = real_split(*args, **kwargs)
        created.extend(parts)
        return parts

    monkeypatch.setattr(rad, "_split_pdf_for_ocr", split_spy)

    result = albert.run(_build_pdf(tmp_path, "parts", 3))

    assert result.provider == PROVIDER_DOC
    assert len(created) >= 3
    assert [path for path in created if os.path.exists(path)] == []


# ===========================================================================
# LightOnOCR (albert_lightonocr)
# ===========================================================================
def test_lightonocr_message_image_only_params(chain, tmp_path):
    albert = chain(mode="chat")
    sizes = _sizes(2)
    albert.router(sizes)

    result = albert.run(_build_pdf(tmp_path, "params", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    calls = albert.chat_calls()
    assert len(calls) == 2
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "stream"}
    for call in calls:
        body = call.json
        assert set(body) <= allowed, sorted(set(body) - allowed)
        assert body["model"] == OCR_CHAT_MODEL
        assert body["max_tokens"] == 4096
        assert body["temperature"] == pytest.approx(0.2)
        assert body["top_p"] == pytest.approx(0.9)
        assert body.get("stream") in (None, False)
        assert len(body["messages"]) == 1
        message = body["messages"][0]
        assert message["role"] == "user"
        assert set(message) == {"role", "content"}
        assert isinstance(message["content"], list) and len(message["content"]) == 1
        part = message["content"][0]
        assert part["type"] == "image_url"
        assert _image_url(call).startswith("data:image/png;base64,")
        _png_size(_image_url(call))
        assert call.headers.get("authorization") == "Bearer " + FAKE_ALBERT_KEY
        assert FAKE_ALBERT_KEY not in call.url


def test_alias_config_sends_pinned_id(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_OCR_CHAT_MODEL": OCR_CHAT_ALIAS})
    sizes = _sizes(1)
    albert.router(sizes)

    result = albert.run(_build_pdf(tmp_path, "alias", sizes=sizes))
    listings_after_first = len(albert.fake.calls_to("GET", "/v1/models"))
    albert.run(_build_pdf(tmp_path, "alias_bis", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert [call.json["model"] for call in albert.chat_calls()] == [OCR_CHAT_MODEL, OCR_CHAT_MODEL]
    assert listings_after_first == 1
    assert len(albert.fake.calls_to("GET", "/v1/models")) == 1, "alias résolu une fois par processus"


def test_pinned_ids_need_no_model_listing(chain, tmp_path):
    albert = chain(mode="auto")
    sizes = _sizes(1)
    albert.router(sizes)

    albert.run(_build_pdf(tmp_path, "epingle", sizes=sizes))

    assert albert.fake.calls_to("GET", "/v1/models") == []
    assert {call.json["model"] for call in albert.chat_calls()} == {OCR_CHAT_MODEL}
    assert {call.json["model"] for call in albert.v1_calls()} <= {OCR_DOC_MODEL}


def test_rasterization_a4_max_side_le_1541(chain, tmp_path):
    ocr = _ocr_module()
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # A4 portrait
    doc.new_page(width=842, height=595)  # A4 paysage
    doc.new_page(width=200, height=280)  # petite page : 200 dpi atteints
    try:
        sizes = [_png_size(ocr.render_page_png_b64(page, 200, 1540)) for page in doc]
    finally:
        doc.close()
    for width, height in sizes[:2]:
        assert max(width, height) <= 1541
        assert max(width, height) >= 1535
    assert sizes[0][0] < sizes[0][1] and sizes[1][0] > sizes[1][1]
    expected_small = _expected_size(200, 280)
    assert abs(sizes[2][0] - expected_small[0]) <= 2 and abs(sizes[2][1] - expected_small[1]) <= 2

    albert = chain(mode="chat")
    a4 = [(595, 842)]
    albert.router(a4)
    result = albert.run(_build_pdf(tmp_path, "a4", sizes=a4))
    assert result.provider == PROVIDER_CHAT
    (call,) = albert.chat_calls()
    width, height = _png_size(_image_url(call))
    assert max(width, height) <= 1541


def test_page_markers_continuous_with_failed_page(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_MAX_RETRIES": "0"})
    albert.monkeypatch.setattr(rad, "OCR_MIN_CHARS_PER_PAGE", 1)
    sizes = _sizes(4)
    router = albert.router(sizes)
    router.plan(2, 500)
    router.plan(3, "empty")

    result = albert.run(_build_pdf(tmp_path, "trou", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert _markers(result.text) == [1, 2, 3, 4]
    after_two = result.text.split("<!-- Page 2 -->", 1)[1].split("<!-- Page 3 -->", 1)[0]
    failures = FAILED_PAGE_RE.findall(after_two)
    assert len(failures) == 1, after_two
    assert PAGE_MARKER_RE.findall(failures[0]) == []
    assert FAILED_PAGE_RE.findall(result.text) == failures
    after_three = result.text.split("<!-- Page 3 -->", 1)[1].split("<!-- Page 4 -->", 1)[0]
    assert after_three.strip() == ""
    assert router.text(1) in result.text and router.text(4) in result.text
    assert result.partial is True
    assert (result.pages_done, result.pages_total) == (3, 4)
    assert result.fallback_from is None


@pytest.mark.parametrize("action", ["length", "length_empty"])
def test_finish_length_marks_partial(action, chain, tmp_path):
    albert = chain(mode="chat")
    albert.monkeypatch.setattr(rad, "OCR_MIN_CHARS_PER_PAGE", 1)
    sizes = _sizes(3)
    router = albert.router(sizes)
    router.plan(1, action)

    result = albert.run(_build_pdf(tmp_path, f"tronque_{action}", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert result.partial is True
    assert _markers(result.text) == [1, 2, 3]
    assert result.pages_total == 3
    if action == "length":
        assert router.text(1) in result.text
    else:
        assert len(FAILED_PAGE_RE.findall(result.text)) == 1
    assert sorted(router.requests) == [1, 2, 3], "une page tronquée n'est pas renvoyée"


def test_failed_ratio_raises_chain_continues(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_MAX_RETRIES": "0"})
    sizes = _sizes(3)
    router = albert.router(sizes)
    router.plan(1, 500)
    router.plan(2, 500)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "ratio", sizes=sizes))

    assert result.provider == "mistral"
    assert isinstance(result.fallback_from, str) and result.fallback_from.startswith("albert")
    assert len(mistral.calls) == 1


def test_failed_ratio_below_threshold_keeps_partial_albert(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_MAX_RETRIES": "0", "ALBERT_OCR_MAX_FAILED_RATIO": "0.7"})
    albert.monkeypatch.setattr(rad, "OCR_MIN_CHARS_PER_PAGE", 1)
    sizes = _sizes(3)
    router = albert.router(sizes)
    router.plan(1, 500)
    router.plan(2, 500)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "ratio_ok", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert result.partial is True
    assert (result.pages_done, result.pages_total) == (1, 3)
    assert mistral.calls == []


def test_404_single_request(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_OCR_CONCURRENCY": "1"})
    sizes = _sizes(3)
    router = albert.router(sizes)
    for page in (1, 2, 3):
        router.plan(page, 404)
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "modele_retire", sizes=sizes))

    assert len(albert.chat_calls()) == 1, "un 404 (modèle retiré) n'est ni retenté ni répété page par page"
    assert albert.sleeps == []
    assert result.provider == "mistral"
    assert isinstance(result.fallback_from, str) and result.fallback_from.startswith("albert")
    assert len(mistral.calls) == 1


def test_semaphore_released_during_backoff(chain, tmp_path):
    albert = chain(mode="chat")
    semaphore = threading.BoundedSemaphore(1)
    albert.monkeypatch.setattr(rad, "ALBERT_OCR_SEMAPHORE", semaphore)
    sizes = _sizes(1)
    router = albert.router(sizes)
    router.plan(1, 502, 500)
    held_at_send = []
    held_at_sleep = []
    router.observer = lambda page: held_at_send.append(_held(semaphore))
    albert.sleep_observer = lambda seconds: held_at_sleep.append(_held(semaphore))

    result = albert.run(_build_pdf(tmp_path, "backoff", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert router.requests == [1, 1, 1]
    assert held_at_send == [True, True, True], "le sémaphore OCR Albert est tenu pendant l'envoi"
    assert len(held_at_sleep) >= 2
    assert not any(held_at_sleep), "le sémaphore est libéré pendant le backoff"
    assert _held(semaphore) is False


def test_density_only_lightonocr(chain, tmp_path):
    albert = chain(mode="chat")
    sizes = _sizes(2)
    router = albert.router(sizes)
    router.texts = {1: "court", 2: "bref"}

    sparse = albert.run(_build_pdf(tmp_path, "clairseme", sizes=sizes))

    assert sparse.provider == PROVIDER_CHAT
    assert sparse.partial is True
    assert "albert_lightonocr" in (sparse.error or "")
    assert "car./page" in (sparse.error or "")

    fake = FakeAlbert(ocr_access=True)
    fake.ocr_doc_reply = lambda index: "court"
    doc_albert = chain(fake, mode="auto")
    dense_check = doc_albert.run(_build_pdf(tmp_path, "clairseme_doc", 2))

    assert dense_check.provider == PROVIDER_DOC
    assert dense_check.partial is False
    assert dense_check.error is None


def test_partial_creates_ocr_partial_entry(chain, tmp_path, monkeypatch):
    albert = chain(mode="chat", env={"ALBERT_MAX_RETRIES": "0"})
    monkeypatch.setattr(rad, "OCR_MIN_CHARS_PER_PAGE", 1)
    monkeypatch.setattr(rad, "time", _NoSleepTime())
    sizes = _sizes(3)
    router = albert.router(sizes)
    router.plan(2, 500)
    _build_pdf(tmp_path / "files", "partiel", sizes=sizes)

    outcome = rad._process_single_zotero_item(_zotero_item("ITEMPAR1", "files/partiel.pdf"), str(tmp_path))

    (record,) = outcome.records
    assert record["texteocr_provider"] == PROVIDER_CHAT
    assert record["texteocr_partial"] is True
    assert (record["texteocr_pages_done"], record["texteocr_pages_total"]) == (2, 3)
    partial = [entry for entry in outcome.errors if entry.get("error_type") == "OCR_PARTIAL"]
    assert len(partial) == 1
    assert partial[0]["provider"] == PROVIDER_CHAT
    assert (partial[0]["pages_done"], partial[0]["pages_total"]) == (2, 3)
    assert not [entry for entry in outcome.errors if entry.get("error_type") == "OCR_PROVIDER_FALLBACK"]


def test_max_pages_cap_marks_partial(chain, tmp_path):
    albert = chain(mode="chat", env={"ALBERT_OCR_MAX_PAGES": "2"})
    albert.monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", lambda *a, **k: "Page 1")
    sizes = _sizes(3)
    router = albert.router(sizes)

    result = albert.run(_build_pdf(tmp_path, "plafonne", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert sorted(router.requests) == [1, 2]
    assert result.partial is True
    assert (result.pages_done, result.pages_total) == (2, 3)
    assert "ALBERT_OCR_MAX_PAGES" in (result.error or "")
    assert _markers(result.text) == [1, 2]


# ===========================================================================
# LightOnOCR à plusieurs fils (ALBERT_OCR_CONCURRENCY > 1)
# ===========================================================================
OCR_WORKERS = 3
MULTI_PAGES = 10  # au-delà, deux pages de _sizes() atteignent le plafond de 1 540 px (même largeur)
HOLD_TIMEOUT = 5.0  # garde-fou : une page retenue n'attend jamais indéfiniment
POOL_THREAD_PREFIX = "albert-ocr"


class _PoolProbe:
    """Observateur du routeur LightOnOCR à plusieurs fils : simultanéité, ordre de fin, retenues.

    Appelé par ``PageRouter._route`` dans le fil de l'envoi, avant la réponse :
    ``peak`` est le nombre maximal d'envois simultanés, ``finished`` l'ordre
    dans lequel le faux a servi les pages (réessais compris).
    """

    def __init__(self, delays=None, *, waits=None, pause=0.0):
        """Prépare l'observateur.

        Args:
            delays: retard (secondes) par page, appliqué pendant l'envoi.
            waits: ``{page: page_attendue}`` : la page n'est servie qu'après la
                page attendue (``HOLD_TIMEOUT`` au plus).
            pause: attente supplémentaire (secondes) d'une page retenue, une
                fois la page attendue servie.
        """
        self.delays = dict(delays or {})
        self.waits = dict(waits or {})
        self.pause = float(pause)
        self.current = 0
        self.peak = 0
        self.finished = []
        self._lock = threading.Lock()
        self._served = {}

    def _event(self, page):
        """Événement « page servie » de ``page`` (créé à la demande)."""
        with self._lock:
            return self._served.setdefault(page, threading.Event())

    def __call__(self, page):
        """Compte l'envoi, applique retenue et retard, puis note la page comme servie."""
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        try:
            if page in self.waits:
                self._event(self.waits[page]).wait(timeout=HOLD_TIMEOUT)
                if self.pause:
                    time.sleep(self.pause)
            delay = self.delays.get(page, 0.0)
            if delay:
                time.sleep(delay)
        finally:
            with self._lock:
                self.current -= 1
                self.finished.append(page)
            self._event(page).set()


def _seeded_delays(pages, seed=20260926, high=0.02):
    """Retards pseudo-aléatoires reproductibles (0 à ``high`` secondes) par page."""
    rng = random.Random(seed)
    return {page: rng.uniform(0.0, high) for page in range(1, pages + 1)}


def _multi_worker_chain(chain, workers=OCR_WORKERS, *, env=None, **kwargs):
    """Chaîne Albert à ``workers`` fils ; renvoie ``(chaîne, sémaphore)``.

    ``ALBERT_OCR_CONCURRENCY`` fixe la taille du pool (``cfg.ocr_concurrency``) ;
    le sémaphore du processus est remplacé par celui que l'import aurait créé
    avec cette variable (capacité ``workers``).
    """
    settings = {"ALBERT_OCR_CONCURRENCY": str(workers)}
    settings.update(env or {})
    albert = chain(env=settings, **kwargs)
    assert albert.cfg.ocr_concurrency == workers
    semaphore = threading.BoundedSemaphore(workers)
    albert.monkeypatch.setattr(rad, "ALBERT_OCR_SEMAPHORE", semaphore)
    return albert, semaphore


def _all_slots_free(semaphore, capacity):
    """Vrai si les ``capacity`` places du sémaphore sont libres (aucun envoi ne le tient)."""
    taken = 0
    try:
        while taken < capacity and semaphore.acquire(blocking=False):
            taken += 1
        return taken == capacity
    finally:
        for _ in range(taken):
            semaphore.release()


def _live_pool_threads():
    """Noms des fils du pool LightOnOCR encore vivants (vide après ``shutdown``)."""
    return [
        thread.name for thread in threading.enumerate()
        if thread.name.startswith(POOL_THREAD_PREFIX) and thread.is_alive()
    ]


def _page_segments(text):
    """``{N: contenu}`` entre chaque marqueur ``<!-- Page N -->`` et le suivant (blancs retirés)."""
    parts = PAGE_MARKER_RE.split(text or "")
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def test_lightonocr_multi_worker_markers_in_order(chain, tmp_path):
    albert, semaphore = _multi_worker_chain(chain, mode="chat")
    sizes = _sizes(MULTI_PAGES)
    router = albert.router(sizes)
    # Retards aléatoires (graine fixe) ; la page 1 n'est servie qu'après la page 3 :
    # les fins sont désordonnées à coup sûr.
    probe = _PoolProbe(_seeded_delays(MULTI_PAGES), waits={1: 3})
    router.observer = probe
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, "multi_fils", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert result.fallback_from is None
    assert result.partial is False
    assert (result.pages_done, result.pages_total) == (MULTI_PAGES, MULTI_PAGES)
    assert mistral.calls == []
    assert sorted(router.requests) == list(range(1, MULTI_PAGES + 1)), "chaque page envoyée une fois"
    assert probe.finished.index(3) < probe.finished.index(1), "fins désordonnées attendues"
    assert probe.finished != sorted(probe.finished)
    assert 2 <= probe.peak <= OCR_WORKERS, probe.peak
    assert _markers(result.text) == list(range(1, MULTI_PAGES + 1))
    segments = _page_segments(result.text)
    for number in range(1, MULTI_PAGES + 1):
        assert segments[number] == router.text(number), f"texte de la page {number} hors de son marqueur"
    assert FAILED_PAGE_RE.findall(result.text) == []
    assert _all_slots_free(semaphore, OCR_WORKERS)
    assert _live_pool_threads() == []


def test_lightonocr_render_ahead_window_bounded(tmp_path):
    ocr = _ocr_module()
    window = 2 * OCR_WORKERS
    fake = FakeAlbert()
    sizes = _sizes(MULTI_PAGES)
    router = PageRouter(fake, sizes)
    lock = threading.Lock()
    gate = threading.Event()
    served = []
    renders = []
    state = {"current": 0, "peak": 0}

    def observer(page):
        """Retient les envois jusqu'à l'ouverture de la barrière, puis note la page servie."""
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        gate.wait(timeout=HOLD_TIMEOUT)
        with lock:
            state["current"] -= 1
            served.append(page)

    def render_spy(page, dpi, max_side):
        """Note chaque rendu (index, pages déjà servies) ; ouvre la barrière peu après la fenêtre pleine."""
        with lock:
            renders.append((page.number, len(served)))
        if page.number == window - 1:
            timer = threading.Timer(0.15, gate.set)
            timer.daemon = True
            timer.start()
        return ocr.render_page_png_b64(page, dpi, max_side)

    router.observer = observer
    cfg = AlbertConfig.from_env({**ALBERT_TEST_ENV, "ALBERT_OCR_CONCURRENCY": str(OCR_WORKERS)})
    semaphore = threading.BoundedSemaphore(OCR_WORKERS)
    client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport, sleep=lambda seconds: None)
    try:
        outcome = ocr.ocr_pdf_lightonocr(
            _build_pdf(tmp_path, "fenetre", sizes=sizes), client,
            cfg=cfg, semaphore=semaphore, render_fn=render_spy,
        )
    finally:
        gate.set()
        client.close()

    assert [index for index, _served in renders] == list(range(MULTI_PAGES)), "rendu séquentiel, dans l'ordre"
    # Barrière fermée : exactement 2 × fils pages rendues d'avance, aucune servie.
    assert [count for _index, count in renders[:window]] == [0] * window
    # Au-delà, une page n'est rendue que si une page en vol a été rangée (fenêtre bornée).
    for index, count in renders[window:]:
        assert count >= index + 1 - window, (index, count, renders)
    assert state["peak"] <= OCR_WORKERS
    assert outcome.provider == PROVIDER_CHAT
    assert outcome.partial is False
    assert _markers(outcome.text) == list(range(1, MULTI_PAGES + 1))
    assert sorted(served) == list(range(1, MULTI_PAGES + 1))
    assert _all_slots_free(semaphore, OCR_WORKERS)
    assert _live_pool_threads() == []


@pytest.mark.parametrize("scope", ["first", "every"])
def test_lightonocr_multi_worker_account_error_stops_pool(scope, chain, tmp_path):
    albert, semaphore = _multi_worker_chain(chain, mode="chat")
    sizes = _sizes(MULTI_PAGES)
    router = albert.router(sizes)
    if scope == "first":
        # 401 sur la page 1 seule ; les autres envois ne répondent qu'une fois l'arrêt posé.
        router.plan(1, 401)
        router.observer = _PoolProbe(waits={page: 1 for page in range(2, MULTI_PAGES + 1)}, pause=0.2)
    else:
        for page in range(1, MULTI_PAGES + 1):
            router.plan(page, 401)
    mistral = albert.use_mistral()

    first = albert.run(_build_pdf(tmp_path, f"compte_{scope}", sizes=sizes))
    calls_after_first = len(albert.fake.calls)

    requested = set(router.requests)
    assert 1 <= len(albert.chat_calls()) <= OCR_WORKERS, router.requests
    assert requested <= set(range(1, OCR_WORKERS + 1)), "aucune page en file n'est envoyée après l'arrêt"
    if scope == "first":
        assert 1 in requested
    assert albert.sleeps == [], "une erreur de compte n'est pas retentée"
    assert first.provider == "mistral"
    assert isinstance(first.fallback_from, str) and first.fallback_from.startswith("albert")
    assert rad._ALBERT_ACCOUNT_DISABLED
    assert FAKE_ALBERT_KEY not in str(rad._ALBERT_ACCOUNT_DISABLED)
    assert _all_slots_free(semaphore, OCR_WORKERS)
    assert _live_pool_threads() == []

    second = albert.run(_build_pdf(tmp_path, f"compte_{scope}_suivant", sizes=sizes))

    assert len(albert.fake.calls) == calls_after_first, "document suivant : aucun appel Albert"
    assert second.provider == "mistral"
    assert isinstance(second.fallback_from, str) and second.fallback_from.startswith("albert")
    assert len(mistral.calls) == 2


@pytest.mark.parametrize("outcome", ["retried", "exhausted"])
def test_lightonocr_multi_worker_transient_page_markers_continuous(outcome, chain, tmp_path):
    env = {"ALBERT_MAX_RETRIES": "0"} if outcome == "exhausted" else None
    albert, semaphore = _multi_worker_chain(chain, mode="chat", env=env)
    sizes = _sizes(MULTI_PAGES)
    router = albert.router(sizes)
    failing = 5
    router.plan(failing, 502 if outcome == "retried" else 500)
    probe = _PoolProbe(_seeded_delays(MULTI_PAGES, seed=7), waits={1: 3})
    router.observer = probe
    mistral = albert.use_mistral()

    result = albert.run(_build_pdf(tmp_path, f"transitoire_{outcome}", sizes=sizes))

    assert result.provider == PROVIDER_CHAT
    assert result.fallback_from is None
    assert mistral.calls == []
    assert probe.finished.index(3) < probe.finished.index(1), "fins désordonnées attendues"
    assert _markers(result.text) == list(range(1, MULTI_PAGES + 1)), "marqueurs continus, dans l'ordre"
    segments = _page_segments(result.text)
    for number in range(1, MULTI_PAGES + 1):
        if number != failing:
            assert segments[number] == router.text(number)
    if outcome == "retried":
        assert router.requests.count(failing) == 2, "page transitoire réessayée par le client"
        assert len(router.requests) == MULTI_PAGES + 1
        assert albert.sleeps, "backoff du réessai"
        assert segments[failing] == router.text(failing)
        assert FAILED_PAGE_RE.findall(result.text) == []
        assert result.partial is False
        assert (result.pages_done, result.pages_total) == (MULTI_PAGES, MULTI_PAGES)
    else:
        assert router.requests.count(failing) == 1
        failures = FAILED_PAGE_RE.findall(segments[failing])
        assert len(failures) == 1, segments[failing]
        assert FAILED_PAGE_RE.findall(result.text) == failures
        assert result.partial is True
        assert (result.pages_done, result.pages_total) == (MULTI_PAGES - 1, MULTI_PAGES)
    assert _all_slots_free(semaphore, OCR_WORKERS)
    assert _live_pool_threads() == []
