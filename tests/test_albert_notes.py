"""Tests du chat Albert pour les notes Zotero et les fiches de livre (lot 3, tâches 6 et 7).

* Chemin OFF : le golden G11 (kwargs des appels LLM web) est rejoué en lecture
  seule avec une clé Albert factice présente, Albert désactivé puis activé pour
  des modèles sans préfixe ``albert/`` (invariants 1 et 4).
* ``run_llm_slot`` sans politique Albert : strictement l'appel historique
  (sémaphore global puis ``run_in_executor``).
* Chemin Albert (``tests/albert_fakes.FakeAlbert`` branché sur tout
  ``AlbertClient`` construit pendant le test) : budget et marge de raisonnement
  de gpt-oss, réessai unique d'une réponse vide tronquée, aucun repli vers
  gpt-4o-mini (invariant 19), jamais de clé lue dans l'environnement, erreurs
  de compte propagées (invariant 38, fiches de livre comprises), sommeil hors
  sémaphores (invariant 17), une seule couche de retry (invariant 37), clé
  transmise aux trois phases des fiches et rôle ``long_context`` pour les
  invites trop longues.

Les coroutines sont exécutées avec ``asyncio.run`` ; aucun test n'attend
réellement (sommeils enregistrés ou ramenés à zéro).
"""

from __future__ import annotations

import asyncio
import difflib
import importlib
import inspect
import os
import sys
import threading
from datetime import date
from types import SimpleNamespace

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests import test_albert_off_golden as golden  # noqa: E402

from app.utils import book_note_generator as bng  # noqa: E402
from scripts.rad_albert import catalog as albert_catalog  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402
from scripts.rad_albert.config import AlbertConfig  # noqa: E402
from scripts.rad_albert.errors import (  # noqa: E402
    AlbertAuthError,
    AlbertError,
    AlbertQuotaExhausted,
    AlbertTruncatedError,
)
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert  # noqa: E402

lng = golden.lng

GOLDEN_SCRIPTS_DIR = os.path.join(_THIS, "fixtures", "albert", "golden_off", "scripts")
CHAT_PATH = "/v1/chat/completions"
REASONING_MODEL = "gpt-oss-120b"
SMALL_MODEL = "ministral-3-8b-instruct-2512"
FAKE_OPENAI_KEY = golden.FAKE_ENV["OPENAI_API_KEY"]
FAKE_OPENROUTER_KEY = golden.FAKE_ENV["OPENROUTER_API_KEY"]
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
METADATA = {
    "title": "Le langage ordinaire",
    "authors": "Dupont, Jeanne",
    "date": "2021",
    "abstract": "Résumé de l'article sur le langage ordinaire.",
    "language": "fr",
}
TEXT = "Texte intégral de l'article. " * 40
NOTE_HTML = "<h2>Fiche</h2><p>Analyse de l'article sur le langage ordinaire.</p>"


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
class SleepRecorder:
    """Sommeil factice (synchrone) : enregistre les durées demandées et l'état observé."""

    def __init__(self, probe=None):
        """Crée l'enregistreur ; ``probe()`` est appelé à chaque sommeil (état des sémaphores)."""
        self.calls = []
        self.states = []
        self._probe = probe
        self._lock = threading.Lock()

    def __call__(self, seconds):
        """Enregistre la durée et, si demandé, l'état au moment du sommeil."""
        with self._lock:
            self.calls.append(float(seconds))
            if self._probe is not None:
                self.states.append(self._probe())


def _client_classes():
    """Classes ``AlbertClient`` (import ``scripts.rad_albert`` et, s'il est chargé, import CLI ``rad_albert``).

    Le module client est importé ici : les implémentations le chargent
    paresseusement, après l'installation du double.
    """
    importlib.import_module("scripts.rad_albert.client")
    classes = []
    for name in ("scripts.rad_albert.client", "rad_albert.client"):
        module = sys.modules.get(name)
        cls = getattr(module, "AlbertClient", None) if module is not None else None
        if cls is not None and cls not in classes:
            classes.append(cls)
    return classes


def route_albert_to_fake(monkeypatch, fake, sleep):
    """Branche tout ``AlbertClient`` construit pendant le test sur ``fake`` (transport et sommeil)."""
    for cls in _client_classes():
        original = cls.__init__

        def patched(self, *args, __original=original, **kwargs):
            """Constructeur d'origine avec le transport du faux et le sommeil enregistreur."""
            if kwargs.get("transport") is None:
                kwargs["transport"] = fake.transport
            kwargs.setdefault("sleep", sleep)
            __original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", patched)


def _assert_matches_golden_json(name, obj):
    """Compare le rendu JSON de ``obj`` au golden ``name`` sans jamais le réécrire."""
    with open(os.path.join(GOLDEN_SCRIPTS_DIR, name), "rb") as fh:
        expected = fh.read()
    data = golden._dump(obj).encode("utf-8")
    if expected != data:
        diff = difflib.unified_diff(
            expected.decode("utf-8", "replace").splitlines(True),
            data.decode("utf-8", "replace").splitlines(True),
            fromfile=f"golden/{name}", tofile="actuel", n=2,
        )
        pytest.fail("Écart avec le golden " + name + " :\n" + "".join(list(diff)[:200]), pytrace=False)


def _chat_calls(fake):
    """Appels de chat reçus par le faux."""
    return fake.calls_to("POST", CHAT_PATH)


def _module_semaphores():
    """Sémaphores asyncio tenus au niveau du module des notes (global et Albert)."""
    return {name: value for name, value in vars(lng).items() if isinstance(value, asyncio.Semaphore)}


def _free_semaphores():
    """Noms des sémaphores du module actuellement libres (aucune place tenue)."""
    capacities = {"_llm_semaphore": 1}
    capacities.update({name: AlbertConfig.from_env().notes_concurrency
                       for name in _module_semaphores() if name != "_llm_semaphore"})
    return {name: sem._value >= capacities.get(name, 1) for name, sem in _module_semaphores().items()}


def _chat_ok(content, model=SMALL_MODEL, finish="stop"):
    """Réponse 200 de chat minimale (forme de la fixture P7), injectable dans ``FakeAlbert``."""
    return (200, {
        "id": "chatcmpl-fake-injected",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.0},
    })


def _book_text(chapters=3):
    """Texte de livre OCR avec marqueurs de page et ``chapters`` titres de chapitre markdown."""
    parts = []
    page = 1
    for num in range(1, chapters + 1):
        parts.append(f"<!-- Page {page} -->")
        parts.append(f"# Chapitre {num}. Titre du chapitre {num}")
        body = " ".join(
            f"Paragraphe {i} du chapitre {num} : l'analyse du langage ordinaire progresse."
            for i in range(1, 25)
        )
        parts.append(body)
        page += 1
        parts.append(f"<!-- Page {page} -->")
        parts.append(body)
        page += 1
    return "\n\n".join(parts)


BOOK_METADATA = {
    "title": "Le langage ordinaire, un livre",
    "authors": "Dupont, Jeanne",
    "date": "2021",
    "language": "fr",
    "itemType": "book",
}


def _book_reply(body):
    """Réponse de chat générique pour les trois phases d'une fiche de livre."""
    user = body["messages"][-1]["content"]
    if "===SECTION_A===" in user:
        return ("===SECTION_A===\n<h3>1. Identification</h3><p>Livre.</p>\n"
                "===SECTION_B===\n<h3>4. Synthèse</h3><p>Synthèse.</p>\n===END===")
    if "SUMMARY:" in user:
        return "<h3>Chapitre</h3><p>Analyse du chapitre.</p>\nSUMMARY: résumé du chapitre."
    return '{"book_type": "monograph", "chapters": []}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Retire identifiants et réglages ; sémaphores du module, limiteurs et preflight remis à zéro."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in list(_module_semaphores()):
        monkeypatch.setattr(lng, name, None)
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def openai_double(monkeypatch):
    """``_get_llm_clients`` remplacé par des clients OpenAI/OpenRouter enregistreurs."""
    calls, key_log = [], []
    monkeypatch.setattr(lng, "_get_llm_clients", golden._llm_clients_double(calls, key_log))
    return SimpleNamespace(calls=calls, key_log=key_log)


def _albert_env(monkeypatch, enabled):
    """Albert ON ou OFF, débits élevés ; la clé n'est jamais posée dans l'environnement ici."""
    monkeypatch.setenv("ALBERT_ENABLED", "1" if enabled else "0")
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def albert_on(monkeypatch, openai_double):
    """Albert ON ; ``FakeAlbert`` répond ``NOTE_HTML`` ; sommeils synchrones enregistrés."""
    fake = FakeAlbert()
    fake.chat_reply = NOTE_HTML
    sleeps = SleepRecorder(probe=_free_semaphores)
    route_albert_to_fake(monkeypatch, fake, sleeps)
    _albert_env(monkeypatch, enabled=True)
    return SimpleNamespace(fake=fake, sleeps=sleeps, openai=openai_double)


@pytest.fixture
def fast_async_sleep(monkeypatch):
    """``asyncio.sleep`` enregistré (durée, état des sémaphores du module) et ramené à zéro."""
    original = asyncio.sleep
    records = []

    async def spy(delay, *args, **kwargs):
        """Enregistre les attentes non nulles puis cède la main sans attendre."""
        if delay and delay > 0:
            records.append((float(delay), _free_semaphores()))
        return await original(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return records


# ---------------------------------------------------------------------------
# Chemin OFF et fournisseurs historiques
# ---------------------------------------------------------------------------
def _g11_entries(monkeypatch, extra_kwargs):
    """Rejoue le scénario G11 (notes puis filtre de citations) ; ``extra_kwargs`` ajoutés à chaque appel."""
    prompt = "PROMPT GOLDEN G11 : produire une fiche de lecture sur « Le langage ordinaire »."
    note_cases = [
        ("short_openai", "short", "gpt-4o-mini", True),
        ("extended_openrouter", "extended", "google/gemini-2.5-flash", True),
        ("extended_default_model", "extended", None, True),
        ("short_openrouter_missing_fallback", "short", "google/gemini-2.5-flash", False),
    ]
    filter_cases = [
        ("filter_openai", "gpt-4o-mini", True),
        ("filter_openrouter", "google/gemini-2.5-flash", True),
        ("filter_openrouter_missing", "google/gemini-2.5-flash", False),
    ]
    out = []
    for case, mode, model, with_or in note_cases:
        calls, key_log = [], []
        monkeypatch.setattr(lng, "_get_llm_clients", golden._llm_clients_double(calls, key_log, with_or))
        entry = {"function": "_generate_with_llm", "case": case, "mode": mode, "model": model}
        try:
            entry["returned"] = lng._generate_with_llm(
                prompt, model=model, mode=mode,
                openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY, **extra_kwargs,
            )
            entry["error"] = None
        except Exception as exc:  # noqa: BLE001 — l'échec fait partie du golden
            entry["returned"] = None
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
        entry["credentials_passed"] = key_log
        entry["calls"] = calls
        out.append(entry)
    for case, model, with_or in filter_cases:
        calls, key_log = [], []
        monkeypatch.setattr(golden.cfilter, "_get_llm_clients", golden._llm_clients_double(calls, key_log, with_or))
        entry = {"function": "_call_llm_api", "case": case, "model": model}
        try:
            entry["returned"] = golden.cfilter._call_llm_api(
                prompt, model=model,
                openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY, **extra_kwargs,
            )
            entry["error"] = None
        except Exception as exc:  # noqa: BLE001 — l'échec fait partie du golden
            entry["returned"] = None
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
        entry["credentials_passed"] = key_log
        entry["calls"] = calls
        out.append(entry)
    return out


@pytest.mark.parametrize("variant", ["off_env_key", "off_kwarg_key", "on_kwarg_key"])
def test_off_generate_kwargs_unchanged(variant, monkeypatch):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=variant.startswith("on"))
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    extra = {"albert_api_key": FAKE_ALBERT_KEY} if variant.endswith("kwarg_key") else {}
    _assert_matches_golden_json("g11_llm_call_kwargs.json", _g11_entries(monkeypatch, extra))
    assert fake.calls == []


def test_albert_api_key_is_last_kwarg_default_none():
    for func in (lng.build_note_html, lng.build_abstract_text, lng.build_note_html_async,
                 lng.build_abstract_text_async):
        params = list(inspect.signature(func).parameters.values())
        assert params[-2].name == "albert_api_key", func.__name__
        assert params[-2].default is None, func.__name__
        assert params[-1].name == "albert_usage_ledger", func.__name__
        assert params[-1].kind is inspect.Parameter.KEYWORD_ONLY, func.__name__
        assert params[-1].default is None, func.__name__
    params = inspect.signature(lng._generate_with_llm).parameters
    assert params["albert_api_key"].default is None


def test_run_llm_slot_without_policy_equals_executor_call():
    run_llm_slot = lng.run_llm_slot
    params = inspect.signature(run_llm_slot).parameters
    assert list(params)[:2] == ["semaphore", "thunk"]
    assert params["albert_policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["albert_policy"].default is None
    observed = []

    class Boom(Exception):
        """Erreur propagée telle quelle par le créneau."""

    boom = Boom("échec du thunk")

    async def scenario():
        """Un appel réussi puis un appel en échec, sémaphore de taille 1."""
        semaphore = asyncio.Semaphore(1)
        loop_thread = threading.get_ident()

        def thunk():
            """Note l'état du sémaphore et le thread d'exécution."""
            observed.append({"held": semaphore._value == 0, "in_worker": threading.get_ident() != loop_thread})
            return "résultat"

        def failing():
            """Lève l'erreur prévue, sémaphore tenu."""
            observed.append({"held": semaphore._value == 0, "in_worker": threading.get_ident() != loop_thread})
            raise boom

        value = await run_llm_slot(semaphore, thunk)
        with pytest.raises(Boom) as excinfo:
            await run_llm_slot(semaphore, failing)
        return value, excinfo.value, semaphore._value

    value, raised, final = asyncio.run(scenario())
    assert value == "résultat"
    assert raised is boom
    assert final == 1
    assert observed == [{"held": True, "in_worker": True}, {"held": True, "in_worker": True}]


# ---------------------------------------------------------------------------
# Chemin Albert : budgets, troncature, souveraineté, clés
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["extended", "pedagogique", "evaluation", "short"])
def test_gptoss_budget_headroom_effort(mode, albert_on):
    fake = albert_on.fake
    headroom = AlbertConfig.from_env().reasoning_headroom
    result = lng._generate_with_llm("Invite de test.", model="albert/" + REASONING_MODEL, mode=mode,
                                    albert_api_key=FAKE_ALBERT_KEY)
    assert result == NOTE_HTML
    body = _chat_calls(fake)[-1].json
    assert body["model"] == REASONING_MODEL
    base = lng.NOTE_MODE_MAX_TOKENS[mode]
    if mode == "short":
        assert body["max_tokens"] >= base + headroom
    else:
        assert body["max_tokens"] == base + headroom
    assert body["reasoning_effort"] == AlbertConfig.from_env().reasoning_effort
    # Modèle sans raisonnement : pas de marge ni d'effort.
    lng._generate_with_llm("Invite de test.", model="albert/" + SMALL_MODEL, mode="extended",
                           albert_api_key=FAKE_ALBERT_KEY)
    body = _chat_calls(fake)[-1].json
    assert body["model"] == SMALL_MODEL
    assert body["max_tokens"] == lng.NOTE_MODE_MAX_TOKENS["extended"]
    assert "reasoning_effort" not in body
    assert albert_on.openai.calls == []


def test_empty_length_retry_then_error(albert_on):
    fake = albert_on.fake
    trap = {"content": None, "finish_reason": "length", "reasoning": "raisonnement tronqué"}
    fake.queue_chat_replies(trap, trap)
    with pytest.raises(AlbertTruncatedError):
        lng._generate_with_llm("Invite.", model="albert/" + REASONING_MODEL, mode="extended",
                               albert_api_key=FAKE_ALBERT_KEY)
    bodies = [call.json for call in _chat_calls(fake)]
    assert len(bodies) == 2
    assert bodies[1]["max_tokens"] > bodies[0]["max_tokens"]
    assert bodies[1]["reasoning_effort"] == "low"
    # Une réponse valide au réessai est renvoyée telle quelle.
    fake.reset_calls()
    fake.queue_chat_replies(trap, "Contenu valide au réessai.")
    result = lng._generate_with_llm("Invite.", model="albert/" + REASONING_MODEL, mode="extended",
                                    albert_api_key=FAKE_ALBERT_KEY)
    assert result == "Contenu valide au réessai."
    assert len(_chat_calls(fake)) == 2
    assert albert_on.openai.calls == []


@pytest.mark.parametrize("failure", ["server_error", "not_found", "model_busy"])
def test_final_failure_raises_no_gpt4o_mini(failure, albert_on):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *([failure] * 60))
    with pytest.raises(AlbertError):
        lng._generate_with_llm("Invite.", model="albert/" + SMALL_MODEL, mode="short",
                               openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
                               albert_api_key=FAKE_ALBERT_KEY)
    assert albert_on.openai.calls == []
    # Note complète : un échec non lié au compte retombe sur le gabarit, jamais sur OpenAI.
    fake.clear_injections()
    fake.inject("POST", CHAT_PATH, *([failure] * 60))
    sentinel, html = lng.build_note_html(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL,
                                         mode="short", openai_api_key=FAKE_OPENAI_KEY,
                                         openrouter_api_key=FAKE_OPENROUTER_KEY,
                                         albert_api_key=FAKE_ALBERT_KEY)
    assert "template" in html
    assert sentinel in html
    assert albert_on.openai.calls == []


def test_no_env_key_fallback(albert_on, monkeypatch):
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    fake = albert_on.fake
    with pytest.raises((ValueError, AlbertError)):
        lng._generate_with_llm("Invite.", model="albert/" + SMALL_MODEL, mode="short", albert_api_key=None)
    with pytest.raises((ValueError, AlbertError)):
        lng._get_albert_client(None)
    with pytest.raises((ValueError, AlbertError)):
        lng.build_abstract_text(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL)
    assert fake.calls == []
    assert albert_on.openai.calls == []
    # Aucun module du lot ne lit la clé Albert dans l'environnement.
    for path in ("app/utils/llm_note_generator.py", "app/utils/book_note_generator.py",
                 "app/utils/citation_filter.py", "app/utils/parallel_citation_processor.py"):
        with open(os.path.join(RAGPY_ROOT, path), encoding="utf-8") as fh:
            source = fh.read()
        assert "getenv(\"ALBERT_API_KEY" not in source and "getenv('ALBERT_API_KEY" not in source, path
        assert "environ[\"ALBERT_API_KEY" not in source and "environ['ALBERT_API_KEY" not in source, path
        assert "environ.get(\"ALBERT_API_KEY" not in source and "environ.get('ALBERT_API_KEY" not in source, path


@pytest.mark.parametrize("failure,error_class", [
    ("invalid_key", AlbertAuthError),
    ("account_expired", AlbertAuthError),
    ({"status": 429, "body": {"detail": "Too many requests."}, "retry_after": 3600}, AlbertQuotaExhausted),
])
def test_account_error_propagates(failure, error_class, albert_on):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *([failure] * 20))
    with pytest.raises(error_class) as excinfo:
        lng.build_note_html(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL, mode="extended",
                            albert_api_key=FAKE_ALBERT_KEY)
    assert excinfo.value.credential_required == "albert_api_key"
    with pytest.raises(error_class):
        lng.build_abstract_text(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL,
                                albert_api_key=FAKE_ALBERT_KEY)
    with pytest.raises(error_class):
        asyncio.run(lng.build_note_html_async(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL,
                                              mode="short", albert_api_key=FAKE_ALBERT_KEY))
    with pytest.raises(error_class):
        asyncio.run(lng.build_abstract_text_async(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL,
                                                  albert_api_key=FAKE_ALBERT_KEY))
    assert albert_on.openai.calls == []


def test_book_account_error_propagates(albert_on, monkeypatch):
    fake = albert_on.fake
    fake.chat_reply = _book_reply
    # Phase 1 (bout en bout, via le faux) : la 401 n'est plus avalée.
    fake.inject("POST", CHAT_PATH, *(["invalid_key"] * 20))
    with pytest.raises(AlbertAuthError):
        asyncio.run(bng.build_book_note_async(BOOK_METADATA, _book_text(), model="albert/" + SMALL_MODEL,
                                              albert_api_key=FAKE_ALBERT_KEY))
    assert len(_chat_calls(fake)) == 1
    fake.clear_injections()

    # Phases 2 et 3 : les deux sites try/except du chef d'orchestre relancent les erreurs de compte.
    real_phase2 = bng._phase2_analyse_chapter
    real_phase3 = bng._phase3_synthesise

    async def phase2_quota(*args, **kwargs):
        """Phase 2 en quota épuisé."""
        raise AlbertQuotaExhausted(status=429, retry_after=3600)

    monkeypatch.setattr(bng, "_phase2_analyse_chapter", phase2_quota)
    with pytest.raises(AlbertQuotaExhausted):
        asyncio.run(bng.build_book_note_async(BOOK_METADATA, _book_text(), model="albert/" + SMALL_MODEL,
                                              albert_api_key=FAKE_ALBERT_KEY))
    monkeypatch.setattr(bng, "_phase2_analyse_chapter", real_phase2)

    async def phase3_auth(*args, **kwargs):
        """Phase 3 en clé invalide."""
        raise AlbertAuthError(reason="invalid_key", status=401)

    monkeypatch.setattr(bng, "_phase3_synthesise", phase3_auth)
    with pytest.raises(AlbertAuthError):
        asyncio.run(bng.build_book_note_async(BOOK_METADATA, _book_text(), model="albert/" + SMALL_MODEL,
                                              albert_api_key=FAKE_ALBERT_KEY))
    monkeypatch.setattr(bng, "_phase3_synthesise", real_phase3)

    # Une erreur ordinaire reste avalée comme avant (chapitre marqué indisponible).
    async def phase2_generic(*args, **kwargs):
        """Phase 2 en erreur quelconque."""
        raise RuntimeError("panne quelconque")

    monkeypatch.setattr(bng, "_phase2_analyse_chapter", phase2_generic)
    sentinel, html = asyncio.run(bng.build_book_note_async(BOOK_METADATA, _book_text(),
                                                           model="albert/" + SMALL_MODEL,
                                                           albert_api_key=FAKE_ALBERT_KEY))
    assert "indisponible" in html
    assert albert_on.openai.calls == []


def test_book_phase_functions_raise_account_errors(albert_on):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *(["invalid_key"] * 20))
    full = bng._load_book_prompt()
    tpl1 = bng._extract_phase_prompt(full, "phase1")
    tpl2 = bng._extract_phase_prompt(full, "phase2")
    tpl3 = bng._extract_phase_prompt(full, "phase3")
    text = _book_text()
    structure = bng._build_initial_structure(text, BOOK_METADATA)
    meta = dict(BOOK_METADATA, language_label="français")
    common = {"model": "albert/" + SMALL_MODEL, "openai_api_key": None, "openrouter_api_key": None,
              "albert_api_key": FAKE_ALBERT_KEY}
    with pytest.raises(AlbertAuthError):
        asyncio.run(bng._phase1_detect_structure(text, meta, tpl1, **common))
    with pytest.raises(AlbertAuthError):
        asyncio.run(bng._phase2_analyse_chapter(structure.chapters[0], structure, meta, [], tpl2, **common))
    with pytest.raises(AlbertAuthError):
        asyncio.run(bng._phase3_synthesise(structure, ["Ch.1 : résumé."], meta, tpl3, **common))


def test_semaphore_released_during_backoff(albert_on, fast_async_sleep):
    fake = albert_on.fake
    # Deux refus transitoires (429 court, puis 500) avant le succès.
    fake.inject("POST", CHAT_PATH, {"status": 429, "body": {"detail": "Too many requests."}, "retry_after": 1},
                "server_error")

    async def scenario():
        """Note asynchrone, sémaphore global de taille 1."""
        lng._llm_semaphore = asyncio.Semaphore(1)
        return await lng.build_note_html_async(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL,
                                               mode="short", albert_api_key=FAKE_ALBERT_KEY)

    sentinel, html = asyncio.run(scenario())
    assert NOTE_HTML in html
    assert len(_chat_calls(fake)) == 3
    waits = [(seconds, state) for seconds, state in fast_async_sleep] + list(
        zip(albert_on.sleeps.calls, albert_on.sleeps.states))
    waits = [(seconds, state) for seconds, state in waits if seconds > 0]
    assert len(waits) >= 2
    for _seconds, state in waits:
        assert state and all(state.values()), state
    assert albert_on.openai.calls == []


def test_single_retry_layer_notes(albert_on, fast_async_sleep):
    fake = albert_on.fake
    max_retries = AlbertConfig.from_env().max_retries
    # Synchrone : exactement 1 + ALBERT_MAX_RETRIES envois (pas de boucle max_attempts en plus).
    fake.inject("POST", CHAT_PATH, *(["server_error"] * 60))
    with pytest.raises(AlbertError):
        lng._generate_with_llm("Invite.", model="albert/" + SMALL_MODEL, mode="short",
                               albert_api_key=FAKE_ALBERT_KEY)
    assert len(_chat_calls(fake)) == 1 + max_retries
    # Asynchrone (run_llm_slot) : même compte, puis gabarit.
    fake.reset_calls()
    fake.clear_injections()
    fake.inject("POST", CHAT_PATH, *(["server_error"] * 60))

    async def scenario():
        """Note asynchrone qui échoue sur tous les essais."""
        return await lng.build_note_html_async(METADATA, text_content=TEXT, model="albert/" + SMALL_MODEL,
                                               mode="short", albert_api_key=FAKE_ALBERT_KEY)

    _sentinel, html = asyncio.run(scenario())
    assert "template" in html
    assert len(_chat_calls(fake)) == 1 + max_retries
    assert albert_on.openai.calls == []


def test_book_three_phases_forward_key(albert_on, monkeypatch):
    fake = albert_on.fake
    fake.chat_reply = _book_reply
    for func in (bng._phase1_detect_structure, bng._phase2_analyse_chapter, bng._phase3_synthesise,
                 bng.build_book_note_async):
        param = inspect.signature(func).parameters.get("albert_api_key")
        assert param is not None, func.__name__
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, func.__name__
    seen = []
    for name in ("_phase1_detect_structure", "_phase2_analyse_chapter", "_phase3_synthesise"):
        real = getattr(bng, name)

        async def spy(*args, __real=real, __name=name, **kwargs):
            """Note si la clé Albert a été transmise (booléen), puis délègue."""
            seen.append((__name, kwargs.get("albert_api_key") == FAKE_ALBERT_KEY))
            return await __real(*args, **kwargs)

        monkeypatch.setattr(bng, name, spy)
    sentinel, html = asyncio.run(bng.build_book_note_async(BOOK_METADATA, _book_text(),
                                                           model="albert/" + SMALL_MODEL,
                                                           albert_api_key=FAKE_ALBERT_KEY))
    names = [name for name, _ok in seen]
    assert names[0] == "_phase1_detect_structure"
    assert "_phase2_analyse_chapter" in names
    assert names[-1] == "_phase3_synthesise"
    assert all(ok for _name, ok in seen)
    chats = _chat_calls(fake)
    assert len(chats) == len(seen)
    assert all(call.headers.get("authorization") == "Bearer " + FAKE_ALBERT_KEY for call in chats)
    # Phase 1 : sortie structurée (JSON) ; phases 2 et 3 : texte libre.
    assert "response_format" in chats[0].json
    assert all("response_format" not in call.json for call in chats[1:])
    assert albert_on.openai.calls == []
    assert sentinel in html


def test_long_prompt_uses_long_context_model(albert_on):
    fake = albert_on.fake
    context = albert_catalog.context_length(REASONING_MODEL)
    long_prompt = "mot " * int(context * 0.9 * 3.2 / 4 + 1000)
    assert len(long_prompt) / 3.2 > 0.9 * context
    lng._generate_with_llm(long_prompt, model="albert/" + REASONING_MODEL, mode="extended",
                           albert_api_key=FAKE_ALBERT_KEY)
    expected = albert_catalog.fallback_chain("long_context", today=date.today())[0]
    assert _chat_calls(fake)[-1].json["model"] == expected
    # Une invite courte reste sur gpt-oss.
    lng._generate_with_llm("Invite courte.", model="albert/" + REASONING_MODEL, mode="extended",
                           albert_api_key=FAKE_ALBERT_KEY)
    assert _chat_calls(fake)[-1].json["model"] == REASONING_MODEL
