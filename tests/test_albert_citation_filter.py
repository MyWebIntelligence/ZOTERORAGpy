"""Tests du filtre de citations sur Albert (lot 3, tâches 8 et 9).

* ``_call_llm_api`` : un contenu ``None`` donne ``AlbertTruncatedError`` (jamais
  ``AttributeError``), budget 1500 (+ marge et effort ``low`` pour un modèle à
  raisonnement), aucun appel OpenAI ou OpenRouter sur le chemin Albert
  (invariant 19).
* ``pre_filter_citation`` / ``filter_citation_with_llm`` : passage par
  ``run_llm_slot`` (sommeil hors sémaphores, invariant 17 ; aucune acquisition
  imbriquée, invariant 36 ; une seule couche de retry, invariant 37), erreurs de
  compte qui remontent au lieu du repli « pertinent » (invariant 38), lecture du
  pré-filtre par jeton exact sur la branche Albert seulement, lecture historique
  inchangée sur le chemin OFF.
* ``process_citations_parallel`` : appel positionnel historique toujours valide ;
  à la première erreur de compte, les tâches restantes sont annulées et
  l'erreur est relancée.

``tests/albert_fakes.FakeAlbert`` est branché sur tout ``AlbertClient`` construit
pendant le test ; la récupération web est remplacée par un double (aucun
réseau). Coroutines exécutées avec ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests import test_albert_off_golden as golden  # noqa: E402

from app.utils import citation_fetcher  # noqa: E402
from app.utils import parallel_citation_processor as pcp  # noqa: E402
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
cfilter = golden.cfilter

CHAT_PATH = "/v1/chat/completions"
SMALL_MODEL = "ministral-3-8b-instruct-2512"
REASONING_MODEL = "gpt-oss-120b"
ALBERT_SMALL = "albert/" + SMALL_MODEL
FAKE_OPENAI_KEY = golden.FAKE_ENV["OPENAI_API_KEY"]
FAKE_OPENROUTER_KEY = golden.FAKE_ENV["OPENROUTER_API_KEY"]
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
PREFILTER_MARK = 'Réponds UNIQUEMENT par "RELEVANT" ou "NA"'
PROJECT = {
    "project_name": "Langage ordinaire",
    "project_description": "Philosophie du langage ordinaire et sociologie des usages.",
    "collection_name": "Corpus",
    "collection_description": "Articles retenus.",
}
FILTER_JSON = {
    "relevance_score": 82,
    "relevance_reason": "Article central pour le projet.",
    "zotero_item": {
        "itemType": "journalArticle",
        "title": "Le langage ordinaire",
        "creators": [{"creatorType": "author", "firstName": "Jeanne", "lastName": "Dupont"}],
        "date": "2021",
    },
}
FILTER_REPLY = "```json\n" + json.dumps(FILTER_JSON, ensure_ascii=False) + "\n```"


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
class SleepRecorder:
    """Sommeil factice (synchrone) : enregistre les durées et l'état observé des sémaphores."""

    def __init__(self, probe=None):
        """Crée l'enregistreur ; ``probe()`` est appelé à chaque sommeil."""
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


def _chat_calls(fake):
    """Appels de chat reçus par le faux."""
    return fake.calls_to("POST", CHAT_PATH)


def _module_semaphores():
    """Sémaphores asyncio tenus au niveau du module des notes (global et Albert)."""
    return {name: value for name, value in vars(lng).items() if isinstance(value, asyncio.Semaphore)}


def _free_semaphores():
    """Pour chaque sémaphore du module, vrai s'il n'est tenu par personne (taille fixée par le test)."""
    capacities = {"_llm_semaphore": 1}
    capacities.update({name: AlbertConfig.from_env().notes_concurrency
                       for name in _module_semaphores() if name != "_llm_semaphore"})
    return {name: sem._value >= capacities.get(name, 1) for name, sem in _module_semaphores().items()}


def _filter_reply(body):
    """Réponse de chat : « RELEVANT » au pré-filtre, JSON valide au filtre complet."""
    user = body["messages"][-1]["content"]
    if PREFILTER_MARK in user:
        return "RELEVANT"
    return FILTER_REPLY


def _citation(i=1):
    """Citation Publish or Perish minimale (sans URL : aucune récupération web)."""
    return {
        "title": f"Le langage ordinaire, étude {i}",
        "authors": ["Dupont, Jeanne"],
        "abstract": "Étude des usages du langage ordinaire.",
        "source": "Revue de philosophie",
        "year": 2021,
        "cites": 3,
    }


def _content_double(calls, content_fn):
    """Remplaçant de ``_get_llm_clients`` : clients OpenAI/OpenRouter qui répondent ``content_fn``."""
    def fake_get_llm_clients(*args, **extra):
        """Renvoie deux clients enregistreurs et le modèle par défaut historique."""
        def content(label, kwargs):
            """Contenu imposé par le test."""
            return content_fn(kwargs)

        return (golden._FakeLLMClient("openai", calls, content),
                golden._FakeLLMClient("openrouter", calls, content),
                "gpt-4o-mini")
    return fake_get_llm_clients


def _openai_content_for_filter(kwargs):
    """Réponse historique : « RELEVANT » au pré-filtre, JSON au filtre complet."""
    prompt = kwargs["messages"][-1]["content"]
    return "RELEVANT" if PREFILTER_MARK in prompt else FILTER_REPLY


async def _no_fetch(*args, **kwargs):
    """Double de ``fetch_citation_content`` : aucun contenu web, aucune requête."""
    return "", "none"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Identifiants retirés, sémaphores du module, limiteurs et preflight remis à zéro, web neutralisé."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in list(_module_semaphores()):
        monkeypatch.setattr(lng, name, None)
    monkeypatch.setattr(citation_fetcher, "fetch_citation_content", _no_fetch)
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def openai_calls(monkeypatch):
    """``_get_llm_clients`` du filtre remplacé par des clients enregistreurs (réponses historiques)."""
    calls = []
    monkeypatch.setattr(cfilter, "_get_llm_clients", _content_double(calls, _openai_content_for_filter))
    return calls


def _albert_env(monkeypatch, enabled):
    """Albert ON ou OFF, débits élevés ; la clé n'est jamais posée dans l'environnement."""
    monkeypatch.setenv("ALBERT_ENABLED", "1" if enabled else "0")
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def albert_on(monkeypatch, openai_calls):
    """Albert ON ; ``FakeAlbert`` répond « RELEVANT » puis le JSON du filtre."""
    fake = FakeAlbert()
    fake.chat_reply = _filter_reply
    sleeps = SleepRecorder(probe=_free_semaphores)
    route_albert_to_fake(monkeypatch, fake, sleeps)
    _albert_env(monkeypatch, enabled=True)
    return SimpleNamespace(fake=fake, sleeps=sleeps, openai=openai_calls)


@pytest.fixture
def fast_async_sleep(monkeypatch):
    """``asyncio.sleep`` enregistré (durée, état des sémaphores) et ramené à zéro."""
    original = asyncio.sleep
    records = []

    async def spy(delay, *args, **kwargs):
        """Enregistre les attentes non nulles puis cède la main sans attendre."""
        if delay and delay > 0:
            records.append((float(delay), _free_semaphores()))
        return await original(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return records


def _prefilter(model=ALBERT_SMALL, **kwargs):
    """Coroutine de pré-filtrage d'une citation pour le projet de test."""
    return cfilter.pre_filter_citation(
        _citation(), PROJECT["project_name"], PROJECT["project_description"],
        PROJECT["collection_name"], PROJECT["collection_description"], model=model, **kwargs,
    )


def _full_filter(model=ALBERT_SMALL, citation=None, **kwargs):
    """Coroutine de filtrage complet d'une citation pour le projet de test."""
    return cfilter.filter_citation_with_llm(
        citation or _citation(), "", "none", PROJECT["project_name"], PROJECT["project_description"],
        PROJECT["collection_name"], PROJECT["collection_description"], model=model, **kwargs,
    )


async def _collect(generator):
    """Consomme un générateur asynchrone d'événements et renvoie la liste."""
    return [event async for event in generator]


# ---------------------------------------------------------------------------
# _call_llm_api
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_none_content_no_attributeerror(finish, albert_on):
    albert_on.fake.chat_reply = {"content": None, "finish_reason": finish}
    with pytest.raises(AlbertTruncatedError):
        cfilter._call_llm_api("Invite.", model=ALBERT_SMALL, albert_api_key=FAKE_ALBERT_KEY)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(_full_filter(albert_api_key=FAKE_ALBERT_KEY))
    assert not isinstance(excinfo.value, AttributeError)
    # Pré-filtre : repli « pertinent » conservé pour une erreur qui n'est pas de compte.
    assert asyncio.run(_prefilter(albert_api_key=FAKE_ALBERT_KEY)) is True
    assert albert_on.openai == []


def test_reasoning_budget_low_effort(albert_on):
    fake = albert_on.fake
    fake.chat_reply = "RELEVANT"
    headroom = AlbertConfig.from_env().reasoning_headroom
    cfilter._call_llm_api("Invite.", model="albert/" + REASONING_MODEL, albert_api_key=FAKE_ALBERT_KEY)
    body = _chat_calls(fake)[-1].json
    assert body["model"] == REASONING_MODEL
    assert body["max_tokens"] == 1500 + headroom
    assert body["reasoning_effort"] == "low"
    cfilter._call_llm_api("Invite.", model=ALBERT_SMALL, albert_api_key=FAKE_ALBERT_KEY)
    body = _chat_calls(fake)[-1].json
    assert body["model"] == SMALL_MODEL
    assert body["max_tokens"] == 1500
    assert "reasoning_effort" not in body
    assert albert_on.openai == []


@pytest.mark.parametrize("failure", ["server_error", "not_found", "model_busy"])
def test_albert_never_calls_openai(failure, albert_on, fast_async_sleep):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *([failure] * 200))
    with pytest.raises(AlbertError):
        cfilter._call_llm_api("Invite.", model=ALBERT_SMALL, openai_api_key=FAKE_OPENAI_KEY,
                              openrouter_api_key=FAKE_OPENROUTER_KEY, albert_api_key=FAKE_ALBERT_KEY)
    assert asyncio.run(_prefilter(openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
                                  albert_api_key=FAKE_ALBERT_KEY)) is True
    with pytest.raises(Exception):
        asyncio.run(_full_filter(openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
                                 albert_api_key=FAKE_ALBERT_KEY))
    assert albert_on.openai == []
    # Succès : la réponse d'Albert est utilisée, toujours sans OpenAI.
    fake.clear_injections()
    result = asyncio.run(_full_filter(openai_api_key=FAKE_OPENAI_KEY, albert_api_key=FAKE_ALBERT_KEY))
    assert isinstance(result, dict) and result["relevance_score"] == 82
    assert albert_on.openai == []


# ---------------------------------------------------------------------------
# Signatures et appel positionnel historique
# ---------------------------------------------------------------------------
def test_parallel_positional_call_without_key_still_works(openai_calls, monkeypatch):
    _albert_env(monkeypatch, enabled=False)
    for func in (cfilter.pre_filter_citation, cfilter.filter_citation_with_llm, pcp.process_citations_parallel):
        params = list(inspect.signature(func).parameters.values())
        assert params[-1].name == "albert_api_key", func.__name__
        assert params[-1].default is None, func.__name__
    assert inspect.signature(cfilter._call_llm_api).parameters["albert_api_key"].default is None
    config = dict(PROJECT, model="gpt-4o-mini")
    citations = [_citation(1), _citation(2)]
    events = asyncio.run(_collect(pcp.process_citations_parallel(citations, config, FAKE_OPENAI_KEY,
                                                                 FAKE_OPENROUTER_KEY)))
    kinds = [kind for kind, _data in events]
    assert kinds[0] == "init" and kinds[-1] == "complete"
    assert kinds.count("progress") == 2
    assert events[-1][1]["relevant"] == 2
    assert {call["client"] for call in openai_calls} == {"openai"}
    # Appel positionnel historique du pré-filtre (sans clé Albert).
    ok = asyncio.run(cfilter.pre_filter_citation(
        _citation(), PROJECT["project_name"], PROJECT["project_description"], PROJECT["collection_name"],
        PROJECT["collection_description"], "gpt-4o-mini", FAKE_OPENAI_KEY, FAKE_OPENROUTER_KEY))
    assert ok is True


# ---------------------------------------------------------------------------
# Sémaphores, retry, erreurs de compte
# ---------------------------------------------------------------------------
def test_semaphore_released_during_backoff(albert_on, fast_async_sleep):
    fake = albert_on.fake
    transient = [{"status": 429, "body": {"detail": "Too many requests."}, "retry_after": 1}, "server_error"]

    async def scenario():
        """Filtre complet puis pré-filtre, chacun après deux refus transitoires ; sémaphore global de taille 1."""
        lng._llm_semaphore = asyncio.Semaphore(1)
        fake.inject("POST", CHAT_PATH, *transient)
        full = await _full_filter(albert_api_key=FAKE_ALBERT_KEY)
        fake.inject("POST", CHAT_PATH, *transient)
        pre = await _prefilter(albert_api_key=FAKE_ALBERT_KEY)
        return full, pre

    full, pre = asyncio.run(scenario())
    assert isinstance(full, dict) and full["relevance_score"] == 82
    assert pre is True
    assert len(_chat_calls(fake)) == 6
    waits = list(fast_async_sleep) + list(zip(albert_on.sleeps.calls, albert_on.sleeps.states))
    waits = [(seconds, state) for seconds, state in waits if seconds > 0]
    assert len(waits) >= 4
    for _seconds, state in waits:
        assert state and all(state.values()), state
    assert albert_on.openai == []


def test_no_nested_acquire_deadlock(albert_on, monkeypatch):
    monkeypatch.setenv("ALBERT_NOTES_CONCURRENCY", "1")
    config = dict(PROJECT, model=ALBERT_SMALL)

    async def scenario():
        """Deux citations, sémaphore global de taille 1, délai de garde contre l'interblocage."""
        lng._llm_semaphore = asyncio.Semaphore(1)
        events = await asyncio.wait_for(
            _collect(pcp.process_citations_parallel([_citation(1), _citation(2)], config, None, None,
                                                    albert_api_key=FAKE_ALBERT_KEY)),
            timeout=30,
        )
        both = await asyncio.wait_for(
            asyncio.gather(_full_filter(citation=_citation(3), albert_api_key=FAKE_ALBERT_KEY),
                           _full_filter(citation=_citation(4), albert_api_key=FAKE_ALBERT_KEY)),
            timeout=30,
        )
        # Filtrage par lots : le sémaphore global n'est pas tenu autour de l'appel Albert.
        batch = await asyncio.wait_for(
            pcp._filter_batch_parallel([_citation(5), _citation(6)], [("", "none"), ("", "none")], config, 0,
                                       None, None, albert_api_key=FAKE_ALBERT_KEY),
            timeout=30,
        )
        return events, both, batch, lng._llm_semaphore._value

    events, both, batch, final_value = asyncio.run(scenario())
    assert events[-1][0] == "complete"
    assert events[-1][1]["relevant"] == 2
    assert all(isinstance(result, dict) for result in both)
    assert [result.status for result in batch] == ["relevant", "relevant"]
    assert final_value == 1
    assert albert_on.openai == []


def test_single_retry_layer_citations(albert_on, fast_async_sleep):
    fake = albert_on.fake
    max_retries = AlbertConfig.from_env().max_retries
    fake.inject("POST", CHAT_PATH, *(["server_error"] * 200))
    with pytest.raises(Exception) as excinfo:
        asyncio.run(_full_filter(max_retries=1, albert_api_key=FAKE_ALBERT_KEY))
    assert not isinstance(excinfo.value, AttributeError)
    assert len(_chat_calls(fake)) == 1 + max_retries
    fake.reset_calls()
    assert asyncio.run(_prefilter(albert_api_key=FAKE_ALBERT_KEY)) is True
    assert len(_chat_calls(fake)) == 1 + max_retries
    assert albert_on.openai == []


@pytest.mark.parametrize("failure,error_class", [
    ("invalid_key", AlbertAuthError),
    ("budget_exhausted", AlbertAuthError),
    ({"status": 429, "body": {"detail": "Too many requests."}, "retry_after": 3600}, AlbertQuotaExhausted),
])
def test_account_error_aborts_job(failure, error_class, albert_on):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *([failure] * 200))
    with pytest.raises(error_class):
        asyncio.run(_prefilter(albert_api_key=FAKE_ALBERT_KEY))
    with pytest.raises(error_class):
        asyncio.run(_full_filter(albert_api_key=FAKE_ALBERT_KEY))
    fake.reset_calls()
    config = dict(PROJECT, model=ALBERT_SMALL)
    citations = [_citation(i) for i in range(1, 9)]
    leftovers = {}

    async def scenario():
        """Traitement parallèle interrompu par l'erreur de compte ; tâches restantes relevées."""
        events = []
        try:
            async for event in pcp.process_citations_parallel(citations, config, None, None, batch_size=2,
                                                              albert_api_key=FAKE_ALBERT_KEY):
                events.append(event)
        finally:
            for _ in range(10):
                await asyncio.sleep(0)
            current = asyncio.current_task()
            leftovers["pending"] = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        return events

    with pytest.raises(error_class) as excinfo:
        asyncio.run(scenario())
    assert excinfo.value.credential_required == "albert_api_key"
    assert leftovers["pending"] == []
    # L'arrêt est immédiat : bien moins d'appels que de citations.
    assert len(_chat_calls(fake)) < len(citations)
    assert albert_on.openai == []


class _GuardScaledAsyncio:
    """Module ``asyncio`` vu par ``parallel_citation_processor`` : délais de ``wait_for`` réduits.

    Tout autre attribut est celui du vrai module. Le remplacement est limité au
    module testé (le client et la couche de réessai gardent le vrai ``asyncio``).
    """

    def __init__(self, factor):
        """Crée le module de substitution ; ``factor`` multiplie chaque délai de ``wait_for``."""
        self._factor = factor
        self.expired = []

    def __getattr__(self, name):
        """Attribut du vrai module ``asyncio``."""
        return getattr(asyncio, name)

    async def wait_for(self, aw, timeout=None):
        """``asyncio.wait_for`` au délai réduit ; note chaque expiration (délai d'origine)."""
        scaled = None if timeout is None else timeout * self._factor
        try:
            return await asyncio.wait_for(aw, scaled)
        except asyncio.TimeoutError:
            self.expired.append(timeout)
            raise


def test_account_error_after_result_guard_aborts_job(albert_on, monkeypatch):
    # Garde d'attente des résultats (120 s) ramenée à 0,25 s ; 429 courts persistants
    # (Retry-After 1, un réessai) : AlbertQuotaExhausted n'arrive qu'après ~1 s, donc
    # après l'expiration de la garde. L'erreur de compte doit quand même arrêter le job.
    fake = albert_on.fake
    monkeypatch.setenv("ALBERT_MAX_RETRIES", "1")
    guard = _GuardScaledAsyncio(0.25 / 120.0)
    monkeypatch.setattr(pcp, "asyncio", guard)
    short_429 = {"status": 429, "body": {"detail": "Too many requests."}, "retry_after": 1}
    fake.inject("POST", CHAT_PATH, *([short_429] * 200))
    config = dict(PROJECT, model=ALBERT_SMALL)
    citations = [_citation(1), _citation(2)]
    leftovers = {}

    async def scenario():
        """Traitement parallèle ; renvoie les événements reçus et l'exception levée (ou ``None``)."""
        events, error = [], None
        try:
            async for event in pcp.process_citations_parallel(citations, config, None, None, batch_size=2,
                                                              albert_api_key=FAKE_ALBERT_KEY):
                events.append(event)
        except Exception as exc:  # noqa: BLE001 - l'assertion porte sur le type
            error = exc
        finally:
            for _ in range(10):
                await asyncio.sleep(0)
            current = asyncio.current_task()
            leftovers["pending"] = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        return events, error

    events, error = asyncio.run(scenario())
    kinds = [kind for kind, _data in events]
    # Le scénario n'a de sens que si la garde a expiré avant l'erreur de compte.
    assert guard.expired, "la garde d'attente n'a pas expiré : scénario à ajuster"
    assert "complete" not in kinds, kinds
    assert isinstance(error, AlbertQuotaExhausted), repr(error)
    assert error.credential_required == "albert_api_key"
    assert leftovers["pending"] == []
    # La couche de réessai a bien tourné (1 envoi + 1 réessai au moins).
    assert len(_chat_calls(fake)) >= 2
    assert albert_on.openai == []


# ---------------------------------------------------------------------------
# Lecture du pré-filtre
# ---------------------------------------------------------------------------
ALBERT_PREFILTER_CASES = [
    ("RELEVANT", True),
    ("NA", False),
    ("relevant", True),
    ("na", False),
    ("NA.", False),
    ("RELEVANT.", True),
    ("**RELEVANT**", True),
    ("**NA**", False),
    ("  relevant\n", True),
    ("RELEVANT — l'article traite de NA et de sa grammaire", True),
    ("NA : hors sujet, bien que RELEVANT pour un autre projet", False),
    ("IRRELEVANT", True),
    ("Analysons la demande avant de répondre NA", True),
    ("Peut-être", True),
]


@pytest.mark.parametrize("reply,expected", ALBERT_PREFILTER_CASES)
def test_albert_prefilter_exact_token_parse(reply, expected, albert_on):
    albert_on.fake.chat_reply = reply
    assert asyncio.run(_prefilter(albert_api_key=FAKE_ALBERT_KEY)) is expected
    assert albert_on.openai == []


OFF_PREFILTER_CASES = [
    ("RELEVANT", True),
    ("NA", False),
    ("relevant", True),
    ("RELEVANT — l'article traite de NA", False),
    ("IRRELEVANT", True),
    ("Relevant : analyse du corpus", False),
    ("Peut-être", False),
]


@pytest.mark.parametrize("reply,expected", OFF_PREFILTER_CASES)
@pytest.mark.parametrize("enabled", [False, True], ids=["albert_off", "albert_on_openai_model"])
def test_off_prefilter_parse_unchanged(reply, expected, enabled, monkeypatch):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled)
    calls = []
    monkeypatch.setattr(cfilter, "_get_llm_clients", _content_double(calls, lambda kwargs: reply))
    result = asyncio.run(_prefilter(model="gpt-4o-mini", openai_api_key=FAKE_OPENAI_KEY,
                                    albert_api_key=FAKE_ALBERT_KEY))
    assert result is expected
    assert [call["client"] for call in calls] == ["openai"]
    assert fake.calls == []
