"""Régressions de la revue Albert (correctifs G1), sans réseau ni identifiant réel.

* **budget du limiteur web selon le modèle** : ``albert_slot_policy`` accepte
  ``model=`` et ``_albert_slot_chat`` construit la politique pour chaque modèle
  de la chaîne de repli. Un modèle à raisonnement (gpt-oss) consomme le budget
  ``notes`` même sous un rôle du budget ``recode`` : phase 1 des fiches de
  livre (rôle ``book_structure``) et filtre de citations asynchrone (rôle
  ``citation``) avec ``albert/gpt-oss-120b`` ; un repli ministral de la même
  chaîne revient au budget ``recode`` ;
* **filtrage des replis par /v1/models sur le chemin web** : la liste
  ``/v1/models`` mise en cache par le module des notes est installée dans
  chaque nouveau client (``AlbertClient.set_model_listing``), à la
  construction comme lors d'un succès de cache, de sorte que ``chat_chain``
  retire les replis absents du compte, y compris pour un id épinglé. Aucune
  requête ``/v1/models`` supplémentaire n'est émise, et la liste d'une clé
  n'est jamais installée dans le client d'une autre clé ;
* **liste chargée juste avant le premier repli** : sans liste en cache, une
  chaîne à plusieurs modèles charge ``/v1/models`` une seule fois, après les 503
  du modèle en tête et avant le premier repli (fil de travail sur le chemin
  asynchrone, chemins synchrone et asynchrone), de sorte qu'un repli absent du
  compte n'est jamais envoyé ; un seul ``GET /v1/models`` par clé et par 600 s
  (deux jobs consécutifs), aucun si le modèle en tête répond ; un échec de la
  liste (hors erreur de compte) garde la chaîne du catalogue.

``FakeAlbert`` derrière un ``httpx.MockTransport`` ; les sommeils de réessai
sont enregistrés, jamais attendus.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import os
import sys

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
SCRIPTS_DIR = os.path.join(RAGPY_ROOT, "scripts")
for _p in (RAGPY_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests import test_albert_off_golden as golden  # noqa: E402
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert, load_fixture  # noqa: E402
from tests.test_albert_notes import SleepRecorder, _client_classes, route_albert_to_fake  # noqa: E402

from app.utils import book_note_generator as bng  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402

lng = golden.lng
cfilter = golden.cfilter

CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
REASONING_MODEL = "gpt-oss-120b"
REASONING_ALIAS = "openweight-large"
MINISTRAL = "ministral-3-8b-instruct-2512"
OTHER_FAKE_KEY = "fake-albert-key-0002"
NOTE_TEXT = "<p>Fiche factice.</p>"
BUSY_REPLIES = 12
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
RECODE_BUDGET_ROLES = ("book_structure", "citation", "long_context", "recode")


# ---------------------------------------------------------------------------
# Environnement
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Retire identifiants et réglages Albert ; sémaphores, limiteurs et caches remis à zéro."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in vars(lng).items():
        if isinstance(value, asyncio.Semaphore):
            monkeypatch.setattr(lng, name, None)
    lng.clear_albert_listing_cache()
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)
    yield
    lng.clear_albert_listing_cache()
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


def _listing_without_ministral():
    """Liste ``/v1/models`` du compte sans ministral (repli de ``notes`` et de ``citation``)."""
    models = copy.deepcopy(load_fixture("P2_models.json")["body"]["data"])
    return [m for m in models if m["id"] != MINISTRAL]


def _production_scope(client):
    """Portée de cache d'un client sans transport injecté : empreinte de clé et URL de base.

    ``route_albert_to_fake`` injecte un transport, ce qui donne à chaque client
    une portée propre ; les tests du cache partagé rétablissent la portée de
    production (deux clients de même clé et même URL partagent la liste).
    """
    return (client.key_fingerprint, client.base_url, None)


def _share_scope_between_clients(monkeypatch):
    """Donne à tout ``AlbertClient`` la portée de cache de production (clé, URL)."""
    for cls in _client_classes():
        monkeypatch.setattr(cls, "cache_scope", property(_production_scope))


@pytest.fixture
def fake(monkeypatch):
    """``FakeAlbert`` au catalogue complet (P2), branché sur tout ``AlbertClient`` construit."""
    double = FakeAlbert()
    double.chat_reply = NOTE_TEXT
    route_albert_to_fake(monkeypatch, double, SleepRecorder())
    return double


@pytest.fixture
def shared(monkeypatch):
    """``FakeAlbert`` dont le compte n'offre pas ministral ; portée de cache partagée entre clients."""
    double = FakeAlbert(models=_listing_without_ministral())
    double.chat_reply = NOTE_TEXT
    route_albert_to_fake(monkeypatch, double, SleepRecorder())
    _share_scope_between_clients(monkeypatch)
    return double


class SlotSpy:
    """Espion de ``run_llm_slot`` : modèle envoyé et budget du limiteur de chaque créneau Albert.

    Le créneau d'origine est ensuite exécuté, avec un sommeil de réessai
    enregistré au lieu d'être attendu.
    """

    def __init__(self, monkeypatch):
        """Installe l'espion dans le module des notes."""
        self.slots = []
        self.sleeps = []
        self._original = lng.run_llm_slot
        monkeypatch.setattr(lng, "run_llm_slot", self._slot)

    async def _no_sleep(self, seconds):
        """Sommeil de réessai enregistré, sans attente."""
        self.sleeps.append(float(seconds))

    async def _slot(self, semaphore, thunk, *, albert_policy=None):
        """Enregistre ``(modèle, budget)`` puis exécute le créneau d'origine sans attente réelle."""
        if albert_policy is not None:
            wire = thunk.args[1] if isinstance(thunk, functools.partial) else None
            self.slots.append((wire, getattr(albert_policy.limiter, "name", None)))
            albert_policy = dataclasses.replace(albert_policy, sleep=self._no_sleep)
        return await self._original(semaphore, thunk, albert_policy=albert_policy)


def _sent_models(double):
    """Modèles envoyés à ``/v1/chat/completions``, dans l'ordre."""
    return [call.json["model"] for call in double.calls_to("POST", CHAT_PATH)]


def _listing_requests(double):
    """Nombre de requêtes ``GET /v1/models`` reçues."""
    return len(double.calls_to("GET", MODELS_PATH))


# ---------------------------------------------------------------------------
# 1. Budget du limiteur web selon le modèle envoyé
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", RECODE_BUDGET_ROLES)
def test_slot_policy_budget_follows_model(role):
    """``albert_slot_policy(model=gpt-oss)`` prend le budget ``notes`` sous un rôle du budget ``recode``."""
    cfg = lng.AlbertConfig.from_env()
    limiter = lng._albert_module("limiter")
    notes = limiter.get_limiter("notes", cfg)
    recode = limiter.get_limiter("recode", cfg)
    assert lng.albert_slot_policy(role, 10, cfg=cfg, model=REASONING_MODEL).limiter is notes
    assert lng.albert_slot_policy(role, 10, cfg=cfg, model="albert/" + REASONING_ALIAS).limiter is notes
    assert lng.albert_slot_policy(role, 10, cfg=cfg, model=MINISTRAL).limiter is recode
    assert lng.albert_slot_policy(role, 10, cfg=cfg, model=None).limiter is recode
    assert lng.albert_slot_policy(role, 10, cfg=cfg).limiter is recode
    assert lng.albert_slot_policy("notes", 10, cfg=cfg, model=MINISTRAL).limiter is notes


def test_book_phase1_with_gpt_oss_takes_notes_budget(fake, monkeypatch):
    """Phase 1 des fiches de livre (rôle ``book_structure``) avec gpt-oss : budget ``notes``."""
    spy = SlotSpy(monkeypatch)
    asyncio.run(bng._phase1_detect_structure(
        "Texte du livre. " * 50,
        {"title": "Livre factice", "authors": "Dupont, Jeanne"},
        "Structure de {TITLE} : {TOC_RAW}\n{FIRST_PAGES}\n{LAST_PAGES}",
        model="albert/" + REASONING_MODEL,
        openai_api_key=None,
        openrouter_api_key=None,
        albert_api_key=FAKE_ALBERT_KEY,
    ))
    assert _sent_models(fake) == [REASONING_MODEL]
    assert spy.slots == [(REASONING_MODEL, "notes")]


@pytest.mark.parametrize("model", [REASONING_MODEL, REASONING_ALIAS])
def test_async_citation_filter_with_gpt_oss_takes_notes_budget(model, fake, monkeypatch):
    """Filtre de citations asynchrone (rôle ``citation``) avec gpt-oss (id ou alias) : budget ``notes``."""
    spy = SlotSpy(monkeypatch)
    answer = asyncio.run(cfilter._albert_citation_call(
        "Invite.", "albert/" + model, temperature=0.1, albert_api_key=FAKE_ALBERT_KEY,
    ))
    assert answer == NOTE_TEXT
    assert _sent_models(fake) == [REASONING_MODEL]
    assert spy.slots == [(REASONING_MODEL, "notes")]


def test_fallback_chain_budget_is_rebuilt_per_model(fake, monkeypatch):
    """Chaîne ``citation`` : gpt-oss surchargé (budget ``notes``), puis repli ministral (budget ``recode``)."""
    fake.inject("POST", CHAT_PATH, "model_busy", "model_busy")
    spy = SlotSpy(monkeypatch)
    answer = asyncio.run(cfilter._albert_citation_call(
        "Invite.", "albert/" + REASONING_MODEL, temperature=0.1, albert_api_key=FAKE_ALBERT_KEY,
    ))
    assert answer == NOTE_TEXT
    assert _sent_models(fake) == [REASONING_MODEL, REASONING_MODEL, MINISTRAL]
    assert spy.slots == [(REASONING_MODEL, "notes"), (MINISTRAL, "recode")]


# ---------------------------------------------------------------------------
# 2. Replis filtrés par la liste /v1/models du cache web
# ---------------------------------------------------------------------------
def test_cached_listing_installed_into_new_client(shared):
    """Nouveau client, liste en cache : les replis absents du compte sont retirés, sans requête."""
    first = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    lng._albert_account_listing(first)
    assert _listing_requests(shared) == 1
    second = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    try:
        assert second.chat_chain("notes", REASONING_MODEL) == [REASONING_MODEL]
        assert second.chat_chain("notes") == [REASONING_MODEL]
    finally:
        first.close()
        second.close()
    assert _listing_requests(shared) == 1


def test_cache_hit_installs_listing_into_existing_client(shared):
    """Client construit avant le cache : un succès de cache lui installe la liste du compte."""
    early = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    warm = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    try:
        lng._albert_account_listing(warm)
        assert lng.albert_resolve_alias(early, REASONING_ALIAS) == REASONING_MODEL
        assert early.chat_chain("notes", REASONING_MODEL) == [REASONING_MODEL]
    finally:
        early.close()
        warm.close()
    assert _listing_requests(shared) == 1


def test_cached_listing_never_crosses_keys(shared):
    """La liste d'une clé n'est jamais installée dans le client d'une autre clé."""
    warm = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    other = None
    try:
        lng._albert_account_listing(warm)
        other = lng._get_albert_client(OTHER_FAKE_KEY, use_limiter=False)
        assert other.chat_chain("notes", REASONING_MODEL) == [REASONING_MODEL, MINISTRAL]
    finally:
        warm.close()
        if other is not None:
            other.close()


def test_expired_cached_listing_not_installed(shared, monkeypatch):
    """Liste expirée (au-delà de ``ALBERT_LISTING_TTL_SECONDS``) : jamais installée."""
    warm = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    try:
        lng._albert_account_listing(warm)
    finally:
        warm.close()
    monkeypatch.setattr(lng, "ALBERT_LISTING_TTL_SECONDS", 0.0)
    fresh = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    try:
        assert fresh.chat_chain("notes", REASONING_MODEL) == [REASONING_MODEL, MINISTRAL]
    finally:
        fresh.close()


def _warm_listing():
    """Charge la liste ``/v1/models`` du compte dans le cache web (une requête)."""
    client = lng._get_albert_client(FAKE_ALBERT_KEY, use_limiter=False)
    try:
        lng._albert_account_listing(client)
    finally:
        client.close()


def _generate_async(model):
    """Note asynchrone (``agenerate_with_albert``, rôle ``notes``)."""
    return asyncio.run(lng.agenerate_with_albert(
        "Invite.", model, semaphore=asyncio.Semaphore(5), temperature=0.2, mode="short",
        albert_api_key=FAKE_ALBERT_KEY,
    ))


def _generate_sync(model):
    """Note synchrone (``_generate_with_albert``, repli du client lui-même)."""
    return lng._generate_with_albert("Invite.", model, temperature=0.2, mode="short", albert_api_key=FAKE_ALBERT_KEY)


def _citation_async(model):
    """Filtre de citations asynchrone (``_albert_citation_call``, rôle ``citation``)."""
    return asyncio.run(cfilter._albert_citation_call(
        "Invite.", "albert/" + model, temperature=0.1, albert_api_key=FAKE_ALBERT_KEY,
    ))


WEB_PATHS = {
    "generate_async": _generate_async,
    "generate_sync": _generate_sync,
    "citation_async": _citation_async,
}


@pytest.mark.parametrize("path", sorted(WEB_PATHS))
def test_pinned_id_fallback_skips_models_absent_from_cached_listing(path, shared, monkeypatch):
    """Id épinglé surchargé, liste en cache : aucun repli vers un modèle absent du compte."""
    SlotSpy(monkeypatch)
    _warm_listing()
    shared.inject("POST", CHAT_PATH, *(["model_busy"] * BUSY_REPLIES))
    with pytest.raises(lng.AlbertModelBusy):
        WEB_PATHS[path](REASONING_MODEL)
    sent = _sent_models(shared)
    assert sent and sent[0] == REASONING_MODEL
    assert MINISTRAL not in sent
    assert _listing_requests(shared) == 1


@pytest.mark.parametrize("path", ["citation_async", "generate_async"])
def test_alias_cache_hit_fallback_skips_models_absent_from_listing(path, shared, monkeypatch):
    """Alias résolu par un succès de cache : les replis absents du compte restent exclus."""
    SlotSpy(monkeypatch)
    assert WEB_PATHS[path](REASONING_ALIAS) == NOTE_TEXT
    assert _listing_requests(shared) == 1
    shared.reset_calls()
    shared.inject("POST", CHAT_PATH, *(["model_busy"] * BUSY_REPLIES))
    with pytest.raises(lng.AlbertModelBusy):
        WEB_PATHS[path](REASONING_ALIAS)
    sent = _sent_models(shared)
    assert sent and sent[0] == REASONING_MODEL
    assert MINISTRAL not in sent
    assert _listing_requests(shared) == 0


# ---------------------------------------------------------------------------
# 3. Liste /v1/models chargée avant le premier repli, sans liste en cache
# ---------------------------------------------------------------------------
LATE_FALLBACK = "mistral-small-3-2-24b-instruct-2506"


def _citation_sync(model):
    """Filtre de citations synchrone (``_generate_with_albert``, rôle ``citation``)."""
    return lng._generate_with_albert(
        "Invite.", model, temperature=0.1, mode="short", albert_api_key=FAKE_ALBERT_KEY, role="citation",
    )


def _trace(double):
    """Suite ``(verbe, chemin)`` des requêtes reçues par le faux."""
    return [(call.method, call.path) for call in double.calls]


def test_first_fallback_loads_listing_just_before_it_async(shared, monkeypatch):
    """Chemin asynchrone : id épinglé surchargé deux fois, aucune liste en cache. La
    liste ``/v1/models`` est chargée juste avant le premier repli (fil de travail) :
    le repli absent du compte (ministral) n'est jamais envoyé, le second repli sert.
    Le job suivant réutilise la liste en cache : un seul ``GET /v1/models`` en tout."""
    SlotSpy(monkeypatch)
    for _job in range(2):
        shared.inject("POST", CHAT_PATH, "model_busy", "model_busy")
        assert _citation_async(REASONING_MODEL) == NOTE_TEXT
    assert _sent_models(shared) == [REASONING_MODEL, REASONING_MODEL, LATE_FALLBACK] * 2
    assert _trace(shared)[:4] == [("POST", CHAT_PATH), ("POST", CHAT_PATH), ("GET", MODELS_PATH), ("POST", CHAT_PATH)]
    assert _listing_requests(shared) == 1


def test_listing_not_requested_when_primary_answers_async(shared, monkeypatch):
    """Chemin asynchrone : le modèle en tête répond, aucun repli, donc aucune requête ``/v1/models``."""
    SlotSpy(monkeypatch)
    assert _citation_async(REASONING_MODEL) == NOTE_TEXT
    assert _listing_requests(shared) == 0


def test_first_fallback_loads_listing_just_before_it_sync(shared):
    """Chemin synchrone : même scénario. Le modèle en tête part seul (réessais du client
    compris) ; la liste n'est chargée qu'après ses deux 503, juste avant le premier
    repli ; le repli absent n'est jamais envoyé ; un seul ``GET /v1/models`` sur deux jobs."""
    for _job in range(2):
        shared.inject("POST", CHAT_PATH, "model_busy", "model_busy")
        assert _citation_sync(REASONING_MODEL) == NOTE_TEXT
    assert _sent_models(shared) == [REASONING_MODEL, REASONING_MODEL, LATE_FALLBACK] * 2
    assert _trace(shared)[:4] == [("POST", CHAT_PATH), ("POST", CHAT_PATH), ("GET", MODELS_PATH), ("POST", CHAT_PATH)]
    assert _listing_requests(shared) == 1


def test_listing_not_requested_when_primary_answers_sync(shared):
    """Chemin synchrone : le modèle en tête répond, donc aucune requête ``/v1/models``."""
    assert _citation_sync(REASONING_MODEL) == NOTE_TEXT
    assert _sent_models(shared) == [REASONING_MODEL]
    assert _listing_requests(shared) == 0


def test_listing_failure_keeps_unfiltered_chain_async(shared, monkeypatch):
    """``GET /v1/models`` en échec (hors erreur de compte) : la chaîne du catalogue est
    gardée telle quelle (comportement antérieur), le job n'échoue pas pour la liste."""
    SlotSpy(monkeypatch)
    shared.models = copy.deepcopy(load_fixture("P2_models.json")["body"]["data"])  # ministral servi
    shared.inject("GET", MODELS_PATH, *(["server_error"] * BUSY_REPLIES))
    shared.inject("POST", CHAT_PATH, "model_busy", "model_busy")
    assert _citation_async(REASONING_MODEL) == NOTE_TEXT
    assert _sent_models(shared) == [REASONING_MODEL, REASONING_MODEL, MINISTRAL]
    assert _listing_requests(shared) >= 1
