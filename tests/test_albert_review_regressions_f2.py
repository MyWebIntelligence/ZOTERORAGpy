"""Régressions de la revue Albert (correctifs F2), sans réseau ni identifiant réel.

* garde de client de la phase ``initial`` : elle suit le routage effectif du
  recodage durci (``RECODE_HARDEN_ENABLED``), pas le fournisseur brut de
  ``--model`` ; une garde franchie donne toujours un client utilisable ;
* aide morte ``albert_recode_startup_error`` retirée (point d'entrée unique :
  ``albert_recode_startup_exception``) ;
* chemin web (notes, filtre de citations) : un alias Albert est résolu par
  ``/v1/models``, jamais par la table figée du catalogue, et un id épinglé part
  sans requête supplémentaire ;
* ``run_llm_slot`` : les tâches Albert en attente du sémaphore Albert ne
  tiennent aucune place du sémaphore LLM global ;
* ``load_dotenv_guarded`` : les secrets serveur retirés de l'environnement
  non-admin ne sont jamais rechargés depuis le ``.env`` ;
* ``tests/test_albert_catalog.py`` lit ``decisions.json`` quel que soit le
  répertoire courant.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
SCRIPTS_DIR = os.path.join(RAGPY_ROOT, "scripts")
for _p in (RAGPY_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests import test_albert_off_golden as golden  # noqa: E402
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert, load_fixture  # noqa: E402
from tests.test_albert_notes import SleepRecorder, route_albert_to_fake  # noqa: E402
from tests.test_env_isolation import _user, _write_dotenv  # noqa: E402

from scripts import rad_env  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402

rc = golden.rc
lng = golden.lng
cfilter = golden.cfilter

CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
MINISTRAL = "ministral-3-8b-instruct-2512"
MISTRAL_SMALL = "mistral-small-3-2-24b-instruct-2506"
GEMMA = "gemma-4-31b-it"
MEDIUM_ALIAS = "openweight-medium"
CODE_ALIAS = "openweight-code"
NOTE_TEXT = "<p>Fiche factice.</p>"
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
SERVER_SECRETS = ("FLOWER_PASSWORD", "JWT_SECRET_KEY", "RESEND_API_KEY")
FAKE_SERVER_DOTENV = {
    "FLOWER_PASSWORD": "fake-flower-dotenv-0001",
    "JWT_SECRET_KEY": "fake-jwt-dotenv-0001",
    "RESEND_API_KEY": "fake-resend-dotenv-0001",
    "MISTRAL_OCR_MODEL": "fake-mistral-ocr-model-dotenv-0001",
}


# ---------------------------------------------------------------------------
# Environnement
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Retire identifiants et réglages Albert/recodage ; sémaphores, limiteurs et caches remis à zéro."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in vars(lng).items():
        if isinstance(value, asyncio.Semaphore):
            monkeypatch.setattr(lng, name, None)
    clear_listing = getattr(lng, "clear_albert_listing_cache", None)
    if callable(clear_listing):
        clear_listing()
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    if callable(clear_listing):
        clear_listing()
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


# ---------------------------------------------------------------------------
# 1. Garde de client de la phase initial (recodage durci)
# ---------------------------------------------------------------------------
class ChatDouble:
    """Client compatible OpenAI factice : chaque appel réussit (``finish_reason=stop``)."""

    def __init__(self):
        """Crée le double et son journal d'appels."""
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        """Enregistre les kwargs et renvoie une réponse de chat minimale."""
        self.calls.append(kwargs)
        message = SimpleNamespace(content="Texte recodé.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


# (nom, variables RECODE_*, client OpenAI présent, client OpenRouter présent, garde attendue)
GUARD_CASES = [
    ("harden_prefer_openai_default_only_openrouter", {"RECODE_HARDEN_ENABLED": "1"}, False, True, True),
    ("harden_prefer_openai_default_openai_present", {"RECODE_HARDEN_ENABLED": "1"}, True, False, False),
    ("harden_openrouter_pinned_slug", {
        "RECODE_HARDEN_ENABLED": "1", "RECODE_PREFER_OPENAI": "0",
        "RECODE_OPENROUTER_PROVIDER": "fake-provider", "RECODE_MODEL": OPENROUTER_MODEL,
    }, False, True, False),
    ("harden_openrouter_pinned_slug_no_client", {
        "RECODE_HARDEN_ENABLED": "1", "RECODE_PREFER_OPENAI": "0",
        "RECODE_OPENROUTER_PROVIDER": "fake-provider", "RECODE_MODEL": OPENROUTER_MODEL,
    }, False, False, True),
    ("harden_openrouter_not_pinned", {
        "RECODE_HARDEN_ENABLED": "1", "RECODE_PREFER_OPENAI": "0", "RECODE_MODEL": OPENROUTER_MODEL,
    }, False, True, True),
    ("harden_prefer_openrouter_openai_snapshot", {
        "RECODE_HARDEN_ENABLED": "1", "RECODE_PREFER_OPENAI": "0", "RECODE_OPENROUTER_PROVIDER": "fake-provider",
    }, False, True, True),
    ("historical_slug_openrouter_only", {}, False, True, False),
    ("historical_slug_openai_only", {}, True, False, False),
    ("historical_slug_no_client", {}, False, False, True),
]


@pytest.mark.parametrize("case,env,with_openai,with_openrouter,expected", GUARD_CASES,
                         ids=[c[0] for c in GUARD_CASES])
def test_initial_guard_follows_effective_recode_route(case, env, with_openai, with_openrouter, expected,
                                                      monkeypatch):
    """La garde ``initial`` exige le client que le recodage durci utilisera réellement."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(rc, "client", ChatDouble() if with_openai else None)
    monkeypatch.setattr(rc, "openrouter_client", ChatDouble() if with_openrouter else None)
    assert rc.missing_llm_client_for_phase("initial", OPENROUTER_MODEL) is expected


@pytest.mark.parametrize("case,env,with_openai,with_openrouter,expected", GUARD_CASES,
                         ids=[c[0] for c in GUARD_CASES])
def test_passed_guard_never_degrades_to_raw_text(case, env, with_openai, with_openrouter, expected, monkeypatch):
    """Garde franchie ⇒ le lot est recodé par un client réel (jamais de ``fallback_raw`` faute de client)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    openai_double = ChatDouble() if with_openai else None
    openrouter_double = ChatDouble() if with_openrouter else None
    monkeypatch.setattr(rc, "client", openai_double)
    monkeypatch.setattr(rc, "openrouter_client", openrouter_double)
    monkeypatch.setattr(rc.time, "sleep", lambda seconds: None)
    if rc.missing_llm_client_for_phase("initial", OPENROUTER_MODEL):
        return
    cfg = rc.rad_recode_cache.RecodeConfig.from_env()
    texts, statuses = rc.gpt_recode_batch(["Texte brut."], rc._RECODE_INSTRUCTIONS, model=OPENROUTER_MODEL,
                                          recode_cfg=cfg)
    assert statuses == ["recoded"], case
    assert texts == ["Texte recodé."]


# ---------------------------------------------------------------------------
# 2. Aide morte retirée
# ---------------------------------------------------------------------------
def test_dead_startup_error_helper_removed():
    """``albert_recode_startup_error`` (jamais appelée) n'existe plus ; l'exception reste l'unique contrôle."""
    assert not hasattr(rc, "albert_recode_startup_error")
    assert rc.albert_recode_startup_exception("gpt-4o-mini") is None
    with open(os.path.join(SCRIPTS_DIR, "rad_chunk.py"), encoding="utf-8") as fh:
        assert "albert_recode_startup_error" not in fh.read()


# ---------------------------------------------------------------------------
# 3. Chemin web : alias résolus par /v1/models
# ---------------------------------------------------------------------------
def _repointed_listing():
    """Liste ``/v1/models`` après le retrait de mistral-small : ``openweight-medium`` → gemma-4-31b-it.

    ``openweight-code`` n'y apparaît plus (qwen3-coder retiré, alias non encore repris).
    """
    models = copy.deepcopy(load_fixture("P2_models.json")["body"]["data"])
    kept = [m for m in models if m["id"] not in (MISTRAL_SMALL, "qwen3-coder-30b-a3b-instruct")]
    for model in kept:
        if model["id"] == GEMMA:
            model["aliases"] = list(model.get("aliases") or []) + [MEDIUM_ALIAS]
    return kept


@pytest.fixture
def repointed(monkeypatch):
    """Albert ON, ``FakeAlbert`` servant la liste réorientée ; tout ``AlbertClient`` y est branché."""
    fake = FakeAlbert(models=_repointed_listing())
    fake.chat_reply = NOTE_TEXT
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)
    return fake


def _sent_models(fake):
    """Modèles envoyés à ``/v1/chat/completions``, dans l'ordre."""
    return [call.json["model"] for call in fake.calls_to("POST", CHAT_PATH)]


def _run_generate_sync(alias):
    """Note synchrone (``_generate_with_albert``)."""
    return lng._generate_with_albert("Invite.", alias, temperature=0.2, mode="short", albert_api_key=FAKE_ALBERT_KEY)


def _run_generate_async(alias):
    """Note asynchrone (``agenerate_with_albert``, via ``run_llm_slot``)."""
    return asyncio.run(lng.agenerate_with_albert(
        "Invite.", alias, semaphore=asyncio.Semaphore(5), temperature=0.2, mode="short",
        albert_api_key=FAKE_ALBERT_KEY,
    ))


def _run_citation_async(alias):
    """Filtre de citations asynchrone (``_albert_citation_call``)."""
    return asyncio.run(cfilter._albert_citation_call(
        "Invite.", "albert/" + alias, temperature=0.1, albert_api_key=FAKE_ALBERT_KEY,
    ))


def _run_citation_sync(alias):
    """Filtre de citations synchrone (``_call_albert_api``)."""
    return cfilter._call_albert_api("Invite.", alias, temperature=0.1, albert_api_key=FAKE_ALBERT_KEY)


WEB_PATHS = {
    "generate_sync": _run_generate_sync,
    "generate_async": _run_generate_async,
    "citation_async": _run_citation_async,
    "citation_sync": _run_citation_sync,
}


@pytest.mark.parametrize("path", sorted(WEB_PATHS))
def test_web_alias_resolved_by_models_listing(path, repointed):
    """Alias réorienté : l'id servi par ``/v1/models`` part sur le fil, jamais l'id retiré du catalogue figé."""
    assert WEB_PATHS[path](MEDIUM_ALIAS) == NOTE_TEXT
    sent = _sent_models(repointed)
    assert sent == [GEMMA]
    assert MISTRAL_SMALL not in sent
    assert repointed.calls_to("GET", MODELS_PATH)


@pytest.mark.parametrize("path", sorted(WEB_PATHS))
def test_web_pinned_id_needs_no_listing(path, repointed):
    """Id épinglé : envoyé tel quel, sans requête ``/v1/models`` supplémentaire."""
    assert WEB_PATHS[path](MINISTRAL) == NOTE_TEXT
    assert _sent_models(repointed) == [MINISTRAL]
    assert repointed.calls_to("GET", MODELS_PATH) == []


@pytest.mark.parametrize("path", sorted(WEB_PATHS))
def test_web_alias_absent_from_listing_is_explicit_error(path, repointed):
    """Alias absent du compte : erreur explicite ``ModelNotFoundError``, aucun envoi de chat."""
    catalog = lng._albert_module("catalog")
    with pytest.raises(catalog.ModelNotFoundError):
        WEB_PATHS[path](CODE_ALIAS)
    assert _sent_models(repointed) == []


# ---------------------------------------------------------------------------
# 4. Ordre des sémaphores de run_llm_slot
# ---------------------------------------------------------------------------
def test_albert_waiters_hold_no_global_llm_slot(monkeypatch):
    """Dix envois Albert (2 places Albert) : 3 places globales restent libres pour les autres fournisseurs."""
    monkeypatch.setenv("ALBERT_NOTES_CONCURRENCY", "2")
    retry = lng._albert_module("retry")
    release = threading.Event()

    def thunk():
        """Envoi factice bloqué jusqu'à la fin des mesures."""
        release.wait(10)
        return "ok"

    async def scenario():
        """Lance les envois, mesure le sémaphore global, puis tente une acquisition « historique »."""
        glob = asyncio.Semaphore(5)
        policy = lng.AlbertSlotPolicy(retry=retry.RetryPolicy(), limiter=None)
        tasks = [asyncio.create_task(lng.run_llm_slot(glob, thunk, albert_policy=policy)) for _ in range(10)]
        try:
            await asyncio.sleep(0.2)
            free_global = glob._value
            try:
                await asyncio.wait_for(glob.acquire(), timeout=0.5)
            except asyncio.TimeoutError:
                acquired = False
            else:
                acquired = True
                glob.release()
        finally:
            release.set()
        results = await asyncio.gather(*tasks)
        return free_global, acquired, results

    free_global, acquired, results = asyncio.run(scenario())
    assert free_global == 3
    assert acquired is True
    assert results == ["ok"] * 10


# ---------------------------------------------------------------------------
# 5. Secrets serveur jamais rechargés du .env (non-admin)
# ---------------------------------------------------------------------------
def _probe(env, cwd, names):
    """``python -c`` : ``rad_env.load_dotenv_guarded()`` puis présence (booléens) de ``names``."""
    code = (
        "import sys, os, json\n"
        f"sys.path.insert(0, {SCRIPTS_DIR!r})\n"
        "import rad_env\n"
        "rad_env.load_dotenv_guarded()\n"
        f"print(json.dumps({{n: n in os.environ for n in {sorted(names)!r}}}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_subprocess_cannot_refill_server_secrets_from_dotenv(tmp_path):
    """Non-admin : JWT_SECRET_KEY, RESEND_API_KEY et FLOWER_PASSWORD restent absents après le chargement gardé."""
    from app.core import credentials as creds

    _write_dotenv(tmp_path, FAKE_SERVER_DOTENV)
    env = creds.build_subprocess_env(_user(is_admin=False))
    for name in FAKE_SERVER_DOTENV:
        env.pop(name, None)
    presence = _probe(env, tmp_path, FAKE_SERVER_DOTENV)
    assert presence == {"FLOWER_PASSWORD": False, "JWT_SECRET_KEY": False, "MISTRAL_OCR_MODEL": True,
                        "RESEND_API_KEY": False}


def test_admin_subprocess_still_reloads_server_secrets(tmp_path):
    """Admin (sans ``RAGPY_DOTENV_DENY``) : chargement ``dotenv`` inchangé, secrets serveur compris."""
    from app.core import credentials as creds

    _write_dotenv(tmp_path, FAKE_SERVER_DOTENV)
    env = creds.build_subprocess_env(_user(is_admin=True))
    assert rad_env.DOTENV_DENY_ENV_VAR not in env
    for name in FAKE_SERVER_DOTENV:
        env.pop(name, None)
    assert _probe(env, tmp_path, FAKE_SERVER_DOTENV) == {name: True for name in FAKE_SERVER_DOTENV}


def test_server_secret_names_match_credentials():
    """La liste des secrets serveur du chargeur gardé est celle que ``build_subprocess_env`` retire."""
    from app.core import credentials as creds

    assert tuple(rad_env.SERVER_SECRET_ENV_VARS) == tuple(creds.SERVER_SECRET_ENV_VARS) == SERVER_SECRETS


# ---------------------------------------------------------------------------
# 6. Catalogue : decisions.json indépendant du répertoire courant
# ---------------------------------------------------------------------------
def test_catalog_decisions_read_from_any_cwd(monkeypatch, tmp_path):
    """``_decisions()`` de ``test_albert_catalog`` fonctionne depuis un répertoire courant étranger."""
    module = importlib.import_module("tests.test_albert_catalog")
    monkeypatch.chdir(tmp_path)
    decisions = module._decisions()
    assert "D2" in decisions
    assert module.FIXTURES.is_absolute()
