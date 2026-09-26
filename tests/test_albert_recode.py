"""Tests du recodage Albert de ``rad_chunk`` (lot 3, tâches 1 à 5).

Deux familles :

* **chemin OFF et fournisseurs historiques** : les goldens G2, G4 et G12 sont
  rejoués **en lecture seule** (jamais régénérés ici) avec une clé Albert
  factice présente dans l'environnement, Albert désactivé puis activé pour des
  modèles sans préfixe ``albert/`` : sorties, clés de cache et kwargs
  identiques à l'octet (invariants 1 et 4) ;
* **chemin Albert** : ``albert/<id>`` est routé vers ``tests/albert_fakes.FakeAlbert``
  (``httpx.MockTransport`` injecté dans tout ``AlbertClient`` construit pendant
  le test ; aucun réseau, clé ``FAKE_ALBERT_KEY`` seulement). On vérifie le
  modèle envoyé sur le fil, l'absence de repli silencieux vers OpenAI ou
  OpenRouter (invariant 19), la correspondance des statuts, le cache jamais
  pollué (invariant 23), l'arrêt du job sur erreur de compte (invariant 38),
  la concurrence plafonnée par processus (invariant 40), la traçabilité
  (invariant 31) et la couche de retry unique (invariant 37).

Les sommeils des réessais sont enregistrés (aucune attente réelle) et les
clients OpenAI/OpenRouter du module sont remplacés par des doubles qui
journalisent chaque appel.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import json
import logging
import os
import runpy
import sqlite3
import sys
import threading
import time
from datetime import date
from types import SimpleNamespace

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Le harnais golden importe ``rad_chunk`` (nom de module du CLI) avec une clé
# OpenAI factice ; on réutilise le même objet module et ses aides, sans jamais
# réécrire un golden.
from tests import test_albert_off_golden as golden  # noqa: E402

from scripts.rad_albert import catalog as albert_catalog  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402
from scripts.rad_albert.config import AlbertConfig  # noqa: E402
from scripts.rad_albert.errors import AlbertAuthError  # noqa: E402
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert  # noqa: E402

rc = golden.rc
rad_recode_cache = rc.rad_recode_cache
rad_dedup = rc.rad_dedup
# Fonction de clé d'origine, capturée avant tout espion.
_REAL_RECODE_KEY = rad_recode_cache.recode_key

GOLDEN_SCRIPTS_DIR = os.path.join(_THIS, "fixtures", "albert", "golden_off", "scripts")
CHAT_PATH = "/v1/chat/completions"
PRIMARY_RECODE = "ministral-3-8b-instruct-2512"
ALBERT_PRIMARY = "albert/" + PRIMARY_RECODE
REASONING_MODEL = "gpt-oss-120b"
RAW_SHORT = golden.G4_RAW
# Débits très élevés : le limiteur proactif ne fait jamais attendre un test.
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
# Constantes de module du CLI susceptibles de porter l'interrupteur Albert
# (lecture à l'import, décision 11) : forcées à ON par ``albert_on``.
MODULE_SWITCHES = ("ALBERT_ENABLED",)
MODULE_KEYS = ("ALBERT_API_KEY",)


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
class SleepRecorder:
    """Sommeil factice : enregistre les durées demandées sans attendre."""

    def __init__(self):
        """Crée un enregistreur vide."""
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, seconds):
        """Enregistre une demande de sommeil de ``seconds`` secondes."""
        with self._lock:
            self.calls.append(float(seconds))


def _golden_bytes(name):
    """Contenu d'un golden ``scripts/`` (lecture seule)."""
    with open(os.path.join(GOLDEN_SCRIPTS_DIR, name), "rb") as fh:
        return fh.read()


def _assert_matches_golden_bytes(name, data):
    """Compare ``data`` au golden ``name`` sans jamais le réécrire (diff lisible en cas d'écart)."""
    expected = _golden_bytes(name)
    if expected != data:
        diff = difflib.unified_diff(
            expected.decode("utf-8", "replace").splitlines(True),
            data.decode("utf-8", "replace").splitlines(True),
            fromfile=f"golden/{name}", tofile="actuel", n=2,
        )
        pytest.fail("Écart avec le golden " + name + " :\n" + "".join(list(diff)[:200]), pytrace=False)


def _assert_matches_golden_json(name, obj):
    """Compare le rendu JSON de ``obj`` (format du harnais golden) au golden ``name``."""
    _assert_matches_golden_bytes(name, golden._dump(obj).encode("utf-8"))


def _client_classes():
    """Classes ``AlbertClient`` (import ``scripts.rad_albert`` et, s'il est chargé, import CLI ``rad_albert``).

    Le module client est importé ici : les implémentations le chargent
    paresseusement, après l'installation du double. La racine dont ``rad_chunk``
    tire ses sous-modules (``_rad_albert``) est importée aussi, pour que le
    client chargé à la demande soit celui qui est branché.
    """
    importlib.import_module("scripts.rad_albert.client")
    root = getattr(getattr(rc, "_rad_albert", None), "__name__", None)
    if root:
        importlib.import_module(f"{root}.client")
    classes = []
    for name in ("scripts.rad_albert.client", "rad_albert.client"):
        module = sys.modules.get(name)
        cls = getattr(module, "AlbertClient", None) if module is not None else None
        if cls is not None and cls not in classes:
            classes.append(cls)
    return classes


def route_albert_to_fake(monkeypatch, fake, sleep):
    """Branche tout ``AlbertClient`` construit pendant le test sur ``fake`` (transport et sommeil).

    Le transport n'est remplacé que s'il n'est pas fourni ; le sommeil injecté
    enregistre les attentes sans dormir.
    """
    for cls in _client_classes():
        original = cls.__init__

        def patched(self, *args, __original=original, **kwargs):
            """Constructeur d'origine avec le transport du faux et le sommeil enregistreur."""
            if kwargs.get("transport") is None:
                kwargs["transport"] = fake.transport
            kwargs.setdefault("sleep", sleep)
            __original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", patched)


def _set_module_attr_if_present(monkeypatch, module, name, value):
    """Pose ``module.name = value`` (monkeypatch) seulement si l'attribut existe."""
    if hasattr(module, name):
        monkeypatch.setattr(module, name, value)


def _reset_rc_albert_state():
    """Oublie le client Albert en cache et l'arrêt de job mémorisé du module."""
    reset = getattr(rc, "reset_albert_state", None)
    if callable(reset):
        reset()


def _abort_error():
    """Erreur qui a arrêté le recodage Albert du processus (``None`` si aucun arrêt)."""
    abort = getattr(rc, "_ALBERT_ABORT", None)
    if hasattr(abort, "is_set"):
        return getattr(rc, "_ALBERT_ABORT_ERROR", None) if abort.is_set() else None
    return abort


def _echo_reply(body):
    """Réponse de chat : le chunk reçu, préfixé (rapport de longueur ≈ 1)."""
    user = body["messages"][-1]["content"]
    chunk = user.split("Texte à recoder :\n", 1)[-1].rsplit("\n\nTexte recodé :", 1)[0]
    return "Recodé. " + chunk


def _chat_calls(fake):
    """Appels de chat reçus par le faux."""
    return fake.calls_to("POST", CHAT_PATH)


def _raw_chunks(n=2):
    """``n`` chunks bruts distincts, assez longs pour la garde de ratio."""
    return [f"Chunk {i} : " + RAW_SHORT for i in range(1, n + 1)]


def _cache_rows(path):
    """Nombre total de lignes du cache SQLite (0 si le fichier n'existe pas)."""
    if not os.path.exists(path):
        return 0
    conn = sqlite3.connect(path)
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return sum(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables)
    finally:
        conn.close()


def _read_chunks(path):
    """Chunks d'un fichier ``*_chunks.json``."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _expected_fallback(role="recode"):
    """Premier modèle de repli du rôle à la date du jour (chaîne du catalogue)."""
    return albert_catalog.fallback_chain(role, today=date.today())[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Retire les identifiants et réglages de cache ou de dédup ; état Albert du processus remis à zéro."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    _reset_rc_albert_state()
    yield
    _reset_rc_albert_state()
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def chunk_env(monkeypatch):
    """``rad_chunk`` figé (workers, lots, doc_id) avec des clients OpenAI/OpenRouter enregistreurs."""
    monkeypatch.setattr(rc, "DEFAULT_MAX_WORKERS", 1)
    monkeypatch.setattr(rc, "DEFAULT_DOC_WORKERS", 3)
    monkeypatch.setattr(rc, "DEFAULT_BATCH_SIZE_GPT", 5)
    monkeypatch.setattr(rc, "DEFAULT_EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(rc, "random", golden._RandomProxy())
    sleeps = []
    monkeypatch.setattr(rc, "time", golden._TimeProxy(sleeps))
    calls = []
    monkeypatch.setattr(rc, "client", golden._FakeLLMClient("openai", calls))
    monkeypatch.setattr(rc, "openrouter_client", golden._FakeLLMClient("openrouter", calls))
    return SimpleNamespace(calls=calls, legacy_sleeps=sleeps)


def _albert_env(monkeypatch, enabled):
    """Pose une clé Albert factice (et les débits élevés) ; Albert ON ou OFF."""
    monkeypatch.setenv("ALBERT_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)
    for name in MODULE_SWITCHES:
        _set_module_attr_if_present(monkeypatch, rc, name, bool(enabled))
    for name in MODULE_KEYS:
        _set_module_attr_if_present(monkeypatch, rc, name, FAKE_ALBERT_KEY)


@pytest.fixture
def albert_off_with_key(monkeypatch, chunk_env):
    """Albert OFF mais clé factice présente ; tout client Albert construit irait au faux."""
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=False)
    return SimpleNamespace(fake=fake, calls=chunk_env.calls)


@pytest.fixture
def albert_on(monkeypatch, chunk_env):
    """Albert ON avec clé factice ; ``FakeAlbert`` répond en écho du chunk."""
    fake = FakeAlbert()
    fake.chat_reply = _echo_reply
    sleeps = SleepRecorder()
    route_albert_to_fake(monkeypatch, fake, sleeps)
    _albert_env(monkeypatch, enabled=True)
    return SimpleNamespace(fake=fake, sleeps=sleeps, calls=chunk_env.calls,
                           legacy_sleeps=chunk_env.legacy_sleeps)


# ---------------------------------------------------------------------------
# Chemin OFF et fournisseurs historiques : goldens rejoués en lecture seule
# ---------------------------------------------------------------------------
def _g4_cells(monkeypatch, tmp_path, calls):
    """Rejoue la matrice G4 (modèle × prefer_openai × harden) ; renvoie les cellules du golden."""
    cache_mod = rad_recode_cache
    real_recode_key = cache_mod.recode_key
    real_embed_key = cache_mod.embed_key
    key_calls = []

    def recode_key_spy(*args, **kwargs):
        """Enregistre les arguments connus et la valeur de ``recode_key``."""
        value = real_recode_key(*args, **kwargs)
        bound = golden._bind_known(args, kwargs, golden.G4_RECODE_KEY_ARGS)
        key_calls.append(("recode", {"content_hash": bound.get("content_hash"),
                                     "model": bound.get("model"),
                                     "provider": bound.get("provider"),
                                     "prompt_version": bound.get("prompt_version"),
                                     "decode_params": bound.get("decode_params_json")}, value))
        return value

    def embed_key_spy(*args, **kwargs):
        """Enregistre les arguments connus et la valeur de ``embed_key``."""
        value = real_embed_key(*args, **kwargs)
        bound = golden._bind_known(args, kwargs, golden.G4_EMBED_KEY_ARGS)
        key_calls.append(("embed", {"text_sha12": golden._fingerprint(bound.get("text")),
                                    "embed_model": bound.get("embed_model"),
                                    "embed_params": bound.get("embed_params_json")}, value))
        return value

    monkeypatch.setattr(cache_mod, "recode_key", recode_key_spy)
    monkeypatch.setattr(cache_mod, "embed_key", embed_key_spy)
    cells = []
    cell_no = 0
    for model in golden.G4_MODELS:
        for prefer_openai in (0, 1):
            for harden in (0, 1):
                cell_no += 1
                key_calls.clear()
                calls.clear()
                cell_dir = tmp_path / f"cell_{cell_no:02d}"
                cell_dir.mkdir()
                monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
                monkeypatch.setenv("RECODE_EMBED_CACHE_ENABLED", "1")
                monkeypatch.setenv("RECODE_CACHE_PATH", str(cell_dir / "cache.sqlite"))
                monkeypatch.setenv("RECODE_PREFER_OPENAI", str(prefer_openai))
                monkeypatch.setenv("RECODE_HARDEN_ENABLED", str(harden))
                _df, row = golden._golden_row(cell_dir, text=golden.G4_RAW)
                chunks_json = cell_dir / "output_chunks.json"
                rc.process_document_chunks(row, json_file=str(chunks_json), model=model)
                rc.generate_and_save_embeddings(str(chunks_json), str(cell_dir / "output_dense.json"))
                statuses = [chunk.get("recode_status") for chunk in _read_chunks(chunks_json)]
                recode = [c for c in key_calls if c[0] == "recode"]
                embed = [c for c in key_calls if c[0] == "embed"]
                chat = [c for c in calls if c["endpoint"] == "chat.completions.create"]
                cells.append({
                    "model": model,
                    "prefer_openai": prefer_openai,
                    "harden": harden,
                    "key_model": recode[0][1]["model"],
                    "provider_label": recode[0][1]["provider"],
                    "recode_key": recode[0][2],
                    "recode_key_args": recode[0][1],
                    "routed_client": chat[0]["client"] if chat else None,
                    "sent_model": chat[0]["kwargs"]["model"] if chat else None,
                    "statuses": statuses,
                    "embed_key": embed[0][2],
                    "embed_key_args": embed[0][1],
                })
    return cells


@pytest.mark.parametrize("enabled", [False, True], ids=["albert_off", "albert_on_non_albert_models"])
def test_off_keys_match_golden(enabled, chunk_env, monkeypatch, tmp_path):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled)
    cells = _g4_cells(monkeypatch, tmp_path, chunk_env.calls)
    _assert_matches_golden_json("g4_recode_embed_keys.json", cells)
    assert fake.calls == []


@pytest.mark.parametrize("enabled", [False, True], ids=["albert_off", "albert_on_non_albert_models"])
@pytest.mark.parametrize("case,model,with_openrouter", golden.G2_CASES, ids=[c[0] for c in golden.G2_CASES])
def test_off_create_kwargs_match_golden(case, model, with_openrouter, enabled, chunk_env, monkeypatch, tmp_path):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled)
    if not with_openrouter:
        monkeypatch.setattr(rc, "openrouter_client", None)
    _df, row = golden._golden_row(tmp_path)
    json_file = tmp_path / "output_chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model=model)
    _assert_matches_golden_bytes(f"g2_chunks_{case}.json", json_file.read_bytes())
    _assert_matches_golden_json(f"g2_create_kwargs_{case}.json", chunk_env.calls)
    assert fake.calls == []


def test_off_stdout_matches_golden(albert_off_with_key, chunk_env, tmp_path, capsys, monkeypatch):
    df, row = golden._golden_row(tmp_path)
    capsys.readouterr()
    rc.process_document_chunks(row, json_file=str(tmp_path / "direct_chunks.json"), model="gpt-4o-mini")
    direct = golden._stdout_lines(capsys.readouterr().out, tmp_path)
    rc.process_all_documents(df, json_file=str(tmp_path / "output_chunks.json"), model="google/gemini-2.5-flash")
    all_docs = golden._stdout_lines(capsys.readouterr().out, tmp_path)
    input_path = golden._write_chunks_file(tmp_path)
    rc.generate_and_save_embeddings(input_path, str(tmp_path / "output_chunks_with_embeddings.json"))
    dense = golden._stdout_lines(capsys.readouterr().out, tmp_path)
    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    _skip_df, skip_row = golden._golden_row(skip_dir, provider="mistral")
    rc.process_document_chunks(skip_row, json_file=str(skip_dir / "skip_chunks.json"), model="gpt-4o-mini")
    skipped = golden._stdout_lines(capsys.readouterr().out, tmp_path)
    monkeypatch.setattr(rc, "openrouter_client", None)
    rc.process_document_chunks(row, json_file=str(tmp_path / "fallback_chunks.json"),
                               model="google/gemini-2.5-flash")
    fallback = golden._stdout_lines(capsys.readouterr().out, tmp_path)
    _assert_matches_golden_json("g12_rad_chunk_stdout.json", {
        "process_document_chunks_gpt-4o-mini": direct,
        "process_all_documents_google/gemini-2.5-flash": all_docs,
        "generate_and_save_embeddings": dense,
        "process_document_chunks_skip_recode_mistral": skipped,
        "process_document_chunks_google/gemini-2.5-flash_without_openrouter_client": fallback,
    })
    assert albert_off_with_key.fake.calls == []


# ---------------------------------------------------------------------------
# Chemin Albert : routage, souveraineté, statuts, cache
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested", [ALBERT_PRIMARY, "Albert/" + PRIMARY_RECODE, "ALBERT/" + PRIMARY_RECODE])
def test_albert_prefix_routes_wire_model(requested, albert_on):
    raws = _raw_chunks(2)
    texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=requested)
    chats = _chat_calls(albert_on.fake)
    assert len(chats) == 2
    for call in chats:
        body = call.json
        assert body["model"] == PRIMARY_RECODE
        assert "user" not in body
        assert call.headers.get("authorization") == "Bearer " + FAKE_ALBERT_KEY
        assert body["messages"][0] == {"role": "system", "content": rc._RECODE_SYSTEM}
        assert body["messages"][1]["role"] == "user"
    sent_prompts = sorted(call.json["messages"][1]["content"] for call in chats)
    expected_prompts = sorted(
        rc._RECODE_TEMPLATE.format(instructions=rc._RECODE_INSTRUCTIONS, chunk=raw) for raw in raws
    )
    assert sent_prompts == expected_prompts
    assert statuses == ["recoded", "recoded"]
    assert texts == ["Recodé. " + raw for raw in raws]
    assert albert_on.calls == []


def test_effective_recode_model_is_single_source(albert_on):
    assert callable(getattr(rc, "effective_recode_model", None))
    params = list(inspect.signature(rc.effective_recode_model).parameters)
    assert params[:2] == ["cli_model", "cfg"]


@pytest.mark.parametrize("failure", ["server_error", "model_busy", "not_found", "invalid_key"])
def test_no_silent_openai_fallback(failure, albert_on, capsys, monkeypatch):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *([failure] * 60))
    raws = _raw_chunks(2)
    texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY)
    assert albert_on.calls == []
    assert statuses == ["fallback_raw", "fallback_raw"]
    assert texts == raws
    out = capsys.readouterr().out
    assert "gpt-4o-mini" not in out
    # Même sans aucun client OpenAI ni OpenRouter, le chemin Albert ne casse pas.
    monkeypatch.setattr(rc, "client", None)
    monkeypatch.setattr(rc, "openrouter_client", None)
    fake.clear_injections()
    fake.inject("POST", CHAT_PATH, *([failure] * 60))
    texts2, statuses2 = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY)
    assert statuses2 == ["fallback_raw", "fallback_raw"]
    assert texts2 == raws


def test_length_is_fallback_truncated_not_cached(albert_on, monkeypatch, tmp_path):
    cache_path = str(tmp_path / "cache.sqlite")
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", cache_path)
    fake = albert_on.fake
    raws = _raw_chunks(3)
    # Tronqué avec contenu, filtré, tronqué sans contenu (AlbertTruncatedError côté client).
    replies = {
        raws[0]: {"content": "Recodé partiel. " + raws[0], "finish_reason": "length"},
        raws[1]: {"content": "Recodé filtré. " + raws[1], "finish_reason": "content_filter"},
        raws[2]: {"content": None, "finish_reason": "length"},
    }

    def reply(body):
        """Réponse tronquée propre à chaque chunk."""
        user = body["messages"][-1]["content"]
        for raw, answer in replies.items():
            if raw in user:
                return answer
        raise AssertionError("chunk inattendu")

    fake.chat_reply = reply
    texts, statuses = rc.recode_batch_cached(raws, rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    assert statuses == ["fallback_truncated"] * 3
    assert texts == raws
    first_calls = len(_chat_calls(fake))
    assert first_calls == 3
    # Rien n'a été mis en cache : un second passage refait les trois appels.
    fake.chat_reply = _echo_reply
    texts2, statuses2 = rc.recode_batch_cached(raws, rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    assert statuses2 == ["recoded"] * 3
    assert len(_chat_calls(fake)) == first_calls + 3
    assert albert_on.calls == []


@pytest.mark.parametrize("ratio_case", ["too_short", "too_long"])
def test_suspicious_ratio_not_cached(ratio_case, albert_on, monkeypatch, tmp_path):
    cache_path = str(tmp_path / "cache.sqlite")
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", cache_path)
    fake = albert_on.fake
    raw = _raw_chunks(1)[0]

    def suspicious(body):
        """Réponse hors de l'intervalle de ratio [0.3, 1.5]."""
        if ratio_case == "too_short":
            return raw[: max(1, int(len(raw) * 0.2))]
        return raw + " " + raw

    fake.chat_reply = suspicious
    texts, statuses = rc.recode_batch_cached([raw], rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    assert statuses == ["fallback_raw"]
    assert texts == [raw]
    assert _cache_rows(cache_path) == 0
    # Une réponse dans l'intervalle est recodée et mise en cache ; le HIT n'appelle plus Albert.
    fake.chat_reply = _echo_reply
    texts2, statuses2 = rc.recode_batch_cached([raw], rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    assert statuses2 == ["recoded"]
    calls_before = len(_chat_calls(fake))
    texts3, statuses3 = rc.recode_batch_cached([raw], rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    assert statuses3 == ["cached"]
    assert texts3 == texts2
    assert len(_chat_calls(fake)) == calls_before


def test_ratio_guard_is_albert_only(chunk_env, monkeypatch):
    raw = _raw_chunks(1)[0]
    short_calls = []

    def short_content(label, kwargs):
        """Réponse très courte (ratio < 0,3) du double OpenAI."""
        return "Court."

    monkeypatch.setattr(rc, "client", golden._FakeLLMClient("openai", short_calls, short_content))
    texts, statuses = rc.gpt_recode_batch([raw], rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
    assert statuses == ["recoded"]
    assert texts == ["Court."]


def test_reasoning_content_never_used(albert_on):
    fake = albert_on.fake
    raws = _raw_chunks(2)
    leaked = "RAISONNEMENT INTERNE : je dois d'abord analyser la consigne. " * 3

    def reply(body):
        """Contenu vide (premier chunk) ou valide (second), avec un raisonnement non vide."""
        user = body["messages"][-1]["content"]
        if raws[0] in user:
            return {"content": None, "finish_reason": "stop", "reasoning": leaked}
        return {"content": "Recodé. " + raws[1], "finish_reason": "stop", "reasoning": leaked}

    fake.chat_reply = reply
    texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model="albert/" + REASONING_MODEL)
    assert statuses == ["fallback_raw", "recoded"]
    assert texts == [raws[0], "Recodé. " + raws[1]]
    assert all("RAISONNEMENT INTERNE" not in text for text in texts)
    for call in _chat_calls(fake):
        assert call.json["model"] == REASONING_MODEL
        assert "reasoning_effort" in call.json


def _spy_recode_keys(monkeypatch):
    """Espionne ``recode_key`` : arguments connus (hash, modèle, provider, prompt, dp) de chaque appel."""
    seen = []
    real = rad_recode_cache.recode_key

    def spy(*args, **kwargs):
        """Enregistre les arguments connus puis calcule la vraie clé."""
        bound = golden._bind_known(args, kwargs, golden.G4_RECODE_KEY_ARGS)
        seen.append(bound)
        return real(*args, **kwargs)

    monkeypatch.setattr(rad_recode_cache, "recode_key", spy)
    return seen


@pytest.fixture
def recode_key_spy(monkeypatch):
    """Liste des appels à ``recode_key`` pendant le test (arguments liés par nom)."""
    return _spy_recode_keys(monkeypatch)


def test_503_twice_fallback_recorded_in_key(albert_on, monkeypatch, tmp_path, recode_key_spy):
    cache_path = str(tmp_path / "cache.sqlite")
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", cache_path)
    fake = albert_on.fake
    fallback = _expected_fallback()
    fake.inject("POST", CHAT_PATH, "model_busy", "model_busy")
    _df, row = golden._golden_row(tmp_path, text=RAW_SHORT)
    json_file = tmp_path / "output_chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model=ALBERT_PRIMARY)
    models = [call.json["model"] for call in _chat_calls(fake)]
    assert models == [PRIMARY_RECODE, PRIMARY_RECODE, fallback]
    chunks = _read_chunks(json_file)
    assert [c["recode_status"] for c in chunks] == ["recoded"]
    assert [c["recode_model"] for c in chunks] == ["albert/" + fallback]
    # La valeur recodée est rangée sous la clé du modèle SERVI (repli), fournisseur « albert ».
    served = [b for b in recode_key_spy if b.get("model") == "albert/" + fallback]
    assert served and all(b["provider"] == "albert" for b in served)
    cfg = rad_recode_cache.RecodeConfig.from_env()
    stored = rad_recode_cache.get_recode(cfg, _REAL_RECODE_KEY(**served[-1]))
    assert stored == chunks[0]["text"]
    primary = [b for b in recode_key_spy if b.get("model") == ALBERT_PRIMARY]
    for bound in primary:
        assert rad_recode_cache.get_recode(cfg, _REAL_RECODE_KEY(**bound)) is None


@pytest.mark.parametrize("requested", ["albert/modele-disparu-2026", ALBERT_PRIMARY],
                         ids=["absent_from_models", "chat_404"])
def test_404_permanent_abort_message(requested, albert_on, capsys, caplog):
    fake = albert_on.fake
    fake.inject("POST", CHAT_PATH, *(["not_found"] * 20))
    raws = _raw_chunks(2)
    with caplog.at_level("WARNING"):
        texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=requested)
    chats = _chat_calls(fake)
    wire = requested.split("/", 1)[1]
    # Permanent : aucun réessai, aucun repli de rôle vers un autre modèle (un id
    # absent de /v1/models est même détecté avant tout envoi).
    assert len(chats) <= len(raws)
    assert {call.json["model"] for call in chats} <= {wire}
    assert statuses == ["fallback_raw", "fallback_raw"]
    assert texts == raws
    shown = capsys.readouterr().out + caplog.text
    assert "/v1/models" in shown
    assert albert_on.calls == []
    # L'erreur permanente arrête le recodage du job : le lot suivant n'appelle plus Albert.
    before = len(_chat_calls(fake))
    texts2, statuses2 = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=requested)
    assert statuses2 == ["fallback_raw", "fallback_raw"]
    assert len(_chat_calls(fake)) == before


def test_account_error_abort_exit_2(albert_on, monkeypatch, tmp_path):
    fake = albert_on.fake
    monkeypatch.setattr(rc, "DEFAULT_BATCH_SIZE_GPT", 1)
    fake.inject("POST", CHAT_PATH, *(["invalid_key"] * 50))
    df, row = golden._golden_row(tmp_path)
    json_file = tmp_path / "output_chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model=ALBERT_PRIMARY)
    chunks = _read_chunks(json_file)
    assert len(chunks) >= 2
    # Un seul envoi (le premier lot) : les lots restants n'appellent plus Albert.
    assert len(_chat_calls(fake)) == 1
    assert all(c["recode_status"] == "fallback_raw" for c in chunks)
    assert [c["text"] for c in chunks] == [rc.TEXT_SPLITTER.split_text(row["texteocr"].strip())[i]
                                           for i in range(len(chunks))]
    assert albert_on.calls == []
    assert _abort_error() is not None
    assert isinstance(_abort_error(), AlbertAuthError)


def _write_cli_csv(tmp_path, docs=2):
    """CSV d'entrée du CLI : ``docs`` documents au texte golden."""
    import csv

    path = tmp_path / "input.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "title", "texteocr", "texteocr_provider"])
        for i in range(docs):
            writer.writerow([f"doc{i}.pdf", f"Document {i}", golden._golden_ocr_text(), "legacy"])
    return str(path)


@pytest.mark.parametrize("failure", ["invalid_key", "account_expired", "budget_exhausted"])
def test_account_error_abort_exit_2_cli(failure, monkeypatch, tmp_path):
    fake = FakeAlbert()
    fake.chat_reply = _echo_reply
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=True)
    for name in golden.BASELINE_ENV_NAMES:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("DEFAULT_MAX_WORKERS", "1")
    monkeypatch.setenv("DEFAULT_DOC_WORKERS", "1")
    for module_name in ("scripts.rad_env", "rad_env"):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "load_dotenv_guarded", lambda *a, **k: False)
    fake.inject("POST", CHAT_PATH, *([failure] * 50))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    argv = ["rad_chunk.py", "--input", _write_cli_csv(tmp_path), "--output", str(out_dir),
            "--phase", "initial", "--model", ALBERT_PRIMARY]
    monkeypatch.setattr(sys, "argv", argv)
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(os.path.join(RAGPY_ROOT, "scripts", "rad_chunk.py"), run_name="__main__")
    finally:
        for handler in list(root_logger.handlers):
            if handler not in handlers_before:
                root_logger.removeHandler(handler)
                handler.close()
    assert excinfo.value.code == 2
    assert len(_chat_calls(fake)) == 1


def test_usage_ledger_only_if_albert_called(albert_on, tmp_path, capsys):
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    # Aucun appel Albert : ni fichier ni ligne de synthèse.
    assert rc.report_albert_usage(str(off_dir)) is None
    assert os.listdir(off_dir) == []
    assert "Albert :" not in capsys.readouterr().out
    rc.gpt_recode_batch(_raw_chunks(1), rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY)
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    path = rc.report_albert_usage(str(on_dir))
    assert path is not None and os.path.basename(path) == "albert_usage.jsonl"
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert [r.get("model") for r in records] == [PRIMARY_RECODE]
    assert "Albert :" in capsys.readouterr().out


def test_cache_key_uses_albert_provider_and_resolved_id(albert_on, monkeypatch, tmp_path, recode_key_spy):
    cache_path = str(tmp_path / "cache.sqlite")
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", cache_path)
    fake = albert_on.fake
    for requested in ("albert/openweight-small", ALBERT_PRIMARY):
        recode_key_spy.clear()
        fake.reset_calls()
        _df, row = golden._golden_row(tmp_path, text=RAW_SHORT + " " + requested)
        rc.process_document_chunks(row, json_file=str(tmp_path / "chunks.json"), model=requested)
        assert recode_key_spy, requested
        for bound in recode_key_spy:
            assert bound["provider"] == "albert"
            assert bound["model"] == ALBERT_PRIMARY
            dp = json.loads(bound["decode_params_json"])
            assert dp != json.loads(rad_recode_cache.RecodeConfig.from_env().decode_params_json())
        # Seul l'id épinglé part sur le fil (jamais l'alias).
        assert {call.json["model"] for call in _chat_calls(fake)} == {PRIMARY_RECODE}
    # Les clés OpenAI sont inchangées : même modèle et même libellé qu'avant.
    recode_key_spy.clear()
    _df, row = golden._golden_row(tmp_path, text=RAW_SHORT + " openai")
    rc.process_document_chunks(row, json_file=str(tmp_path / "chunks_openai.json"), model="gpt-4o-mini")
    assert [(b["model"], b["provider"]) for b in recode_key_spy] == [("gpt-4o-mini", "openai")]
    assert recode_key_spy[0]["decode_params_json"] == rad_recode_cache.RecodeConfig.from_env().decode_params_json()


def test_harden_albert_ignores_prefer_openai(albert_on, monkeypatch, tmp_path):
    monkeypatch.setenv("RECODE_HARDEN_ENABLED", "1")
    monkeypatch.setenv("RECODE_PREFER_OPENAI", "1")
    cfg = rad_recode_cache.RecodeConfig.from_env()
    assert rc.effective_recode_model(ALBERT_PRIMARY, cfg) == ALBERT_PRIMARY
    raws = _raw_chunks(1)
    texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY, recode_cfg=cfg)
    assert statuses == ["recoded"]
    assert [call.json["model"] for call in _chat_calls(albert_on.fake)] == [PRIMARY_RECODE]
    assert albert_on.calls == []


def test_harden_cli_albert_routes_and_keys_consistent(albert_on, monkeypatch, tmp_path, recode_key_spy):
    fake = albert_on.fake
    cell = 0
    for harden in (0, 1):
        for prefer_openai in (0, 1):
            for cache in (0, 1):
                cell += 1
                cell_dir = tmp_path / f"cell_{cell:02d}"
                cell_dir.mkdir()
                monkeypatch.setenv("RECODE_HARDEN_ENABLED", str(harden))
                monkeypatch.setenv("RECODE_PREFER_OPENAI", str(prefer_openai))
                monkeypatch.setenv("RECODE_CACHE_ENABLED", str(cache))
                monkeypatch.setenv("RECODE_CACHE_PATH", str(cell_dir / "cache.sqlite"))
                cfg = rad_recode_cache.RecodeConfig.from_env()
                assert rc.effective_recode_model(ALBERT_PRIMARY, cfg) == ALBERT_PRIMARY
                assert rc.effective_recode_model("gpt-4o-mini", cfg) == (cfg.model if harden else "gpt-4o-mini")
                recode_key_spy.clear()
                fake.reset_calls()
                albert_on.calls.clear()
                _df, row = golden._golden_row(cell_dir, text=RAW_SHORT + f" cellule {cell}")
                chunks_json = cell_dir / "chunks.json"
                rc.process_document_chunks(row, json_file=str(chunks_json), model=ALBERT_PRIMARY)
                # Routage : Albert seul, id épinglé.
                assert [call.json["model"] for call in _chat_calls(fake)] == [PRIMARY_RECODE]
                assert albert_on.calls == []
                # Clé (si cache) : même fournisseur et même modèle que le routage.
                if cache:
                    assert [(b["provider"], b["model"]) for b in recode_key_spy] == [("albert", ALBERT_PRIMARY)]
                else:
                    assert recode_key_spy == []
                chunk = _read_chunks(chunks_json)[0]
                assert chunk["recode_model"] == ALBERT_PRIMARY
                assert chunk["recode_status"] == "recoded"


def test_harden_recode_model_albert_with_cli_gpt4omini(albert_on, monkeypatch, tmp_path):
    monkeypatch.setenv("RECODE_HARDEN_ENABLED", "1")
    monkeypatch.setenv("RECODE_MODEL", ALBERT_PRIMARY)
    cfg = rad_recode_cache.RecodeConfig.from_env()
    assert rc.effective_recode_model("gpt-4o-mini", cfg) == ALBERT_PRIMARY
    _df, row = golden._golden_row(tmp_path, text=RAW_SHORT)
    json_file = tmp_path / "chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model="gpt-4o-mini")
    assert [call.json["model"] for call in _chat_calls(albert_on.fake)] == [PRIMARY_RECODE]
    assert albert_on.calls == []
    chunk = _read_chunks(json_file)[0]
    assert chunk["recode_model"] == ALBERT_PRIMARY
    assert chunk["recode_status"] == "recoded"


def test_skip_providers_mistral_ocr_only_by_default(albert_on, monkeypatch, tmp_path):
    assert "albert_mistral_ocr" in rc.RECODE_SKIP_PROVIDERS
    assert "albert_lightonocr" not in rc.RECODE_SKIP_PROVIDERS
    for legacy in ("mistral", "csv", "docling", "mineru", "marker"):
        assert legacy in rc.RECODE_SKIP_PROVIDERS
    fake = albert_on.fake
    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    _df, row = golden._golden_row(skip_dir, provider="albert_mistral_ocr", text=RAW_SHORT)
    rc.process_document_chunks(row, json_file=str(skip_dir / "chunks.json"), model=ALBERT_PRIMARY)
    assert _chat_calls(fake) == []
    recode_dir = tmp_path / "recode"
    recode_dir.mkdir()
    _df, row = golden._golden_row(recode_dir, provider="albert_lightonocr", text=RAW_SHORT)
    rc.process_document_chunks(row, json_file=str(recode_dir / "chunks.json"), model=ALBERT_PRIMARY)
    assert len(_chat_calls(fake)) == 1


def test_skip_providers_lightonocr_when_flag_set():
    assert "albert_lightonocr" in rc.recode_skip_providers(True)
    assert "albert_lightonocr" not in rc.recode_skip_providers(False)
    assert "albert_mistral_ocr" in rc.recode_skip_providers(False)
    assert rc.recode_skip_providers(False) == rc.RECODE_SKIP_PROVIDERS
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'scripts'); import rad_chunk as rc; "
        "print('SKIP', 'albert_lightonocr' in rc.RECODE_SKIP_PROVIDERS, 'albert_mistral_ocr' in rc.RECODE_SKIP_PROVIDERS)"
    )
    env = {k: v for k, v in os.environ.items()
           if not any(part in k for part in ("KEY", "SECRET", "TOKEN", "PASSWORD"))}
    for name in golden.BASELINE_ENV_NAMES:
        env[name] = ""
    env["ALBERT_OCR_SKIP_RECODE"] = "1"
    env["ALBERT_ENABLED"] = "1"
    env["ALBERT_API_KEY"] = ""
    proc = subprocess.run([sys.executable, "-c", code], cwd=RAGPY_ROOT, env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=300)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("SKIP ")]
    assert lines, proc.stderr[-2000:]
    assert lines[-1] == "SKIP True True"


def test_recode_concurrency_is_process_wide(albert_on, monkeypatch, tmp_path):
    semaphore = getattr(rc, "ALBERT_RECODE_SEMAPHORE", None)
    assert semaphore is not None
    assert hasattr(semaphore, "acquire") and hasattr(semaphore, "release")
    limit = int(getattr(rc, "ALBERT_RECODE_CONCURRENCY", AlbertConfig().recode_concurrency))
    assert limit == AlbertConfig().recode_concurrency
    monkeypatch.setattr(rc, "DEFAULT_MAX_WORKERS", 8)
    monkeypatch.setattr(rc, "DEFAULT_DOC_WORKERS", 3)
    state = {"inflight": 0, "max": 0, "calls": 0}
    lock = threading.Lock()

    def slow_echo(body):
        """Écho du chunk après une courte attente, en comptant les requêtes simultanées."""
        with lock:
            state["inflight"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["inflight"])
        try:
            time.sleep(0.05)
            return _echo_reply(body)
        finally:
            with lock:
                state["inflight"] -= 1

    albert_on.fake.chat_reply = slow_echo
    import pandas as pd

    rows = []
    for i in range(3):
        rows.append({"filename": f"doc{i}.pdf", "title": f"Document {i}",
                     "texteocr": golden._golden_ocr_text() + f"\n\nFin du document {i}.",
                     "texteocr_provider": "legacy"})
    df = pd.DataFrame(rows)
    rc.process_all_documents(df, json_file=str(tmp_path / "chunks.json"), model=ALBERT_PRIMARY)
    chunks = _read_chunks(tmp_path / "chunks.json")
    assert state["calls"] == len(chunks) >= 3
    assert state["max"] <= limit
    # Le sémaphore est celui du module (partagé par tous les workers), jamais recréé.
    assert rc.ALBERT_RECODE_SEMAPHORE is semaphore


def test_albert_run_always_emits_recode_model(albert_on, tmp_path):
    _df, row = golden._golden_row(tmp_path)
    json_file = tmp_path / "chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model=ALBERT_PRIMARY)
    chunks = _read_chunks(json_file)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["recode_status"] == "recoded"
        assert chunk["recode_model"] == ALBERT_PRIMARY
    # Chemin OpenAI (ni cache ni durcissement) : aucun des deux champs, comme avant.
    openai_file = tmp_path / "openai_chunks.json"
    rc.process_document_chunks(row, json_file=str(openai_file), model="gpt-4o-mini")
    for chunk in _read_chunks(openai_file):
        assert "recode_status" not in chunk
        assert "recode_model" not in chunk


def test_single_retry_layer_recode(albert_on, capsys):
    fake = albert_on.fake
    max_retries = AlbertConfig.from_env().max_retries
    # Deux échecs transitoires puis succès : trois envois, aucun passage séquentiel en plus.
    fake.inject("POST", CHAT_PATH, "server_error", "bad_gateway")
    raws = _raw_chunks(1)
    texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY)
    assert statuses == ["recoded"]
    assert len(_chat_calls(fake)) == 3
    # Échec persistant : exactement 1 + ALBERT_MAX_RETRIES envois (pas de 2e passe).
    fake.reset_calls()
    fake.inject("POST", CHAT_PATH, *(["server_error"] * 40))
    texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY)
    assert statuses == ["fallback_raw"]
    assert len(_chat_calls(fake)) == 1 + max_retries
    out = capsys.readouterr().out
    assert "2ᵉ passe" not in out
    assert albert_on.legacy_sleeps == []
    assert len(albert_on.sleeps.calls) >= max_retries


def test_seed_only_if_deterministic(albert_on, monkeypatch, tmp_path, recode_key_spy):
    raws = _raw_chunks(1)
    fake = albert_on.fake
    # Sans durcissement : jamais de seed envoyé.
    rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model=ALBERT_PRIMARY)
    assert all("seed" not in call.json for call in _chat_calls(fake))
    # Durci : seed envoyé et présent dans la clé si et seulement si le déterminisme est établi (D11).
    monkeypatch.setenv("RECODE_HARDEN_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("RECODE_SEED", "1234")
    deterministic = _decisions_send_seed()
    fake.reset_calls()
    recode_key_spy.clear()
    rc.recode_batch_cached(raws, rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    bodies = [call.json for call in _chat_calls(fake)]
    assert bodies
    dps = [json.loads(b["decode_params_json"]) for b in recode_key_spy]
    assert dps
    assert rc.ALBERT_SEED_DETERMINISTIC is deterministic
    if deterministic:
        assert all(body.get("seed") == 1234 for body in bodies)
        assert all(dp.get("seed") == 1234 for dp in dps)
    else:
        assert all("seed" not in body for body in bodies)
        assert all("seed" not in dp for dp in dps)
    # Seed non déterministe : ni envoyé, ni dans la clé (nouvelle entrée de cache).
    monkeypatch.setattr(rc, "ALBERT_SEED_DETERMINISTIC", False)
    monkeypatch.setenv("RECODE_CACHE_PATH", str(tmp_path / "cache_no_seed.sqlite"))
    fake.reset_calls()
    recode_key_spy.clear()
    rc.recode_batch_cached(raws, rc._RECODE_INSTRUCTIONS, ALBERT_PRIMARY)
    assert [("seed" in call.json) for call in _chat_calls(fake)] == [False]
    assert all("seed" not in json.loads(b["decode_params_json"]) for b in recode_key_spy)


def _decisions_send_seed():
    """Valeur mesurée de D11 (``send_seed``) dans ``decisions.json``."""
    with open(os.path.join(_THIS, "fixtures", "albert", "decisions.json"), encoding="utf-8") as fh:
        decisions = json.load(fh)
    return bool(decisions["D11"]["value"]["send_seed"])
