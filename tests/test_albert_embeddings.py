"""Tests des embeddings Albert de ``rad_chunk`` (lot 6, tâches 1 à 3).

Deux familles :

* **chemin OpenAI (espace par défaut)** : les goldens G3, G4 et G12 sont
  rejoués **en lecture seule** (jamais régénérés ici) avec une clé Albert
  factice présente, Albert désactivé puis activé sans ``EMBEDDING_PROVIDER=albert`` :
  fichier dense, kwargs de ``embeddings.create``, clés de cache et stdout
  identiques à l'octet (invariants 1 et 26) ;
* **espace Albert (``EMBEDDING_PROVIDER=albert``, bge-m3, 1024 dimensions)** :
  tout ``AlbertClient`` construit pendant le test est branché sur
  ``tests/albert_fakes.FakeAlbert`` (``httpx.MockTransport`` ; aucun réseau,
  clé ``FAKE_ALBERT_KEY`` seulement). On vérifie le contrat d'embeddings
  (invariant 25 : 64 textes au plus par requête, tri par index, ni
  ``dimensions`` ni chaîne vide), l'absence de vecteurs nuls et de tout repli
  vers OpenAI (invariants 19 et 26), le cache cloisonné par espace, les codes
  de sortie (manques : 1 ; compte ou configuration : 2) et le ledger d'usage
  écrit seulement si Albert a été appelé (invariant 31).

Les sommeils des réessais sont enregistrés (aucune attente réelle) et les
clients OpenAI/OpenRouter du module sont remplacés par des doubles qui
journalisent chaque appel. Les exécutions de la CLI passent par ``runpy``
dans le processus de test, ``.env`` neutralisé.
"""

from __future__ import annotations

import difflib
import importlib
import json
import logging
import math
import os
import re
import runpy
import sys
import threading
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

from scripts import rad_providers  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402
from scripts.rad_albert.usage import USAGE_FILENAME  # noqa: E402
from tests.albert_fakes import (  # noqa: E402
    EMBED_DIM,
    FAKE_ALBERT_KEY,
    MAX_EMBED_BATCH,
    FakeAlbert,
    fake_embedding,
    load_fixture,
)

rc = golden.rc
rad_recode_cache = rc.rad_recode_cache
# Fonction de clé d'origine, capturée avant tout espion.
_REAL_EMBED_KEY = rad_recode_cache.embed_key

RAD_CHUNK = os.path.join(RAGPY_ROOT, "scripts", "rad_chunk.py")
GOLDEN_SCRIPTS_DIR = os.path.join(_THIS, "fixtures", "albert", "golden_off", "scripts")
EMBED_PATH = "/v1/embeddings"
ALBERT_MODEL = "bge-m3"
OPENAI_MODEL = "text-embedding-3-large"
SPACE_FIELDS = ("embedding_provider", "embedding_model", "embedding_dim")
ALBERT_CACHE_MODEL = "albert:bge-m3"
D12 = load_fixture("decisions.json")["D12"]["value"]
# Débits très élevés : le limiteur proactif ne fait jamais attendre un test.
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
# Réglages de concurrence du CLI (lus à l'import) retirés avant chaque test.
CLI_TUNING_NAMES = ("DEFAULT_EMBEDDING_BATCH_SIZE", "DEFAULT_MAX_WORKERS", "DEFAULT_DOC_WORKERS",
                    "DEFAULT_BATCH_SIZE_GPT")


def _params_json(norm):
    """Paramètres d'espace bge-m3 attendus dans la clé de cache (``norm`` = ``'l2'`` ou ``'none'``)."""
    return json.dumps({"dim": EMBED_DIM, "model": ALBERT_MODEL, "norm": norm, "provider": "albert"},
                      sort_keys=True)


OPENAI_PARAMS_JSON = json.dumps({"model": OPENAI_MODEL}, sort_keys=True)


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


class MetricsRecorder:
    """Double de ``metrics_collector`` : journalise les compteurs et histogrammes."""

    def __init__(self):
        """Crée un journal vide."""
        self.events = []

    def increment_counter(self, name, value=1, **labels):
        """Enregistre un compteur et ses libellés."""
        self.events.append(("counter", name, dict(labels)))

    def observe_histogram(self, name, value, **labels):
        """Enregistre une observation d'histogramme et ses libellés."""
        self.events.append(("histogram", name, dict(labels)))


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
    """Classes ``AlbertClient`` chargées (racine ``scripts.rad_albert`` et, si présente, ``rad_albert``).

    Le module client est importé ici : ``rad_chunk`` le charge paresseusement,
    après l'installation du double, depuis la racine de son paquet léger.
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
    """Oublie le client Albert en cache et l'arrêt mémorisé du module ``rad_chunk``."""
    reset = getattr(rc, "reset_albert_state", None)
    if callable(reset):
        reset()


def _albert_env(monkeypatch, *, enabled, provider=None, extra=None):
    """Clé Albert factice et débits élevés ; Albert ON ou OFF ; ``EMBEDDING_PROVIDER`` si fourni."""
    monkeypatch.setenv("ALBERT_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)
    for name, value in (extra or {}).items():
        monkeypatch.setenv(name, value)
    if provider is not None:
        monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    _set_module_attr_if_present(monkeypatch, rc, "ALBERT_ENABLED", bool(enabled))
    _set_module_attr_if_present(monkeypatch, rc, "ALBERT_API_KEY", FAKE_ALBERT_KEY)


def _albert_space(extra_env=None):
    """Espace bge-m3 tel que ``EmbeddingConfig.from_env`` le décrit (Albert ON)."""
    env = {"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": "1"}
    env.update(extra_env or {})
    return rad_providers.EmbeddingConfig.from_env(env).space


def _embed_posts(fake):
    """Requêtes ``POST /v1/embeddings`` reçues par le faux."""
    return fake.calls_to("POST", EMBED_PATH)


def _posted_inputs(fake):
    """Tous les textes envoyés à ``/v1/embeddings``, requête par requête."""
    return [list(call.json["input"]) for call in _embed_posts(fake)]


def _openai_embed_calls(calls):
    """Appels ``embeddings.create`` journalisés par les doubles OpenAI/OpenRouter."""
    return [c for c in calls if c["endpoint"] == "embeddings.create"]


def _close(vector, expected, tol=1e-9):
    """Vrai si ``vector`` a la longueur de ``expected`` et en diffère d'au plus ``tol`` par composante."""
    if not isinstance(vector, list) or len(vector) != len(expected):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(vector, expected))


def _is_zero_vector(vector):
    """Vrai pour une liste non vide dont toutes les composantes sont nulles."""
    return isinstance(vector, list) and len(vector) > 0 and all(float(v) == 0.0 for v in vector)


def _norm(vector):
    """Norme L2 d'un vecteur."""
    return math.sqrt(sum(float(v) * float(v) for v in vector))


def _chunks(n, prefix="Passage"):
    """``n`` chunks de forme ``output_chunks.json`` aux textes distincts (10 chunks par document)."""
    out = []
    for i in range(n):
        doc_id = f"{(i // 10) + 1:012d}"
        out.append({
            "id": f"{doc_id}_{(i % 10) + 1}",
            "doc_id": doc_id,
            "chunk_index": (i % 10) + 1,
            "total_chunks": 10,
            "text": f"{prefix} {i} — texte recodé stable pour l'embedding dense de l'espace souverain.",
            "title": f"Document {(i // 10) + 1}",
            "texteocr_provider": "mistral",
        })
    return out


def _write_chunks(directory, chunks, name="output_chunks.json"):
    """Écrit ``chunks`` en JSON dans ``directory`` ; renvoie le chemin (str)."""
    os.makedirs(str(directory), exist_ok=True)
    path = os.path.join(str(directory), name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    return path


def _read_json(path):
    """Contenu JSON d'un fichier."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _run_dense(directory, chunks):
    """Phase dense (``generate_and_save_embeddings``) sur ``chunks`` ; renvoie (retour, sortie, chunks lus)."""
    input_path = _write_chunks(directory, chunks)
    output_path = os.path.join(str(directory), "output_chunks_with_embeddings.json")
    returned = rc.generate_and_save_embeddings(input_path, output_path)
    written = _read_json(output_path) if os.path.exists(output_path) else None
    return returned, output_path, written


def _run_dense_allow_exit(directory, chunks, allowed=(1,)):
    """Comme ``_run_dense`` mais tolère un ``SystemExit`` de code ``allowed`` ; renvoie aussi le code."""
    try:
        returned, output_path, written = _run_dense(directory, chunks)
        return 0, returned, output_path, written
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
        assert code in allowed, f"code de sortie inattendu : {code}"
        output_path = os.path.join(str(directory), "output_chunks_with_embeddings.json")
        written = _read_json(output_path) if os.path.exists(output_path) else None
        return code, None, output_path, written


def _usage_files(root):
    """Chemins de tous les ``albert_usage.jsonl`` sous ``root``."""
    found = []
    for base, _dirs, files in os.walk(str(root)):
        if USAGE_FILENAME in files:
            found.append(os.path.join(base, USAGE_FILENAME))
    return found


def _neutralise_dotenv(monkeypatch):
    """Le ``.env`` du dépôt n'est jamais chargé par une CLI exécutée dans le test."""
    for module_name in ("scripts.rad_env", "rad_env"):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, "load_dotenv_guarded", lambda *a, **k: False)


def _run_cli(monkeypatch, args):
    """Exécute ``scripts/rad_chunk.py`` comme ``__main__`` ; renvoie le code de sortie (0 si fin normale).

    Les gestionnaires de logs ajoutés par la CLI (``chunking.log``, console)
    sont retirés et fermés, et le niveau du logger racine est restauré.
    """
    _neutralise_dotenv(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rad_chunk.py"] + [str(a) for a in args])
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    level_before = root_logger.level
    try:
        runpy.run_path(RAD_CHUNK, run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
    finally:
        for handler in list(root_logger.handlers):
            if handler not in handlers_before:
                root_logger.removeHandler(handler)
                handler.close()
        root_logger.setLevel(level_before)
    return code


def _cli_env(monkeypatch, *, openai_key=False, batch=2):
    """Réglages d'import de la CLI : un worker, lots de ``batch`` ; clé OpenAI factice ou vide."""
    for name in golden.BASELINE_ENV_NAMES:
        monkeypatch.setenv(name, "")
    if openai_key:
        monkeypatch.setenv("OPENAI_API_KEY", golden.FAKE_ENV["OPENAI_API_KEY"])
    monkeypatch.setenv("DEFAULT_MAX_WORKERS", "1")
    monkeypatch.setenv("DEFAULT_DOC_WORKERS", "1")
    monkeypatch.setenv("DEFAULT_EMBEDDING_BATCH_SIZE", str(batch))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")


def _patch_openai_class(monkeypatch, calls):
    """``openai.OpenAI`` renvoie un double journalisant (``openai`` ou ``openrouter`` selon ``base_url``)."""
    import openai

    def factory(*args, base_url=None, **kwargs):
        """Double OpenAI-compatible, sans aucun appel réseau."""
        return golden._FakeLLMClient("openrouter" if base_url else "openai", calls)

    monkeypatch.setattr(openai, "OpenAI", factory)


def _cli_input(tmp_path, n=4):
    """Fichier ``in/output_chunks.json`` de ``n`` chunks et dossier de sortie ``out`` ; renvoie (entrée, sortie)."""
    input_path = _write_chunks(tmp_path / "in", _chunks(n))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return input_path, out_dir


def _cli_dense_output(out_dir):
    """Fichier dense écrit par la CLI pour une entrée ``output_chunks.json``."""
    return os.path.join(str(out_dir), "output_chunks_with_embeddings.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Retire identifiants, réglages Albert/cache/dédup et réglages CLI ; état Albert remis à zéro."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in CLI_TUNING_NAMES:
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


@pytest.fixture
def albert_on(monkeypatch, chunk_env):
    """Albert ON, ``EMBEDDING_PROVIDER=albert``, clé factice ; ``FakeAlbert`` sert bge-m3 en 1024 d."""
    fake = FakeAlbert()
    sleeps = SleepRecorder()
    route_albert_to_fake(monkeypatch, fake, sleeps)
    _albert_env(monkeypatch, enabled=True, provider="albert")
    return SimpleNamespace(fake=fake, sleeps=sleeps, calls=chunk_env.calls)


# ---------------------------------------------------------------------------
# Espace OpenAI (défaut) : goldens rejoués en lecture seule
# ---------------------------------------------------------------------------
OPENAI_CASES = [
    ("albert_off", False, None),
    ("albert_on_provider_unset", True, None),
    ("albert_on_provider_openai", True, "openai"),
]


@pytest.mark.parametrize("case,enabled,provider", OPENAI_CASES, ids=[c[0] for c in OPENAI_CASES])
def test_openai_request_and_output_identical(case, enabled, provider, chunk_env, monkeypatch, tmp_path):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled, provider=provider)
    input_path = golden._write_chunks_file(tmp_path)
    output_path = str(tmp_path / "output_chunks_with_embeddings.json")
    returned = rc.generate_and_save_embeddings(input_path, output_path)
    assert returned == output_path
    with open(output_path, "rb") as fh:
        _assert_matches_golden_bytes("g3_dense_output.json", fh.read())
    _assert_matches_golden_json("g3_embeddings_create_kwargs.json", chunk_env.calls)
    # Requête OpenAI directe : exactement (input, model, timeout=60.0), jamais encoding_format.
    for space in (None, rad_providers.OPENAI_DEFAULT):
        chunk_env.calls.clear()
        kwargs = {} if space is None else {"space": space}
        vectors = rc.get_embeddings_batch(["un", "deux"], **kwargs)
        assert vectors == [golden._fake_vector("un"), golden._fake_vector("deux")]
        assert [c["client"] for c in chunk_env.calls] == ["openai"]
        assert chunk_env.calls[0]["kwargs"] == {"input": ["un", "deux"], "model": OPENAI_MODEL, "timeout": 60.0}
    assert fake.calls == []
    assert _usage_files(tmp_path) == []


def test_openai_zero_fallback_unchanged(chunk_env, monkeypatch):
    """Espace OpenAI : le repli historique en vecteurs nuls de 3072 dimensions est conservé."""
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=True, provider="openai")

    def failing_create(**kwargs):
        """``embeddings.create`` qui échoue toujours (erreur non liée au débit)."""
        raise RuntimeError("panne simulée")

    monkeypatch.setattr(rc.client.embeddings, "create", failing_create)
    vectors = rc.get_embeddings_batch(["un", "deux"])
    assert vectors == [[0.0] * 3072, [0.0] * 3072]
    assert fake.calls == []


def test_off_stdout_matches_golden(chunk_env, tmp_path, capsys, monkeypatch):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=False)
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
    assert fake.calls == []


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
                statuses = [chunk.get("recode_status") for chunk in _read_json(chunks_json)]
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


@pytest.mark.parametrize("case,enabled,provider", OPENAI_CASES, ids=[c[0] for c in OPENAI_CASES])
def test_openai_embed_key_unchanged(case, enabled, provider, chunk_env, monkeypatch, tmp_path):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled, provider=provider)
    cells = _g4_cells(monkeypatch, tmp_path, chunk_env.calls)
    _assert_matches_golden_json("g4_recode_embed_keys.json", cells)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Espace Albert : contrat de requête (invariant 25)
# ---------------------------------------------------------------------------
def test_albert_130_texts_three_posts(albert_on):
    fake = albert_on.fake
    texts = [f"Phrase de test numéro {i} sur la sociologie des sciences." for i in range(130)]
    vectors = rc.get_embeddings_batch(texts, space=_albert_space())
    posts = _embed_posts(fake)
    assert sorted(len(call.json["input"]) for call in posts) == [2, 64, 64]
    assert sorted(t for inputs in _posted_inputs(fake) for t in inputs) == sorted(texts)
    assert len(vectors) == 130
    for text, vector in zip(texts, vectors):
        assert len(vector) == EMBED_DIM
        assert _close(vector, fake_embedding(text))
    assert _openai_embed_calls(albert_on.calls) == []


def test_albert_sorted_by_index(albert_on, tmp_path):
    fake = albert_on.fake
    fake.embed_shuffle = True  # le faux renvoie ``data`` en ordre inverse de ``index``
    texts = [f"Texte ordonné {i} : l'ordre des vecteurs suit l'index de la réponse." for i in range(5)]
    vectors = rc.get_embeddings_batch(texts, space=_albert_space())
    assert all(_close(v, fake_embedding(t)) for t, v in zip(texts, vectors))
    chunks = _chunks(6)
    _returned, _out, written = _run_dense(tmp_path, chunks)
    assert [c["id"] for c in written] == [c["id"] for c in chunks]
    for original, chunk in zip(chunks, written):
        assert _close(chunk["embedding"], fake_embedding(original["text"]))


@pytest.mark.parametrize("configured_model", [None, "openweight-embeddings", "BAAI/bge-m3"],
                         ids=["default", "alias_openweight", "alias_hf"])
def test_request_float_no_dimensions(configured_model, albert_on, monkeypatch, tmp_path):
    fake = albert_on.fake
    if configured_model is not None:
        monkeypatch.setenv("ALBERT_EMBED_MODEL", configured_model)
    _returned, _out, written = _run_dense(tmp_path, _chunks(3))
    posts = _embed_posts(fake)
    assert posts, "aucune requête d'embeddings Albert"
    for call in posts:
        body = call.json
        assert set(body) == {"model", "input", "encoding_format"}
        assert body["encoding_format"] == "float"
        assert "dimensions" not in body
        assert "user" not in body
        # D12 : l'alias est résolu, c'est l'id épinglé qui part sur le fil.
        assert body["model"] == ALBERT_MODEL
        assert all(isinstance(t, str) and t.strip() for t in body["input"])
        assert FAKE_ALBERT_KEY not in call.url
    assert all(len(c["embedding"]) == D12["embedding_dim"] for c in written)


def test_preflight_embed_before_first_post(albert_on, tmp_path):
    """Preflight du rôle embed (``/v1/me`` puis ``/v1/models``) avant la première requête d'embeddings."""
    fake = albert_on.fake
    _run_dense(tmp_path, _chunks(2))
    paths = [call.path for call in fake.calls]
    first_post = paths.index(EMBED_PATH)
    assert "/v1/me" in paths[:first_post]
    assert "/v1/models" in paths[:first_post]


def test_empty_text_not_sent(albert_on):
    fake = albert_on.fake
    texts = [
        "Alpha : un premier passage non vide du corpus.",
        "",
        "Gamma : un troisième passage non vide du corpus.",
        "   \n\t ",
    ]
    vectors = rc.get_embeddings_batch(texts, space=_albert_space())
    assert len(vectors) == 4
    assert vectors[1] is None and vectors[3] is None
    assert _close(vectors[0], fake_embedding(texts[0]))
    assert _close(vectors[2], fake_embedding(texts[2]))
    sent = [t for inputs in _posted_inputs(fake) for t in inputs]
    assert sorted(sent) == sorted([texts[0], texts[2]])
    assert all(t.strip() for t in sent)
    # Lot entièrement vide : aucun envoi, aucun vecteur (jamais de zéros).
    fake.reset_calls()
    assert rc.get_embeddings_batch(["", "  "], space=_albert_space()) == [None, None]
    assert _embed_posts(fake) == []


# ---------------------------------------------------------------------------
# Espace Albert : jamais de vecteur nul, jamais OpenAI (invariants 19 et 26)
# ---------------------------------------------------------------------------
def test_albert_never_writes_zero_vectors(albert_on, monkeypatch, tmp_path):
    fake = albert_on.fake
    monkeypatch.setenv("ALBERT_MAX_RETRIES", "0")
    monkeypatch.setenv("ALBERT_EMBED_MAX_MISSING_RATIO", "1.0")
    fake.inject("POST", EMBED_PATH, *([500] * 200))
    # Appel direct : échec → None pour chaque texte.
    vectors = rc.get_embeddings_batch(["un texte", "un autre texte"], space=_albert_space())
    assert vectors == [None, None]
    # Phase dense : fichier écrit sans aucun vecteur nul.
    code, _returned, output_path, written = _run_dense_allow_exit(tmp_path / "http_failure", _chunks(4))
    assert written is not None, "le fichier dense doit être écrit"
    assert len(written) == 4
    assert all(c.get("embedding") is None for c in written)
    assert not any(_is_zero_vector(c.get("embedding")) for c in written)
    # Exception levée par le traitement d'un lot : jamais de zéros non plus.
    fake.clear_injections()
    boom_calls = []

    def boom(*args, **kwargs):
        """Traitement de lot qui lève une exception inattendue."""
        boom_calls.append(1)
        raise RuntimeError("panne simulée du lot")

    monkeypatch.setattr(rc, "process_chunks_for_embedding", boom)
    code, _returned, output_path, written = _run_dense_allow_exit(tmp_path / "batch_exception", _chunks(4))
    assert boom_calls, "process_chunks_for_embedding doit traiter les lots de l'espace Albert"
    assert written is not None, "le fichier dense doit être écrit"
    assert not any(_is_zero_vector(c.get("embedding")) for c in written)
    assert all(c.get("embedding") is None for c in written)
    assert _openai_embed_calls(albert_on.calls) == []


def test_albert_never_calls_openai_embeddings(albert_on, monkeypatch, tmp_path):
    fake = albert_on.fake
    calls = albert_on.calls
    monkeypatch.setenv("ALBERT_MAX_RETRIES", "0")
    monkeypatch.setenv("ALBERT_EMBED_MAX_MISSING_RATIO", "1.0")
    # Succès : un client OpenAI existe mais n'est jamais utilisé.
    _run_dense(tmp_path / "ok", _chunks(3))
    assert _embed_posts(fake)
    # Échecs transitoires et permanents : aucun repli vers OpenAI.
    fake.inject("POST", EMBED_PATH, *([500] * 50))
    rc.get_embeddings_batch(["texte en échec"], space=_albert_space())
    _run_dense_allow_exit(tmp_path / "transient", _chunks(3))
    fake.clear_injections()
    fake.inject("POST", EMBED_PATH, *(["wrong_model_type"] * 50))
    rc.get_embeddings_batch(["texte en échec"], space=_albert_space())
    _run_dense_allow_exit(tmp_path / "permanent", _chunks(3))
    fake.clear_injections()
    # Erreur de compte : arrêt (code 2), toujours sans OpenAI.
    fake.inject("POST", EMBED_PATH, *(["invalid_key"] * 50))
    _run_dense_allow_exit(tmp_path / "account", _chunks(3), allowed=(2,))
    assert _openai_embed_calls(calls) == []
    # Processus sans client OpenAI : l'espace Albert fonctionne quand même.
    fake.clear_injections()
    monkeypatch.setattr(rc, "client", None)
    monkeypatch.setattr(rc, "openrouter_client", None)
    _reset_rc_albert_state()
    _returned, _out, written = _run_dense(tmp_path / "no_openai_client", _chunks(2))
    assert all(len(c["embedding"]) == EMBED_DIM for c in written)


def test_batch_clamped_64(albert_on, chunk_env, monkeypatch, tmp_path):
    fake = albert_on.fake
    monkeypatch.setattr(rc, "DEFAULT_EMBEDDING_BATCH_SIZE", 100)
    chunks = _chunks(130)
    _returned, _out, written = _run_dense(tmp_path / "default_batch", chunks)
    sizes = sorted(len(call.json["input"]) for call in _embed_posts(fake))
    assert sizes == [2, 64, 64]
    assert max(sizes) <= MAX_EMBED_BATCH == D12["ALBERT_EMBED_BATCH"]
    assert all(_close(c["embedding"], fake_embedding(o["text"])) for o, c in zip(chunks, written))
    # ALBERT_EMBED_BATCH plus petit : respecté (jamais plus de 64 de toute façon).
    fake.reset_calls()
    _reset_rc_albert_state()
    monkeypatch.setenv("ALBERT_EMBED_BATCH", "16")
    _run_dense(tmp_path / "small_batch", chunks)
    sizes = [len(call.json["input"]) for call in _embed_posts(fake)]
    assert max(sizes) <= 16
    assert sum(sizes) == 130
    # Espace OpenAI : la taille de lot historique n'est pas plafonnée à 64.
    fake.reset_calls()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    chunk_env.calls.clear()
    _run_dense(tmp_path / "openai", chunks)
    assert sorted(len(c["kwargs"]["input"]) for c in _openai_embed_calls(chunk_env.calls)) == [30, 100]
    assert _embed_posts(fake) == []


# ---------------------------------------------------------------------------
# Espace Albert : champs d'espace, cache cloisonné, métriques
# ---------------------------------------------------------------------------
def test_space_fields_only_non_default(albert_on, monkeypatch, tmp_path):
    chunks = _chunks(3)
    original_keys = set(chunks[0])
    # Espace OpenAI (Albert ON) : aucun champ d'espace.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    _returned, _out, written = _run_dense(tmp_path / "openai", chunks)
    for chunk in written:
        assert not any(field in chunk for field in SPACE_FIELDS)
        assert set(chunk) == original_keys | {"embedding"}
    assert albert_on.fake.calls == []
    # Espace Albert : les trois champs, cohérents avec le vecteur.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "albert")
    _returned, _out, written = _run_dense(tmp_path / "albert", chunks)
    for chunk in written:
        assert chunk["embedding_provider"] == "albert"
        assert chunk["embedding_model"] == ALBERT_MODEL
        assert chunk["embedding_dim"] == EMBED_DIM == len(chunk["embedding"])
        assert set(chunk) == original_keys | {"embedding"} | set(SPACE_FIELDS)
    space = rad_providers.check_uniform_space(written)
    assert (space.provider, space.model, space.dim) == ("albert", ALBERT_MODEL, EMBED_DIM)
    assert space.is_default is False


def test_metrics_label_is_space_model(albert_on, monkeypatch, tmp_path):
    """Libellé ``model`` des métriques = modèle de l'espace (bge-m3 ou text-embedding-3-large)."""
    recorder = MetricsRecorder()
    monkeypatch.setattr(rc, "METRICS_AVAILABLE", True)
    monkeypatch.setattr(rc, "metrics_collector", recorder)
    _run_dense(tmp_path / "albert", _chunks(2))
    models = {labels.get("model") for _kind, _name, labels in recorder.events if "model" in labels}
    assert models == {ALBERT_MODEL}
    recorder.events.clear()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    _run_dense(tmp_path / "openai", _chunks(2))
    models = {labels.get("model") for _kind, _name, labels in recorder.events if "model" in labels}
    assert models == {OPENAI_MODEL}


@pytest.fixture
def embed_key_spy(monkeypatch):
    """Espion de ``rad_recode_cache.embed_key`` : (texte, modèle, paramètres, clé) de chaque appel."""
    calls = []

    def spy(*args, **kwargs):
        """Enregistre les arguments connus et la valeur réelle de la clé."""
        value = _REAL_EMBED_KEY(*args, **kwargs)
        bound = golden._bind_known(args, kwargs, golden.G4_EMBED_KEY_ARGS)
        calls.append((bound.get("text"), bound.get("embed_model"), bound.get("embed_params_json"), value))
        return value

    monkeypatch.setattr(rad_recode_cache, "embed_key", spy)
    return calls


def _enable_embed_cache(monkeypatch, tmp_path):
    """Active le cache d'embeddings dans un SQLite du test ; renvoie sa ``RecodeConfig``."""
    monkeypatch.setenv("RECODE_EMBED_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    return rad_recode_cache.RecodeConfig.from_env()


def test_albert_cache_namespaced_with_norm(albert_on, monkeypatch, tmp_path, embed_key_spy):
    fake = albert_on.fake
    _enable_embed_cache(monkeypatch, tmp_path)
    chunks = _chunks(3)
    texts = [c["text"] for c in chunks]
    _returned, _out, first = _run_dense(tmp_path / "run1", chunks)
    posts_first = len(_embed_posts(fake))
    assert posts_first > 0
    assert embed_key_spy, "le cache d'embeddings n'a pas été consulté"
    assert {(model, params) for _t, model, params, _k in embed_key_spy} == {(ALBERT_CACHE_MODEL, _params_json("l2"))}
    assert sorted(t for t, _m, _p, _k in embed_key_spy) == sorted(texts)
    l2_keys = {t: k for t, _m, _p, k in embed_key_spy}
    # Second passage : HIT → aucun appel, vecteurs identiques.
    embed_key_spy.clear()
    _returned, _out, second = _run_dense(tmp_path / "run2", chunks)
    assert len(_embed_posts(fake)) == posts_first
    assert [c["embedding"] for c in second] == [c["embedding"] for c in first]
    # Normalisation désactivée : autre espace, autres clés → MISS.
    embed_key_spy.clear()
    _reset_rc_albert_state()
    monkeypatch.setenv("ALBERT_EMBED_L2_NORMALIZE", "0")
    _run_dense(tmp_path / "run3", chunks)
    assert len(_embed_posts(fake)) > posts_first
    assert {(model, params) for _t, model, params, _k in embed_key_spy} == {(ALBERT_CACHE_MODEL, _params_json("none"))}
    none_keys = {t: k for t, _m, _p, k in embed_key_spy}
    for text in texts:
        openai_key = _REAL_EMBED_KEY(text, OPENAI_MODEL, OPENAI_PARAMS_JSON)
        assert len({l2_keys[text], none_keys[text], openai_key}) == 3


def test_wrong_dim_hit_is_miss(albert_on, monkeypatch, tmp_path):
    fake = albert_on.fake
    cfg = _enable_embed_cache(monkeypatch, tmp_path)
    chunks = _chunks(2)
    stale_text = chunks[0]["text"]
    stale_key = _REAL_EMBED_KEY(stale_text, ALBERT_CACHE_MODEL, _params_json("l2"))
    rad_recode_cache.put_embed(cfg, stale_key, [0.25] * 3072)
    assert rad_recode_cache.get_embed(cfg, stale_key) == [0.25] * 3072
    _returned, _out, written = _run_dense(tmp_path / "run", chunks)
    sent = [t for inputs in _posted_inputs(fake) for t in inputs]
    assert stale_text in sent, "un HIT de mauvaise dimension doit être traité comme un MISS"
    assert all(len(c["embedding"]) == EMBED_DIM for c in written)
    assert _close(written[0]["embedding"], fake_embedding(stale_text))
    # Un échec (None) n'est jamais mis en cache.
    failed = _chunks(1, prefix="Passage en échec")
    failed_key = _REAL_EMBED_KEY(failed[0]["text"], ALBERT_CACHE_MODEL, _params_json("l2"))
    monkeypatch.setenv("ALBERT_MAX_RETRIES", "0")
    monkeypatch.setenv("ALBERT_EMBED_MAX_MISSING_RATIO", "1.0")
    fake.inject("POST", EMBED_PATH, *([500] * 50))
    _run_dense_allow_exit(tmp_path / "failed", failed)
    assert rad_recode_cache.get_embed(cfg, failed_key) is None
    fake.clear_injections()
    _reset_rc_albert_state()
    fake.reset_calls()
    _returned, _out, written = _run_dense(tmp_path / "retry", failed)
    assert [t for inputs in _posted_inputs(fake) for t in inputs] == [failed[0]["text"]]
    cached = rad_recode_cache.get_embed(cfg, failed_key)
    assert cached is not None and len(cached) == EMBED_DIM


# ---------------------------------------------------------------------------
# Codes de sortie et configuration
# ---------------------------------------------------------------------------
CONFIG_REFUSALS = [
    ("albert_disabled", "albert", False, {}, True, "ALBERT_ENABLED"),
    ("unknown_on", "cohere", True, {}, True, "EMBEDDING_PROVIDER"),
    ("unknown_off", "cohere", False, {}, True, "EMBEDDING_PROVIDER"),
    ("unsupported_model", "albert", True, {"ALBERT_EMBED_MODEL": "qwen3-vl-embedding-8b"}, True,
     "ALBERT_EMBED_MODEL"),
    ("missing_key", "albert", True, {}, False, "Clé API Albert"),
]


@pytest.mark.parametrize("case,value,enabled,extra,with_key,expected_text", CONFIG_REFUSALS,
                         ids=[c[0] for c in CONFIG_REFUSALS])
def test_albert_provider_when_disabled_exit_2(case, value, enabled, extra, with_key, expected_text, chunk_env,
                                              monkeypatch, tmp_path, capsys, caplog):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled, provider=value, extra=extra)
    if not with_key:
        monkeypatch.setenv("ALBERT_API_KEY", "")
        _set_module_attr_if_present(monkeypatch, rc, "ALBERT_API_KEY", None)
    input_path = _write_chunks(tmp_path, _chunks(2))
    output_path = str(tmp_path / "output_chunks_with_embeddings.json")
    with pytest.raises(SystemExit) as excinfo:
        rc.generate_and_save_embeddings(input_path, output_path)
    assert excinfo.value.code == 2
    shown = capsys.readouterr()
    assert expected_text in shown.out + shown.err + caplog.text
    assert not os.path.exists(output_path)
    assert fake.calls == []
    assert _openai_embed_calls(chunk_env.calls) == []


def test_albert_provider_when_disabled_exit_2_cli(monkeypatch, tmp_path, capsys):
    """CLI : ``--embedding-provider albert`` avec Albert désactivé → code 2, aucun appel."""
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=False)
    _cli_env(monkeypatch, openai_key=True)
    openai_calls = []
    _patch_openai_class(monkeypatch, openai_calls)
    input_path, out_dir = _cli_input(tmp_path)
    code = _run_cli(monkeypatch, ["--input", input_path, "--output", out_dir, "--phase", "dense",
                                  "--embedding-provider", "albert"])
    assert code == 2
    assert "ALBERT_ENABLED" in capsys.readouterr().out
    assert fake.calls == []
    assert _openai_embed_calls(openai_calls) == []


@pytest.mark.parametrize("threshold,expected_code", [("0.0", 1), ("0.75", 0)], ids=["default_ratio", "tolerant"])
def test_missing_ratio_exit_1(threshold, expected_code, monkeypatch, tmp_path, capsys):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=True, extra={"ALBERT_EMBED_MAX_MISSING_RATIO": threshold,
                                                  "ALBERT_EMBED_CONCURRENCY": "1"})
    _cli_env(monkeypatch, openai_key=True, batch=2)
    openai_calls = []
    _patch_openai_class(monkeypatch, openai_calls)
    # Un seul envoi en échec permanent (400) : 2 chunks manquants sur 4 (ratio 0.5).
    fake.inject("POST", EMBED_PATH, "bad_request")
    input_path, out_dir = _cli_input(tmp_path, n=4)
    code = _run_cli(monkeypatch, ["--input", input_path, "--output", out_dir, "--phase", "dense",
                                  "--embedding-provider", "albert"])
    out = capsys.readouterr().out
    assert code == expected_code
    output_path = _cli_dense_output(out_dir)
    assert os.path.exists(output_path), "le fichier dense est écrit avant la sortie"
    written = _read_json(output_path)
    assert len(written) == 4
    missing = [c for c in written if c.get("embedding") is None]
    assert len(missing) == 2
    assert not any(_is_zero_vector(c.get("embedding")) for c in written)
    assert all(len(c["embedding"]) == EMBED_DIM for c in written if c.get("embedding") is not None)
    assert _openai_embed_calls(openai_calls) == []
    if expected_code == 1:
        assert "ALBERT_EMBED_MAX_MISSING_RATIO" in out or "manqu" in out.lower()
    else:
        # Albert appelé : ledger d'usage écrit dans le dossier de sortie.
        assert os.path.exists(os.path.join(str(out_dir), USAGE_FILENAME))


ACCOUNT_FAILURES = [
    ("invalid_key", "POST", EMBED_PATH, ("invalid_key",), None, "invalid_key"),
    ("account_expired", "POST", EMBED_PATH, ("account_expired",), None, "account_expired"),
    ("budget_exhausted", "POST", EMBED_PATH, ("budget_exhausted",), None, "budget_exhausted"),
    ("quota_retry_after", "POST", EMBED_PATH, ("rate_limited",), 3600, "quota_exhausted"),
    ("preflight_me", "GET", "/v1/me", ("invalid_key",), None, "invalid_key"),
]
# Ligne lue par les routes et le runner Celery (événement SSE ``credential_required``).
ABORT_LINE_RE = re.compile(r"^Albert abort: kind=account reason=(\w+) credential_required=albert_api_key$")


@pytest.mark.parametrize("case,method,path,entries,retry_after,reason", ACCOUNT_FAILURES,
                         ids=[c[0] for c in ACCOUNT_FAILURES])
def test_account_error_exit_2(case, method, path, entries, retry_after, reason, monkeypatch, tmp_path, capsys):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=True, extra={"ALBERT_EMBED_CONCURRENCY": "1"})
    _cli_env(monkeypatch, openai_key=True, batch=2)
    openai_calls = []
    _patch_openai_class(monkeypatch, openai_calls)
    fake.inject(method, path, *(entries * 20), retry_after=retry_after)
    input_path, out_dir = _cli_input(tmp_path, n=4)
    code = _run_cli(monkeypatch, ["--input", input_path, "--output", out_dir, "--phase", "dense",
                                  "--embedding-provider", "albert"])
    assert code == 2
    markers = [m.group(1) for m in map(ABORT_LINE_RE.match, capsys.readouterr().out.splitlines()) if m]
    assert markers == [reason]
    posts = _embed_posts(fake)
    if method == "GET":
        assert posts == []  # preflight en échec : aucun envoi de textes
    else:
        assert len(posts) == 1  # erreur de compte : le job s'arrête, aucun autre lot n'est envoyé
    output_path = _cli_dense_output(out_dir)
    if os.path.exists(output_path):
        assert not any(_is_zero_vector(c.get("embedding")) for c in _read_json(output_path))
    assert _openai_embed_calls(openai_calls) == []


# ---------------------------------------------------------------------------
# CLI : --embedding-provider et ledger d'usage
# ---------------------------------------------------------------------------
def test_cli_flag_sets_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("COLUMNS", "200")
    assert _run_cli(monkeypatch, ["--help"]) == 0
    help_text = capsys.readouterr().out
    assert len([line for line in help_text.splitlines() if re.match(r"^  --embedding-provider", line)]) == 1
    assert "{openai,albert}" in help_text
    # --embedding-provider albert : pose EMBEDDING_PROVIDER, sans clé OpenAI dans le processus.
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=True)
    _cli_env(monkeypatch, openai_key=False, batch=2)
    input_path, out_dir = _cli_input(tmp_path, n=3)
    code = _run_cli(monkeypatch, ["--input", input_path, "--output", out_dir, "--phase", "dense",
                                  "--embedding-provider", "albert"])
    assert code == 0
    assert os.environ.get("EMBEDDING_PROVIDER") == "albert"
    assert _embed_posts(fake)
    written = _read_json(_cli_dense_output(out_dir))
    assert all(c.get("embedding_provider") == "albert" and len(c["embedding"]) == EMBED_DIM for c in written)
    usage_path = os.path.join(str(out_dir), USAGE_FILENAME)
    assert os.path.exists(usage_path), "ledger d'usage attendu : Albert a été appelé"
    with open(usage_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records and {r.get("endpoint") for r in records} == {EMBED_PATH}
    assert {r.get("model") for r in records} == {ALBERT_MODEL}
    # --embedding-provider openai l'emporte sur EMBEDDING_PROVIDER=albert.
    fake.reset_calls()
    _cli_env(monkeypatch, openai_key=True, batch=2)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "albert")
    openai_calls = []
    _patch_openai_class(monkeypatch, openai_calls)
    input_path = _write_chunks(tmp_path / "in2", _chunks(3))
    out_dir2 = tmp_path / "out2"
    out_dir2.mkdir()
    code = _run_cli(monkeypatch, ["--input", input_path, "--output", out_dir2, "--phase", "dense",
                                  "--embedding-provider", "openai"])
    assert code == 0
    assert os.environ.get("EMBEDDING_PROVIDER") == "openai"
    assert fake.calls == []
    assert _openai_embed_calls(openai_calls)
    written = _read_json(_cli_dense_output(out_dir2))
    assert not any(field in c for c in written for field in SPACE_FIELDS)
    assert not os.path.exists(os.path.join(str(out_dir2), USAGE_FILENAME))


@pytest.mark.parametrize("enabled,flag", [(False, None), (True, None), (True, "openai")],
                         ids=["albert_off", "albert_on_provider_unset", "albert_on_flag_openai"])
def test_no_usage_file_when_off(enabled, flag, monkeypatch, tmp_path, capsys):
    fake = FakeAlbert()
    route_albert_to_fake(monkeypatch, fake, SleepRecorder())
    _albert_env(monkeypatch, enabled=enabled)
    _cli_env(monkeypatch, openai_key=True, batch=2)
    openai_calls = []
    _patch_openai_class(monkeypatch, openai_calls)
    input_path, out_dir = _cli_input(tmp_path, n=3)
    args = ["--input", input_path, "--output", out_dir, "--phase", "dense"]
    if flag:
        args += ["--embedding-provider", flag]
    code = _run_cli(monkeypatch, args)
    out = capsys.readouterr().out
    assert code == 0
    assert _openai_embed_calls(openai_calls)
    assert fake.calls == []
    assert _usage_files(tmp_path) == []
    assert "albert_usage" not in out
    assert not any(line.startswith("Albert") for line in out.splitlines())
