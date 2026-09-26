"""Tests de l'infrastructure de test Albert : ``conftest.py`` racine et ``tests/albert_fakes.py``.

Les comportements de session (refus d'un ``.env`` contenant ``ALBERT_LIVE``, saut
des tests live, neutralisation de l'environnement avant l'import de ``app``,
échec au teardown d'une requête bloquée puis avalée, compteur des tests
historiques, refus des variantes trio) sont vérifiés par un pytest lancé en
sous-processus sur un petit projet temporaire qui embarque une copie du
``conftest.py`` racine. Ces sous-processus tournent derrière un proxy mort et ne
font aucun appel réseau externe ; ``ALBERT_LIVE`` n'y est jamais défini.
"""

import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import types
import xml.etree.ElementTree as ET

import httpx
import pytest
import requests

from tests.albert_fakes import (
    FAKE_ALBERT_KEY,
    FAKE_BASE_URL,
    ROUTES,
    FakeAlbert,
    FakeRedis,
    fake_embedding,
    load_fixture,
)

RAGPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_CONFTEST = os.path.join(RAGPY_ROOT, "conftest.py")
DEAD_PROXY = "http://127.0.0.1:9"
ALBERT_URL = "https://albert.api.etalab.gouv.fr/v1/models"
ALBERT_ENV_NAMES = ("OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER")


# ---------------------------------------------------------------------------
# Projets pytest temporaires (sous-processus)
# ---------------------------------------------------------------------------
def _subprocess_env(extra=None):
    """Environnement du sous-processus : sans variables Albert ni ``PYTHONPATH``, proxy mort."""
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith("ALBERT_") and name not in ALBERT_ENV_NAMES and name != "PYTHONPATH"
    }
    env.update({
        "HTTPS_PROXY": DEAD_PROXY,
        "HTTP_PROXY": DEAD_PROXY,
        "NO_PROXY": "127.0.0.1,localhost,testserver",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.update(extra or {})
    return env


def _make_project(root, files):
    """Crée un projet pytest isolé : copie du conftest racine, ``pytest.ini`` vide, fichiers donnés."""
    os.makedirs(root, exist_ok=True)
    shutil.copyfile(ROOT_CONFTEST, os.path.join(root, "conftest.py"))
    with open(os.path.join(root, "pytest.ini"), "w", encoding="utf-8") as fh:
        fh.write("[pytest]\n")
    for rel, content in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(content))


def _run_pytest(root, *, extra_env=None, args=()):
    """Lance pytest en sous-processus dans ``root`` et renvoie le ``CompletedProcess``."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml=report.xml", *args]
    return subprocess.run(cmd, cwd=str(root), env=_subprocess_env(extra_env), capture_output=True,
                          text=True, timeout=180)


def _junit_statuses(path):
    """``{nom du test: [statuts]}`` d'un rapport JUnit (``passed``, ``skipped``, ``failure``, ``error``)."""
    statuses = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        found = [child.tag for child in case if child.tag in ("failure", "error", "skipped")]
        statuses.setdefault(case.get("name"), []).extend(found or ["passed"])
    return statuses


INNER_APP = '''
    """Faux paquet ``app`` : photographie l'environnement Albert à son import."""
    import os

    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    NAMES = ("ALBERT_ENABLED", "OCR_ENABLE_ALBERT", "ALBERT_API_KEY", "EMBEDDING_PROVIDER", "ALBERT_LIMITER_BACKEND")
    SNAPSHOT = {name: os.environ.get(name) for name in NAMES}
'''

INNER_TESTS = {
    "app/__init__.py": INNER_APP,
    "test_env_probe.py": '''
        import os

        import app


        def test_env_neutral_at_app_import():
            assert app.SNAPSHOT == {
                "ALBERT_ENABLED": "0",
                "OCR_ENABLE_ALBERT": "0",
                "ALBERT_API_KEY": "",
                "EMBEDDING_PROVIDER": "",
                "ALBERT_LIMITER_BACKEND": "local",
            }


        def test_albert_env_removed_inside_test():
            leftovers = [n for n in os.environ if n.startswith("ALBERT_") or n in ("OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER")]
            assert leftovers == []
    ''',
    "test_live_probe.py": '''
        import pytest


        @pytest.mark.albert_live
        def test_live_must_not_run():
            raise AssertionError("un test live a été exécuté sans ALBERT_LIVE=1")
    ''',
    "test_swallow_probe.py": '''
        import httpx


        def test_swallowed_albert_call():
            try:
                httpx.get("https://albert.api.etalab.gouv.fr/v1/models", timeout=1.0)
            except Exception:
                pass
    ''',
    "test_legacy_external_probe.py": '''
        import requests


        def test_legacy_external_recorded_only():
            try:
                requests.get("https://legacy-host.invalid/ping", timeout=1.0)
            except requests.RequestException:
                pass
    ''',
    "test_albert_new_probe.py": '''
        import pytest
        import requests


        def test_new_file_external_blocked(network_guard):
            with pytest.raises(network_guard.error_class):
                requests.get("https://new-host.invalid/ping", timeout=1.0)
            assert [r.host for r in network_guard.consume()] == ["new-host.invalid"]
    ''',
}

INNER_EXTRA_ENV = {
    "ALBERT_ENABLED": "1",
    "OCR_ENABLE_ALBERT": "1",
    "ALBERT_API_KEY": FAKE_ALBERT_KEY,
    "EMBEDDING_PROVIDER": "albert",
    "ALBERT_LIMITER_BACKEND": "redis",
    "ALBERT_BASE_URL": "https://albert.api.etalab.gouv.fr/v1",
}


@pytest.fixture(scope="module")
def inner_run(tmp_path_factory):
    """Exécute une fois le projet temporaire principal ; renvoie (processus, statuts JUnit, racine)."""
    root = tmp_path_factory.mktemp("inner_project")
    _make_project(str(root), INNER_TESTS)
    with open(os.path.join(str(root), ".env"), "w", encoding="utf-8") as fh:
        fh.write("ALBERT_API_KEY=fake-dotenv-albert-key-0002\nALBERT_ENABLED=1\n")
    proc = _run_pytest(root, extra_env=INNER_EXTRA_ENV)
    report = os.path.join(str(root), "report.xml")
    assert os.path.exists(report), proc.stdout + proc.stderr
    return proc, _junit_statuses(report), root


# ---------------------------------------------------------------------------
# Marqueur, mode live, refus du .env
# ---------------------------------------------------------------------------
def test_marker_registered(request):
    markers = request.config.getini("markers")
    assert any(line.split(":", 1)[0].strip() == "albert_live" for line in markers)


@pytest.mark.albert_live
def test_live_marker_placeholder_is_skipped_here():
    raise AssertionError("ce test live ne doit jamais s'exécuter sans ALBERT_LIVE=1")


def test_live_skipped_without_ALBERT_LIVE(inner_run):
    proc, statuses, _root = inner_run
    assert statuses["test_live_must_not_run"] == ["skipped"], proc.stdout
    assert "ALBERT_LIVE" not in os.environ


def test_live_refused_if_env_file_has_ALBERT_LIVE(tmp_path):
    _make_project(str(tmp_path), {"test_trivial.py": "def test_ok():\n    assert True\n"})
    sentinels = ("fake-live-sentinel-7341", "fake-openai-sentinel-9127")
    with open(tmp_path / ".env", "w", encoding="utf-8") as fh:
        fh.write(f"OPENAI_API_KEY={sentinels[1]}\nALBERT_LIVE={sentinels[0]}\n")
    proc = _run_pytest(tmp_path)
    output = proc.stdout + proc.stderr
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "ALBERT_LIVE ne doit jamais figurer dans un fichier .env" in output
    for sentinel in sentinels:
        assert sentinel not in output


# ---------------------------------------------------------------------------
# Environnement neutralisé
# ---------------------------------------------------------------------------
def test_env_neutralised_before_app_import(inner_run):
    proc, statuses, _root = inner_run
    assert statuses["test_env_neutral_at_app_import"] == ["passed"], proc.stdout
    assert statuses["test_albert_env_removed_inside_test"] == ["passed"], proc.stdout
    assert "fake-dotenv-albert-key-0002" not in proc.stdout + proc.stderr
    # Dans cette session aussi : aucune variable Albert pendant un test.
    assert [n for n in os.environ if n.startswith("ALBERT_") or n in ALBERT_ENV_NAMES] == []


def test_env_neutral_values_restored_between_tests(monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "albert")
    assert os.environ["ALBERT_ENABLED"] == "1"


def test_env_not_leaked_from_previous_test():
    assert "ALBERT_ENABLED" not in os.environ
    assert "EMBEDDING_PROVIDER" not in os.environ


@pytest.fixture(scope="module")
def albert_constants_module():
    """Module factice « du dépôt » (``scripts/``) portant des constantes Albert à ON."""
    name = "scripts._albert_infra_constants_probe"
    module = types.ModuleType(name)
    module.__file__ = os.path.join(RAGPY_ROOT, "scripts", "_albert_infra_constants_probe.py")
    module.OCR_ENABLE_ALBERT = True
    module.ALBERT_API_KEY = FAKE_ALBERT_KEY
    sys.modules[name] = module
    yield module
    sys.modules.pop(name, None)


def test_module_constants_forced_off(albert_constants_module):
    assert albert_constants_module.OCR_ENABLE_ALBERT is False
    assert albert_constants_module.ALBERT_API_KEY is None


# ---------------------------------------------------------------------------
# Garde réseau
# ---------------------------------------------------------------------------
def test_network_guard_blocks_and_records_etalab(network_guard, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    error = network_guard.error_class
    with pytest.raises(error):
        httpx.get(ALBERT_URL, timeout=2.0)
    with pytest.raises(error):
        requests.get("https://albert.api.etalab.gouv.fr/health", timeout=2.0)

    async def call_async():
        """Requête async vers Albert."""
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get("https://albert.api.etalab.gouv.fr/v1/me")

    with pytest.raises(error):
        asyncio.run(call_async())
    blocked = network_guard.blocked()
    assert [r.host for r in blocked] == ["albert.api.etalab.gouv.fr"] * 3
    assert sorted(r.library for r in blocked) == ["httpx", "httpx-async", "requests"]
    assert all(r.reason == "albert" and r.blocked for r in blocked)
    assert network_guard.consume() == blocked
    assert network_guard.blocked() == []


def test_network_guard_blocks_other_hosts_in_new_tests(network_guard):
    with pytest.raises(network_guard.error_class):
        httpx.get("https://api.example.invalid/v1", timeout=1.0)
    assert [(r.host, r.reason) for r in network_guard.consume()] == [("api.example.invalid", "external")]


def test_guard_fails_test_even_if_exception_swallowed(inner_run):
    proc, statuses, _root = inner_run
    assert statuses["test_swallowed_albert_call"] == ["error"], proc.stdout
    assert "ERROR at teardown of test_swallowed_albert_call" in proc.stdout
    assert "Garde réseau" in proc.stdout
    assert "albert.api.etalab.gouv.fr" in proc.stdout


def test_legacy_external_hosts_only_counted(inner_run):
    proc, statuses, root = inner_run
    assert statuses["test_legacy_external_recorded_only"] == ["passed"], proc.stdout
    assert statuses["test_new_file_external_blocked"] == ["passed"], proc.stdout
    with open(os.path.join(str(root), "data", "albert_gates", "net_legacy.json"), encoding="utf-8") as fh:
        assert json.load(fh) == {"count": 1}


def test_mock_transport_unaffected(network_guard):
    fake = FakeAlbert()
    with httpx.Client(transport=fake.transport, headers={"Authorization": f"Bearer {FAKE_ALBERT_KEY}"}) as client:
        response = client.get(ALBERT_URL)
    assert response.status_code == 200
    with fake.client() as client:
        assert client.get("/me").status_code == 200
    raw = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(204)))
    assert raw.get("https://albert.api.etalab.gouv.fr/health").status_code == 204
    assert network_guard.records() == []


def test_local_hosts_allowed(network_guard):
    with httpx.Client(trust_env=False, timeout=1.0) as direct:
        with pytest.raises(httpx.ConnectError):
            direct.get("http://127.0.0.1:9/")
    assert network_guard.records() == []


# ---------------------------------------------------------------------------
# anyio / trio
# ---------------------------------------------------------------------------
def test_anyio_backend_is_asyncio(anyio_backend):
    assert anyio_backend == "asyncio"


@pytest.mark.anyio
async def test_anyio_marked_test_runs_on_asyncio(anyio_backend):
    import sniffio

    assert sniffio.current_async_library() == "asyncio"


def test_trio_variant_refused(tmp_path):
    _make_project(str(tmp_path), {
        "test_trio_probe.py": '''
            import pytest


            @pytest.mark.parametrize("backend", ["trio"])
            def test_backend(backend):
                assert backend
        ''',
    })
    proc = _run_pytest(tmp_path)
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR, proc.stdout + proc.stderr
    assert "trio" in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# FakeAlbert / FakeRedis
# ---------------------------------------------------------------------------
@pytest.fixture
def fake():
    """API Albert simulée."""
    return FakeAlbert()


@pytest.fixture
def client(fake):
    """Client httpx authentifié sur ``fake``."""
    with fake.client() as http:
        yield http


def test_fake_routes_cover_client_surface():
    paths = {path for _method, path in ROUTES}
    for path in ("/health", "/v1/me", "/v1/models", "/v1/chat/completions", "/v1/embeddings", "/v1/ocr",
                 "/v1/collections", "/v1/documents", "/v1/documents/{document_id}/chunks", "/v1/search"):
        assert path in paths
    fake_obj = FakeAlbert()
    for method, template in ROUTES:
        assert hasattr(fake_obj, "_route_" + fake_obj._handler_name(method, template)), (method, template)


def test_fake_models_me_health(fake, client):
    models = client.get("/models").json()
    assert models == {"object": "list", "data": load_fixture("P2_models.json")["body"]["data"]}
    assert client.get("/models/openweight-large").json()["id"] == "gpt-oss-120b"
    assert client.get("/models/does-not-exist").status_code == 404
    me = client.get("/me").json()
    assert me["object"] == "userInfo" and me["expires"] is None and me["limits"]
    assert FakeAlbert(me_overrides={"expires": 1800000000}).client().get("/me").json()["expires"] == 1800000000
    assert client.get("https://albert.api.etalab.gouv.fr/health").json() == {"status": "ok"}
    assert client.get("/health").status_code == 404
    with httpx.Client(transport=fake.transport) as anonymous:
        assert anonymous.get("https://albert.api.etalab.gouv.fr/health").status_code == 200
        unauthorised = anonymous.get(ALBERT_URL)
    assert unauthorised.status_code == 401
    assert unauthorised.json() == load_fixture("P6_invalid_key.json")["body"]


def test_fake_chat_usage_reasoning_and_image(fake, client):
    body = {"model": "ministral-3-8b-instruct-2512", "messages": [{"role": "user", "content": "Bonjour"}],
            "max_tokens": 8}
    data = client.post("/chat/completions", json=body).json()
    choice = data["choices"][0]
    assert choice["message"]["content"] == "OK." and choice["finish_reason"] == "stop"
    assert choice["message"]["reasoning"] is None
    assert {"cost", "impacts", "prompt_tokens", "completion_tokens"} <= set(data["usage"])
    assert data["model"] == "ministral-3-8b-instruct-2512"

    alias = client.post("/chat/completions", json=dict(body, model="openweight-small")).json()
    assert alias["model"] == "openweight-small"

    trap = client.post("/chat/completions", json=dict(body, model="gpt-oss-120b", max_tokens=64)).json()
    assert trap["choices"][0]["message"]["content"] is None
    assert trap["choices"][0]["finish_reason"] == "length"
    assert trap["choices"][0]["message"]["reasoning"]
    ok = client.post("/chat/completions", json=dict(body, model="gpt-oss-120b", max_tokens=2048)).json()
    assert ok["choices"][0]["message"]["content"] == "OK." and ok["choices"][0]["message"]["reasoning"]

    fake.queue_chat_replies("premier", {"content": "", "finish_reason": "length"})
    fake.chat_reply = lambda request_body: "rappel:" + request_body["messages"][0]["content"]
    replies = [client.post("/chat/completions", json=body).json()["choices"][0] for _ in range(3)]
    assert [r["message"]["content"] for r in replies] == ["premier", "", "rappel:Bonjour"]
    assert replies[1]["finish_reason"] == "length"

    image = {"model": "lightonocr-2-1b", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}], "max_tokens": 4096}
    assert client.post("/chat/completions", json=image).json()["choices"][0]["message"]["content"] == "PAGE_ONE_7Q"
    fake.ocr_chat_reply = lambda url, index: f"page-{index}"
    assert client.post("/chat/completions", json=image).json()["choices"][0]["message"]["content"] == "page-1"

    assert client.post("/chat/completions", json=dict(body, model="inconnu")).json() == {"detail": "Model not found."}
    assert client.post("/chat/completions", json=dict(body, model="bge-m3")).status_code == 422


def test_fake_embeddings_contract(fake, client):
    texts = [f"texte {i}" for i in range(64)]
    data = client.post("/embeddings", json={"model": "bge-m3", "input": texts, "encoding_format": "float"}).json()
    assert [item["index"] for item in data["data"]] == list(range(64))
    first = data["data"][0]["embedding"]
    assert len(first) == 1024 and first == fake_embedding("texte 0")
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)
    too_many = client.post("/embeddings", json={"model": "bge-m3", "input": texts + ["x"]})
    assert too_many.status_code == 413
    assert too_many.json() == load_fixture("P12_batch_65.json")["body"]
    empty = client.post("/embeddings", json={"model": "bge-m3", "input": ["ok", ""]})
    assert (empty.status_code, empty.json()) == (400, load_fixture("P12_empty_string.json")["body"])
    wrong = client.post("/embeddings", json={"model": "gpt-oss-120b", "input": ["x"]})
    assert (wrong.status_code, wrong.json()) == (422, load_fixture("P6_wrong_model_type.json")["body"])
    fake.embed_shuffle = True
    shuffled = client.post("/embeddings", json={"model": "openweight-embeddings", "input": ["a", "b", "c"]}).json()
    assert [item["index"] for item in shuffled["data"]] == [2, 1, 0]
    assert shuffled["model"] == "openweight-embeddings"


def test_fake_ocr_document(client):
    body = {"model": "mistral-ocr-2512", "document": {"type": "document_url",
                                                      "document_url": "data:application/pdf;base64,JVBERi0="},
            "pages": [0], "include_image_base64": False}
    denied = client.post("/ocr", json=body)
    assert (denied.status_code, denied.json()) == (404, load_fixture("P3_ocr_page0.json")["body"])
    with FakeAlbert(ocr_access=True).client() as allowed:
        pages = allowed.post("/ocr", json=dict(body, pages=[0, 2])).json()["pages"]
        assert [(p["index"], p["markdown"]) for p in pages] == [(0, "PAGE_ONE_7Q"), (2, "PAGE_3_7Q")]
        assert [p["index"] for p in allowed.post("/ocr", json=dict(body, pages="0-2")).json()["pages"]] == [0, 1, 2]


def test_fake_collections_documents_chunks(fake, client):
    cid = client.post("/collections", json={"name": "corpus", "description": "d", "visibility": "private"}).json()["id"]
    assert client.post("/collections", json={"name": "corpus"}).json()["id"] == cid + 1
    listed = client.get("/collections", params={"name": "corpus", "limit": 100}).json()["data"]
    assert [c["id"] for c in listed] == [cid, cid + 1] and listed[0]["visibility"] == "private"
    assert client.get("/collections", params={"limit": 101}).json() == load_fixture("P14_limit_101.json")["body"]

    created = client.post("/documents", files={"name": (None, "doc"), "collection_id": (None, str(cid))})
    assert created.status_code == 201
    did = created.json()["id"]
    call = fake.calls_to("POST", "/v1/documents")[-1]
    assert call.form == {"name": "doc", "collection_id": str(cid)} and call.files == {}
    assert client.post("/documents", files={"name": (None, "x"), "collection_id": (None, "999")}).status_code == 404

    chunks = [{"content": f"txt {i} : chunk de sonde", "metadata": {"content_hash": f"h{i}", "chunk_index": i,
                                                                     "title": "T", "year": 2026, "flag": True}}
              for i in range(64)]
    pushed = client.post(f"/documents/{did}/chunks", json={"chunks": chunks})
    assert pushed.status_code == 201 and pushed.json() == {"document_id": did, "ids": list(range(64))}
    assert client.post(f"/documents/{did}/chunks", json={"chunks": chunks + chunks[:1]}).status_code == 422
    for bad in ({f"k{i}": i for i in range(11)}, {"title": ""}, {"title": None}, {"tags": ["a"]}, {"t": "x" * 256}):
        response = client.post(f"/documents/{did}/chunks", json={"chunks": [{"content": "c", "metadata": bad}]})
        assert response.status_code == 422, bad
    assert client.get(f"/documents/{did}").json()["chunks"] == 64

    page1 = client.get(f"/documents/{did}/chunks", params={"limit": 50}).json()["data"]
    page2 = client.get(f"/documents/{did}/chunks", params={"limit": 50, "offset": 50}).json()["data"]
    assert [c["id"] for c in page1 + page2] == list(range(64))
    assert page1[0]["metadata"]["content_hash"] == "h0"
    assert client.get(f"/documents/{did}/chunks/3").json()["content"] == "txt 3 : chunk de sonde"
    assert client.delete(f"/documents/{did}/chunks/3").status_code == 204
    assert client.get(f"/documents/{did}/chunks/3").status_code == 404
    assert [d["id"] for d in client.get("/documents", params={"collection_id": cid}).json()["data"]] == [did]

    hit = client.post("/search", json={"query": "txt", "collection_ids": [cid], "method": "semantic", "limit": 10,
                                       "metadata_filters": {"key": "content_hash", "type": "eq", "value": "h7"}}).json()
    assert [item["chunk"]["id"] for item in hit["data"]] == [7] and "usage" in hit
    ors = {"operator": "or", "filters": [{"key": "content_hash", "type": "eq", "value": f"h{i}"} for i in (0, 1, 2, 4)]}
    assert len(client.post("/search", json={"query": "txt", "collection_ids": [cid], "method": "semantic",
                                            "metadata_filters": ors}).json()["data"]) == 4
    ors["filters"].append({"key": "content_hash", "type": "eq", "value": "h9"})
    assert client.post("/search", json={"query": "txt", "collection_ids": [cid], "method": "semantic",
                                        "metadata_filters": ors}).status_code == 422
    hybrid = client.post("/search", json={"query": "txt", "collection_ids": [cid], "method": "hybrid", "limit": 5,
                                          "rff_k": 20}).json()["data"]
    assert len(hybrid) == 5 and all(item["method"] == "hybrid" for item in hybrid)
    threshold = client.post("/search", json={"query": "txt", "collection_ids": [cid], "method": "hybrid",
                                             "score_threshold": 0.5})
    assert (threshold.status_code, threshold.json()) == (400, load_fixture("P17_hybrid_score_threshold.json")["body"])
    assert client.post("/search", json={"query": "txt", "collection_ids": [cid], "method": "exact"}).status_code == 422
    assert client.post("/search", json={"query": None, "collection_ids": [cid]}).status_code == 422

    assert client.delete(f"/collections/{cid}").status_code == 204
    assert client.get(f"/documents/{did}").status_code == 404
    assert client.get(f"/collections/{cid}").json() == load_fixture("P14_get_deleted_main.json")["body"]


def test_fake_injection_retry_after_and_errors(fake, client):
    fake.inject("POST", "/v1/chat/completions", 429, "model_busy", retry_after=3)
    fake.inject("GET", "/v1/me", (429, {"detail": "quota"}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}))
    fake.inject("GET", "/v1/models", "account_expired", {"status": 400, "body": {"detail": "InsufficientBudget"}})
    fake.inject("GET", "/v1/documents/*/chunks", httpx.ReadTimeout)
    body = {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "x"}]}
    first = client.post("/chat/completions", json=body)
    assert (first.status_code, first.headers["retry-after"]) == (429, "3")
    second = client.post("/chat/completions", json=body)
    assert second.status_code == 503 and "too busy" in second.json()["detail"]
    assert client.post("/chat/completions", json=body).status_code == 200
    assert client.get("/me").headers["retry-after"] == "Wed, 21 Oct 2026 07:28:00 GMT"
    expired = client.get("/models")
    assert expired.status_code == 403 and "account has expired" in expired.json()["detail"]
    assert client.get("/models").json() == {"detail": "InsufficientBudget"}
    with pytest.raises(httpx.ReadTimeout):
        client.get("/documents/1/chunks")
    assert fake.pending_injections() == 0
    injected = [c for c in fake.calls if c.injected]
    assert len(injected) == 6 and injected[-1].status is None
    assert all(c.headers["authorization"] == f"Bearer {FAKE_ALBERT_KEY}" for c in fake.calls)


def test_fake_async_client_and_custom_base(fake):
    async def run():
        """Deux requêtes async, dont une via une URL de base personnalisée."""
        async with fake.async_client() as http:
            first = await http.get("/models")
        async with fake.async_client(base_url="https://albert.example.org/api/v1") as http:
            second = await http.get("/me")
        return first.status_code, second.status_code

    assert asyncio.run(run()) == (200, 200)
    assert [c.path for c in fake.calls] == ["/v1/models", "/v1/me"]
    assert FAKE_BASE_URL.endswith("/v1")


def test_fake_redis_ttl_and_clock():
    now = [100.0]
    redis = FakeRedis(clock=lambda: now[0])
    assert redis.get("absent") is None and redis.ttl("absent") == -2
    assert redis.incr("rpm") == 1 and redis.incr("rpm", 2) == 3
    assert redis.ttl("rpm") == -1
    assert redis.expire("rpm", 60) is True and redis.ttl("rpm") == 60
    now[0] += 59.4
    assert redis.get("rpm") == b"3" and redis.ttl("rpm") == 1
    now[0] += 1.0
    assert redis.get("rpm") is None and redis.ttl("rpm") == -2
    assert redis.set("k", "v", ex=5) is True and redis.set("k", "w", nx=True) is None
    assert redis.get("k") == b"v"
    decoded = FakeRedis.from_url("redis://localhost:6379/0", decode_responses=True, clock=lambda: now[0])
    decoded.set("k", 7)
    assert decoded.get("k") == "7" and decoded.url == "redis://localhost:6379/0"
    pipe = redis.pipeline()
    assert pipe.incr("w").expire("w", 10).execute() == [1, True]
    assert redis.ttl("w") == 10
    assert redis.expire("absent", 5) is False
    assert ("incrby", ("rpm", 1)) in redis.calls
