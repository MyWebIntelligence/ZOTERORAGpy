"""Tests hors ligne de ``scripts/albert_probe.py`` (MockTransport, aucun .env, aucun réseau).

Couvre l'assainissement (en-têtes, Bearer, clé, e-mails, identité du compte,
idempotence), la garde ``ALBERT_LIVE``, un run P1+P2 simulé (fixtures
assainies et ``decisions.json`` D1 à D22), le nettoyage ``--cleanup`` et la
suppression garantie de la collection de sonde.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "albert_probe.py"
FAKE_ALBERT_KEY = "fake-albert-key-0001"
FAKE_WALL = 1790000000.0
DECISION_IDS = {"D%d" % i for i in range(1, 23)}


def _load_probe_module():
    """Charge le script de sondes comme module isolé (enregistré dans sys.modules)."""
    spec = importlib.util.spec_from_file_location("albert_probe_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()

ME_BODY = {
    "object": "userInfo", "id": 4242, "email": "prenom.testeur@exemple.gouv.fr", "name": "Prénom Testeur",
    "organization_id": 77, "budget": None, "permissions": [],
    "limits": [{"router_id": 3, "type": "rpm", "value": 100}, {"router_id": 3, "type": "rpd", "value": 50000}],
    "expires": None,
}
MODELS_BODY = {"object": "list", "data": [
    {"id": "gpt-oss-120b", "object": "model", "type": "text-generation", "owned_by": "support@exemple.org",
     "description": "catalogue consulté par Prénom Testeur",
     "aliases": ["openai/gpt-oss-120b", "openweight-large"], "max_context_length": 131072,
     "costs": {"prompt_tokens": 0.0, "completion_tokens": 0.0}},
    {"id": "ministral-3-8b-instruct-2512", "object": "model", "type": "image-text-to-text",
     "aliases": ["mistralai/Ministral-3-8B-Instruct-2512", "openweight-small"], "max_context_length": 262144},
    {"id": "lightonocr-2-1b", "object": "model", "type": "image-text-to-text",
     "aliases": ["lighton/LightOn-OCR-2-1B", "openweight-ocr"], "max_context_length": 16384},
    {"id": "bge-m3", "object": "model", "type": "text-embeddings-inference",
     "aliases": ["BAAI/bge-m3", "openweight-embeddings"], "max_context_length": 8192},
]}


class FakeClock:
    """Horloge factice : ``sleep`` avance le temps sans attendre."""

    def __init__(self):
        """Démarre le temps à zéro."""
        self.t = 0.0

    def now(self):
        """Temps courant factice."""
        return self.t

    def sleep(self, seconds):
        """Avance le temps factice."""
        self.t += seconds


class FakeAlbert:
    """API Albert simulée pour MockTransport : collections en mémoire, requêtes enregistrées."""

    def __init__(self, me_status=200, collections=None, honour_visibility=True):
        """Prépare l'état simulé."""
        self.requests = []
        self.me_status = me_status
        self.collections = {c["id"]: dict(c) for c in (collections or [])}
        self.honour_visibility = honour_visibility
        self.next_id = 500

    def handler(self, request):
        """Répond comme l'API pour les routes utilisées par les tests (écho volontaire de l'en-tête)."""
        self.requests.append(request)
        path = request.url.path
        auth = request.headers.get("authorization", "")
        params = request.url.params
        if path == "/v1/me":
            if self.me_status != 200:
                return httpx.Response(self.me_status, json={"detail": "Invalid authentication"})
            return httpx.Response(200, json=dict(ME_BODY, echo=auth),
                                  headers={"x-ratelimit-remaining-requests": "99", "x-echo": auth})
        if path == "/v1/models":
            return httpx.Response(200, json=dict(MODELS_BODY, echo=auth), headers={"x-echo": auth})
        if path == "/v1/collections" and request.method == "POST":
            payload = json.loads(request.content)
            cid = self.next_id
            self.next_id += 1
            self.collections[cid] = dict(payload, id=cid, owner="Prénom Testeur")
            return httpx.Response(201, json={"id": cid})
        if path == "/v1/collections" and request.method == "GET":
            limit = int(params.get("limit", "10"))
            if limit > 100:
                return httpx.Response(422, json={"detail": [{"loc": ["query", "limit"], "msg": "trop grand",
                                                             "type": "less_than_equal"}]})
            offset = int(params.get("offset", "0"))
            name = params.get("name")
            visibility = params.get("visibility") if self.honour_visibility else None
            items = [c for c in self.collections.values()
                     if (name is None or name in c["name"]) and (visibility is None or c["visibility"] == visibility)]
            return httpx.Response(200, json={"object": "list", "data": items[offset:offset + limit]})
        match = re.fullmatch(r"/v1/collections/(\d+)", path)
        if match:
            cid = int(match.group(1))
            if request.method == "DELETE":
                if self.collections.pop(cid, None) is None:
                    return httpx.Response(404, json={"detail": "Collection not found"})
                return httpx.Response(204)
            if cid in self.collections:
                return httpx.Response(200, json=self.collections[cid])
            return httpx.Response(404, json={"detail": "Collection not found"})
        return httpx.Response(404, json={"detail": "Not Found"})


def make_ctx(tmp_path, fake):
    """Contexte de sondes branché sur l'API simulée, horloges factices, journal muet."""
    clock = FakeClock()
    client = probe.build_client(FAKE_ALBERT_KEY, httpx.MockTransport(fake.handler))
    return probe.ProbeContext(client, FAKE_ALBERT_KEY, tmp_path / "run", tmp_path / "fixtures",
                              clock=clock.now, sleep=clock.sleep, wall=lambda: FAKE_WALL, log=lambda message: None)


def run_main(args, fake, monkeypatch, tmp_path):
    """Lance ``main`` dans ``tmp_path`` (sans .env) avec l'API simulée et une clé factice."""
    monkeypatch.chdir(tmp_path)
    clock = FakeClock()
    return probe.main(args, transport=httpx.MockTransport(fake.handler),
                      environ={"ALBERT_LIVE": "1", "ALBERT_API_KEY": FAKE_ALBERT_KEY},
                      clock=clock.now, sleep=clock.sleep, wall=lambda: FAKE_WALL)


def all_output_text(root):
    """Concatène le texte de tous les fichiers écrits sous ``root``."""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path(root).rglob("*")) if p.is_file())


def subprocess_env(**extra):
    """Environnement sans ALBERT_LIVE ni ALBERT_API_KEY, proxy mort, plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if k not in ("ALBERT_LIVE", "ALBERT_API_KEY")}
    env.update({"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"})
    env.update(extra)
    return env


# --- Assainissement -----------------------------------------------------------


def test_sanitize_masks_authorization_bearer_key_and_emails():
    payload = {
        "headers": {"Authorization": "Bearer " + FAKE_ALBERT_KEY, "Content-Type": "application/json",
                    "Set-Cookie": "session=abc"},
        "body": {"detail": "clé %s refusée pour jean.dupont@exemple.gouv.fr" % FAKE_ALBERT_KEY,
                 "trace": "Bearer fake-other-token-123456"},
    }
    clean = probe.sanitize_payload(payload, secrets=[FAKE_ALBERT_KEY])
    text = json.dumps(clean, ensure_ascii=False)
    assert FAKE_ALBERT_KEY not in text
    assert "fake-other-token" not in text
    assert "jean.dupont" not in text and "@exemple" not in text
    assert clean["headers"]["Authorization"] == probe.MASK
    assert clean["headers"]["Set-Cookie"] == probe.MASK
    assert clean["headers"]["Content-Type"] == "application/json"


def test_sanitize_masks_account_identity_but_keeps_useful_ids():
    payload = {
        "me": dict(ME_BODY, user_id=5),
        "models": MODELS_BODY,
        "collection": {"id": 42, "name": "ragpy-probe-1", "owner": "Prénom Testeur", "visibility": "private"},
        "key": {"object": "key", "id": 3, "name": "cle-projet", "value": "fake-albert-key-0002", "user_id": 5,
                "expires": None},
        "note": "créée par Prénom Testeur",
    }
    clean = probe.sanitize_payload(payload, identity=["Prénom Testeur", "Testeur"])
    for field in ("id", "email", "name", "organization_id", "user_id"):
        assert clean["me"][field] == probe.MASK
    assert clean["me"]["limits"][0] == {"router_id": 3, "type": "rpm", "value": 100}
    assert clean["models"]["data"][0]["id"] == "gpt-oss-120b"
    assert clean["collection"]["id"] == 42 and clean["collection"]["name"] == "ragpy-probe-1"
    assert clean["collection"]["owner"] == probe.MASK
    assert clean["key"]["value"] == probe.MASK and clean["key"]["name"] == probe.MASK
    assert "Testeur" not in json.dumps(clean, ensure_ascii=False)
    assert "support@" not in json.dumps(clean, ensure_ascii=False)


def test_sanitize_is_idempotent():
    payload = {
        "me": ME_BODY,
        "headers": {"authorization": "Bearer " + FAKE_ALBERT_KEY},
        "texte": "Bearer %s et prenom.testeur@exemple.gouv.fr, Prénom Testeur" % FAKE_ALBERT_KEY,
        "liste": [FAKE_ALBERT_KEY, {"email": "a.b@exemple.fr"}, 3, None, True],
    }
    kwargs = {"secrets": [FAKE_ALBERT_KEY], "identity": ["Prénom Testeur", "Testeur"],
              "replacements": [("/chemin/local", ".")]}
    once = probe.sanitize_payload(payload, **kwargs)
    twice = probe.sanitize_payload(once, **kwargs)
    assert once == twice
    assert FAKE_ALBERT_KEY not in json.dumps(once)


def test_request_summary_hides_base64_and_long_lists():
    body = {"document": {"document_url": "data:application/pdf;base64," + "A" * 5000},
            "chunks": [{"content": "c%d" % i} for i in range(64)]}
    summary = probe.summarize_request_body(body)
    assert "AAAA" not in json.dumps(summary)
    assert summary["document"]["document_url"].startswith("data:application/pdf;base64,<")
    assert len(summary["chunks"]) == 4
    compacted, notes = probe.compact_for_fixture({"data": [{"embedding": [0.1] * 1024}]})
    assert len(compacted["data"][0]["embedding"]) == probe.FIXTURE_FLOAT_KEEP
    assert notes == [{"chemin": "body.data[].embedding", "longueur_originale": 1024}]


# --- Garde ALBERT_LIVE ----------------------------------------------------------


def test_cli_refuses_without_albert_live(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), "--all"], cwd=str(tmp_path), env=subprocess_env(),
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 2
    assert "ALBERT_LIVE=1" in proc.stderr
    assert not (tmp_path / "data").exists()


def test_cli_refuses_when_env_file_contains_albert_live(tmp_path):
    (tmp_path / ".env").write_text("ALBERT_LIVE=1\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(SCRIPT), "--all"], cwd=str(tmp_path),
                          env=subprocess_env(ALBERT_LIVE="1"), capture_output=True, text=True, encoding="utf-8",
                          timeout=120)
    assert proc.returncode == 2
    assert ".env" in proc.stderr
    assert not (tmp_path / "data").exists()


def test_main_refuses_same_fixtures_and_diff_dirs(tmp_path, monkeypatch):
    fake = FakeAlbert()
    rc = run_main(["--all", "--fixtures-dir", "fx", "--diff-against", "fx"], fake, monkeypatch, tmp_path)
    assert rc == 2
    assert fake.requests == []


# --- Run P1 + P2 simulé -------------------------------------------------------------


def test_mocked_p1_p2_run_writes_sanitized_fixtures_and_decisions(tmp_path, monkeypatch, capsys):
    fake = FakeAlbert()
    rc = run_main(["--only", "P1,P2", "--out", "out", "--fixtures-dir", "fx", "--report", "rapport.md"],
                  fake, monkeypatch, tmp_path)
    captured = capsys.readouterr()
    assert FAKE_ALBERT_KEY not in captured.out and FAKE_ALBERT_KEY not in captured.err
    lines = captured.out.strip().splitlines()
    assert rc == 0
    assert lines[-1] == "SONDES: 2 exécutées, 0 erreur de script"
    assert fake.requests and all(r.headers.get("authorization") == "Bearer " + FAKE_ALBERT_KEY
                                 for r in fake.requests)

    fx = tmp_path / "fx"
    me = json.loads((fx / "P1_me.json").read_text(encoding="utf-8"))
    models = json.loads((fx / "P2_models.json").read_text(encoding="utf-8"))
    assert set(me) >= {"request", "status", "headers", "body"} and me["synthetic"] is False
    assert me["request"]["method"] == "GET" and me["request"]["path"] == "/v1/me" and me["status"] == 200
    assert set(me["headers"]) <= {"content-type", "retry-after"} | {h for h in me["headers"]
                                                                  if h.startswith("x-ratelimit")}
    assert "x-ratelimit-remaining-requests" in me["headers"] and "x-echo" not in me["headers"]
    for field in ("id", "email", "name", "organization_id"):
        assert me["body"][field] == probe.MASK
    assert me["body"]["limits"][0]["value"] == 100
    assert [m["id"] for m in models["body"]["data"]][:2] == ["gpt-oss-120b", "ministral-3-8b-instruct-2512"]

    decisions = json.loads((fx / "decisions.json").read_text(encoding="utf-8"))
    assert set(decisions) == DECISION_IDS
    for entry in decisions.values():
        assert set(entry) == {"value", "default", "source", "consumer"}
        assert entry["source"] in ("probe", "default") and entry["consumer"]
        assert entry["value"] is not None
    assert decisions["D1"]["source"] == "probe" and decisions["D2"]["source"] == "probe"
    assert decisions["D5"]["source"] == "default" and decisions["D21"]["source"] == "default"

    everything = all_output_text(tmp_path)
    assert FAKE_ALBERT_KEY not in everything
    assert "Bearer fake" not in everything
    assert "prenom.testeur" not in everything and "@exemple" not in everything
    assert "Testeur" not in everything
    assert str(tmp_path) not in (tmp_path / "rapport.md").read_text(encoding="utf-8")


def test_account_error_aborts_run(tmp_path, monkeypatch, capsys):
    fake = FakeAlbert(me_status=401)
    rc = run_main(["--only", "P1,P2", "--out", "out", "--fixtures-dir", "fx"], fake, monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert rc == 3
    assert "clé Albert invalide" in out
    assert out.strip().splitlines()[-1] == "SONDES: 1 exécutées, 0 erreur de script"
    assert [r.url.path for r in fake.requests] == ["/v1/me"]
    decisions = json.loads((tmp_path / "fx" / "decisions.json").read_text(encoding="utf-8"))
    assert set(decisions) == DECISION_IDS


def test_diff_against_reports_changes_and_never_overwrites(tmp_path, monkeypatch, capsys):
    fake = FakeAlbert()
    assert run_main(["--only", "P1,P2", "--out", "out", "--fixtures-dir", "committed"], fake, monkeypatch,
                    tmp_path) == 0
    committed = tmp_path / "committed" / "P2_models.json"
    fixture = json.loads(committed.read_text(encoding="utf-8"))
    fixture["body"]["data"] = fixture["body"]["data"][:2]
    committed.write_text(json.dumps(fixture), encoding="utf-8")
    snapshot = {p.name: p.read_bytes() for p in (tmp_path / "committed").iterdir()}
    capsys.readouterr()
    assert run_main(["--only", "P1,P2", "--out", "out2", "--diff-against", "committed"], fake, monkeypatch,
                    tmp_path) == 0
    out = capsys.readouterr().out
    assert "DIFF: P2_models.json : ids retirés" in out
    assert {p.name: p.read_bytes() for p in (tmp_path / "committed").iterdir()} == snapshot


# --- Nettoyage ---------------------------------------------------------------------

CLEANUP_COLLECTIONS = [
    {"id": 1, "name": "ragpy-probe-111", "visibility": "private"},
    {"id": 2, "name": "ragpy-probe-e2e-222", "visibility": "private"},
    {"id": 3, "name": "mon-corpus", "visibility": "private"},
    {"id": 4, "name": "ragpy-probe-publique", "visibility": "public"},
    {"id": 5, "name": "xragpy-probe-5", "visibility": "private"},
]


def test_cleanup_dry_run_counts_only_private_probe_collections(tmp_path, monkeypatch, capsys):
    fake = FakeAlbert(collections=CLEANUP_COLLECTIONS, honour_visibility=False)
    rc = run_main(["--cleanup", "--dry-run"], fake, monkeypatch, tmp_path)
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[-1] == "ragpy-probe restantes: 2"
    assert not [r for r in fake.requests if r.method != "GET"]
    assert len(fake.collections) == 5


def test_cleanup_deletes_probe_collections_and_reports_zero(tmp_path, monkeypatch, capsys):
    fake = FakeAlbert(collections=CLEANUP_COLLECTIONS, honour_visibility=False)
    rc = run_main(["--cleanup"], fake, monkeypatch, tmp_path)
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[-1] == "ragpy-probe restantes: 0"
    assert sorted(fake.collections) == [3, 4, 5]


# --- Collection de sonde toujours supprimée ------------------------------------------


def test_collection_probes_delete_collection_even_when_a_probe_raises(tmp_path, monkeypatch):
    fake = FakeAlbert()
    ctx = make_ctx(tmp_path, fake)

    def boom(context):
        """Sonde qui échoue par une exception Python."""
        raise RuntimeError("panne simulée")

    monkeypatch.setitem(probe.COLLECTION_PROBES, "P16", boom)
    probe.run_collection_group(ctx, ["P14", "P16"])
    assert fake.collections == {}
    assert ctx.created_collections == []
    assert ctx.script_errors == 1
    assert ctx.results["P16"]["statut"] == "erreur"
    assert ctx.results["P14"]["constats"]["doublon_autorise"] is True
    created = [json.loads(r.content) for r in fake.requests if r.method == "POST" and r.url.path == "/v1/collections"]
    assert len(created) == 2
    assert all(body["visibility"] == "private" and body["name"].startswith("ragpy-probe-") for body in created)
    deleted = {r.url.path for r in fake.requests if r.method == "DELETE"}
    assert deleted == {"/v1/collections/500", "/v1/collections/501"}


def test_collection_group_cleans_up_on_account_error(tmp_path, monkeypatch):
    fake = FakeAlbert()
    ctx = make_ctx(tmp_path, fake)

    def budget_exhausted(context):
        """Sonde qui rencontre une erreur de compte."""
        raise probe.AccountLevelError("budget Albert épuisé (HTTP 400)")

    monkeypatch.setitem(probe.COLLECTION_PROBES, "P15", budget_exhausted)
    message = probe.run_probes(ctx, ["P15", "P20"])
    assert message and "budget" in message
    assert fake.collections == {}
    assert "P20" not in ctx.results
    assert ctx.results["P15"]["statut"] == "interrompu"
    assert [r.method for r in fake.requests if r.url.path == "/v1/collections/500"] == ["DELETE", "GET"]


# --- Run complet simulé (P1 à P21) ------------------------------------------------------

KNOWN_CHAT_MODELS = {
    "gpt-oss-120b": "gpt-oss-120b", "openweight-large": "gpt-oss-120b",
    "ministral-3-8b-instruct-2512": "ministral-3-8b-instruct-2512",
    "openweight-small": "ministral-3-8b-instruct-2512",
    "lightonocr-2-1b": "lightonocr-2-1b", "openweight-ocr": "lightonocr-2-1b",
}
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0,
         "impacts": {"kWh": 0.0, "kgCO2eq": 0.0}, "requests": 1}
RATE_HEADERS = {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "99"}


class FullFakeAlbert(FakeAlbert):
    """API simulée couvrant toutes les routes des sondes P1 à P21."""

    def __init__(self, ocr_access=False):
        """Ajoute documents, chunks et compteurs d'usage à l'état simulé."""
        super().__init__()
        self.ocr_access = ocr_access
        self.documents = {}
        self.chunks = []
        self.embed_requests = 0

    def handler(self, request):
        """Aiguille vers la route simulée ; refuse toute clé autre que la clé factice."""
        path = request.url.path
        if path == "/health":
            self.requests.append(request)
            return httpx.Response(200, json={"status": "ok"})
        if request.headers.get("authorization") != "Bearer " + FAKE_ALBERT_KEY:
            self.requests.append(request)
            return httpx.Response(401, json={"detail": "Invalid authentication credentials"})
        routes = {"/v1/ocr": self._ocr, "/v1/chat/completions": self._chat, "/v1/embeddings": self._embed,
                  "/v1/search": self._search, "/v1/usage": self._usage, "/health/models": self._health_models}
        if path in routes:
            self.requests.append(request)
            return routes[path](request)
        if path == "/v1/documents" or path.startswith("/v1/documents/"):
            self.requests.append(request)
            return self._documents(request)
        if path == "/v1/collections" and request.method == "GET" and request.url.params.get("visibility") == "public":
            self.requests.append(request)
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": 999, "name": "mediatech-legifrance", "visibility": "public"}]})
        return super().handler(request)

    def _ocr(self, request):
        """/v1/ocr : 403 sans accès, sinon une page par indice (markdown échappé)."""
        if not self.ocr_access:
            return httpx.Response(403, json={"detail": "Insufficient permissions to access model"})
        pages = json.loads(request.content)["pages"]
        text = {0: "PAGE\\_ONE\\_7Q", 1: "PAGE\\_TWO\\_9Z"}
        return httpx.Response(200, json={
            "id": "ocr-1", "model": "mistral-ocr-2512",
            "pages": [{"index": i, "markdown": text.get(i, "page %d" % i)} for i in pages],
            "usage_info": {"pages_processed": len(pages), "doc_size_bytes": len(request.content)}, "usage": USAGE})

    def _chat(self, request):
        """/v1/chat/completions : OCR, JSON, raisonnement et bornes simulés."""
        body = json.loads(request.content)
        model = KNOWN_CHAT_MODELS.get(body.get("model"))
        if model is None:
            return httpx.Response(404, json={"detail": "Model not found"})
        content_in = body["messages"][-1]["content"]
        bound = body.get("max_tokens") or body.get("max_completion_tokens")
        message = {"role": "assistant", "content": "Texte recodé propre."}
        finish, completion = "stop", 40
        if isinstance(content_in, list):
            message["content"] = "PAGE\\_ONE\\_7Q"
        elif body.get("response_format"):
            message["content"] = json.dumps({"keep": [1, 3, 5]})
        elif model == "gpt-oss-120b":
            message["reasoning_content"] = "17*23 = 391"
            completion = 300 if body.get("reasoning_effort") == "high" else 120
            if bound is not None and bound <= 64:
                message["content"], finish, completion = None, "length", bound
            else:
                message["content"] = "391"
        if bound is not None and bound <= 5:
            finish, completion = "length", bound
        usage = dict(USAGE, completion_tokens=completion)
        return httpx.Response(200, headers=RATE_HEADERS, json={
            "id": "chat-1", "object": "chat.completion", "model": model, "system_fingerprint": "fp-1",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})

    def _embed(self, request):
        """/v1/embeddings : 422 mauvais type, lot > 64 ou chaîne vide ; sinon vecteurs 1024 normés."""
        body = json.loads(request.content)
        inputs = body["input"]
        if body["model"] not in ("bge-m3", "openweight-embeddings"):
            return httpx.Response(422, json={"detail": [{"loc": ["body", "model"], "msg": "Wrong model type",
                                                         "type": "value_error"}]})
        if len(inputs) > 64 or "" in inputs:
            return httpx.Response(422, json={"detail": [{"loc": ["body", "input"], "msg": "invalid input",
                                                         "type": "value_error"}]})
        vector = [1.0 / 32.0] * 1024
        return httpx.Response(200, headers=RATE_HEADERS, json={
            "object": "list", "model": "bge-m3", "usage": USAGE,
            "data": [{"object": "embedding", "index": i, "embedding": vector} for i in range(len(inputs))]})

    @staticmethod
    def _meta_ok(meta):
        """Règles de métadonnées d'un chunk (10 propriétés, clés et chaînes de 1 à 255)."""
        if len(meta) > 10:
            return False
        for key, value in meta.items():
            if not 1 <= len(key) <= 255 or value is None or isinstance(value, (list, dict)):
                return False
            if isinstance(value, str) and not 1 <= len(value) <= 255:
                return False
        return True

    def _documents(self, request):
        """Documents et chunks (multipart exigé à la création, envoi atomique)."""
        path = request.url.path
        if path == "/v1/documents" and request.method == "POST":
            if not request.headers.get("content-type", "").startswith("multipart/form-data"):
                return httpx.Response(422, json={"detail": [{"msg": "multipart attendu"}]})
            raw = request.content.decode("utf-8")
            fields = dict(re.findall(r'name="([^"]+)"\r\n\r\n(.*?)\r\n', raw, re.S))
            doc_id = self.next_id
            self.next_id += 1
            self.documents[doc_id] = {"id": doc_id, "name": fields.get("name"),
                                      "collection_id": int(fields["collection_id"]),
                                      "metadata": json.loads(fields["metadata"]) if fields.get("metadata") else {}}
            return httpx.Response(201, json={"id": doc_id})
        match = re.fullmatch(r"/v1/documents/(\d+)(/chunks)?", path)
        doc = self.documents.get(int(match.group(1))) if match else None
        if doc is None:
            return httpx.Response(404, json={"detail": "Document not found"})
        own = [c for c in self.chunks if c["document_id"] == doc["id"]]
        if match.group(2) and request.method == "POST":
            items = json.loads(request.content)["chunks"]
            merged = [dict(doc["metadata"], **(item.get("metadata") or {})) for item in items]
            if not 1 <= len(items) <= 64 or not all(self._meta_ok(m) for m in merged):
                return httpx.Response(422, json={"detail": [{"msg": "invalid chunks"}]})
            for item, meta in zip(items, merged):
                self.chunks.append({"id": len(own), "collection_id": doc["collection_id"],
                                    "document_id": doc["id"], "content": item["content"], "metadata": meta})
                own.append(self.chunks[-1])
            self.embed_requests += 1
            return httpx.Response(201, json=None)
        if match.group(2):
            return httpx.Response(200, json={"object": "list", "data": own[:100]})
        return httpx.Response(200, json={"object": "document", "id": doc["id"], "name": doc["name"],
                                         "collection_id": doc["collection_id"], "created": 0,
                                         "chunks": len(own), "size": 0})

    def _search(self, request):
        """/v1/search : méthodes, seuil, filtres composés (2 à 4), rrf_k refusé."""
        body = json.loads(request.content)
        if body.get("method") not in ("semantic", "lexical", "hybrid") or "rrf_k" in body:
            return httpx.Response(422, json={"detail": [{"msg": "validation"}]})
        if body.get("score_threshold") and body["method"] != "semantic":
            return httpx.Response(422, json={"detail": [{"msg": "score_threshold"}]})
        filters = body.get("metadata_filters")
        if filters and "filters" in filters and not 2 <= len(filters["filters"]) <= 4:
            return httpx.Response(422, json={"detail": [{"msg": "2 à 4 filtres"}]})

        def keep(chunk):
            """Filtre eq simple ou composé (or)."""
            if not filters:
                return True
            subs = filters.get("filters", [filters])
            return any(chunk["metadata"].get(f["key"]) == f["value"] for f in subs)

        hits = [c for c in self.chunks if c["collection_id"] in body.get("collection_ids", []) and keep(c)]
        if body["method"] == "lexical" and body.get("query"):
            hits = [c for c in hits if body["query"] in c["content"]]
        hits = hits[:body.get("limit", 10)]
        return httpx.Response(200, json={"object": "list", "usage": USAGE,
                                         "data": [{"method": body["method"], "score": 0.9, "chunk": c}
                                                  for c in hits]})

    def _usage(self, request):
        """/v1/usage : un seau journalier ; les envois de chunks comptent comme embeddings."""
        requests_count = self.embed_requests if request.url.params.get("endpoint") == "/v1/embeddings" else 0
        return httpx.Response(200, json={"object": "list", "data": [
            {"object": "usage.bucket", "requests": requests_count, "prompt_tokens": 64 * requests_count}]})

    def _health_models(self, request):
        """/health/models : état par modèle."""
        return httpx.Response(200, json={"models": [{"id": "bge-m3", "status": "healthy"}]})


def test_full_mocked_run_executes_21_probes_without_script_error(tmp_path, monkeypatch, capsys):
    fake = FullFakeAlbert(ocr_access=False)
    rc = run_main(["--all", "--out", "out", "--fixtures-dir", "fx", "--report", "rapport.md"], fake, monkeypatch,
                  tmp_path)
    captured = capsys.readouterr()
    assert FAKE_ALBERT_KEY not in captured.out and FAKE_ALBERT_KEY not in captured.err
    out = captured.out.strip().splitlines()
    assert out[-1] == "SONDES: 21 exécutées, 0 erreur de script"
    assert rc == 0
    fx = tmp_path / "fx"
    decisions = json.loads((fx / "decisions.json").read_text(encoding="utf-8"))
    defaults = sorted(did for did, entry in decisions.items() if entry["source"] == "default")
    assert defaults == ["D21", "D4"]
    assert decisions["D3"]["value"]["ocr_access"] is False
    assert decisions["D16"]["value"]["atomic_post"] is True
    assert decisions["D19"]["value"]["push_counts_embed_quota"] is True
    assert (fx / "P3_ocr_page0_synthetic.json").exists()
    assert json.loads((fx / "P3_ocr_page0_synthetic.json").read_text(encoding="utf-8"))["synthetic"] is True
    assert fake.collections == {}
    assert FAKE_ALBERT_KEY not in all_output_text(tmp_path)
    assert "Testeur" not in all_output_text(tmp_path)


def test_full_mocked_run_with_ocr_access_measures_p4(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe, "P4_SIZES_MB", (1, 2))
    fake = FullFakeAlbert(ocr_access=True)
    rc = run_main(["--only", "P3,P4", "--out", "out", "--fixtures-dir", "fx"], fake, monkeypatch, tmp_path)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "SONDES: 2 exécutées, 0 erreur de script"
    assert rc == 0
    decisions = json.loads((tmp_path / "fx" / "decisions.json").read_text(encoding="utf-8"))
    assert decisions["D3"]["value"]["pages_zero_based"] is True
    assert decisions["D4"]["source"] == "probe"
    assert decisions["D4"]["value"]["ALBERT_OCR_PART_PAGES"] == 100
    assert not (tmp_path / "fx" / "P3_ocr_page0_synthetic.json").exists()


# --- Retry 429 et identité hors P1 ------------------------------------------------------


def test_call_honours_retry_after_and_records_natural_429(tmp_path):
    calls = []

    def handler(request):
        """429 avec Retry-After au premier appel, puis 200."""
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"detail": "Too many requests (rpm)"})
        return httpx.Response(200, json={"object": "list", "data": []})

    fake = FakeAlbert()
    fake.handler = handler
    ctx = make_ctx(tmp_path, fake)
    slept = []
    ctx.sleep = slept.append
    ex = ctx.call("P13", "chat_headers", "GET", "/models")
    assert ex.status == 200 and len(calls) == 2
    assert slept == [7.0]
    assert ctx.rate_info["retry_after_formats"] == {"delta"}
    assert (tmp_path / "fixtures" / "P13_natural_429.json").exists()
    seconds, fmt = probe.parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT", 1792567650.0)
    assert fmt == "http-date" and seconds == 30.0


def test_only_run_without_p1_prefetches_identity_before_writing(tmp_path, monkeypatch, capsys):
    fake = FakeAlbert()
    rc = run_main(["--only", "P2", "--out", "out", "--fixtures-dir", "fx"], fake, monkeypatch, tmp_path)
    assert rc == 0
    assert [r.url.path for r in fake.requests] == ["/v1/me", "/v1/models"]
    assert not (tmp_path / "fx" / "P0_identity_prefetch.json").exists()
    everything = all_output_text(tmp_path)
    assert "Testeur" not in everything and FAKE_ALBERT_KEY not in everything


# --- Fixtures versionnées : aucune valeur ressemblant à une clé --------------------------

FIXTURES_ROOT = ROOT / "tests" / "fixtures" / "albert"

# Empreinte tronquée utilisée par les goldens (golden_off/) : valeur déjà non réversible.
FINGERPRINT_RE = re.compile(r"sha12:[0-9a-f]{12}")


def key_like_patterns():
    """Motifs de valeurs secrètes (préfixes assemblés par morceaux pour ne jamais les écrire en clair)."""
    return [
        ("jeton Bearer", re.compile(r"\b" + "Bea" + r"rer\s+(?!<)[A-Za-z0-9._~+/=-]{20,}")),
        ("clé de style OpenAI", re.compile("(?<![A-Za-z])" + re.escape("s" + "k-") + "[A-Za-z0-9_-]{16,}")),
        ("clé de style Pinecone", re.compile("(?<![A-Za-z])" + re.escape("pc" + "sk" + "_"))),
        ("JWT", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
        ("adresse e-mail", probe.EMAIL_RE),
    ]


def authorization_values(node):
    """Valeurs de toutes les clés ``authorization`` (toute casse) d'une structure JSON."""
    found = []
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str) and key.lower() == "authorization":
                    found.append(value)
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return found


def test_committed_fixtures_contain_no_key_like_values():
    files = sorted(p for p in FIXTURES_ROOT.rglob("*") if p.is_file() and p.suffix in (".json", ".txt", ".md"))
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT)
        for label, pattern in key_like_patterns():
            if pattern.search(text):
                hits.append("%s : %s" % (rel, label))
        if path.suffix == ".json":
            for value in authorization_values(json.loads(text)):
                if value != probe.MASK and not FINGERPRINT_RE.fullmatch(str(value)):
                    hits.append("%s : en-tête authorization non masqué" % rel)
    assert hits == []


def test_key_like_patterns_catch_obvious_leaks():
    samples = {
        "jeton Bearer": "Bea" + "rer " + "A" * 24,
        "clé de style OpenAI": "s" + "k-" + "b" * 20,
        "clé de style Pinecone": "pc" + "sk" + "_" + "c" * 20,
        "JWT": "eyJ" + "d" * 12 + "." + "e" * 12,
        "adresse e-mail": "quelqu.un@exemple.gouv.fr",
    }
    for label, pattern in key_like_patterns():
        assert pattern.search(samples[label]), label
    assert not key_like_patterns()[0][1].search("Bea" + "rer " + probe.MASK)


# --- Collection dont la réponse de création se perd ---------------------------------------


class LossyCollectionsAlbert(FakeAlbert):
    """API simulée dont la création de collection s'applique côté serveur mais répond mal."""

    def __init__(self, mode):
        """``mode`` : timeout_main, timeout_duplicate ou string_id."""
        super().__init__()
        self.mode = mode
        self.creations = 0

    def handler(self, request):
        """Enregistre la collection, puis perd la réponse ou renvoie un id en chaîne selon le mode."""
        if request.url.path == "/v1/collections" and request.method == "POST":
            self.creations += 1
            response = super().handler(request)
            cid = json.loads(response.content)["id"]
            if self.mode == "string_id":
                return httpx.Response(201, json={"id": str(cid)})
            if (self.mode == "timeout_main" and self.creations == 1) or \
                    (self.mode == "timeout_duplicate" and self.creations == 2):
                raise httpx.ReadTimeout("réponse perdue", request=request)
            return response
        return super().handler(request)


@pytest.mark.parametrize("mode, selected", [("timeout_main", ["P15"]), ("timeout_duplicate", ["P14"]),
                                            ("string_id", ["P14"])])
def test_collection_with_lost_or_odd_create_response_is_still_deleted(tmp_path, mode, selected):
    fake = LossyCollectionsAlbert(mode)
    ctx = make_ctx(tmp_path, fake)
    assert probe.run_probes(ctx, selected) is None
    assert fake.creations >= 1
    assert fake.collections == {}
    assert ctx.created_collections == []
    report = probe.build_report(ctx, {}, None, [])
    assert "aucune collection créée" not in report
    if mode == "timeout_main":
        assert ctx.state["balayage"]["supprimees"] == [500]
        assert "création tentée (sans réponse)" in report
    if mode == "string_id":
        assert ctx.results["P14"]["constats"]["type_id_creation"] == "str"
        assert ctx.state["balayage"]["supprimees"] == []


def test_report_says_no_collection_when_none_was_attempted(tmp_path):
    ctx = make_ctx(tmp_path, FakeAlbert())
    assert "- aucune collection créée" in probe.build_report(ctx, {}, None, [])


class FailingMainDeleteAlbert(FakeAlbert):
    """API simulée dont le premier DELETE de la collection principale échoue (500)."""

    def __init__(self):
        """Prépare le compteur d'échecs."""
        super().__init__()
        self.failed = False

    def handler(self, request):
        """Répond 500 au premier DELETE de la collection 500, normalement ensuite."""
        if request.method == "DELETE" and request.url.path == "/v1/collections/500" and not self.failed:
            self.failed = True
            return httpx.Response(500, json={"detail": "erreur interne"})
        return super().handler(request)


def test_main_collection_teardown_is_asserted_in_p14(tmp_path):
    fake = FakeAlbert()
    ctx = make_ctx(tmp_path, fake)
    probe.run_collection_group(ctx, ["P14"])
    labels = {a["attendu"]: a["ok"] for a in ctx.results["P14"]["assertions"]}
    assert labels[probe.MAIN_DELETE_LABEL] is True and labels[probe.MAIN_GET_LABEL] is True
    assert ctx.results["P14"]["statut"] == "ok"

    fake = FailingMainDeleteAlbert()
    ctx = make_ctx(tmp_path / "echec", fake)
    probe.run_collection_group(ctx, ["P14"])
    labels = {a["attendu"]: a["ok"] for a in ctx.results["P14"]["assertions"]}
    assert labels[probe.MAIN_DELETE_LABEL] is False
    assert ctx.results["P14"]["statut"] == "écart"
    assert fake.collections == {}
    assert ctx.state["balayage"]["supprimees"] == [500]


# --- P20 sans clé, quota journalier -------------------------------------------------------


def test_p20_unauthenticated_health_401_is_not_an_account_error(tmp_path):
    fake = FakeAlbert()
    base = fake.handler

    def handler(request):
        """/health répond 401 sans clé ; le reste comme l'API simulée."""
        if request.url.path == "/health":
            fake.requests.append(request)
            return httpx.Response(401, json={"detail": "Not authenticated"})
        if request.url.path == "/health/models":
            fake.requests.append(request)
            return httpx.Response(200, json={"models": []})
        return base(request)

    fake.handler = handler
    ctx = make_ctx(tmp_path, fake)
    assert probe.run_probes(ctx, ["P20"]) is None
    assert ctx.results["P20"]["constats"]["health"] == 401
    health = [r for r in fake.requests if r.url.path == "/health"]
    assert len(health) == 1 and "authorization" not in health[0].headers


def test_call_stops_on_daily_quota_retry_after(tmp_path):
    calls = []

    def handler(request):
        """429 avec un Retry-After d'une heure (quota journalier)."""
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "3600"}, json={"detail": "Too many requests (rpd)"})

    fake = FakeAlbert()
    fake.handler = handler
    ctx = make_ctx(tmp_path, fake)
    slept, logs = [], []
    ctx.sleep = slept.append
    ctx._log = logs.append
    with pytest.raises(probe.AccountLevelError, match="quota journalier atteint"):
        ctx.call("P7", "chat_usage", "GET", "/models")
    assert len(calls) == 1 and slept == []
    assert any("quota journalier atteint (Retry-After 3600 s)" in line for line in logs)
    ex = ctx.call("P14", "delete_main", "DELETE", "/collections/1", expect_auth_error=True)
    assert ex.status == 429 and len(calls) == 2 and slept == []


# --- Identité : pas de corruption du vocabulaire ------------------------------------------


def test_identity_with_generic_words_keeps_fixture_keys_and_values(tmp_path):
    body = {"object": "list", "data": [{"role": "user", "content": "Data User a écrit ceci."}],
            "detail": "Albert API ok", "model": "bge-m3"}

    def handler(request):
        """/v1/me avec un nom générique, puis une réponse riche en vocabulaire d'API."""
        if request.url.path == "/v1/me":
            return httpx.Response(200, json=dict(ME_BODY, name="Data User", email="data.user@exemple.fr"))
        return httpx.Response(200, json=body)

    fake = FakeAlbert()
    fake.handler = handler
    ctx = make_ctx(tmp_path, fake)
    ctx.call("P0", "identity_prefetch", "GET", "/me", record=False)
    assert "Data User" in ctx.identity and "data.user" in ctx.identity
    assert not {"Data", "User"} & ctx.identity
    ctx.call("P2", "models", "GET", "/models")
    fixture = json.loads((tmp_path / "fixtures" / "P2_models.json").read_text(encoding="utf-8"))
    assert fixture["body"]["data"][0]["role"] == "user"
    assert fixture["body"]["detail"] == "Albert API ok"
    assert fixture["body"]["data"][0]["content"] == probe.MASK + " a écrit ceci."
    assert set(fixture["body"]) == {"object", "data", "detail", "model"}
    raw = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "run" / "raw").glob("*.json")]
    assert all(r["request"]["headers"]["user-agent"] == probe.USER_AGENT for r in raw)
    assert "albert.api.etalab.gouv.fr" in raw[0]["request"]["url"]
    assert probe.sanitize_payload({"Testeur": "Testeur"}, identity=["Testeur"]) == {"Testeur": probe.MASK}


# --- Clé mal formée, exception pendant --cleanup -----------------------------------------------


def test_malformed_key_is_refused_without_echo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake = FakeAlbert()
    bad_key = "fake-albert-key" + "\x07" + "0001"
    rc = probe.main(["--cleanup", "--dry-run"], transport=httpx.MockTransport(fake.handler),
                    environ={"ALBERT_LIVE": "1", "ALBERT_API_KEY": bad_key})
    captured = capsys.readouterr()
    assert rc == 2
    assert "mal formée" in captured.err
    assert "fake-albert-key" not in captured.out + captured.err
    assert fake.requests == []
    assert probe.load_api_key({"ALBERT_API_KEY": "  " + FAKE_ALBERT_KEY + "\n"}) == (FAKE_ALBERT_KEY, None)
    key, refusal = probe.load_api_key({})
    assert key is None and "absente" in refusal


def test_cleanup_unexpected_exception_prints_only_its_type(tmp_path, monkeypatch, capsys):
    def handler(request):
        """Panne inattendue dont le message contient la clé."""
        raise RuntimeError("en-tête refusé : " + request.headers.get("authorization", ""))

    fake = FakeAlbert()
    fake.handler = handler
    rc = run_main(["--cleanup", "--dry-run"], fake, monkeypatch, tmp_path)
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err.strip() == "Arrêt : RuntimeError"
    assert FAKE_ALBERT_KEY not in captured.out + captured.err


# --- Fixtures synthétiques et fixtures périmées --------------------------------------------------


def test_synthetic_fixtures_cover_every_documented_endpoint(tmp_path):
    ctx = make_ctx(tmp_path, FakeAlbert())
    ctx.executed_ids.extend(["P2", "P16", "P19", "P20"])
    written = probe.write_synthetic_fixtures(ctx)
    assert sorted(written) == ["P16_push_64_synthetic.json", "P19_usage_embeddings_before_synthetic.json",
                               "P20_health_models_synthetic.json", "P2_models_synthetic.json"]
    for name in written:
        fixture = json.loads((tmp_path / "fixtures" / name).read_text(encoding="utf-8"))
        assert fixture["synthetic"] is True and fixture["source"].startswith("référence API Albert")
        assert isinstance(fixture["status"], int) and 200 <= fixture["status"] < 300
    push = json.loads((tmp_path / "fixtures" / "P16_push_64_synthetic.json").read_text(encoding="utf-8"))
    assert push["status"] == 201 and push["body"] is None and "D16" in push["forme_corps"]
    models = json.loads((tmp_path / "fixtures" / "P2_models_synthetic.json").read_text(encoding="utf-8"))
    assert {m["id"] for m in models["body"]["data"]} >= set(probe.EXPECTED_MODELS)
    usage = json.loads((tmp_path / "fixtures" / "P19_usage_embeddings_before_synthetic.json").read_text("utf-8"))
    assert usage["body"]["data"][0]["object"] == "usage.bucket"


def test_stale_fixtures_of_executed_probes_are_pruned(tmp_path, monkeypatch, capsys):
    fx = tmp_path / "fx"
    (fx / "golden_off").mkdir(parents=True)
    stale = {"P2_models_synthetic.json", "P2_ancien_slug.json", "P13_natural_429.json", "P13_chat_headers_2.json"}
    kept = {"P5_ocr_chat_alias.json", "P1_me.json", "decisions.json", "golden_off/P2_models.json"}
    for name in stale | kept:
        (fx / name).write_text(json.dumps({"status": 200, "body": {"ancien": True}}), encoding="utf-8")
    fake = FakeAlbert()
    assert run_main(["--only", "P2,P13", "--out", "out", "--fixtures-dir", "fx"], fake, monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert "Fixtures périmées supprimées : 4" in out
    for name in stale:
        assert not (fx / name).exists(), name
    for name in kept:
        assert (fx / name).exists(), name
    assert json.loads((fx / "P2_models.json").read_text(encoding="utf-8"))["status"] == 200
    assert json.loads((fx / "P1_me.json").read_text(encoding="utf-8"))["body"] == {"ancien": True}
