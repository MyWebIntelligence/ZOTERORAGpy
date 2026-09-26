"""Régressions de la dette corrigée au lot 2 du sprint Albert.

* ``rad_chunk`` ne demande plus jamais de clé par ``input()`` ; sans clé, le
  client vaut ``None`` et la CLI s'arrête (exit 1) seulement si la phase l'exige ;
* les deux lanceurs de sous-processus des routes ferment le stdin (``DEVNULL``) ;
* aucun littéral de clé réelle dans les fichiers suivis (gabarits en X ignorés) ;
* la CLI ``rad_vectordb`` n'affiche jamais la clé, même tronquée ;
* ``_get_llm_clients`` n'a plus de repli ``.env`` / environnement pour les clés,
  et tous ses appelants passent les clés explicitement ;
* le modèle par défaut web est relu du ``.env`` à chaque appel, sans muter
  ``os.environ``, et le ``.env`` l'emporte sur l'environnement du processus
  quelles que soient les clés passées ;
* le stdout de ``import rad_chunk`` avec clé reste celui du golden G10.

Aucun réseau ; les valeurs de clé sont factices et ne sont jamais affichées :
les assertions portent sur des noms, des booléens et des identités d'objets.
"""

import ast
import asyncio
import json
import os
import re
import runpy
import subprocess
import sys
import types
from types import SimpleNamespace

import dotenv
import dotenv.main
import pytest

RAGPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(RAGPY_ROOT, "scripts")
for _p in (RAGPY_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDEN_DIR = os.path.join(RAGPY_ROOT, "tests", "fixtures", "albert", "golden_off", "scripts")
RAD_CHUNK = os.path.join(SCRIPTS_DIR, "rad_chunk.py")
RAD_VECTORDB = os.path.join(SCRIPTS_DIR, "rad_vectordb.py")

GUARD_MESSAGE = "Erreur critique: Client OpenAI non initialisé (OPENAI_API_KEY manquante?). Arrêt."
NO_KEY_LINE = "OPENAI_API_KEY not found in environment variables or .env file."
NO_OPENROUTER_LINE = "OpenRouter API key not found. Will use OpenAI for all LLM calls."

FAKE_OPENAI_KEY = "fake-openai-0001"
FAKE_OPENROUTER_KEY = "fake-openrouter-0001"


# ----------------------------------------------------------------------
# Aides
# ----------------------------------------------------------------------
def _child_env(**overrides):
    """Env d'un sous-processus : copie sans famille Albert/Recode/Dedup ni liste de refus, plus ``overrides``.

    Une valeur ``None`` retire la variable ; une chaîne vide signifie « pas de clé »
    (``load_dotenv`` ne remplace jamais une variable déjà présente).
    """
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("ALBERT_", "RECODE_", "DEDUP_"))}
    env.pop("RAGPY_DOTENV_DENY", None)
    env.update({"ALBERT_ENABLED": "0", "OCR_ENABLE_ALBERT": "0", "EMBEDDING_PROVIDER": ""})
    for name, value in overrides.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


def _run_python(code, cwd, env, timeout=300):
    """Lance ``python -c code`` (stdin fermé) et renvoie le ``CompletedProcess``."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
    )


def _import_rad_chunk_code():
    """Code ``python -c`` qui importe ``rad_chunk`` depuis ``scripts/`` (motif du golden G10)."""
    return f"import sys; sys.path.insert(0, {SCRIPTS_DIR!r}); import rad_chunk"


def _parse(path):
    """Arbre AST du fichier ``path``."""
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _call_name(node):
    """Nom appelé par un nœud ``ast.Call`` (``f(...)`` ou ``obj.f(...)``), sinon ``None``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_node(tree, name):
    """Définition de fonction (sync ou async) de premier niveau ``name`` dans ``tree``."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"fonction {name} introuvable")


@pytest.fixture
def rc():
    """Module ``rad_chunk`` importé comme par les goldens (``scripts/`` sur le chemin).

    S'il n'est pas encore importé, l'import se fait sans découverte du ``.env``
    de l'hôte (``find_dotenv`` renvoie une chaîne vide le temps de l'import).
    """
    if "rad_chunk" in sys.modules:
        return sys.modules["rad_chunk"]
    original = dotenv.main.find_dotenv
    dotenv.main.find_dotenv = lambda *args, **kwargs: ""
    try:
        import rad_chunk
    finally:
        dotenv.main.find_dotenv = original
    return rad_chunk


# ----------------------------------------------------------------------
# rad_chunk : plus de saisie interactive, garde de phase
# ----------------------------------------------------------------------
def test_import_without_openai_key_never_calls_input(tmp_path):
    tree = _parse(RAD_CHUNK)
    input_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n) == "input"]
    assert input_calls == []
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "update_env_file" not in names | defs

    env = _child_env(OPENAI_API_KEY="", OPENROUTER_API_KEY="")
    proc = _run_python(_import_rad_chunk_code() + "; print('client', rad_chunk.client)", tmp_path, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.splitlines() == [NO_KEY_LINE, NO_OPENROUTER_LINE, "client None"]
    assert "Please enter" not in proc.stdout + proc.stderr


@pytest.mark.parametrize("phase,model,openai_ok,openrouter_ok,expected_missing", [
    ("sparse", "gpt-4o-mini", False, False, False),
    ("dense", "gpt-4o-mini", False, True, True),
    ("dense", "gpt-4o-mini", True, False, False),
    ("initial", "gpt-4o-mini", False, True, True),
    ("initial", "gpt-4o-mini", True, False, False),
    ("initial", "google/gemini-2.5-flash", False, True, False),
    ("initial", "google/gemini-2.5-flash", True, False, False),
    ("initial", "google/gemini-2.5-flash", False, False, True),
    ("all", "google/gemini-2.5-flash", False, True, True),
    ("all", "gpt-4o-mini", True, False, False),
])
def test_phase_guard_matrix(rc, monkeypatch, phase, model, openai_ok, openrouter_ok, expected_missing):
    monkeypatch.setattr(rc, "client", object() if openai_ok else None)
    monkeypatch.setattr(rc, "openrouter_client", object() if openrouter_ok else None)
    assert rc.missing_llm_client_for_phase(phase, model) is expected_missing


def _run_cli(tmp_path, argv, env):
    """Exécute ``rad_chunk.py`` en ``__main__`` (``scripts/`` en tête de ``sys.path``, comme
    ``python scripts/rad_chunk.py``), sans découverte du ``.env`` de l'hôte."""
    code = (
        "import sys, runpy, dotenv.main\n"
        "dotenv.main.find_dotenv = lambda *args, **kwargs: ''\n"
        f"sys.path.insert(0, {SCRIPTS_DIR!r})\n"
        f"sys.argv = {['rad_chunk.py'] + list(argv)!r}\n"
        f"runpy.run_path({RAD_CHUNK!r}, run_name='__main__')\n"
    )
    return _run_python(code, tmp_path, env)


def test_cli_phase_guard_without_openai_key(tmp_path):
    env = _child_env(OPENAI_API_KEY="", OPENROUTER_API_KEY="")

    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    chunks = tmp_path / "doc_chunks.json"
    chunks.write_text(json.dumps([{"id": "1_1", "text": "Un texte court."}]), encoding="utf-8")
    proc = _run_cli(tmp_path, ["--input", str(chunks), "--output", str(dense_dir), "--phase", "dense"], env)
    assert proc.returncode == 1
    assert GUARD_MESSAGE in proc.stdout.splitlines()

    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    dense_file = tmp_path / "doc_chunks_with_embeddings.json"
    dense_file.write_text(json.dumps([
        {"id": "1_1", "text": "Le langage ordinaire structure l'expérience sociale.", "embedding": [0.1, 0.2]},
    ], ensure_ascii=False), encoding="utf-8")
    proc = _run_cli(tmp_path, ["--input", str(dense_file), "--output", str(sparse_dir), "--phase", "sparse"], env)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert GUARD_MESSAGE not in proc.stdout
    with open(sparse_dir / "doc_chunks_with_embeddings_sparse.json", encoding="utf-8") as fh:
        assert "sparse_embedding" in json.load(fh)[0]


# ----------------------------------------------------------------------
# G10 : stdout de l'import avec clé inchangé (avec ou sans liste de refus)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("golden_name,with_openrouter", [
    ("g10_rad_chunk_import_stdout.json", False),
    ("g10_rad_chunk_import_stdout_openrouter.json", True),
])
@pytest.mark.parametrize("deny", [False, True], ids=["admin_env", "nonadmin_deny"])
def test_import_stdout_with_key_equals_golden(golden_name, with_openrouter, deny, tmp_path):
    with open(os.path.join(GOLDEN_DIR, golden_name), encoding="utf-8") as fh:
        golden = json.load(fh)
    overrides = {"OPENAI_API_KEY": FAKE_OPENAI_KEY,
                 "OPENROUTER_API_KEY": FAKE_OPENROUTER_KEY if with_openrouter else ""}
    if deny:
        # Clés personnelles réinjectées : elles ne figurent pas dans la liste de refus.
        denied = ["ALBERT_API_KEY", "MISTRAL_API_KEY", "PINECONE_API_KEY", "QDRANT_API_KEY",
                  "WEAVIATE_API_KEY", "ZOTERO_API_KEY"]
        if not with_openrouter:
            # Clé absente de l'env mais présente dans le .env trouvé : elle est
            # refusée, donc jamais rechargée (stdout identique au golden).
            denied.append("OPENROUTER_API_KEY")
            overrides["OPENROUTER_API_KEY"] = None
            (tmp_path / ".env").write_text("OPENROUTER_API_KEY=fake-openrouter-dotenv-0011\n", encoding="utf-8")
        overrides["RAGPY_DOTENV_DENY"] = ",".join(sorted(denied))
    proc = _run_python(_import_rad_chunk_code(), tmp_path, _child_env(**overrides))
    assert {"returncode": proc.returncode, "stdout_lines": proc.stdout.splitlines()} == golden


# ----------------------------------------------------------------------
# Lanceurs de sous-processus des routes : stdin fermé
# ----------------------------------------------------------------------
class _EmptyStream:
    """Flux asynchrone vide (``readline`` renvoie ``b''``)."""

    async def readline(self):
        """Fin de flux immédiate."""
        return b""


class _FakeProcess:
    """Processus asynchrone factice terminé avec le code 0."""

    pid = 424242
    returncode = 0

    def __init__(self):
        """Crée les flux stdout/stderr vides."""
        self.stdout = _EmptyStream()
        self.stderr = _EmptyStream()

    async def communicate(self):
        """Sorties vides."""
        return b"", b""

    async def wait(self):
        """Code de sortie 0."""
        return 0

    def kill(self):
        """Sans effet."""


def test_route_subprocesses_use_devnull_stdin(monkeypatch):
    from app.routes import processing
    from app.utils import sse_helpers

    launches = []

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        """Enregistre les kwargs du lancement et renvoie un processus factice."""
        launches.append(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    for module in (processing, sse_helpers):
        monkeypatch.setattr(module.process_manager, "register", lambda *args, **kwargs: None)
        monkeypatch.setattr(module.process_manager, "unregister", lambda *args, **kwargs: None)

    async def scenario():
        """Lance une commande par chacun des deux lanceurs."""
        await processing.run_tracked_subprocess(["true"], "debt-session", timeout=5)
        async for _event in sse_helpers.run_subprocess_with_sse(
                ["true"], lambda line: None, session_folder="debt-session", timeout=5):
            pass

    asyncio.run(scenario())
    assert len(launches) == 2
    for kwargs in launches:
        assert kwargs.get("stdin") is asyncio.subprocess.DEVNULL
        assert kwargs.get("stdout") is asyncio.subprocess.PIPE
        assert kwargs.get("stderr") is asyncio.subprocess.PIPE

    # Même contrat vu dans le source : chaque create_subprocess_exec des deux
    # modules passe stdin=asyncio.subprocess.DEVNULL.
    for path in (processing.__file__, sse_helpers.__file__):
        calls = [n for n in ast.walk(_parse(path))
                 if isinstance(n, ast.Call) and _call_name(n) == "create_subprocess_exec"]
        assert calls
        for call in calls:
            stdin = {k.arg: k.value for k in call.keywords}.get("stdin")
            assert stdin is not None and ast.unparse(stdin) == "asyncio.subprocess.DEVNULL"


# ----------------------------------------------------------------------
# Aucun littéral de clé dans le dépôt
# ----------------------------------------------------------------------
# Motifs construits par concaténation : ce fichier ne se détecte pas lui-même.
_TOKEN_TAIL = r"[A-Za-z0-9_\-]{20,}"
_NO_WORD_BEFORE = r"(?<![A-Za-z0-9_\-])"
SECRET_PATTERNS = (
    ("pinecone", re.compile(_NO_WORD_BEFORE + "pc" + "sk_" + _TOKEN_TAIL)),
    ("openai", re.compile(_NO_WORD_BEFORE + "s" + "k-" + _TOKEN_TAIL)),
    ("bearer", re.compile("Bear" + "er" + r"\s+([A-Za-z0-9._\-]{20,})")),
)
_TEMPLATE_SEGMENT = re.compile(r"[Xx]{4,}")
MAX_SCANNED_BYTES = 5 * 1024 * 1024


def _is_placeholder(token):
    """Vrai pour un gabarit : dernier segment en X (``pc…_XXXX``, ``s…-proj-XXXX``) ou valeur ``fake-``."""
    if token.startswith("fake-"):
        return True
    last = re.split(r"[-_]", token)[-1]
    return bool(_TEMPLATE_SEGMENT.fullmatch(last))


def _secret_hits(text):
    """Motifs de clé non gabarits trouvés dans ``text`` : liste de (type, ligne)."""
    hits = []
    for kind, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(match.lastindex or 0)
            if not _is_placeholder(token):
                hits.append((kind, text.count("\n", 0, match.start()) + 1))
    return hits


def _is_env_file(path):
    """Vrai pour un fichier ``.env`` réel (jamais lu) ; les ``*.example`` restent examinés."""
    base = os.path.basename(path)
    return base == ".env" or (base.startswith(".env.") and not base.endswith((".example", ".sample", ".template")))


def _repo_files():
    """Fichiers suivis ou nouveaux non ignorés (``git ls-files``), sans les ``.env``."""
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=RAGPY_ROOT, capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    names = [name for name in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if name]
    return [name for name in names if not _is_env_file(name)]


def test_secret_patterns_detect_real_shapes_and_ignore_templates():
    real = "pc" + "sk_" + "A1b2C3d4" * 4
    assert _secret_hits(f"key = '{real}'\n") == [("pinecone", 1)]
    assert _secret_hits("x\n" + "s" + "k-" + "proj-" + "Zy9Qa1Wb2Vc3Xd4Ue5Tf6" + "\n") == [("openai", 2)]
    assert _secret_hits("Bear" + "er " + "Q1w2E3r4T5y6U7i8O9p0A1s2") == [("bearer", 1)]
    for template in ("pc" + "sk_" + "X" * 40, "s" + "k-" + "proj-" + "X" * 40, "s" + "k-" + "x" * 21,
                     "s" + "k-or-v1-" + "X" * 30, "Bear" + "er fake-other-token-123456"):
        assert _secret_hits(template) == []
    assert _secret_hits("active-task" + "-notification-banner-element") == []


def test_no_secret_literals_in_repo():
    files = _repo_files()
    assert "scripts/rad_vectordb.py" in files
    offenders = []
    for name in files:
        path = os.path.join(RAGPY_ROOT, name)
        if not os.path.isfile(path) or os.path.getsize(path) > MAX_SCANNED_BYTES:
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        if b"\0" in data[:8192]:
            continue
        for kind, line in _secret_hits(data.decode("utf-8", "ignore")):
            offenders.append(f"{name}:{line} ({kind})")
    assert offenders == []


# ----------------------------------------------------------------------
# rad_vectordb : la clé n'est jamais affichée
# ----------------------------------------------------------------------
FAKE_VECTORDB_KEY = "fake-vdb-7Qz9Lm3Rt8-0001"


def _vectordb_chunks():
    """Deux chunks avec vecteurs dense et sparse."""
    return [
        {"id": "424242424242_1", "doc_id": "424242424242", "chunk_index": 1, "total_chunks": 2,
         "text": "alpha", "title": "Doc", "embedding": [0.1] * 8,
         "sparse_embedding": {"indices": [3], "values": [0.5]}},
        {"id": "424242424242_2", "doc_id": "424242424242", "chunk_index": 2, "total_chunks": 2,
         "text": "beta", "title": "Doc", "embedding": [0.2] * 8,
         "sparse_embedding": {"indices": [1], "values": [1.0]}},
    ]


def _fake_pinecone_module():
    """Module ``pinecone`` factice : un index ``debt-index`` (dotproduct, 8 dimensions)."""
    index = SimpleNamespace(upsert=lambda **kwargs: {"upserted_count": len(kwargs.get("vectors", []))})

    class _FakePinecone:
        """Client Pinecone factice."""

        def __init__(self, api_key=None, **kwargs):
            """Accepte la clé sans la conserver."""

        def list_indexes(self):
            """Liste l'unique index."""
            return SimpleNamespace(indexes=[SimpleNamespace(name="debt-index")])

        def Index(self, name, *args, **kwargs):
            """Renvoie l'index factice."""
            return index

        def describe_index(self, name, *args, **kwargs):
            """Métrique et dimension de l'index."""
            return SimpleNamespace(metric="dotproduct", dimension=8)

    module = types.ModuleType("pinecone")
    module.Pinecone = _FakePinecone
    return {"pinecone": module}


def _fake_qdrant_modules():
    """Paquet ``qdrant_client`` factice : collection existante, upserts terminés."""
    client = SimpleNamespace(
        get_collections=lambda *a, **k: SimpleNamespace(collections=[]),
        get_collection=lambda *a, **k: SimpleNamespace(status="green"),
        create_collection=lambda *a, **k: None,
        upsert=lambda *a, **k: SimpleNamespace(status="completed"),
        close=lambda: None,
    )
    root = types.ModuleType("qdrant_client")
    models = types.ModuleType("qdrant_client.models")
    models.PointStruct = lambda id=None, vector=None, payload=None, **kwargs: SimpleNamespace(
        id=id, vector=vector, payload=payload)
    models.UpdateStatus = SimpleNamespace(COMPLETED="completed")
    models.VectorParams = lambda size=None, distance=None, **kwargs: SimpleNamespace(size=size, distance=distance)
    models.Distance = SimpleNamespace(COSINE="Cosine")
    models.Filter = lambda *args, **kwargs: SimpleNamespace()
    models.FieldCondition = lambda *args, **kwargs: SimpleNamespace()
    models.MatchValue = lambda *args, **kwargs: SimpleNamespace()
    root.models = models
    root.QdrantClient = lambda *args, **kwargs: client
    return {"qdrant_client": root, "qdrant_client.models": models}


@pytest.mark.parametrize("db,db_args,env_name", [
    ("pinecone", ["--db", "pinecone", "--index", "debt-index"], "PINECONE_API_KEY"),
    ("qdrant", ["--db", "qdrant", "--collection", "debt-collection"], "QDRANT_API_KEY"),
])
def test_vectordb_cli_stdout_has_no_key(db, db_args, env_name, tmp_path, monkeypatch, capsys):
    import time

    input_path = tmp_path / "debt_sparse.json"
    input_path.write_text(json.dumps(_vectordb_chunks()), encoding="utf-8")
    for name in list(os.environ):
        if name.startswith("DEDUP_"):
            monkeypatch.delenv(name)  # dédup OFF : comportement par défaut du connecteur
    monkeypatch.setenv(env_name, FAKE_VECTORDB_KEY)
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.test")
    monkeypatch.setattr(time, "sleep", lambda *_args: None)
    fakes = _fake_pinecone_module() if db == "pinecone" else _fake_qdrant_modules()
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["rad_vectordb.py", "--input", str(input_path)] + db_args)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(RAD_VECTORDB, run_name="__main__")
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0, output[-2000:]
    assert "=== Result ===" in output
    assert FAKE_VECTORDB_KEY not in output
    assert FAKE_VECTORDB_KEY[:10] not in output
    assert "API Key" not in output


def test_vectordb_source_never_prints_a_key_variable():
    offenders = []
    for node in ast.walk(_parse(RAD_VECTORDB)):
        if isinstance(node, ast.Call) and _call_name(node) == "print":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and "api_key" in sub.id.lower():
                    offenders.append(node.lineno)
    assert offenders == []


# ----------------------------------------------------------------------
# llm_note_generator : clés explicites seulement, modèle par défaut relu
# ----------------------------------------------------------------------
@pytest.fixture
def lng():
    """Module ``app.utils.llm_note_generator``."""
    from app.utils import llm_note_generator
    return llm_note_generator


@pytest.fixture
def isolated_dotenv(tmp_path, monkeypatch, lng):
    """``.env`` factice renvoyé par tout ``find_dotenv`` (module et paquet dotenv) ; renvoie son chemin."""
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    finder = lambda *args, **kwargs: str(path)  # noqa: E731
    monkeypatch.setattr(lng, "find_dotenv", finder)
    monkeypatch.setattr(dotenv.main, "find_dotenv", finder)
    return path


def test_get_llm_clients_has_no_env_fallback(lng, isolated_dotenv, monkeypatch):
    isolated_dotenv.write_text(
        "OPENAI_API_KEY=fake-openai-dotenv-0009\n"
        "OPENROUTER_API_KEY=fake-openrouter-dotenv-0009\n"
        "RAGPY_DEBT_SENTINEL=1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-env-0009")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-env-0009")
    monkeypatch.setenv("RAGPY_DEBT_SENTINEL", "placeholder")
    monkeypatch.delenv("RAGPY_DEBT_SENTINEL")
    built = []

    def fake_openai(**kwargs):
        """Enregistre la construction d'un client (kwargs) et renvoie un objet neutre."""
        built.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(lng, "OpenAI", fake_openai)
    before = dict(os.environ)

    for keys in ({}, {"openai_api_key": None, "openrouter_api_key": None},
                 {"openai_api_key": "", "openrouter_api_key": ""}):
        openai_client, openrouter_client, _default = lng._get_llm_clients(**keys)
        assert openai_client is None and openrouter_client is None
    assert built == []
    assert dict(os.environ) == before  # ni rechargement du .env ni mutation

    openai_key, openrouter_key = "fake-openai-explicit-0010", "fake-openrouter-explicit-0010"
    openai_client, openrouter_client, _default = lng._get_llm_clients(
        openai_api_key=openai_key, openrouter_api_key=openrouter_key)
    assert openai_client is not None and openrouter_client is not None
    assert [sorted(kwargs) for kwargs in built] == [["api_key"], ["api_key", "base_url"]]
    assert built[0]["api_key"] is openai_key and built[1]["api_key"] is openrouter_key

    built.clear()
    openai_client, openrouter_client, _default = lng._get_llm_clients(openai_api_key=openai_key)
    assert openai_client is not None and openrouter_client is None
    assert len(built) == 1

    # Source : aucun chargement dotenv ni lecture d'env de clé dans ces fonctions.
    tree = _parse(lng.__file__)
    for func_name in ("_get_llm_clients", "resolve_default_llm_model"):
        func = _function_node(tree, func_name)
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                assert _call_name(node) != "load_dotenv", func_name
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.endswith("_API_KEY"), (func_name, node.value)
    with open(lng.__file__, encoding="utf-8") as fh:
        assert "override=True" not in fh.read()


def _enclosing_functions(tree):
    """Associe chaque nœud ``Call`` au nom de la fonction de premier niveau qui le contient."""
    owner = {}
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(top):
                if isinstance(node, ast.Call):
                    owner[node] = top.name
    return owner


# Points d'entrée LLM dont chaque appel doit transmettre les deux clés explicitement.
LLM_KEY_ENTRY_POINTS = frozenset({
    "_get_llm_clients", "_generate_with_llm", "build_note_html", "build_abstract_text",
    "build_note_html_async", "build_abstract_text_async", "_call_llm_api",
    "filter_citation_with_llm", "pre_filter_citation", "process_citations_parallel",
    "build_book_note_async",
})


def _app_python_files():
    """Fichiers Python de ``app/`` hors tests."""
    for dirpath, _dirnames, filenames in os.walk(os.path.join(RAGPY_ROOT, "app")):
        if "__pycache__" in dirpath:
            continue
        for filename in sorted(filenames):
            if filename.endswith(".py") and not filename.startswith("test_"):
                yield os.path.join(dirpath, filename)


def test_get_llm_clients_callers_pass_explicit_keys():
    sites = []
    bad = []
    for path in _app_python_files():
        tree = _parse(path)
        owners = _enclosing_functions(tree)
        rel = os.path.relpath(path, RAGPY_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in LLM_KEY_ENTRY_POINTS:
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            if _call_name(node) == "_get_llm_clients":
                sites.append((rel, owners.get(node)))
            for key in ("openai_api_key", "openrouter_api_key"):
                value = keywords.get(key)
                if value is None or (isinstance(value, ast.Constant) and value.value is None):
                    bad.append(f"{rel}:{node.lineno} {_call_name(node)} sans {key}")
                elif isinstance(value, ast.Call) and _call_name(value) in ("getenv", "get"):
                    bad.append(f"{rel}:{node.lineno} {_call_name(node)} {key} lu dans l'env")
    assert sorted(sites) == [
        ("app/utils/citation_filter.py", "_call_llm_api"),
        ("app/utils/llm_note_generator.py", "_generate_with_llm"),
        ("app/utils/llm_note_generator.py", "build_abstract_text"),
        ("app/utils/llm_note_generator.py", "build_note_html"),
    ]
    assert bad == []


def test_default_model_read_fresh_from_dotenv(lng, isolated_dotenv, monkeypatch):
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", "fake-env-model")
    before = dict(os.environ)

    isolated_dotenv.write_text("OPENROUTER_DEFAULT_MODEL=fake-dotenv-model-a\n", encoding="utf-8")
    assert lng.resolve_default_llm_model() == "fake-dotenv-model-a"
    isolated_dotenv.write_text("OPENROUTER_DEFAULT_MODEL=fake-dotenv-model-b\n", encoding="utf-8")
    assert lng.resolve_default_llm_model() == "fake-dotenv-model-b"
    assert lng._get_llm_clients()[2] == "fake-dotenv-model-b"
    assert dict(os.environ) == before  # lu sans muter l'environnement

    assert lng.resolve_default_llm_model("fake-explicit-model") == "fake-explicit-model"
    assert lng._get_llm_clients(openrouter_model="fake-explicit-model")[2] == "fake-explicit-model"

    isolated_dotenv.write_text("OPENROUTER_DEFAULT_MODEL=\nOTHER=1\n", encoding="utf-8")
    assert lng.resolve_default_llm_model() == "fake-env-model"
    isolated_dotenv.unlink()
    assert lng.resolve_default_llm_model() == "fake-env-model"
    monkeypatch.setattr(lng, "find_dotenv", lambda *args, **kwargs: "")
    assert lng.resolve_default_llm_model() == "fake-env-model"
    monkeypatch.delenv("OPENROUTER_DEFAULT_MODEL")
    assert lng.resolve_default_llm_model() == "gpt-4o-mini"
    assert lng._get_llm_clients()[2] == "gpt-4o-mini"


def test_default_model_dotenv_wins_over_process_env_whatever_the_keys(lng, isolated_dotenv, monkeypatch):
    # Le .env et l'environnement du processus divergent (export shell, ou .env
    # modifié par /save_credentials sans toucher os.environ) : le .env gagne,
    # que l'appelant passe une clé, les deux ou aucune (ordre fixé par le sprint).
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", "fake-env-model")
    isolated_dotenv.write_text("OPENROUTER_DEFAULT_MODEL=fake-dotenv-model\n", encoding="utf-8")
    monkeypatch.setattr(lng, "OpenAI", lambda **kwargs: SimpleNamespace())
    before = dict(os.environ)

    for keys in ({}, {"openai_api_key": "fake-openai-explicit-0011"},
                 {"openrouter_api_key": "fake-openrouter-explicit-0011"},
                 {"openai_api_key": "fake-openai-explicit-0011",
                  "openrouter_api_key": "fake-openrouter-explicit-0011"}):
        assert lng._get_llm_clients(**keys)[2] == "fake-dotenv-model", sorted(keys)
    assert dict(os.environ) == before

    # Édition du .env en cours de route : prise en compte sans redémarrage.
    isolated_dotenv.write_text("OPENROUTER_DEFAULT_MODEL=fake-dotenv-model-edited\n", encoding="utf-8")
    assert lng._get_llm_clients(
        openai_api_key="fake-openai-explicit-0011",
        openrouter_api_key="fake-openrouter-explicit-0011",
    )[2] == "fake-dotenv-model-edited"
    assert os.environ["OPENROUTER_DEFAULT_MODEL"] == "fake-env-model"

    # Sans valeur dans le .env, l'environnement du processus reprend la main.
    isolated_dotenv.write_text("OTHER=1\n", encoding="utf-8")
    assert lng._get_llm_clients(
        openai_api_key="fake-openai-explicit-0011",
        openrouter_api_key="fake-openrouter-explicit-0011",
    )[2] == "fake-env-model"
