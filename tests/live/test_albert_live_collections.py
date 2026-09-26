"""Tests live des collections Albert (lot 5) : envoi, relance idempotente, recherche lexicale.

Marqueur ``albert_live`` : sautés sauf si ``ALBERT_LIVE=1`` est exporté dans le
shell (``ALBERT_LIVE`` ne doit jamais figurer dans un fichier ``.env``). La clé
vient de l'environnement du shell, sinon de ``ALBERT_API_KEY`` dans le ``.env``
racine ; elle n'est jamais affichée (sorties masquées avant toute assertion).

Scénario, par la CLI ``scripts/rad_vectordb.py --db albert`` :

1. envoi de 5 chunks vers une collection privée ``ragpy-probe-<ts>`` créée
   pour l'occasion (``--albert-create-collection --albert-ack-retention``) ;
2. relance à l'identique : ``Inserted: 0`` et ``Skipped (existing): 5``
   (dédup désactivée pour ce test : l'idempotence par ``content_id`` suffit) ;
3. après l'attente d'indexation (D18, ``decisions.json``), recherche lexicale
   sur un marqueur propre au test ;
4. suppression de la collection par un finaliseur, même en cas d'échec
   (``scripts/albert_probe.py --cleanup --dry-run`` doit ensuite afficher
   ``ragpy-probe restantes: 0``).

Commande : ``ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_collections.py -m albert_live -v -p no:cacheprovider``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import rad_dedup
from scripts.rad_albert import collections as col
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig

pytestmark = pytest.mark.albert_live

RAGPY_ROOT = Path(__file__).resolve().parents[2]
VECTORDB_SCRIPT = RAGPY_ROOT / "scripts" / "rad_vectordb.py"
DECISIONS_PATH = RAGPY_ROOT / "tests" / "fixtures" / "albert" / "decisions.json"
PROBE_PREFIX = "ragpy-probe-"
MARKER = "ragpyprobe7qlot5"
CHUNK_COUNT = 5
CLI_TIMEOUT_S = 900


def _live_key():
    """Clé Albert du shell, sinon du ``.env`` racine (valeur jamais affichée)."""
    key = (os.environ.get("ALBERT_API_KEY") or "").strip()
    if not key:
        try:
            from dotenv import dotenv_values
        except ImportError:
            dotenv_values = None
        if dotenv_values is not None:
            key = (dotenv_values(RAGPY_ROOT / ".env").get("ALBERT_API_KEY") or "").strip()
    return key


def _index_wait_s():
    """Attente d'indexation des tests live (D18 : ``live_test_index_wait_s``, repli sur le défaut)."""
    try:
        decision = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))["D18"]
    except (OSError, ValueError, KeyError):
        return 60.0
    value = (decision.get("value") or {}).get("live_test_index_wait_s")
    if value is None:
        value = (decision.get("default") or {}).get("live_test_index_wait_s", 60)
    return float(value)


def _masked(text, key):
    """Texte sans la clé (remplacée par ``***``)."""
    return text.replace(key, "***") if key else text


def _probe_chunks():
    """Cinq chunks d'un même document de sonde, avec champs de dédup et vecteurs factices.

    Chaque texte porte le marqueur ``MARKER`` (recherche lexicale) et un mot
    propre (``sondeN``) ; les vecteurs sont retirés par le connecteur.
    """
    chunks = []
    for index in range(CHUNK_COUNT):
        text = (f"{MARKER} sonde{index} : la souveraineté des infrastructures de recherche passe par "
                f"des collections hébergées en France, sans copie chez un tiers (passage {index}).")
        chash, eligible, content_id = rad_dedup.compute_dedup_fields(text, index)
        chunks.append({
            "id": content_id,
            "doc_id": "424242424242",
            "chunk_index": index,
            "total_chunks": CHUNK_COUNT,
            "text": text,
            "title": "Sonde RAGpy lot 5",
            "authors": "RAGpy",
            "date": "2026-09-26",
            "itemKey": "PROBEL05",
            "filename": "ragpy-probe-l5.pdf",
            "content_hash": chash,
            "dedup_eligible": eligible,
            "embedding": [0.125] * 8,
        })
    return chunks


def _run_cli(input_path, collection_name, key):
    """Lance la CLI ``--db albert`` dans un sous-processus ; renvoie (code, stdout masqué, stderr masqué)."""
    env = dict(os.environ)
    env.update({
        "ALBERT_ENABLED": "1",
        "ALBERT_API_KEY": key,
        "DEDUP_ENABLED": "0",
        "ALBERT_PUSH_CONCURRENCY": "1",
    })
    cmd = [
        sys.executable, str(VECTORDB_SCRIPT),
        "--input", str(input_path),
        "--db", "albert",
        "--albert-collection-name", collection_name,
        "--albert-create-collection",
        "--albert-ack-retention",
    ]
    proc = subprocess.run(cmd, cwd=str(RAGPY_ROOT), env=env, capture_output=True, text=True,
                          timeout=CLI_TIMEOUT_S, stdin=subprocess.DEVNULL)
    return proc.returncode, _masked(proc.stdout, key), _masked(proc.stderr, key)


def _result_lines(stdout):
    """Lignes du bloc ``=== Result ===``."""
    lines = stdout.splitlines()
    assert "=== Result ===" in lines, stdout[-3000:]
    return lines[lines.index("=== Result ==="):]


def _value(block, label):
    """Valeur d'une ligne ``label: valeur`` du bloc Result (``None`` si absente)."""
    prefix = label + ": "
    for line in block:
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


@pytest.fixture(scope="module")
def live_key():
    """Clé Albert réelle ; saute le module si elle est absente."""
    key = _live_key()
    if not key:
        pytest.skip("ALBERT_API_KEY absente du shell et du .env : tests live impossibles")
    return key


@pytest.fixture(scope="module")
def live_client(live_key):
    """Client Albert réel (configuration lue dans l'environnement, Albert activé)."""
    env = dict(os.environ)
    env["ALBERT_ENABLED"] = "1"
    client = AlbertClient(AlbertConfig.from_env(env), live_key)
    yield client
    client.close()


@pytest.fixture
def probe_collection_name(live_client, request):
    """Nom ``ragpy-probe-<ts>`` ; le finaliseur supprime toute collection de ce nom exact."""
    name = f"{PROBE_PREFIX}{time.strftime('%Y%m%dT%H%M%S')}-l5-{os.getpid()}"

    def cleanup():
        """Supprime les collections de sonde créées par le test (nom exact)."""
        for collection in live_client.list_collections(name=name):
            live_client.delete_collection(collection["id"])
        assert live_client.list_collections(name=name) == []

    request.addfinalizer(cleanup)
    return name


def test_live_push_rerun_idempotent_then_lexical_search(tmp_path, live_key, live_client, probe_collection_name):
    chunks = _probe_chunks()
    input_path = tmp_path / "output_chunks_with_embeddings_sparse.json"
    input_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    # 1) Premier envoi : 5 chunks, collection privée créée, manifeste écrit.
    code, out, err = _run_cli(input_path, probe_collection_name, live_key)
    assert code == 0, (out[-3000:], err[-3000:])
    block = _result_lines(out)
    assert _value(block, "Status") == "success", block
    assert _value(block, "Inserted") == str(CHUNK_COUNT), block
    assert _value(block, "Skipped (existing)") == "0", block
    manifest_path = _value(block, "Albert manifest")
    assert manifest_path == col.manifest_path_for(str(tmp_path)), block
    rows = col.read_manifest(manifest_path)
    assert sum(row["count"] for row in rows) == CHUNK_COUNT
    content_ids = {cid for row in rows for cid in row["content_ids"]}
    assert content_ids == {col.content_id_for(chunk) for chunk in chunks}
    # D16 : les ids de chunks renvoyés par l'API sont recopiés dans le manifeste.
    assert all(isinstance(i, int) for row in rows for i in (row["chunk_ids"] or []))
    assert sum(len(row["chunk_ids"] or []) for row in rows) == CHUNK_COUNT
    assert all(live_key not in text for text in (out, err))

    (collection,) = live_client.list_collections(name=probe_collection_name)
    assert col.is_private(collection)
    assert {row["collection_id"] for row in rows} == {int(collection["id"])}

    # 2) Relance à l'identique : rien n'est renvoyé.
    code, out, err = _run_cli(input_path, probe_collection_name, live_key)
    assert code == 0, (out[-3000:], err[-3000:])
    block = _result_lines(out)
    assert _value(block, "Status") == "success", block
    assert _value(block, "Inserted") == "0", block
    assert _value(block, "Skipped (existing)") == str(CHUNK_COUNT), block
    assert len(live_client.list_collections(name=probe_collection_name)) == 1

    # 3) Recherche lexicale après l'attente d'indexation (D18).
    wait = _index_wait_s()
    results = []
    for _attempt in range(3):
        time.sleep(wait)
        results = live_client.search(f"{MARKER} sonde3", collection_ids=[int(collection["id"])],
                                     method="lexical", limit=10)
        if results:
            break
    assert results, "aucun résultat de recherche lexicale après l'attente d'indexation"
    found = [r.get("chunk") or {} for r in results]
    assert any(MARKER in (chunk.get("content") or "") for chunk in found)
    assert any((chunk.get("metadata") or {}).get("content_id") in content_ids for chunk in found)
