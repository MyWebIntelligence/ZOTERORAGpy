"""Tests live des embeddings Albert (lot 6) : bge-m3 en 1024 dimensions, phase dense.

Marqueur ``albert_live`` : sautés sauf si ``ALBERT_LIVE=1`` est exporté dans le
shell (``ALBERT_LIVE`` ne doit jamais figurer dans un fichier ``.env``). Appels
d'inférence seulement (aucune collection). La clé vient de l'environnement du
shell, sinon de ``ALBERT_API_KEY`` dans le ``.env`` racine ; elle n'est jamais
affichée ni comparée à une valeur.

* 3 textes → 3 vecteurs de 1024 dimensions, déjà normalisés en L2 par le
  service (décision D12 : ``norm_range`` = [1.0, 1.0]) : le client est construit
  avec ``ALBERT_EMBED_L2_NORMALIZE=0`` pour mesurer la norme brute ; l'id
  épinglé ``bge-m3`` part sur le fil et revient dans ``response.model`` ;
* phase dense de ``rad_chunk`` sur 5 chunks avec ``EMBEDDING_PROVIDER=albert`` :
  5 vecteurs bge-m3, champs d'espace, aucun vecteur nul, aucun client OpenAI,
  journal d'usage écrit.

Commande : ``ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_embed.py -m albert_live -v -p no:cacheprovider``
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

RAGPY_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(RAGPY_ROOT), str(RAGPY_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.rad_albert.client import AlbertClient  # noqa: E402
from scripts.rad_albert.config import AlbertConfig  # noqa: E402
from scripts.rad_albert.usage import USAGE_FILENAME  # noqa: E402

pytestmark = pytest.mark.albert_live

EMBED_MODEL = "bge-m3"
TEXTS = [
    "Le langage ordinaire structure l'expérience sociale.",
    "La loi de finances est votée par le Parlement.",
    "Les enquêtes de terrain décrivent des épreuves de justification.",
]
with open(RAGPY_ROOT / "tests" / "fixtures" / "albert" / "decisions.json", encoding="utf-8") as _fh:
    D12 = json.load(_fh)["D12"]["value"]
NORM_TOLERANCE = 1e-3


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


@pytest.fixture(scope="module")
def live_key():
    """Clé réelle ; saute le module si elle est absente."""
    key = _live_key()
    if not key:
        pytest.skip("ALBERT_API_KEY absente du shell et du .env : tests live impossibles")
    return key


def _norm(vector):
    """Norme L2 d'un vecteur."""
    return math.sqrt(sum(float(v) * float(v) for v in vector))


def test_live_embed_three_texts_1024_norm_d12(live_key):
    env = dict(os.environ)
    env["ALBERT_ENABLED"] = "1"
    env["ALBERT_EMBED_L2_NORMALIZE"] = "0"  # norme brute du service (D12)
    client = AlbertClient(AlbertConfig.from_env(env), live_key)
    try:
        vectors = client.embed(TEXTS, model=EMBED_MODEL)
    finally:
        client.close()
    assert len(vectors) == len(TEXTS)
    low, high = D12["norm_range"]
    for vector in vectors:
        assert len(vector) == D12["embedding_dim"] == 1024
        assert low - NORM_TOLERANCE <= _norm(vector) <= high + NORM_TOLERANCE
        assert any(v != 0.0 for v in vector)
    records = [r for r in client.ledger if r.get("endpoint") == "/v1/embeddings"]
    assert records, "appel d'embeddings absent du ledger"
    assert {r.get("model") for r in records} == {EMBED_MODEL}
    # D12 : response.model renvoie le nom envoyé ; l'id épinglé part sur le fil.
    assert {r.get("response_model") for r in records} == {EMBED_MODEL}


@pytest.fixture
def rad_chunk_albert(monkeypatch, live_key):
    """``rad_chunk`` avec Albert ON, ``EMBEDDING_PROVIDER=albert`` et aucun client OpenAI."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", live_key)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "albert")
    import rad_chunk

    for name, value in (("ALBERT_ENABLED", True), ("ALBERT_API_KEY", live_key)):
        if hasattr(rad_chunk, name):
            monkeypatch.setattr(rad_chunk, name, value)
    monkeypatch.setattr(rad_chunk, "client", None)
    monkeypatch.setattr(rad_chunk, "openrouter_client", None)
    rad_chunk.reset_albert_state()
    yield rad_chunk
    rad_chunk.reset_albert_state()


def test_live_dense_phase_five_chunks(rad_chunk_albert, tmp_path):
    rc = rad_chunk_albert
    chunks = []
    for i in range(1, 6):
        chunks.append({
            "id": f"424242424242_{i}",
            "doc_id": "424242424242",
            "chunk_index": i,
            "total_chunks": 5,
            "text": f"Passage {i} : la sociologie pragmatique décrit les épreuves de justification ({i}).",
            "title": "Test live embeddings",
            "texteocr_provider": "mistral",
        })
    input_path = tmp_path / "output_chunks.json"
    input_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    output_path = tmp_path / "output_chunks_with_embeddings.json"
    returned = rc.generate_and_save_embeddings(str(input_path), str(output_path))
    assert returned == str(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert [c["id"] for c in written] == [c["id"] for c in chunks]
    for chunk in written:
        vector = chunk["embedding"]
        assert isinstance(vector, list) and len(vector) == 1024
        assert any(v != 0.0 for v in vector)
        assert abs(_norm(vector) - 1.0) <= NORM_TOLERANCE
        assert chunk["embedding_provider"] == "albert"
        assert chunk["embedding_model"] == EMBED_MODEL
        assert chunk["embedding_dim"] == 1024
    usage_path = tmp_path / USAGE_FILENAME
    assert usage_path.exists(), "journal d'usage attendu : Albert a été appelé"
    records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records and {r.get("model") for r in records} == {EMBED_MODEL}
