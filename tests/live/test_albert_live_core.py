"""Tests live du socle Albert : compte, catalogue, embeddings, santé, clé invalide.

Marqueur ``albert_live`` : sautés sauf si ``ALBERT_LIVE=1`` est exporté dans le
shell (``ALBERT_LIVE`` ne doit jamais figurer dans un fichier ``.env``). Ils
n'utilisent que des appels de lecture et d'inférence (aucune collection créée).
La clé vient de l'environnement du shell, sinon de ``ALBERT_API_KEY`` dans le
``.env`` racine ; elle n'est jamais affichée.

Commande : ``ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_core.py -m albert_live -v -p no:cacheprovider``
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import httpx
import pytest

from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig, root_url
from scripts.rad_albert.errors import AlbertAuthError

pytestmark = pytest.mark.albert_live

RAGPY_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MODELS = {
    "ministral-3-8b-instruct-2512": {"text-generation", "image-text-to-text"},
    "gpt-oss-120b": {"text-generation"},
    "lightonocr-2-1b": {"image-text-to-text", "image-to-text"},
    "bge-m3": {"text-embeddings-inference"},
}
INVALID_KEY = "fake-invalid-albert-key-0000"


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
def live_cfg():
    """Configuration Albert ON lue depuis l'environnement (URL de base comprise)."""
    env = dict(os.environ)
    env["ALBERT_ENABLED"] = "1"
    return AlbertConfig.from_env(env)


@pytest.fixture(scope="module")
def live_client(live_cfg):
    """Client Albert réel ; saute le module si aucune clé n'est disponible."""
    key = _live_key()
    if not key:
        pytest.skip("ALBERT_API_KEY absente du shell et du .env : tests live impossibles")
    return AlbertClient(live_cfg, key)


def test_live_me_account_valid(live_client):
    me = live_client.me()
    assert isinstance(me, dict)
    expires = me.get("expires")
    assert expires is None or float(expires) > time.time()
    assert isinstance(me.get("limits"), list)


def test_live_required_models_present_and_typed(live_client):
    listing = live_client.models()
    data = listing.get("data", listing) if isinstance(listing, dict) else listing
    by_id = {m["id"]: m for m in data}
    for model_id, types in REQUIRED_MODELS.items():
        assert model_id in by_id, model_id
        assert by_id[model_id].get("type") in types, (model_id, by_id[model_id].get("type"))


def test_live_embed_three_texts_1024_sorted(live_client):
    texts = ["Le chat dort sur le canapé.", "La loi de finances est votée.", "Le chat dort sur le canapé."]
    vectors = live_client.embed(texts)
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == 1024
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-3)
    reversed_vectors = live_client.embed(list(reversed(texts)))

    def cosine(a, b):
        """Similarité cosinus de deux vecteurs."""
        return sum(x * y for x, y in zip(a, b)) / (
            math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        )

    # Ordre conservé (tri par index) : chaque vecteur correspond à son texte.
    for i, vector in enumerate(vectors):
        assert cosine(vector, reversed_vectors[2 - i]) > 0.999
    assert cosine(vectors[0], vectors[2]) > 0.999
    assert cosine(vectors[0], vectors[1]) < 0.99


def test_live_health_at_root_not_under_v1(live_client, live_cfg):
    live_client.health()
    with httpx.Client(timeout=30.0, follow_redirects=False) as http:
        assert http.get(root_url(live_cfg.base_url) + "/health").status_code == 200
        assert http.get(live_cfg.base_url.rstrip("/") + "/health").status_code == 404


def test_live_invalid_key_raises_auth_error(live_cfg):
    client = AlbertClient(live_cfg, INVALID_KEY)
    with pytest.raises(AlbertAuthError) as excinfo:
        client.models()
    assert excinfo.value.reason == "invalid_key"
    assert excinfo.value.status == 401
    assert INVALID_KEY not in str(excinfo.value)
