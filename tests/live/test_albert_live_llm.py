"""Tests live du chat Albert (lot 3) : recodage, note courte gpt-oss, filtre de citations.

Marqueur ``albert_live`` : sautés sauf si ``ALBERT_LIVE=1`` est exporté dans le
shell (``ALBERT_LIVE`` ne doit jamais figurer dans un fichier ``.env``). Appels
d'inférence seulement (aucune collection). La clé vient de l'environnement du
shell, sinon de ``ALBERT_API_KEY`` dans le ``.env`` racine ; elle n'est jamais
affichée ni comparée à une valeur.

Commande : ``ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_llm.py -m albert_live -v -p no:cacheprovider``
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

RAGPY_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(RAGPY_ROOT), str(RAGPY_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.albert_live

RECODE_MODEL = "ministral-3-8b-instruct-2512"
NOTES_MODEL = "gpt-oss-120b"
RAW_CHUNKS = [
    "12 REVUE DE PHILOSOPHIE\n\nLe langage ordinaire structure l'expé- rience sociale ; chaque énoncé "
    "renvoie à un contexte d'usage précis et situé. Page 12",
    "Chapitre 3\n\nWittgenstein insiste sur les jeux de langage : la signification d'un mot tient à "
    "son usage dans une forme de vie. 47",
    "Les enquêtes de terrain montrent que les locuteurs ajustent leurs formulations aux attentes "
    "de leurs interlocuteurs , ce que la sociologie pragmatique décrit comme une épreuve . 103",
]


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


@pytest.fixture
def albert_on_env(monkeypatch, live_key):
    """Albert ON pour le test : variables d'environnement et constantes des modules CLI."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", live_key)
    import rad_chunk

    for name, value in (("ALBERT_ENABLED", True), ("ALBERT_API_KEY", live_key)):
        if hasattr(rad_chunk, name):
            monkeypatch.setattr(rad_chunk, name, value)
    rad_chunk.reset_albert_state()
    yield rad_chunk
    rad_chunk.reset_albert_state()


CHAT_ENDPOINT = "/v1/chat/completions"


def test_live_recode_three_chunks_ministral(albert_on_env):
    rc = albert_on_env
    served = []
    texts, statuses = rc.gpt_recode_batch(RAW_CHUNKS, rc._RECODE_INSTRUCTIONS, model="albert/" + RECODE_MODEL,
                                          models_out=served)
    assert statuses == ["recoded", "recoded", "recoded"]
    assert all(isinstance(text, str) and text.strip() for text in texts)
    assert served == [RECODE_MODEL] * 3
    # Les trois requêtes de recodage elles-mêmes : l'id épinglé part sur le fil et
    # la réponse le recopie (pas un alias), sans modèle de repli.
    assert rc._ALBERT_CLIENT is not None
    chats = [record for record in rc._ALBERT_CLIENT.ledger.records if record["endpoint"] == CHAT_ENDPOINT]
    assert len(chats) == 3
    for record in chats:
        assert record["role"] == "recode"
        assert record["model"] == RECODE_MODEL
        assert record["response_model"] == RECODE_MODEL
        assert record["fallback_from"] is None


def test_live_short_note_gpt_oss_not_empty(albert_on_env, live_key):
    from app.utils import llm_note_generator as lng

    metadata = {"title": "Le langage ordinaire", "authors": "Dupont, Jeanne", "date": "2021",
                "abstract": "Étude des usages du langage ordinaire en sociologie.", "language": "fr"}
    prompt = lng._build_prompt(metadata, " ".join(RAW_CHUNKS), "fr", mode="short")
    content = lng._generate_with_llm(prompt, model="albert/" + NOTES_MODEL, mode="short", albert_api_key=live_key)
    assert isinstance(content, str) and content.strip()


def test_live_citation_filter_valid_json(albert_on_env, live_key):
    from app.utils import citation_filter as cfilter

    citation = {
        "title": "Philosophical Investigations",
        "authors": ["Wittgenstein, Ludwig"],
        "abstract": "Language games and meaning as use.",
        "source": "Blackwell",
        "year": 1953,
    }
    result = asyncio.run(cfilter.filter_citation_with_llm(
        citation, "", "none",
        "Langage ordinaire", "Philosophie du langage ordinaire et usages sociaux du langage.",
        "Corpus", "Textes fondateurs.",
        model="albert/" + RECODE_MODEL, albert_api_key=live_key,
    ))
    assert result == "NA" or (isinstance(result, dict) and 0 <= result["relevance_score"] <= 100
                              and result["zotero_item"]["title"])
