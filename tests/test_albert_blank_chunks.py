"""Chunks au texte vide dans l'espace Albert (lot 9, tâche 0, option A).

Un chunk dont le texte est vide ou blanc n'est jamais envoyé aux embeddings
Albert (bge-m3). Il ne doit donc pas compter comme un embedding manquant :

* **phase dense** (``rad_chunk._finish_albert_embeddings``) : la part de
  manquants est calculée sur les seuls chunks au texte non vide ; une ligne
  unique donne le nombre de chunks vides ignorés ;
* **envoi** (``rad_vectordb._missing_vectors_error`` et la garde des
  connecteurs) : les chunks vides sans vecteur sont retirés (ni envoyés, ni
  comptés comme invalides), avec une ligne unique ; l'envoi réussit ;
* **``rebuild_pinecone_index --all``** ne refuse plus tout le lot pour des
  chunks vides seulement ;
* un vrai vecteur manquant (texte non vide) fait toujours échouer ;
* le chemin par défaut (OpenAI, Albert désactivé) est inchangé : aucun chunk
  retiré, aucune ligne en plus.

Aucun réseau : ``FakeAlbert`` (``httpx.MockTransport``) pour Albert, doubles en
mémoire pour Pinecone, Qdrant et Weaviate ; clés factices seulement.
"""

from __future__ import annotations

import json
import os
import re
import sys
from types import SimpleNamespace

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import rad_vectordb as rv  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402
from tests import test_albert_embeddings as emb  # noqa: E402  (aides de la phase dense Albert)
from tests import test_vector_space_guards as vsg  # noqa: E402  (doubles des bases vectorielles)
from tests.albert_fakes import EMBED_DIM, FakeAlbert, fake_embedding  # noqa: E402

import rebuild_pinecone_index as rebuild  # noqa: E402  (script CLI, importé par son nom)

rc = emb.rc
golden = emb.golden
EMBED_PATH = emb.EMBED_PATH
BLANK_TEXTS = ("", "   \n\t ")
# Ligne unique de la phase dense et de la garde d'envoi (nombre de chunks vides ignorés).
DENSE_BLANK_RE = re.compile(r"^Albert : (\d+) chunk\(s\) au texte vide ignoré\(s\)")
UPLOAD_BLANK_RE = re.compile(r"^Avertissement: (\d+) chunk\(s\) au texte vide de l'espace \S+ ignoré\(s\)")
MISSING_DENSE_WARNING = "Embedding dense manquant"


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
def _dense_blank_counts(out):
    """Nombres portés par les lignes « chunks au texte vide » de la phase dense."""
    return [int(m.group(1)) for m in map(DENSE_BLANK_RE.match, out.splitlines()) if m]


def _upload_blank_counts(out):
    """Nombres portés par les lignes « chunks au texte vide » de la garde d'envoi."""
    return [int(m.group(1)) for m in map(UPLOAD_BLANK_RE.match, out.splitlines()) if m]


def _any_blank_line(out):
    """Vrai si une ligne quelconque mentionne des chunks au texte vide."""
    return "au texte vide" in out


def _dense_chunks_with_blanks(n_text, blanks=BLANK_TEXTS):
    """``n_text`` chunks au texte non vide suivis d'un chunk par texte vide de ``blanks``."""
    chunks = emb._chunks(n_text)
    for j, text in enumerate(blanks):
        base = emb._chunks(1, prefix=f"Vide {j}")[0]
        base.update({"id": f"{999000 + j:012d}_1", "doc_id": f"{999000 + j:012d}", "text": text})
        chunks.append(base)
    return chunks


def _finish(chunks, threshold):
    """Appelle ``_finish_albert_embeddings`` (fichier fictif, seuil ``threshold``) ; renvoie le code de sortie."""
    cfg = SimpleNamespace(embed_max_missing_ratio=threshold)
    try:
        rc._finish_albert_embeddings(chunks, os.path.join(os.sep, "nonexistent", "out.json"), cfg,
                                     emb._albert_space())
        return 0
    except SystemExit as exc:
        return 0 if exc.code is None else exc.code


def _written_albert_chunk(i, *, embedding="auto", text=None):
    """Chunk d'un fichier bge-m3 (espace Albert) : texte ``text`` (défaut : texte non vide), vecteur ``embedding``."""
    chunk = vsg._chunk(i, space="albert")
    if text is not None:
        chunk["text"] = text
    if embedding != "auto":
        chunk["embedding"] = embedding
    return chunk


def _albert_file_with_blanks(n_text=3, *, missing_text=0):
    """Fichier bge-m3 : ``n_text`` chunks vectorisés, ``missing_text`` chunks non vides sans vecteur,
    puis deux chunks au texte vide sans vecteur (``None`` et liste vide)."""
    chunks = [_written_albert_chunk(i) for i in range(1, n_text + 1)]
    for k in range(missing_text):
        chunks.append(_written_albert_chunk(100 + k, embedding=None))
    chunks.append(_written_albert_chunk(200, embedding=None, text=BLANK_TEXTS[0]))
    chunks.append(_written_albert_chunk(201, embedding=[], text=BLANK_TEXTS[1]))
    return chunks


def _upserted_ids_pinecone(fake):
    """Ids envoyés à l'index Pinecone double."""
    return [v["id"] for u in fake.index.upserts for v in u["vectors"]]


def _plain_vectordb():
    """Module ``rad_vectordb`` utilisé par ``rebuild_pinecone_index`` (import par son nom de CLI)."""
    return sys.modules[rebuild.insert_to_pinecone.__module__]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Identifiants, réglages Albert/cache/dédup et réglages CLI retirés ; état Albert remis à zéro."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in emb.CLI_TUNING_NAMES + vsg.CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rv.time, "sleep", lambda *_a, **_k: None)
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    emb._reset_rc_albert_state()
    yield
    emb._reset_rc_albert_state()
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def chunk_env(monkeypatch):
    """``rad_chunk`` figé (un worker, lots de 2) avec des clients OpenAI/OpenRouter enregistreurs."""
    monkeypatch.setattr(rc, "DEFAULT_MAX_WORKERS", 1)
    monkeypatch.setattr(rc, "DEFAULT_DOC_WORKERS", 3)
    monkeypatch.setattr(rc, "DEFAULT_BATCH_SIZE_GPT", 5)
    monkeypatch.setattr(rc, "DEFAULT_EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(rc, "random", golden._RandomProxy())
    monkeypatch.setattr(rc, "time", golden._TimeProxy([]))
    calls = []
    monkeypatch.setattr(rc, "client", golden._FakeLLMClient("openai", calls))
    monkeypatch.setattr(rc, "openrouter_client", golden._FakeLLMClient("openrouter", calls))
    return SimpleNamespace(calls=calls)


@pytest.fixture
def albert_on(monkeypatch, chunk_env):
    """Albert ON, ``EMBEDDING_PROVIDER=albert``, clé factice ; ``FakeAlbert`` sert bge-m3 en 1024 d."""
    fake = FakeAlbert()
    emb.route_albert_to_fake(monkeypatch, fake, emb.SleepRecorder())
    emb._albert_env(monkeypatch, enabled=True, provider="albert")
    return SimpleNamespace(fake=fake, calls=chunk_env.calls)


# ---------------------------------------------------------------------------
# Phase dense : la part de manquants ignore les chunks vides
# ---------------------------------------------------------------------------
def test_dense_blank_chunks_not_missing(albert_on, tmp_path, capsys):
    """Seuil par défaut (0.0) : des chunks vides seuls ne font plus échouer la phase dense."""
    chunks = _dense_chunks_with_blanks(3)
    returned, output_path, written = emb._run_dense(tmp_path, chunks)
    out = capsys.readouterr().out
    assert returned == output_path
    assert written is not None and len(written) == len(chunks)
    by_text = {c["text"]: c for c in written}
    for text in BLANK_TEXTS:
        assert by_text[text].get("embedding") is None
        assert by_text[text]["embedding_provider"] == "albert"
    assert all(len(c["embedding"]) == EMBED_DIM for c in written if c["text"].strip())
    sent = [t for inputs in emb._posted_inputs(albert_on.fake) for t in inputs]
    assert all(t.strip() for t in sent)
    assert len(sent) == 3
    assert _dense_blank_counts(out) == [len(BLANK_TEXTS)]
    assert "ALBERT_EMBED_MAX_MISSING_RATIO" not in out
    assert emb._openai_embed_calls(albert_on.calls) == []


def test_dense_real_missing_still_fails(albert_on, monkeypatch, tmp_path, capsys):
    """Un vrai embedding manquant (texte non vide) fait toujours sortir en 1 ; la part exclut les vides."""
    monkeypatch.setenv("ALBERT_MAX_RETRIES", "0")
    # Lots de 2 dans l'ordre : le premier envoi (2 textes non vides) échoue en 400.
    albert_on.fake.inject("POST", EMBED_PATH, "bad_request")
    chunks = _dense_chunks_with_blanks(3)
    code, _returned, output_path, written = emb._run_dense_allow_exit(tmp_path, chunks, allowed=(1,))
    out = capsys.readouterr().out
    assert code == 1
    assert written is not None and len(written) == len(chunks)
    assert "2/3 embedding(s) Albert manquant(s)" in out
    assert "ALBERT_EMBED_MAX_MISSING_RATIO" in out
    assert _dense_blank_counts(out) == [len(BLANK_TEXTS)]


def test_finish_ratio_over_non_blank_chunks(capsys):
    """Unité : dénominateur = chunks non vides ; seuil comparé à 1/4 et non à 1/8."""
    space = emb._albert_space()
    chunks = [{"text": f"Passage {i} non vide.", "embedding": fake_embedding(f"p{i}", space.dim)}
              for i in range(3)]
    chunks.append({"text": "Passage sans vecteur.", "embedding": None})
    chunks += [{"text": text, "embedding": None} for text in ("", " ", "\n\t", None)]
    assert _finish(chunks, 0.25) == 0  # 1/4 = 0.25, non au-delà du seuil
    out = capsys.readouterr().out
    assert _dense_blank_counts(out) == [4]
    assert _finish(chunks, 0.2) == 1  # 1/4 > 0.2 (l'ancien calcul 1/8 serait passé)
    out = capsys.readouterr().out
    assert "1/4 embedding(s) Albert manquant(s)" in out
    assert _dense_blank_counts(out) == [4]


def test_finish_without_blank_chunks_prints_no_line(capsys):
    """Aucun chunk vide : aucune ligne en plus, comportement antérieur."""
    space = emb._albert_space()
    chunks = [{"text": f"Passage {i}.", "embedding": fake_embedding(f"p{i}", space.dim)} for i in range(3)]
    assert _finish(chunks, 0.0) == 0
    assert capsys.readouterr().out == ""


def test_finish_blank_chunk_with_vector_not_skipped(capsys):
    """Un chunk vide portant un vecteur exploitable n'est ni compté ignoré ni compté manquant."""
    space = emb._albert_space()
    chunks = [{"text": "Passage.", "embedding": fake_embedding("p", space.dim)},
              {"text": "  ", "embedding": fake_embedding("v", space.dim)}]
    assert _finish(chunks, 0.0) == 0
    assert not _any_blank_line(capsys.readouterr().out)


def test_dense_off_path_unchanged(chunk_env, tmp_path, capsys):
    """Chemin OpenAI (Albert non sélectionné) : aucune ligne « texte vide », aucun champ d'espace."""
    chunks = _dense_chunks_with_blanks(3)
    returned, output_path, written = emb._run_dense(tmp_path, chunks)
    out = capsys.readouterr().out
    assert returned == output_path
    assert not _any_blank_line(out)
    assert all(not any(field in c for field in emb.SPACE_FIELDS) for c in written)
    assert emb._openai_embed_calls(chunk_env.calls)


# ---------------------------------------------------------------------------
# Garde d'envoi : _missing_vectors_error et _skip_blank_chunks
# ---------------------------------------------------------------------------
def test_missing_vectors_error_ignores_blank_chunks(capsys):
    """Chunks vides sans vecteur : aucun refus ; contrôle pur, sans sortie."""
    chunks = _albert_file_with_blanks(3)
    space = rv._providers_module().check_uniform_space(chunks)
    assert space is not None and space.dim == EMBED_DIM
    assert rv._missing_vectors_error(chunks, space) is None
    assert capsys.readouterr().out == ""


def test_missing_vectors_error_real_missing_counted_over_non_blank(capsys):
    """Un chunk non vide sans vecteur reste manquant ; la part exclut les chunks vides (1/4)."""
    chunks = _albert_file_with_blanks(3, missing_text=1)
    space = rv._providers_module().check_uniform_space(chunks)
    message = rv._missing_vectors_error(chunks, space, "l'index Pinecone 'articles'")
    assert message is not None
    assert "1/4" in message and "ALBERT_EMBED_MAX_MISSING_RATIO" in message
    assert capsys.readouterr().out == ""


def test_skip_blank_chunks_only_non_default_space(capsys):
    """Seuls les chunks vides sans vecteur d'un espace hors défaut sont retirés, avec une ligne unique."""
    chunks = _albert_file_with_blanks(2)
    ids_before = [c["id"] for c in chunks]
    space = rv._providers_module().check_uniform_space(chunks)
    assert rv._skip_blank_chunks(chunks, space) == 2
    assert [c["id"] for c in chunks] == ids_before[:2]
    assert _upload_blank_counts(capsys.readouterr().out) == [2]
    # Espace par défaut (fichier OpenAI historique) : liste intacte, aucune sortie.
    legacy = vsg._openai_chunks(2) + [vsg._chunk(9, space="openai", embedding=None)]
    legacy[-1]["text"] = ""
    snapshot = json.dumps(legacy, sort_keys=True)
    legacy_space = rv._providers_module().check_uniform_space(legacy)
    assert rv._skip_blank_chunks(legacy, legacy_space) == 0
    assert rv._missing_vectors_error(legacy, legacy_space) is None
    assert json.dumps(legacy, sort_keys=True) == snapshot
    assert capsys.readouterr().out == ""


def test_check_vector_space_dry_run_does_not_skip(capsys):
    """Contrôle préalable (``drop_foreign=False``) : aucune modification, aucune ligne."""
    chunks = _albert_file_with_blanks(2)
    snapshot = json.dumps(chunks, sort_keys=True)
    space, error = rv._check_vector_space(chunks, target_dim=EMBED_DIM, drop_foreign=False)
    assert error is None and space.dim == EMBED_DIM
    assert json.dumps(chunks, sort_keys=True) == snapshot
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Connecteurs : l'envoi ignore les chunks vides et réussit
# ---------------------------------------------------------------------------
def test_pinecone_upload_skips_blank_chunks(monkeypatch, tmp_path, capsys):
    """Pinecone : statut ``success``, chunks vides non envoyés, une ligne unique."""
    fake = vsg._install_pinecone(monkeypatch, SimpleNamespace(metric="dotproduct", dimension=EMBED_DIM))
    chunks = _albert_file_with_blanks(3)
    expected_ids = [c["id"] for c in chunks[:3]]
    result = rv.insert_to_pinecone(vsg._write(tmp_path, chunks), index_name="articles",
                                   pinecone_api_key=vsg.FAKE_PINECONE_KEY)
    out = capsys.readouterr().out
    assert result["status"] == "success"
    assert result["inserted_count"] == 3
    assert _upserted_ids_pinecone(fake) == expected_ids
    assert _upload_blank_counts(out) == [2]
    assert MISSING_DENSE_WARNING not in out


def test_qdrant_upload_skips_blank_chunks(monkeypatch, tmp_path, capsys):
    """Qdrant (collection créée depuis le fichier) : ``success``, 3 points, une ligne unique."""
    fake = vsg._install_qdrant(monkeypatch, vsg.FakeQdrant(existing_vectors=None))
    result = rv.insert_to_qdrant(vsg._write(tmp_path, _albert_file_with_blanks(3)), "corpus",
                                 qdrant_url=vsg.FAKE_QDRANT_URL)
    out = capsys.readouterr().out
    assert result["status"] == "success"
    assert result["inserted_count"] == 3
    assert len(fake.created) == 1 and fake.created[0]["vectors_config"].size == EMBED_DIM
    assert sum(len(u["points"]) for u in fake.upserts) == 3
    assert _upload_blank_counts(out) == [2]


def test_weaviate_upload_skips_blank_chunks(monkeypatch, tmp_path, capsys):
    """Weaviate : ``success``, 3 objets insérés, une ligne unique."""
    fake = vsg._install_weaviate(monkeypatch, vsg.FakeWeaviate(target_vectors=()))
    result = vsg._insert_weaviate(vsg._write(tmp_path, _albert_file_with_blanks(3)))
    out = capsys.readouterr().out
    assert result["status"] == "success"
    assert result["inserted_count"] == 3
    objects = [obj for call in fake.tenant.data.insert_many.call_args_list for obj in call.args[0]]
    assert len(objects) == 3
    assert _upload_blank_counts(out) == [2]


@pytest.mark.parametrize("connector", ["pinecone", "qdrant", "weaviate"])
def test_upload_real_missing_vector_still_fails(connector, monkeypatch, tmp_path, capsys):
    """Un chunk non vide sans vecteur : refus avant tout envoi, même avec des chunks vides."""
    path = vsg._write(tmp_path, _albert_file_with_blanks(3, missing_text=1))
    if connector == "pinecone":
        fake = vsg._install_pinecone(monkeypatch, SimpleNamespace(metric="dotproduct", dimension=EMBED_DIM))
        result = rv.insert_to_pinecone(path, index_name="articles", pinecone_api_key=vsg.FAKE_PINECONE_KEY)
        touched = fake.index.upserts + fake.index.fetches
    elif connector == "qdrant":
        fake = vsg._install_qdrant(monkeypatch, vsg.FakeQdrant(existing_vectors=vsg._vectors_params(EMBED_DIM)))
        result = rv.insert_to_qdrant(path, "corpus", qdrant_url=vsg.FAKE_QDRANT_URL)
        touched = fake.upserts + fake.created + fake.retrieves
    else:
        fake = vsg._install_weaviate(monkeypatch, vsg.FakeWeaviate(target_vectors=()))
        result = vsg._insert_weaviate(path)
        touched = [None] * fake.inserted
    out = capsys.readouterr().out
    assert result["status"] == "error"
    assert result["inserted_count"] == 0
    assert "1/4" in vsg._message(result)
    assert "ALBERT_EMBED_MAX_MISSING_RATIO" in vsg._message(result)
    assert touched == []
    assert not _any_blank_line(out)


def test_upload_off_path_unchanged(monkeypatch, tmp_path, capsys):
    """Fichier de l'espace par défaut avec un chunk vide sans vecteur : comportement historique."""
    fake = vsg._install_pinecone(monkeypatch, SimpleNamespace(metric="dotproduct", dimension=vsg.OPENAI_DIM))
    chunks = vsg._openai_chunks(2) + [vsg._chunk(9, space="openai", embedding=None)]
    chunks[-1]["text"] = ""
    result = rv.insert_to_pinecone(vsg._write(tmp_path, chunks), index_name="articles",
                                   pinecone_api_key=vsg.FAKE_PINECONE_KEY)
    out = capsys.readouterr().out
    assert result["status"] == "success_partial_data"
    assert result["inserted_count"] == 2
    assert len(_upserted_ids_pinecone(fake)) == 2
    assert MISSING_DENSE_WARNING in out
    assert not _any_blank_line(out)


# ---------------------------------------------------------------------------
# rebuild_pinecone_index --all
# ---------------------------------------------------------------------------
@pytest.fixture
def rebuild_real_upload(monkeypatch):
    """``rebuild_pinecone_index`` avec le vrai ``insert_to_pinecone`` sur un double Pinecone partagé."""
    monkeypatch.setenv("PINECONE_API_KEY", vsg.FAKE_PINECONE_KEY)
    fake_pc = vsg.FakePinecone(SimpleNamespace(metric="dotproduct", dimension=EMBED_DIM, status={"ready": True}))
    plain = _plain_vectordb()
    monkeypatch.setattr(rebuild, "Pinecone", lambda *a, **k: fake_pc)
    monkeypatch.setattr(plain, "Pinecone", lambda *a, **k: fake_pc)
    monkeypatch.setattr(plain.time, "sleep", lambda *_a, **_k: None)
    recreated = []

    def fake_recreate(pc, name, dimension, metric, cloud, region):
        """Enregistre la recréation demandée (aucune suppression réelle)."""
        recreated.append(dimension)

    monkeypatch.setattr(rebuild, "recreate_index", fake_recreate)
    return SimpleNamespace(pc=fake_pc, recreated=recreated)


def test_rebuild_all_accepts_blank_chunks(rebuild_real_upload, monkeypatch, tmp_path, capsys):
    """``--all`` : deux sessions bge-m3 avec des chunks vides → recréation, envoi, code 0."""
    uploads = tmp_path / "uploads"
    first = vsg._session_file(tmp_path, "session_a", _albert_file_with_blanks(3))
    second_chunks = [_written_albert_chunk(i, text=f"Autre passage {i} du second corpus.") for i in (1, 2)]
    for chunk in second_chunks:
        chunk["embedding"] = fake_embedding(chunk["text"], EMBED_DIM)
        chunk["id"] = f"222222222222_{chunk['chunk_index']}"
    second_chunks.append(_written_albert_chunk(3, embedding=None, text=""))
    vsg._session_file(tmp_path, "session_b", second_chunks)
    assert os.path.isfile(first)
    code = vsg._rebuild_main(monkeypatch, ["--index", "articles", "--recreate", "--dimension", EMBED_DIM,
                                           "--uploads-dir", uploads, "--all"])
    out = capsys.readouterr().out
    assert code == 0
    assert rebuild_real_upload.recreated == [EMBED_DIM]
    sent = _upserted_ids_pinecone(rebuild_real_upload.pc)
    assert len(sent) == 5
    assert re.findall(r"status=(\w+), inserted=(\d+)", out) == [("success", "3"), ("success", "2")]
    assert _upload_blank_counts(out) == [2, 1]
    assert "ERREUR" not in out


def test_rebuild_all_real_missing_refused(rebuild_real_upload, monkeypatch, tmp_path, capsys):
    """``--all`` : un vrai vecteur manquant reste refusé en pré-vol (code 2, rien recréé ni envoyé)."""
    uploads = tmp_path / "uploads"
    vsg._session_file(tmp_path, "session_a", _albert_file_with_blanks(3, missing_text=1))
    code = vsg._rebuild_main(monkeypatch, ["--index", "articles", "--recreate", "--dimension", EMBED_DIM,
                                           "--uploads-dir", uploads, "--all"])
    out = capsys.readouterr().out
    assert code == 2
    assert rebuild_real_upload.recreated == []
    assert rebuild_real_upload.pc.index.upserts == []
    assert "ALBERT_EMBED_MAX_MISSING_RATIO" in out
