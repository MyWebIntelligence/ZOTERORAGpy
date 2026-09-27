"""Audit du 2026-09-27 : intégrité du pipeline (A05, A06, A07, A10).

A05 — identité des chunks dédupliqués = contenu + source stable, sans la
position (scénarios reproduits par l'audit, adaptateur Pinecone réel sur un
index factice) :

* même texte, titres différents dans un lot : les deux sont conservés ;
* mêmes données sur deux runs avec un autre titre : ids distincts, aucune
  provenance remplacée, les deux gardées dans le JSON fusionné ;
* texte et titre identiques, index décalé : reconnu comme déjà présent ;
* migration : un vecteur indexé sous l'id v1 historique reste reconnu ;
* chunk sans titre : jamais dédupliqué, id aléatoire conservé.

A06 — encodage sparse stable entre processus (graines de hachage différentes)
et collisions additionnées.

A07 — plus aucun vecteur nul d'échec : 401, 429 épuisé, délai dépassé,
mauvais nombre de réponses, mauvaise dimension, NaN et vecteur nul donnent un
embedding absent, jamais mis en cache ni compté ni inséré ; la phase sort en 1
au-delà du seuil ; une entrée de cache invalide est purgée.

A10 — sauvegarde de la phase initiale : écritures bornées pendant un lot,
écriture atomique, fichier corrompu conservé à part.

Aucun réseau : clients OpenAI et index Pinecone factices.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from openai import APITimeoutError, AuthenticationError, RateLimitError

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rad_dedup as d  # noqa: E402
import rad_sparse  # noqa: E402
from rad_providers import valid_dense_vector  # noqa: E402

TEXT = ("Le langage ordinaire structure l'expérience sociale ; cet énoncé renvoie à un contexte "
        "d'usage précis et situé, assez long pour être éligible à la déduplication.")


# ======================================================================
# A05 — identité (contenu, source) et migration
# ======================================================================
def _chunk(title, index=1, text=TEXT, cid=None):
    """Chunk dédupliqué comme l'écrit ``rad_chunk`` (id v2 quand la source est stable)."""
    skey = d.source_key({"title": title})
    chash, eligible, new_id = d.compute_dedup_fields(text, index, 64, source=skey)
    return {"id": cid or new_id, "content_hash": chash, "dedup_eligible": eligible,
            "chunk_index": index, "title": title, "text": text, "doc_id": "123"}


class _FakePineconeIndex:
    """Index Pinecone factice : ``fetch`` par ids, ``upsert`` qui remplace par id."""

    def __init__(self):
        """Index vide."""
        self.vectors = {}
        self.fetched = []

    def fetch(self, ids, namespace=None):
        """Vecteurs présents parmi ``ids`` (métadonnées seules)."""
        self.fetched.append(list(ids))
        found = {vid: SimpleNamespace(metadata=dict(self.vectors[vid])) for vid in ids if vid in self.vectors}
        return SimpleNamespace(vectors=found)

    def upsert_chunks(self, chunks):
        """Insère ou remplace chaque chunk sous son id (sémantique d'upsert)."""
        for chunk in chunks:
            meta = {k: v for k, v in chunk.items() if k not in ("id", "embedding")}
            self.vectors[chunk["id"]] = meta


@pytest.fixture
def dedup_on(monkeypatch, tmp_path):
    """``DEDUP_ENABLED=1``, champs de source par défaut (titre), journal temporaire."""
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    monkeypatch.delenv("DEDUP_META_FIELDS", raising=False)
    monkeypatch.setenv("DEDUP_JOURNAL_DIR", str(tmp_path / "journal"))
    return d.DedupConfig.from_env()


def _adapter(index):
    """Adaptateur Pinecone réel de ``rad_vectordb`` sur l'index factice."""
    import rad_vectordb

    return rad_vectordb._PineconeDedupAdapter(index)


def test_same_text_different_titles_in_one_batch_are_both_kept(dedup_on):
    """Tier 1 : même texte, titres différents -> deux provenances conservées (avant : une rejetée)."""
    a, b = _chunk("Titre A"), _chunk("Titre B")
    kept, rejected = d.dedup_filter([a, b], _adapter(_FakePineconeIndex()), dedup_on)
    assert kept == [a, b] and rejected == []
    assert a["id"] != b["id"]


def test_same_text_same_title_in_one_batch_is_rejected_once(dedup_on):
    """Tier 1 : même texte ET même source -> un seul conservé (doublon exact)."""
    a, b = _chunk("Titre A", 1), _chunk("Titre A", 7)
    kept, rejected = d.dedup_filter([a, b], _adapter(_FakePineconeIndex()), dedup_on)
    assert kept == [a] and [r["reason"] for r in rejected] == ["in_batch_duplicate"]


def test_second_run_with_other_title_keeps_both_provenances(dedup_on):
    """Deux runs, même texte, autre titre : id distinct, l'upsert ne remplace pas la première provenance."""
    index = _FakePineconeIndex()
    first = _chunk("Titre A")
    kept, _ = d.dedup_filter([first], _adapter(index), dedup_on)
    index.upsert_chunks(kept)
    second = _chunk("Titre B")
    kept, rejected = d.dedup_filter([second], _adapter(index), dedup_on)
    assert kept == [second] and rejected == []
    index.upsert_chunks(kept)
    titles = sorted(meta["title"] for meta in index.vectors.values())
    assert titles == ["Titre A", "Titre B"]


def test_shifted_index_same_source_is_recognised(dedup_on):
    """Texte et titre identiques, index passé de 1 à 2 : déjà présent (l'id ne contient plus la position)."""
    index = _FakePineconeIndex()
    index.upsert_chunks([_chunk("Titre A", 1)])
    moved = _chunk("Titre A", 2)
    kept, rejected = d.dedup_filter([moved], _adapter(index), dedup_on)
    assert kept == [] and [r["reason"] for r in rejected] == ["existing_hash_match"]


def test_legacy_v1_vector_is_still_recognised(dedup_on):
    """Migration : un vecteur indexé sous l'id v1 ``{hash}_{index}`` est retrouvé (double lecture)."""
    index = _FakePineconeIndex()
    legacy = _chunk("Titre A", 3)
    legacy["id"] = d.content_id(legacy["content_hash"], 3)
    index.upsert_chunks([legacy])
    fresh = _chunk("Titre A", 3)
    assert fresh["id"] != legacy["id"]
    kept, rejected = d.dedup_filter([fresh], _adapter(index), dedup_on)
    assert kept == [] and rejected and rejected[0]["matched_existing"][0]["existing_id"] == legacy["id"]


def test_legacy_v1_vector_of_other_source_is_not_a_duplicate(dedup_on):
    """Vecteur v1 d'un autre document (même texte, même index) : pas de corroboration, nouveau chunk gardé."""
    index = _FakePineconeIndex()
    legacy = _chunk("Titre A", 3)
    legacy["id"] = d.content_id(legacy["content_hash"], 3)
    index.upsert_chunks([legacy])
    other = _chunk("Titre B", 3)
    kept, rejected = d.dedup_filter([other], _adapter(index), dedup_on)
    assert kept == [other] and rejected == []
    assert other["id"] != legacy["id"]  # l'upsert ne remplacera pas le vecteur v1


def test_chunk_without_title_is_never_deduplicated(dedup_on):
    """Sans source stable (titre vide) : inéligible, id conservé, jamais rejeté."""
    chash, eligible, cid = d.compute_dedup_fields(TEXT, 1, 64, source=d.source_key({"title": ""}))
    assert eligible is False
    chunk = {"id": "random_1", "content_hash": chash, "dedup_eligible": eligible, "chunk_index": 1,
             "title": "", "text": TEXT}
    kept, rejected = d.dedup_filter([chunk, dict(chunk, id="random_2")], _adapter(_FakePineconeIndex()), dedup_on)
    assert len(kept) == 2 and rejected == []


def test_write_path_ids_distinguish_sources_and_merge_keeps_both(dedup_on, monkeypatch, tmp_path):
    """``process_document_chunks`` : deux documents au même texte -> deux ids, le JSON fusionné garde les deux."""
    rc = _rad_chunk(monkeypatch)
    json_file = tmp_path / "output_chunks.json"
    for title in ("Titre A", "Titre B"):
        row = pd.Series({"filename": "f.pdf", "title": title, "texteocr": TEXT, "texteocr_provider": "mistral"})
        rc.process_document_chunks(row, json_file=str(json_file), model="gpt-4o-mini")
    chunks = json.loads(json_file.read_text(encoding="utf-8"))
    assert sorted(chunk["title"] for chunk in chunks) == ["Titre A", "Titre B"]
    assert len({chunk["id"] for chunk in chunks}) == 2
    # Ré-ingest du même document : idempotent (même id, fusion par id).
    row = pd.Series({"filename": "f.pdf", "title": "Titre A", "texteocr": TEXT, "texteocr_provider": "mistral"})
    rc.process_document_chunks(row, json_file=str(json_file), model="gpt-4o-mini")
    assert len(json.loads(json_file.read_text(encoding="utf-8"))) == 2


def test_backfill_tool_targets_v2_ids():
    """Outil de back-fill in-place : cible v2 par défaut (migration), v1 sur demande, sans titre -> intact."""
    import backfill_pinecone_inplace as tool

    chash = d.content_hash(TEXT)
    target, eligible = tool.target_id_for({"title": "Titre A"}, chash, 4, True, "old")
    assert eligible and target == d.source_content_id(chash, d.source_key({"title": "Titre A"}))
    target, eligible = tool.target_id_for({"title": "Titre A"}, chash, 4, True, "old", id_scheme="v1")
    assert eligible and target == d.content_id(chash, 4)
    target, eligible = tool.target_id_for({"title": ""}, chash, 4, True, "old")
    assert (target, eligible) == ("old", False)


# ======================================================================
# A06 — encodage sparse stable
# ======================================================================
def test_sparse_index_is_stable_across_processes():
    """``recherche`` : même indice dans deux processus aux graines de hachage différentes."""
    code = ("import sys; sys.path.insert(0, 'scripts'); import rad_sparse; "
            "print(rad_sparse.sparse_index('recherche'), rad_sparse.sparse_vector_from_lemmas(['a', 'b', 'a']))")
    outputs = set()
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run([sys.executable, "-c", code], cwd=RAGPY_ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1
    assert outputs.pop().startswith(str(rad_sparse.sparse_index("recherche")))


def test_sparse_collisions_are_summed(monkeypatch):
    """Deux lemmes sur le même indice : poids additionnés (avant : le second écrasait le premier)."""
    monkeypatch.setattr(rad_sparse, "sparse_index", lambda lemma: 42)
    vec = rad_sparse.sparse_vector_from_lemmas(["alpha", "beta", "beta", "gamma"])
    assert vec == {"indices": ["42"], "values": [1.0]}


def test_sparse_vector_sorted_and_normalised():
    """Indices triés (chaînes), poids TF dont la somme vaut 1."""
    vec = rad_sparse.sparse_vector_from_lemmas(["b", "a", "a", "c"])
    assert vec["indices"] == sorted(vec["indices"], key=int)
    assert abs(sum(vec["values"]) - 1.0) < 1e-12
    assert rad_sparse.sparse_vector_from_lemmas([]) == {"indices": [], "values": []}


def test_extract_sparse_features_uses_the_stable_encoding(monkeypatch):
    """``rad_chunk.extract_sparse_features`` délègue à ``rad_sparse`` (plus de ``hash()`` salé)."""
    rc = _rad_chunk(monkeypatch)
    doc = [SimpleNamespace(lemma_="Recherche", pos_="NOUN", is_stop=False, is_punct=False),
           SimpleNamespace(lemma_="sociale", pos_="ADJ", is_stop=False, is_punct=False),
           SimpleNamespace(lemma_="le", pos_="DET", is_stop=True, is_punct=False)]
    monkeypatch.setattr(rc, "nlp", _CallableNlp(1000000, doc))
    assert rc.extract_sparse_features("x") == rad_sparse.sparse_vector_from_lemmas(["recherche", "sociale"])


class _CallableNlp:
    """Modèle spaCy factice : ``max_length`` et appel renvoyant des jetons fixes."""

    def __init__(self, max_length, doc):
        """Mémorise la longueur maximale et les jetons."""
        self.max_length = max_length
        self._doc = doc

    def __call__(self, text):
        """Jetons fixes, quel que soit le texte."""
        return self._doc


# ======================================================================
# A07 — embeddings : aucun faux succès durable
# ======================================================================
def _rad_chunk(monkeypatch):
    """Module ``rad_chunk`` avec une clé factice le temps de l'import."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "fake-openai-audit-0001")
    import rad_chunk as rc

    return rc


def _http_error(cls, status, message="erreur simulée"):
    """Exception OpenAI d'un statut HTTP donné (requête factice, aucun réseau)."""
    request = httpx.Request("POST", "https://api.openai.invalid/v1/embeddings")
    return cls(message, response=httpx.Response(status, request=request), body=None)


class _FakeEmbeddings:
    """``client.embeddings`` factice : ``behaviour(input)`` renvoie une réponse ou lève."""

    def __init__(self, behaviour):
        """Mémorise le comportement ; compte les appels."""
        self.behaviour = behaviour
        self.calls = []

    def create(self, **kwargs):
        """Enregistre l'appel et applique le comportement."""
        self.calls.append(list(kwargs["input"]))
        return self.behaviour(list(kwargs["input"]))


def _vector(seed, dim=8):
    """Vecteur valide déterministe."""
    return [0.1 * (seed + 1) + 0.01 * i for i in range(dim)]


def _response(vectors):
    """Réponse ``embeddings.create`` factice (index = position)."""
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)])


@pytest.fixture
def openai_embed(monkeypatch):
    """``rad_chunk`` avec un client d'embeddings factice réglable, sans attente réelle ni cache."""
    rc = _rad_chunk(monkeypatch)
    monkeypatch.setattr(rc.time, "sleep", lambda *_a: None)
    monkeypatch.delenv("RECODE_EMBED_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBED_MAX_MISSING_RATIO", raising=False)
    rc.reset_openai_embed_state()
    state = SimpleNamespace(rc=rc)

    def install(behaviour):
        """Installe un client dont ``embeddings.create`` suit ``behaviour``."""
        fake = _FakeEmbeddings(behaviour)
        monkeypatch.setattr(rc, "client", SimpleNamespace(embeddings=fake))
        state.fake = fake
        return fake

    state.install = install
    yield state
    rc.reset_openai_embed_state()


def _raise(exc):
    """Comportement qui lève toujours ``exc``."""
    def behaviour(_texts):
        """Comportement simulé de ``embeddings.create`` pour ce test."""
        raise exc
    return behaviour


def test_authentication_error_gives_none_without_cascade(openai_embed):
    """401 : ``None`` pour tout le lot, aucun appel individuel, lots suivants sans appel."""
    fake = openai_embed.install(_raise(_http_error(AuthenticationError, 401)))
    assert openai_embed.rc.get_embeddings_batch(["un", "deux", "trois"]) == [None, None, None]
    assert len(fake.calls) == 1
    assert openai_embed.rc.get_embeddings_batch(["quatre"]) == [None]
    assert len(fake.calls) == 1


def test_rate_limit_exhausted_raises_and_batch_gets_no_vector(openai_embed, tmp_path):
    """429 après les reprises : le lot échoue sans vecteur nul ; la phase sort en 1, fichier écrit."""
    openai_embed.install(_raise(_http_error(RateLimitError, 429)))
    with pytest.raises(RateLimitError):
        openai_embed.rc.get_embeddings_batch(["un"], max_retries=1)
    chunks_file = _chunks_file(tmp_path, 2)
    out = tmp_path / "dense.json"
    with pytest.raises(SystemExit) as exc:
        openai_embed.rc.generate_and_save_embeddings(str(chunks_file), str(out))
    assert exc.value.code == 1
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [chunk["embedding"] for chunk in written] == [None, None]


def test_timeout_falls_back_individually_and_marks_failures(openai_embed):
    """Délai dépassé sur le lot : reprise texte par texte ; un texte en échec reste ``None``."""
    def behaviour(texts):
        """Comportement simulé de ``embeddings.create`` pour ce test."""
        if len(texts) > 1:
            raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.invalid"))
        if texts == ["deux"]:
            raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.invalid"))
        return _response([_vector(1)])

    openai_embed.install(behaviour)
    vectors = openai_embed.rc.get_embeddings_batch(["un", "deux"])
    assert vectors[0] == _vector(1) and vectors[1] is None


@pytest.mark.parametrize("bad", ["short_response", "nan", "zero", "empty", "strings"])
def test_invalid_responses_never_become_vectors(openai_embed, bad):
    """Mauvais nombre de réponses, NaN, vecteur nul, vide ou non numérique : ``None`` à cette place."""
    def behaviour(texts):
        """Comportement simulé de ``embeddings.create`` pour ce test."""
        vectors = [_vector(i) for i in range(len(texts))]
        if bad == "short_response":
            vectors = vectors[:-1]
        elif bad == "nan":
            vectors[-1] = [float("nan")] * 8
        elif bad == "zero":
            vectors[-1] = [0.0] * 8
        elif bad == "empty":
            vectors[-1] = []
        elif bad == "strings":
            vectors[-1] = ["x"] * 8
        return _response(vectors)

    openai_embed.install(behaviour)
    vectors = openai_embed.rc.get_embeddings_batch(["un", "deux"])
    assert vectors[0] == _vector(0) and vectors[1] is None


def _chunks_file(tmp_path, count, blank_last=False):
    """Fichier ``output_chunks.json`` de ``count`` chunks (le dernier au texte vide si demandé)."""
    chunks = [{"id": f"doc_{i}", "chunk_index": i, "text": f"texte {i} du chunk", "title": "Doc"}
              for i in range(count)]
    if blank_last:
        chunks[-1]["text"] = "   "
    path = tmp_path / "output_chunks.json"
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return path


def test_dense_phase_counts_only_real_successes_and_fails_above_threshold(openai_embed, tmp_path, capsys):
    """Un chunk sans vecteur : compté comme manquant (pas comme généré), phase en échec (seuil 0)."""
    def behaviour(texts):
        """Comportement simulé de ``embeddings.create`` pour ce test."""
        return _response([[0.0] * 8 if text.startswith("texte 1") else _vector(i) for i, text in enumerate(texts)])

    openai_embed.install(behaviour)
    out = tmp_path / "dense.json"
    with pytest.raises(SystemExit) as exc:
        openai_embed.rc.generate_and_save_embeddings(str(_chunks_file(tmp_path, 3)), str(out))
    assert exc.value.code == 1
    stdout = capsys.readouterr().out
    assert "Performance: 2 embeddings" in stdout
    assert "1/3 embedding(s) OpenAI manquant(s)" in stdout
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written[1]["embedding"] is None
    assert all(valid_dense_vector(c["embedding"]) for i, c in enumerate(written) if i != 1)


def test_dense_phase_threshold_allows_partial_runs(openai_embed, tmp_path, monkeypatch):
    """``EMBED_MAX_MISSING_RATIO=0.5`` : un manque sur trois est toléré (fichier écrit, sortie normale)."""
    monkeypatch.setenv("EMBED_MAX_MISSING_RATIO", "0.5")
    openai_embed.install(lambda texts: _response(
        [[0.0] * 8 if text.startswith("texte 1") else _vector(i) for i, text in enumerate(texts)]))
    out = tmp_path / "dense.json"
    assert openai_embed.rc.generate_and_save_embeddings(str(_chunks_file(tmp_path, 3)), str(out)) == str(out)


def test_blank_texts_are_never_sent_and_not_missing(openai_embed, tmp_path):
    """Texte vide : jamais envoyé à OpenAI, sans vecteur, non compté comme manquant."""
    fake = openai_embed.install(lambda texts: _response([_vector(i) for i in range(len(texts))]))
    out = tmp_path / "dense.json"
    openai_embed.rc.generate_and_save_embeddings(str(_chunks_file(tmp_path, 3, blank_last=True)), str(out))
    assert all("   " not in batch for batch in fake.calls)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written[-1]["embedding"] is None


def test_mismatched_dimension_is_removed(openai_embed, tmp_path):
    """Un vecteur d'une autre dimension que la majorité du run : retiré (manquant)."""
    openai_embed.install(lambda texts: _response(
        [_vector(i, dim=4) if text.startswith("texte 2") else _vector(i) for i, text in enumerate(texts)]))
    out = tmp_path / "dense.json"
    with pytest.raises(SystemExit):
        openai_embed.rc.generate_and_save_embeddings(str(_chunks_file(tmp_path, 3)), str(out))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written[2]["embedding"] is None and len(written[0]["embedding"]) == 8


def test_invalid_cached_vector_is_purged_and_recomputed(openai_embed, tmp_path, monkeypatch):
    """Cache d'embeddings : une ancienne entrée nulle est purgée puis recalculée ; un échec n'est jamais caché."""
    rc = openai_embed.rc
    monkeypatch.setenv("RECODE_EMBED_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    cfg = rc.rad_recode_cache.RecodeConfig.from_env()
    params = json.dumps({"model": "text-embedding-3-large"}, sort_keys=True)
    key = rc.rad_recode_cache.embed_key("un", "text-embedding-3-large", params)
    # Entrée invalide d'une ancienne version (vecteur nul), insérée directement.
    conn = rc.rad_recode_cache._get_conn(cfg.cache_path)
    conn.execute("INSERT OR REPLACE INTO embed_cache(key, vector) VALUES(?, ?)", (key, json.dumps([0.0] * 8)))
    conn.commit()
    fake = openai_embed.install(lambda texts: _response([_vector(7) for _ in texts]))
    assert rc._embed_with_cache(["un"], cfg) == [_vector(7)]
    assert fake.calls == [["un"]]
    assert rc.rad_recode_cache.get_embed(cfg, key) == _vector(7)
    # Un vecteur invalide n'est jamais mis en cache.
    other = rc.rad_recode_cache.embed_key("deux", "text-embedding-3-large", params)
    rc.rad_recode_cache.put_embed(cfg, other, [0.0] * 8)
    assert rc.rad_recode_cache.get_embed(cfg, other) is None


def test_connectors_never_insert_zero_or_nan_vectors(capsys):
    """Chemin d'insertion commun : vecteurs nuls ou NaN retirés (``_drop_foreign_vectors``, préparation Pinecone)."""
    import rad_vectordb

    chunks = [
        {"id": "ok", "text": "a", "embedding": [0.1] * 8},
        {"id": "zero", "text": "b", "embedding": [0.0] * 8},
        {"id": "nan", "text": "c", "embedding": [float("nan")] + [0.1] * 7},
    ]
    space, error = rad_vectordb._check_vector_space(chunks)
    assert error is None and space is not None
    assert [c["embedding"] is not None for c in chunks] == [True, False, False]
    assert "vecteur nul ou non fini" in capsys.readouterr().out
    prepared = rad_vectordb.prepare_vectors_for_pinecone([
        {"id": "ok", "text": "a", "embedding": [0.1] * 8},
        {"id": "zero", "text": "b", "embedding": [0.0] * 8},
    ])
    assert [v["id"] for v in prepared] == ["ok"]


# ======================================================================
# A10 — checkpoints de la phase initiale
# ======================================================================
def test_batch_save_writes_are_bounded(monkeypatch, tmp_path):
    """Lot de 50 documents : 2 écritures (premier point de contrôle + fin), plus une par document."""
    rc = _rad_chunk(monkeypatch)
    monkeypatch.setattr(rc, "CHUNK_CHECKPOINT_SECONDS", 3600.0)
    writes = []
    real_write = rc.write_json_atomic

    def counting_write(data, json_file):
        """Compte les écritures puis écrit réellement."""
        writes.append(len(data))
        real_write(data, json_file)

    monkeypatch.setattr(rc, "write_json_atomic", counting_write)
    json_file = str(tmp_path / "output_chunks.json")
    rc._forget_chunk_accumulator(json_file)
    for i in range(50):
        rc.save_raw_chunks_to_json_incrementally([{"id": f"d{i}_1", "text": "t"}], json_file, flush=False)
    rc.flush_chunk_checkpoint(json_file)
    assert writes == [1, 50]
    assert len(json.loads(open(json_file, encoding="utf-8").read())) == 50


def test_direct_save_keeps_immediate_write(monkeypatch, tmp_path):
    """Appel direct (``flush=True``) : fichier écrit à chaque appel, relu d'un appel à l'autre."""
    rc = _rad_chunk(monkeypatch)
    json_file = tmp_path / "output_chunks.json"
    rc.save_raw_chunks_to_json_incrementally([{"id": "a"}], str(json_file))
    rc.save_raw_chunks_to_json_incrementally([{"id": "b"}], str(json_file))
    assert [c["id"] for c in json.loads(json_file.read_text(encoding="utf-8"))] == ["a", "b"]


def test_interrupted_write_keeps_previous_file(monkeypatch, tmp_path):
    """Écriture interrompue : l'ancien fichier reste intact (remplacement atomique), aucun temporaire laissé."""
    rc = _rad_chunk(monkeypatch)
    json_file = tmp_path / "output_chunks.json"
    json_file.write_text('[{"id": "old"}]', encoding="utf-8")

    def broken_dump(*_args, **_kwargs):
        """Simule une panne pendant la sérialisation."""
        raise OSError("disque plein")

    monkeypatch.setattr(rc.json, "dump", broken_dump)
    with pytest.raises(OSError):
        rc.write_json_atomic([{"id": "new"}], str(json_file))
    assert json_file.read_text(encoding="utf-8") == '[{"id": "old"}]'
    assert sorted(os.listdir(tmp_path)) == ["output_chunks.json"]


def test_corrupt_file_is_preserved_aside(monkeypatch, tmp_path, capsys):
    """Fichier JSON illisible : conservé sous ``.corrupt-*`` (plus jamais écrasé en silence)."""
    rc = _rad_chunk(monkeypatch)
    json_file = tmp_path / "output_chunks.json"
    json_file.write_text('[{"id": "a"', encoding="utf-8")
    rc.save_raw_chunks_to_json_incrementally([{"id": "b"}], str(json_file))
    backups = [name for name in os.listdir(tmp_path) if name.startswith("output_chunks.json.corrupt-")]
    assert len(backups) == 1
    assert (tmp_path / backups[0]).read_text(encoding="utf-8") == '[{"id": "a"'
    assert json.loads(json_file.read_text(encoding="utf-8")) == [{"id": "b"}]
    assert "conservé sous" in capsys.readouterr().out
