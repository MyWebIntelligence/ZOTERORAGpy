"""Tests de la déduplication des chunks (Sprint dédup — Lots 0-7).

Couches :
  L1  normalize/hash/content_id/eligibility (déterministe, byte-pinné)
  L2  cascade dedup_filter (Tier 1 in-batch, Tier 2 existence, Tier 3 sémantique, no-op)
  L3  intégration par connecteur (clients mockés ; Pinecone fetch JAMAIS query+$in)
  L0  retours connecteurs unifiés (dict) + parser ancré
  L5  back-fill content-addressed
  L7  cache de recodage déterministe (hit byte-identité, échec non caché, embed cache)

Tout est mocké : aucune base vectorielle ni credential requis en CI.
    pytest tests/test_dedup.py
"""

import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest

# --- sys.path : rendre 'scripts' et la racine ragpy importables (pattern du repo) ---
_THIS = os.path.dirname(os.path.abspath(__file__))
_RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (_RAGPY_ROOT, os.path.join(_RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Évite le prompt interactif input() de rad_chunk si OPENAI_API_KEY est absent en CI.
os.environ.setdefault("OPENAI_API_KEY", "dummy-for-tests")

import rad_dedup as d  # noqa: E402
import rad_recode_cache as rcache  # noqa: E402


# ======================================================================
# L1 — normalize / hash / content_id / eligibility
# ======================================================================
class TestNormalizeHash:
    def test_nfc_equivalence(self):
        import unicodedata
        assert d.content_hash("été") == d.content_hash(unicodedata.normalize("NFD", "été"))

    def test_no_over_normalization_accents(self):
        # Pas d'accent-strip / NFKC / case-fold : FR distinct → hash distinct.
        assert d.content_hash("côté") != d.content_hash("cote")
        assert d.content_hash("Été") != d.content_hash("été")

    def test_html_comment_markers_removed(self):
        # <!-- Page N -->, <!-- Part N/M -->, blancs → collapsés. Livre entier == splitté.
        a = "Le langage est <!-- Page 12 -->un système.   "
        b = "Le langage\test  <!-- Page 512 --><!-- Part 2/3 -->un système."
        assert d.content_hash(a) == d.content_hash(b)

    def test_normalize_byte_pinned(self):
        # Spec byte-pinné de normalize_text_for_hash.
        got = d.normalize_text_for_hash("Le langage est <!-- Page 12 -->un système.   ")
        assert got == "Le langage est un système."

    def test_content_hash_stable_random_doc_id(self):
        # RÉGRESSION RACINE : le hash ne dépend QUE du texte normalisé.
        raw = "Texte académique identique réinséré sous deux doc_id aléatoires distincts."
        assert d.content_hash(raw) == d.content_hash(raw)
        assert len(d.content_hash(raw)) == 64  # sha256 hex

    def test_content_id_deterministic(self):
        h = d.content_hash("x" * 200)
        assert d.content_id(h, 3) == f"{h[:16]}_3"

    def test_eligibility_floor(self):
        assert not d.is_dedup_eligible(d.normalize_text_for_hash("Références"))
        assert not d.is_dedup_eligible(d.normalize_text_for_hash("<!-- OCR ECHOUE -->"))
        assert d.is_dedup_eligible(d.normalize_text_for_hash("x" * 100))

    def test_compute_dedup_fields(self):
        chash, eligible, cid = d.compute_dedup_fields("a" * 100, 5)
        assert eligible is True and cid == d.content_id(chash, 5)
        _, eligible2, _ = d.compute_dedup_fields("ab", 1)
        assert eligible2 is False


# ======================================================================
# L2 — cascade dedup_filter (adapters factices)
# ======================================================================
class _FakeAdapter:
    """Adapter mock stateful : existing()/nearest() depuis des dicts en mémoire."""

    def __init__(self, existing=None, nearest=None, db_name="pinecone", metric="dotproduct"):
        """Mémorise les réponses simulées (existence, voisins), l'identité de la base
        et initialise les compteurs d'appels."""
        self._existing = existing or {}
        self._nearest = nearest or []
        self.db_name = db_name
        self.metric = metric
        self.existing_calls = 0
        self.nearest_calls = 0

    def existing(self, batch):
        """Compte l'appel et renvoie les entrées simulées des ``content_hash`` du lot
        présents dans ``existing``."""
        self.existing_calls += 1
        return {c["content_hash"]: self._existing[c["content_hash"]]
                for c in batch if c["content_hash"] in self._existing}

    def nearest(self, embedding, meta_filter):
        """Compte l'appel et renvoie la liste de voisins simulée."""
        self.nearest_calls += 1
        return self._nearest


def _chunk(cid, chash, idx=1, title="Doc A", text="alpha", eligible=True, **extra):
    """Fabrique un chunk de test (``doc_id`` fixe ``"111"``), complété par ``extra``."""
    base = {"id": cid, "content_hash": chash, "dedup_eligible": eligible,
            "chunk_index": idx, "title": title, "text": text, "doc_id": "111"}
    base.update(extra)
    return base


class TestCascade:
    def test_noop_when_disabled(self):
        cfg = d.DedupConfig.from_env({})  # DEDUP_ENABLED absent → False
        adapter = _FakeAdapter()
        chunks = [_chunk("a", "h1")]
        kept, rej = d.dedup_filter(chunks, adapter, cfg)
        assert kept == chunks and rej == []
        assert adapter.existing_calls == 0  # adapter JAMAIS touché

    def test_tier1_in_batch_duplicate(self):
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1"})
        adapter = _FakeAdapter()
        c1 = _chunk("111_1", "hX", idx=1)
        c2 = _chunk("999_1", "hX", idx=1)  # même hash, doc_id différent
        kept, rej = d.dedup_filter([c1, c2], adapter, cfg, journal_dir=tempfile.mkdtemp())
        assert [c["id"] for c in kept] == ["111_1"]
        assert rej[0]["reason"] == "in_batch_duplicate"
        assert rej[0]["matched_existing"][0]["score"] is None  # exact → jamais 1.0

    def test_tier2_existing_with_title_match_refused(self):
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1"})
        adapter = _FakeAdapter(existing={"hY": [{"id": "old", "content_hash": "hY",
                                                 "title": "Doc A", "chunk_index": 2}]})
        kept, rej = d.dedup_filter([_chunk("222_2", "hY", idx=2)], adapter, cfg,
                                   journal_dir=tempfile.mkdtemp())
        assert kept == [] and rej[0]["reason"] == "existing_hash_match"
        assert rej[0]["matched_existing"][0]["metadata_match"] == ["title"]

    def test_tier2_title_mismatch_kept(self):
        # Hash hit MAIS titre discordant (boilerplate cross-papier) → conservé.
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1"})
        adapter = _FakeAdapter(existing={"hZ": [{"id": "old", "content_hash": "hZ",
                                                 "title": "Autre Doc", "chunk_index": 3}]})
        kept, rej = d.dedup_filter([_chunk("333_3", "hZ", idx=3, title="Doc A")], adapter, cfg,
                                   journal_dir=tempfile.mkdtemp())
        assert len(kept) == 1 and rej == []

    def test_min_chars_floor_never_matched(self):
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1"})
        adapter = _FakeAdapter(existing={"hShort": [{"title": "Doc A"}]})
        # dedup_eligible=False → toujours gardé, adapter jamais consulté pour lui
        kept, rej = d.dedup_filter([_chunk("a", "hShort", eligible=False)], adapter, cfg,
                                   journal_dir=tempfile.mkdtemp())
        assert len(kept) == 1 and rej == [] and adapter.existing_calls == 0

    def test_tier2_batching_call_count(self):
        # 250 survivants, lot=100 → ceil(250/100)=3 appels existing (pas 250).
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1", "DEDUP_QUERY_BATCH": "100"})
        adapter = _FakeAdapter()
        chunks = [_chunk(f"c{i}", f"h{i}", idx=i) for i in range(250)]
        d.dedup_filter(chunks, adapter, cfg, journal_dir=tempfile.mkdtemp())
        assert adapter.existing_calls == 3

    def test_tier3_semantic_refuse_and_keep(self):
        same = "the quick brown fox jumps over the lazy dog repeatedly each morning here ok"
        diff = "completely unrelated content about astrophysics and quantum tunnelling now"
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1", "DEDUP_SEMANTIC": "1"})

        ch = _chunk("cid_5", "hsem", idx=5, title="Doc", text=same, embedding=[0.1] * 8)
        refuse = _FakeAdapter(metric="cosine",
                              nearest=[{"score": 0.99, "meta": {"title": "Doc", "chunk_index": 5}, "text": same}])
        kept, rej = d.dedup_filter([dict(ch)], refuse, cfg, journal_dir=tempfile.mkdtemp())
        assert kept == [] and rej[0]["reason"] == "semantic_match"

        low_jacc = _FakeAdapter(metric="cosine",
                                nearest=[{"score": 0.99, "meta": {"title": "Doc", "chunk_index": 5}, "text": diff}])
        kept2, rej2 = d.dedup_filter([dict(ch)], low_jacc, cfg, journal_dir=tempfile.mkdtemp())
        assert len(kept2) == 1 and rej2 == []  # jamais de refus sur score seul

    def test_tier3_cap_skips_nearest(self):
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1", "DEDUP_SEMANTIC": "1",
                                      "DEDUP_SEMANTIC_MAX_SURVIVORS": "0"})
        adapter = _FakeAdapter(metric="cosine", nearest=[{"score": 0.99, "meta": {}, "text": "x"}])
        ch = _chunk("c", "h", text="y" * 80, embedding=[0.1] * 8)
        kept, _ = d.dedup_filter([ch], adapter, cfg, journal_dir=tempfile.mkdtemp())
        assert len(kept) == 1 and adapter.nearest_calls == 0

    def test_journal_format(self):
        cfg = d.DedupConfig.from_env({"DEDUP_ENABLED": "1"})
        adapter = _FakeAdapter(existing={"hY": [{"id": "old", "content_hash": "hY", "title": "Doc A"}]})
        jdir = tempfile.mkdtemp()
        d.dedup_filter([_chunk("222_2", "hY")], adapter, cfg, journal_dir=jdir, target_desc="idx/ns")
        path = os.path.join(jdir, "dedup_journal.jsonl")
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["reason"] == "existing_hash_match" and rec["target"] == "idx/ns"
        assert rec["refused"]["doc_id"] == "111"  # traçabilité, jamais clé de match
        assert len(rec["refused"]["text_preview"]) <= 200
        summary = json.load(open(os.path.join(jdir, "dedup_summary.json"), encoding="utf-8"))
        assert summary["skipped_total"] == 1 and summary["by_reason"]["existing_hash_match"] == 1


# ======================================================================
# L3 — adapters concrets + intégration (clients mockés)
# ======================================================================
class TestConnectorAdapters:
    def _vmod(self):
        """Importe et renvoie le module ``rad_vectordb``."""
        import rad_vectordb as v
        return v

    def test_pinecone_adapter_uses_fetch_not_query(self):
        v = self._vmod()
        idx = mock.MagicMock()
        cid = d.content_id("hYYYY", 2)
        fake_vec = SimpleNamespace(metadata={"content_hash": "hYYYY", "title": "Doc A", "chunk_index": 2})
        idx.fetch.return_value = SimpleNamespace(vectors={cid: fake_vec})
        ad = v._PineconeDedupAdapter(idx, namespace="ns", metric="dotproduct")
        res = ad.existing([{"id": cid, "content_hash": "hYYYY", "chunk_index": 2, "title": "Doc A"}])
        idx.fetch.assert_called_once()
        assert not idx.query.called  # jamais query+$in
        assert res["hYYYY"][0]["title"] == "Doc A"

    def test_qdrant_adapter_uses_retrieve(self):
        v = self._vmod()
        cli = mock.MagicMock()
        cid = d.content_id("hZZ", 1)
        pt = SimpleNamespace(id=v.generate_uuid(cid), payload={"content_hash": "hZZ", "title": "Doc B", "chunk_index": 1})
        cli.retrieve.return_value = [pt]
        ad = v._QdrantDedupAdapter(cli, "coll")
        res = ad.existing([{"id": cid, "content_hash": "hZZ", "chunk_index": 1, "title": "Doc B"}])
        cli.retrieve.assert_called_once()
        assert res["hZZ"][0]["title"] == "Doc B"

    def _pinecone_mocks(self, MockPC, fetched=None):
        """Configure le client ``Pinecone`` mocké (index ``articles`` existant, métrique
        dotproduct, ``fetch`` renvoyant ``fetched``) ; renvoie ``(pc, index)``."""
        pc = MockPC.return_value
        pc.list_indexes.return_value = SimpleNamespace(indexes=[SimpleNamespace(name="articles")])
        pc.describe_index.return_value = SimpleNamespace(metric="dotproduct")
        index = pc.Index.return_value
        index.fetch.return_value = SimpleNamespace(vectors=fetched or {})
        return pc, index

    def test_insert_to_pinecone_dedup_end_to_end(self):
        v = self._vmod()
        chunks = [
            {"id": d.content_id("h1", 1), "content_hash": "h1", "dedup_eligible": True,
             "chunk_index": 1, "title": "Art", "doc_id": "111", "embedding": [0.1] * 8, "text": "alpha"},
            {"id": d.content_id("h2", 2), "content_hash": "h2", "dedup_eligible": True,
             "chunk_index": 2, "title": "Art", "doc_id": "111", "embedding": [0.2] * 8, "text": "beta"},
        ]
        jf = os.path.join(tempfile.mkdtemp(), "emb.json")
        with open(jf, "w") as f:
            json.dump(chunks, f)
        with mock.patch.object(v, "Pinecone") as MockPC, \
                mock.patch.object(v, "upsert_batch_to_pinecone", return_value=True) as up, \
                mock.patch.dict(os.environ, {"DEDUP_ENABLED": "1"}):
            existing = {d.content_id("h1", 1): SimpleNamespace(
                metadata={"content_hash": "h1", "title": "Art", "chunk_index": 1})}
            _, index = self._pinecone_mocks(MockPC, fetched=existing)
            res = v.insert_to_pinecone(jf, index_name="articles", pinecone_api_key="k", namespace="ns")
        assert res["skipped_count"] == 1 and res["status"] == "success"
        assert index.fetch.called and not index.query.called
        upserted = [vv["id"] for call in up.call_args_list for vv in call.args[1]]
        assert d.content_id("h2", 2) in upserted and d.content_id("h1", 1) not in upserted
        assert os.path.exists(res["journal_path"])

    def test_insert_to_pinecone_noop_when_disabled(self):
        v = self._vmod()
        chunks = [{"id": "x_1", "content_hash": "h1", "dedup_eligible": True, "chunk_index": 1,
                   "doc_id": "111", "embedding": [0.1] * 8, "text": "alpha"}]
        jf = os.path.join(tempfile.mkdtemp(), "emb.json")
        with open(jf, "w") as f:
            json.dump(chunks, f)
        with mock.patch.object(v, "Pinecone") as MockPC, \
                mock.patch.object(v, "upsert_batch_to_pinecone", return_value=True), \
                mock.patch.dict(os.environ, {"DEDUP_ENABLED": "0"}):
            _, index = self._pinecone_mocks(MockPC)
            res = v.insert_to_pinecone(jf, index_name="articles", pinecone_api_key="k")
        assert not index.fetch.called  # adapter jamais touché
        assert "skipped_count" not in res


# ======================================================================
# L0 — retours connecteurs unifiés (dict) + parser ancré
# ======================================================================
class TestLot0:
    def test_weaviate_qdrant_return_dict_on_missing_file(self):
        import rad_vectordb as v
        rw = v.insert_to_weaviate_hybrid("/nope/missing.json", url="http://x", api_key="k")
        rq = v.insert_to_qdrant("/nope/missing.json", "coll", qdrant_url="http://x")
        for r in (rw, rq):
            assert isinstance(r, dict) and r["status"] == "error" and r["inserted_count"] == 0

    def test_vectordb_result_helper(self):
        import rad_vectordb as v
        full = v._vectordb_result("success", "m", inserted_count=5, skipped_count=3, journal_path="/j")
        assert full == {"status": "success", "message": "m", "inserted_count": 5,
                        "skipped_count": 3, "journal_path": "/j"}
        assert "skipped_count" not in v._vectordb_result("success", "m", inserted_count=5)

    def test_anchored_parser_resists_leading_digits(self):
        # T-LOT0-03 : un nombre (date) avant le compte ne corrompt plus l'extraction.
        stdout = ("=== Result ===\nStatus: success\n"
                  "Message: 2026 chunks traités. Total: 99\n"
                  "Inserted: 42\nSkipped (dedup): 7\nDedup journal: /u/42/dedup_journal.jsonl\n")
        assert re.search(r'^Inserted:\s*(\d+)', stdout, re.MULTILINE).group(1) == "42"
        assert re.search(r'^Skipped \(dedup\):\s*(\d+)', stdout, re.MULTILINE).group(1) == "7"
        assert re.search(r'^Dedup journal:\s*(\S+)', stdout, re.MULTILINE).group(1).endswith(".jsonl")


# ======================================================================
# L5 — back-fill content-addressed
# ======================================================================
class TestBackfill:
    def test_backfill_adds_fields_and_preserves_existing(self):
        chunks = [
            {"id": "rand_1", "doc_id": "111", "chunk_index": 1, "text": "x" * 100},
            {"id": "rand_2", "doc_id": "111", "chunk_index": 2, "text": "Réf"},
            {"id": "keep_3", "content_hash": "PRE", "chunk_index": 3, "text": "y" * 100},
        ]
        n = d.backfill_chunk_dedup_fields(chunks)
        assert n == 2
        assert chunks[0]["dedup_eligible"] is True
        assert chunks[0]["id"] == d.content_id(chunks[0]["content_hash"], 1)
        assert chunks[1]["dedup_eligible"] is False and chunks[1]["id"] == "rand_2"
        assert chunks[2]["content_hash"] == "PRE" and chunks[2]["id"] == "keep_3"


# ======================================================================
# L7 — cache de recodage déterministe
# ======================================================================
def _resp(content, finish="stop"):
    """Réponse chat-completion factice (``content`` + ``finish_reason``)."""
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish)])


@pytest.fixture
def recode_env(tmp_path, monkeypatch):
    """Active le cache de recodage sur une base SQLite temporaire et renvoie le
    module ``rad_chunk``."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-for-tests")
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", str(tmp_path / "recode.sqlite"))
    import rad_chunk as rc
    return rc


class TestRecodeCache:
    def test_cache_hit_byte_identity_zero_calls(self, recode_env):
        rc = recode_env
        raw = ["texte ocr brut numero un avec suffisamment de contenu pour etre eligible"]
        m1 = mock.MagicMock(return_value=_resp("CLEAN"))
        with mock.patch.object(rc.client.chat.completions, "create", m1):
            t1, s1 = rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
        assert t1 == ["CLEAN"] and s1 == ["recoded"] and m1.call_count == 1
        m2 = mock.MagicMock(return_value=_resp("SHOULD_NOT_APPEAR"))
        with mock.patch.object(rc.client.chat.completions, "create", m2):
            t2, s2 = rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
        assert t2 == ["CLEAN"] and s2 == ["cached"] and m2.call_count == 0

    def test_prompt_edit_invalidates(self, recode_env):
        rc = recode_env
        raw = ["un autre texte ocr brut assez long pour le cache de recodage gpt ici"]
        with mock.patch.object(rc.client.chat.completions, "create", mock.MagicMock(return_value=_resp("A"))):
            rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
        m = mock.MagicMock(return_value=_resp("B"))
        with mock.patch.object(rc.client.chat.completions, "create", m):
            t, _ = rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS + " EDIT", model="gpt-4o-mini")
        assert m.call_count == 1 and t == ["B"]

    def test_failure_not_cached(self, recode_env):
        rc = recode_env
        raw = ["chunk qui echoue au recodage mais assez long pour etre eligible au cache"]
        with mock.patch.object(rc.client.chat.completions, "create",
                               mock.MagicMock(side_effect=RuntimeError("boom"))), \
                mock.patch.object(rc.time, "sleep", lambda *_a: None):
            t, s = rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
        assert s == ["fallback_raw"] and t == raw
        m_ok = mock.MagicMock(return_value=_resp("NOWCLEAN"))
        with mock.patch.object(rc.client.chat.completions, "create", m_ok):
            t2, s2 = rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
        assert s2 == ["recoded"] and m_ok.call_count == 1  # ré-appel (pas de gel)

    def test_truncation_not_cached(self, recode_env):
        rc = recode_env
        raw = ["chunk tronque assez long pour etre eligible au cache de recodage maintenant"]
        with mock.patch.object(rc.client.chat.completions, "create",
                               mock.MagicMock(return_value=_resp("partial", "length"))), \
                mock.patch.object(rc.time, "sleep", lambda *_a: None):
            t, s = rc.recode_batch_cached(raw, instructions=rc._RECODE_INSTRUCTIONS, model="gpt-4o-mini")
        assert s == ["fallback_truncated"] and t == raw

    def test_embed_cache(self, recode_env, monkeypatch):
        rc = recode_env
        monkeypatch.setenv("RECODE_EMBED_CACHE_ENABLED", "1")
        cfg = rcache.RecodeConfig.from_env()
        calls = {"n": 0}

        def fake_embed(texts, model="text-embedding-3-large"):
            calls["n"] += 1
            return [[0.1, 0.2, 0.3] for _ in texts]

        with mock.patch.object(rc, "get_embeddings_batch", side_effect=fake_embed):
            e1 = rc._embed_with_cache(["meme texte a embedder ici"], cfg)
            e2 = rc._embed_with_cache(["meme texte a embedder ici"], cfg)
        assert e1 == e2 == [[0.1, 0.2, 0.3]] and calls["n"] == 1


# ======================================================================
# Config
# ======================================================================
class TestConfig:
    def test_dedup_config_from_env_defaults(self):
        cfg = d.DedupConfig.from_env({})
        assert cfg.enabled is False and cfg.semantic is False
        assert cfg.min_chars == 64 and cfg.query_batch == 100 and cfg.meta_fields == ("title",)

    def test_dedup_config_from_env_overrides(self):
        cfg = d.DedupConfig.from_env({
            "DEDUP_ENABLED": "1", "DEDUP_SEMANTIC": "true", "DEDUP_MIN_CHARS": "32",
            "DEDUP_META_FIELDS": "title, authors", "DEDUP_SIM_THRESHOLD": "0.95",
        })
        assert cfg.enabled and cfg.semantic and cfg.min_chars == 32
        assert cfg.meta_fields == ("title", "authors") and cfg.sim_threshold == 0.95

    def test_recode_config_defaults_off(self):
        cfg = rcache.RecodeConfig.from_env({})
        assert cfg.cache_enabled is False and cfg.harden_enabled is False
        assert cfg.embed_cache_enabled is False and cfg.model == "gpt-4o-mini-2024-07-18"
        assert cfg.prefer_openai is True
