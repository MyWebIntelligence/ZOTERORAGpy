"""Benchmark collections Albert (lot 5) — invariants SANS RÉSEAU (durs) + débit (info).

L'envoi ``insert_to_albert`` tourne contre ``FakeAlbert`` (``httpx.MockTransport``,
temps virtuel du client : aucune attente réelle). On ne mesure donc pas une
latence réseau ; les seuls gates DURS sont des comptages déterministes :

  - POST de chunks == ceil(n / 64) par document (jamais plus de 64 par POST) ;
  - Tier 2 de la dédup : 0 appel HTTP après le préchargement (index en mémoire),
    et ``existing()`` bien appelé ;
  - préchargement borné : ``GET /v1/documents/{id}/chunks`` == n // 100 + 1 par
    document listé (pages de 100) ;
  - 0 appel à ``/v1/search`` ni à ``/v1/embeddings`` (embeddings côté serveur,
    Tier 3 sauté même avec ``DEDUP_SEMANTIC=1``), aucun vecteur dans les corps ;
  - relance identique : 0 écriture.

Usage :
    python tests/load/bench_albert.py            # N = 1 000 et 10 000 chunks
    pytest tests/load/bench_albert.py -q -p no:cacheprovider
"""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import rad_dedup  # noqa: E402
from scripts import rad_vectordb as rv  # noqa: E402
from scripts.rad_albert import collections as col  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert  # noqa: E402

CHUNK_POSTS = "/v1/documents/*/chunks"
CHUNKS_PER_POST = 64
LIST_PAGE = 100
COLLECTION = "ragpy-bench-corpus"
DOC_SIZES = (1, 63, 64, 65, 128, 130, 200)


def _no_sleep(_seconds=0.0):
    """Sommeil neutralisé (aucune attente réelle)."""
    return None


@contextlib.contextmanager
def _env(values):
    """Pose temporairement des variables d'environnement (``None`` = retirée), puis restaure.

    Les variables ``DEDUP_*`` et ``RECODE_*`` absentes de ``values`` sont
    retirées pendant le bloc (mesure indépendante du ``.env``).
    """
    saved = {}
    names = set(values) | {n for n in os.environ if n.startswith(("DEDUP_", "RECODE_"))}
    for name in names:
        saved[name] = os.environ.get(name)
        value = values.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _spy_tier2(fake):
    """Remplace l'adaptateur Albert par une sous-classe qui compte ``existing()`` et ses appels HTTP.

    Renvoie un dictionnaire de compteurs : ``existing_calls``, ``http_in_existing``
    (requêtes reçues par le faux serveur pendant ``existing``) et ``preloads``.
    """
    stats = {"existing_calls": 0, "http_in_existing": 0, "preloads": 0}
    original = rv._AlbertDedupAdapter

    class _CountingAdapter(original):
        """Adaptateur Albert instrumenté (préchargement et Tier 2 comptés)."""

        def preload(self, *args, **kwargs):
            """Préchargement réel, compté."""
            stats["preloads"] += 1
            return super().preload(*args, **kwargs)

        def existing(self, chunks_batch):
            """Tier 2 réel ; compte l'appel et les requêtes HTTP émises pendant l'appel."""
            before = len(fake.calls)
            try:
                return super().existing(chunks_batch)
            finally:
                stats["existing_calls"] += 1
                stats["http_in_existing"] += len(fake.calls) - before

    rv._AlbertDedupAdapter = _CountingAdapter
    try:
        yield stats
    finally:
        rv._AlbertDedupAdapter = original


def _make_chunks(sizes, *, prefix="ITEM", same_text_as=None):
    """Chunks de plusieurs documents (un document par taille), champs de dédup compris.

    Args:
        sizes: nombre de chunks de chaque document.
        prefix: préfixe de l'``itemKey`` (des préfixes différents donnent des
            documents Albert différents).
        same_text_as: préfixe dont on reprend les textes (doublons inter-documents).
    """
    chunks = []
    for d, size in enumerate(sizes):
        item_key = f"{prefix}{d:04d}"
        text_key = f"{same_text_as}{d:04d}" if same_text_as else item_key
        title = f"Document de banc {text_key}"
        for i in range(size):
            text = (f"{title} — passage {i} : texte de charge pour mesurer l'envoi vers une "
                    f"collection Albert sans aucun appel réseau réel ({text_key}-{i}).")
            chash, eligible, content_id = rad_dedup.compute_dedup_fields(text, i)
            chunks.append({
                "id": content_id, "doc_id": "424242424242", "chunk_index": i, "total_chunks": size,
                "text": text, "title": title, "itemKey": item_key, "filename": f"{item_key}.pdf",
                "content_hash": chash, "dedup_eligible": eligible,
                "embedding": [0.125] * 8, "sparse_embedding": {"indices": [1], "values": [1.0]},
            })
    return chunks


def _push(fake, directory, chunks):
    """Écrit ``chunks`` puis appelle ``insert_to_albert`` (sortie standard absorbée)."""
    path = os.path.join(directory, "output_chunks_with_embeddings_sparse.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return rv.insert_to_albert(
            path, collection_name=COLLECTION, albert_api_key=FAKE_ALBERT_KEY, create_collection=True,
            ack_retention=True, transport=fake.transport, sleep=_no_sleep,
        )


def _forbidden_calls(fake):
    """Appels interdits sur le chemin d'envoi : recherche et embeddings explicites."""
    return [c for c in fake.calls if c.path in ("/v1/search", "/v1/embeddings")]


def run_push(sizes=DOC_SIZES):
    """Envoie un document par taille ; renvoie les POST par document et les compteurs."""
    albert_limiter.reset_limiters()
    fake = FakeAlbert()
    with _env({"ALBERT_ENABLED": "1"}), tempfile.TemporaryDirectory() as tmp:
        chunks = _make_chunks(sizes)
        t0 = time.perf_counter()
        result = _push(fake, tmp, chunks)
        elapsed = time.perf_counter() - t0
        posts_by_doc = {}
        sizes_by_doc = {}
        for call in fake.calls_to("POST", CHUNK_POSTS):
            did = int(call.path.split("/")[3])
            posts_by_doc[did] = posts_by_doc.get(did, 0) + 1
            sizes_by_doc.setdefault(did, []).append(len(call.json["chunks"]))
        names = {d["id"]: d["name"] for d in fake.documents.values()}
        expected = {col.document_name_for({"itemKey": f"ITEM{d:04d}", "filename": f"ITEM{d:04d}.pdf"}): size
                    for d, size in enumerate(sizes)}
        vectors_sent = any(b"embedding" in (c.content or b"") for c in fake.calls)
        forbidden = len(_forbidden_calls(fake))
        fake.reset_calls()
        rerun = _push(fake, tmp, chunks)
        rerun_writes = sum(1 for c in fake.calls if c.method in ("POST", "PATCH", "DELETE"))
    return {
        "result": result,
        "elapsed_s": elapsed,
        "posts_by_name": {names[did]: count for did, count in posts_by_doc.items()},
        "max_chunks_per_post": max((n for v in sizes_by_doc.values() for n in v), default=0),
        "expected_sizes_by_name": expected,
        "vectors_sent": vectors_sent,
        "forbidden_calls": forbidden,
        "rerun": rerun,
        "rerun_writes": rerun_writes,
    }


def run_tier2(sizes=(150, 40, 100), query_batch=25):
    """Relance avec dédup active sur des doublons inter-documents ; compte le HTTP du Tier 2.

    1er envoi : documents ``SRC*``. 2e envoi : documents ``DUP*`` aux mêmes
    textes et titres (refusés par le Tier 2) plus un document neuf.
    """
    albert_limiter.reset_limiters()
    fake = FakeAlbert()
    env = {"ALBERT_ENABLED": "1", "DEDUP_ENABLED": "1", "DEDUP_SEMANTIC": "1",
           "DEDUP_QUERY_BATCH": str(query_batch)}
    with _env(env), tempfile.TemporaryDirectory() as tmp:
        first_dir = os.path.join(tmp, "first")
        second_dir = os.path.join(tmp, "second")
        os.makedirs(first_dir)
        os.makedirs(second_dir)
        first = _push(fake, first_dir, _make_chunks(sizes, prefix="SRC"))
        duplicates = _make_chunks(sizes, prefix="DUP", same_text_as="SRC")
        fresh = _make_chunks((7,), prefix="NEW")
        fake.reset_calls()
        with _spy_tier2(fake) as stats:
            second = _push(fake, second_dir, duplicates + fresh)
        chunk_listings = len(fake.calls_to("GET", CHUNK_POSTS))
        forbidden = len(_forbidden_calls(fake))
        new_posts = len(fake.calls_to("POST", CHUNK_POSTS))
    total = sum(sizes) + 7
    return {
        "first": first,
        "second": second,
        "stats": stats,
        "chunk_listings": chunk_listings,
        "expected_chunk_listings": sum(n // LIST_PAGE + 1 for n in sizes),
        "expected_existing_calls": math.ceil(total / query_batch),
        "forbidden_calls": forbidden,
        "new_posts": new_posts,
        "duplicates": sum(sizes),
    }


# ---------------------------------------------------------------------------
# Invariants (pytest)
# ---------------------------------------------------------------------------
def test_invariant_chunk_posts_ceil_64_per_document():
    out = run_push()
    assert out["result"]["status"] == "success"
    assert out["result"]["inserted_count"] == sum(DOC_SIZES)
    assert set(out["posts_by_name"]) == set(out["expected_sizes_by_name"])
    for name, size in out["expected_sizes_by_name"].items():
        assert out["posts_by_name"][name] == math.ceil(size / CHUNKS_PER_POST), name
    assert out["max_chunks_per_post"] <= CHUNKS_PER_POST
    assert out["rerun"]["inserted_count"] == 0 and out["rerun"]["existing_count"] == sum(DOC_SIZES)
    assert out["rerun_writes"] == 0


def test_invariant_tier2_zero_http_after_preload():
    out = run_tier2()
    stats = out["stats"]
    assert stats["preloads"] == 1
    assert stats["existing_calls"] == out["expected_existing_calls"]
    assert stats["http_in_existing"] == 0
    assert out["chunk_listings"] == out["expected_chunk_listings"]
    assert out["second"]["skipped_count"] == out["duplicates"]
    assert out["second"]["inserted_count"] == 7
    assert out["new_posts"] == 1


def test_invariant_no_search_no_embeddings_calls():
    push = run_push((65, 3))
    tier2 = run_tier2((70,), query_batch=100)
    assert push["forbidden_calls"] == 0 and tier2["forbidden_calls"] == 0
    assert push["vectors_sent"] is False


# ---------------------------------------------------------------------------
# Banc manuel
# ---------------------------------------------------------------------------
def _bench_cli():
    """Affiche les compteurs et le débit pour N = 1 000 et 10 000 chunks (documents de 500)."""
    for n in (1000, 10000):
        sizes = tuple([500] * (n // 500))
        out = run_push(sizes)
        posts = sum(out["posts_by_name"].values())
        rate = n / out["elapsed_s"] if out["elapsed_s"] else float("inf")
        print(f"N={n}: {len(sizes)} documents, {posts} POST de chunks "
              f"(attendu {sum(math.ceil(s / CHUNKS_PER_POST) for s in sizes)}), "
              f"{out['elapsed_s']:.2f} s ({rate:.0f} chunks/s, faux serveur en mémoire), "
              f"relance : {out['rerun_writes']} écriture(s)")
    tier2 = run_tier2((500, 500), query_batch=100)
    print(f"Tier 2 : {tier2['stats']['existing_calls']} appel(s) existing(), "
          f"{tier2['stats']['http_in_existing']} requête(s) HTTP pendant le Tier 2, "
          f"{tier2['forbidden_calls']} appel(s) search/embeddings")


if __name__ == "__main__":
    _bench_cli()
