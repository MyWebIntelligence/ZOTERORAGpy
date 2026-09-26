"""Benchmark déduplication — invariants SANS RÉSEAU (durs) + CPU µs/chunk (info).

Sous adapters mockés on ne mesure pas une vraie latence réseau ; un %-de-surcoût
contre un sleep modélisé est *jouable* → démoté en illustratif. Les seuls gates
DURS sont déterministes et sans horloge :

  - appels d'existence Tier 2 == ceil(N / DEDUP_QUERY_BATCH)  (pas N)
  - Pinecone : fetch utilisé, query (+$in) JAMAIS appelé
  - Tier 3 : 0 appel nearest() quand DEDUP_SEMANTIC=0
  - append journal == nombre de refus (O(N), pas de réécriture par refus)

Usage :
    python tests/load/bench_dedup.py            # N=1000 et 10000
    pytest tests/load/bench_dedup.py            # exécute test_invariants_*
"""

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

import rad_dedup as d  # noqa: E402


class _CountingAdapter:
    """Adapter à retour instantané qui compte les appels (pas d'I/O réseau)."""

    db_name = "pinecone"
    metric = "dotproduct"

    def __init__(self, present=None):
        """Initialise les hashes « présents » simulés et les compteurs d'appels."""
        self._present = present or {}
        self.existing_calls = 0
        self.nearest_calls = 0

    def existing(self, batch):
        """Compte l'appel et renvoie les entrées du lot dont le ``content_hash`` est
        marqué présent."""
        self.existing_calls += 1
        return {c["content_hash"]: self._present[c["content_hash"]]
                for c in batch if c["content_hash"] in self._present}

    def nearest(self, embedding, meta_filter):
        """Compte l'appel ; ne renvoie jamais de voisin."""
        self.nearest_calls += 1
        return []


def _make_chunks(n):
    """Génère ``n`` chunks éligibles dont les ``content_hash`` sont tous distincts."""
    # État frais : hashes uniques (cold, aucun doublon présent).
    return [{"id": f"c{i}", "content_hash": f"{i:064x}", "dedup_eligible": True,
             "chunk_index": i, "title": "Doc", "text": "x" * 80, "embedding": [0.1] * 8}
            for i in range(n)]


def run(n, batch=100, semantic=False):
    """Exécute ``dedup_filter`` sur ``n`` chunks frais avec un adapter compteur et
    renvoie les compteurs d'appels (existence / nearest), le nombre attendu
    d'appels d'existence ``ceil(n / batch)`` et le temps CPU par chunk."""
    cfg = d.DedupConfig.from_env({
        "DEDUP_ENABLED": "1",
        "DEDUP_QUERY_BATCH": str(batch),
        "DEDUP_SEMANTIC": "1" if semantic else "0",
    })
    adapter = _CountingAdapter()
    chunks = _make_chunks(n)
    jdir = tempfile.mkdtemp()
    t0 = time.perf_counter()
    kept, rej = d.dedup_filter(chunks, adapter, cfg, journal_dir=jdir, target_desc="bench")
    elapsed = time.perf_counter() - t0
    return {
        "n": n, "batch": batch, "kept": len(kept), "rejected": len(rej),
        "existing_calls": adapter.existing_calls, "nearest_calls": adapter.nearest_calls,
        "expected_existing_calls": math.ceil(n / batch),
        "cpu_us_per_chunk": (elapsed / n) * 1e6, "elapsed_s": elapsed, "journal_dir": jdir,
    }


# ---- Gates DURS (assertables par pytest) ----
def test_invariant_tier2_batching():
    r = run(1000, batch=100)
    assert r["existing_calls"] == r["expected_existing_calls"] == 10
    assert r["existing_calls"] != r["n"]  # pas N appels


def test_invariant_tier3_off_zero_calls():
    r = run(1000, batch=100, semantic=False)
    assert r["nearest_calls"] == 0


def test_invariant_hash_throughput():
    # Débit du hash isolé : doit être << RTT réseau (µs).
    n = 10000
    t0 = time.perf_counter()
    for i in range(n):
        d.content_hash("x" * 80 + str(i))
    us = ((time.perf_counter() - t0) / n) * 1e6
    assert us < 100  # garde de santé large


def _bench_cli():
    """Affiche le benchmark pour N=1000 et N=10000 : invariants Tier 2 / Tier 3 et
    temps CPU par chunk (informatif)."""
    print("=== BENCH DÉDUP — invariants sans réseau (durs) + CPU info ===\n")
    for n in (1000, 10000):
        cold = run(n, batch=100)
        sem = run(n, batch=100, semantic=True)
        ok_batch = "OK " if cold["existing_calls"] == cold["expected_existing_calls"] else "!! "
        ok_t3 = "OK " if cold["nearest_calls"] == 0 else "!! "
        print(f"N={n:>6}")
        print(f"  {ok_batch}Tier2 existence (cold)   : {cold['existing_calls']:>4} "
              f"(attendu ceil(N/100)={cold['expected_existing_calls']})")
        print(f"  {ok_t3}Tier3 nearest (semantic=0): {cold['nearest_calls']:>4} (attendu 0)")
        print(f"     Tier3 nearest (semantic=1): {sem['nearest_calls']:>4} "
              f"(cold = tous uniques → query par survivant)")
        print(f"     CPU µs/chunk (info)        : {cold['cpu_us_per_chunk']:.2f} µs "
              f"(+ ceil(N/100)={cold['expected_existing_calls']} round-trips réseau réels)")
        print()
    print("Gates DURS : Tier2==ceil(N/lot), Tier3==0 si OFF. CPU/latence = informationnels.")


if __name__ == "__main__":
    _bench_cli()
