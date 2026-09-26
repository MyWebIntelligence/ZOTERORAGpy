#!/usr/bin/env python3
"""Recreate a Pinecone index and bulk re-upload session corpora into it.

This tool exists because RAGpy has no index-creation helper (indexes are normally
pre-created by hand) and ``rad_vectordb.py`` only uploads one file per invocation.
It is meant for *rebuilding* an index whose configuration must change — most
notably switching an index to ``metric="dotproduct"`` so it can store the hybrid
(dense + sparse) vectors produced by the pipeline. Pinecone's metric is immutable,
so a metric change requires delete + recreate, which is **destructive**: every
vector currently in the index is lost and must be re-uploaded from source JSON.

Workflow:
    1. (optional) ``--recreate``: delete the index if present, then create it with
       the requested dimension / metric / serverless spec, waiting until ready.
    2. Discover the input files: every
       ``<uploads-dir>/*/<filename>`` (with ``--all``) or an explicit
       ``--sessions`` list.
    3. Upload each file sequentially via ``insert_to_pinecone`` (imported from
       ``rad_vectordb``), which automatically includes sparse vectors when the
       index metric is ``dotproduct``.
    4. Print a per-session + grand-total summary and exit non-zero on any failure.

Example (inside the container, PINECONE_API_KEY comes from .env):
    python3 scripts/rebuild_pinecone_index.py \
        --index articles --recreate \
        --uploads-dir /app/uploads --all
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# When run as ``python3 scripts/rebuild_pinecone_index.py`` the script directory
# (scripts/) is on sys.path[0], so a plain sibling import works. Add it explicitly
# too so the tool also works when imported from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinecone import Pinecone, ServerlessSpec  # noqa: E402
from rad_vectordb import insert_to_pinecone  # noqa: E402
from rad_vectordb import _missing_vectors_error  # noqa: E402  (même garde que l'envoi)
import rad_dedup  # noqa: E402  (Lot 5 back-fill)
from rad_providers import check_uniform_space, target_mismatch_message  # noqa: E402  (espaces vectoriels)

DEFAULT_FILENAME = "output_chunks_with_embeddings_sparse.json"


def check_input_dimensions(paths: list[str], target_dimension) -> str | None:
    """Pre-flight: every embeddings file must hold one vector space matching the index.

    Checked locally, before any destructive step (``recreate_index``) or upload:
    a file mixing spaces (``check_uniform_space``), a file of a non-default
    space (e.g. Albert) left incomplete by its dense phase (share of missing
    vectors above ``ALBERT_EMBED_MAX_MISSING_RATIO``, the refusal
    ``insert_to_pinecone`` would otherwise raise only after the index was
    recreated), files of different spaces (e.g. OpenAI 3072-d and Albert bge-m3
    1024-d) or a space whose dimension differs from ``target_dimension`` (when
    it is an int) are refused. Files without usable vectors are otherwise
    ignored; an unreadable file is reported and left to the upload step (as
    before).

    Args:
        paths: embeddings files to upload.
        target_dimension: dimension of the (re)created or existing index.

    Returns:
        A French error message, or ``None`` when every file is compatible.
    """
    import json

    spaces = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as exc:
            print(f"Avertissement: contrôle de dimension impossible pour {path} ({exc}).")
            continue
        try:
            space = check_uniform_space(chunks)
        except ValueError as exc:
            return f"{path} : {exc}"
        missing = _missing_vectors_error(chunks, space)
        if missing:
            return f"{path} : {missing}"
        if space is None:
            continue
        mismatch = target_mismatch_message(target_dimension, space)
        if mismatch:
            return f"{path} : {mismatch}"
        spaces.setdefault((space.provider, space.model, space.dim), []).append(path)
    if len(spaces) > 1:
        detail = " ; ".join(
            f"{provider}/{model} ({dim} dimensions) : {len(files)} fichier(s)"
            for (provider, model, dim), files in spaces.items()
        )
        return (f"espaces d'embeddings différents entre les corpus ({detail}) : "
                "un index ne contient qu'un seul espace vectoriel.")
    return None


def backfill_inputs(paths: list[str]) -> list[str]:
    """Lot 5 — réécrit chaque JSON d'embeddings vers un sibling ``*.dedup.json`` doté
    de ``content_hash`` / ``dedup_eligible`` / id adressé par contenu (hash sur le
    texte STOCKÉ). Rend des corpus pré-dédup déduplicables lors du rebuild (idempotence
    + cross-session). Renvoie la liste des chemins à uploader (les temporaires)."""
    import json

    out = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            n = rad_dedup.backfill_chunk_dedup_fields(chunks)
            tmp = path + ".dedup.json"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False)
            print(f"  back-fill: {n}/{len(chunks)} chunk(s) dotés d'un content_hash → {os.path.basename(tmp)}")
            out.append(tmp)
        except Exception as exc:
            print(f"  back-fill ÉCHEC pour {path} ({exc}); upload du fichier original.")
            out.append(path)
    return out


def _index_exists(pc: Pinecone, name: str) -> bool:
    """Return True if an index named ``name`` exists in the account."""
    try:
        return name in [idx.name for idx in pc.list_indexes().indexes]
    except Exception as exc:  # pragma: no cover - network/SDK shape guard
        print(f"Avertissement: list_indexes() a échoué ({exc}).")
        return False


def _wait_until_absent(pc: Pinecone, name: str, timeout: int = 120, interval: float = 2.0) -> None:
    """Block until the named index disappears from list_indexes (post-delete)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _index_exists(pc, name):
            return
        time.sleep(interval)
    raise TimeoutError(f"L'index '{name}' n'a pas disparu après {timeout}s.")


def _wait_until_ready(pc: Pinecone, name: str, timeout: int = 300, interval: float = 2.0) -> None:
    """Block until the named index reports status.ready == True (post-create)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pc.describe_index(name).status.get("ready"):
                return
        except Exception:
            pass  # index may not be queryable for a beat after creation
        time.sleep(interval)
    raise TimeoutError(f"L'index '{name}' n'est pas devenu 'ready' après {timeout}s.")


def recreate_index(pc: Pinecone, name: str, dimension: int, metric: str,
                   cloud: str, region: str) -> None:
    """Delete the index if it exists, then create it with the given spec.

    DESTRUCTIVE: all vectors in an existing index named ``name`` are permanently
    removed before recreation.
    """
    if _index_exists(pc, name):
        print(f"Suppression de l'index existant '{name}'…")
        pc.delete_index(name)
        _wait_until_absent(pc, name)
        print(f"  -> index '{name}' supprimé.")
    else:
        print(f"Aucun index '{name}' existant à supprimer.")

    print(f"Création de l'index '{name}' (dim={dimension}, metric={metric}, "
          f"{cloud}/{region})…")
    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    _wait_until_ready(pc, name)
    print(f"  -> index '{name}' prêt.")


def discover_inputs(uploads_dir: str, filename: str) -> list[str]:
    """Return sorted paths of every <uploads_dir>/*/<filename> embeddings file."""
    pattern = os.path.join(uploads_dir, "*", filename)
    return sorted(glob.glob(pattern))


def resolve_sessions(sessions: list[str], uploads_dir: str, filename: str) -> list[str]:
    """Map explicit --sessions entries (folder name or path) to embeddings files."""
    resolved = []
    for s in sessions:
        if os.path.isfile(s):
            resolved.append(s)
            continue
        candidate = s if os.path.isdir(s) else os.path.join(uploads_dir, s)
        path = os.path.join(candidate, filename)
        if os.path.isfile(path):
            resolved.append(path)
        else:
            print(f"Avertissement: aucun '{filename}' trouvé pour la session '{s}' "
                  f"(cherché: {path}). Ignoré.")
    return resolved


def main() -> int:
    """CLI entry point: resolve the embeddings files (--all or --sessions), check
    their vector space against the index dimension (pre-flight, before any
    deletion), optionally recreate the index, optionally back-fill content
    hashes, upload each corpus with insert_to_pinecone and print a summary.

    Returns:
        int: 0 if every corpus was uploaded, 1 if any upload failed, 2 on a
        configuration error (missing key, index, arguments or input files) or a
        vector-space mismatch (mixed spaces, wrong dimension) — in that case the
        index is never recreated.
    """
    parser = argparse.ArgumentParser(
        description="Recreate a Pinecone index and bulk re-upload session corpora."
    )
    parser.add_argument("--index", default="articles", help="Index name (default: articles)")
    parser.add_argument("--dimension", type=int, default=3072,
                        help="Vector dimension (default: 3072, text-embedding-3-large)")
    parser.add_argument("--metric", default="dotproduct",
                        choices=["dotproduct", "cosine", "euclidean"],
                        help="Index metric (default: dotproduct — required for sparse/hybrid)")
    parser.add_argument("--cloud", default="aws", help="Serverless cloud (default: aws)")
    parser.add_argument("--region", default="us-east-1", help="Serverless region (default: us-east-1)")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete (if present) and recreate the index before uploading. "
                             "DESTRUCTIVE: drops all existing vectors.")
    parser.add_argument("--uploads-dir", default="uploads",
                        help="Base directory holding session folders (default: uploads)")
    parser.add_argument("--all", action="store_true",
                        help="Upload every <uploads-dir>/*/<filename> found")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Explicit session folders/paths to upload (alternative to --all)")
    parser.add_argument("--filename", default=DEFAULT_FILENAME,
                        help=f"Embeddings filename within each session (default: {DEFAULT_FILENAME})")
    parser.add_argument("--namespace", default=None, help="Pinecone namespace (default: index default)")
    parser.add_argument("--backfill-hash", action="store_true",
                        help="(Lot 5) Doter chaque corpus pré-dédup d'un content_hash + id "
                             "adressé par contenu (hash sur le texte stocké) avant upload, pour "
                             "une dédup cross-session lors du rebuild. Requiert DEDUP_ENABLED=1.")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERREUR: PINECONE_API_KEY non défini dans l'environnement.")
        return 2

    if not args.all and not args.sessions:
        print("ERREUR: préciser --all ou --sessions <...>.")
        return 2

    pc = Pinecone(api_key=api_key)

    if args.recreate:
        target_dimension = args.dimension
    else:
        if not _index_exists(pc, args.index):
            print(f"ERREUR: l'index '{args.index}' n'existe pas et --recreate n'est pas fourni.")
            return 2
        description = pc.describe_index(args.index)
        metric = description.metric
        target_dimension = getattr(description, "dimension", None)
        print(f"Index '{args.index}' existant (metric={metric}). Upload sans recréation.")
        if metric != "dotproduct":
            print(f"  Attention: metric={metric} (non-dotproduct) → les vecteurs sparse "
                  f"seront omis par insert_to_pinecone (upsert dense uniquement).")

    # Resolve the list of embeddings files to upload (before any deletion).
    if args.all:
        inputs = discover_inputs(args.uploads_dir, args.filename)
    else:
        inputs = resolve_sessions(args.sessions, args.uploads_dir, args.filename)

    if not inputs:
        print(f"ERREUR: aucun fichier d'embeddings trouvé (uploads-dir={args.uploads_dir}, "
              f"filename={args.filename}).")
        return 2

    # Pre-flight des espaces vectoriels : un seul espace, de la dimension de
    # l'index, sinon exit 2 AVANT recreate_index (aucun vecteur supprimé).
    space_error = check_input_dimensions(inputs, target_dimension)
    if space_error:
        print(f"ERREUR: espace vectoriel incompatible avec l'index '{args.index}' — {space_error}")
        if args.recreate:
            print("  Aucun index supprimé ni recréé.")
        return 2

    if args.recreate:
        recreate_index(pc, args.index, args.dimension, args.metric, args.cloud, args.region)

    backfill_temps = []
    if args.backfill_hash:
        if os.getenv("DEDUP_ENABLED", "0").strip().lower() not in ("1", "true", "yes", "on"):
            print("  Attention: --backfill-hash sans DEDUP_ENABLED=1 → les content_hash sont "
                  "écrits mais insert_to_pinecone ne déduplique pas. Exporter DEDUP_ENABLED=1.")
        print("Back-fill des content_hash (hash sur texte stocké — cf. caveat raw vs recodé):")
        inputs = backfill_inputs(inputs)
        backfill_temps = [p for p in inputs if p.endswith(".dedup.json")]

    print(f"\n{len(inputs)} corpus à uploader vers l'index '{args.index}':")
    for p in inputs:
        print(f"  - {p}")
    print()

    results = []
    grand_total = 0
    for i, path in enumerate(inputs, 1):
        session = os.path.basename(os.path.dirname(path))
        print(f"\n========== [{i}/{len(inputs)}] {session} ==========")
        try:
            res = insert_to_pinecone(
                embeddings_json_file=path,
                index_name=args.index,
                pinecone_api_key=api_key,
                namespace=args.namespace,
            )
        except Exception as exc:  # keep going; record the failure
            print(f"EXCEPTION pendant l'upload de {session}: {exc}")
            res = {"status": "error", "message": str(exc), "inserted_count": 0}
        status = res.get("status", "unknown")
        count = res.get("inserted_count", 0)
        grand_total += count
        results.append((session, status, count))
        print(f"--> {session}: status={status}, inserted={count}")

    # Nettoyage des fichiers temporaires de back-fill.
    for tmp in backfill_temps:
        try:
            os.remove(tmp)
        except OSError:
            pass

    # Summary.
    print("\n" + "=" * 60)
    print(f"RÉSUMÉ — index '{args.index}'")
    print("=" * 60)
    for session, status, count in results:
        flag = "OK " if status in ("success", "success_partial_data") else "!! "
        print(f"  {flag}{session:<32} {status:<22} {count:>7}")
    print("-" * 60)
    print(f"  Total vecteurs insérés (somme upload): {grand_total}")

    failures = [s for s, st, _ in results if st not in ("success", "success_partial_data")]
    if failures:
        print(f"\n{len(failures)} corpus en échec: {', '.join(failures)}")
        return 1
    print("\nTous les corpus ont été uploadés avec succès.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
