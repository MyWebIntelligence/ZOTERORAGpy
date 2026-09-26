#!/usr/bin/env python3
"""Back-fill *in-place* d'un index Pinecone : dote les vecteurs EXISTANTS d'un
``content_hash`` et les re-clé en ID adressé par contenu, **sans recréer l'index**
ni dépendre des fichiers sources (qui peuvent avoir été nettoyés).

Pourquoi in-place plutôt que ``rebuild --recreate`` :
  - les sources locales (``uploads/*/...json``) sont **incomplètes** → un recreate
    perdrait les vecteurs des sessions nettoyées ;
  - le corpus est **100% Mistral** (recodage sauté) → le texte stocké == texte brut,
    donc hacher le texte stocké donne le MÊME ``content_hash`` qu'un futur ingest
    frais : le back-fill est forward-compatible (pas de caveat raw-vs-stocké) ;
  - ni re-OCR ni re-embedding (coût API nul ; les vecteurs existants sont réutilisés).

Effet :
  1. **Protection forward** : chaque vecteur éligible reçoit ``content_hash`` +
     ``dedup_eligible`` et un ID ``"{hash[:16]}_{chunk_index}"`` → un futur ré-ingest
     du même document est reconnu par la dédup (Tier 2 ``index.fetch``).
  2. **Nettoyage des doublons EXISTANTS** : deux ré-ingests d'un même doc (même texte,
     même ``chunk_index``, ``doc_id`` aléatoires distincts) collapsent vers le même ID
     adressé par contenu → un seul survit.

Garde anti-fusion cross-document : deux documents DIFFÉRENTS partageant un chunk
identique au même ``chunk_index`` (boilerplate) auraient le même ID cible. La
corroboration **titre** détecte ce cas : ces groupes ne sont PAS fusionnés (ID d'origine
conservés), seul ``content_hash`` leur est ajouté (la dédup d'insertion les gèrera avec
sa corroboration titre).

Idempotent : ré-exécuter ne change que ce qui doit l'être (vecteurs déjà adressés
contenu = simple ajout de ``content_hash`` si absent). Sûr : upsert du nouvel ID AVANT
suppression de l'ancien (aucune fenêtre de perte).

Usage :
    python3 scripts/backfill_pinecone_inplace.py --index articles --dry-run
    python3 scripts/backfill_pinecone_inplace.py --index articles --apply
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402
import rad_dedup  # noqa: E402

FETCH_BATCH = 100   # ids par appel fetch (les vecteurs portent les values → lots modestes)
UPSERT_BATCH = 100
DEFAULT_MIN_CHARS = 64


def _load_key():
    """Charge ``.env`` (répertoire courant) et renvoie ``PINECONE_API_KEY`` ;
    quitte le processus (code 2) si la clé est absente."""
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        print("ERREUR: PINECONE_API_KEY non défini.")
        sys.exit(2)
    return key


def list_all_ids(index, namespace):
    """Liste paginée de TOUS les ids d'un namespace (léger : ids seuls)."""
    ids = []
    for page in index.list(namespace=namespace, limit=100):
        ids.extend(page)
    return ids


def scan(index, namespace, min_chars=DEFAULT_MIN_CHARS):
    """Pass 1 — récupère pour chaque vecteur : content_hash, éligibilité,
    chunk_index, title, id cible. Construit la map des collisions (target_id →
    titres distincts). Les values sont récupérées mais DISCARD (analyse seule)."""
    ids = list_all_ids(index, namespace)
    total = len(ids)
    print(f"  {total} ids listés. Récupération des métadonnées par lots de {FETCH_BATCH}…")

    records = []                 # {old_id, content_hash, eligible, chunk_index, title, target_id}
    target_titles = defaultdict(set)   # target_id -> {titres}
    n_no_text = 0
    done = 0
    for i in range(0, total, FETCH_BATCH):
        batch = ids[i:i + FETCH_BATCH]
        vectors = index.fetch(ids=batch, namespace=namespace).vectors
        for vid, vec in vectors.items():
            meta = vec.metadata or {}
            text = meta.get("text", "") or ""
            title = str(meta.get("title", "") or "").strip().lower()
            try:
                cidx = int(float(meta.get("chunk_index", 0)))
            except (TypeError, ValueError):
                cidx = 0
            norm = rad_dedup.normalize_text_for_hash(text)
            chash = rad_dedup.content_hash(text)
            eligible = rad_dedup.is_dedup_eligible(norm, min_chars)
            if not norm:
                n_no_text += 1
            has_ch = bool(meta.get("content_hash"))
            target = rad_dedup.content_id(chash, cidx) if eligible else vid
            records.append({
                "old_id": vid, "content_hash": chash, "eligible": eligible,
                "chunk_index": cidx, "title": title, "target_id": target,
                "has_content_hash": has_ch,
            })
            if eligible:
                target_titles[target].add(title)
        done += len(vectors)
        if done % 2000 < FETCH_BATCH:
            print(f"    … {done}/{total}")
    return records, target_titles, n_no_text


def plan(records, target_titles):
    """Pass 2 — classe chaque vecteur : rekey (survivant), delete (doublon),
    collision (laissé intact + metadata), metadata_only (déjà adressé contenu),
    skip (inéligible)."""
    # Représentant par target_id (préférence : un id déjà == target_id, sinon le 1er).
    groups = defaultdict(list)
    for r in records:
        if r["eligible"]:
            groups[r["target_id"]].append(r)

    actions = {"rekey": [], "delete": [], "collision_meta": [], "meta_only": [], "skip_ineligible": 0}
    for r in records:
        if not r["eligible"]:
            actions["skip_ineligible"] += 1

    for target, members in groups.items():
        titles = target_titles[target]
        if len(titles) > 1:
            # Collision cross-document : NE PAS fusionner. Ajouter content_hash si absent.
            for m in members:
                if not m["has_content_hash"]:
                    actions["collision_meta"].append(m)
            continue
        # Groupe sûr (1 seul titre) : 1 survivant adressé contenu, le reste supprimé.
        canonical = next((m for m in members if m["old_id"] == target), members[0])
        for m in members:
            if m is canonical:
                if canonical["old_id"] == target:
                    if not canonical["has_content_hash"]:
                        actions["meta_only"].append(canonical)  # déjà bien clé, juste metadata
                else:
                    actions["rekey"].append(canonical)          # re-clé vers target
            else:
                actions["delete"].append(m)                     # doublon → supprimé
    return actions


def report(total, records, target_titles, actions, n_no_text):
    """Affiche le plan de back-fill (compteurs de re-clés, doublons, collisions,
    inéligibles) et renvoie la taille finale projetée de l'index (``total`` moins
    les doublons à supprimer)."""
    eligible = sum(1 for r in records if r["eligible"])
    rekey = len(actions["rekey"])
    delete = len(actions["delete"])
    coll = len(actions["collision_meta"])
    meta = len(actions["meta_only"])
    skip = actions["skip_ineligible"]
    collision_groups = sum(1 for t, ts in target_titles.items() if len(ts) > 1)
    final = total - delete
    print("\n" + "=" * 64)
    print("PLAN DE BACK-FILL IN-PLACE")
    print("=" * 64)
    print(f"  Vecteurs scannés                          : {total}")
    print(f"  Éligibles (texte >= {DEFAULT_MIN_CHARS} car. normalisés) : {eligible}")
    print(f"  Inéligibles (laissés intacts)             : {skip}")
    print(f"  Texte normalisé vide                      : {n_no_text}")
    print("  ----")
    print(f"  Re-clés vers ID adressé contenu           : {rekey}")
    print(f"  Doublons EXACTS à supprimer (collapse)    : {delete}")
    print(f"  Déjà adressé contenu (ajout content_hash) : {meta}")
    print(f"  Collisions cross-document (laissées, +hash): {coll}  (sur {collision_groups} groupes multi-titres)")
    print("  ----")
    print(f"  Taille finale projetée                    : {final}  (était {total}, -{delete} doublons)")
    print("=" * 64)
    return final


def apply(index, namespace, actions):
    """Exécute : re-clé (fetch values → upsert nouvel id → delete ancien), supprime
    les doublons, patche content_hash (metadata-only via index.update)."""
    # 1) Metadata-only (déjà adressé contenu + collisions) : index.update, pas de values.
    meta_targets = actions["meta_only"] + actions["collision_meta"]
    print(f"\n[apply] {len(meta_targets)} mises à jour metadata (content_hash)…")
    for m in meta_targets:
        try:
            index.update(id=m["old_id"], namespace=namespace,
                         set_metadata={"content_hash": m["content_hash"], "dedup_eligible": True})
        except Exception as e:
            print(f"  update {m['old_id']} échec: {e}")

    # 2) Re-clé : fetch values du représentant, upsert sous target_id, delete l'ancien.
    rekey = actions["rekey"]
    print(f"[apply] {len(rekey)} re-clés (fetch→upsert→delete)…")
    for i in range(0, len(rekey), UPSERT_BATCH):
        if i and i % 2000 == 0:
            print(f"    … re-clé {i}/{len(rekey)}")
        batch = rekey[i:i + UPSERT_BATCH]
        fetched = index.fetch(ids=[m["old_id"] for m in batch], namespace=namespace).vectors
        vectors = []
        for m in batch:
            vec = fetched.get(m["old_id"])
            if vec is None:
                continue
            md = dict(vec.metadata or {})
            md["content_hash"] = m["content_hash"]
            md["dedup_eligible"] = True
            vectors.append({"id": m["target_id"], "values": list(vec.values), "metadata": md})
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
        # supprimer les anciens ids re-clés (après upsert réussi du nouvel id)
        old_ids = [m["old_id"] for m in batch if m["old_id"] != m["target_id"]]
        if old_ids:
            index.delete(ids=old_ids, namespace=namespace)

    # 3) Supprimer les doublons exacts.
    delete = [m["old_id"] for m in actions["delete"]]
    print(f"[apply] {len(delete)} doublons à supprimer…")
    for i in range(0, len(delete), UPSERT_BATCH):
        index.delete(ids=delete[i:i + UPSERT_BATCH], namespace=namespace)

    print("[apply] Terminé.")


def main():
    """Point d'entrée CLI : scanne l'index, construit et affiche le plan, puis
    l'exécute seulement avec ``--apply`` (après 5 s de délai) ; ``--dry-run`` reste
    en lecture seule. Renvoie 0."""
    p = argparse.ArgumentParser(description="Back-fill in-place d'un index Pinecone (content_hash + re-clé).")
    p.add_argument("--index", required=True)
    p.add_argument("--namespace", default="")
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Analyse seule (read-only), aucun écriture.")
    g.add_argument("--apply", action="store_true", help="Applique le plan (re-clé/supprime/patche). DESTRUCTIF.")
    args = p.parse_args()

    key = _load_key()
    from pinecone import Pinecone
    index = Pinecone(api_key=key).Index(args.index)

    print(f"=== Back-fill in-place : index '{args.index}', namespace '{args.namespace or '(default)'}' ===")
    t0 = time.time()
    records, target_titles, n_no_text = scan(index, args.namespace, args.min_chars)
    actions = plan(records, target_titles)
    report(len(records), records, target_titles, actions, n_no_text)
    print(f"(scan {time.time() - t0:.0f}s)")

    if args.apply:
        print("\n⚠️  Mode --apply : exécution des écritures dans 5s (Ctrl-C pour annuler)…")
        time.sleep(5)
        apply(index, args.namespace, actions)
        print("\nVérification post-apply :")
        stats = index.describe_index_stats().get("namespaces", {}) or {}
        print(f"  namespace '{args.namespace or '(default)'}': "
              f"{(stats.get(args.namespace, {}) or {}).get('vector_count')} vecteurs")
    else:
        print("\nDRY-RUN : aucune écriture effectuée. Relancer avec --apply pour exécuter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
