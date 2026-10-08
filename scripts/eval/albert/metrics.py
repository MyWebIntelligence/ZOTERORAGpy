"""Métriques et rapport de décision des fiches d'évaluation Albert.

Relit les fiches remplies (export HTML ``<fiche>_rempli.csv``, sinon la fiche
CSV remplie au tableur ; « ; » ou « , », BOM toléré) et leur clé privée
(``ocr_items.jsonl``, ``audio_items.jsonl``), calcule les mesures puis applique
les critères **fixés avant tout jugement** (``README.md`` du dossier) :

OCR (D21 : recoder ou non les textes LightOnOCR), pages LightOnOCR réussies
(version brute) : fidélité moyenne ≥ 2,5, au plus 5 % de pages notées ≤ 1 et,
si une comparaison A/B existe, version comparée (recodée) préférée dans moins
de 60 % des paires tranchées. Sinon le recodage est gardé (défaut actuel).

Transcription audio (Whisper) : la transcription est jugée utilisable sans
relecture humaine si la fidélité moyenne des extraits est ≥ 2,5 et si au plus
5 % des extraits sont notés ≤ 1 ; sinon une relecture humaine est nécessaire.

Le script ne modifie aucune configuration. L'évaluation de la variante de
recherche a été retirée le 2026-10-03 avec la fonction de réponses sourcées.

Exemple (une ligne) ::

    .venv/bin/python scripts/eval/albert/metrics.py --dir data/albert_eval/2026-10-02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.eval.albert import common as C  # noqa: E402

KAPPA_MIN = 0.6
D21_MEAN_MIN = 2.5
D21_LOW_SHARE_MAX = 0.05
D21_COMPARED_PREF_MAX = 0.60
D21_PROVIDER = "albert_lightonocr"
AUDIO_MEAN_MIN = 2.5
AUDIO_LOW_SHARE_MAX = 0.05
EPS = 1e-9


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------
def locate_sheet(directory: Path, stem: str, explicit: Optional[str]) -> Optional[Path]:
    """Fiche remplie à relire : chemin explicite, sinon export HTML, sinon CSV rempli en place.

    Args:
        directory: dossier de campagne.
        stem: nom de base de la fiche (``fiche_ocr``, ``fiche_audio``).
        explicit: chemin donné en option (prioritaire).

    Returns:
        Le chemin, ou ``None`` si aucune fiche n'existe.
    """
    if explicit:
        return Path(explicit)
    for name in (f"{stem}_rempli.csv", f"{stem}.csv"):
        candidate = Path(directory) / name
        if candidate.exists():
            return candidate
    return None


def load_key(path: Path) -> Dict[str, Any]:
    """Lit la clé privée d'une fiche (``*_items.jsonl``).

    Args:
        path: fichier de clé.

    Returns:
        ``{meta, items, documents}``.

    Raises:
        C.EvalError: fichier illisible ou sans en-tête ``meta``.
    """
    meta: Dict[str, Any] = {}
    items: Dict[str, Dict[str, Any]] = {}
    documents: List[Dict[str, Any]] = []
    for record in C.read_jsonl(path):
        kind = record.get("record")
        if kind == "meta":
            meta = record
        elif kind == "item":
            items[str(record["item_id"])] = record
        elif kind == "document":
            documents.append(record)
    if not meta:
        raise C.EvalError(f"{Path(path).name} : en-tête meta absent (clé incomplète)")
    return {"meta": meta, "items": items, "documents": documents}


def read_judgments(sheet: Path, key: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """Relit une fiche remplie et la confronte à sa clé.

    Une note ou une préférence illisible, un identifiant inconnu, deux
    saisies contradictoires du même élément ou une ligne qui ne correspond
    plus à la clé (document, page ou début différents) sont des erreurs nommées, avec
    fichier et ligne ; l'élément concerné compte alors comme non jugé.

    Args:
        sheet: fiche CSV remplie.
        key: clé (``load_key``).

    Returns:
        ``(saisies {item_id: {note, preference, commentaire}}, erreurs, avertissements)``.

    Raises:
        C.EvalError: fichier illisible.
    """
    rows, warnings = C.read_csv_rows(sheet)
    values: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    name = Path(sheet).name
    if rows and "item_id" not in rows[0][1]:
        raise C.EvalError(f"{name} : colonne item_id absente")
    for line, row in rows:
        item_id = row.get("item_id", "")
        if not item_id:
            continue
        item = key["items"].get(item_id)
        if item is None:
            errors.append(f"{name} ligne {line} : élément {item_id} inconnu de la clé")
            continue
        mismatch = [col for col, expected in (("document", item.get("document")), ("page", item.get("page")),
                                               ("debut", item.get("debut")))
                    if expected is not None and col in row and row[col] != str(expected)]
        if mismatch:
            errors.append(f"{name} ligne {line} ({item_id}) : {', '.join(mismatch)} différent(s) de la clé")
            continue
        entry: Dict[str, Any] = {"note": None, "preference": None, "commentaire": row.get("commentaire", "")}
        try:
            entry["note"] = C.parse_grade(row.get("note", ""))
        except ValueError as exc:
            errors.append(f"{name} ligne {line} ({item_id}) : {exc}")
        try:
            entry["preference"] = C.parse_preference(row.get("preference", ""))
        except ValueError as exc:
            errors.append(f"{name} ligne {line} ({item_id}) : {exc}")
        previous = values.get(item_id)
        if previous is not None and (previous["note"], previous["preference"]) != (entry["note"],
                                                                                    entry["preference"]):
            errors.append(f"{name} ligne {line} ({item_id}) : saisie contradictoire avec une ligne précédente")
        values[item_id] = entry
    return values, errors, warnings


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------
def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Moyenne des valeurs définies (``None`` si aucune)."""
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def agreement(pairs: Sequence[Tuple[int, int]], labels: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """Accord entre premier et second jugement.

    Args:
        pairs: couples de notes ``(premier, second)``.
        labels: identifiants ``(premier, second)`` de chaque couple.

    Returns:
        ``{n_pairs, kappa, kappa_linear, raw, disagreements}`` (désaccords de 2 degrés ou plus).
    """
    return {
        "n_pairs": len(pairs),
        "kappa": C.weighted_kappa(pairs, weights="quadratic"),
        "kappa_linear": C.weighted_kappa(pairs, weights="linear"),
        "raw": (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None,
        "disagreements": [{"first": labels[i][0], "second": labels[i][1], "grades": [a, b]}
                          for i, (a, b) in enumerate(pairs) if abs(a - b) >= 2 and i < len(labels)],
    }


def _second_pairs(key: Mapping[str, Any], values: Mapping[str, Mapping[str, Any]]
                  ) -> Tuple[List[Tuple[int, int]], List[Tuple[str, str]]]:
    """Couples de notes (premier, second jugement) et leurs identifiants."""
    pairs: List[Tuple[int, int]] = []
    labels: List[Tuple[str, str]] = []
    for item_id, item in sorted(key["items"].items()):
        if item.get("kind") != "second":
            continue
        first = (values.get(item["of"]) or {}).get("note")
        second = (values.get(item_id) or {}).get("note")
        if first is not None and second is not None:
            pairs.append((first, second))
            labels.append((item["of"], item_id))
    return pairs, labels


def _grade_stats(notes: Mapping[str, Optional[int]], ids: Sequence[str]) -> Dict[str, Any]:
    """Effectif, nombre de jugés, moyenne, part des notes ≤ 1 et répartition d'un groupe d'éléments."""
    graded = [notes[i] for i in ids if notes.get(i) is not None]
    return {"n": len(ids), "judged": len(graded), "mean": _mean(graded),
            "low_share": (sum(1 for g in graded if g <= 1) / len(graded)) if graded else None,
            "grades": {str(g): sum(1 for x in graded if x == g) for g in range(4)}}


def ocr_metrics(key: Mapping[str, Any], values: Mapping[str, Mapping[str, Any]],
                provider: str = D21_PROVIDER) -> Dict[str, Any]:
    """Mesures de fidélité OCR et préférences A/B.

    Args:
        key: clé de la fiche OCR.
        values: saisies relues.
        provider: fournisseur visé par D21.

    Returns:
        ``{provider, population, failed, groups, ab, agreement, missing, compare_column}``.
    """
    items = key["items"]
    first = {i: it for i, it in items.items() if it.get("kind") == "fidelity"}
    notes = {i: (values.get(i) or {}).get("note") for i in first}
    missing = sorted([i for i, n in notes.items() if n is None]
                     + [i for i, it in items.items() if it.get("kind") == "ab"
                        and (values.get(i) or {}).get("preference") is None])

    def stats(ids: Sequence[str]) -> Dict[str, Any]:
        """Mesures d'un groupe d'éléments de fidélité."""
        return _grade_stats(notes, ids)

    population = [i for i, it in first.items() if it.get("version") == "brut" and it.get("provider") == provider
                  and not it.get("failed")]
    failed = [i for i, it in first.items() if it.get("version") == "brut" and it.get("provider") == provider
              and it.get("failed")]
    groups = {}
    for (prov, version, is_failed) in sorted({(it.get("provider", ""), it.get("version", ""), bool(it.get("failed")))
                                              for it in first.values()}):
        ids = [i for i, it in first.items() if (it.get("provider", ""), it.get("version", ""),
                                                bool(it.get("failed"))) == (prov, version, is_failed)]
        groups[f"{prov or '?'} · {version} · {'échec' if is_failed else 'réussie'}"] = stats(ids)
    counts = {"brut": 0, "comparée": 0, "égal": 0, "non jugé": 0}
    for item_id, item in items.items():
        if item.get("kind") != "ab" or item.get("provider") != provider:
            continue
        pref = (values.get(item_id) or {}).get("preference")
        if pref is None:
            counts["non jugé"] += 1
        elif pref == "égal":
            counts["égal"] += 1
        else:
            counts[item["a"] if pref == "A" else item["b"]] += 1
    decided = counts["brut"] + counts["comparée"]
    pairs, labels = _second_pairs(key, values)
    return {"provider": provider, "population": stats(population), "failed": stats(failed), "groups": groups,
            "ab": {"counts": counts, "decided": decided, "n": sum(counts.values()),
                   "compared_share": (counts["comparée"] / decided) if decided else None},
            "agreement": agreement(pairs, labels), "missing": missing,
            "compare_column": key["meta"].get("compare_column")}


def decide_d21(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Applique le gate D21 (fixé d'avance).

    Args:
        summary: résultat de ``ocr_metrics``.

    Returns:
        ``{verdict, skip_recode, decided, checks}`` ; ``skip_recode`` vaut
        ``True`` seulement si les trois critères sont tenus.
    """
    checks: List[Dict[str, Any]] = []
    pop = summary["population"]
    provider = summary["provider"]
    keep = "garder le recodage des textes LightOnOCR (défaut actuel, ALBERT_OCR_SKIP_RECODE=0)"
    if summary["missing"]:
        checks.append({"criterion": "complétude de la fiche OCR", "value": f"{len(summary['missing'])} non jugé(s)",
                       "threshold": "0", "ok": False})
        return {"verdict": "jugements incomplets : pas de décision D21", "skip_recode": False, "decided": False,
                "checks": checks}
    if not pop["judged"]:
        return {"verdict": f"pas de décision D21 : aucune page {provider} réussie jugée", "skip_recode": False,
                "decided": False, "checks": checks}
    ok_mean = pop["mean"] >= D21_MEAN_MIN - EPS
    ok_low = pop["low_share"] <= D21_LOW_SHARE_MAX + EPS
    checks.append({"criterion": "fidélité moyenne (version brute)", "value": C.fmt(pop["mean"], 2),
                   "threshold": f"≥ {C.fmt(D21_MEAN_MIN, 1)}", "ok": ok_mean})
    checks.append({"criterion": "part des pages notées ≤ 1", "value": C.pct(pop["low_share"]),
                   "threshold": f"≤ {C.pct(D21_LOW_SHARE_MAX)}", "ok": ok_low})
    share = summary["ab"]["compared_share"]
    if summary["ab"]["n"]:
        ok_ab = share is None or share < D21_COMPARED_PREF_MAX - EPS
        checks.append({"criterion": "version comparée (recodée) préférée, paires tranchées",
                       "value": C.pct(share) + f" de {summary['ab']['decided']}",
                       "threshold": f"< {C.pct(D21_COMPARED_PREF_MAX)}", "ok": ok_ab})
    else:
        ok_ab = True
        checks.append({"criterion": "comparaison A/B", "value": "aucune", "threshold": "sans objet", "ok": True})
    if ok_mean and ok_low and ok_ab:
        return {"verdict": "ne plus recoder les textes LightOnOCR (ALBERT_OCR_SKIP_RECODE=1)", "skip_recode": True,
                "decided": True, "checks": checks}
    return {"verdict": keep, "skip_recode": False, "decided": True, "checks": checks}


def audio_metrics(key: Mapping[str, Any], values: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Mesures de fidélité des transcriptions audio.

    Args:
        key: clé de la fiche audio.
        values: saisies relues.

    Returns:
        ``{population, documents, segments_total, segments_failed, agreement, missing}``.
    """
    first = {i: it for i, it in key["items"].items() if it.get("kind") == "first"}
    notes = {i: (values.get(i) or {}).get("note") for i in first}
    per_doc: Dict[str, List[str]] = {}
    for item_id, item in sorted(first.items()):
        per_doc.setdefault(str(item.get("document")), []).append(item_id)
    pairs, labels = _second_pairs(key, values)
    docs = key["documents"]
    return {"population": _grade_stats(notes, list(first)),
            "documents": {name: _grade_stats(notes, ids) for name, ids in sorted(per_doc.items())},
            "segments_total": sum(int(d.get("segments_total") or 0) for d in docs),
            "segments_failed": sum(int(d.get("segments_failed") or 0) for d in docs),
            "agreement": agreement(pairs, labels),
            "missing": sorted(i for i, n in notes.items() if n is None)}


def decide_audio(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Applique le gate de la transcription audio (fixé d'avance).

    Args:
        summary: résultat de ``audio_metrics``.

    Returns:
        ``{verdict, usable, decided, checks}`` ; ``usable`` vaut ``True``
        seulement si la moyenne est ≥ 2,5 et la part des extraits notés ≤ 1 au plus 5 %.
    """
    checks: List[Dict[str, Any]] = []
    pop = summary["population"]
    if summary["missing"]:
        checks.append({"criterion": "complétude de la fiche audio", "value": f"{len(summary['missing'])} non jugé(s)",
                       "threshold": "0", "ok": False})
        return {"verdict": "jugements incomplets : pas de décision", "usable": False, "decided": False,
                "checks": checks}
    if not pop["judged"]:
        return {"verdict": "pas de décision : aucun extrait jugé", "usable": False, "decided": False,
                "checks": checks}
    ok_mean = pop["mean"] >= AUDIO_MEAN_MIN - EPS
    ok_low = pop["low_share"] <= AUDIO_LOW_SHARE_MAX + EPS
    checks.append({"criterion": "fidélité moyenne des extraits", "value": C.fmt(pop["mean"], 2),
                   "threshold": f"≥ {C.fmt(AUDIO_MEAN_MIN, 1)}", "ok": ok_mean})
    checks.append({"criterion": "part des extraits notés ≤ 1", "value": C.pct(pop["low_share"]),
                   "threshold": f"≤ {C.pct(AUDIO_LOW_SHARE_MAX)}", "ok": ok_low})
    if ok_mean and ok_low:
        return {"verdict": "transcription utilisable sans relecture humaine", "usable": True, "decided": True,
                "checks": checks}
    return {"verdict": "relecture humaine nécessaire avant usage de la transcription", "usable": False,
            "decided": True, "checks": checks}


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def _checks_table(checks: Sequence[Mapping[str, Any]]) -> List[str]:
    """Tableau Markdown des critères appliqués."""
    if not checks:
        return []
    lines = ["| Critère | Valeur | Seuil | Tenu |", "|---|---|---|---|"]
    lines += [f"| {c['criterion']} | {c['value']} | {c['threshold']} | {'oui' if c['ok'] else 'non'} |"
              for c in checks]
    return lines + [""]


def _list(values: Sequence[str], limit: int = 30) -> str:
    """Liste courte d'identifiants (« … » au-delà de ``limit``)."""
    shown = ", ".join(values[:limit])
    return shown + (f" … (+{len(values) - limit})" if len(values) > limit else "")


def _agreement_lines(ag: Mapping[str, Any]) -> List[str]:
    """Lignes Markdown de l'accord inter-juges."""
    lines = [f"- couples (premier, second jugement) : {ag['n_pairs']}",
             f"- kappa pondéré quadratique : **{C.fmt(ag['kappa'])}** ; linéaire : {C.fmt(ag['kappa_linear'])} ; "
             f"accord brut : {C.pct(ag['raw'])}"]
    if ag["disagreements"]:
        lines.append("- désaccords de 2 degrés ou plus (à relire) : " + _list(
            [f"{d['first']}/{d['second']} ({d['grades'][0]} puis {d['grades'][1]})" for d in ag["disagreements"]]))
    return lines


def render_ocr(summary: Mapping[str, Any], decision: Mapping[str, Any], sheet: Path,
               errors: Sequence[str]) -> List[str]:
    """Section Markdown de l'OCR (D21).

    Args:
        summary: ``ocr_metrics``.
        decision: ``decide_d21``.
        sheet: fiche relue.
        errors: erreurs de lecture.

    Returns:
        Les lignes de la section.
    """
    pop, failed, ab = summary["population"], summary["failed"], summary["ab"]
    compare = summary.get("compare_column")
    lines = ["## 1. OCR LightOnOCR : recodage des textes (D21)", "",
             "### Critères fixés d'avance", "",
             f"Population : pages `{summary['provider']}` réussies, transcription brute (premier jugement). "
             "Les pages en échec sont jugées et comptées à part : le recodage ne les corrige pas.", "",
             f"1. fidélité moyenne ≥ {C.fmt(D21_MEAN_MIN, 1)} ;",
             f"2. au plus {C.pct(D21_LOW_SHARE_MAX)} des pages notées ≤ 1 ;",
             f"3. si une comparaison A/B existe, la version comparée (recodée) est préférée dans moins de "
             f"{C.pct(D21_COMPARED_PREF_MAX)} des paires tranchées (A ou B, hors « égal »).", "",
             "Les trois tenus : ne plus recoder (`ALBERT_OCR_SKIP_RECODE=1`) ; sinon garder le recodage (défaut "
             "actuel).", "",
             "### Données", "",
             f"- fiche relue : `{sheet.name}`" + (f" ; colonne comparée : `{compare}`" if compare else ""),
             f"- pages réussies jugées : {pop['judged']} sur {pop['n']} ; fidélité moyenne **{C.fmt(pop['mean'], 2)}** ; "
             f"notées ≤ 1 : {C.pct(pop['low_share'])} ; répartition 0/1/2/3 : "
             + "/".join(str(pop["grades"][str(g)]) for g in range(4)),
             f"- pages en échec ({summary['provider']}) : {failed['n']} (fidélité moyenne {C.fmt(failed['mean'], 2)})"]
    if ab["n"]:
        c = ab["counts"]
        lines.append(f"- comparaisons A/B : {ab['n']} ; préférée brute {c['brut']}, comparée {c['comparée']}, "
                     f"égal {c['égal']}, non jugée {c['non jugé']} ; comparée préférée dans "
                     f"{C.pct(ab['compared_share'])} des paires tranchées")
    if errors:
        lines.append(f"- **{len(errors)} erreur(s) de lecture** (éléments comptés comme non jugés) :")
        lines += [f"  - {e}" for e in errors[:50]]
    lines += ["", "| Groupe (fournisseur · version · statut) | Pages | Jugées | Moyenne | Notées ≤ 1 |",
              "|---|---|---|---|---|"]
    for name, g in summary["groups"].items():
        lines.append(f"| {name} | {g['n']} | {g['judged']} | {C.fmt(g['mean'], 2)} | {C.pct(g['low_share'])} |")
    lines += ["", "### Accord inter-juges (second jugement, information)", ""] + _agreement_lines(summary["agreement"])
    kappa = summary["agreement"]["kappa"]
    if kappa is not None and kappa < KAPPA_MIN:
        lines.append(f"- attention : kappa sous {C.fmt(KAPPA_MIN, 1)}, relire les désaccords avant d'appliquer "
                     "la décision")
    if summary["missing"]:
        lines.append(f"- éléments non jugés : {_list(summary['missing'])}")
    lines += ["", "### Décision", ""] + _checks_table(decision["checks"]) + [f"**Verdict : {decision['verdict']}.**",
                                                                            ""]
    if decision.get("decided"):
        lines += ["Le script ne modifie aucune configuration : `ALBERT_OCR_SKIP_RECODE` reste à changer à la main, "
                  "sur ce rapport.", ""]
    return lines


def render_audio(summary: Mapping[str, Any], decision: Mapping[str, Any], sheet: Path,
                 errors: Sequence[str]) -> List[str]:
    """Section Markdown de la transcription audio.

    Args:
        summary: ``audio_metrics``.
        decision: ``decide_audio``.
        sheet: fiche relue.
        errors: erreurs de lecture.

    Returns:
        Les lignes de la section.
    """
    pop = summary["population"]
    lines = ["## 2. Transcription audio (Whisper) : utilisable sans relecture ?", "",
             "### Critères fixés d'avance", "",
             f"Transcription utilisable sans relecture humaine si la fidélité moyenne des extraits est ≥ "
             f"{C.fmt(AUDIO_MEAN_MIN, 1)} et si au plus {C.pct(AUDIO_LOW_SHARE_MAX)} des extraits sont notés ≤ 1 ; "
             "sinon relecture humaine nécessaire. Les segments en échec ne sont pas jugés (aucun énoncé) : leur "
             "nombre est donné à part.", "",
             "### Données", "",
             f"- fiche relue : `{sheet.name}`",
             f"- extraits jugés : {pop['judged']} sur {pop['n']} ; fidélité moyenne **{C.fmt(pop['mean'], 2)}** ; "
             f"notés ≤ 1 : {C.pct(pop['low_share'])} ; répartition 0/1/2/3 : "
             + "/".join(str(pop["grades"][str(g)]) for g in range(4)),
             f"- segments en échec (non jugés) : {summary['segments_failed']} sur {summary['segments_total']}"]
    if errors:
        lines.append(f"- **{len(errors)} erreur(s) de lecture** (éléments comptés comme non jugés) :")
        lines += [f"  - {e}" for e in errors[:50]]
    lines += ["", "| Enregistrement | Extraits | Jugés | Moyenne | Notés ≤ 1 |", "|---|---|---|---|---|"]
    for name, g in summary["documents"].items():
        lines.append(f"| {name} | {g['n']} | {g['judged']} | {C.fmt(g['mean'], 2)} | {C.pct(g['low_share'])} |")
    lines += ["", "### Accord inter-juges (second jugement, information)", ""] + _agreement_lines(summary["agreement"])
    if summary["missing"]:
        lines.append(f"- éléments non jugés : {_list(summary['missing'])}")
    lines += ["", "### Décision", ""] + _checks_table(decision["checks"]) + [f"**Verdict : {decision['verdict']}.**",
                                                                            ""]
    return lines


def render_report(campaign: str, ocr: Optional[Tuple[Any, ...]], audio: Optional[Tuple[Any, ...]] = None) -> str:
    """Rapport de décision complet (Markdown, français).

    Args:
        campaign: nom de la campagne (dossier).
        ocr: ``(summary, decision, sheet, errors)`` de l'OCR, ou ``None``.
        audio: ``(summary, decision, sheet, errors)`` de l'audio, ou ``None``.

    Returns:
        Le texte du rapport.
    """
    lines = [f"# Rapport de décision — évaluation humaine Albert (campagne {campaign})", "",
             "Produit par `scripts/eval/albert/metrics.py` à partir des fiches remplies. Les critères ci-dessous "
             "ont été fixés avant tout jugement (`scripts/eval/albert/README.md`).", ""]
    lines += render_ocr(*ocr) if ocr else ["## 1. OCR (D21)", "", "Fiche OCR absente : non évaluée.", ""]
    if audio:
        lines += render_audio(*audio)
    return "\n".join(lines).rstrip() + "\n"


def _jsonable(value: Any) -> Any:
    """Copie sérialisable en JSON (clés en chaînes)."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée de la ligne de commande.

    Args:
        argv: arguments (défaut : ``sys.argv[1:]``).

    Returns:
        0 quand le rapport est écrit (avec ou sans décision), 2 si rien n'est à évaluer.
    """
    parser = argparse.ArgumentParser(description="Métriques et rapport de décision des fiches Albert.")
    parser.add_argument("--dir", help="dossier de campagne (défaut : data/albert_eval/<date du jour>)")
    parser.add_argument("--ocr-sheet", help="fiche OCR remplie (défaut : fiche_ocr_rempli.csv, sinon fiche_ocr.csv)")
    parser.add_argument("--ocr-items", help="clé de la fiche OCR (défaut : ocr_items.jsonl)")
    parser.add_argument("--audio-sheet", help="fiche audio remplie (défaut : fiche_audio_rempli.csv, sinon fiche_audio.csv)")
    parser.add_argument("--audio-items", help="clé de la fiche audio (défaut : audio_items.jsonl)")
    parser.add_argument("--d21-provider", default=D21_PROVIDER, help="fournisseur visé par D21")
    parser.add_argument("--out", help="rapport (défaut : <dossier>/rapport_decision.md)")
    args = parser.parse_args(argv)
    directory = Path(args.dir) if args.dir else C.default_out_dir()
    ocr = audio = None
    out: Dict[str, Any] = {}
    try:
        ocr_key_path = Path(args.ocr_items) if args.ocr_items else directory / "ocr_items.jsonl"
        ocr_sheet = locate_sheet(directory, "fiche_ocr", args.ocr_sheet)
        if ocr_key_path.exists() and ocr_sheet is not None:
            key = load_key(ocr_key_path)
            values, errors, warnings = read_judgments(ocr_sheet, key)
            summary = ocr_metrics(key, values, args.d21_provider)
            decision = decide_d21(summary)
            ocr = (summary, decision, ocr_sheet, errors + warnings)
            out["ocr"] = {"metrics": summary, "decision": decision, "errors": errors + warnings}
        audio_key_path = Path(args.audio_items) if args.audio_items else directory / "audio_items.jsonl"
        audio_sheet = locate_sheet(directory, "fiche_audio", args.audio_sheet)
        if audio_key_path.exists() and audio_sheet is not None:
            key = load_key(audio_key_path)
            values, errors, warnings = read_judgments(audio_sheet, key)
            summary = audio_metrics(key, values)
            decision = decide_audio(summary)
            audio = (summary, decision, audio_sheet, errors + warnings)
            out["audio"] = {"metrics": summary, "decision": decision, "errors": errors + warnings}
        if ocr is None and audio is None:
            raise C.EvalError(f"aucune fiche ni clé à évaluer dans {C.display_path(directory)}")
        report_path = Path(args.out) if args.out else directory / "rapport_decision.md"
        C.write_text(report_path, render_report(directory.name, ocr, audio))
        C.write_text(report_path.with_name("metriques.json"),
                     json.dumps(_jsonable(out), ensure_ascii=False, indent=2) + "\n")
    except C.EvalError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    if ocr:
        print(f"OCR (D21) : {ocr[1]['verdict']}")
    if audio:
        print(f"audio : {audio[1]['verdict']}")
    print(f"rapport : {C.display_path(report_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
