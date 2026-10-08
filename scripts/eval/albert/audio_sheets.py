"""Fiche de jugement de la transcription audio (Albert, Whisper) : utilisable sans relecture ?

À partir d'une session audio de RAGpy (``scripts/rad_audio.py``) :

- ``output.csv`` : une ligne par enregistrement (``filename``, ``texteocr`` avec
  ses marqueurs ``<!-- Segment N (hh:mm:ss–hh:mm:ss) -->``, ``texteocr_provider``) ;
- ``<base>_audio_segments.json`` : segments et énoncés (sous-segments de
  Whisper) aux horodatages absolus de l'enregistrement ;
- le dossier des enregistrements (``--audio-dir``).

Environ 20 énoncés par enregistrement (``--clips-per-file``) sont tirés à
graine fixe parmi les énoncés non vides des segments réussis, de 30 s au plus
(``--max-clip-seconds``). Chaque extrait est découpé par ffmpeg (mono, 16 kHz,
mp3 32 kb/s, petite marge avant et après) et intégré en base64 dans la page
HTML (``<audio controls>``), à côté de sa transcription. Sans ffmpeg, aucun
extrait n'est découpé : la fiche n'est pas écrite et la raison est donnée.
Environ 20 % des extraits sont présentés une seconde fois (``r-<n>``, autre
ordre) pour mesurer l'accord.

Les segments en échec (``<!-- TRANSCRIPTION ÉCHOUÉE … -->``) n'ont pas
d'énoncé : ils ne sont pas jugés, mais leur nombre est consigné dans la clé et
repris par le rapport.

Fichiers écrits dans le dossier de campagne (``--out``, défaut
``data/albert_eval/<AAAA-MM-JJ>``) : ``fiche_audio.html`` (à juger dans un
navigateur, l'écoute est nécessaire), ``fiche_audio.csv`` (même fiche, sans le
son) et ``audio_items.jsonl`` (clé privée). Aucun chemin local n'y figure.

Exemple (une ligne) ::

    .venv/bin/python scripts/eval/albert/audio_sheets.py --audio-dir sources/X/audio --csv sources/X/output.csv
"""

from __future__ import annotations

import argparse
import base64
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.eval.albert import common as C  # noqa: E402

COLUMNS: Tuple[str, ...] = ("item_id", "document", "debut", "fin", "transcription", "note", "commentaire")
TEXT_COLUMNS: Tuple[str, ...] = ("transcription", "commentaire")
SHEET_STEM = "fiche_audio"
ITEMS_FILE = "audio_items.jsonl"
DEFAULT_CLIPS_PER_FILE = 20
DEFAULT_MAX_CLIP_SECONDS = 30.0
CLIP_PADDING_SECONDS = 0.25
WARN_HTML_MB = 40
DEFAULT_MAX_HTML_MB = 120
SECONDS_OVERHEAD = 10.0
LISTEN_FACTOR = 1.5
"""Durée estimée d'un élément : 1,5 × la durée de l'extrait (écoute et réécoute) + 10 s."""
FFMPEG_MISSING = "ffmpeg introuvable : aucun extrait ne peut être découpé (installer ffmpeg puis relancer)"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".webm", ".mp4", ".aac", ".wma")


class ClipError(Exception):
    """Découpage d'un extrait impossible (message français, sans chemin local)."""


@dataclass
class Utterance:
    """Énoncé tiré pour la fiche.

    Attributes:
        document: nom de l'enregistrement.
        start: début absolu (secondes).
        end: fin absolue (secondes).
        text: transcription de l'énoncé.
        segment: numéro du segment d'origine.
        provider: fournisseur de la transcription.
    """

    document: str
    start: float
    end: float
    text: str
    segment: int
    provider: str


@dataclass
class Recording:
    """Enregistrement retenu.

    Attributes:
        name: nom de fichier affiché (unique dans la fiche).
        path: chemin local (jamais écrit dans les fiches).
        provider: ``texteocr_provider`` de la ligne.
        segments_total: segments du relevé.
        segments_failed: segments en échec.
        pool: énoncés éligibles.
        sampled: énoncés tirés.
        too_long: énoncés écartés car plus longs que l'extrait maximal.
    """

    name: str
    path: Path
    provider: str
    segments_total: int = 0
    segments_failed: int = 0
    pool: List[Utterance] = field(default_factory=list)
    sampled: List[Utterance] = field(default_factory=list)
    too_long: int = 0


# ---------------------------------------------------------------------------
# Découpage (point d'injection des tests)
# ---------------------------------------------------------------------------
def ffmpeg_path() -> Optional[str]:
    """Chemin de ``ffmpeg`` dans le ``PATH``, ou ``None``.

    Returns:
        Le chemin de l'exécutable, ou ``None`` s'il est absent.
    """
    return shutil.which("ffmpeg")


def cut_clip(source: Path, start: float, duration: float) -> Optional[Tuple[bytes, str]]:
    """Découpe un extrait audio par ffmpeg (mono, 16 kHz, mp3 32 kb/s).

    Args:
        source: enregistrement.
        start: début (secondes).
        duration: durée (secondes).

    Returns:
        ``(octets, type MIME)``, ou ``None`` si ffmpeg est absent.

    Raises:
        ClipError: ffmpeg en échec ou extrait vide.
    """
    exe = ffmpeg_path()
    if exe is None:
        return None
    with tempfile.TemporaryDirectory(prefix="ragpy-eval-audio-") as tmp:
        target = Path(tmp) / "extrait.mp3"
        args = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{max(start, 0.0):.3f}",
                "-t", f"{max(duration, 0.1):.3f}", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                "-b:a", "32k", "-f", "mp3", str(target)]
        try:
            done = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClipError(f"ffmpeg n'a pas pu découper l'extrait ({type(exc).__name__})") from None
        if done.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            raise ClipError(f"échec du découpage ffmpeg (code {done.returncode})")
        return target.read_bytes(), "audio/mpeg"


# ---------------------------------------------------------------------------
# Lecture de la session
# ---------------------------------------------------------------------------
def hms(seconds: float) -> str:
    """Horodatage ``hh:mm:ss.d`` (dixième de seconde).

    Args:
        seconds: position en secondes.

    Returns:
        Le libellé.
    """
    tenths = int(round(max(seconds, 0.0) * 10))
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    return f"{hours:02d}:{minutes:02d}:{rest // 10:02d}.{rest % 10}"


def default_segments_path(csv_path: Path) -> Path:
    """Relevé des segments écrit par ``rad_audio.py`` à côté du CSV.

    Args:
        csv_path: ``output.csv`` de la session.

    Returns:
        ``<base>_audio_segments.json``.
    """
    return csv_path.with_name(csv_path.stem + "_audio_segments.json")


def _find_audio(root: Path, name: str, index: Dict[str, List[Path]]) -> Tuple[Optional[Path], Optional[str]]:
    """Enregistrement d'un nom de fichier dans le dossier (recherche récursive, homonymes signalés)."""
    direct = root / name
    if direct.is_file():
        return direct, None
    if not index:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                index.setdefault(path.name.lower(), []).append(path)
        index.setdefault("", [])
    matches = index.get(name.lower(), [])
    if len(matches) > 1:
        return matches[0], f"{name} : {len(matches)} fichiers homonymes, le premier est retenu"
    return (matches[0], None) if matches else (None, None)


def load_recordings(csv_path: Path, segments_path: Path, audio_dir: Path, *, providers: Sequence[str],
                    clips_per_file: int, max_clip_seconds: float, seed: int) -> Tuple[List[Recording], List[str]]:
    """Lit la session audio et tire les énoncés à juger.

    Args:
        csv_path: ``output.csv`` de la session.
        segments_path: ``<base>_audio_segments.json``.
        audio_dir: dossier des enregistrements.
        providers: fournisseurs gardés (vide : tous).
        clips_per_file: énoncés tirés par enregistrement.
        max_clip_seconds: durée maximale d'un extrait.
        seed: graine.

    Returns:
        ``(enregistrements, avertissements)`` (noms de fichier seulement).

    Raises:
        C.EvalError: CSV ou relevé illisible, colonne obligatoire absente.
    """
    import json

    rows, warnings = C.read_csv_rows(csv_path)
    if not rows:
        raise C.EvalError(f"{csv_path.name} : aucune ligne")
    if "filename" not in rows[0][1] and "path" not in rows[0][1]:
        raise C.EvalError(f"{csv_path.name} : colonne filename ou path absente")
    try:
        report_data = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    except OSError:
        raise C.EvalError(f"relevé des segments introuvable : {Path(segments_path).name} (écrit par rad_audio.py "
                          "à côté du CSV ; sinon --segments)") from None
    except json.JSONDecodeError as exc:
        raise C.EvalError(f"{Path(segments_path).name} : JSON invalide ({exc.msg})") from None
    by_name = {str(f.get("filename") or ""): f for f in report_data.get("files") or [] if isinstance(f, Mapping)}
    by_key = {str(f.get("itemKey") or ""): f for f in report_data.get("files") or [] if isinstance(f, Mapping)}
    wanted = {p.strip().lower() for p in providers if p.strip()}
    index: Dict[str, List[Path]] = {}
    recordings: List[Recording] = []
    names: Dict[str, int] = {}
    for line, row in rows:
        provider = row.get("texteocr_provider", "")
        if wanted and provider.lower() not in wanted:
            continue
        filename = re.split(r"[\\/]", row.get("filename", "").strip() or row.get("path", "").strip())[-1]
        if not filename:
            warnings.append(f"{csv_path.name} ligne {line} : ni filename ni path, ligne ignorée")
            continue
        entry = by_key.get(row.get("itemkey", "")) if row.get("itemkey") else None
        entry = entry or by_name.get(filename)
        if entry is None:
            warnings.append(f"{filename} : absent du relevé des segments, ignoré")
            continue
        path, note = _find_audio(audio_dir, filename, index)
        if note:
            warnings.append(note)
        if path is None:
            warnings.append(f"{filename} : enregistrement introuvable dans le dossier audio, ignoré")
            continue
        names[filename] = names.get(filename, 0) + 1
        name = filename if names[filename] == 1 else f"{filename} ({names[filename]})"
        rec = Recording(name=name, path=path, provider=provider)
        for segment in entry.get("segments") or []:
            if not isinstance(segment, Mapping):
                continue
            rec.segments_total += 1
            if segment.get("status") != "ok":
                rec.segments_failed += 1
                continue
            for utt in segment.get("utterances") or []:
                try:
                    start, end = float(utt["start"]), float(utt["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = str(utt.get("text") or "").strip()
                if not text or end <= start:
                    continue
                if end - start > max_clip_seconds:
                    rec.too_long += 1
                    continue
                rec.pool.append(Utterance(name, start, end, text, int(segment.get("index") or 0), provider))
        rec.pool.sort(key=lambda u: (u.start, u.end))
        if not rec.pool:
            warnings.append(f"{name} : aucun énoncé exploitable (segments réussis sans texte, ou énoncés trop longs)")
            continue
        count = min(clips_per_file, len(rec.pool))
        rec.sampled = sorted(random.Random(f"{seed}:audio:{name}").sample(rec.pool, count),
                             key=lambda u: (u.start, u.end))
        if rec.too_long:
            warnings.append(f"{name} : {rec.too_long} énoncé(s) de plus de {max_clip_seconds:g} s écarté(s)")
        recordings.append(rec)
    return recordings, [C.scrub_paths(w, [audio_dir]) for w in warnings]


def clip_window(utt: Utterance, max_clip_seconds: float) -> Tuple[float, float]:
    """Fenêtre de l'extrait : l'énoncé avec une petite marge, sans dépasser la durée maximale.

    Args:
        utt: énoncé.
        max_clip_seconds: durée maximale.

    Returns:
        ``(début, durée)`` en secondes.
    """
    start = max(utt.start - CLIP_PADDING_SECONDS, 0.0)
    end = min(utt.end + CLIP_PADDING_SECONDS, start + max_clip_seconds)
    return start, max(end - start, 0.1)


def cut_clips(recordings: Sequence[Recording], *, max_clip_seconds: float, max_bytes: int
              ) -> Tuple[Dict[Tuple[str, float, float], str], List[str], bool]:
    """Découpe les extraits tirés, dans un budget de taille.

    Args:
        recordings: enregistrements et énoncés tirés.
        max_clip_seconds: durée maximale d'un extrait.
        max_bytes: taille totale maximale des extraits encodés.

    Returns:
        ``(URI data: par (document, début, fin), avertissements, ffmpeg absent)``.
    """
    clips: Dict[Tuple[str, float, float], str] = {}
    warnings: List[str] = []
    used = 0
    failed = over = 0
    for rec in recordings:
        for utt in rec.sampled:
            start, duration = clip_window(utt, max_clip_seconds)
            try:
                result = cut_clip(rec.path, start, duration)
            except ClipError as exc:
                failed += 1
                warnings.append(f"{rec.name} {hms(utt.start)} : {exc}")
                continue
            if result is None:
                return {}, [FFMPEG_MISSING], True
            data, mime = result
            uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
            if used + len(uri) > max_bytes:
                over += 1
                continue
            used += len(uri)
            clips[(utt.document, utt.start, utt.end)] = uri
    if failed:
        warnings.append(f"{failed} extrait(s) non découpé(s) : éléments retirés de la fiche")
    if over:
        warnings.append(f"taille maximale atteinte : {over} extrait(s) retiré(s) ; réduire --clips-per-file ou "
                        "relever --max-html-mb")
    return clips, warnings, False


# ---------------------------------------------------------------------------
# Fiche
# ---------------------------------------------------------------------------
def build_sheet(recordings: Sequence[Recording], clips: Mapping[Tuple[str, float, float], str], *, seed: int,
                second_ratio: float, audio_dir: Path, max_clip_seconds: float) -> Dict[str, Any]:
    """Construit la fiche : lignes CSV, éléments HTML (extrait + transcription), clé privée.

    Args:
        recordings: enregistrements et énoncés tirés.
        clips: extraits découpés, par (document, début, fin) ; un énoncé sans extrait est retiré.
        seed: graine.
        second_ratio: part des extraits rejugés.
        audio_dir: dossier des enregistrements (retiré de tout texte écrit).
        max_clip_seconds: durée maximale d'un extrait (consignée dans la clé).

    Returns:
        ``{rows, items, key_records, audio, sheet_id, n_first, n_second, seconds}``.
    """
    entries = [u for rec in recordings for u in rec.sampled if (u.document, u.start, u.end) in clips]
    entries = C.seeded_shuffle(sorted(entries, key=lambda u: (u.document, u.start, u.end)), seed, "audio:extraits")
    clip_ids: Dict[Tuple[str, float, float], str] = {}
    for n, key in enumerate(sorted(clips), 1):
        clip_ids[key] = f"clip-{n:0{C.id_width(len(clips))}d}"
    audio = {clip_ids[key]: uri for key, uri in clips.items()}
    rows: List[Dict[str, str]] = []
    items: List[Dict[str, Any]] = []
    key_records: List[Dict[str, Any]] = []
    for rec in recordings:
        key_records.append({"record": "document", "document": rec.name, "provider": rec.provider,
                            "segments_total": rec.segments_total, "segments_failed": rec.segments_failed,
                            "utterances_eligible": len(rec.pool), "utterances_too_long": rec.too_long,
                            "sampled": len(rec.sampled)})
    seconds = 0.0

    def add(item_id: str, utt: Utterance, section: str) -> None:
        """Ajoute une ligne CSV et l'élément HTML correspondant."""
        nonlocal seconds
        text = C.scrub_paths(utt.text, [audio_dir])
        row = {"item_id": item_id, "document": utt.document, "debut": hms(utt.start), "fin": hms(utt.end),
               "transcription": text, "note": "", "commentaire": ""}
        rows.append(row)
        items.append({"id": item_id, "section": section, "input": "note",
                      "row": {c: C.csv_cell(row[c], text=c in TEXT_COLUMNS) for c in COLUMNS},
                      "blocks": [{"label": "Enregistrement",
                                  "text": f"{utt.document} — de {row['debut']} à {row['fin']}", "cls": "source"},
                                 {"label": "Transcription", "text": text, "cls": "transcription"}],
                      "audio": clip_ids[(utt.document, utt.start, utt.end)]})
        seconds += LISTEN_FACTOR * min(utt.end - utt.start, max_clip_seconds) + SECONDS_OVERHEAD

    width = C.id_width(len(entries))
    by_id: Dict[str, Utterance] = {}
    for n, utt in enumerate(entries, 1):
        item_id = f"a-{n:0{width}d}"
        by_id[item_id] = utt
        add(item_id, utt, "Premier jugement")
        key_records.append({"record": "item", "item_id": item_id, "kind": "first", "document": utt.document,
                            "debut": hms(utt.start), "start": utt.start, "end": utt.end, "segment": utt.segment,
                            "provider": utt.provider})
    second = C.second_sample(list(by_id), second_ratio, seed, "audio")
    width_r = C.id_width(len(second))
    for n, original in enumerate(second, 1):
        utt = by_id[original]
        item_id = f"r-{n:0{width_r}d}"
        add(item_id, utt, "Second jugement")
        key_records.append({"record": "item", "item_id": item_id, "kind": "second", "of": original,
                            "document": utt.document, "debut": hms(utt.start), "start": utt.start, "end": utt.end,
                            "segment": utt.segment, "provider": utt.provider})
    sheet_id = C.sheet_fingerprint(SHEET_STEM, [seed, [(r["item_id"], r["document"], r["debut"], r["transcription"])
                                                       for r in rows]])
    meta = {"record": "meta", "sheet": "audio", "sheet_id": sheet_id, "seed": seed, "second_ratio": second_ratio,
            "max_clip_seconds": max_clip_seconds, "providers": sorted({r.provider for r in recordings}),
            "columns": list(COLUMNS)}
    return {"rows": rows, "items": items, "key_records": [meta, *key_records], "audio": audio,
            "sheet_id": sheet_id, "n_first": len(entries), "n_second": len(second), "seconds": seconds}


INTRO = (
    "Fiche de jugement de la fidélité des transcriptions audio : écoutez chaque extrait et comparez-le à sa "
    "transcription. Les extraits sont présentés dans un ordre aléatoire.",
    "Les éléments « Second jugement » (r-…) reprennent environ 20 % des extraits dans un autre ordre : "
    "jugez-les sans revenir au premier jugement, de préférence à un autre moment.",
)
RULES = (
    "Jugez les mots : ponctuation, majuscules et hésitations (euh, ben) ne comptent pas.",
    "Un mot inventé, un nom propre déformé ou une phrase manquante est une erreur, même si la suite est juste.",
    "L'extrait déborde un peu avant et après l'énoncé : ignorez les mots coupés aux bords.",
    "En cas d'hésitation entre deux notes, prenez la plus basse et notez la raison en commentaire.",
)


def write_outputs(out_dir: Path, sheet: Mapping[str, Any], *, seed: int, force: bool) -> Dict[str, Path]:
    """Écrit la fiche (CSV, HTML) et la clé.

    Args:
        out_dir: dossier de campagne.
        sheet: résultat de ``build_sheet``.
        seed: graine.
        force: écraser une fiche déjà remplie (copie gardée).

    Returns:
        Les chemins écrits, par nom.

    Raises:
        C.EvalError: fiche déjà remplie, ou export rempli d'une autre fiche, sans ``force``.
    """
    out_dir = Path(out_dir)
    csv_path, html_path, items_path = (out_dir / f"{SHEET_STEM}.csv", out_dir / f"{SHEET_STEM}.html",
                                       out_dir / ITEMS_FILE)
    filled_export = out_dir / f"{SHEET_STEM}_rempli.csv"
    previous_id = None
    if items_path.exists():
        try:
            previous_id = next((r.get("sheet_id") for r in C.read_jsonl(items_path) if r.get("record") == "meta"),
                               None)
        except C.EvalError:
            previous_id = None
    stale = filled_export.exists() and previous_id != sheet["sheet_id"]
    if stale and not force:
        raise C.EvalError(f"{filled_export.name} est l'export d'une autre fiche : rien n'a été écrit. "
                          "Relancer avec --force pour le ranger dans sauvegardes/.")
    backups = C.guard_overwrite([csv_path], force)
    if stale:
        backups.append(C.backup_file(filled_export))
        filled_export.unlink()
    written = {"csv": C.write_csv(csv_path, sheet["rows"], COLUMNS, TEXT_COLUMNS)}
    written["html"] = C.write_text(html_path, C.render_sheet_html(
        title=f"Transcription audio — campagne {out_dir.name}", intro=INTRO, rules=RULES,
        sheet_id=sheet["sheet_id"], seed=seed, file_stem=SHEET_STEM, columns=COLUMNS, items=sheet["items"],
        scale=C.AUDIO_SCALE, scale_title="Échelle de fidélité de la transcription à l'extrait écouté :",
        audio=sheet["audio"]))
    written["items"] = C.write_jsonl(items_path, sheet["key_records"])
    for path in backups:
        print(f"copie de sauvegarde : {C.display_path(path)}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée de la ligne de commande.

    Args:
        argv: arguments (défaut : ``sys.argv[1:]``).

    Returns:
        0 si la fiche est écrite, 2 en cas d'entrée invalide ou d'outil manquant (ffmpeg).
    """
    parser = argparse.ArgumentParser(description="Fiche de fidélité des transcriptions audio (Whisper).")
    parser.add_argument("--audio-dir", required=True, help="dossier des enregistrements de la session")
    parser.add_argument("--csv", required=True, help="output.csv écrit par rad_audio.py")
    parser.add_argument("--segments", help="relevé des segments (défaut : <base du CSV>_audio_segments.json)")
    parser.add_argument("--out", help="dossier de campagne (défaut : data/albert_eval/<date du jour>)")
    parser.add_argument("--clips-per-file", type=int, default=DEFAULT_CLIPS_PER_FILE)
    parser.add_argument("--max-clip-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS)
    parser.add_argument("--provider", default="", help="fournisseurs gardés, séparés par des virgules (défaut : tous)")
    parser.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    parser.add_argument("--second-ratio", type=float, default=C.DEFAULT_SECOND_RATIO)
    parser.add_argument("--max-html-mb", type=float, default=DEFAULT_MAX_HTML_MB,
                        help="taille maximale des extraits de la page HTML (Mo)")
    parser.add_argument("--force", action="store_true", help="écraser une fiche déjà remplie (copie gardée)")
    args = parser.parse_args(argv)
    try:
        if args.clips_per_file < 1:
            raise C.EvalError("--clips-per-file doit être au moins 1")
        if not 1 <= args.max_clip_seconds <= 120:
            raise C.EvalError("--max-clip-seconds doit être compris entre 1 et 120")
        if not 0 <= args.second_ratio < 1:
            raise C.EvalError("--second-ratio doit être dans [0, 1[")
        audio_dir, csv_path = Path(args.audio_dir), Path(args.csv)
        if not audio_dir.is_dir():
            raise C.EvalError(f"dossier audio introuvable : {audio_dir.name}")
        segments_path = Path(args.segments) if args.segments else default_segments_path(csv_path)
        out_dir = Path(args.out) if args.out else C.default_out_dir()
        recordings, warnings = load_recordings(csv_path, segments_path, audio_dir, providers=args.provider.split(","),
                                               clips_per_file=args.clips_per_file,
                                               max_clip_seconds=args.max_clip_seconds, seed=args.seed)
        if not recordings:
            for message in warnings:
                print(f"attention : {message}", file=sys.stderr)
            raise C.EvalError("aucun enregistrement utilisable (voir les avertissements)")
        clips, more, missing_ffmpeg = cut_clips(recordings, max_clip_seconds=args.max_clip_seconds,
                                                max_bytes=int(args.max_html_mb * 1024 * 1024))
        if missing_ffmpeg:
            raise C.EvalError(FFMPEG_MISSING + " ; aucune fiche n'a été écrite")
        warnings += [C.scrub_paths(m, [audio_dir]) for m in more]
        if not clips:
            raise C.EvalError("aucun extrait découpé : aucune fiche n'a été écrite (voir les avertissements)")
        sheet = build_sheet(recordings, clips, seed=args.seed, second_ratio=args.second_ratio, audio_dir=audio_dir,
                            max_clip_seconds=args.max_clip_seconds)
        written = write_outputs(out_dir, sheet, seed=args.seed, force=args.force)
    except C.EvalError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    for message in warnings:
        print(f"attention : {message}")
    size_mb = written["html"].stat().st_size / (1024 * 1024)
    if size_mb > WARN_HTML_MB:
        print(f"attention : fiche HTML de {size_mb:.0f} Mo (au-delà de {WARN_HTML_MB} Mo, un navigateur peut "
              "ralentir) : réduire --clips-per-file")
    where = C.display_path(out_dir)
    failed = sum(r.segments_failed for r in recordings)
    print(f"fiche audio : {len(recordings)} enregistrements, {sheet['n_first']} extraits à noter, "
          f"{sheet['n_second']} en second jugement (r-…) ; {failed} segment(s) en échec non jugé(s) ; "
          f"page HTML de {size_mb:.1f} Mo")
    formula = f"Σ ({LISTEN_FACTOR:g} × durée de l'extrait + {SECONDS_OVERHEAD:g} s)".replace(".", ",")
    print(f"durée estimée : {formula} ≈ {C.duration_label(sheet['seconds'])}")
    print(f"à juger : {where}/{SHEET_STEM}.html (l'écoute est nécessaire)")
    print(f"ne pas ouvrir avant d'avoir jugé : {where}/{ITEMS_FILE}")
    print(f"ensuite : .venv/bin/python scripts/eval/albert/metrics.py --dir {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
