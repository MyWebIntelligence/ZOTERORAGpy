"""Transcription d'enregistrements audio par Albert (Whisper) vers le ``output.csv`` du pipeline.

Étape CLI parallèle à ``rad_dataframe.py`` : elle transforme des enregistrements
(entretiens, séminaires) en un CSV de même forme que celui de l'OCR, pour que la
suite du pipeline (``rad_chunk.py`` → embeddings → base vectorielle ou
collections Albert) tourne sans changement.

Usage (une ligne) ::

    ALBERT_ENABLED=1 ALBERT_AUDIO_ENABLED=1 .venv/bin/python scripts/rad_audio.py --input sources/X/audio --output sources/X/output.csv

Interrupteurs : ``ALBERT_ENABLED=1`` **et** ``ALBERT_AUDIO_ENABLED=1``, plus la
clé ``ALBERT_API_KEY`` (environnement, ou ``.env`` chargé par
``rad_env.load_dotenv_guarded()``) ; sinon refus en code 2. La clé n'est jamais
affichée.

Déroulement, fichier par fichier :

* empreinte SHA-256 du fichier (``itemKey = audio-<sha16>``, stable) ;
* un fichier mp3 ou wav de 20 Mo au plus et pas plus long que
  ``ALBERT_AUDIO_SEGMENT_SECONDS`` est envoyé tel quel ; sinon ffmpeg le
  convertit et le découpe (mono, 16 kHz, mp3 48 kb/s, segments de
  ``ALBERT_AUDIO_SEGMENT_SECONDS`` secondes, plafonnés à ``MAX_SEGMENT_SECONDS``)
  dans un dossier temporaire supprimé en fin de fichier ;
* chaque segment est transcrit par ``AlbertClient.transcribe`` (nom de fichier
  neutre ``audio-<sha16>-segNNNN.mp3`` : le nom local ne part jamais) ;
* point de reprise par segment, écrit atomiquement sous
  ``<dossier de sortie>/albert_audio_checkpoints/<sha16 fichier>/<sha16 réglages>/seg_NNNN.json``
  (réglages = modèle, langue, amorce, durée des segments, version) : une
  relance réutilise les segments validés sans les renvoyer ;
* ``texteocr`` = pour chaque segment, ``<!-- Segment N (hh:mm:ss–hh:mm:ss) -->``
  puis le texte ; un segment en échec reçoit
  ``<!-- TRANSCRIPTION ÉCHOUÉE (albert_whisper) : raison -->`` et la ligne est
  marquée partielle (jamais de succès silencieux).

Sorties, à côté du CSV : ``<base>_errors.json`` (même structure que celui de
``rad_dataframe``), ``<base>_audio_segments.json`` (horodatages absolus des
énoncés, décalés de l'offset de leur segment, croissants) et
``albert_usage.jsonl`` (journal d'usage Albert, si ``ALBERT_USAGE_LOG=1``).
Aucun chemin local n'est écrit dans le CSV (``filename`` et ``path`` = nom du
fichier seul).

Lignes de progression (stdout, vidées à chaque ligne) : ``Fichier i/n : <nom>``
puis ``  Segment j/m : …``. Une erreur de compte ou de quota Albert arrête tout
le traitement, affiche la ligne ``Albert abort: kind=<kind> reason=<reason>[
credential_required=albert_api_key]`` (même contrat que ``rad_chunk.py``, lu
par les routes) et sort en code 2.

Codes de sortie : 0 (au moins un fichier transcrit), 1 (aucun texte produit),
2 (refus : Albert ou l'audio désactivé, clé absente, entrée invalide ; ou arrêt
sur erreur de compte ou de quota).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import Counter
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Chargement du .env gardé (RAGPY_DOTENV_DENY : secrets non rechargés pour un
# sous-processus non-admin). Import double : paquet (app, Celery, tests) ou CLI.
try:
    from scripts.rad_env import load_dotenv_guarded
except ImportError:
    from rad_env import load_dotenv_guarded

# Socle Albert : à l'import, seul le paquet léger (``config`` et ``errors``,
# stdlib) est chargé, depuis une SEULE racine (``scripts.rad_albert`` ou
# ``rad_albert``) ; le client (httpx) et le ledger sont importés à la demande
# depuis la même racine : les classes d'erreur interceptées ici sont donc
# celles que lève le client.
try:
    from scripts import rad_albert as _rad_albert
except ImportError:
    import rad_albert as _rad_albert

logger = logging.getLogger(__name__)


def _albert_module(name: str) -> Any:
    """Importe le sous-module ``name`` du socle Albert depuis la racine de ``_rad_albert``.

    Args:
        name: nom du sous-module (``client``, ``usage``, ``config``…).

    Returns:
        Le module importé (``scripts.rad_albert.<name>`` ou ``rad_albert.<name>``).
    """
    return importlib.import_module(f"{_rad_albert.__name__}.{name}")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROVIDER = "albert_whisper"
"""Valeur de ``texteocr_provider`` des lignes produites."""

AUDIO_INPUT_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".webm", ".mp4", ".mkv",
)
"""Extensions reconnues dans ``--input`` (recherche non récursive)."""

DIRECT_EXTENSIONS = (".mp3", ".wav")
"""Formats acceptés tels quels par la transcription Albert (les autres passent par ffmpeg)."""

MAX_AUDIO_BYTES = int(_albert_module("config").AUDIO_MAX_MB * 1024 * 1024)
"""Taille maximale d'un fichier envoyé à Albert (20 Mo)."""

MAX_SEGMENT_SECONDS = 3000
"""Durée maximale d'un segment : 3 000 s à 48 kb/s ≈ 18 Mo, sous la limite de 20 Mo."""

SEGMENT_SAMPLE_RATE = 16000
SEGMENT_BITRATE = "48k"
CHECKPOINT_DIRNAME = "albert_audio_checkpoints"
CHECKPOINT_VERSION = 1
"""Version du format des points de reprise (entre dans l'empreinte des réglages)."""

PLAN_FILENAME = "plan.json"
PROMPT_MAX_CHARS = 1000
FFPROBE_TIMEOUT_SECONDS = 60
FFMPEG_MIN_TIMEOUT_SECONDS = 600
FFMPEG_MAX_TIMEOUT_SECONDS = 21600
FFMPEG_MISSING_MESSAGE = "ffmpeg introuvable : convertir en mp3 ou wav de moins de 20 Mo"
ERROR_DETAIL_MAX_CHARS = 300

ALBERT_ABORT_MARKER = "Albert abort:"
"""Préfixe de la ligne d'arrêt Albert (contrat partagé avec ``rad_chunk.py`` et les routes)."""
_ALBERT_MODEL_REASONS = ("not_found", "validation")

MISSING_KEY_MESSAGE = (
    "Clé API Albert (DINUM) requise (ALBERT_API_KEY). Configurez-la dans Paramètres > Mes Identifiants."
)

CSV_COLUMNS = (
    "itemKey",
    "title",
    "authors",
    "date",
    "filename",
    "path",
    "texteocr",
    "texteocr_provider",
    "texteocr_partial",
    "texteocr_pages_done",
    "texteocr_pages_total",
    "language",
    "duration_s",
    "segments",
)
"""Colonnes du CSV produit (``texteocr`` et métadonnées recopiées dans chaque chunk)."""

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$|^[a-z]{4,20}$")
_SEGMENT_FILE_RE = re.compile(r"^seg_(\d{4,})\.mp3$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")


# ---------------------------------------------------------------------------
# Erreurs et structures
# ---------------------------------------------------------------------------
class AudioConversionError(RuntimeError):
    """Conversion ou découpage ffmpeg impossible (message français, sans chemin local)."""


class CheckpointPathError(ValueError):
    """Dossier de reprise qui sortirait du dossier de sortie (lien symbolique) : refusé."""


class _RunAbort(Exception):
    """Arrêt de tout le traitement sur une erreur de compte ou de quota Albert.

    Attributes:
        error: l'``AlbertAuthError`` (ou ``AlbertQuotaExhausted``) d'origine.
    """

    def __init__(self, error: BaseException) -> None:
        """Garde l'erreur Albert d'origine.

        Args:
            error: erreur de compte ou de quota.
        """
        super().__init__(str(error))
        self.error = error


@dataclass
class _Segment:
    """Segment à transcrire.

    Attributes:
        index: numéro du segment, à partir de 1.
        source: fichier envoyé (original ou segment ffmpeg).
        upload_name: nom neutre transmis à Albert (``.mp3`` ou ``.wav``).
        start: début du segment dans l'enregistrement (secondes).
        duration: durée connue du segment (secondes), ou ``None``.
    """

    index: int
    source: str
    upload_name: str
    start: float
    duration: Optional[float]


@dataclass
class _SegmentResult:
    """Résultat d'un segment (transcrit, repris ou en échec).

    Attributes:
        index: numéro du segment, à partir de 1.
        start: début du segment (secondes, offset cumulé).
        duration: durée du segment (plan, sinon réponse d'Albert), ou ``None``.
        ok: vrai si la transcription a réussi.
        text: texte transcrit (vide en cas d'échec).
        utterances: énoncés, horodatages **relatifs** au début du segment.
        language: langue renvoyée par le modèle.
        error: raison de l'échec (sans chemin ni clé).
        reused: vrai si le segment vient d'un point de reprise.
    """

    index: int
    start: float
    duration: Optional[float]
    ok: bool
    text: str = ""
    utterances: List[Dict[str, Any]] = field(default_factory=list)
    language: Optional[str] = None
    error: Optional[str] = None
    reused: bool = False


@dataclass
class _RunContext:
    """État partagé d'un traitement.

    Attributes:
        cfg: configuration Albert (``AlbertConfig``).
        client: client Albert.
        ledger: journal d'usage partagé avec le client.
        output_dir: dossier de sortie (chemin réel).
        model: modèle de transcription (préfixe ``albert/`` retiré).
        language: langue envoyée (``''`` = détection automatique).
        prompt: amorce de vocabulaire (``''`` si aucune).
        segment_seconds: durée effective des segments.
        params: empreinte des réglages (16 caractères hexadécimaux).
        stop: drapeau d'arrêt posé par une erreur de compte.
        seen: empreintes déjà traitées pendant le traitement (doublons).
    """

    cfg: Any
    client: Any
    ledger: Any
    output_dir: str
    model: str
    language: str
    prompt: str
    segment_seconds: int
    params: str
    stop: threading.Event = field(default_factory=threading.Event)
    seen: Dict[str, str] = field(default_factory=dict)


@dataclass
class _FileOutcome:
    """Bilan d'un fichier.

    Attributes:
        filename: nom du fichier (sans dossier).
        title: nom sans extension.
        row: ligne du CSV, ou ``None`` si aucun texte.
        errors: entrées de ``*_errors.json``.
        segments: résultats par segment (fichier transcrit).
        segments_ok: segments transcrits ou repris.
        segments_total: nombre de segments.
        segments_reused: segments repris d'un point de reprise.
        item_key: ``audio-<sha16>``, ou ``''`` si l'empreinte n'a pas pu être calculée.
    """

    filename: str
    title: str
    row: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    segments: List[Tuple[_SegmentResult, float, float]] = field(default_factory=list)
    segments_ok: int = 0
    segments_total: int = 0
    segments_reused: int = 0
    item_key: str = ""


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _say(message: str) -> None:
    """Affiche une ligne sur stdout et la vide aussitôt (progression lue en direct).

    Args:
        message: ligne à afficher (jamais la clé).
    """
    print(message, flush=True)


def _now() -> str:
    """Horodatage local des entrées d'erreur.

    Returns:
        ``AAAA-MM-JJ HH:MM:SS``.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _hms(seconds: Optional[float]) -> str:
    """Formate une durée en ``hh:mm:ss`` (arrondi à la seconde, heures non bornées).

    Args:
        seconds: durée en secondes (``None`` ou négative : 0).

    Returns:
        La durée formatée.
    """
    value = 0.0 if seconds is None or not math.isfinite(seconds) else max(0.0, float(seconds))
    total = int(math.floor(value + 0.5))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _marker_safe(text: Any, limit: int = ERROR_DETAIL_MAX_CHARS) -> str:
    """Texte sûr dans un commentaire HTML : une ligne, sans ``--``, tronqué.

    Args:
        text: texte brut (message d'erreur).
        limit: longueur maximale.

    Returns:
        Le texte nettoyé (jamais ``-->`` ni ``<!--``).
    """
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"-{2,}", "-", cleaned).replace("<", "‹").replace(">", "›")
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned or "raison inconnue"


def _error_reason(exc: BaseException) -> str:
    """Raison lisible d'un échec de segment, sans chemin local ni secret.

    Args:
        exc: exception levée pendant la transcription du segment.

    Returns:
        Un message français court.
    """
    if isinstance(exc, _rad_albert.AlbertError):
        return _marker_safe(str(exc))
    if isinstance(exc, OSError):
        return _marker_safe(f"lecture du segment impossible ({exc.strerror or type(exc).__name__})")
    return _marker_safe(f"erreur inattendue ({type(exc).__name__})")


def _model_name(model: Any) -> str:
    """Nom du modèle sans préfixe ``albert/`` (empreinte des réglages).

    Args:
        model: valeur configurée (``ALBERT_AUDIO_MODEL``).

    Returns:
        Le nom nettoyé.
    """
    name = str(model or "").strip()
    return name[7:].strip() if name[:7].lower() == "albert/" else name


def normalise_language(value: Optional[str], default: str) -> str:
    """Langue de transcription : code ISO 639-1 (``fr``) ou nom anglais, ``auto`` = détection.

    Args:
        value: valeur de ``--language`` (``None`` : défaut de la configuration).
        default: ``ALBERT_AUDIO_LANGUAGE`` lu par ``AlbertConfig`` (``''`` = détection).

    Returns:
        La langue en minuscules, ou ``''`` pour la détection automatique.

    Raises:
        ValueError: valeur non reconnue.
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return ""
    if not _LANGUAGE_RE.match(text):
        raise ValueError(f"langue {value!r} non reconnue (code ISO 639-1 comme fr, ou auto)")
    return text


def params_key(model: str, language: str, prompt: str, segment_seconds: int) -> str:
    """Empreinte des réglages qui déterminent une transcription (dossier de reprise).

    Args:
        model: modèle de transcription.
        language: langue envoyée (``''`` = détection).
        prompt: amorce de vocabulaire.
        segment_seconds: durée effective des segments.

    Returns:
        ``sha256(modèle|langue|amorce|durée|version)[:16]``.
    """
    raw = f"{model}|{language}|{prompt}|{segment_seconds}|{CHECKPOINT_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def file_digest(path: str) -> Tuple[str, int]:
    """Empreinte SHA-256 d'un fichier, lue par blocs (fichiers longs).

    Args:
        path: chemin du fichier.

    Returns:
        ``(empreinte hexadécimale, taille en octets)``.

    Raises:
        OSError: fichier illisible.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


# ---------------------------------------------------------------------------
# Entrées
# ---------------------------------------------------------------------------
def list_audio_files(input_path: str, max_files: Optional[int] = None) -> List[str]:
    """Fichiers audio à traiter : un fichier, ou un dossier parcouru sans récursion.

    Les fichiers cachés (``.``, dont les ``._*`` de macOS) et les extensions non
    reconnues d'un dossier sont ignorés ; l'ordre est alphabétique (casse ignorée).

    Args:
        input_path: fichier ou dossier.
        max_files: nombre maximal de fichiers (``None`` : tous).

    Returns:
        Les chemins retenus.

    Raises:
        FileNotFoundError: entrée introuvable.
        ValueError: fichier unique d'extension non reconnue.
    """
    if os.path.isdir(input_path):
        names = sorted(os.listdir(input_path), key=lambda n: (n.lower(), n))
        files = [
            os.path.join(input_path, name)
            for name in names
            if not name.startswith(".")
            and os.path.splitext(name)[1].lower() in AUDIO_INPUT_EXTENSIONS
            and os.path.isfile(os.path.join(input_path, name))
        ]
    elif os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in AUDIO_INPUT_EXTENSIONS:
            raise ValueError(
                f"extension {ext or '(aucune)'} non reconnue (attendu : {' '.join(AUDIO_INPUT_EXTENSIONS)})"
            )
        files = [input_path]
    else:
        raise FileNotFoundError(input_path)
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    return files


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe (points d'injection des tests)
# ---------------------------------------------------------------------------
def ffmpeg_available() -> bool:
    """Indique si ``ffmpeg`` est disponible dans le ``PATH``.

    Returns:
        Vrai si l'exécutable est trouvé.
    """
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: Sequence[str], *, timeout: float) -> None:
    """Lance ``ffmpeg`` (sans shell, entrée standard fermée, durée bornée).

    Args:
        args: arguments après le nom de l'exécutable.
        timeout: délai maximal (secondes) ; au-delà, ffmpeg est arrêté.

    Raises:
        AudioConversionError: ffmpeg absent, interrompu ou en échec.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioConversionError(FFMPEG_MISSING_MESSAGE)
    try:
        proc = subprocess.run(
            [exe, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise AudioConversionError(f"conversion ffmpeg interrompue après {int(timeout)} s") from None
    except OSError as exc:
        raise AudioConversionError(f"ffmpeg n'a pas pu être lancé ({type(exc).__name__})") from None
    if proc.returncode != 0:
        lines = [line.strip() for line in (proc.stderr or "").splitlines() if line.strip()]
        tail = lines[-1] if lines else "aucun détail"
        raise AudioConversionError(f"échec de la conversion ffmpeg (code {proc.returncode}) : {tail}")


def _probe_duration(path: str) -> Optional[float]:
    """Durée d'un fichier audio ou vidéo lue par ``ffprobe`` (sans shell, durée bornée).

    Args:
        path: chemin du fichier.

    Returns:
        La durée en secondes, ou ``None`` (ffprobe absent, erreur, valeur illisible).
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", os.path.abspath(path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        try:
            value = float(line.strip())
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _wav_duration(path: str) -> Optional[float]:
    """Durée d'un WAV PCM lue par le module ``wave`` (repli sans ffprobe).

    Args:
        path: chemin du fichier ``.wav``.

    Returns:
        La durée en secondes, ou ``None`` si l'en-tête n'est pas lisible.
    """
    try:
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (wave.Error, EOFError, OSError):
        return None
    if rate <= 0:
        return None
    return frames / float(rate)


def media_duration(path: str) -> Optional[float]:
    """Durée d'un enregistrement : ffprobe, sinon en-tête WAV.

    Args:
        path: chemin du fichier.

    Returns:
        La durée en secondes, ou ``None`` si elle est inconnue.
    """
    duration = _probe_duration(path)
    if duration is None and os.path.splitext(path)[1].lower() == ".wav":
        duration = _wav_duration(path)
    return duration


def _ffmpeg_timeout(duration: Optional[float]) -> float:
    """Délai accordé à ffmpeg : au moins 10 min, au plus 6 h, une fois la durée sinon.

    Args:
        duration: durée de l'enregistrement, ou ``None``.

    Returns:
        Le délai en secondes.
    """
    if duration is None:
        return float(FFMPEG_MIN_TIMEOUT_SECONDS)
    return float(min(FFMPEG_MAX_TIMEOUT_SECONDS, max(FFMPEG_MIN_TIMEOUT_SECONDS, duration)))


def split_audio(src: str, tmpdir: str, segment_seconds: int, source_duration: Optional[float]) -> List[str]:
    """Convertit et découpe un enregistrement par ffmpeg (mono, 16 kHz, mp3 48 kb/s).

    Args:
        src: fichier source.
        tmpdir: dossier temporaire de sortie.
        segment_seconds: durée des segments.
        source_duration: durée connue de la source (délai de ffmpeg), ou ``None``.

    Returns:
        Les fichiers ``seg_NNNN.mp3`` produits, dans l'ordre.

    Raises:
        AudioConversionError: échec, ou aucun segment produit (message sans chemin local).
    """
    source = os.path.abspath(src)
    pattern = os.path.join(tmpdir, "seg_%04d.mp3")
    args = [
        "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", source,
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(SEGMENT_SAMPLE_RATE),
        "-c:a", "libmp3lame", "-b:a", SEGMENT_BITRATE,
        "-f", "segment", "-segment_time", str(int(segment_seconds)), "-reset_timestamps", "1",
        pattern,
    ]
    try:
        _run_ffmpeg(args, timeout=_ffmpeg_timeout(source_duration))
    except AudioConversionError as exc:
        message = str(exc).replace(source, os.path.basename(source)).replace(tmpdir, "<temporaire>")
        raise AudioConversionError(_marker_safe(message)) from None
    produced = []
    for name in os.listdir(tmpdir):
        match = _SEGMENT_FILE_RE.match(name)
        if match:
            produced.append((int(match.group(1)), os.path.join(tmpdir, name)))
    if not produced:
        raise AudioConversionError("ffmpeg n'a produit aucun segment (piste audio absente ?)")
    return [path for _number, path in sorted(produced)]


def segment_windows(
    files: Sequence[str], segment_seconds: int, source_duration: Optional[float]
) -> List[Tuple[float, Optional[float]]]:
    """Début et durée de chaque segment découpé.

    Offsets cumulés des durées mesurées par ffprobe quand toutes sont connues,
    sinon ``index × durée des segments`` ; la durée d'un segment non mesuré vaut
    la durée des segments (sauf le dernier : reste de la durée source, ou inconnue).

    Args:
        files: segments produits par ffmpeg, dans l'ordre.
        segment_seconds: durée demandée des segments.
        source_duration: durée de la source, ou ``None``.

    Returns:
        ``[(début, durée ou None), …]``, débuts croissants.
    """
    measured = [_probe_duration(path) for path in files]
    if measured and all(d is not None for d in measured):
        starts, cursor = [], 0.0
        for duration in measured:
            starts.append(cursor)
            cursor += float(duration)
    else:
        starts = [float(i * segment_seconds) for i in range(len(files))]
    windows: List[Tuple[float, Optional[float]]] = []
    last = len(files) - 1
    for i, start in enumerate(starts):
        duration = measured[i]
        if duration is None:
            if i < last:
                duration = float(segment_seconds)
            elif source_duration is not None and source_duration > start:
                duration = float(source_duration) - start
        windows.append((round(start, 3), round(duration, 3) if duration is not None else None))
    return windows


# ---------------------------------------------------------------------------
# Points de reprise (confinés au dossier de sortie, écritures atomiques)
# ---------------------------------------------------------------------------
def _is_within(base: str, target: str) -> bool:
    """Vrai si ``target`` (chemin réel) est ``base`` ou se trouve dessous.

    Args:
        base: dossier de référence (chemin réel).
        target: chemin à contrôler.

    Returns:
        Le résultat du contrôle.
    """
    real = os.path.realpath(target)
    try:
        return os.path.commonpath([base, real]) == base
    except ValueError:
        return False


def checkpoint_dir(output_dir: str, file_sha16: str, params: str) -> str:
    """Dossier de reprise d'un fichier et de ses réglages, confiné au dossier de sortie.

    Args:
        output_dir: dossier de sortie.
        file_sha16: 16 premiers caractères de l'empreinte du fichier.
        params: empreinte des réglages.

    Returns:
        Le chemin réel du dossier (créé au besoin).

    Raises:
        CheckpointPathError: empreinte invalide ou dossier hors du dossier de sortie.
    """
    if not _HEX16_RE.match(file_sha16 or "") or not _HEX16_RE.match(params or ""):
        raise CheckpointPathError("empreinte de reprise invalide")
    base = os.path.realpath(output_dir)
    root = os.path.join(base, CHECKPOINT_DIRNAME)
    message = f"dossier de reprise {CHECKPOINT_DIRNAME} hors du dossier de sortie : refusé"
    if os.path.lexists(root) and not _is_within(base, root):
        raise CheckpointPathError(message)
    target = os.path.join(root, file_sha16, params)
    for partial in (os.path.join(root, file_sha16), target):
        if os.path.lexists(partial) and not _is_within(base, partial):
            raise CheckpointPathError(message)
    os.makedirs(target, exist_ok=True)
    if not _is_within(base, target):
        raise CheckpointPathError(message)
    return os.path.realpath(target)


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Écrit un JSON atomiquement (fichier temporaire du même dossier, puis ``os.replace``).

    Args:
        path: fichier cible.
        payload: contenu sérialisable.

    Raises:
        OSError: écriture impossible (le fichier temporaire est supprimé).
    """
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Optional[Any]:
    """Lit un JSON de reprise ; un fichier illisible est mis de côté en ``.corrupt-<date>``.

    Args:
        path: fichier à lire.

    Returns:
        Le contenu, ou ``None`` (absent ou illisible).
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        corrupt = f"{path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            os.replace(path, corrupt)
            logger.warning("Point de reprise illisible mis de côté : %s", os.path.basename(corrupt))
        except OSError:
            pass
        return None


def _plan_payload(file_sha16: str, params: str, mode: str, segments: Sequence[_Segment],
                  source_duration: Optional[float]) -> Dict[str, Any]:
    """Contenu de ``plan.json`` : découpage retenu pour un fichier et des réglages.

    Args:
        file_sha16: empreinte courte du fichier.
        params: empreinte des réglages.
        mode: ``whole`` (fichier envoyé tel quel) ou ``split`` (découpage ffmpeg).
        segments: segments du plan.
        source_duration: durée de la source, ou ``None``.

    Returns:
        Le dictionnaire à écrire.
    """
    return {
        "version": CHECKPOINT_VERSION,
        "file_sha16": file_sha16,
        "params": params,
        "mode": mode,
        "segment_count": len(segments),
        "source_duration": source_duration,
        "segments": [{"index": s.index, "start": s.start, "duration": s.duration} for s in segments],
    }


def _load_plan(ckpt_dir: str, file_sha16: str, params: str, mode: str) -> Optional[List[_Segment]]:
    """Relit le plan enregistré s'il correspond au fichier, aux réglages et au mode.

    Args:
        ckpt_dir: dossier de reprise.
        file_sha16: empreinte courte du fichier.
        params: empreinte des réglages.
        mode: mode attendu.

    Returns:
        Les segments du plan (sans fichier source), ou ``None``.
    """
    data = _read_json(os.path.join(ckpt_dir, PLAN_FILENAME))
    if not isinstance(data, dict):
        return None
    if (data.get("version"), data.get("file_sha16"), data.get("params"), data.get("mode")) != (
        CHECKPOINT_VERSION, file_sha16, params, mode
    ):
        return None
    raw = data.get("segments")
    if not isinstance(raw, list) or not raw or len(raw) != data.get("segment_count"):
        return None
    segments: List[_Segment] = []
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or item.get("index") != position:
            return None
        start, duration = item.get("start"), item.get("duration")
        if not _is_number(start) or (duration is not None and not _is_number(duration)):
            return None
        segments.append(_Segment(position, "", "", float(start), float(duration) if duration is not None else None))
    return segments


def _is_number(value: Any) -> bool:
    """Vrai pour un nombre fini (booléens exclus).

    Args:
        value: valeur à tester.

    Returns:
        Le résultat du test.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _checkpoint_path(ckpt_dir: str, index: int) -> str:
    """Chemin du point de reprise d'un segment.

    Args:
        ckpt_dir: dossier de reprise.
        index: numéro du segment (à partir de 1).

    Returns:
        ``<ckpt_dir>/seg_NNNN.json``.
    """
    return os.path.join(ckpt_dir, f"seg_{index:04d}.json")


def _load_checkpoint(ckpt_dir: str, segment: _Segment, file_sha16: str, params: str, mode: str,
                     count: int) -> Optional[_SegmentResult]:
    """Relit et valide le point de reprise d'un segment.

    Un point de reprise n'est valide que pour le même fichier, les mêmes
    réglages, le même mode et le même nombre de segments ; seuls les succès
    sont enregistrés (un échec est toujours retenté).

    Args:
        ckpt_dir: dossier de reprise.
        segment: segment du plan courant.
        file_sha16: empreinte courte du fichier.
        params: empreinte des réglages.
        mode: mode du plan courant.
        count: nombre de segments du plan courant.

    Returns:
        Le résultat repris, ou ``None``.
    """
    data = _read_json(_checkpoint_path(ckpt_dir, segment.index))
    if not isinstance(data, dict):
        return None
    expected = (CHECKPOINT_VERSION, file_sha16, params, mode, count, segment.index, "ok")
    found = tuple(data.get(k) for k in ("version", "file_sha16", "params", "mode", "segment_count", "index", "status"))
    if found != expected or not isinstance(data.get("text"), str):
        return None
    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        return None
    clean: List[Dict[str, Any]] = []
    for item in utterances:
        if not isinstance(item, dict) or not _is_number(item.get("start")) or not _is_number(item.get("end")):
            return None
        entry = {"start": float(item["start"]), "end": float(item["end"]), "text": str(item.get("text") or "")}
        if item.get("speaker") is not None:
            entry["speaker"] = str(item["speaker"])
        clean.append(entry)
    duration = segment.duration
    if duration is None and _is_number(data.get("duration")):
        duration = float(data["duration"])
    return _SegmentResult(
        index=segment.index, start=segment.start, duration=duration, ok=True, text=data["text"],
        utterances=clean, language=data.get("language") or None, reused=True,
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def normalise_utterances(raw: Any, window: Optional[float]) -> List[Dict[str, Any]]:
    """Énoncés d'un segment, horodatages relatifs, croissants et bornés au segment.

    Args:
        raw: ``segments`` normalisés par ``AlbertClient.transcribe``.
        window: durée du segment (borne supérieure), ou ``None``.

    Returns:
        ``[{"start", "end", "text"[, "speaker"]}, …]`` : débuts non décroissants,
        fin >= début, valeurs dans ``[0, window]``.
    """
    utterances: List[Dict[str, Any]] = []
    previous = 0.0
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping) or not _is_number(item.get("start")) or not _is_number(item.get("end")):
            continue
        start = max(float(item["start"]), previous, 0.0)
        end = max(float(item["end"]), start)
        if window is not None:
            start, end = min(start, window), min(end, window)
        entry: Dict[str, Any] = {"start": round(start, 3), "end": round(end, 3),
                                 "text": str(item.get("text") or "").strip()}
        if item.get("speaker") is not None:
            entry["speaker"] = str(item["speaker"])
        utterances.append(entry)
        previous = start
    return utterances


def _transcribe_segment(ctx: _RunContext, segment: _Segment, ckpt_dir: str, file_sha16: str, mode: str,
                        count: int) -> Optional[_SegmentResult]:
    """Transcrit un segment et écrit son point de reprise (exécuté dans un fil).

    Args:
        ctx: contexte du traitement.
        segment: segment à transcrire.
        ckpt_dir: dossier de reprise.
        file_sha16: empreinte courte du fichier.
        mode: mode du plan.
        count: nombre de segments du plan.

    Returns:
        Le résultat (succès ou échec), ou ``None`` si l'arrêt a été demandé avant l'envoi.

    Raises:
        AlbertAuthError: erreur de compte ou de quota (le drapeau d'arrêt est posé).
    """
    if ctx.stop.is_set():
        return None
    try:
        with open(segment.source, "rb") as handle:
            data = handle.read()
        response = ctx.client.transcribe(
            data,
            segment.upload_name,
            language=ctx.language,
            prompt=ctx.prompt or None,
            response_format="verbose_json",
        )
    except _rad_albert.AlbertAuthError:
        ctx.stop.set()
        raise
    except Exception as exc:  # noqa: BLE001 - tout autre échec est propre au segment
        return _SegmentResult(index=segment.index, start=segment.start, duration=segment.duration, ok=False,
                              error=_error_reason(exc))
    duration = segment.duration
    if duration is None and _is_number(response.get("duration")):
        duration = float(response["duration"])
    utterances = normalise_utterances(response.get("segments"), duration)
    text = str(response.get("text") or "").strip()
    language = response.get("language") or None
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "file_sha16": file_sha16,
        "params": ctx.params,
        "mode": mode,
        "segment_count": count,
        "index": segment.index,
        "status": "ok",
        "offset": segment.start,
        "duration": duration,
        "text": text,
        "language": language,
        "utterances": utterances,
        "model": response.get("model"),
        "response_model": response.get("response_model"),
        "written_at": _now(),
    }
    try:
        _atomic_write_json(_checkpoint_path(ckpt_dir, segment.index), checkpoint)
    except OSError as exc:
        logger.warning("Point de reprise du segment %d non écrit (%s).", segment.index, type(exc).__name__)
    return _SegmentResult(index=segment.index, start=segment.start, duration=duration, ok=True, text=text,
                          utterances=utterances, language=language)


def _run_segments(ctx: _RunContext, segments: Sequence[_Segment], ckpt_dir: str, file_sha16: str,
                  mode: str) -> List[_SegmentResult]:
    """Reprend les segments validés et transcrit les autres (``ALBERT_AUDIO_CONCURRENCY`` fils).

    Les lignes ``  Segment j/m : …`` sont affichées dans l'ordre des segments.

    Args:
        ctx: contexte du traitement.
        segments: segments du plan courant (avec leur fichier source).
        ckpt_dir: dossier de reprise.
        file_sha16: empreinte courte du fichier.
        mode: mode du plan.

    Returns:
        Les résultats, dans l'ordre des segments.

    Raises:
        _RunAbort: erreur de compte ou de quota (les autres envois ne partent plus).
    """
    count = len(segments)
    reused: Dict[int, _SegmentResult] = {}
    todo: List[_Segment] = []
    for segment in segments:
        cached = _load_checkpoint(ckpt_dir, segment, file_sha16, ctx.params, mode, count)
        if cached is not None:
            reused[segment.index] = cached
        else:
            todo.append(segment)
    results: List[_SegmentResult] = []
    abort: Optional[BaseException] = None
    workers = max(1, int(getattr(ctx.cfg, "audio_concurrency", 1) or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rad-audio") as pool:
        futures = {
            s.index: pool.submit(_transcribe_segment, ctx, s, ckpt_dir, file_sha16, mode, count) for s in todo
        }
        for segment in segments:
            label = f"  Segment {segment.index}/{count}"
            if segment.index in reused:
                result = reused[segment.index]
                results.append(result)
                _say(f"{label} : repris (point de reprise)")
                continue
            try:
                result = futures[segment.index].result()
            except FutureCancelledError:
                continue
            except _rad_albert.AlbertAuthError as exc:
                if abort is None:
                    abort = exc
                    for pending in futures.values():
                        pending.cancel()
                _say(f"{label} : arrêt (erreur de compte Albert)")
                continue
            except Exception as exc:  # noqa: BLE001 - défensif : un fil ne doit jamais casser le fichier
                result = _SegmentResult(index=segment.index, start=segment.start, duration=segment.duration,
                                        ok=False, error=_error_reason(exc))
            if result is None:  # arrêt demandé avant l'envoi : seulement après une erreur de compte
                continue
            results.append(result)
            if result.ok:
                _say(f"{label} : transcrit")
            else:
                _say(f"{label} : échec : {result.error}")
    if abort is not None:
        raise _RunAbort(abort)
    return results


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------
def _segment_body(result: _SegmentResult) -> str:
    """Texte d'un segment réussi ; tours de parole étiquetés par segment si le modèle en fournit.

    Un locuteur n'est jamais identifié d'un segment à l'autre : l'étiquette
    porte le numéro du segment (``[SPEAKER_00 · segment 2]``).

    Args:
        result: résultat du segment.

    Returns:
        Le texte à placer sous le marqueur du segment.
    """
    if not any(u.get("speaker") for u in result.utterances):
        return result.text
    lines: List[str] = []
    current: Optional[str] = None
    for utterance in result.utterances:
        text = utterance.get("text") or ""
        speaker = utterance.get("speaker") or "?"
        if not text:
            continue
        if speaker == current and lines:
            lines[-1] += " " + text
        else:
            lines.append(f"[{speaker} · segment {result.index}] {text}")
            current = speaker
    return "\n".join(lines)


def _windows(results: Sequence[_SegmentResult]) -> List[Tuple[float, float]]:
    """Fenêtres absolues ``(début, fin)`` des segments, croissantes.

    Fin d'un segment = début du suivant ; pour le dernier : début + durée
    (plan, sinon réponse d'Albert), sinon fin du dernier énoncé.

    Args:
        results: résultats ordonnés.

    Returns:
        Les fenêtres, dans l'ordre.
    """
    windows: List[Tuple[float, float]] = []
    for position, result in enumerate(results):
        start = result.start if not windows else max(result.start, windows[-1][1])
        if position + 1 < len(results):
            end = results[position + 1].start
        elif result.duration is not None:
            end = result.start + result.duration
        elif result.utterances:
            end = result.start + max(u["end"] for u in result.utterances)
        else:
            end = start
        windows.append((start, max(end, start)))
    return windows


def build_texteocr(results: Sequence[_SegmentResult]) -> str:
    """Texte du champ ``texteocr`` : marqueur horodaté par segment, puis texte ou échec.

    Args:
        results: résultats ordonnés.

    Returns:
        ``<!-- Segment N (hh:mm:ss–hh:mm:ss) -->`` suivi du texte, ou du marqueur
        ``<!-- TRANSCRIPTION ÉCHOUÉE (albert_whisper) : raison -->``, segments séparés
        par une ligne vide.
    """
    blocks: List[str] = []
    for result, (start, end) in zip(results, _windows(results)):
        marker = f"<!-- Segment {result.index} ({_hms(start)}–{_hms(end)}) -->"
        if result.ok:
            body = _segment_body(result)
        else:
            body = f"<!-- TRANSCRIPTION ÉCHOUÉE ({PROVIDER}) : {_marker_safe(result.error)} -->"
        blocks.append(f"{marker}\n{body}" if body else marker)
    return "\n\n".join(blocks)


def _absolute_utterances(result: _SegmentResult, start: float) -> List[Dict[str, Any]]:
    """Énoncés d'un segment décalés de son offset (horodatages absolus).

    Args:
        result: résultat du segment.
        start: début absolu du segment.

    Returns:
        Les énoncés avec ``start``/``end`` absolus.
    """
    shifted = []
    for utterance in result.utterances:
        entry = dict(utterance)
        entry["start"] = round(start + float(utterance["start"]), 3)
        entry["end"] = round(start + float(utterance["end"]), 3)
        shifted.append(entry)
    return shifted


# ---------------------------------------------------------------------------
# Traitement d'un fichier
# ---------------------------------------------------------------------------
def _error_entry(outcome: _FileOutcome, error_type: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Entrée de ``*_errors.json`` (structure de ``rad_dataframe``, sans chemin local).

    Args:
        outcome: bilan du fichier.
        error_type: type d'erreur (``FFMPEG_MISSING``, ``TRANSCRIPTION_PARTIAL``…).
        message: message français.
        **extra: champs complémentaires.

    Returns:
        L'entrée.
    """
    entry: Dict[str, Any] = {
        "itemKey": outcome.item_key,
        "title": outcome.title,
        "filename": outcome.filename,
        "error_type": error_type,
        "error_message": message,
    }
    entry.update(extra)
    entry["timestamp"] = _now()
    return entry


def _fail(outcome: _FileOutcome, error_type: str, message: str, **extra: Any) -> _FileOutcome:
    """Enregistre un échec de fichier et l'affiche.

    Args:
        outcome: bilan du fichier.
        error_type: type d'erreur.
        message: message français.
        **extra: champs complémentaires de l'entrée.

    Returns:
        Le bilan complété.
    """
    outcome.errors.append(_error_entry(outcome, error_type, message, **extra))
    _say(f"  Échec : {message}")
    return outcome


def _prepare_segments(ctx: _RunContext, path: str, mode: str, item_key: str, tmpdir: Optional[str],
                      source_duration: Optional[float]) -> List[_Segment]:
    """Segments du plan courant : le fichier entier, ou les segments ffmpeg.

    Args:
        ctx: contexte du traitement.
        path: fichier source.
        mode: ``whole`` ou ``split``.
        item_key: ``audio-<sha16>`` (noms neutres envoyés à Albert).
        tmpdir: dossier temporaire (mode ``split``).
        source_duration: durée de la source, ou ``None``.

    Returns:
        Les segments, numérotés à partir de 1.

    Raises:
        AudioConversionError: découpage impossible.
    """
    if mode == "whole":
        ext = os.path.splitext(path)[1].lower()
        return [_Segment(1, path, f"{item_key}{ext}", 0.0, source_duration)]
    files = split_audio(path, tmpdir or "", ctx.segment_seconds, source_duration)
    windows = segment_windows(files, ctx.segment_seconds, source_duration)
    return [
        _Segment(i, source, f"{item_key}-seg{i:04d}.mp3", start, duration)
        for i, (source, (start, duration)) in enumerate(zip(files, windows), start=1)
    ]


def process_file(ctx: _RunContext, path: str, position: int, total: int) -> _FileOutcome:
    """Transcrit un enregistrement et construit sa ligne de CSV.

    Args:
        ctx: contexte du traitement.
        path: fichier audio ou vidéo.
        position: rang du fichier (à partir de 1).
        total: nombre de fichiers.

    Returns:
        Le bilan du fichier (ligne éventuelle, erreurs, segments).

    Raises:
        _RunAbort: erreur de compte ou de quota Albert.
    """
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    outcome = _FileOutcome(filename=name, title=stem)
    _say(f"Fichier {position}/{total} : {name}")
    try:
        digest, size = file_digest(path)
        mtime = os.path.getmtime(path)
    except OSError as exc:
        return _fail(outcome, "AUDIO_UNREADABLE", f"fichier illisible ({exc.strerror or type(exc).__name__})")
    sha16 = digest[:16]
    outcome.item_key = f"audio-{sha16}"
    if sha16 in ctx.seen:
        return _fail(outcome, "DUPLICATE_AUDIO", f"contenu identique à {ctx.seen[sha16]} : fichier ignoré")
    ctx.seen[sha16] = name
    try:
        ckpt_dir = checkpoint_dir(ctx.output_dir, sha16, ctx.params)
    except (CheckpointPathError, OSError) as exc:
        reason = str(exc) if isinstance(exc, CheckpointPathError) else type(exc).__name__
        return _fail(outcome, "CHECKPOINT_REFUSED", f"points de reprise impossibles : {reason}")

    source_duration = media_duration(path)
    needs_conversion = ext not in DIRECT_EXTENSIONS or size > MAX_AUDIO_BYTES
    too_long = source_duration is not None and source_duration > ctx.segment_seconds
    mode = "split" if (needs_conversion or too_long) else "whole"

    planned = _load_plan(ckpt_dir, sha16, ctx.params, mode)
    results: Optional[List[_SegmentResult]] = None
    if planned is not None:
        cached = [_load_checkpoint(ckpt_dir, s, sha16, ctx.params, mode, len(planned)) for s in planned]
        if all(c is not None for c in cached):
            results = [c for c in cached if c is not None]
            for c in results:
                _say(f"  Segment {c.index}/{len(results)} : repris (point de reprise)")

    if results is None:
        if mode == "split" and not ffmpeg_available():
            if needs_conversion:
                return _fail(outcome, "FFMPEG_MISSING", FFMPEG_MISSING_MESSAGE)
            _say(f"  ffmpeg introuvable : enregistrement de plus de {ctx.segment_seconds} s envoyé en un seul segment.")
            mode = "whole"
        tmpdir = tempfile.mkdtemp(prefix="rad_audio_") if mode == "split" else None
        try:
            segments = _prepare_segments(ctx, path, mode, outcome.item_key, tmpdir, source_duration)
            try:
                _atomic_write_json(os.path.join(ckpt_dir, PLAN_FILENAME),
                                   _plan_payload(sha16, ctx.params, mode, segments, source_duration))
            except OSError as exc:
                logger.warning("Plan de reprise non écrit (%s).", type(exc).__name__)
            results = _run_segments(ctx, segments, ckpt_dir, sha16, mode)
        except AudioConversionError as exc:
            return _fail(outcome, "AUDIO_CONVERSION_FAILED", f"conversion impossible : {exc}")
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    windows = _windows(results)
    outcome.segments = [(r, start, end) for r, (start, end) in zip(results, windows)]
    outcome.segments_total = len(results)
    outcome.segments_ok = sum(1 for r in results if r.ok)
    outcome.segments_reused = sum(1 for r in results if r.reused)
    failed = [r for r in results if not r.ok]
    failed_detail = [{"segment": r.index, "error": r.error} for r in failed]

    if outcome.segments_ok == 0:
        reasons = "; ".join(f"segment {r.index} : {r.error}" for r in failed[:3])
        return _fail(outcome, "TRANSCRIPTION_FAILED", f"aucun segment transcrit ({reasons})",
                     provider=PROVIDER, pages_done=0, pages_total=len(results), failed_segments=failed_detail)
    if not any(r.ok and r.text.strip() for r in results):
        return _fail(outcome, "TRANSCRIPTION_EMPTY", "aucune parole transcrite (texte vide)",
                     provider=PROVIDER, pages_done=outcome.segments_ok, pages_total=len(results))

    languages = Counter(r.language for r in results if r.ok and r.language)
    language = languages.most_common(1)[0][0] if languages else ctx.language
    duration = source_duration if source_duration is not None else (windows[-1][1] if windows else None)
    outcome.row = {
        "itemKey": outcome.item_key,
        "title": stem,
        "authors": "",
        "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"),
        "filename": name,
        "path": name,
        "texteocr": build_texteocr(results),
        "texteocr_provider": PROVIDER,
        "texteocr_partial": bool(failed),
        "texteocr_pages_done": outcome.segments_ok,
        "texteocr_pages_total": len(results),
        "language": language or "",
        "duration_s": round(duration, 1) if duration is not None else None,
        "segments": len(results),
    }
    if failed:
        message = (f"Transcription partielle : {outcome.segments_ok}/{len(results)} segments ("
                   + "; ".join(f"segment {r.index} : {r.error}" for r in failed[:3]) + ")")
        outcome.errors.append(_error_entry(outcome, "TRANSCRIPTION_PARTIAL", message, provider=PROVIDER,
                                           pages_done=outcome.segments_ok, pages_total=len(results),
                                           failed_segments=failed_detail))
        _say(f"  ⚠ {message}")
    return outcome


# ---------------------------------------------------------------------------
# Écritures
# ---------------------------------------------------------------------------
def errors_file_path(output_csv: str) -> str:
    """Chemin du fichier d'erreurs (``<base>_errors.json``, comme ``rad_dataframe``).

    Args:
        output_csv: CSV de sortie.

    Returns:
        Le chemin.
    """
    return f"{os.path.splitext(output_csv)[0]}_errors.json"


def segments_file_path(output_csv: str) -> str:
    """Chemin du relevé des horodatages (``<base>_audio_segments.json``).

    Args:
        output_csv: CSV de sortie.

    Returns:
        Le chemin.
    """
    return f"{os.path.splitext(output_csv)[0]}_audio_segments.json"


def write_csv(rows: Sequence[Mapping[str, Any]], output_csv: str) -> None:
    """Écrit le CSV atomiquement (``utf-8-sig``, conventions de ``rad_dataframe``).

    Args:
        rows: lignes à écrire.
        output_csv: chemin du CSV.
    """
    import pandas as pd

    folder = os.path.dirname(os.path.abspath(output_csv))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".csv")
    os.close(fd)
    try:
        pd.DataFrame(list(rows), columns=list(CSV_COLUMNS)).to_csv(
            tmp, index=False, encoding="utf-8-sig", escapechar="\\"
        )
        os.replace(tmp, output_csv)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_errors(output_csv: str, errors: Sequence[Mapping[str, Any]]) -> str:
    """Écrit ``<base>_errors.json`` (``total_errors``, ``errors``, ``last_updated``).

    Args:
        output_csv: CSV de sortie.
        errors: entrées d'erreur.

    Returns:
        Le chemin écrit.
    """
    path = errors_file_path(output_csv)
    _atomic_write_json(path, {"total_errors": len(errors), "errors": list(errors), "last_updated": _now()})
    return path


def write_segments(output_csv: str, outcomes: Sequence[_FileOutcome]) -> Optional[str]:
    """Écrit le relevé des segments et des énoncés aux horodatages absolus.

    Args:
        output_csv: CSV de sortie.
        outcomes: bilans des fichiers ; seuls ceux qui ont une ligne de CSV y figurent.

    Returns:
        Le chemin écrit, ou ``None`` si aucun fichier n'a été transcrit.
    """
    files = []
    for outcome in outcomes:
        if outcome.row is None:
            continue
        segments = []
        for result, start, end in outcome.segments:
            entry: Dict[str, Any] = {
                "index": result.index,
                "start": round(start, 3),
                "end": round(end, 3),
                "status": "ok" if result.ok else "failed",
                "reused": result.reused,
                "utterances": _absolute_utterances(result, start) if result.ok else [],
            }
            if not result.ok:
                entry["error"] = result.error
            segments.append(entry)
        files.append({
            "itemKey": outcome.item_key,
            "filename": outcome.filename,
            "language": outcome.row.get("language"),
            "duration_s": outcome.row.get("duration_s"),
            "segments": segments,
        })
    if not files:
        return None
    path = segments_file_path(output_csv)
    _atomic_write_json(path, {"version": CHECKPOINT_VERSION, "provider": PROVIDER, "files": files,
                              "last_updated": _now()})
    return path


# ---------------------------------------------------------------------------
# Albert : client, ligne d'arrêt, journal d'usage
# ---------------------------------------------------------------------------
def _build_client(cfg: Any, api_key: str, *, ledger: Any) -> Any:
    """Construit le client Albert du traitement (point d'injection des tests).

    Args:
        cfg: configuration Albert.
        api_key: clé Bearer (jamais lue par la bibliothèque dans l'environnement).
        ledger: journal d'usage partagé.

    Returns:
        Un ``AlbertClient``.
    """
    return _albert_module("client").AlbertClient(cfg, api_key, ledger=ledger)


def albert_abort_marker(exc: BaseException) -> str:
    """Ligne ``Albert abort: …`` qui qualifie un arrêt Albert (même contrat que ``rad_chunk``).

    Motif lisible par ``^Albert abort: kind=(\\w+) reason=(\\w+)(?: credential_required=(\\w+))?$`` :
    ``account`` (avec ``credential_required``) pour une erreur de compte ou de
    quota, ``model``/``request``/``service`` pour les autres erreurs Albert,
    ``config`` pour Albert désactivé (``disabled``) ou une configuration refusée
    (``invalid_config``), ``error`` sinon.

    Args:
        exc: erreur qui arrête le traitement.

    Returns:
        La ligne, sans retour chariot.
    """
    credential = None
    if isinstance(exc, _rad_albert.AlbertAuthError):
        kind, reason = "account", exc.reason
        credential = getattr(exc, "credential_required", None)
    elif isinstance(exc, _rad_albert.AlbertPermanentError):
        kind = "model" if exc.reason in _ALBERT_MODEL_REASONS else "request"
        reason = exc.reason
    elif isinstance(exc, _rad_albert.AlbertError):
        kind, reason = "service", exc.message_key
    elif isinstance(exc, _rad_albert.AlbertDisabledError):
        kind, reason = "config", "disabled"
    elif isinstance(exc, ValueError):
        kind, reason = "config", "invalid_config"
    else:
        kind, reason = "error", "unexpected"
    line = f"{ALBERT_ABORT_MARKER} kind={kind} reason={reason}"
    if credential:
        line += f" credential_required={credential}"
    return line


def _flush_usage(ctx: _RunContext) -> Optional[str]:
    """Ajoute à ``<dossier de sortie>/albert_usage.jsonl`` les appels non encore écrits.

    Rien n'est écrit si ``ALBERT_USAGE_LOG=0`` ou si aucun appel n'est en attente.

    Args:
        ctx: contexte du traitement.

    Returns:
        Le chemin écrit, ou ``None``.
    """
    if not getattr(ctx.cfg, "usage_log", True):
        return None
    try:
        return ctx.ledger.write_jsonl(ctx.output_dir)
    except OSError as exc:
        logger.warning("Journal d'usage Albert non écrit (%s).", type(exc).__name__)
        return None


def _load_env() -> None:
    """Charge le ``.env`` gardé (point d'injection des tests, qui ne le chargent jamais)."""
    load_dotenv_guarded()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _positive_int(value: str) -> int:
    """Type argparse : entier strictement positif.

    Args:
        value: texte de l'option.

    Returns:
        L'entier.

    Raises:
        argparse.ArgumentTypeError: valeur invalide.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"entier attendu : {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("entier strictement positif attendu")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Analyseur des options de la ligne de commande.

    Returns:
        L'analyseur.
    """
    parser = argparse.ArgumentParser(
        description=("Transcrit des enregistrements audio (entretiens, séminaires) par Albert (Whisper) "
                     "en un output.csv compatible avec rad_chunk.py."),
    )
    parser.add_argument("--input", required=True,
                        help="Fichier audio, ou dossier parcouru sans récursion (" + " ".join(AUDIO_INPUT_EXTENSIONS) + ").")
    parser.add_argument("--output", required=True, help="CSV de sortie (ex. sources/X/output.csv).")
    parser.add_argument("--language", default=None,
                        help="Langue (code ISO 639-1, ex. fr) ; auto = détection ; défaut : ALBERT_AUDIO_LANGUAGE.")
    parser.add_argument("--prompt", default=None, help="Amorce de vocabulaire (noms propres, sigles), facultative.")
    parser.add_argument("--max-files", type=_positive_int, default=None, help="Nombre maximal de fichiers traités.")
    return parser


def _refuse(message: str, exc: BaseException) -> int:
    """Affiche un refus, la ligne ``Albert abort: …`` et renvoie le code 2.

    Args:
        message: message français.
        exc: erreur qui qualifie le refus.

    Returns:
        2.
    """
    _say(f"Transcription audio refusée : {message}")
    _say(albert_abort_marker(exc))
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée : contrôles, transcription fichier par fichier, écritures.

    Args:
        argv: arguments (défaut : ``sys.argv[1:]``).

    Returns:
        0 (au moins un fichier transcrit), 1 (aucun texte), 2 (refus ou arrêt Albert).
    """
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.output.lower().endswith(".csv"):
        parser.error("--output doit désigner un fichier .csv")

    try:
        cfg = _rad_albert.AlbertConfig.from_env()
    except ValueError as exc:
        return _refuse(f"configuration Albert invalide ({exc}).", exc)
    if not cfg.enabled:
        return _refuse("Albert est désactivé (ALBERT_ENABLED=1 requis).",
                       _rad_albert.AlbertDisabledError(model="audio"))
    if not cfg.audio_enabled:
        return _refuse("l'import audio est désactivé (ALBERT_AUDIO_ENABLED=1 requis).",
                       _rad_albert.AlbertDisabledError(model="audio"))
    api_key = (os.environ.get("ALBERT_API_KEY") or "").strip()
    if not api_key:
        return _refuse(MISSING_KEY_MESSAGE, _rad_albert.AlbertAuthError(MISSING_KEY_MESSAGE, reason="invalid_key"))

    try:
        language = normalise_language(args.language, cfg.audio_language)
    except ValueError as exc:
        parser.error(str(exc))
    prompt = (args.prompt or "").strip()[:PROMPT_MAX_CHARS]
    try:
        files = list_audio_files(args.input, args.max_files)
    except FileNotFoundError:
        _say(f"Entrée introuvable : {os.path.basename(os.path.normpath(args.input))}")
        return 2
    except ValueError as exc:
        _say(f"Entrée refusée : {exc}")
        return 2
    if not files:
        _say("Aucun fichier audio trouvé dans l'entrée : aucun CSV écrit.")
        return 1

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    segment_seconds = int(cfg.audio_segment_seconds)
    if segment_seconds > MAX_SEGMENT_SECONDS:
        _say(f"Segments ramenés de {segment_seconds} à {MAX_SEGMENT_SECONDS} s (limite de 20 Mo par envoi).")
        segment_seconds = MAX_SEGMENT_SECONDS
    model = _model_name(cfg.audio_model)
    ledger = _albert_module("usage").UsageLedger()
    try:
        client = _build_client(cfg, api_key, ledger=ledger)
    except ValueError as exc:
        return _refuse(f"configuration Albert invalide ({exc}).", exc)
    ctx = _RunContext(
        cfg=cfg, client=client, ledger=ledger, output_dir=os.path.realpath(output_dir), model=model,
        language=language, prompt=prompt, segment_seconds=segment_seconds,
        params=params_key(model, language, prompt, segment_seconds),
    )
    _say(f"Transcription audio Albert : {len(files)} fichier(s), modèle {model}, "
         f"langue {language or 'détection automatique'}, segments de {segment_seconds} s.")

    outcomes: List[_FileOutcome] = []
    abort: Optional[BaseException] = None
    try:
        for position, path in enumerate(files, start=1):
            try:
                outcomes.append(process_file(ctx, path, position, len(files)))
            except _RunAbort as stop:
                abort = stop.error
                remaining = [os.path.basename(p) for p in files[position - 1:]]
                failed = _FileOutcome(filename=os.path.basename(path), title=os.path.splitext(os.path.basename(path))[0])
                failed.errors.append(_error_entry(
                    failed, "ALBERT_ACCOUNT_ERROR", str(abort), reason=getattr(abort, "reason", None),
                    unprocessed_files=remaining,
                ))
                outcomes.append(failed)
                break
            finally:
                _flush_usage(ctx)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001 - fermeture best effort
            pass

    rows = [o.row for o in outcomes if o.row is not None]
    errors = [e for o in outcomes for e in o.errors]
    errors_path = write_errors(args.output, errors)
    if rows:
        write_csv(rows, args.output)
        write_segments(args.output, outcomes)
    seg_total = sum(o.segments_total for o in outcomes)
    seg_ok = sum(o.segments_ok for o in outcomes)
    seg_reused = sum(o.segments_reused for o in outcomes)
    partial = sum(1 for r in rows if r["texteocr_partial"])
    _say("=== Résumé de la transcription ===")
    _say(f"Fichiers transcrits : {len(rows)}/{len(files)} (partiels : {partial})")
    _say(f"Segments transcrits : {seg_ok}/{seg_total} (repris d'un point de reprise : {seg_reused})")
    if rows:
        _say(f"Output CSV saved to: {args.output}")
    else:
        _say("Aucun texte transcrit : CSV non écrit.")
    if errors:
        _say(f"Erreurs : {len(errors)} (voir {os.path.basename(errors_path)})")
    if ledger.called:
        _say(ledger.summary_line())
    if abort is not None:
        _say(f"Erreur critique Albert : {abort} Arrêt.")
        _say(albert_abort_marker(abort))
        return 2
    return 0 if rows else 1


def _sigterm_to_exit(signum: int, _frame: Any) -> None:
    """Transforme SIGTERM en ``SystemExit`` pour exécuter les ``finally`` (dossier temporaire).

    Args:
        signum: numéro du signal.
        _frame: cadre courant (inutilisé).
    """
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    signal.signal(signal.SIGTERM, _sigterm_to_exit)
    sys.exit(main())
