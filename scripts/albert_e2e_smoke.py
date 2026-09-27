#!/usr/bin/env python3
"""Test de bout en bout souverain d'Albert (DINUM) par les vraies routes de RAGpy.

Le script démarre le serveur FastAPI réel (``uvicorn`` en sous-processus) sur
une base SQLite temporaire, puis vérifie par HTTP les 12 contrôles du lot 8 :
7 avec Albert activé, 2 d'isolation des identifiants, 3 avec Albert
désactivé. Il appelle l'API Albert de production : il refuse de tourner
(code 2) sans ``ALBERT_LIVE=1`` dans le shell, et si ``ALBERT_LIVE`` figure
dans le ``.env``.

Préparation :

* corpus construit avec PyMuPDF : ``N-1`` PDF texte et 1 PDF image seule
  (page numérisée, sans couche texte), plus un export Zotero JSON ; le tout
  est zippé à plat (JSON et dossier ``files/`` à la racine), comme l'attend
  ``POST /api/pipeline/projects/{project_id}/upload_zip`` ;
* base temporaire désignée par ``DATABASE_URL``, schéma créé par
  ``init_database`` ; un admin et deux non-admins A et B (actifs, vérifiés)
  et un projet pour chacun des non-admins sont insérés avec SQLAlchemy
  **avant** le démarrage du serveur ; les jetons sont frappés par
  ``create_access_token`` avec le même ``JWT_SECRET_KEY`` que le serveur ;
* serveur ``[sys.executable, -m, uvicorn, app.main:app]`` sur un port libre.
  Son environnement place ``.venv/bin`` en tête du ``PATH`` (les routes
  lancent ``python3``) et porte ``DATABASE_URL``, ``ALBERT_ENABLED=1`` et
  ``OCR_ENABLE_ALBERT=1`` (environnement du processus seulement, jamais le
  ``.env``). La dédup et les caches de recodage y sont ramenés à leur défaut
  (``0``) : la relance de l'envoi compte alors des chunks « déjà présents »
  et aucune trace ne reste dans les caches partagés. ``ALBERT_OCR_SKIP_RECODE``
  (``0``) et ``ALBERT_USAGE_LOG`` (``1``) y sont aussi ramenés à leur défaut :
  le texte LightOnOCR passe toujours par le recodage contrôlé, et le ledger
  ``albert_usage.jsonl`` est toujours écrit. Avant tout,
  ``shutil.which('python3', path=<PATH du serveur>)`` doit pointer dans
  ``.venv/bin`` (sinon code 2) ;
* A enregistre sa clé (lue par ``dotenv_values`` dans le ``.env``, jamais
  affichée) par ``PUT /users/me/credentials`` ; B n'en a pas.

Contrôles (tableau du lot 8) :

* ON, en tant que A : ``process_dataframe_sse`` (PDF numérisé : fournisseur
  ``albert_…``, marqueurs ``<!-- Page N -->`` continus, ``texteocr_partial``
  faux et aucune entrée ``OCR_FAILED``, ``OCR_PARTIAL``,
  ``OCR_PROVIDER_FALLBACK`` ou ``EXTRACTION_FAILED`` dans
  ``output_errors.json``) ;
  ``initial_text_chunking_sse`` avec ``albert/ministral-3-8b-instruct-2512``
  (``recode_model`` = ``albert/<id résolu>`` ; ``recode_status`` vaut
  ``recoded`` pour chaque chunk, ``skipped`` n'étant admis que pour un
  fournisseur exclu du recodage autre que ``albert_lightonocr`` ; au moins un
  enregistrement ``role=recode`` avec son modèle dans ``albert_usage.jsonl``) ;
  ``dense_embedding_generation_sse`` avec ``embedding_provider=albert``
  (``embedding_dim=1024`` partout, aucun vecteur nul) ; ``upload_db`` vers
  une collection ``ragpy-probe-e2e-<ts>`` créée avec acquittement
  (``Inserted`` > 0, manifeste copié sous ``data/albert_manifests/``, entrée
  d'audit) ; ``upload_db`` relancé (``Inserted: 0``, ``Skipped (existing): N``) ;
  ``generate_zotero_notes_sse`` avec ``albert/gpt-oss-120b`` en mode court
  (note différente du gabarit ; dans ``albert_usage.jsonl``, au moins un
  enregistrement des notes servi par l'id épinglé de gpt-oss avec un coût
  numérique) ;
  ``GET /api/albert/collections`` puis ``DELETE …?confirm=true`` (collection
  listée, puis absente) ;
* isolation : chunking ``albert/…`` de B en 403
  ``credential_required=albert_api_key`` alors que le ``.env`` admin contient
  la clé ; ``GET /users/me/credentials`` de A masqué et sans la clé brute,
  celui de B sans valeur Albert ;
* OFF, après redémarrage avec ``ALBERT_ENABLED=0`` : aucune occurrence
  d'« albert » dans le HTML de l'index et du profil ; ``/api/albert/status``
  en 404 et chunking ``albert/…`` en 400 ; ``GET /get_credentials`` (admin)
  sans clé ``ALBERT_API_KEY``.

Les flux SSE sont lus jusqu'à l'événement ``complete``, ou jusqu'à la fin du
flux : l'issue est alors le dernier événement ``complete`` ou ``error`` (un
événement ``error`` intermédiaire peut n'être qu'une ligne de journal du
sous-processus, qui continue).

Ledger des notes (on6) : la route ``generate_zotero_notes_sse`` ajoute les
enregistrements de ses appels Albert à ``<session>/albert_usage.jsonl`` dans
son bloc ``finally``, donc éventuellement juste après l'événement
``complete`` : le fichier est relu quelques instants
(``NOTES_LEDGER_ATTEMPTS`` × ``NOTES_LEDGER_WAIT_S``) tant qu'aucun
enregistrement des notes (rôles ``notes`` ou ``long_context``) n'y figure.
Avec ``NOTES_LEDGER_REQUIRED`` à ``True`` (valeur du script), on6 exige un
enregistrement des notes dont le modèle est l'id épinglé de gpt-oss
(``gpt-oss-120b``, pas un alias) et le coût numérique ; à ``False``,
l'absence d'enregistrement serait seulement informative. Des enregistrements
de notes présents sont toujours vérifiés.

Fin, dans un bloc ``finally`` : arrêt des serveurs (groupe de processus
compris) avant toute suppression de collection, suppression par
``AlbertClient`` de la collection du run (nom exact), recherche de la clé
dans les journaux produits pendant le run, suppression des dossiers de
session et des archives créés sous ``uploads/``, des copies de manifestes du
run sous ``data/albert_manifests/`` et de la base temporaire. Chaque étape
est isolée : une étape en échec devient une anomalie (run en ÉCHEC) sans
empêcher les suivantes. SIGTERM et SIGHUP interrompent le run comme un
Ctrl-C (le nettoyage s'exécute) ; pendant le nettoyage et l'écriture du
résumé, SIGINT, SIGTERM et SIGHUP sont ignorés. Avec ``--cleanup``, les
collections restantes d'un run précédent de ce script sont aussi supprimées
au démarrage : seulement les noms ``ragpy-probe-e2e-AAAAMMJJ-HHMMSS``, jamais
une collection de la suite live (``ragpy-probe-e2e-AAAAMMJJTHHMMSS-<pid>``)
qui tournerait en même temps.

Sortie : résumé JSON ``data/e2e_albert/summary-<ts>.json`` (clé masquée),
puis, en dernière ligne, ``E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)``
ou ``E2E ALBERT ÉCHEC (…)`` avec les compteurs. Codes de sortie : 0 (OK),
1 (échec), 2 (refus).

Commande :

    ALBERT_LIVE=1 .venv/bin/python scripts/albert_e2e_smoke.py --corpus-size 3 --cleanup
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / ".env"
VENV_BIN = REPO / ".venv" / "bin"
UPLOAD_DIR = REPO / "uploads"
MANIFEST_ROOT = REPO / "data" / "albert_manifests"
SUMMARY_DIR = REPO / "data" / "e2e_albert"
LOG_DIR = REPO / "logs"

COLLECTION_PREFIX = "ragpy-probe-e2e-"
CORPUS_PREFIX = "e2e-corpus-"
RECODE_MODEL = "albert/ministral-3-8b-instruct-2512"
NOTES_MODEL = "albert/gpt-oss-120b"
ALBERT_PREFIX = "albert/"
EMBED_DIM = 1024
ALBERT_CREDENTIAL_KEY = "albert_api_key"
ALBERT_ENV_KEY = "ALBERT_API_KEY"
UPLOAD_AUDIT_ACTION = "ALBERT_COLLECTION_UPLOAD"
NOTES_ROLES = ("notes", "long_context")

DEFAULT_CORPUS_SIZE = 3
MIN_CORPUS_SIZE = 2
PAGES_PER_DOC = 2
SCAN_DPI = 200
TOKEN_TTL = timedelta(hours=12)
# Ids explicites des utilisateurs de la base temporaire : les copies de
# manifeste du run (``data/albert_manifests/<id>/``) ne se mêlent jamais à
# celles d'un utilisateur réel.
E2E_USER_IDS = {"admin": 990001, "a": 990002, "b": 990003}

SERVER_START_TIMEOUT_S = 120.0
SERVER_STOP_TIMEOUT_S = 15.0
HTTP_TIMEOUT_S = 120.0
UPLOAD_TIMEOUT_S = 3700.0
SSE_READ_TIMEOUT_S = 1800.0
STEP_TIMEOUT_S = 3 * 3600.0
DELETE_CONFIRM_ATTEMPTS = 5
DELETE_CONFIRM_WAIT_S = 2.0

# Variables ramenées à leur défaut dans l'environnement du serveur : la relance
# de l'envoi compte des chunks « déjà présents » (et non dédupliqués), aucune
# trace du run ne reste dans les caches partagés, le texte LightOnOCR est recodé
# (on2 contrôle un vrai recodage) et le ledger d'usage est écrit.
PINNED_ENV: Dict[str, str] = {
    "DEDUP_ENABLED": "0",
    "RECODE_CACHE_ENABLED": "0",
    "RECODE_EMBED_CACHE_ENABLED": "0",
    "ALBERT_OCR_SKIP_RECODE": "0",
    "ALBERT_USAGE_LOG": "1",
}

# Nom des collections de ce script : ``ragpy-probe-e2e-AAAAMMJJ-HHMMSS``. Le
# balayage initial ne supprime que ce format, jamais une collection de la suite
# live (``ragpy-probe-e2e-AAAAMMJJTHHMMSS-<pid>``) qui tournerait en même temps.
SMOKE_COLLECTION_RE = re.compile(re.escape(COLLECTION_PREFIX) + r"\d{8}-\d{6}")

# OCR (on1) : fournisseur LightOnOCR, erreurs bloquantes du fichier d'erreurs,
# valeurs de ``texteocr_partial`` qui signalent un OCR complet.
LIGHTONOCR_PROVIDER = "albert_lightonocr"
BLOCKING_OCR_ERRORS = ("OCR_FAILED", "OCR_PARTIAL", "OCR_PROVIDER_FALLBACK", "EXTRACTION_FAILED")
PARTIAL_FALSE_VALUES = ("false", "0", "")

# Recodage (on2) : statut d'un chunk recodé dans ce run, statut d'un chunk dont
# le fournisseur d'OCR est exclu du recodage, rôle des enregistrements du ledger.
RECODED_STATUS = "recoded"
SKIPPED_STATUS = "skipped"
RECODE_ROLE = "recode"

# Ledger des notes (on6). La route ``generate_zotero_notes_sse`` ajoute les
# enregistrements de ses appels Albert à ``albert_usage.jsonl`` (dans son bloc
# ``finally``, éventuellement juste après l'événement ``complete``). on6 exige
# un enregistrement des notes servi par l'id épinglé de gpt-oss avec un coût
# numérique. ``NOTES_LEDGER_PENDING`` n'est employé qu'avec
# ``NOTES_LEDGER_REQUIRED = False`` (absence d'enregistrement informative).
# Le journal est relu au plus ``NOTES_LEDGER_ATTEMPTS`` fois, à
# ``NOTES_LEDGER_WAIT_S`` secondes d'intervalle, tant qu'il ne porte aucun
# enregistrement des notes.
NOTES_LEDGER_REQUIRED = True
NOTES_LEDGER_PENDING = "informatif : journal des notes non exigé (NOTES_LEDGER_REQUIRED = False)"
NOTES_LEDGER_ATTEMPTS = 10
NOTES_LEDGER_WAIT_S = 0.5

# Signaux qui interrompent le run comme un Ctrl-C (le nettoyage s'exécute), et
# signaux ignorés pendant le nettoyage et l'écriture du résumé.
INTERRUPT_SIGNALS: Tuple[int, ...] = tuple(
    s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s is not None)
SHIELDED_SIGNALS: Tuple[int, ...] = (signal.SIGINT,) + INTERRUPT_SIGNALS

# Clé Bearer : au moins 8 caractères ASCII imprimables, sans espace (un
# caractère de contrôle dans l'en-tête ferait afficher la clé par la pile HTTP).
API_KEY_RE = re.compile(r"[\x21-\x7e]{8,}")
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
SSE_TYPE_RE = re.compile(r'"type"\s*:\s*"([A-Za-z_]+)"')
SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")
SAFE_LABEL_MAX_CHARS = 120
EXCLUDED_JSON_PREFIXES = ("output_", "output.", "generated_")
TEMPLATE_MARKERS = ("à compléter", "a completer", "to be completed")
MIN_NOTE_CHARS = 40
MASK = "[masqué]"

ON_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("on1", "POST /process_dataframe_sse"),
    ("on2", "POST /initial_text_chunking_sse (albert/ministral-3-8b-instruct-2512)"),
    ("on3", "POST /dense_embedding_generation_sse (embedding_provider=albert)"),
    ("on4", "POST /upload_db (db_choice=albert, création de la collection, acquittement)"),
    ("on5", "POST /upload_db (relance)"),
    ("on6", "POST /generate_zotero_notes_sse (albert/gpt-oss-120b, mode short)"),
    ("on7", "GET /api/albert/collections puis DELETE /api/albert/collections/{id}?confirm=true"),
)
ISOLATION_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("iso1", "POST /initial_text_chunking (B, albert/…)"),
    ("iso2", "GET /users/me/credentials (A puis B)"),
)
OFF_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("off1", "GET /, /pipeline, /profile, /project/{id} (HTML sans « albert »)"),
    ("off2", "GET /api/albert/status puis POST /initial_text_chunking (albert/…)"),
    ("off3", "GET /get_credentials (admin)"),
)
CHECK_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    ("on", ON_CHECKS),
    ("isolation", ISOLATION_CHECKS),
    ("off", OFF_CHECKS),
)
CHECK_ROUTES: Dict[str, str] = {cid: route for _group, checks in CHECK_GROUPS for cid, route in checks}

_TEXT_SENTENCES = (
    "Les bibliothèques universitaires numérisent leurs fonds et transforment les pratiques de lecture savante.",
    "La recherche documentaire articule désormais catalogues, archives ouvertes et bases de données.",
    "Les chercheurs annotent, classent et partagent leurs lectures dans des outils de gestion bibliographique.",
    "L'indexation automatique des textes suppose une reconnaissance fiable des caractères imprimés.",
    "Les corpus numérisés permettent de comparer des revues, des époques et des disciplines.",
    "La qualité des métadonnées conditionne la découverte et la citation des travaux.",
    "Les sciences humaines et sociales étudient ces infrastructures comme des objets de recherche.",
    "Une chaîne de traitement reproductible documente chaque étape, de la source au résultat.",
    "La conservation des documents numériques exige des formats ouverts et des identifiants pérennes.",
    "Les politiques de science ouverte encouragent le dépôt des publications et des données.",
    "Les enquêtes de terrain montrent que la lecture à l'écran fragmente l'attention des étudiants.",
    "Les revues savantes expérimentent de nouveaux modèles éditoriaux et de nouvelles formes d'évaluation.",
)
_TEXT_TITLES = (
    "Lire à l'ère des bibliothèques numériques",
    "Métadonnées et découverte des travaux savants",
    "Archives ouvertes et science ouverte en sciences sociales",
    "Chaînes de traitement reproductibles pour les corpus textuels",
)
_SCAN_TITLE = "Numérisation d'un ouvrage de sciences sociales"
_AUTHORS = (("Durand", "Claire"), ("Martin", "Paul"), ("Petit", "Anne"), ("Moreau", "Luc"))


# ---------------------------------------------------------------------------
# Aides pures (testées hors ligne)
# ---------------------------------------------------------------------------
class CheckFailed(Exception):
    """Échec d'un contrôle : message court et détail structuré (clé masquée ensuite)."""

    def __init__(self, message: str, detail: Optional[Mapping[str, Any]] = None) -> None:
        """Construit l'échec.

        Args:
            message: raison lisible, en français.
            detail: éléments observés, versés au résumé JSON.
        """
        super().__init__(message)
        self.message = message
        self.detail: Dict[str, Any] = dict(detail or {})


def _ensure_repo_on_path() -> None:
    """Place la racine du dépôt en tête de ``sys.path`` (imports ``app`` et ``scripts``)."""
    root = str(REPO)
    if root not in sys.path:
        sys.path.insert(0, root)


def _dotenv_values(env_file: Any) -> Dict[str, Optional[str]]:
    """Lit un fichier ``.env`` sans l'afficher (dictionnaire vide s'il manque)."""
    if not env_file or not os.path.isfile(env_file):
        return {}
    from dotenv import dotenv_values

    return dict(dotenv_values(env_file))


def live_gate_refusal(environ: Mapping[str, str], env_file: Any = ENV_FILE) -> Optional[str]:
    """Message de refus si le run live n'est pas explicitement autorisé, sinon ``None``.

    Args:
        environ: environnement du shell.
        env_file: fichier ``.env`` du dépôt (seuls les noms de variables sont examinés).

    Returns:
        Le message de refus, ou ``None``.
    """
    if environ.get("ALBERT_LIVE") != "1":
        return ("Refus : ce test de bout en bout appelle l'API Albert de production. Lancer avec "
                "ALBERT_LIVE=1 dans le shell (jamais dans .env).")
    if "ALBERT_LIVE" in _dotenv_values(env_file):
        return (f"Refus : ALBERT_LIVE figure dans {env_file}. Cette variable doit venir du shell "
                "uniquement ; la retirer du fichier.")
    return None


def load_api_key(env_file: Any = ENV_FILE) -> Tuple[Optional[str], Optional[str]]:
    """Clé Albert du ``.env`` (``dotenv_values``), jamais affichée.

    Le ``.env`` est la seule source admise : le contrôle d'isolation vérifie
    qu'un non-admin sans clé reste refusé alors que le ``.env`` admin la
    contient.

    Args:
        env_file: fichier ``.env`` du dépôt.

    Returns:
        ``(clé, None)``, ou ``(None, message de refus)``.
    """
    value = _dotenv_values(env_file).get(ALBERT_ENV_KEY)
    key = value.strip() if isinstance(value, str) else ""
    if not key:
        return None, f"Refus : {ALBERT_ENV_KEY} absente de {env_file} (clé de la personne A et repli admin)."
    if not API_KEY_RE.fullmatch(key):
        return None, f"Refus : {ALBERT_ENV_KEY} mal formée (caractères non imprimables ou trop courte)."
    return key, None


def build_server_path(base_path: Optional[str], venv_bin: Any = VENV_BIN) -> str:
    """``PATH`` du serveur : ``<dépôt>/.venv/bin`` en tête, puis le ``PATH`` du shell."""
    head = str(venv_bin)
    return head + os.pathsep + base_path if base_path else head


def venv_python_problem(server_path: str, venv_bin: Any = VENV_BIN) -> Optional[str]:
    """Vérifie que ``python3`` (lancé par les routes) se résout dans ``.venv/bin``.

    Args:
        server_path: ``PATH`` de l'environnement du serveur.
        venv_bin: dossier ``bin`` de l'environnement virtuel du dépôt.

    Returns:
        ``None`` si ``shutil.which('python3', path=server_path)`` est dans
        ``venv_bin``, sinon le message d'erreur.
    """
    found = shutil.which("python3", path=server_path)
    if not found:
        return f"python3 introuvable dans le PATH du serveur (attendu dans {venv_bin})."
    here = os.path.normcase(os.path.dirname(os.path.abspath(found)))
    expected = os.path.normcase(os.path.abspath(str(venv_bin)))
    if here != expected:
        return f"python3 du PATH du serveur ({found}) n'est pas celui de {venv_bin}."
    return None


def free_port() -> int:
    """Port TCP libre sur 127.0.0.1 (attribué par le système)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def redact(value: Any, hidden: Sequence[str]) -> Any:
    """Masque récursivement les secrets dans une valeur (chaînes, listes, dictionnaires).

    Args:
        value: valeur quelconque.
        hidden: secrets à masquer (les chaînes de moins de 4 caractères sont ignorées).

    Returns:
        Une copie où chaque occurrence d'un secret est remplacée par ``[masqué]``.
    """
    needles = [s for s in hidden if isinstance(s, str) and len(s) >= 4]
    if isinstance(value, str):
        for needle in needles:
            value = value.replace(needle, MASK)
        return value
    if isinstance(value, Mapping):
        return {redact(k, needles) if isinstance(k, str) else k: redact(v, needles) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact(v, needles) for v in value]
    return value


def summary_line(on_ok: int, off_ok: int, isolation_ok: int, extras: Iterable[Any] = ()) -> str:
    """Dernière ligne du run.

    Args:
        on_ok: contrôles ON réussis.
        off_ok: contrôles OFF réussis.
        isolation_ok: contrôles d'isolation réussis.
        extras: anomalies hors contrôles (collection restante, fuite de clé…).

    Returns:
        ``E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)`` si tout est réussi
        et sans anomalie, sinon ``E2E ALBERT ÉCHEC (ON a/7, OFF b/3,
        isolation c/2[ ; anomalies])``.
    """
    counts = (f"ON {on_ok}/{len(ON_CHECKS)}, OFF {off_ok}/{len(OFF_CHECKS)}, "
              f"isolation {isolation_ok}/{len(ISOLATION_CHECKS)}")
    notes = [str(e).strip() for e in extras if str(e).strip()]
    complete = (on_ok == len(ON_CHECKS) and off_ok == len(OFF_CHECKS)
                and isolation_ok == len(ISOLATION_CHECKS))
    if complete and not notes:
        return f"E2E ALBERT OK ({counts})"
    if notes:
        counts += " ; " + " ; ".join(notes)
    return f"E2E ALBERT ÉCHEC ({counts})"


def parse_sse_data(payload: str) -> Dict[str, Any]:
    """Décode la charge d'un événement SSE (JSON, sinon type extrait par motif).

    Certains événements des routes sont formatés à la main et peuvent ne pas
    être du JSON valide (guillemets d'un message) : leur type reste lu.

    Args:
        payload: texte après ``data:``.

    Returns:
        Le dictionnaire de l'événement (``brut=True`` s'il n'était pas du JSON).
    """
    try:
        data = json.loads(payload)
    except ValueError:
        data = None
    if isinstance(data, dict):
        return data
    match = SSE_TYPE_RE.search(payload)
    return {"type": match.group(1) if match else "inconnu", "message": payload[:500], "brut": True}


def iter_sse_events(lines: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    """Événements d'un flux SSE, ligne ``data: {...}`` par ligne.

    Args:
        lines: lignes du flux (``str`` ou ``bytes``).

    Yields:
        Chaque événement décodé.
    """
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        text = str(line).strip()
        if not text.startswith("data:"):
            continue
        payload = text[len("data:"):].strip()
        if payload:
            yield parse_sse_data(payload)


def terminal_event(events: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Dernier événement ``complete`` ou ``error`` d'un flux lu jusqu'au bout (``None`` sinon)."""
    for event in reversed(list(events)):
        if event.get("type") in ("complete", "error"):
            return dict(event)
    return None


def page_markers(text: Any) -> List[int]:
    """Numéros des marqueurs ``<!-- Page N -->`` d'un texte, dans l'ordre d'apparition."""
    return [int(n) for n in PAGE_MARKER_RE.findall(str(text or ""))]


def markers_continuous(numbers: Sequence[int], expected_pages: Optional[int] = None) -> bool:
    """Vrai si les marqueurs valent exactement 1, 2, …, n (n = ``expected_pages`` s'il est donné)."""
    if not numbers:
        return False
    if list(numbers) != list(range(1, len(numbers) + 1)):
        return False
    return expected_pages is None or len(numbers) == int(expected_pages)


def embedding_problems(chunks: Sequence[Mapping[str, Any]], dim: int = EMBED_DIM, limit: int = 10) -> List[str]:
    """Défauts des embeddings denses d'une liste de chunks.

    Args:
        chunks: chunks de ``output_chunks_with_embeddings.json``.
        dim: dimension attendue (``embedding_dim`` et longueur du vecteur).
        limit: nombre maximal de défauts renvoyés.

    Returns:
        Les défauts (liste vide si tous les chunks portent ``embedding_dim=dim``
        et un vecteur fini, de longueur ``dim``, non nul).
    """
    if not chunks:
        return ["aucun chunk"]
    problems: List[str] = []
    for index, chunk in enumerate(chunks):
        label = str(chunk.get("id") or f"#{index}")
        declared = chunk.get("embedding_dim")
        vector = chunk.get("embedding")
        if isinstance(declared, bool) or declared != dim:
            problems.append(f"{label} : embedding_dim={declared!r} (attendu {dim})")
        if not isinstance(vector, list) or len(vector) != dim:
            size = len(vector) if isinstance(vector, list) else None
            problems.append(f"{label} : vecteur de longueur {size} (attendu {dim})")
        elif not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector):
            problems.append(f"{label} : valeur non finie ou non numérique dans le vecteur")
        elif not any(x != 0 for x in vector):
            problems.append(f"{label} : vecteur nul")
        if len(problems) >= limit:
            break
    return problems[:limit]


def recode_model_problem(value: Any, requested: str = RECODE_MODEL, today: Optional[date] = None) -> Optional[str]:
    """Défaut du champ ``recode_model`` d'un chunk recodé par Albert (``None`` si conforme).

    Conforme : ``albert/<id>`` où ``<id>`` est l'id épinglé du catalogue (pas
    un alias) du modèle demandé ou d'un modèle de la chaîne de repli du rôle
    ``recode`` à la date ``today``.

    Args:
        value: valeur du champ.
        requested: modèle demandé à la route (``albert/<nom>``).
        today: date de la chaîne de repli (défaut : aujourd'hui).

    Returns:
        Le défaut, ou ``None``.
    """
    _ensure_repo_on_path()
    from scripts.rad_albert import catalog

    if not isinstance(value, str) or not value.startswith(ALBERT_PREFIX):
        return f"recode_model {value!r} sans préfixe {ALBERT_PREFIX}"
    served = value[len(ALBERT_PREFIX):]
    if not served:
        return "recode_model sans identifiant après albert/"
    if catalog.model_spec(served) is None:
        return f"recode_model {value!r} : modèle inconnu du catalogue"
    canonical = catalog.canonical_id(served)
    if canonical != served:
        return f"recode_model {value!r} : alias non résolu (id épinglé : {canonical})"
    wanted = requested[len(ALBERT_PREFIX):] if requested.startswith(ALBERT_PREFIX) else requested
    wanted_id = catalog.canonical_id(wanted)
    allowed = [wanted_id] + [m for m in catalog.fallback_chain("recode", today=today or date.today()) if m != wanted_id]
    if served not in allowed:
        return f"recode_model {value!r} hors de la chaîne du rôle recode ({', '.join(allowed)})"
    return None


def note_problem(text: Any, abstract: Any = "") -> Optional[str]:
    """Défaut d'un résumé généré en mode court (``None`` s'il n'est pas un gabarit).

    Args:
        text: résumé de ``generated_notes.json``.
        abstract: résumé Zotero du document (une note qui le recopie n'est pas générée).

    Returns:
        Le défaut, ou ``None``.
    """
    if not isinstance(text, str) or len(text.strip()) < MIN_NOTE_CHARS:
        return "note vide ou trop courte"
    lowered = text.lower()
    if any(marker in lowered for marker in TEMPLATE_MARKERS):
        return "note issue du gabarit de repli"
    if isinstance(abstract, str) and abstract.strip() and text.strip() == abstract.strip():
        return "note identique au résumé Zotero"
    return None


def partial_flag_set(value: Any) -> bool:
    """Vrai si ``texteocr_partial`` signale un OCR partiel (tout sauf ``False``, ``0`` ou vide)."""
    if isinstance(value, bool):
        return value
    text = "" if value is None else str(value)
    return text.strip().lower() not in PARTIAL_FALSE_VALUES


def read_ocr_errors(path: Any) -> List[Dict[str, Any]]:
    """Entrées du fichier d'erreurs de ``rad_dataframe`` (``output_errors.json``).

    Args:
        path: chemin du fichier.

    Returns:
        Les entrées (liste vide si le fichier n'existe pas).

    Raises:
        CheckFailed: fichier illisible ou sans liste ``errors``.
    """
    target = Path(path)
    if not target.is_file():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckFailed(f"{target.name} illisible : {type(exc).__name__}")
    entries = payload.get("errors") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise CheckFailed(f"{target.name} sans liste errors")
    return [e for e in entries if isinstance(e, dict)]


def blocking_ocr_errors(entries: Iterable[Mapping[str, Any]], item_key: str) -> List[str]:
    """Types d'erreur bloquants (``BLOCKING_OCR_ERRORS``) consignés pour ``item_key``, dans l'ordre."""
    return [str(e.get("error_type")) for e in entries
            if str(e.get("itemKey")) == str(item_key) and e.get("error_type") in BLOCKING_OCR_ERRORS]


def scan_row_problems(doc: "CorpusDoc", row: Mapping[str, Any], errors: Iterable[Mapping[str, Any]]) -> List[str]:
    """Défauts de la ligne ``output.csv`` du PDF numérisé (liste vide si l'OCR Albert est complet).

    Conforme : fournisseur ``albert_…``, marqueurs ``<!-- Page N -->`` de 1 à
    ``doc.pages``, ``texteocr_partial`` faux, et aucune entrée bloquante
    (``BLOCKING_OCR_ERRORS``) pour son ``itemKey`` dans le fichier d'erreurs.
    Une page LightOnOCR en échec garde son marqueur : seuls le drapeau partiel
    et le fichier d'erreurs la révèlent.

    Args:
        doc: le document numérisé du corpus.
        row: sa ligne de ``output.csv``.
        errors: entrées de ``output_errors.json``.

    Returns:
        Les défauts.
    """
    problems = []
    provider = str(row.get("texteocr_provider") or "")
    if not provider.startswith("albert_"):
        problems.append(f"{doc.filename} (numérisé) : fournisseur {provider!r} (attendu albert_…)")
    markers = page_markers(row.get("texteocr"))
    if not markers_continuous(markers, doc.pages):
        problems.append(f"{doc.filename} (numérisé) : marqueurs {markers} (attendu 1 à {doc.pages})")
    if partial_flag_set(row.get("texteocr_partial")):
        problems.append(f"{doc.filename} (numérisé) : OCR partiel (texteocr_partial="
                        f"{row.get('texteocr_partial')!r}, pages {row.get('texteocr_pages_done')!r}/"
                        f"{row.get('texteocr_pages_total')!r})")
    blocking = blocking_ocr_errors(errors, doc.key)
    if blocking:
        problems.append(f"{doc.filename} (numérisé) : output_errors.json signale {', '.join(blocking)}")
    return problems


def recode_status_counts(chunks: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Nombre de chunks par ``recode_status`` (statut absent compté sous ``None``)."""
    counts: Dict[str, int] = {}
    for chunk in chunks:
        status = str(chunk.get("recode_status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_counts(counts: Mapping[str, int]) -> str:
    """Compteurs ``statut=n`` triés, séparés par des virgules."""
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "aucun"


def recode_status_problems(chunks: Sequence[Mapping[str, Any]]) -> List[str]:
    """Défauts de recodage d'une liste de chunks (liste vide si Albert a bien recodé).

    Conforme : au moins un chunk ``recoded`` ; tout autre chunk est
    ``recoded``, ou ``skipped`` pour un fournisseur d'OCR exclu du recodage.
    ``albert_lightonocr`` n'est jamais exclu ici (le serveur tourne avec
    ``ALBERT_OCR_SKIP_RECODE=0``). Un repli (``fallback_raw``,
    ``fallback_truncated``) ou un statut absent est un défaut : le recodage
    retombe sur le texte brut sans faire échouer la route.

    Args:
        chunks: chunks de ``output_chunks.json``.

    Returns:
        Les défauts, chacun avec les compteurs par statut.
    """
    if not chunks:
        return ["aucun chunk"]
    counts = _format_counts(recode_status_counts(chunks))
    recoded = [c for c in chunks if c.get("recode_status") == RECODED_STATUS]
    others = [c for c in chunks if c.get("recode_status") not in (RECODED_STATUS, SKIPPED_STATUS)]
    lighton_skipped = [c for c in chunks if c.get("recode_status") == SKIPPED_STATUS
                       and str(c.get("texteocr_provider") or "").strip().lower() == LIGHTONOCR_PROVIDER]
    problems = []
    if not recoded:
        problems.append(f"aucun chunk recodé par Albert (recode_status : {counts})")
    if others:
        problems.append(f"{len(others)} chunk(s) non recodé(s) par Albert (recode_status : {counts})")
    if lighton_skipped:
        problems.append(f"{len(lighton_skipped)} chunk(s) {LIGHTONOCR_PROVIDER} non recodé(s) alors que "
                        f"ALBERT_OCR_SKIP_RECODE=0 (recode_status : {counts})")
    return problems


def ledger_info(records: Iterable[Mapping[str, Any]], roles: Sequence[str]) -> Dict[str, Any]:
    """Synthèse des enregistrements du ledger d'usage dont le rôle est dans ``roles``.

    Args:
        records: enregistrements de ``albert_usage.jsonl``.
        roles: rôles retenus.

    Returns:
        ``{"enregistrements", "modeles_servis", "avec_cout", "cout"}`` : nombre
        d'enregistrements, modèles non vides, enregistrements portant un modèle
        et un coût numérique, et somme de ces coûts (``None`` s'il n'y en a pas).
    """
    selected = [r for r in records if r.get("role") in roles]
    costed = [r for r in selected if r.get("model") and isinstance(r.get("cost"), (int, float))
              and not isinstance(r.get("cost"), bool) and math.isfinite(float(r["cost"]))]
    return {
        "enregistrements": len(selected),
        "modeles_servis": sorted({str(r.get("model")) for r in selected if r.get("model")}),
        "avec_cout": len(costed),
        "cout": round(sum(float(r["cost"]) for r in costed), 6) if costed else None,
    }


def recode_ledger_problem(records: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """Défaut du ledger après le recodage : aucun enregistrement ``role=recode`` portant son modèle."""
    if not ledger_info(records, (RECODE_ROLE,))["modeles_servis"]:
        return "albert_usage.jsonl sans enregistrement role=recode portant le modèle servi"
    return None


def notes_pinned_model() -> str:
    """Id épinglé du catalogue pour ``NOTES_MODEL``, sans le préfixe ``albert/`` (``gpt-oss-120b``)."""
    _ensure_repo_on_path()
    from scripts.rad_albert import catalog

    name = NOTES_MODEL[len(ALBERT_PREFIX):] if NOTES_MODEL.startswith(ALBERT_PREFIX) else NOTES_MODEL
    return catalog.canonical_id(name)


def notes_ledger_verdict(records: Iterable[Mapping[str, Any]],
                         required: Optional[bool] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """Évalue le ledger des notes (rôles ``NOTES_ROLES``).

    Args:
        records: enregistrements de ``albert_usage.jsonl``.
        required: absence d'enregistrement de notes bloquante (défaut :
            ``NOTES_LEDGER_REQUIRED``).

    Returns:
        ``(défaut ou None, détail)`` : sans enregistrement de notes, défaut si
        ``required``, sinon détail marqué informatif ; avec des enregistrements,
        défaut si aucun n'a pour modèle l'id épinglé de gpt-oss
        (``notes_pinned_model``, un alias ne suffit pas) avec un coût numérique.
        Le détail est celui de ``ledger_info``, complété de ``modele_epingle``,
        ``avec_modele_epingle`` (enregistrements conformes) et ``statut``.
    """
    records = list(records)
    strict = NOTES_LEDGER_REQUIRED if required is None else bool(required)
    info = ledger_info(records, NOTES_ROLES)
    pinned = notes_pinned_model()
    info["modele_epingle"] = pinned
    info["avec_modele_epingle"] = ledger_info(
        [r for r in records if r.get("model") == pinned], NOTES_ROLES)["avec_cout"]
    if not info["enregistrements"]:
        if strict:
            info["statut"] = "exigé"
            return "albert_usage.jsonl sans enregistrement des notes", info
        info["statut"] = NOTES_LEDGER_PENDING
        return None, info
    info["statut"] = "vérifié"
    if not info["avec_modele_epingle"]:
        served = ", ".join(info["modeles_servis"]) or "aucun"
        return (f"albert_usage.jsonl : aucun enregistrement des notes servi par {pinned} avec un coût "
                f"numérique (modèles servis : {served})"), info
    return None, info


def wait_for_notes_ledger(path: Any, attempts: Optional[int] = None, wait_s: Optional[float] = None,
                          sleep: Optional[Callable[[float], None]] = None) -> List[Dict[str, Any]]:
    """Enregistrements de ``albert_usage.jsonl``, relus tant qu'aucun ne vient des notes.

    La route des notes écrit le journal dans son bloc ``finally``, donc
    éventuellement juste après l'événement ``complete`` lu par le client.

    Args:
        path: chemin du journal.
        attempts: relectures au plus (défaut : ``NOTES_LEDGER_ATTEMPTS``).
        wait_s: attente avant chaque relecture, en secondes (défaut :
            ``NOTES_LEDGER_WAIT_S``).
        sleep: fonction d'attente (défaut : ``time.sleep``).

    Returns:
        Les enregistrements de la dernière lecture (vide si le fichier manque).
    """
    attempts = NOTES_LEDGER_ATTEMPTS if attempts is None else attempts
    wait_s = NOTES_LEDGER_WAIT_S if wait_s is None else wait_s
    sleep = time.sleep if sleep is None else sleep
    records = read_jsonl(path)
    for _attempt in range(max(0, int(attempts))):
        if ledger_info(records, NOTES_ROLES)["enregistrements"]:
            break
        sleep(wait_s)
        records = read_jsonl(path)
    return records


def manifest_label(session: Any) -> str:
    """Préfixe des copies de manifeste d'une session (même règle que la route ``upload_db``)."""
    label = SAFE_LABEL_RE.sub("_", str(session or "")).strip("._")
    return label[:SAFE_LABEL_MAX_CHARS] or "session"


def manifest_copies(manifest_dir: Any, session: Any, before: Iterable[str] = ()) -> List[Path]:
    """Copies de manifeste d'une session créées pendant le run.

    Args:
        manifest_dir: ``data/albert_manifests/<user_id>``.
        session: dossier de session (chemin relatif sous ``uploads/``).
        before: noms de fichiers présents avant le run (jamais renvoyés).

    Returns:
        Les fichiers ``<label>-<ts>.jsonl`` nouveaux, triés.
    """
    folder = Path(manifest_dir)
    if not folder.is_dir():
        return []
    prefix = manifest_label(session) + "-"
    known = set(before)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.name.startswith(prefix) and p.name.endswith(".jsonl") and p.name not in known)


def upload_entries_for(upload_dir: Any, zip_stem: str) -> List[Path]:
    """Dossiers de session et archives créés sous ``uploads/`` pour le zip ``<zip_stem>.zip``.

    La route ``upload_zip`` nomme la session ``<8 hex>_<nom du zip>`` et garde
    l'archive ``<8 hex>_<nom du zip>.zip`` à côté.

    Args:
        upload_dir: dossier ``uploads/``.
        zip_stem: nom du zip sans extension (unique par run).

    Returns:
        Les entrées correspondantes, triées (jamais d'autre entrée).
    """
    base = Path(upload_dir)
    if not zip_stem or not base.is_dir():
        return []
    pattern = re.compile(r"[0-9a-f]{8}_" + re.escape(zip_stem) + r"(?:\.zip)?")
    return sorted(p for p in base.iterdir() if pattern.fullmatch(p.name))


def remove_upload_entries(upload_dir: Any, entries: Iterable[Path]) -> List[str]:
    """Supprime des entrées renvoyées par ``upload_entries_for`` (jamais hors de ``upload_dir``).

    Returns:
        Les noms supprimés.
    """
    root = os.path.realpath(str(upload_dir))
    removed: List[str] = []
    for entry in entries:
        path = Path(entry)
        if os.path.realpath(str(path.parent)) != root:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        else:
            continue
        removed.append(path.name)
    return removed


def files_containing(entries: Iterable[Tuple[Any, int]], secret: str) -> List[str]:
    """Fichiers dont la partie écrite après ``offset`` contient ``secret``.

    Args:
        entries: couples ``(chemin, offset)`` ; un offset hors du fichier
            (journal tourné) fait relire depuis le début.
        secret: chaîne recherchée (jamais affichée).

    Returns:
        Les chemins où la chaîne apparaît.
    """
    if not secret:
        return []
    needle = secret.encode("utf-8")
    overlap = max(len(needle) - 1, 0)
    hits: List[str] = []
    for path, offset in entries:
        try:
            with open(path, "rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                handle.seek(offset if 0 <= offset <= size else 0)
                tail = b""
                while True:
                    block = handle.read(1 << 20)
                    if not block:
                        break
                    data = tail + block
                    if needle in data:
                        hits.append(str(path))
                        break
                    tail = data[-overlap:] if overlap else b""
        except OSError:
            continue
    return hits


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusDoc:
    """Un document du corpus : item Zotero et PDF joint."""

    key: str
    title: str
    filename: str
    kind: str
    pages: int
    relpath: str
    abstract: str


@dataclass
class Corpus:
    """Corpus construit : dossier racine (JSON Zotero et ``files/``) et documents."""

    root: Path
    json_name: str
    docs: List[CorpusDoc] = field(default_factory=list)

    @property
    def scanned(self) -> List[CorpusDoc]:
        """Documents sans couche texte (PDF image seule)."""
        return [d for d in self.docs if d.kind == "scan"]

    def path_of(self, doc: CorpusDoc) -> Path:
        """Chemin local du PDF d'un document."""
        return self.root / doc.relpath


def _page_texts(title: str, stamp: str, index: int, pages: int) -> List[str]:
    """Textes des pages d'un document (marqueur unique au run : contenus jamais dédupliqués entre runs)."""
    texts = []
    for page in range(1, pages + 1):
        sentences = [_TEXT_SENTENCES[(index * 3 + page + k) % len(_TEXT_SENTENCES)] for k in range(9)]
        header = f"{title}\n" if page == 1 else ""
        header += f"Référence de contrôle {stamp}, document {index}, page {page} sur {pages}.\n\n"
        texts.append(header + " ".join(sentences))
    return texts


def _text_document(page_texts: Sequence[str]) -> Any:
    """PDF texte en mémoire (une page A4 par texte)."""
    import fitz

    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        left = page.insert_textbox(fitz.Rect(56, 56, 539, 786), text, fontsize=11, fontname="helv")
        if left < 0:
            doc.close()
            raise ValueError("texte trop long pour une page A4")
    return doc


def _image_only_document(page_texts: Sequence[str], dpi: int = SCAN_DPI) -> Any:
    """PDF image seule : chaque page texte est rastérisée puis réinsérée comme image."""
    import fitz

    source = _text_document(page_texts)
    scan = fitz.open()
    try:
        for page in source:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            target = scan.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=pixmap.tobytes("png"))
    finally:
        source.close()
    return scan


def build_corpus(dest_dir: Any, size: int = DEFAULT_CORPUS_SIZE, stamp: str = "run",
                 pages: int = PAGES_PER_DOC) -> Corpus:
    """Construit le corpus : ``size-1`` PDF texte, 1 PDF image seule et l'export Zotero JSON.

    Args:
        dest_dir: dossier racine du corpus (créé ; le JSON et ``files/`` y sont écrits).
        size: nombre de documents (2 au moins).
        stamp: horodatage du run (nom du JSON, marqueur unique dans les textes).
        pages: pages par document.

    Returns:
        Le corpus.

    Raises:
        ValueError: ``size`` inférieur à 2.
    """
    if size < MIN_CORPUS_SIZE:
        raise ValueError(f"corpus de {size} document(s) : {MIN_CORPUS_SIZE} au moins (texte et numérisé)")
    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(root=root, json_name=f"{CORPUS_PREFIX}{stamp}.json")
    items = []
    for index in range(1, size + 1):
        scanned = index == size
        key = f"E2E{index:05d}"
        if scanned:
            title = _SCAN_TITLE
            filename = f"numerise-{index}.pdf"
        else:
            base = _TEXT_TITLES[(index - 1) % len(_TEXT_TITLES)]
            title = base if index <= len(_TEXT_TITLES) else f"{base} ({index})"
            filename = f"texte-{index}.pdf"
        relpath = f"files/{key}/{filename}"
        abstract = f"Document de contrôle {index} du corpus de bout en bout ({stamp})."
        doc = CorpusDoc(key=key, title=title, filename=filename, kind="scan" if scanned else "text",
                        pages=pages, relpath=relpath, abstract=abstract)
        target = corpus.path_of(doc)
        target.parent.mkdir(parents=True, exist_ok=True)
        texts = _page_texts(title, stamp, index, pages)
        pdf = _image_only_document(texts) if scanned else _text_document(texts)
        try:
            pdf.save(str(target), garbage=4, deflate=True)
        finally:
            pdf.close()
        corpus.docs.append(doc)
        last, first = _AUTHORS[(index - 1) % len(_AUTHORS)]
        items.append({
            "key": key,
            "itemType": "book" if scanned else "journalArticle",
            "title": title,
            "creators": [{"creatorType": "author", "lastName": last, "firstName": first}],
            "date": "2024",
            "abstractNote": abstract,
            "DOI": "",
            "url": "",
            "language": "fr",
            "attachments": [{"title": "Texte intégral PDF", "path": relpath, "contentType": "application/pdf"}],
        })
    (root / corpus.json_name).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return corpus


def zip_corpus(corpus: Corpus, zip_path: Any) -> Path:
    """Zippe le corpus à plat : le JSON Zotero et ``files/`` à la racine de l'archive.

    Deux entrées à la racine : la route ``upload_zip`` traite alors le dossier
    de session lui-même (pas de sous-dossier unique à descendre).

    Args:
        corpus: le corpus construit.
        zip_path: chemin de l'archive à écrire.

    Returns:
        Le chemin de l'archive.
    """
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(corpus.root / corpus.json_name, arcname=corpus.json_name)
        for doc in corpus.docs:
            archive.write(corpus.path_of(doc), arcname=doc.relpath)
    return target


def write_minimal_output_csv(path: Any) -> Path:
    """Écrit un ``output.csv`` d'une ligne (format de ``rad_dataframe``) s'il n'existe pas.

    Sert aux contrôles qui exigent la présence du fichier mais échouent avant
    tout traitement (refus d'identifiant, Albert désactivé).

    Returns:
        Le chemin du fichier.
    """
    target = Path(path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["itemKey", "type", "title", "abstract", "date", "url", "doi", "authors", "filename", "path",
              "attachment_title", "texteocr", "texteocr_provider", "texteocr_partial",
              "texteocr_pages_done", "texteocr_pages_total"]
    row = {name: "" for name in fields}
    row.update({"itemKey": "E2E00000", "type": "journalArticle", "title": "Document de contrôle minimal",
                "filename": "controle.pdf", "texteocr": "<!-- Page 1 -->\nTexte minimal de contrôle.",
                "texteocr_provider": "legacy", "texteocr_partial": "False", "texteocr_pages_done": "1",
                "texteocr_pages_total": "1"})
    with open(target, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, escapechar="\\", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerow(row)
    return target


def read_output_csv(path: Any) -> List[Dict[str, str]]:
    """Lignes de ``output.csv`` (UTF-8 avec BOM, échappement ``\\`` comme ``rad_dataframe``)."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, escapechar="\\"))


def load_json_list(path: Any) -> List[Dict[str, Any]]:
    """Liste JSON d'un fichier de chunks ou de notes.

    Raises:
        CheckFailed: fichier absent, illisible ou qui n'est pas une liste.
    """
    target = Path(path)
    if not target.is_file():
        raise CheckFailed(f"{target.name} absent")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckFailed(f"{target.name} illisible : {type(exc).__name__}")
    if not isinstance(data, list):
        raise CheckFailed(f"{target.name} n'est pas une liste JSON")
    return [d for d in data if isinstance(d, dict)]


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    """Enregistrements d'un fichier JSONL (lignes illisibles ignorées ; vide s'il manque)."""
    target = Path(path)
    if not target.is_file():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Collections Albert (nettoyage)
# ---------------------------------------------------------------------------
def delete_collections_named(client: Any, name: str) -> List[int]:
    """Supprime les collections privées dont le nom vaut exactement ``name``.

    Args:
        client: ``AlbertClient`` (ou double de test de même interface).
        name: nom exact (correspondance exacte, doublons compris).

    Returns:
        Les ids supprimés.
    """
    _ensure_repo_on_path()
    from scripts.rad_albert.collections import is_private

    deleted: List[int] = []
    for collection in client.list_collections(name=name):
        if not isinstance(collection, Mapping) or collection.get("name") != name or not is_private(collection):
            continue
        client.delete_collection(collection["id"])
        deleted.append(int(collection["id"]))
    return deleted


def count_collections_named(client: Any, name: str) -> int:
    """Nombre de collections dont le nom vaut exactement ``name``."""
    return sum(1 for c in client.list_collections(name=name) if isinstance(c, Mapping) and c.get("name") == name)


def is_smoke_collection_name(name: Any) -> bool:
    """Vrai si ``name`` est un nom de collection de ce script (``ragpy-probe-e2e-AAAAMMJJ-HHMMSS``)."""
    return isinstance(name, str) and SMOKE_COLLECTION_RE.fullmatch(name) is not None


def sweep_collections(client: Any) -> List[Tuple[int, str]]:
    """Supprime les collections privées restantes des runs précédents de ce script.

    Seuls les noms ``ragpy-probe-e2e-AAAAMMJJ-HHMMSS`` sont balayés : une
    collection de la suite live (``ragpy-probe-e2e-AAAAMMJJTHHMMSS-<pid>``),
    peut-être en cours d'utilisation, ne l'est jamais.

    Args:
        client: ``AlbertClient`` (ou double de test de même interface).

    Returns:
        Les couples ``(id, nom)`` supprimés.
    """
    _ensure_repo_on_path()
    from scripts.rad_albert.collections import is_private

    deleted: List[Tuple[int, str]] = []
    for collection in client.list_collections(visibility="private"):
        if not isinstance(collection, Mapping):
            continue
        name = collection.get("name")
        if not is_smoke_collection_name(name) or not is_private(collection):
            continue
        client.delete_collection(collection["id"])
        deleted.append((int(collection["id"]), name))
    return deleted


# ---------------------------------------------------------------------------
# Serveur et HTTP
# ---------------------------------------------------------------------------
def auth_headers(token: str) -> Dict[str, str]:
    """En-têtes d'authentification (Bearer, et cookie pour les pages HTML)."""
    return {"Authorization": "Bearer " + token, "Cookie": "access_token=" + token}


@dataclass
class SseOutcome:
    """Issue d'un appel SSE : statut HTTP, événements lus et événement final."""

    status_code: int
    events: List[Dict[str, Any]]
    terminal: Optional[Dict[str, Any]]
    body: str = ""
    timed_out: bool = False

    def error_messages(self, limit: int = 5) -> List[str]:
        """Messages des derniers événements ``error`` (lignes de journal comprises)."""
        errors = [str(e.get("message", ""))[:300] for e in self.events if e.get("type") == "error"]
        return errors[-limit:]


class ServerProcess:
    """Serveur ``uvicorn`` en sous-processus, dans son propre groupe de processus."""

    def __init__(self, env: Mapping[str, str], log_path: Path, label: str) -> None:
        """Prépare le serveur (aucun processus lancé).

        Args:
            env: environnement complet du serveur.
            log_path: fichier qui reçoit stdout et stderr.
            label: ``on`` ou ``off``.
        """
        self.env = dict(env)
        self.log_path = log_path
        self.label = label
        self.port: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self._log = None
        self._stopped = False

    @property
    def base_url(self) -> str:
        """URL de base du serveur."""
        return f"http://127.0.0.1:{self.port}"

    def command(self, port: int) -> List[str]:
        """Commande du serveur : ``[sys.executable, -m, uvicorn, app.main:app]`` sur ``port``."""
        return [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]

    def start(self) -> None:
        """Lance la commande du serveur sur un port libre, dans une nouvelle session (groupe propre)."""
        self.port = free_port()
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.command(self.port), cwd=str(REPO), env=self.env,
                                     stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def wait_ready(self, timeout: float = SERVER_START_TIMEOUT_S) -> Optional[str]:
        """Attend ``GET /health`` en 200.

        Returns:
            ``None`` quand le serveur répond, sinon la raison de l'échec.
        """
        import httpx

        deadline = time.monotonic() + timeout
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.proc is None or self.proc.poll() is not None:
                    code = self.proc.returncode if self.proc is not None else None
                    return f"le serveur s'est arrêté (code {code})"
                try:
                    if client.get(self.base_url + "/health").status_code == 200:
                        return None
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
        return f"le serveur ne répond pas après {int(timeout)} s"

    def stop(self) -> None:
        """Arrête le serveur et tout son groupe de processus (SIGTERM, puis SIGKILL) ; une seule fois."""
        if self._stopped:
            return
        self._stopped = True
        try:
            proc = self.proc
            if proc is not None:
                if proc.poll() is None:
                    self._signal_group(signal.SIGTERM)
                    try:
                        proc.wait(timeout=SERVER_STOP_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        self._signal_group(signal.SIGKILL)
                        proc.wait(timeout=SERVER_STOP_TIMEOUT_S)
                # Sous-processus de traitement éventuellement restés dans le groupe.
                self._signal_group(signal.SIGKILL)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    def _signal_group(self, sig: int) -> None:
        """Envoie ``sig`` au groupe du serveur (processus seul hors POSIX)."""
        if self.proc is None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(self.proc.pid, sig)
            elif self.proc.poll() is None:
                self.proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def log_tail(self, lines: int = 30) -> List[str]:
        """Dernières lignes du journal du serveur."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]


class Api:
    """Client HTTP du serveur local (sans proxy, sans redirection)."""

    def __init__(self, base_url: str) -> None:
        """Ouvre le client vers ``base_url``."""
        import httpx

        self._httpx = httpx
        self.client = httpx.Client(base_url=base_url, trust_env=False, follow_redirects=False,
                                   timeout=httpx.Timeout(HTTP_TIMEOUT_S, connect=10.0))

    def close(self) -> None:
        """Ferme le client."""
        self.client.close()

    def request(self, method: str, path: str, token: Optional[str] = None, *,
                timeout: Optional[float] = None, **kwargs: Any) -> Any:
        """Requête authentifiée (``httpx.Response``)."""
        headers = dict(kwargs.pop("headers", None) or {})
        if token:
            headers.update(auth_headers(token))
        return self.client.request(method, path, headers=headers,
                                   timeout=timeout if timeout is not None else HTTP_TIMEOUT_S, **kwargs)

    def sse(self, path: str, token: str, form: Mapping[str, str], *,
            step_timeout: float = STEP_TIMEOUT_S) -> SseOutcome:
        """POST d'un formulaire vers une route SSE, flux lu jusqu'à ``complete`` ou jusqu'à sa fin.

        Args:
            path: route.
            token: jeton de l'utilisateur.
            form: champs du formulaire.
            step_timeout: durée maximale de l'étape (vérifiée à chaque événement ;
                les routes à sous-processus émettent un battement toutes les 15 s).

        Returns:
            L'issue du flux.
        """
        httpx = self._httpx
        timeout = httpx.Timeout(connect=10.0, read=SSE_READ_TIMEOUT_S, write=60.0, pool=10.0)
        events: List[Dict[str, Any]] = []
        deadline = time.monotonic() + step_timeout
        with self.client.stream("POST", path, data=dict(form), headers=auth_headers(token), timeout=timeout) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                return SseOutcome(resp.status_code, [], None, body[:2000])
            for event in iter_sse_events(resp.iter_lines()):
                if event.get("type") != "heartbeat":
                    events.append(event)
                if event.get("type") == "complete":
                    return SseOutcome(200, events, dict(event))
                if time.monotonic() > deadline:
                    return SseOutcome(200, events, None, "délai de l'étape dépassé", timed_out=True)
        return SseOutcome(200, events, terminal_event(events))


def _json_body(resp: Any) -> Dict[str, Any]:
    """Corps JSON d'une réponse (dictionnaire vide s'il n'en est pas un)."""
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _excerpt(resp: Any, limit: int = 600) -> str:
    """Début du corps d'une réponse."""
    try:
        return resp.text[:limit]
    except Exception:
        return ""


def require_complete(outcome: SseOutcome) -> Dict[str, Any]:
    """Exige un flux SSE terminé par ``complete`` ; renvoie cet événement.

    Raises:
        CheckFailed: statut HTTP inattendu, délai dépassé, flux sans issue ou issue ``error``.
    """
    if outcome.status_code != 200:
        raise CheckFailed(f"HTTP {outcome.status_code}", {"corps": outcome.body[:600]})
    if outcome.timed_out:
        raise CheckFailed("délai de l'étape dépassé", {"erreurs": outcome.error_messages()})
    terminal = outcome.terminal
    if terminal is None:
        raise CheckFailed("flux SSE terminé sans événement complete ni error", {"evenements": len(outcome.events)})
    if terminal.get("type") != "complete":
        detail = {"erreurs": outcome.error_messages()}
        if terminal.get("credential_required"):
            detail["credential_required"] = terminal.get("credential_required")
        raise CheckFailed(f"événement final error : {str(terminal.get('message', ''))[:300]}", detail)
    return terminal


# ---------------------------------------------------------------------------
# Déroulé du test
# ---------------------------------------------------------------------------
class SmokeRun:
    """Un run E2E : préparation, contrôles ON / isolation / OFF, nettoyage et résumé."""

    def __init__(self, args: argparse.Namespace, key: str, server_path: str, *,
                 env_file: Any = ENV_FILE, log: Callable[[str], None] = print) -> None:
        """Prépare le run (aucun effet de bord).

        Args:
            args: options de la ligne de commande.
            key: clé Albert de la personne A (jamais affichée).
            server_path: ``PATH`` du serveur (``.venv/bin`` en tête).
            env_file: fichier ``.env`` du dépôt (clé admin du contrôle d'isolation).
            log: fonction d'affichage (chaque ligne est masquée avant).
        """
        self.args = args
        self.key = key
        self.server_path = server_path
        self.env_file = env_file
        self._log = log
        self.stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.started = datetime.now(timezone.utc)
        self.collection_name = f"{COLLECTION_PREFIX}{self.stamp}"
        self.zip_stem = f"{CORPUS_PREFIX}{self.stamp}"
        self.tmpdir: Optional[Path] = None
        self.db_url: Optional[str] = None
        self.corpus: Optional[Corpus] = None
        self.zip_path: Optional[Path] = None
        self.jwt_secret: Optional[str] = None
        self.user_ids: Dict[str, int] = {}
        self.project_ids: Dict[str, int] = {}
        self.tokens: Dict[str, str] = {}
        self.sessions: Dict[str, str] = {}
        self.server: Optional[ServerProcess] = None
        self.servers: List[ServerProcess] = []
        self.api: Optional[Api] = None
        self.ports: Dict[str, int] = {}
        self.first_inserted: Optional[int] = None
        self.manifest_before: set = set()
        self.manifest_dir_existed = False
        self.manifest_root_existed = MANIFEST_ROOT.is_dir()
        self.log_offsets: Dict[str, int] = {}
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {group: {} for group, _checks in CHECK_GROUPS}
        self.setup_errors: List[str] = []
        self.extras: List[str] = []
        self.cleanup_report: Dict[str, Any] = {}
        self._engine = None
        self._session_factory = None

    # ----------------------------------------------------------------- sorties
    def say(self, message: str) -> None:
        """Affiche une ligne, clé masquée."""
        self._log(redact(str(message), [self.key]))

    def setup_error(self, step: str, exc: BaseException) -> None:
        """Consigne l'échec d'une étape de préparation (message masqué)."""
        text = f"{step} : {type(exc).__name__} : {exc}"
        self.setup_errors.append(redact(text, [self.key])[:600])
        self.say(f"[préparation] ÉCHEC {text[:400]}")

    # ------------------------------------------------------------- préparation
    def execute(self) -> None:
        """Déroule le run complet (le nettoyage est appelé par ``main`` dans un ``finally``)."""
        self.say(f"E2E Albert : run {self.stamp}, corpus de {self.args.corpus_size} PDF, "
                 f"collection {self.collection_name}")
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ragpy-e2e-albert-"))
        self.db_url = f"sqlite:///{self.tmpdir / 'e2e.db'}"
        self._snapshot_logs()
        self.corpus = build_corpus(self.tmpdir / "corpus", self.args.corpus_size, self.stamp)
        self.zip_path = zip_corpus(self.corpus, self.tmpdir / f"{self.zip_stem}.zip")
        self.say(f"Corpus : {len(self.corpus.docs)} PDF ({len(self.corpus.scanned)} numérisé), "
                 f"archive {self.zip_path.name}")
        self.prepare_database()
        self._snapshot_manifests()
        if self.args.cleanup:
            self.initial_sweep()
        self.phase_on()
        self.phase_off()

    def _snapshot_logs(self) -> None:
        """Mémorise la taille des journaux de ``logs/`` (recherche de la clé dans leur suite)."""
        if LOG_DIR.is_dir():
            for path in LOG_DIR.glob("*.log"):
                try:
                    self.log_offsets[str(path)] = path.stat().st_size
                except OSError:
                    continue

    def prepare_database(self) -> None:
        """Crée la base temporaire, les utilisateurs, les projets et les jetons (serveur arrêté)."""
        os.environ["DATABASE_URL"] = str(self.db_url)
        _ensure_repo_on_path()
        from app.config import settings
        from app.core.security import create_access_token, get_password_hash
        from app.database.init_db import init_database
        from app.database.session import SessionLocal, engine
        from app.models.project import Project
        from app.models.user import User

        if settings.DATABASE_URL != self.db_url:
            raise RuntimeError("DATABASE_URL non pris en compte : module de base déjà importé.")
        init_database()
        self._engine = engine
        self._session_factory = SessionLocal
        db = SessionLocal()
        try:
            for persona, roles in (("admin", ["USER", "ADMIN"]), ("a", ["USER"]), ("b", ["USER"])):
                user = User(
                    id=E2E_USER_IDS[persona],
                    email=f"e2e-{persona}-{self.stamp}@example.org",
                    hashed_password=get_password_hash(secrets.token_urlsafe(24)),
                    first_name="E2E",
                    last_name=persona.upper(),
                    roles=list(roles),
                    is_active=True,
                    is_verified=True,
                    is_pending_approval=False,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                self.user_ids[persona] = int(user.id)
            for persona in ("a", "b"):
                project = Project(
                    name=f"Corpus de bout en bout {persona.upper()} {self.stamp}",
                    description="Projet de test de bout en bout, supprimé avec la base temporaire.",
                    owner_id=self.user_ids[persona],
                )
                db.add(project)
                db.commit()
                db.refresh(project)
                self.project_ids[persona] = int(project.id)
        finally:
            db.close()
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.tokens = {persona: create_access_token(subject=str(uid), expires_delta=TOKEN_TTL)
                       for persona, uid in self.user_ids.items()}
        self.say(f"Base temporaire prête : admin, A et B (ids {self.user_ids['admin']}, "
                 f"{self.user_ids['a']}, {self.user_ids['b']}), projets de A et B.")

    def _manifest_dir(self) -> Optional[Path]:
        """``data/albert_manifests/<id de A>``."""
        if "a" not in self.user_ids:
            return None
        return MANIFEST_ROOT / str(self.user_ids["a"])

    def _snapshot_manifests(self) -> None:
        """Mémorise les copies de manifeste déjà présentes pour l'id de A (jamais supprimées)."""
        folder = self._manifest_dir()
        if folder is not None and folder.is_dir():
            self.manifest_dir_existed = True
            self.manifest_before = {p.name for p in folder.iterdir()}

    def albert_client(self) -> Any:
        """``AlbertClient`` de nettoyage (clé de A, configuration du serveur, sans limiteur)."""
        _ensure_repo_on_path()
        from scripts.rad_albert.client import AlbertClient
        from scripts.rad_albert.config import AlbertConfig

        try:
            cfg = AlbertConfig.from_env({**os.environ, "ALBERT_ENABLED": "1"})
        except ValueError:
            cfg = AlbertConfig.from_env({})
        return AlbertClient(cfg, self.key, use_limiter=False)

    def initial_sweep(self) -> None:
        """``--cleanup`` : supprime les collections ``ragpy-probe-e2e-*`` restantes."""
        client = None
        try:
            client = self.albert_client()
            deleted = sweep_collections(client)
            self.cleanup_report["balayage_initial"] = [{"id": cid, "nom": name} for cid, name in deleted]
            self.say(f"Balayage initial : {len(deleted)} collection(s) {COLLECTION_PREFIX}AAAAMMJJ-HHMMSS "
                     "supprimée(s).")
        except Exception as exc:
            self.setup_error("balayage initial des collections", exc)
        finally:
            if client is not None:
                client.close()

    def server_env(self, enabled: bool) -> Dict[str, str]:
        """Environnement du serveur (Albert ON ou OFF), jamais écrit dans un fichier."""
        env = dict(os.environ)
        env.pop("ALBERT_LIVE", None)
        env["PATH"] = self.server_path
        env["DATABASE_URL"] = str(self.db_url)
        env["JWT_SECRET_KEY"] = str(self.jwt_secret)
        env["ALBERT_ENABLED"] = "1" if enabled else "0"
        env["OCR_ENABLE_ALBERT"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(PINNED_ENV)
        return env

    def start_server(self, enabled: bool) -> None:
        """Démarre le serveur et attend qu'il réponde.

        Raises:
            RuntimeError: le serveur ne démarre pas (fin du journal affichée, masquée).
        """
        label = "on" if enabled else "off"
        server = ServerProcess(self.server_env(enabled), Path(self.tmpdir) / f"serveur-{label}.log", label)
        self.server = server
        self.servers.append(server)
        server.start()
        self.ports[label] = int(server.port or 0)
        problem = server.wait_ready()
        if problem:
            for line in server.log_tail():
                self.say(f"  | {line}")
            raise RuntimeError(f"serveur {label.upper()} : {problem}")
        self.api = Api(server.base_url)
        self.say(f"Serveur {label.upper()} prêt sur le port {server.port} (ALBERT_ENABLED={'1' if enabled else '0'}).")

    def _close_api(self) -> None:
        """Ferme le client HTTP du serveur courant (s'il est ouvert)."""
        api, self.api = self.api, None
        if api is not None:
            api.close()

    def stop_server(self) -> None:
        """Arrête le serveur courant et ferme le client HTTP."""
        self._close_api()
        if self.server is not None:
            self.server.stop()
            self.server = None

    def session_dir(self, persona: str) -> Path:
        """Dossier de session d'un utilisateur.

        Raises:
            CheckFailed: aucune session (envoi du corpus en échec).
        """
        path = self.sessions.get(persona)
        if not path:
            raise CheckFailed(f"prérequis : aucune session pour {persona.upper()} (envoi du corpus en échec)")
        return UPLOAD_DIR / path

    def upload_corpus(self, persona: str) -> str:
        """Envoie le corpus zippé dans le projet de ``persona`` ; renvoie le dossier de session."""
        with open(self.zip_path, "rb") as handle:
            payload = handle.read()
        resp = self.api.request(
            "POST", f"/api/pipeline/projects/{self.project_ids[persona]}/upload_zip", self.tokens[persona],
            files={"file": (Path(self.zip_path).name, payload, "application/zip")},
        )
        body = _json_body(resp)
        if resp.status_code != 200 or not body.get("path"):
            raise RuntimeError(f"upload_zip de {persona.upper()} : HTTP {resp.status_code} {_excerpt(resp, 300)}")
        self.sessions[persona] = str(body["path"])
        return self.sessions[persona]

    def setup_on(self) -> None:
        """Clé de A par ``PUT /users/me/credentials``, envoi du corpus pour A et B, ``output.csv`` minimal de B."""
        try:
            resp = self.api.request("PUT", "/users/me/credentials", self.tokens["a"],
                                    json={ALBERT_CREDENTIAL_KEY: self.key})
            updated = _json_body(resp).get("updated") or []
            if resp.status_code != 200 or ALBERT_CREDENTIAL_KEY not in updated:
                raise RuntimeError(f"HTTP {resp.status_code} {_excerpt(resp, 300)}")
            self.say("Clé Albert de A enregistrée par PUT /users/me/credentials.")
        except Exception as exc:
            self.setup_error("enregistrement de la clé de A", exc)
        for persona in ("a", "b"):
            try:
                path = self.upload_corpus(persona)
                self.say(f"Corpus envoyé pour {persona.upper()} : session {path}")
            except Exception as exc:
                self.setup_error(f"envoi du corpus pour {persona.upper()}", exc)
        if self.sessions.get("b"):
            try:
                write_minimal_output_csv(UPLOAD_DIR / self.sessions["b"] / "output.csv")
            except Exception as exc:
                self.setup_error("output.csv minimal de B", exc)

    # ------------------------------------------------------------------ phases
    def run_check(self, group: str, cid: str, func: Callable[[], Optional[Mapping[str, Any]]]) -> bool:
        """Exécute un contrôle et consigne son issue (détail masqué)."""
        route = CHECK_ROUTES[cid]
        self.say(f"[{cid}] {route} …")
        started = time.monotonic()
        try:
            detail = dict(func() or {})
            ok, message = True, ""
        except CheckFailed as exc:
            ok, message, detail = False, exc.message, exc.detail
        except Exception as exc:
            ok, message, detail = False, f"{type(exc).__name__} : {exc}", {}
        entry = {"id": cid, "route": route, "ok": ok, "duree_s": round(time.monotonic() - started, 1),
                 "message": message, "detail": detail}
        self.results[group][cid] = redact(entry, [self.key])
        suffix = f" : {message[:400]}" if message else ""
        self.say(f"[{cid}] {'OK' if ok else 'ÉCHEC'}{suffix}")
        return ok

    def phase_on(self) -> None:
        """Serveur ON : préparation, contrôles d'isolation puis contrôles ON 1 à 7."""
        try:
            self.start_server(True)
        except Exception as exc:
            self.setup_error("démarrage du serveur ON", exc)
            self.stop_server()
            return
        try:
            self.setup_on()
            self.run_check("isolation", "iso1", self.check_iso1)
            self.run_check("isolation", "iso2", self.check_iso2)
            for cid, func in (("on1", self.check_on1), ("on2", self.check_on2), ("on3", self.check_on3),
                              ("on4", self.check_on4), ("on5", self.check_on5), ("on6", self.check_on6),
                              ("on7", self.check_on7)):
                self.run_check("on", cid, func)
        finally:
            self.stop_server()

    def phase_off(self) -> None:
        """Serveur redémarré avec ``ALBERT_ENABLED=0`` : contrôles OFF 1 à 3."""
        try:
            self.start_server(False)
        except Exception as exc:
            self.setup_error("démarrage du serveur OFF", exc)
            self.stop_server()
            return
        try:
            self.run_check("off", "off1", self.check_off1)
            self.run_check("off", "off2", self.check_off2)
            self.run_check("off", "off3", self.check_off3)
        finally:
            self.stop_server()

    # ------------------------------------------------------------- contrôles ON
    def check_on1(self) -> Dict[str, Any]:
        """OCR : numérisé lu en entier par Albert (fournisseur, marqueurs, drapeau partiel, fichier d'erreurs)."""
        folder = self.session_dir("a")
        require_complete(self.api.sse("/process_dataframe_sse", self.tokens["a"], {"path": self.sessions["a"]}))
        csv_path = folder / "output.csv"
        if not csv_path.is_file():
            raise CheckFailed("output.csv absent après l'extraction")
        rows = read_output_csv(csv_path)
        errors_path = folder / "output_errors.json"
        errors = read_ocr_errors(errors_path)
        by_name = {row.get("filename"): row for row in rows}
        detail: Dict[str, Any] = {
            "documents": len(rows),
            "fournisseurs": {name: row.get("texteocr_provider") for name, row in by_name.items()},
            "partiels": {name: row.get("texteocr_partial") for name, row in by_name.items()},
            "fichier_erreurs": errors_path.is_file(),
            "erreurs_bloquantes": {doc.filename: blocking_ocr_errors(errors, doc.key)
                                   for doc in self.corpus.docs if blocking_ocr_errors(errors, doc.key)},
        }
        problems = []
        if len(rows) != len(self.corpus.docs):
            problems.append(f"{len(rows)} ligne(s) dans output.csv pour {len(self.corpus.docs)} document(s)")
        for doc in self.corpus.docs:
            row = by_name.get(doc.filename)
            if row is None:
                problems.append(f"{doc.filename} absent de output.csv")
                continue
            if doc.kind == "scan":
                problems.extend(scan_row_problems(doc, row, errors))
                continue
            markers = page_markers(row.get("texteocr"))
            if markers and not markers_continuous(markers):
                problems.append(f"{doc.filename} : marqueurs non continus {markers}")
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def check_on2(self) -> Dict[str, Any]:
        """Chunking recodé par Albert : ``recode_model`` = ``albert/<id résolu>``, chunks ``recoded``, ledger ``recode``."""
        folder = self.session_dir("a")
        require_complete(self.api.sse("/initial_text_chunking_sse", self.tokens["a"],
                                      {"path": self.sessions["a"], "model": RECODE_MODEL}))
        chunks = load_json_list(folder / "output_chunks.json")
        ledger = read_jsonl(folder / "albert_usage.jsonl")
        models = sorted({str(c.get("recode_model")) for c in chunks})
        problems = recode_status_problems(chunks)
        today = date.today()
        model_issues = []
        for chunk in chunks:
            issue = recode_model_problem(chunk.get("recode_model"), RECODE_MODEL, today)
            if issue:
                model_issues.append(f"{chunk.get('id')} : {issue}")
        problems.extend(model_issues[:10])
        ledger_issue = recode_ledger_problem(ledger)
        if ledger_issue:
            problems.append(ledger_issue)
        detail = {"chunks": len(chunks), "recode_status": recode_status_counts(chunks), "recode_model": models,
                  "ledger_recodage": ledger_info(ledger, (RECODE_ROLE,))}
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def check_on3(self) -> Dict[str, Any]:
        """Embeddings bge-m3 : ``embedding_dim=1024`` partout, aucun vecteur nul."""
        folder = self.session_dir("a")
        require_complete(self.api.sse("/dense_embedding_generation_sse", self.tokens["a"],
                                      {"path": self.sessions["a"], "embedding_provider": "albert"}))
        chunks = load_json_list(folder / "output_chunks_with_embeddings.json")
        detail = {
            "chunks": len(chunks),
            "embedding_dim": sorted({str(c.get("embedding_dim")) for c in chunks}),
            "embedding_model": sorted({str(c.get("embedding_model")) for c in chunks}),
            "embedding_provider": sorted({str(c.get("embedding_provider")) for c in chunks}),
        }
        problems = embedding_problems(chunks, EMBED_DIM)
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def _upload_form(self) -> Dict[str, str]:
        """Formulaire ``/upload_db`` de la cible Albert (création autorisée, acquittement donné)."""
        return {
            "path": self.sessions["a"],
            "db_choice": "albert",
            "albert_collection_name": self.collection_name,
            "albert_create_collection": "true",
            "albert_gdpr_ack": "true",
        }

    def _post_upload(self) -> Dict[str, Any]:
        """``POST /upload_db`` vers Albert ; renvoie le corps JSON d'une réponse 200 en succès.

        Raises:
            CheckFailed: statut autre que 200 ou ``status`` autre que ``success``.
        """
        self.session_dir("a")
        resp = self.api.request("POST", "/upload_db", self.tokens["a"], data=self._upload_form(),
                                timeout=UPLOAD_TIMEOUT_S)
        body = _json_body(resp)
        if resp.status_code != 200 or body.get("status") != "success":
            detail = {k: body.get(k) for k in ("error", "details", "credential_required") if body.get(k)}
            raise CheckFailed(f"HTTP {resp.status_code}, statut {body.get('status')!r}", detail or
                              {"corps": _excerpt(resp)})
        return body

    def _audit_upload_count(self) -> int:
        """Entrées d'audit ``ALBERT_COLLECTION_UPLOAD`` de A pour la collection du run."""
        from app.models.audit import AuditLog

        db = self._session_factory()
        try:
            rows = db.query(AuditLog).filter(AuditLog.action == UPLOAD_AUDIT_ACTION,
                                             AuditLog.user_id == self.user_ids["a"]).all()
            return sum(1 for row in rows
                       if isinstance(row.details, dict) and row.details.get("collection_name") == self.collection_name)
        finally:
            db.close()

    def check_on4(self) -> Dict[str, Any]:
        """Envoi vers une collection créée : ``Inserted`` > 0, manifeste copié, entrée d'audit."""
        body = self._post_upload()
        inserted = body.get("inserted_count")
        copies = manifest_copies(self._manifest_dir(), self.sessions["a"], self.manifest_before)
        audits = self._audit_upload_count()
        detail = {"inserted_count": inserted, "existing_count": body.get("existing_count"),
                  "manifestes_copies": [p.name for p in copies], "entrees_audit": audits}
        problems = []
        if not isinstance(inserted, int) or inserted <= 0:
            problems.append(f"Inserted: {inserted!r} (attendu > 0)")
        else:
            self.first_inserted = inserted
        if not copies:
            problems.append("aucune copie de manifeste sous data/albert_manifests/")
        if audits < 1:
            problems.append("aucune entrée d'audit ALBERT_COLLECTION_UPLOAD")
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def check_on5(self) -> Dict[str, Any]:
        """Relance de l'envoi : ``Inserted: 0`` et ``Skipped (existing): N`` (N du premier envoi)."""
        if not self.first_inserted:
            raise CheckFailed("prérequis : premier envoi sans chunk inséré")
        body = self._post_upload()
        detail = {"inserted_count": body.get("inserted_count"), "existing_count": body.get("existing_count"),
                  "skipped_count": body.get("skipped_count"), "attendu": self.first_inserted}
        if body.get("inserted_count") != 0 or body.get("existing_count") != self.first_inserted:
            raise CheckFailed(f"Inserted: {body.get('inserted_count')!r}, Skipped (existing): "
                              f"{body.get('existing_count')!r} (attendu 0 et {self.first_inserted})", detail)
        return detail

    def check_on6(self) -> Dict[str, Any]:
        """Notes courtes gpt-oss : note générée (pas le gabarit) ; ledger exigé, évalué par ``notes_ledger_verdict``.

        Le journal ``albert_usage.jsonl`` de la session est relu par
        ``wait_for_notes_ledger`` (écrit par la route dans son ``finally``).
        """
        folder = self.session_dir("a")
        terminal = require_complete(self.api.sse(
            "/generate_zotero_notes_sse", self.tokens["a"],
            {"session": self.sessions["a"], "note_mode": "short", "model": NOTES_MODEL}))
        summary = terminal.get("summary") if isinstance(terminal.get("summary"), dict) else {}
        notes = load_json_list(folder / "generated_notes.json")
        abstracts = {doc.key: doc.abstract for doc in self.corpus.docs}
        ledger_path = folder / "albert_usage.jsonl"
        ledger = wait_for_notes_ledger(ledger_path)
        ledger_issue, ledger_detail = notes_ledger_verdict(ledger)
        detail = {"resume": summary, "notes": len(notes), "journal_present": ledger_path.is_file(),
                  "enregistrements_ledger": len(ledger), "ledger_notes": ledger_detail}
        problems = []
        if summary.get("errors") not in (0, None) or summary.get("created") != len(self.corpus.docs):
            problems.append(f"résumé de la route {summary} (attendu {len(self.corpus.docs)} note(s), 0 erreur)")
        if len(notes) != len(self.corpus.docs):
            problems.append(f"{len(notes)} note(s) dans generated_notes.json pour {len(self.corpus.docs)} document(s)")
        for note in notes:
            issue = note_problem(note.get("summary"), abstracts.get(str(note.get("item_key")), ""))
            if issue:
                problems.append(f"{note.get('item_key')} : {issue}")
        if ledger_issue:
            problems.append(ledger_issue)
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def _find_collection(self, token: str) -> Optional[int]:
        """Id de la collection du run dans ``GET /api/albert/collections`` (toutes les pages).

        Raises:
            CheckFailed: réponse autre que 200.
        """
        page, pages = 1, 1
        while page <= pages:
            resp = self.api.request("GET", f"/api/albert/collections?page={page}&per_page=100", token)
            body = _json_body(resp)
            if resp.status_code != 200:
                raise CheckFailed(f"GET /api/albert/collections : HTTP {resp.status_code}",
                                  {"corps": _excerpt(resp)})
            for collection in body.get("collections") or []:
                if isinstance(collection, dict) and collection.get("name") == self.collection_name:
                    return int(collection["id"])
            pages = int(body.get("pages") or 1)
            page += 1
        return None

    def check_on7(self) -> Dict[str, Any]:
        """Collection listée, supprimée avec ``confirm=true``, puis absente."""
        cid = self._find_collection(self.tokens["a"])
        if cid is None:
            raise CheckFailed(f"collection {self.collection_name} absente de la liste")
        resp = self.api.request("DELETE", f"/api/albert/collections/{cid}?confirm=true", self.tokens["a"])
        body = _json_body(resp)
        if resp.status_code != 200 or body.get("success") is not True:
            raise CheckFailed(f"DELETE : HTTP {resp.status_code}", {"corps": _excerpt(resp)})
        for attempt in range(DELETE_CONFIRM_ATTEMPTS):
            if self._find_collection(self.tokens["a"]) is None:
                return {"collection_id": cid, "supprimee": True}
            if attempt + 1 < DELETE_CONFIRM_ATTEMPTS:
                time.sleep(DELETE_CONFIRM_WAIT_S)
        raise CheckFailed("collection toujours listée après la suppression", {"collection_id": cid})

    # ------------------------------------------------------- contrôles isolation
    def check_iso1(self) -> Dict[str, Any]:
        """B sans clé : chunking ``albert/…`` refusé en 403 (``credential_required=albert_api_key``)."""
        folder = self.session_dir("b")
        env_has_key = bool((_dotenv_values(self.env_file).get(ALBERT_ENV_KEY) or "").strip())
        resp = self.api.request("POST", "/initial_text_chunking", self.tokens["b"],
                                data={"path": self.sessions["b"], "model": RECODE_MODEL})
        body = _json_body(resp)
        detail = {"statut": resp.status_code, "credential_required": body.get("credential_required"),
                  "cle_dans_env_admin": env_has_key,
                  "chunks_produits": (folder / "output_chunks.json").exists()}
        problems = []
        if not env_has_key:
            problems.append("prérequis : le .env admin ne contient pas la clé Albert")
        if resp.status_code != 403 or body.get("credential_required") != ALBERT_CREDENTIAL_KEY:
            problems.append(f"HTTP {resp.status_code}, credential_required={body.get('credential_required')!r} "
                            f"(attendu 403 et {ALBERT_CREDENTIAL_KEY})")
        if detail["chunks_produits"]:
            problems.append("output_chunks.json produit pour B")
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def check_iso2(self) -> Dict[str, Any]:
        """Identifiants : A masqué sans la clé brute ; B sans valeur Albert."""
        resp_a = self.api.request("GET", "/users/me/credentials", self.tokens["a"])
        resp_b = self.api.request("GET", "/users/me/credentials", self.tokens["b"])
        entry_a = _json_body(resp_a).get(ALBERT_CREDENTIAL_KEY)
        entry_b = _json_body(resp_b).get(ALBERT_CREDENTIAL_KEY)
        leak_a = self.key in (resp_a.text or "")
        leak_b = self.key in (resp_b.text or "")
        masked_a = entry_a.get("masked") if isinstance(entry_a, dict) else None
        detail = {"statut_a": resp_a.status_code, "statut_b": resp_b.status_code,
                  "a_has_value": entry_a.get("has_value") if isinstance(entry_a, dict) else None,
                  "a_masque_non_vide": bool(masked_a), "cle_brute_chez_a": leak_a, "cle_brute_chez_b": leak_b,
                  "b_champ_albert": None if entry_b is None else {
                      "has_value": entry_b.get("has_value") if isinstance(entry_b, dict) else None,
                      "masque_non_vide": bool(entry_b.get("masked")) if isinstance(entry_b, dict) else None}}
        problems = []
        if resp_a.status_code != 200 or resp_b.status_code != 200:
            problems.append(f"HTTP A {resp_a.status_code}, B {resp_b.status_code}")
        if not isinstance(entry_a, dict) or entry_a.get("has_value") is not True or not masked_a:
            problems.append("A : valeur Albert masquée absente")
        elif masked_a == self.key or self.key in str(masked_a):
            problems.append("A : valeur Albert non masquée")
        if leak_a or leak_b:
            problems.append("clé brute présente dans une réponse")
        if entry_b is not None and (not isinstance(entry_b, dict) or entry_b.get("has_value") or entry_b.get("masked")):
            problems.append("B : une valeur Albert est présente")
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    # ------------------------------------------------------------- contrôles OFF
    def check_off1(self) -> Dict[str, Any]:
        """Aucune occurrence d'« albert » dans le HTML de l'index et du profil (A et admin)."""
        pid = self.project_ids.get("a")
        urls = ["/", "/pipeline", "/profile"]
        if pid:
            urls.append(f"/project/{pid}")
            if self.sessions.get("a"):
                urls.append(f"/pipeline?project={pid}&session={quote(self.sessions['a'], safe='')}")
        pages: Dict[str, Any] = {}
        problems = []
        for persona in ("a", "admin"):
            for url in urls:
                resp = self.api.request("GET", url, self.tokens[persona])
                found = "albert" in (resp.text or "").lower()
                pages[f"{persona}:{url}"] = {"statut": resp.status_code, "albert": found}
                if resp.status_code != 200:
                    problems.append(f"{persona}:{url} HTTP {resp.status_code}")
                if found:
                    problems.append(f"{persona}:{url} contient « albert »")
        if problems:
            raise CheckFailed(" ; ".join(problems), {"pages": pages})
        return {"pages": pages}

    def check_off2(self) -> Dict[str, Any]:
        """``/api/albert/status`` en 404 et chunking ``albert/…`` en 400 (Albert désactivé)."""
        folder = self.session_dir("a")
        write_minimal_output_csv(folder / "output.csv")
        status = self.api.request("GET", "/api/albert/status", self.tokens["a"])
        chunk = self.api.request("POST", "/initial_text_chunking", self.tokens["a"],
                                 data={"path": self.sessions["a"], "model": RECODE_MODEL})
        error = str(_json_body(chunk).get("error") or "")
        detail = {"status_statut": status.status_code, "chunking_statut": chunk.status_code,
                  "chunking_erreur": error[:300]}
        problems = []
        if status.status_code != 404:
            problems.append(f"/api/albert/status : HTTP {status.status_code} (attendu 404)")
        if chunk.status_code != 400 or "albert" not in error.lower():
            problems.append(f"chunking albert/… : HTTP {chunk.status_code} (attendu 400 citant Albert)")
        if problems:
            raise CheckFailed(" ; ".join(problems), detail)
        return detail

    def check_off3(self) -> Dict[str, Any]:
        """``GET /get_credentials`` (admin) sans clé ``ALBERT_API_KEY`` (aucune valeur conservée)."""
        resp = self.api.request("GET", "/get_credentials", self.tokens["admin"])
        body = _json_body(resp)
        detail = {"statut": resp.status_code, "cles": len(body),
                  "albert_present": ALBERT_ENV_KEY in body, "cle_brute": self.key in (resp.text or "")}
        if resp.status_code != 200 or not body or detail["albert_present"] or detail["cle_brute"]:
            raise CheckFailed(f"HTTP {resp.status_code}, {ALBERT_ENV_KEY} présent : {detail['albert_present']}",
                              detail)
        return detail

    # --------------------------------------------------------------- nettoyage
    def cleanup(self) -> None:
        """Nettoie tout ce que le run a créé ; une étape en échec n'empêche jamais les suivantes.

        Ordre : arrêt des serveurs (avant toute suppression de collection, pour
        qu'aucun traitement ne puisse en recréer une), fermeture du client HTTP,
        collection du run, recherche de la clé, ``uploads/``, copies de
        manifeste, dossier temporaire. Une étape qui lève devient l'anomalie
        « <étape> en échec (<type>) », qui met le run en ÉCHEC.
        """
        for server in list(self.servers):
            self._cleanup_step(f"arrêt du serveur {server.label}", server.stop)
        self._cleanup_step("fermeture du client HTTP", self._close_api)
        for label, step in (("nettoyage des collections", self._cleanup_collections),
                            ("recherche de la clé", self._scan_for_key),
                            ("nettoyage de uploads/", self._cleanup_uploads),
                            ("nettoyage des copies de manifeste", self._cleanup_manifests),
                            ("suppression de la base temporaire", self._cleanup_tmp)):
            self._cleanup_step(label, step)

    def _cleanup_step(self, label: str, step: Callable[[], Any]) -> bool:
        """Exécute une étape de nettoyage ; une exception devient une anomalie consignée (message masqué).

        Args:
            label: nom de l'étape, en français.
            step: fonction sans argument.

        Returns:
            Vrai si l'étape s'est terminée sans exception.
        """
        try:
            step()
            return True
        except Exception as exc:
            self.extras.append(f"{label} en échec ({type(exc).__name__})")
            errors = self.cleanup_report.setdefault("erreurs", [])
            errors.append(redact(f"{label} : {type(exc).__name__} : {exc}", [self.key])[:600])
            return False

    def _cleanup_collections(self) -> None:
        """Supprime la collection du run (par nom exact) puis vérifie qu'il n'en reste aucune."""
        client = None
        try:
            client = self.albert_client()
            deleted = delete_collections_named(client, self.collection_name)
            remaining = count_collections_named(client, self.collection_name)
            self.cleanup_report["collections"] = {"supprimees": deleted, "restantes": remaining}
            if remaining:
                self.extras.append(f"collections restantes : {remaining}")
        except Exception as exc:
            self.cleanup_report["collections"] = {"erreur": redact(f"{type(exc).__name__} : {exc}", [self.key])}
            self.extras.append("nettoyage des collections en échec")
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _scan_for_key(self) -> None:
        """Cherche la clé dans les journaux du serveur, la suite des journaux ``logs/`` et les sessions."""
        entries: List[Tuple[Any, int]] = []
        for server in self.servers:
            entries.append((server.log_path, 0))
        if LOG_DIR.is_dir():
            for path in LOG_DIR.glob("*.log"):
                entries.append((path, self.log_offsets.get(str(path), 0)))
        for entry in upload_entries_for(UPLOAD_DIR, self.zip_stem):
            if entry.is_dir():
                entries.extend((p, 0) for p in entry.rglob("*") if p.is_file())
        folder = self._manifest_dir()
        if folder is not None and self.sessions.get("a"):
            entries.extend((p, 0) for p in manifest_copies(folder, self.sessions["a"], self.manifest_before))
        hits = files_containing(entries, self.key)
        names = sorted({os.path.relpath(h, REPO) if str(h).startswith(str(REPO)) else Path(h).name for h in hits})
        self.cleanup_report["fuite_de_cle"] = names
        if names:
            self.extras.append(f"clé trouvée dans {len(names)} fichier(s)")

    def _cleanup_uploads(self) -> None:
        """Supprime les dossiers de session et les archives du run sous ``uploads/``."""
        try:
            removed = remove_upload_entries(UPLOAD_DIR, upload_entries_for(UPLOAD_DIR, self.zip_stem))
            left = upload_entries_for(UPLOAD_DIR, self.zip_stem)
            self.cleanup_report["uploads"] = {"supprimes": removed, "restants": [p.name for p in left]}
            if left:
                self.extras.append(f"entrées restantes sous uploads/ : {len(left)}")
        except Exception as exc:
            self.cleanup_report["uploads"] = {"erreur": f"{type(exc).__name__} : {exc}"}
            self.extras.append("nettoyage de uploads/ en échec")

    def _cleanup_manifests(self) -> None:
        """Supprime les copies de manifeste du run (et les dossiers créés par le run, une fois vides)."""
        folder = self._manifest_dir()
        if folder is None:
            return
        try:
            removed = []
            for session in {s for s in self.sessions.values() if s}:
                for path in manifest_copies(folder, session, self.manifest_before):
                    path.unlink()
                    removed.append(path.name)
            if folder.is_dir() and not self.manifest_dir_existed and not any(folder.iterdir()):
                folder.rmdir()
            if MANIFEST_ROOT.is_dir() and not self.manifest_root_existed and not any(MANIFEST_ROOT.iterdir()):
                MANIFEST_ROOT.rmdir()
            self.cleanup_report["manifestes"] = removed
        except Exception as exc:
            self.cleanup_report["manifestes"] = {"erreur": f"{type(exc).__name__} : {exc}"}
            self.extras.append("nettoyage des copies de manifeste en échec")

    def _cleanup_tmp(self) -> None:
        """Ferme la base temporaire et supprime le dossier temporaire (base, corpus, journaux du serveur)."""
        try:
            if self._engine is not None:
                self._engine.dispose()
            if self.tmpdir is not None and self.tmpdir.exists():
                shutil.rmtree(self.tmpdir)
            self.cleanup_report["base_temporaire"] = "supprimée"
        except Exception as exc:
            self.cleanup_report["base_temporaire"] = {"erreur": f"{type(exc).__name__} : {exc}"}
            self.extras.append("suppression de la base temporaire en échec")

    # ------------------------------------------------------------------ résumé
    def counts(self) -> Dict[str, int]:
        """Contrôles réussis par groupe (un contrôle non exécuté compte comme échoué)."""
        return {group: sum(1 for cid, _route in checks if self.results[group].get(cid, {}).get("ok"))
                for group, checks in CHECK_GROUPS}

    def finish(self) -> int:
        """Écrit le résumé JSON, affiche la dernière ligne et renvoie le code de sortie."""
        counts = self.counts()
        extras = list(self.extras)
        if self.setup_errors:
            extras.insert(0, f"préparation : {len(self.setup_errors)} erreur(s)")
        line = summary_line(counts["on"], counts["off"], counts["isolation"], extras)
        checks = {}
        for group, entries in CHECK_GROUPS:
            checks[group] = [self.results[group].get(cid) or
                             {"id": cid, "route": route, "ok": False, "message": "non exécuté", "detail": {}}
                             for cid, route in entries]
        summary = {
            "run": self.stamp,
            "debut": self.started.isoformat(),
            "fin": datetime.now(timezone.utc).isoformat(),
            "corpus": [{"fichier": d.filename, "type": d.kind, "pages": d.pages}
                       for d in (self.corpus.docs if self.corpus else [])],
            "collection": self.collection_name,
            "modeles": {"recodage": RECODE_MODEL, "notes": NOTES_MODEL},
            "ports": self.ports,
            "controles": checks,
            "compteurs": {group: f"{counts[group]}/{len(entries)}" for group, entries in CHECK_GROUPS},
            "preparation": self.setup_errors,
            "nettoyage": self.cleanup_report,
            "anomalies": extras,
            "resultat": line,
        }
        path = SUMMARY_DIR / f"summary-{self.stamp}.json"
        try:
            SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(redact(summary, [self.key]), ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")
            self.say(f"Résumé JSON : {os.path.relpath(path, REPO)}")
        except (OSError, TypeError, ValueError) as exc:
            self.say(f"Résumé JSON non écrit : {type(exc).__name__}")
        self.say(line)
        return 0 if line.startswith("E2E ALBERT OK") else 1


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def _corpus_size(value: str) -> int:
    """Taille de corpus (entier, 2 au moins)."""
    try:
        size = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"entier attendu : {value!r}")
    if size < MIN_CORPUS_SIZE:
        raise argparse.ArgumentTypeError(f"{MIN_CORPUS_SIZE} documents au moins (texte et numérisé)")
    return size


def _print_line(message: str) -> None:
    """Affiche une ligne sans tampon (progression visible pendant les étapes longues)."""
    print(message, flush=True)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Options de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="albert_e2e_smoke.py",
        description="Test de bout en bout souverain d'Albert par les vraies routes (exige ALBERT_LIVE=1 dans le shell).",
    )
    parser.add_argument("--corpus-size", type=_corpus_size, default=DEFAULT_CORPUS_SIZE,
                        help="nombre de PDF : N-1 texte et 1 numérisé (défaut 3)")
    parser.add_argument("--cleanup", action="store_true",
                        help=(f"supprimer aussi au démarrage les collections {COLLECTION_PREFIX}AAAAMMJJ-HHMMSS "
                              "restantes d'un run précédent de ce script (jamais celles de la suite live)"))
    return parser.parse_args(argv)


def _raise_interrupt(signum: int, _frame: Any) -> None:
    """Gestionnaire de SIGTERM et SIGHUP : lève ``KeyboardInterrupt`` (le nettoyage du ``finally`` s'exécute)."""
    raise KeyboardInterrupt(f"signal {signum}")


def set_signal_handlers(signals: Iterable[int], handler: Any) -> Dict[int, Any]:
    """Installe ``handler`` pour chaque signal et renvoie les gestionnaires précédents.

    Un signal qui ne peut pas être détourné (fil autre que le fil principal,
    signal non pris en charge) est laissé tel quel et absent du résultat.

    Args:
        signals: numéros de signaux.
        handler: gestionnaire, ``signal.SIG_IGN`` ou ``signal.SIG_DFL``.

    Returns:
        ``{signal: gestionnaire précédent}`` pour ``restore_signal_handlers``.
    """
    previous: Dict[int, Any] = {}
    for sig in signals:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError, RuntimeError):
            continue
    return previous


def restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    """Réinstalle les gestionnaires renvoyés par ``set_signal_handlers`` (``SIG_DFL`` s'il n'y en avait pas)."""
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler if handler is not None else signal.SIG_DFL)
        except (ValueError, OSError, RuntimeError, TypeError):
            continue


def main(argv: Optional[Sequence[str]] = None, *, environ: Optional[Mapping[str, str]] = None,
         env_file: Any = None) -> int:
    """Point d'entrée : gardes (live, clé, python3 du PATH), run, nettoyage, résumé.

    Pendant le run, SIGTERM et SIGHUP lèvent ``KeyboardInterrupt`` comme un
    Ctrl-C : les serveurs (lancés dans leur propre session) ne survivent pas à
    un arrêt du script. Pendant le nettoyage et l'écriture du résumé, SIGINT,
    SIGTERM et SIGHUP sont ignorés ; les gestionnaires d'origine sont
    réinstallés avant le retour.

    Args:
        argv: arguments (défaut : ``sys.argv[1:]``).
        environ: environnement du shell (défaut : ``os.environ``).
        env_file: fichier ``.env`` (défaut : celui du dépôt).

    Returns:
        0 (OK), 1 (échec) ou 2 (refus).
    """
    args = parse_args(argv)
    environ = os.environ if environ is None else environ
    env_file = ENV_FILE if env_file is None else Path(env_file)
    refusal = live_gate_refusal(environ, env_file)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    key, key_refusal = load_api_key(env_file)
    if key_refusal or not key:
        print(key_refusal or f"Refus : {ALBERT_ENV_KEY} absente.", file=sys.stderr)
        return 2
    server_path = build_server_path(environ.get("PATH", ""), VENV_BIN)
    problem = venv_python_problem(server_path, VENV_BIN)
    if problem:
        print(f"Refus : {problem} Les routes lancent « python3 » : créer l'environnement virtuel "
              f"du dépôt (.venv) avant le test.", file=sys.stderr)
        return 2
    run = SmokeRun(args, key, server_path, env_file=env_file, log=_print_line)
    code = 1
    interrupt_handlers = set_signal_handlers(INTERRUPT_SIGNALS, _raise_interrupt)
    try:
        try:
            run.execute()
        except KeyboardInterrupt:
            run.setup_errors.append("run interrompu (Ctrl-C ou signal d'arrêt)")
            run.say("Run interrompu : nettoyage.")
        except Exception as exc:
            run.setup_error("déroulé", exc)
        finally:
            shielded = set_signal_handlers(SHIELDED_SIGNALS, signal.SIG_IGN)
            try:
                run.cleanup()
                code = run.finish()
            finally:
                restore_signal_handlers(shielded)
    finally:
        restore_signal_handlers(interrupt_handlers)
    return code


if __name__ == "__main__":
    sys.exit(main())
