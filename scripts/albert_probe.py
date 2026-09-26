#!/usr/bin/env python3
"""Sondes live de l'API Albert (DINUM) pour le lot 0 du sprint Albert.

Ce script autonome (httpx, bibliothèque standard, python-dotenv, PyMuPDF)
exécute les sondes P1 à P21 décrites dans la spécification du sprint, écrit
des fixtures assainies, un fichier ``decisions.json`` (D1 à D22) et un rapport
Markdown en français.

Garde-fous :

* refus (code 2) si ``ALBERT_LIVE`` ne vaut pas ``1`` dans le shell, ou si
  ``ALBERT_LIVE`` figure dans le ``.env`` du répertoire courant ;
* la clé ``ALBERT_API_KEY`` est lue par ``dotenv_values('.env')`` (repli sur
  l'environnement du processus), n'est jamais affichée ni écrite ; tout ce qui
  est écrit (bruts, fixtures, décisions, rapport) passe par
  :func:`sanitize_payload` ;
* une seule collection privée ``ragpy-probe-<ts>`` est créée pour P14 à P19,
  et elle est supprimée dans un ``finally`` (avec l'éventuel doublon), suivi
  d'un balayage par nom exact qui rattrape une création dont la réponse s'est
  perdue ; aucune collection publique n'est jamais créée ;
* P13 est passive (aucune rafale pour provoquer un 429) ;
* une erreur de compte (401, 403 « account has expired », 400 budget, 429
  avec Retry-After > 120 s = quota journalier) arrête immédiatement le run,
  après le nettoyage des collections ;
* après le run, les fixtures ``P<n>_*.json`` d'un run précédent sont retirées
  pour chaque sonde exécutée (``decisions.json`` et ``golden_off/`` intacts).

Usage (une seule ligne) ::

    ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --all --report .claude/tasks/albert_probe_report.md --fixtures-dir tests/fixtures/albert
    ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --only P2,P5
    ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --cleanup --dry-run
    ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --all --diff-against tests/fixtures/albert

Sorties :

* ``<out>/<ts>/raw/NNNN_<sonde>_<slug>.json`` : requêtes et réponses brutes
  (clé, e-mails et identité masqués) ;
* ``<fixtures-dir>/P<n>_<slug>.json`` : fixtures
  ``{request: {method, path, body_summary}, status, headers, body, synthetic}``
  (en-têtes limités à content-type, retry-after et x-ratelimit-*) ; les
  fixtures ``*_synthetic.json`` (``"synthetic": true``) décrivent un succès
  documenté quand la sonde n'en a obtenu aucun ;
* ``<fixtures-dir>/decisions.json`` : D1 à D22, chacune
  ``{value, default, source, consumer}`` ;
* le rapport Markdown (``--report``, sinon ``<out>/<ts>/report.md``).

La dernière ligne d'un run est exactement
``SONDES: <n> exécutées, <k> erreur de script`` (k compte les exceptions Python
du script, pas les réponses HTTP 4xx attendues). ``--cleanup`` se termine par
``ragpy-probe restantes: <N>``.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import email.utils
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
ENV_FILE = ".env"
PROBE_PREFIX = "ragpy-probe-"
DEFAULT_OUT = "data/albert_probe"
USER_AGENT = "ragpy-albert-probe/1.0"

MASK = "<masqué>"
KEY_MASK = "<clé masquée>"
EMAIL_MASK = "<e-mail masqué>"

TIMEOUTS = {
    "chat": 120.0,
    "notes": 300.0,
    "ocr_page": 120.0,
    "ocr_doc": 300.0,
    "embed": 60.0,
    "collections": 120.0,
    "default": 60.0,
}

RECODE_MODEL = "ministral-3-8b-instruct-2512"
NOTES_MODEL = "gpt-oss-120b"
OCR_CHAT_MODEL = "lightonocr-2-1b"
OCR_CHAT_ALIAS = "openweight-ocr"
OCR_DOC_MODEL = "mistral-ocr-2512"
EMBED_MODEL = "bge-m3"
EMBED_ALIAS = "openweight-embeddings"

EXPECTED_MODELS = {
    NOTES_MODEL: "text-generation",
    RECODE_MODEL: "image-text-to-text",
    OCR_CHAT_MODEL: "image-text-to-text",
    EMBED_MODEL: "text-embeddings-inference",
}
EXPECTED_ALIASES = ("openweight-large", "openweight-small", "openweight-ocr", "openweight-embeddings")

# Quotas de production documentés (repli quand /v1/me ne permet pas le croisement).
DOCUMENTED_RPM = {NOTES_MODEL: 50, RECODE_MODEL: 100, EMBED_MODEL: 2000}
# Indice informatif (jamais utilisé pour une décision) : modèles dont le RPM de production documenté vaut n.
DOCUMENTED_RPM_CANDIDATES = {
    50: (NOTES_MODEL, "deepseek-v4-flash"),
    100: (RECODE_MODEL, "mistral-small-3-2-24b-instruct-2506", "qwen3-coder-30b-a3b-instruct", "whisper-large-v3"),
    2000: (EMBED_MODEL, "bge-reranker-v2-m3"),
}

PROBE_ORDER = ["P%d" % i for i in range(1, 22)]
PROBE_TITLES = {
    "P1": "compte et quotas (/v1/me)",
    "P2": "catalogue (/v1/models)",
    "P3": "accès /v1/ocr et pages à partir de 0",
    "P4": "taille et latence de /v1/ocr",
    "P5": "LightOnOCR par le chat",
    "P6": "forme des erreurs",
    "P7": "présence de usage.cost et impacts",
    "P8": "max_tokens ou max_completion_tokens",
    "P9": "raisonnement de gpt-oss",
    "P10": "sorties JSON",
    "P11": "déterminisme de seed",
    "P12": "embeddings : lot, dimension, norme",
    "P13": "en-têtes de limitation (passif)",
    "P14": "cycle de vie d'une collection",
    "P15": "document sans fichier (multipart ou urlencoded)",
    "P16": "limites d'envoi de chunks et métadonnées",
    "P17": "filtres et paramètres de recherche",
    "P18": "latence d'indexation",
    "P19": "quota d'embeddings consommé par les envois",
    "P20": "points de santé",
    "P21": "débit de référence",
}

# Assainissement
IDENTITY_FIELDS = frozenset({"id", "email", "name", "organization_id", "user_id"})
ALWAYS_MASKED_FIELDS = frozenset({
    "email", "user_id", "organization_id", "owner", "owner_id", "password",
    "current_password", "api_key", "apikey", "access_token", "refresh_token",
})
SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"})
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
BEARER_RE = re.compile(r"(?i)\b(bearer)(\s+)([A-Za-z0-9._~+/=\-]{12,})")
MIN_SECRET_LEN = 8
MIN_IDENTITY_TOKEN_LEN = 3
# Morceaux d'identité récoltés (jetons du nom, partie locale de l'e-mail) : longueur minimale et
# mots génériques jamais masqués (vocabulaire d'API ou de JSON, noms d'hôtes).
MIN_NAME_TOKEN_LEN = 4
IDENTITY_STOPLIST = frozenset({
    "data", "user", "model", "models", "name", "list", "object", "albert", "admin", "test", "owner", "api",
    "chat", "key", "role", "text", "etalab", "gouv", "ragpy", "probe", "json", "http", "https", "health",
    "search", "collection", "document", "none", "null", "true", "false", "service",
})
# Clé d'API : caractères ASCII imprimables sans espace, 8 au moins.
API_KEY_RE = re.compile(r"[!-~]{8,}")

# Retry-After au-delà de ce seuil : quota journalier (RPD/TPD) épuisé, aucun nouvel essai.
RETRY_AFTER_MAX_S = 120.0

# Fixtures
FIXTURE_HEADER_NAMES = ("content-type", "retry-after")
FIXTURE_FLOAT_LIST_MAX = 16
FIXTURE_FLOAT_KEEP = 8
FIXTURE_LIST_MAX = 64
FIXTURE_STR_MAX = 6000
SUMMARY_STR_MAX = 500
SUMMARY_LIST_MAX = 8

P4_SIZES_MB = (5, 15, 25, 40)
P4_MAX_TOTAL_S = 900.0
POLL_MAX_S = 60.0

RECODE_SYSTEM = "Assistant spécialisé en recodage de textes académiques."
RECODE_TEMPLATE = "Instructions : {instructions}\n\nTexte à recoder :\n{chunk}\n\nTexte recodé :"
RECODE_INSTRUCTIONS = (
    "ce chunk est issu d'un ocr brut qui laisse beaucoup de blocs de texte inutiles comme des titres de pages, "
    "des numeros, etc. Nettoie ce chunk pour en faire un texte propre qui commence par une phrase complète et se "
    "termine par un point. Supprime le bruit d'OCR et les imperfections en conservant le sens original. Ne echange "
    "ni ajoute aucun mot du texte d'origine. C'est une correction et un nettoyage de texte (suppression des erreurs) "
    "pas une réécriture"
)

FRENCH_PARAGRAPHS = (
    "La sociologie des sciences s'intéresse aux conditions sociales de production des savoirs. Elle observe "
    "les laboratoires, les controverses et les instruments comme autant de lieux où se fabriquent les faits "
    "scientifiques, et elle refuse de séparer a priori le contenu des connaissances de leur contexte.",
    "Dans cette perspective, l'enquête ethnographique suit les chercheurs au travail : elle décrit la manière "
    "dont les inscriptions circulent, dont les résultats sont stabilisés par des publications et dont les "
    "désaccords se referment, parfois sans que la preuve soit décisive.",
    "Les archives numériques transforment à leur tour la pratique des sciences humaines. La numérisation des "
    "fonds, l'océrisation des imprimés anciens et l'indexation sémantique ouvrent des corpus inédits, mais "
    "elles introduisent aussi des biais de sélection et des erreurs de transcription.",
    "Une lecture attentive des sources impose donc une critique documentaire renouvelée : identifier la "
    "provenance des textes, mesurer la qualité de la reconnaissance optique, et conserver la trace des "
    "transformations successives que subit le document entre le papier et la base de données.",
)
OCR_NOISE = ("- 12 -", "CHAPITRE III", "LA FABRIQUE DES SAVOIRS", "socio-\nlogie", "13", "Introduction générale")

CITATIONS_P10 = (
    "1. Bourdieu analyse le champ scientifique comme un espace de concurrence pour le capital symbolique.",
    "2. La recette de la tarte aux pommes demande une pâte brisée et des pommes acidulées.",
    "3. Latour et Woolgar décrivent la construction sociale des faits dans un laboratoire.",
    "4. Le moteur diesel utilise l'auto-inflammation du carburant comprimé.",
    "5. Merton formule les normes de l'ethos scientifique : universalisme, communalisme, désintéressement.",
)

# ---------------------------------------------------------------------------
# Assainissement (fonction pure)
# ---------------------------------------------------------------------------


def _is_account_object(obj: Dict[str, Any]) -> bool:
    """Indique si un dict ressemble à l'objet de compte renvoyé par /v1/me."""
    keys = set(obj)
    if obj.get("object") in ("userInfo", "user", "me"):
        return True
    if "email" in keys:
        return True
    return {"budget", "limits"} <= keys or {"permissions", "expires"} <= keys


def _is_key_object(obj: Dict[str, Any]) -> bool:
    """Indique si un dict ressemble à un objet clé d'API (``/v1/keys``)."""
    if obj.get("object") == "key":
        return True
    return "value" in obj and ("user_id" in obj or "expires" in obj) and "type" not in obj


def _identity_piece_ok(piece: str, min_len: int = MIN_NAME_TOKEN_LEN) -> bool:
    """Vrai si un morceau d'identité est assez long et n'est pas un mot générique."""
    return len(piece) >= min_len and piece.lower() not in IDENTITY_STOPLIST


def _identity_values(email: Any, name: Any) -> List[str]:
    """Valeurs d'identité à effacer d'un objet de compte (e-mail, partie locale, nom, jetons du nom)."""
    found: List[str] = []
    if isinstance(email, str) and email.strip():
        email = email.strip()
        found.append(email)
        local = email.split("@", 1)[0]
        if _identity_piece_ok(local):
            found.append(local)
    if isinstance(name, str) and name.strip():
        name = name.strip()
        if _identity_piece_ok(name, MIN_IDENTITY_TOKEN_LEN):
            found.append(name)
        found.extend(token for token in re.split(r"[\s,;]+", name) if _identity_piece_ok(token))
    return found


def _identity_patterns(identity: Iterable[str]) -> List["re.Pattern[str]"]:
    """Construit les motifs insensibles à la casse des valeurs d'identité à effacer."""
    patterns = []
    for value in sorted({v for v in identity if isinstance(v, str)}, key=len, reverse=True):
        value = value.strip()
        if len(value) < MIN_IDENTITY_TOKEN_LEN or value in (MASK, KEY_MASK, EMAIL_MASK):
            continue
        patterns.append(re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE))
    return patterns


def _scrub_text(text: str, secrets: Sequence[str], id_patterns: Sequence["re.Pattern[str]"],
                replacements: Sequence[Tuple[str, str]]) -> str:
    """Masque dans une chaîne la clé, les jetons Bearer, les e-mails et l'identité."""
    for secret in secrets:
        if secret and len(secret) >= MIN_SECRET_LEN and secret in text:
            text = text.replace(secret, KEY_MASK)
    text = BEARER_RE.sub(lambda m: m.group(1) + m.group(2) + MASK, text)
    text = EMAIL_RE.sub(EMAIL_MASK, text)
    for pattern in id_patterns:
        text = pattern.sub(MASK, text)
    for old, new in replacements:
        if old and len(old) > 1 and old in text:
            text = text.replace(old, new)
    return text


def _is_empty(value: Any) -> bool:
    """Indique si une valeur est vide (rien à masquer)."""
    return value is None or value == "" or value == [] or value == {}


def sanitize_payload(obj: Any, secrets: Sequence[str] = (), identity: Iterable[str] = (),
                     replacements: Sequence[Tuple[str, str]] = ()) -> Any:
    """Renvoie une copie assainie de ``obj`` (fonction pure et idempotente).

    Sont masqués : les en-têtes sensibles (``Authorization``, cookies…),
    tout jeton ``Bearer …``, toute occurrence exacte des chaînes ``secrets``
    (la clé d'API), les adresses e-mail, les champs d'identité d'un objet de
    compte (``id``, ``email``, ``name``, ``organization_id``, ``user_id``), le
    champ ``value`` (et l'identité) d'un objet clé, les champs ``owner``,
    ``user_id``, ``email``… de tout objet, et les valeurs d'identité fournies
    (``identity``, insensible à la casse, dans les valeurs seulement : les
    clés de dict ne sont jamais réécrites par l'identité, pour ne pas
    corrompre la forme des fixtures). ``replacements`` remplace des
    sous-chaînes (chemins locaux par exemple).
    """
    secrets = [s for s in secrets if isinstance(s, str) and len(s) >= MIN_SECRET_LEN]
    id_patterns = _identity_patterns(identity)
    replacements = [(a, b) for a, b in replacements if a and len(a) > 1]

    def walk(node: Any) -> Any:
        """Parcourt récursivement la structure et masque ce qui doit l'être."""
        if isinstance(node, dict):
            account = _is_account_object(node)
            key_obj = _is_key_object(node)
            out = {}
            for key, value in node.items():
                new_key = _scrub_text(key, secrets, (), replacements) if isinstance(key, str) else key
                low = key.lower() if isinstance(key, str) else ""
                if low in SENSITIVE_HEADERS and not _is_empty(value):
                    out[new_key] = MASK
                elif low in ALWAYS_MASKED_FIELDS and not _is_empty(value):
                    out[new_key] = MASK
                elif (account or key_obj) and low in IDENTITY_FIELDS and not _is_empty(value):
                    out[new_key] = MASK
                elif key_obj and low == "value" and not _is_empty(value):
                    out[new_key] = MASK
                else:
                    out[new_key] = walk(value)
            return out
        if isinstance(node, (list, tuple)):
            return [walk(item) for item in node]
        if isinstance(node, str):
            return _scrub_text(node, secrets, id_patterns, replacements)
        return node

    return walk(obj)


# ---------------------------------------------------------------------------
# Résumés et compaction
# ---------------------------------------------------------------------------


def summarize_request_body(obj: Any) -> Any:
    """Résume un corps de requête pour les fixtures (data URI et listes longues abrégées)."""
    if isinstance(obj, dict):
        return {k: summarize_request_body(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if len(items) > SUMMARY_LIST_MAX:
            head = [summarize_request_body(x) for x in items[:3]]
            return head + ["… (+%d éléments, %d au total)" % (len(items) - 3, len(items))]
        return [summarize_request_body(x) for x in items]
    if isinstance(obj, str):
        if obj.startswith("data:") and ";base64," in obj[:80] and len(obj) > 100:
            prefix = obj.split(",", 1)[0]
            return "%s,<%d caractères base64>" % (prefix, len(obj) - len(prefix) - 1)
        if len(obj) > SUMMARY_STR_MAX:
            return obj[:200] + "… (+%d caractères)" % (len(obj) - 200)
    return obj


def compact_for_fixture(obj: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    """Abrège un corps de réponse (vecteurs, listes, textes longs) et note les coupes."""
    notes: Dict[str, int] = {}

    def note(path: str, length: int) -> None:
        """Enregistre une coupe avec la longueur maximale observée."""
        notes[path] = max(notes.get(path, 0), length)

    def walk(node: Any, path: str) -> Any:
        """Parcourt la structure et applique les coupes."""
        if isinstance(node, list):
            if len(node) > FIXTURE_FLOAT_LIST_MAX and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in node
            ):
                note(path, len(node))
                return node[:FIXTURE_FLOAT_KEEP]
            items = node
            if len(node) > FIXTURE_LIST_MAX:
                note(path, len(node))
                items = node[:FIXTURE_LIST_MAX]
            return [walk(item, path + "[]") for item in items]
        if isinstance(node, dict):
            return {k: walk(v, "%s.%s" % (path, k)) for k, v in node.items()}
        if isinstance(node, str) and len(node) > FIXTURE_STR_MAX:
            note(path, len(node))
            return node[:FIXTURE_STR_MAX] + "…"
        return node

    compacted = walk(obj, "body")
    return compacted, [{"chemin": p, "longueur_originale": n} for p, n in sorted(notes.items())]


def filter_fixture_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Ne garde que content-type, retry-after et les en-têtes x-ratelimit-*."""
    out = {}
    for name, value in headers.items():
        low = name.lower()
        if low in FIXTURE_HEADER_NAMES or low.startswith("x-ratelimit"):
            out[low] = value
    return out


def parse_retry_after(value: Optional[str], now: float) -> Tuple[Optional[float], Optional[str]]:
    """Analyse ``Retry-After`` : renvoie (secondes, format) avec format 'delta' ou 'http-date'."""
    if value is None:
        return None, None
    value = value.strip()
    try:
        return max(0.0, float(value)), "delta"
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None, None
    if parsed is None:
        return None, None
    return max(0.0, parsed.timestamp() - now), "http-date"


def account_error_message(status: Optional[int], text: str) -> Optional[str]:
    """Renvoie le message d'arrêt pour une erreur de niveau compte, sinon None."""
    low = (text or "").lower()
    if status == 401:
        return "clé Albert invalide ou expirée (HTTP 401) : vérifier ALBERT_API_KEY."
    if status == 403 and "account has expired" in low:
        return "compte Albert expiré (HTTP 403) : écrire à albert.api@numerique.gouv.fr pour le renouveler."
    if status == 400 and re.search(r"insufficient\s*budget", low):
        return "budget Albert épuisé (HTTP 400) : aucune nouvelle requête ne sera acceptée."
    return None


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Percentile au rang le plus proche (None si la liste est vide)."""
    data = sorted(values)
    if not data:
        return None
    rank = max(1, math.ceil(pct / 100.0 * len(data)))
    return round(data[rank - 1], 3)


def _norm_text(text: Any) -> str:
    """Normalise un texte OCR pour la recherche de jetons (antislashs et espaces retirés)."""
    if not isinstance(text, str):
        return ""
    return text.replace("\\", "").replace(" ", "")


# ---------------------------------------------------------------------------
# PDF et images (PyMuPDF)
# ---------------------------------------------------------------------------


def make_two_page_pdf() -> bytes:
    """PDF de deux pages portant PAGE_ONE_7Q puis PAGE_TWO_9Z."""
    import fitz

    doc = fitz.open()
    for token in ("PAGE_ONE_7Q", "PAGE_TWO_9Z"):
        page = doc.new_page()
        page.insert_text((72, 144), token, fontsize=32)
    data = doc.tobytes()
    doc.close()
    return data


def render_page_png(pdf_bytes: bytes, index: int, dpi: int = 200, max_side: int = 1540) -> bytes:
    """Rastérise une page en PNG (200 DPI, plus grand côté plafonné à ``max_side``)."""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[index]
        zoom = min(dpi / 72.0, max_side / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


def make_noise_pdf(target_mb: float, seed: int = 7) -> bytes:
    """PDF d'images de bruit (incompressibles) d'environ ``target_mb`` Mo, jeton en page 1."""
    import fitz

    rng = random.Random(seed)
    side = 600
    target = int(target_mb * 1024 * 1024)
    doc = fitz.open()
    total = 0
    while total < target:
        pix = fitz.Pixmap(fitz.csRGB, side, side, rng.randbytes(side * side * 3), 0)
        png = pix.tobytes("png")
        page = doc.new_page()
        page.insert_image(page.rect, stream=png)
        if len(doc) == 1:
            page.insert_text((72, 72), "PAGE_ONE_7Q", fontsize=28)
        total += len(png)
    data = doc.tobytes()
    doc.close()
    return data


def make_text_pdf(n_pages: int) -> bytes:
    """PDF texte de ``n_pages`` pages courtes (mesure de latence de /v1/ocr)."""
    import fitz

    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 100), "Page %d de la sonde de latence." % (i + 1), fontsize=14)
        page.insert_textbox(fitz.Rect(72, 130, 520, 700), FRENCH_PARAGRAPHS[i % len(FRENCH_PARAGRAPHS)], fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_dense_pages_png(n_pages: int = 10) -> List[bytes]:
    """Pages denses de texte français rastérisées (échantillon synthétique de livre)."""
    import fitz

    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        text = "\n\n".join(FRENCH_PARAGRAPHS[(i + k) % len(FRENCH_PARAGRAPHS)] for k in range(6))
        page.insert_text((72, 50), "- %d -" % (i + 1), fontsize=9)
        page.insert_textbox(fitz.Rect(60, 70, 535, 800), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return [render_page_png(data, i) for i in range(n_pages)]


def sample_pages_png(path: str, n_pages: int = 10) -> List[bytes]:
    """Rastérise les ``n_pages`` premières pages d'un PDF fourni (--ocr-sample)."""
    data = Path(path).read_bytes()
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    count = min(n_pages, len(doc))
    doc.close()
    return [render_page_png(data, i) for i in range(count)]


def french_ocr_chunk(target_chars: int = 4300) -> str:
    """Texte français bruité façon OCR d'environ 1 000 tokens (sonde P21)."""
    parts: List[str] = []
    i = 0
    while sum(len(p) for p in parts) < target_chars:
        parts.append(OCR_NOISE[i % len(OCR_NOISE)])
        parts.append(FRENCH_PARAGRAPHS[i % len(FRENCH_PARAGRAPHS)])
        i += 1
    return "\n".join(parts)


def _png_data_uri(png: bytes) -> str:
    """Encode un PNG en data URI."""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------------------
# Échanges et contexte
# ---------------------------------------------------------------------------


class AccountLevelError(RuntimeError):
    """Erreur de niveau compte (401, 403 compte expiré, 400 budget) : arrêt du run."""


class Exchange:
    """Une requête envoyée et la réponse (ou l'erreur réseau) obtenue."""

    def __init__(self, probe: str, slug: str, method: str, path: str, body_summary: Any):
        """Initialise un échange vide pour ``probe``/``slug``."""
        self.probe = probe
        self.slug = slug
        self.method = method
        self.path = path
        self.body_summary = body_summary
        self.status: Optional[int] = None
        self.headers: Dict[str, str] = {}
        self.body: Any = None
        self.text = ""
        self.elapsed = 0.0
        self.error: Optional[str] = None

    def ok(self) -> bool:
        """Vrai pour un statut 2xx."""
        return self.status is not None and 200 <= self.status < 300

    def dig(self, *keys: Any, default: Any = None) -> Any:
        """Accès imbriqué tolérant dans le corps JSON."""
        node = self.body
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif isinstance(node, list) and isinstance(key, int) and -len(node) <= key < len(node):
                node = node[key]
            else:
                return default
        return node


def _files_summary(files: Dict[str, Any]) -> Dict[str, Any]:
    """Résume des champs multipart (valeurs textuelles gardées, fichiers abrégés)."""
    out = {}
    for name, value in files.items():
        if isinstance(value, tuple) and len(value) >= 2 and value[0] is None:
            out[name] = value[1]
        else:
            out[name] = "<fichier>"
    return out


class ProbeContext:
    """État partagé d'un run : client, clé, sorties, résultats, horloges injectables."""

    def __init__(self, client: httpx.Client, api_key: str, run_dir: Path, fixtures_dir: Optional[Path] = None, *,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 wall: Callable[[], float] = time.time, ocr_sample: Optional[str] = None,
                 log: Callable[[str], None] = print):
        """Prépare les répertoires de sortie et les compteurs du run."""
        self.client = client
        self._api_key = api_key
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir else self.run_dir / "fixtures"
        self.clock = clock
        self.sleep = sleep
        self.wall = wall
        self.ocr_sample = ocr_sample
        self._log = log
        self._lock = threading.RLock()
        self._seq = 0
        self.identity: set = set()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.fixtures: Dict[str, Dict[str, Any]] = {}
        self.state: Dict[str, Any] = {}
        self.created_collections: List[int] = []
        self.executed = 0
        self.script_errors = 0
        self.executed_ids: List[str] = []
        self.rate_info: Dict[str, Any] = {
            "entetes_vus": set(), "exemples": [], "retry_after_formats": set(),
            "premier_429": None, "compte_429": 0, "compte_503": 0,
        }
        self._models_cache: Optional[List[Dict[str, Any]]] = None
        self.path_replacements: List[Tuple[str, str]] = []
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        if len(cwd) > 1:
            self.path_replacements.append((cwd, "."))
        if len(home) > 1:
            self.path_replacements.append((home, "~"))

    # -- outils ------------------------------------------------------------

    def log(self, message: str) -> None:
        """Affiche un message assaini."""
        self._log(self.sanitize(message))

    def sanitize(self, obj: Any) -> Any:
        """Assainit ``obj`` avec la clé, l'identité connue et les chemins locaux."""
        with self._lock:
            identity = sorted(self.identity)
        return sanitize_payload(obj, secrets=[self._api_key], identity=identity,
                                replacements=self.path_replacements)

    def _next_seq(self) -> int:
        """Numéro d'ordre des fichiers bruts (sûr entre threads)."""
        with self._lock:
            self._seq += 1
            return self._seq

    def _harvest_identity(self, body: Any) -> None:
        """Mémorise nom et e-mail des objets de compte pour les effacer partout.

        Sont retenus l'e-mail, sa partie locale et le nom complet, puis les
        jetons du nom d'au moins ``MIN_NAME_TOKEN_LEN`` caractères ; un mot
        générique (``IDENTITY_STOPLIST`` : data, user, Albert…) n'est jamais
        retenu, pour ne pas corrompre le vocabulaire des réponses.
        """
        stack = [body]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if _is_account_object(node):
                    found = _identity_values(node.get("email"), node.get("name"))
                    if found:
                        with self._lock:
                            self.identity.update(found)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    def _observe_headers(self, ex: Exchange) -> None:
        """Collecte passivement les en-têtes de limitation (sonde P13)."""
        names = [n for n in ex.headers if n.lower().startswith("x-ratelimit") or n.lower() == "retry-after"]
        with self._lock:
            for name in names:
                self.rate_info["entetes_vus"].add(name.lower())
            if names and len(self.rate_info["exemples"]) < 6:
                self.rate_info["exemples"].append({
                    "chemin": ex.path.split("?", 1)[0], "statut": ex.status,
                    "entetes": {n.lower(): ex.headers[n] for n in names},
                })

    def _write_json(self, path: Path, obj: Any) -> None:
        """Écrit un JSON assaini (UTF-8, indenté)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.sanitize(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_raw(self, ex: Exchange, request: httpx.Request) -> None:
        """Écrit la requête et la réponse brutes (assainies) sous raw/."""
        seq = self._next_seq()
        body = ex.body if ex.body is not None else (ex.text[:200000] if ex.text else None)
        raw = {
            "sonde": ex.probe, "slug": ex.slug,
            "request": {"method": ex.method, "url": str(request.url), "headers": dict(request.headers),
                        "body_summary": ex.body_summary},
            "status": ex.status, "headers": dict(ex.headers), "body": body,
            "elapsed_s": round(ex.elapsed, 3), "error": ex.error,
        }
        self._write_json(self.raw_dir / ("%04d_%s_%s.json" % (seq, ex.probe, ex.slug)), raw)

    # -- requêtes ----------------------------------------------------------

    def _send_once(self, probe: str, slug: str, method: str, url: str, *, json_body: Any, files: Any, data: Any,
                   params: Any, headers: Any, drop_auth: bool, timeout: float, summary: Any) -> Exchange:
        """Envoie une requête (sans retry) et construit l'échange."""
        request = self.client.build_request(method, url, json=json_body, files=files, data=data, params=params,
                                            headers=headers, timeout=timeout)
        if drop_auth and "authorization" in request.headers:
            del request.headers["authorization"]
        raw_path = request.url.raw_path.decode("ascii", "replace")
        ex = Exchange(probe, slug, method, raw_path, summary)
        start = self.clock()
        try:
            response = self.client.send(request)
            response.read()
            ex.status = response.status_code
            ex.headers = dict(response.headers)
            ex.text = response.text
            try:
                ex.body = response.json()
            except ValueError:
                ex.body = None
        except httpx.TimeoutException as exc:
            ex.error = "délai dépassé (%s)" % type(exc).__name__
        except httpx.TransportError as exc:
            ex.error = "erreur réseau (%s)" % type(exc).__name__
        ex.elapsed = self.clock() - start
        self._harvest_identity(ex.body)
        self._observe_headers(ex)
        self._write_raw(ex, request)
        return ex

    def _note_status(self, ex: Exchange) -> None:
        """Compte les 429/503 et capture le premier 429 naturel (fixture P13)."""
        if ex.status == 503:
            with self._lock:
                self.rate_info["compte_503"] += 1
        if ex.status != 429:
            return
        _, fmt = parse_retry_after(ex.headers.get("retry-after"), self.wall())
        first = False
        with self._lock:
            self.rate_info["compte_429"] += 1
            if fmt:
                self.rate_info["retry_after_formats"].add(fmt)
            if self.rate_info["premier_429"] is None:
                self.rate_info["premier_429"] = {"sonde": ex.probe, "chemin": ex.path.split("?", 1)[0],
                                                 "entetes": filter_fixture_headers(ex.headers),
                                                 "corps": ex.body if ex.body is not None else ex.text[:300]}
                first = True
        if first:
            self.record_fixture(ex, name="P13_natural_429.json")

    def call(self, probe: str, slug: str, method: str, url: str, *, json_body: Any = None, files: Any = None,
             data: Any = None, params: Any = None, kind: str = "default", headers: Any = None,
             drop_auth: bool = False, record: bool = True, retries: int = 2,
             expect_auth_error: bool = False) -> Exchange:
        """Envoie une requête avec délai, retry limité (429/503), bruts, fixture et contrôle de compte.

        Un 429 dont le Retry-After dépasse ``RETRY_AFTER_MAX_S`` signale un
        quota journalier épuisé : aucun nouvel essai, et arrêt du run
        (``AccountLevelError``) sauf avec ``expect_auth_error``. Une requête
        envoyée sans clé (``drop_auth``) ne déclenche jamais de diagnostic de
        compte : sa réponse ne dit rien de la clé.
        """
        timeout = TIMEOUTS.get(kind, TIMEOUTS["default"])
        if json_body is not None:
            summary = summarize_request_body(json_body)
        elif files:
            summary = {"multipart": _files_summary(files)}
        elif data:
            summary = {"form": dict(data)}
        else:
            summary = None
        attempt = 0
        quota_message = None
        while True:
            ex = self._send_once(probe, slug, method, url, json_body=json_body, files=files, data=data,
                                 params=params, headers=headers, drop_auth=drop_auth, timeout=timeout,
                                 summary=summary)
            self._note_status(ex)
            if ex.status in (429, 503):
                wait, _ = parse_retry_after(ex.headers.get("retry-after"), self.wall())
                if ex.status == 429 and wait is not None and wait > RETRY_AFTER_MAX_S:
                    self.log("[%s] quota journalier atteint (Retry-After %.0f s)" % (probe, wait))
                    quota_message = "quota journalier atteint (HTTP 429, Retry-After %.0f s)" % wait
                    break
                if attempt < retries:
                    wait = min(60.0, wait if wait is not None else 5.0 * (attempt + 1))
                    self.log("[%s] HTTP %s, nouvel essai dans %.0f s" % (probe, ex.status, wait))
                    self.sleep(wait)
                    attempt += 1
                    continue
            break
        if record:
            self.record_fixture(ex)
        if not expect_auth_error:
            message = quota_message or (None if drop_auth else account_error_message(ex.status, ex.text))
            if message:
                raise AccountLevelError(message)
        return ex

    def record_fixture(self, ex: Exchange, name: Optional[str] = None) -> str:
        """Écrit la fixture assainie d'un échange et la garde en mémoire."""
        body, notes = compact_for_fixture(ex.body if ex.body is not None else (ex.text[:2000] or None))
        fixture: Dict[str, Any] = {
            "request": {"method": ex.method, "path": ex.path, "body_summary": ex.body_summary},
            "status": ex.status,
            "headers": filter_fixture_headers(ex.headers),
            "body": body,
            "synthetic": False,
        }
        if notes:
            fixture["truncated"] = notes
        if ex.error:
            fixture["error"] = ex.error
        return self.write_fixture(name or "%s_%s.json" % (ex.probe, ex.slug), fixture)

    def write_fixture(self, name: str, fixture: Dict[str, Any]) -> str:
        """Écrit une fixture (nom unique dans le run) dans le dossier des fixtures."""
        with self._lock:
            final = name
            n = 2
            while final in self.fixtures:
                final = name[:-5] + "_%d.json" % n
                n += 1
            clean = self.sanitize(fixture)
            self.fixtures[final] = clean
        self._write_json(self.fixtures_dir / final, clean)
        return final

    def models(self) -> List[Dict[str, Any]]:
        """Liste des modèles (/v1/models), mise en cache pour le run."""
        if self._models_cache is None:
            ex = self.call("P1", "models_cache", "GET", "/models", record=False)
            data = ex.dig("data", default=[])
            self._models_cache = [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []
        return self._models_cache

    def run_probe(self, pid: str, func: Callable[["ProbeContext"], Dict[str, Any]]) -> None:
        """Exécute une sonde ; une exception Python est comptée comme erreur de script."""
        self.executed += 1
        self.executed_ids.append(pid)
        self.log("[%s] %s…" % (pid, PROBE_TITLES.get(pid, "")))
        try:
            result = func(self)
        except AccountLevelError:
            self.results[pid] = {"statut": "interrompu", "résumé": "erreur de compte, arrêt du run",
                                 "constats": {}, "assertions": []}
            raise
        except Exception as exc:  # une sonde défaillante n'arrête pas le run
            self.script_errors += 1
            result = {"statut": "erreur", "résumé": "exception du script : %s" % type(exc).__name__,
                      "constats": {}, "assertions": [], "erreur": "%s: %s" % (type(exc).__name__, exc)}
        self.results[pid] = self.sanitize(result)
        self.log("[%s] %s : %s" % (pid, result.get("statut"), result.get("résumé", "")))


# ---------------------------------------------------------------------------
# Aides aux sondes
# ---------------------------------------------------------------------------


def _check(assertions: List[Dict[str, Any]], label: str, ok: Any) -> bool:
    """Ajoute une assertion (attendu, ok) et renvoie son résultat."""
    assertions.append({"attendu": label, "ok": bool(ok)})
    return bool(ok)


def _result(assertions: List[Dict[str, Any]], resume: str, constats: Dict[str, Any],
            inconclusive: bool = False, statut: Optional[str] = None) -> Dict[str, Any]:
    """Construit le résultat d'une sonde (ok, écart, non concluant, sauté)."""
    if statut is None:
        if inconclusive:
            statut = "non concluant"
        else:
            statut = "ok" if all(a["ok"] for a in assertions) else "écart"
    return {"statut": statut, "résumé": resume, "constats": constats, "assertions": assertions}


def _choice(ex: Exchange) -> Dict[str, Any]:
    """Premier choix d'une réponse de chat (dict vide sinon)."""
    choice = ex.dig("choices", 0, default={})
    return choice if isinstance(choice, dict) else {}


def _message(ex: Exchange) -> Dict[str, Any]:
    """Message du premier choix (dict vide sinon)."""
    msg = _choice(ex).get("message")
    return msg if isinstance(msg, dict) else {}


def _content(ex: Exchange) -> Optional[str]:
    """Contenu textuel du premier choix."""
    content = _message(ex).get("content")
    return content if isinstance(content, str) else None


def _finish(ex: Exchange) -> Optional[str]:
    """finish_reason du premier choix."""
    return _choice(ex).get("finish_reason")


def _usage(ex: Exchange) -> Dict[str, Any]:
    """Objet usage de la réponse (dict vide sinon)."""
    usage = ex.dig("usage", default={})
    return usage if isinstance(usage, dict) else {}


def _detail_type(ex: Exchange) -> str:
    """Type du champ detail d'une erreur : str, list, autre ou absent."""
    if not isinstance(ex.body, dict) or "detail" not in ex.body:
        return "absent"
    detail = ex.body["detail"]
    if isinstance(detail, str):
        return "str"
    if isinstance(detail, list):
        return "list"
    return type(detail).__name__


def _error_view(ex: Exchange) -> Dict[str, Any]:
    """Vue compacte d'une réponse d'erreur (statut, type de detail, extrait)."""
    return {"statut": ex.status, "type_detail": _detail_type(ex), "extrait": (ex.text or "")[:300],
            "erreur_reseau": ex.error}


def _chat_payload(model: str, content: Any, **extra: Any) -> Dict[str, Any]:
    """Corps minimal d'un appel de chat avec un message utilisateur."""
    payload: Dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}]}
    payload.update(extra)
    return payload


def _chat(ctx: ProbeContext, probe: str, slug: str, payload: Dict[str, Any], kind: str = "chat",
          **kwargs: Any) -> Exchange:
    """POST /v1/chat/completions."""
    return ctx.call(probe, slug, "POST", "/chat/completions", json_body=payload, kind=kind, **kwargs)


def _usage_presence(ex: Exchange) -> Dict[str, Any]:
    """Présence de usage.cost et usage.impacts{kWh, kgCO2eq} dans une réponse."""
    usage = _usage(ex)
    impacts = usage.get("impacts")
    return {
        "statut": ex.status,
        "cost": "cost" in usage,
        "impacts": isinstance(impacts, dict) and {"kWh", "kgCO2eq"} <= set(impacts),
        "valeurs": {"cost": usage.get("cost"), "impacts": impacts, "requests": usage.get("requests")},
    }


def _router_field(models: List[Dict[str, Any]]) -> Optional[str]:
    """Champ des modèles pouvant être croisé avec limits[].router_id, s'il existe."""
    for candidate in ("router_id", "routerId", "router"):
        if any(isinstance(m.get(candidate), int) for m in models):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Sondes P1 à P13, P20, P21
# ---------------------------------------------------------------------------


def probe_p1(ctx: ProbeContext) -> Dict[str, Any]:
    """P1 : compte, expiration, budget, permissions et limites par routeur."""
    a: List[Dict[str, Any]] = []
    ex = ctx.call("P1", "me", "GET", "/me")
    me = ex.body if isinstance(ex.body, dict) else {}
    now = ctx.wall()
    expires = me.get("expires")
    _check(a, "GET /v1/me renvoie 200", ex.status == 200)
    expires_ok = expires is None or (isinstance(expires, (int, float)) and expires > now + 30 * 86400)
    _check(a, "expires nul ou au-delà de maintenant + 30 jours", ex.status == 200 and expires_ok)
    limits = [lim for lim in (me.get("limits") or []) if isinstance(lim, dict)]
    models = ctx.models() if ex.status == 200 else []
    field = _router_field(models)
    router_to_model: Dict[Any, str] = {}
    if field:
        router_to_model = {m.get(field): m.get("id") for m in models if m.get(field) is not None}
    quotas: Dict[str, Dict[str, Any]] = {}
    for lim in limits:
        model = lim.get("model") if isinstance(lim.get("model"), str) else router_to_model.get(lim.get("router_id"))
        if model:
            quotas.setdefault(model, {})[str(lim.get("type"))] = lim.get("value")
    wanted = [RECODE_MODEL, NOTES_MODEL, OCR_CHAT_MODEL, OCR_DOC_MODEL, EMBED_MODEL]
    candidates: Dict[str, List[str]] = {}
    for lim in limits:
        if lim.get("type") == "rpm" and lim.get("value") in DOCUMENTED_RPM_CANDIDATES:
            candidates[str(lim.get("router_id"))] = list(DOCUMENTED_RPM_CANDIDATES[lim["value"]])
    constats = {
        "me_ok": ex.status == 200,
        "statut_http": ex.status,
        "expires": expires,
        "expires_ok": expires_ok,
        "jours_restants": None if not isinstance(expires, (int, float)) else round((expires - now) / 86400, 1),
        "budget": me.get("budget"),
        "permissions": me.get("permissions"),
        "limits": limits,
        "router_ids": sorted({lim.get("router_id") for lim in limits if lim.get("router_id") is not None},
                             key=str),
        "champ_modele_router": field,
        "correspondance_router_possible": bool(quotas),
        "quotas_par_modele": {m: quotas.get(m) for m in wanted},
        "candidats_par_rpm_documente": candidates,
    }
    resume = "compte %s, %d limites, correspondance routeur→modèle %s" % (
        "valide" if expires_ok else "à vérifier", len(limits), "possible" if quotas else "impossible")
    return _result(a, resume, constats, inconclusive=ex.status is None)


def probe_p2(ctx: ProbeContext) -> Dict[str, Any]:
    """P2 : catalogue, types, alias, contexte et coûts."""
    a: List[Dict[str, Any]] = []
    ex = ctx.call("P2", "models", "GET", "/models")
    data = ex.dig("data", default=[])
    data = [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []
    if data:
        ctx._models_cache = data
    by_id = {m.get("id"): m for m in data}
    all_aliases = set()
    for m in data:
        all_aliases.update(a2 for a2 in (m.get("aliases") or []) if isinstance(a2, str))
    attendus = {}
    for mid, typ in EXPECTED_MODELS.items():
        m = by_id.get(mid)
        attendus[mid] = {
            "present": m is not None, "type": (m or {}).get("type"), "type_attendu": typ,
            "type_ok": bool(m) and m.get("type") == typ, "aliases": (m or {}).get("aliases"),
            "max_context_length": (m or {}).get("max_context_length"), "costs": (m or {}).get("costs"),
        }
        _check(a, "%s présent avec le type %s" % (mid, typ), attendus[mid]["type_ok"])
    missing = [al for al in EXPECTED_ALIASES if al not in all_aliases]
    _check(a, "alias openweight-large/small/ocr/embeddings présents", not missing)
    listed = OCR_DOC_MODEL in by_id
    constats = {
        "n_modeles": len(data),
        "ids": sorted(str(i) for i in by_id),
        "modeles_attendus": attendus,
        "alias_manquants": missing,
        "mistral_ocr_2512_liste": listed,
        "catalogue": [{k: m.get(k) for k in ("id", "type", "aliases", "max_context_length", "costs", "owned_by")}
                      for m in data],
    }
    resume = "%d modèles, alias manquants : %s, mistral-ocr-2512 listé : %s" % (
        len(data), ", ".join(missing) or "aucun", "oui" if listed else "non")
    return _result(a, resume, constats, inconclusive=ex.status is None)


def _ocr_doc_payload(pdf: bytes, pages: List[int]) -> Dict[str, Any]:
    """Corps /v1/ocr pour un PDF en data URI et des pages indexées à partir de 0."""
    uri = "data:application/pdf;base64," + base64.b64encode(pdf).decode("ascii")
    return {"model": OCR_DOC_MODEL, "document": {"type": "document_url", "document_url": uri},
            "pages": pages, "include_image_base64": False}


def probe_p3(ctx: ProbeContext) -> Dict[str, Any]:
    """P3 : accès à /v1/ocr et indices de pages à partir de 0."""
    a: List[Dict[str, Any]] = []
    pdf = make_two_page_pdf()
    ex0 = ctx.call("P3", "ocr_page0", "POST", "/ocr", json_body=_ocr_doc_payload(pdf, [0]), kind="ocr_doc",
                   retries=1)
    constats: Dict[str, Any] = {"statut_page0": ex0.status}
    if ex0.status == 200:
        access: Optional[bool] = True
        pages = ex0.dig("pages", default=[]) or []
        md0 = _norm_text(pages[0].get("markdown")) if pages and isinstance(pages[0], dict) else ""
        _check(a, "pages:[0] renvoie une seule page", len(pages) == 1)
        _check(a, "pages[0].index == 0", bool(pages) and isinstance(pages[0], dict) and pages[0].get("index") == 0)
        _check(a, "PAGE_ONE_7Q présent en page 0", "PAGE_ONE_7Q" in md0)
        _check(a, "PAGE_TWO_9Z absent de la page 0", "PAGE_TWO_9Z" not in md0)
        ex1 = ctx.call("P3", "ocr_page1", "POST", "/ocr", json_body=_ocr_doc_payload(pdf, [1]), kind="ocr_doc",
                       retries=1)
        pages1 = ex1.dig("pages", default=[]) or []
        md1 = _norm_text(pages1[0].get("markdown")) if pages1 and isinstance(pages1[0], dict) else ""
        _check(a, "pages:[1] renvoie PAGE_TWO_9Z", ex1.status == 200 and "PAGE_TWO_9Z" in md1)
        constats.update({
            "statut_page1": ex1.status,
            "index_page1": pages1[0].get("index") if pages1 and isinstance(pages1[0], dict) else None,
            "pages_zero_based": "PAGE_ONE_7Q" in md0 and "PAGE_TWO_9Z" in md1,
            "usage_info": ex0.dig("usage_info"),
            "usage_present": bool(_usage(ex0)),
            "usage_cost_impacts": _usage_presence(ex0),
        })
        resume = "accès /v1/ocr confirmé, pages à partir de 0 : %s" % constats["pages_zero_based"]
    elif ex0.status in (403, 404):
        access = False
        constats["signature_sans_acces"] = _error_view(ex0)
        resume = "pas d'accès à /v1/ocr (HTTP %s), voie chat LightOnOCR" % ex0.status
    else:
        access = None
        constats["reponse"] = _error_view(ex0)
        resume = "réponse inattendue de /v1/ocr (HTTP %s)" % ex0.status
    constats["acces"] = access
    ctx.state["ocr_access"] = access
    return _result(a, resume, constats, inconclusive=access is None)


def probe_p4(ctx: ProbeContext) -> Dict[str, Any]:
    """P4 : tailles de charge utile et latence de /v1/ocr (seulement avec accès)."""
    a: List[Dict[str, Any]] = []
    access = ctx.state.get("ocr_access")
    if access is not True:
        if "P3" not in ctx.results:
            reason = "P3 non exécutée dans ce run (lancer --only P3,P4)"
        elif access is False:
            reason = "pas d'accès à /v1/ocr"
        else:
            reason = "accès à /v1/ocr indéterminé (P3 non concluante)"
        return _result(a, "sauté : %s" % reason, {"acces": access, "raison": reason}, statut="sauté")
    start = ctx.clock()
    essais = []
    largest_ok = None
    first_failure = None
    for size in P4_SIZES_MB:
        if ctx.clock() - start > P4_MAX_TOTAL_S:
            essais.append({"taille_mo": size, "saute": "budget de temps atteint"})
            break
        pdf = make_noise_pdf(size)
        payload = _ocr_doc_payload(pdf, [0])
        ex = ctx.call("P4", "size_%dmb" % size, "POST", "/ocr", json_body=payload, kind="ocr_doc", retries=0)
        essai = {"taille_mo": size, "octets_pdf": len(pdf),
                 "octets_base64": len(payload["document"]["document_url"]),
                 "statut": ex.status, "duree_s": round(ex.elapsed, 2), "erreur": ex.error}
        essais.append(essai)
        if ex.status == 200:
            largest_ok = size
        else:
            first_failure = dict(essai, extrait=(ex.text or "")[:300])
            break
    cent = None
    if ctx.clock() - start <= P4_MAX_TOTAL_S:
        pdf100 = make_text_pdf(100)
        ex = ctx.call("P4", "pages_100", "POST", "/ocr",
                      json_body=_ocr_doc_payload(pdf100, list(range(100))), kind="ocr_doc", retries=0)
        cent = {"statut": ex.status, "duree_s": round(ex.elapsed, 2), "erreur": ex.error,
                "pages_renvoyees": len(ex.dig("pages", default=[]) or []),
                "pages_processed": ex.dig("usage_info", "pages_processed")}
        _check(a, "PDF de 100 pages traité (200)", ex.status == 200)
    _check(a, "au moins une taille acceptée", largest_ok is not None)
    constats = {"essais": essais, "plus_grand_ok_mo": largest_ok, "premier_echec": first_failure,
                "cent_pages": cent}
    resume = "plus grande taille acceptée : %s Mo ; premier échec : %s" % (
        largest_ok, first_failure["statut"] if first_failure else "aucun")
    return _result(a, resume, constats)


def _ocr_chat_payload(model: str, png: bytes, instruction: Optional[str] = None) -> Dict[str, Any]:
    """Corps d'OCR par chat : image seule (et consigne éventuelle), paramètres §9.1."""
    content: List[Dict[str, Any]] = []
    if instruction:
        content.append({"type": "text", "text": instruction})
    content.append({"type": "image_url", "image_url": {"url": _png_data_uri(png)}})
    return _chat_payload(model, content, max_tokens=4096, temperature=0.2, top_p=0.9)


def probe_p5(ctx: ProbeContext) -> Dict[str, Any]:
    """P5 : LightOnOCR par /v1/chat/completions (alias, id, consigne) et latence."""
    a: List[Dict[str, Any]] = []
    png = render_page_png(make_two_page_pdf(), 0)
    variants = {}
    for slug, model, instruction in (("ocr_chat_alias", OCR_CHAT_ALIAS, None),
                                     ("ocr_chat_id", OCR_CHAT_MODEL, None),
                                     ("ocr_chat_instruction", OCR_CHAT_MODEL, "Transcris en Markdown")):
        ex = _chat(ctx, "P5", slug, _ocr_chat_payload(model, png, instruction), kind="ocr_page")
        variants[slug] = {"statut": ex.status, "finish_reason": _finish(ex), "modele_servi": ex.dig("model"),
                          "jeton_trouve": "PAGE_ONE_7Q" in _norm_text(_content(ex)),
                          "extrait": (_content(ex) or "")[:200], "duree_s": round(ex.elapsed, 2)}
        if slug == "ocr_chat_alias":
            _check(a, "alias openweight-ocr accepté (200, pas 422)", ex.status == 200)
            _check(a, "finish_reason == stop", _finish(ex) == "stop")
            _check(a, "PAGE_ONE_7Q dans le contenu", variants[slug]["jeton_trouve"])
            _check(a, "modèle servi == lightonocr-2-1b", ex.dig("model") == OCR_CHAT_MODEL)
    if ctx.ocr_sample:
        pages = sample_pages_png(ctx.ocr_sample)
        echantillon = "fourni"
    else:
        pages = make_dense_pages_png(10)
        echantillon = "synthétique (pages denses générées localement)"
    durations, finishes, tokens = [], [], []
    for i, page_png in enumerate(pages):
        ex = _chat(ctx, "P5", "ocr_chat_latence", _ocr_chat_payload(OCR_CHAT_MODEL, page_png), kind="ocr_page",
                   record=(i == 0), retries=1)
        finishes.append(_finish(ex))
        if ex.status == 200:
            durations.append(ex.elapsed)
            tokens.append(_usage(ex).get("completion_tokens"))
    constats = {
        "chat_accepte": variants["ocr_chat_alias"]["statut"] == 200,
        "variantes": variants,
        "latence": {"echantillon": echantillon, "n": len(pages), "n_ok": len(durations),
                    "p50_s": percentile(durations, 50), "p95_s": percentile(durations, 95),
                    "finish_reasons": finishes, "completion_tokens": tokens},
    }
    resume = "chat OCR %s, p50 %s s, p95 %s s" % (
        "accepté" if constats["chat_accepte"] else "refusé", constats["latence"]["p50_s"],
        constats["latence"]["p95_s"])
    return _result(a, resume, constats)


def probe_p6(ctx: ProbeContext) -> Dict[str, Any]:
    """P6 : forme des erreurs (mauvais type, modèle inconnu, clé invalide)."""
    a: List[Dict[str, Any]] = []
    ex_a = ctx.call("P6", "wrong_model_type", "POST", "/embeddings",
                    json_body={"model": NOTES_MODEL, "input": ["x"]}, kind="embed", retries=0)
    ex_b = _chat(ctx, "P6", "unknown_model", _chat_payload("does-not-exist-ragpy", "x"), retries=0)
    ex_c = ctx.call("P6", "invalid_key", "GET", "/models", retries=0, expect_auth_error=True,
                    headers={"Authorization": "Bearer fake-invalid-ragpy-probe-key"})
    _check(a, "mauvais type de modèle : 422", ex_a.status == 422)
    _check(a, "modèle inconnu : 404", ex_b.status == 404)
    _check(a, "clé invalide : 401", ex_c.status == 401)
    constats = {
        "wrong_model_type": dict(_error_view(ex_a), contient_wrong_model_type="wrong model type" in
                                 (ex_a.text or "").lower()),
        "unknown_model": _error_view(ex_b),
        "invalid_key": _error_view(ex_c),
    }
    resume = "statuts %s / %s / %s ; detail 422 de type %s" % (
        ex_a.status, ex_b.status, ex_c.status, constats["wrong_model_type"]["type_detail"])
    return _result(a, resume, constats)


def probe_p7(ctx: ProbeContext) -> Dict[str, Any]:
    """P7 : présence de usage.cost et usage.impacts sur chat, embeddings, OCR chat et search."""
    a: List[Dict[str, Any]] = []
    par_endpoint = {}
    ex = _chat(ctx, "P7", "chat_usage", _chat_payload(RECODE_MODEL, "Réponds seulement : OK.", max_tokens=8,
                                                      temperature=0.1))
    par_endpoint["chat"] = _usage_presence(ex)
    ex = ctx.call("P7", "embeddings_usage", "POST", "/embeddings",
                  json_body={"model": EMBED_MODEL, "input": ["usage"], "encoding_format": "float"}, kind="embed")
    par_endpoint["embeddings"] = _usage_presence(ex)
    png = render_page_png(make_two_page_pdf(), 0, max_side=800)
    ex = _chat(ctx, "P7", "ocr_chat_usage", _ocr_chat_payload(OCR_CHAT_ALIAS, png), kind="ocr_page")
    par_endpoint["ocr_chat"] = _usage_presence(ex)
    lookup = ctx.call("P7", "public_collection_lookup", "GET", "/collections",
                      params={"visibility": "public", "limit": 10}, kind="collections", record=False)
    items = lookup.dig("data", default=[]) or []
    public_id = next((c.get("id") for c in items if isinstance(c, dict) and c.get("visibility") == "public"), None)
    if isinstance(public_id, int):
        ex = ctx.call("P7", "search_usage", "POST", "/search",
                      json_body={"query": "loi", "collection_ids": [public_id], "method": "lexical", "limit": 1})
        par_endpoint["search"] = _usage_presence(ex)
    else:
        par_endpoint["search"] = {"statut": None, "note": "aucune collection publique listée, voir P17"}
    for name, view in par_endpoint.items():
        if view.get("statut") == 200:
            _check(a, "%s : usage.cost présent" % name, view.get("cost"))
            _check(a, "%s : usage.impacts{kWh,kgCO2eq} présent" % name, view.get("impacts"))
    resume = ", ".join("%s cost=%s impacts=%s" % (k, v.get("cost"), v.get("impacts")) for k, v in par_endpoint.items())
    return _result(a, resume, {"par_endpoint": par_endpoint})


def probe_p8(ctx: ProbeContext) -> Dict[str, Any]:
    """P8 : max_completion_tokens, max_tokens, et les deux ensemble."""
    a: List[Dict[str, Any]] = []
    prompt = "Écris 200 mots sur la Loire."

    def run(slug: str, **limits: int) -> Dict[str, Any]:
        """Un appel ministral avec les bornes données."""
        ex = _chat(ctx, "P8", slug, _chat_payload(RECODE_MODEL, prompt, temperature=0.1, **limits))
        return {"statut": ex.status, "completion_tokens": _usage(ex).get("completion_tokens"),
                "finish_reason": _finish(ex), "extrait": (ex.text or "")[:300] if not ex.ok() else None}

    mct = run("max_completion_tokens_5", max_completion_tokens=5)
    mt = run("max_tokens_5", max_tokens=5)
    both = run("both_5_50", max_tokens=5, max_completion_tokens=50)

    def respected(view: Dict[str, Any], bound: int) -> bool:
        """Vrai si la borne a été respectée (tokens ≤ borne et finish length)."""
        tokens = view.get("completion_tokens")
        return view["statut"] == 200 and isinstance(tokens, int) and tokens <= bound and \
            view.get("finish_reason") == "length"

    mct_ok = respected(mct, 5)
    mt_ok = respected(mt, 5)
    _check(a, "max_completion_tokens:5 respecté", mct_ok)
    _check(a, "max_tokens:5 respecté", mt_ok)
    tokens = both.get("completion_tokens")
    if both["statut"] == 422:
        gagnant = "422"
    elif both["statut"] == 200 and isinstance(tokens, int):
        gagnant = "max_tokens" if tokens <= 5 else "max_completion_tokens"
    else:
        gagnant = None
    constats = {"max_completion_tokens": mct, "max_tokens": mt, "les_deux": both,
                "max_completion_tokens_respecte": mct_ok, "max_tokens_respecte": mt_ok, "gagnant_les_deux": gagnant}
    resume = "max_tokens respecté : %s ; max_completion_tokens respecté : %s ; les deux : %s" % (mt_ok, mct_ok,
                                                                                               gagnant)
    return _result(a, resume, constats)


def probe_p9(ctx: ProbeContext) -> Dict[str, Any]:
    """P9 : piège du raisonnement de gpt-oss, champ de raisonnement, effet de reasoning_effort."""
    a: List[Dict[str, Any]] = []
    question = "Combien font 17*23 ? Réponds par le nombre seul."
    trap = _chat(ctx, "P9", "reasoning_trap_64", _chat_payload(NOTES_MODEL, question, max_tokens=64))
    trap_confirmed = trap.status == 200 and _finish(trap) == "length" and not (_content(trap) or "").strip()
    _check(a, "max_tokens 64 : finish length et contenu vide (piège confirmé)", trap_confirmed)
    views = {}
    field = None
    for effort in ("low", "high"):
        ex = _chat(ctx, "P9", "reasoning_%s_2048" % effort,
                   _chat_payload(NOTES_MODEL, question, max_tokens=2048, reasoning_effort=effort))
        msg = _message(ex)
        found = None
        for name in ("reasoning_content", "reasoning"):
            if isinstance(msg.get(name), str) and msg.get(name).strip():
                found = name
                break
        field = field or found
        usage = _usage(ex)
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"),
                                                                       dict) else {}
        views[effort] = {"statut": ex.status, "finish_reason": _finish(ex), "contenu": (_content(ex) or "")[:100],
                         "contient_391": "391" in (_content(ex) or ""), "champ_raisonnement": found,
                         "cles_message": sorted(msg), "completion_tokens": usage.get("completion_tokens"),
                         "reasoning_tokens": details.get("reasoning_tokens")}
    _check(a, "max_tokens 2048 + effort low : 391 dans le contenu", views["low"]["contient_391"])
    low_tokens = views["low"]["completion_tokens"]
    high_tokens = views["high"]["completion_tokens"]
    includes = None
    if isinstance(low_tokens, int):
        includes = low_tokens > len((views["low"]["contenu"] or "").split()) + 10
    effect = None
    if isinstance(low_tokens, int) and isinstance(high_tokens, int):
        effect = high_tokens != low_tokens
    constats = {"piege": {"statut": trap.status, "finish_reason": _finish(trap), "contenu_vide":
                          not (_content(trap) or "").strip()},
                "piege_confirme": trap_confirmed, "champ_raisonnement": field, "efforts": views,
                "completion_inclut_raisonnement": includes, "effet_reasoning_effort": effect}
    resume = "piège confirmé : %s ; champ : %s ; effort low→high : %s → %s tokens" % (
        trap_confirmed, field, low_tokens, high_tokens)
    return _result(a, resume, constats)


def _valid_keep(content: Optional[str], strict: bool) -> bool:
    """Valide {"keep": [entiers]} (sans clé supplémentaire en mode strict)."""
    try:
        obj = json.loads(content or "")
    except ValueError:
        return False
    if not isinstance(obj, dict) or not isinstance(obj.get("keep"), list):
        return False
    if strict and set(obj) != {"keep"}:
        return False
    return all(isinstance(x, int) and not isinstance(x, bool) for x in obj["keep"])


def probe_p10(ctx: ProbeContext) -> Dict[str, Any]:
    """P10 : json_schema strict et json_object (ministral), json_schema (gpt-oss)."""
    a: List[Dict[str, Any]] = []
    schema = {"type": "json_schema", "json_schema": {
        "name": "filtre",
        "schema": {"type": "object", "properties": {"keep": {"type": "array", "items": {"type": "integer"}}},
                   "required": ["keep"], "additionalProperties": False},
        "strict": True}}
    messages = [
        {"role": "system", "content": "Tu es un filtre de citations. Réponds uniquement en JSON."},
        {"role": "user", "content": "Parmi ces citations, lesquelles relèvent de la sociologie des sciences ? "
                                    "Réponds en JSON sous la forme {\"keep\": [numéros]}.\n" + "\n".join(CITATIONS_P10)},
    ]
    cases = (("json_schema_ministral", RECODE_MODEL, schema, 300, True),
             ("json_object_ministral", RECODE_MODEL, {"type": "json_object"}, 300, False),
             ("json_schema_gpt_oss", NOTES_MODEL, schema, 4096, True))
    views = {}
    for slug, model, fmt, max_tokens, strict in cases:
        ex = _chat(ctx, "P10", slug, {"model": model, "messages": messages, "response_format": fmt,
                                      "max_tokens": max_tokens, "temperature": 0.1})
        ok = ex.status == 200 and _valid_keep(_content(ex), strict)
        views[slug] = {"statut": ex.status, "valide": ok, "finish_reason": _finish(ex),
                       "contenu": (_content(ex) or "")[:200]}
        _check(a, "%s : JSON valide" % slug, ok)
    constats = {k: v["valide"] for k, v in views.items()}
    constats["details"] = views
    resume = ", ".join("%s=%s" % (k, v["valide"]) for k, v in views.items())
    return _result(a, resume, constats)


def probe_p11(ctx: ProbeContext) -> Dict[str, Any]:
    """P11 : deux appels identiques avec seed 42 (déterminisme)."""
    a: List[Dict[str, Any]] = []
    payload = _chat_payload(RECODE_MODEL, "Résume en deux phrases : " + FRENCH_PARAGRAPHS[0],
                            temperature=0.1, seed=42, max_tokens=200)
    runs = [_chat(ctx, "P11", "seed_run_%d" % i, payload) for i in (1, 2)]
    contents = [_content(r) for r in runs]
    models = [r.dig("model") for r in runs]
    fps = [r.dig("system_fingerprint") for r in runs]
    identical = all(r.status == 200 for r in runs) and contents[0] is not None and contents[0] == contents[1]
    model_stable = models[0] is not None and models[0] == models[1]
    fp_stable = None if fps[0] is None and fps[1] is None else fps[0] == fps[1]
    _check(a, "sorties identiques avec seed 42", identical)
    constats = {"identiques": identical, "modele_stable": model_stable, "modeles": models,
                "fingerprint_stable": fp_stable, "statuts": [r.status for r in runs]}
    resume = "identiques : %s ; modèle stable : %s" % (identical, model_stable)
    return _result(a, resume, constats)


def probe_p12(ctx: ProbeContext) -> Dict[str, Any]:
    """P12 : lot de 64, dimension 1024, norme, 65 entrées, chaîne vide, alias."""
    a: List[Dict[str, Any]] = []
    texts = ["Phrase de test numéro %d sur la sociologie des sciences." % i for i in range(65)]

    def embed(slug: str, inputs: List[str], model: str = EMBED_MODEL) -> Exchange:
        """POST /v1/embeddings."""
        return ctx.call("P12", slug, "POST", "/embeddings", kind="embed", retries=1,
                        json_body={"model": model, "input": inputs, "encoding_format": "float"})

    ex64 = embed("batch_64", texts[:64])
    data = ex64.dig("data", default=[]) or []
    items = [d for d in data if isinstance(d, dict)]
    indices = sorted(d.get("index") for d in items if isinstance(d.get("index"), int))
    dims = sorted({len(d.get("embedding") or []) for d in items})
    norms = [math.sqrt(sum(x * x for x in d.get("embedding") or [])) for d in items]
    _check(a, "64 entrées : 200", ex64.status == 200)
    _check(a, "64 vecteurs renvoyés", len(items) == 64)
    _check(a, "indices 0..63", indices == list(range(64)))
    _check(a, "dimension 1024", dims == [1024])
    ex65 = embed("batch_65", texts)
    ex_empty = embed("empty_string", [""])
    ex_alias = embed("alias", ["alias"], model=EMBED_ALIAS)
    _check(a, "alias openweight-embeddings servi par bge-m3", ex_alias.dig("model") == EMBED_MODEL)
    constats = {
        "batch_64_ok": ex64.status == 200 and len(items) == 64, "dimensions": dims,
        "indices_ordonnes": [d.get("index") for d in items] == list(range(len(items))),
        "norme_min": round(min(norms), 6) if norms else None, "norme_max": round(max(norms), 6) if norms else None,
        "modele_servi": ex64.dig("model"), "statut_65": ex65.status, "extrait_65": (ex65.text or "")[:300],
        "statut_vide": ex_empty.status, "extrait_vide": (ex_empty.text or "")[:300],
        "alias_modele_servi": ex_alias.dig("model"),
    }
    resume = "dimension %s, norme %s–%s, 65 entrées → %s, chaîne vide → %s" % (
        dims, constats["norme_min"], constats["norme_max"], ex65.status, ex_empty.status)
    return _result(a, resume, constats)


def rate_info_view(ctx: ProbeContext) -> Dict[str, Any]:
    """Vue sérialisable des informations de limitation collectées pendant le run."""
    info = ctx.rate_info
    return {"entetes_ratelimit_vus": sorted(info["entetes_vus"]), "exemples": info["exemples"],
            "formats_retry_after": sorted(info["retry_after_formats"]), "premier_429": info["premier_429"],
            "compte_429": info["compte_429"], "compte_503": info["compte_503"]}


def probe_p13(ctx: ProbeContext) -> Dict[str, Any]:
    """P13 : en-têtes de limitation observés passivement (jamais de rafale)."""
    a: List[Dict[str, Any]] = []
    _chat(ctx, "P13", "chat_headers", _chat_payload(RECODE_MODEL, "Réponds seulement : OK.", max_tokens=5))
    ctx.call("P13", "embed_headers", "POST", "/embeddings", kind="embed",
             json_body={"model": EMBED_MODEL, "input": ["en-têtes"], "encoding_format": "float"})
    constats = rate_info_view(ctx)
    resume = "en-têtes vus : %s ; 429 naturels : %d" % (", ".join(constats["entetes_ratelimit_vus"]) or "aucun",
                                                        constats["compte_429"])
    return _result(a, resume, constats)


def probe_p20(ctx: ProbeContext) -> Dict[str, Any]:
    """P20 : /health (sans clé), /health/models (avec clé), /v1/health (404 attendu)."""
    a: List[Dict[str, Any]] = []
    root = root_url(str(ctx.client.base_url))
    ex_h = ctx.call("P20", "health", "GET", root + "/health", drop_auth=True, expect_auth_error=True)
    ex_m = ctx.call("P20", "health_models", "GET", root + "/health/models")
    ex_v = ctx.call("P20", "v1_health", "GET", "/health")
    _check(a, "/health : 200", ex_h.status == 200)
    _check(a, "/health/models : 200", ex_m.status == 200)
    _check(a, "/v1/health : 404", ex_v.status == 404)
    body = ex_m.body
    if isinstance(body, dict):
        shape: Any = {"cles": sorted(body)[:30]}
        sample = next((v for v in body.values() if isinstance(v, (dict, list))), None)
        if isinstance(sample, dict):
            shape["cles_exemple"] = sorted(sample)[:30]
        elif isinstance(sample, list) and sample and isinstance(sample[0], dict):
            shape["cles_exemple"] = sorted(sample[0])[:30]
    elif isinstance(body, list):
        shape = {"liste": len(body), "cles_exemple": sorted(body[0])[:30] if body and isinstance(body[0], dict)
                 else None}
    else:
        shape = None
    constats = {"health": ex_h.status, "health_models": ex_m.status, "v1_health": ex_v.status,
                "forme_health_models": shape}
    resume = "/health %s, /health/models %s, /v1/health %s" % (ex_h.status, ex_m.status, ex_v.status)
    return _result(a, resume, constats)


def _timed_batch(ctx: ProbeContext, slug: str, payload: Dict[str, Any], n: int, concurrency: int,
                 kind: str) -> Dict[str, Any]:
    """Lance ``n`` appels identiques avec ``concurrency`` threads ; latences et statuts."""
    def one(i: int) -> Exchange:
        """Un appel de la série (le premier seulement est enregistré en fixture)."""
        return _chat(ctx, "P21", slug, payload, kind=kind, record=(i == 0), retries=0)

    start = ctx.clock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        exchanges = list(pool.map(one, range(n)))
    wall_s = ctx.clock() - start
    durations = [ex.elapsed for ex in exchanges if ex.status == 200]
    out_tokens = [_usage(ex).get("completion_tokens") for ex in exchanges if ex.status == 200]
    statuts: Dict[str, int] = {}
    for ex in exchanges:
        key = str(ex.status if ex.status is not None else ex.error)
        statuts[key] = statuts.get(key, 0) + 1
    numeric = [t for t in out_tokens if isinstance(t, int)]
    return {"n": n, "concurrence": concurrency, "n_ok": len(durations), "p50_s": percentile(durations, 50),
            "p95_s": percentile(durations, 95), "duree_totale_s": round(wall_s, 2), "statuts": statuts,
            "completion_tokens_moyen": round(statistics.mean(numeric), 1) if numeric else None,
            "modeles_servis": sorted({str(ex.dig("model")) for ex in exchanges if ex.status == 200})}


def probe_p21(ctx: ProbeContext) -> Dict[str, Any]:
    """P21 : débit de référence (recodage ministral et notes gpt-oss, concurrence 1 puis 8)."""
    a: List[Dict[str, Any]] = []
    chunk = french_ocr_chunk()
    recode = {"model": RECODE_MODEL, "temperature": 0.1, "max_tokens": 2000, "messages": [
        {"role": "system", "content": RECODE_SYSTEM},
        {"role": "user", "content": RECODE_TEMPLATE.format(instructions=RECODE_INSTRUCTIONS, chunk=chunk)}]}
    note = {"model": NOTES_MODEL, "max_tokens": 6000, "reasoning_effort": "medium", "messages": [
        {"role": "system", "content": "Tu es un assistant de recherche en sciences humaines et sociales."},
        {"role": "user", "content": "Rédige une note de lecture académique structurée (problématique, thèses, "
                                    "méthode, apports, limites) du texte suivant :\n\n" + chunk}]}
    series = {}
    for conc in (1, 8):
        series["c%d" % conc] = {
            "recode": _timed_batch(ctx, "recode_c%d" % conc, recode, 20, conc, "chat"),
            "notes": _timed_batch(ctx, "notes_c%d" % conc, note, 5, conc, "notes"),
        }
    _check(a, "recodage : au moins 15 réponses 200 en concurrence 1", series["c1"]["recode"]["n_ok"] >= 15)
    _check(a, "notes : au moins 3 réponses 200 en concurrence 1", series["c1"]["notes"]["n_ok"] >= 3)
    constats = {"chunk_caracteres": len(chunk), "series": series}
    resume = "recodage p50 %s s (c1) / %s s (c8) ; notes p50 %s s (c1)" % (
        series["c1"]["recode"]["p50_s"], series["c8"]["recode"]["p50_s"], series["c1"]["notes"]["p50_s"])
    return _result(a, resume, constats)


# ---------------------------------------------------------------------------
# Collections : P14 à P19
# ---------------------------------------------------------------------------


def _hexes(n: int = 64) -> List[str]:
    """content_hash factices déterministes (64 caractères hexadécimaux)."""
    return [hashlib.sha256(("ragpy-probe-%d" % i).encode("ascii")).hexdigest() for i in range(n)]


def _collection_id(ctx: ProbeContext) -> Optional[int]:
    """Id de la collection de sonde du run (None si sa création a échoué)."""
    coll = ctx.state.get("collection") or {}
    return coll.get("id")


def _as_int_id(value: Any) -> Optional[int]:
    """Id entier d'une réponse : entier, ou chaîne de chiffres (``"7"``) ; None sinon."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _create_document(ctx: ProbeContext, probe: str, slug: str, name: str, cid: int,
                     metadata: Optional[Dict[str, Any]] = None) -> Tuple[Exchange, Optional[int]]:
    """Crée un document vide en multipart (sans fichier) et renvoie (échange, id)."""
    files: Dict[str, Any] = {"name": (None, name), "collection_id": (None, str(cid))}
    if metadata is not None:
        files["metadata"] = (None, json.dumps(metadata))
    ex = ctx.call(probe, slug, "POST", "/documents", files=files, kind="collections")
    doc_id = ex.dig("id")
    return ex, doc_id if isinstance(doc_id, int) else None


def _push(ctx: ProbeContext, probe: str, slug: str, doc_id: int, chunks: List[Dict[str, Any]],
          **kwargs: Any) -> Exchange:
    """POST /v1/documents/{id}/chunks."""
    return ctx.call(probe, slug, "POST", "/documents/%d/chunks" % doc_id, json_body={"chunks": chunks},
                    kind="collections", **kwargs)


def _hits(ex: Exchange) -> List[Dict[str, Any]]:
    """Résultats d'une recherche (liste de dicts)."""
    data = ex.dig("data", default=[]) or []
    return [d for d in data if isinstance(d, dict)]


def _search(ctx: ProbeContext, probe: str, slug: str, body: Dict[str, Any], **kwargs: Any) -> Exchange:
    """POST /v1/search."""
    return ctx.call(probe, slug, "POST", "/search", json_body=body, **kwargs)


def _no_collection(probe: str) -> Dict[str, Any]:
    """Résultat non concluant faute de collection de sonde."""
    return _result([], "collection de sonde absente", {"collection": None}, inconclusive=True)


def probe_p14(ctx: ProbeContext) -> Dict[str, Any]:
    """P14 : doublon de nom, filtre name, limit=101, suppression et 404 ensuite."""
    a: List[Dict[str, Any]] = []
    coll = ctx.state.get("collection") or {}
    cid = coll.get("id")
    _check(a, "création : 201 avec un id entier", coll.get("create_status") == 201 and isinstance(cid, int))
    if cid is None:
        return _no_collection("P14")
    name = coll["name"]
    dup = ctx.call("P14", "create_duplicate", "POST", "/collections", kind="collections",
                   json_body={"name": name, "description": "probe", "visibility": "private"})
    dup_id = _as_int_id(dup.dig("id")) if dup.ok() else None
    if isinstance(dup_id, int):
        ctx.created_collections.append(dup_id)
    duplicates_allowed = True if isinstance(dup_id, int) else (False if dup.status == 409 else None)
    exact = ctx.call("P14", "filter_exact_name", "GET", "/collections", params={"name": name, "limit": 100},
                     kind="collections")
    prefix = ctx.call("P14", "filter_prefix_name", "GET", "/collections", params={"name": name[:-3], "limit": 100},
                      kind="collections")
    exact_names = [c.get("name") for c in _hits(exact)]
    prefix_ids = [c.get("id") for c in _hits(prefix)]
    if prefix.status == 200 and cid in prefix_ids:
        name_filter = "substring"
    elif prefix.status == 200 and exact.status == 200 and name in exact_names:
        name_filter = "exact"
    else:
        name_filter = None
    lim = ctx.call("P14", "limit_101", "GET", "/collections", params={"limit": 101}, kind="collections")
    _check(a, "limit=101 : 422", lim.status == 422)
    delete_dup = None
    if isinstance(dup_id, int):
        d = ctx.call("P14", "delete_duplicate", "DELETE", "/collections/%d" % dup_id, kind="collections")
        g = ctx.call("P14", "get_deleted_duplicate", "GET", "/collections/%d" % dup_id, kind="collections")
        delete_dup = {"delete": d.status, "get_apres": g.status}
        if d.status in (200, 204) and dup_id in ctx.created_collections:
            ctx.created_collections.remove(dup_id)
        _check(a, "suppression du doublon : 204", d.status == 204)
        _check(a, "GET après suppression : 404", g.status == 404)
    constats = {"type_id_creation": coll.get("id_type"), "doublon_autorise": duplicates_allowed,
                "statut_doublon": dup.status,
                "filtre_nom": name_filter, "n_resultats_nom_exact": len(exact_names),
                "n_resultats_prefixe": len(prefix_ids), "statut_limit_101": lim.status,
                "suppression_doublon": delete_dup}
    resume = "doublon %s, filtre name %s, limit=101 → %s" % (
        "autorisé" if duplicates_allowed else ("refusé (409)" if duplicates_allowed is False else "?"),
        name_filter, lim.status)
    return _result(a, resume, constats)


def probe_p15(ctx: ProbeContext) -> Dict[str, Any]:
    """P15 : document sans fichier en multipart, en urlencoded, avec métadonnées."""
    a: List[Dict[str, Any]] = []
    cid = _collection_id(ctx)
    if cid is None:
        return _no_collection("P15")
    ex_a, doc_a = _create_document(ctx, "P15", "multipart_no_file", "probe-doc", cid)
    ex_b = ctx.call("P15", "urlencoded", "POST", "/documents", kind="collections",
                    data={"name": "probe-doc-urlencoded", "collection_id": str(cid)})
    ex_c, doc_c = _create_document(ctx, "P15", "multipart_metadata", "probe-doc-meta", cid,
                                   metadata={"source": "probe", "annee": 2026})
    ctx.state["doc_a"] = doc_a
    ctx.state["doc_c"] = doc_c
    _check(a, "multipart sans fichier : 201 {id}", ex_a.status == 201 and doc_a is not None)
    shape = None
    chunks0 = None
    if doc_a is not None:
        g = ctx.call("P15", "get_document", "GET", "/documents/%d" % doc_a, kind="collections")
        shape = sorted(g.body) if isinstance(g.body, dict) else None
        chunks0 = g.dig("chunks")
        _check(a, "GET du document : chunks == 0", chunks0 == 0)
    constats = {"multipart_sans_fichier": ex_a.status == 201 and doc_a is not None, "statut_multipart": ex_a.status,
                "urlencoded_accepte": ex_b.ok(), "statut_urlencoded": ex_b.status,
                "extrait_urlencoded": (ex_b.text or "")[:300] if not ex_b.ok() else None,
                "metadata_document_acceptee": ex_c.ok(), "statut_metadata": ex_c.status,
                "forme_document": shape, "chunks_initial": chunks0}
    resume = "multipart %s, urlencoded %s, métadonnées %s" % (ex_a.status, ex_b.status, ex_c.status)
    return _result(a, resume, constats)


def _meta10(i: int, hexes: List[str]) -> Dict[str, Any]:
    """Métadonnées à 10 propriétés d'un chunk de sonde."""
    return {"content_hash": hexes[i], "title": "T", "year": 2026, "score": 0.5, "flag": True, "chunk_index": i,
            "total_chunks": 64, "doi": "10.1/x", "zotero_key": "ABCD1234", "source_file": "a.pdf"}


def _doc_chunk_count(ctx: ProbeContext, probe: str, slug: str, doc_id: int) -> Optional[int]:
    """Nombre de chunks d'un document (GET /v1/documents/{id})."""
    g = ctx.call(probe, slug, "GET", "/documents/%d" % doc_id, kind="collections")
    count = g.dig("chunks")
    return count if isinstance(count, int) else None


def probe_p16(ctx: ProbeContext) -> Dict[str, Any]:
    """P16 : envoi de 64 chunks, limites de lot et de métadonnées, fusion, atomicité."""
    a: List[Dict[str, Any]] = []
    cid = _collection_id(ctx)
    if cid is None:
        return _no_collection("P16")
    doc_a = ctx.state.get("doc_a")
    if doc_a is None:
        _, doc_a = _create_document(ctx, "P16", "setup_document", "probe-doc", cid)
        ctx.state["doc_a"] = doc_a
    if doc_a is None:
        return _result(a, "document de sonde absent", {}, inconclusive=True)
    hexes = _hexes(64)
    chunks = [{"content": "txt %d : chunk de sonde RAGpy numéro %d." % (i, i), "metadata": _meta10(i, hexes)}
              for i in range(64)]
    push = _push(ctx, "P16", "push_64", doc_a, chunks)
    ctx.state["hexes"] = hexes if push.ok() else None
    _check(a, "64 chunks à 10 propriétés : 2xx", push.ok())
    body = push.body
    ids_returned = False
    if isinstance(body, dict):
        ids_returned = any(k in body for k in ("ids", "data", "chunks", "id"))
    elif isinstance(body, list):
        ids_returned = bool(body)
    count = _doc_chunk_count(ctx, "P16", "get_document_after_push", doc_a)
    _check(a, "GET du document : chunks == 64", count == 64)
    listing = ctx.call("P16", "list_chunks", "GET", "/documents/%d/chunks" % doc_a, params={"limit": 100},
                       kind="collections")
    chunk_ids = sorted(c.get("id") for c in _hits(listing) if isinstance(c.get("id"), int))
    numbering = None
    if chunk_ids:
        contiguous = chunk_ids == list(range(chunk_ids[0], chunk_ids[0] + len(chunk_ids)))
        numbering = {"premier": chunk_ids[0], "dernier": chunk_ids[-1], "contigus": contiguous,
                     "a_partir_de_0": chunk_ids[0] == 0}

    _, doc_lim = _create_document(ctx, "P16", "setup_limits_document", "probe-doc-limits", cid)
    limits_views: Dict[str, Any] = {}
    if doc_lim is not None:
        bad_hexes = [h[::-1] for h in _hexes(66)]

        def attempt(slug: str, meta: Dict[str, Any], n: int = 1) -> Optional[int]:
            """Envoie ``n`` chunks avec les métadonnées données et renvoie le statut."""
            items = [{"content": "limite %s %d" % (slug, i), "metadata": dict(meta, content_hash=bad_hexes[i])}
                     for i in range(n)]
            ex = _push(ctx, "P16", slug, doc_lim, items, retries=0)
            limits_views[slug] = {"statut": ex.status, "extrait": (ex.text or "")[:300] if not ex.ok() else None}
            return ex.status

        base = {"title": "T"}
        attempt("push_65", base, n=65)
        attempt("meta_11_props", dict({"k%d" % i: i for i in range(10)}))
        attempt("meta_str_256", dict(base, title="x" * 256))
        attempt("meta_empty_string", dict(base, title=""))
        attempt("meta_none", dict(base, title=None))
        attempt("meta_list", dict(base, tags=["a", "b"]))
        attempt("meta_key_256", {"k" * 256: 1})
        before = _doc_chunk_count(ctx, "P16", "get_limits_before_atomicity", doc_lim)
        batch = [{"content": "atomicité %d" % i, "metadata": {"i": i}} for i in range(2)]
        batch.append({"content": "atomicité 2", "metadata": {"i": 2, "tags": ["a"]}})
        atom = _push(ctx, "P16", "atomicity_batch", doc_lim, batch, retries=0)
        after = _doc_chunk_count(ctx, "P16", "get_limits_after_atomicity", doc_lim)
        atomic = None
        if isinstance(before, int) and isinstance(after, int) and not atom.ok():
            atomic = after == before
        limits_views["atomicite"] = {"statut": atom.status, "chunks_avant": before, "chunks_apres": after,
                                     "atomique": atomic}
        _check(a, "65 chunks : 422", limits_views["push_65"]["statut"] == 422)
        _check(a, "11 propriétés : 422", limits_views["meta_11_props"]["statut"] == 422)
        _check(a, "chaîne de 256 caractères : 422", limits_views["meta_str_256"]["statut"] == 422)
        _check(a, "liste : 422", limits_views["meta_list"]["statut"] == 422)
        _check(a, "clé de 256 caractères : 422", limits_views["meta_key_256"]["statut"] == 422)

    merge: Dict[str, Any] = {}
    doc_c = ctx.state.get("doc_c")
    if doc_c is not None:
        p = _push(ctx, "P16", "merge_push", doc_c, [{"content": "fusion", "metadata": {"b": 1}}])
        g = ctx.call("P16", "merge_get", "GET", "/documents/%d/chunks" % doc_c, params={"limit": 10},
                     kind="collections")
        hits = _hits(g)
        meta = hits[0].get("metadata") if hits and isinstance(hits[0].get("metadata"), dict) else {}
        merged = None if not hits else ("source" in meta and "b" in meta)
        cap = _push(ctx, "P16", "merge_cap_push", doc_c,
                    [{"content": "fusion plafond", "metadata": {"m%d" % i: i for i in range(9)}}], retries=0)
        merge = {"statut_push": p.status, "cles_chunk": sorted(meta), "fusion": merged,
                 "premier_id_chunk_doc_c": hits[0].get("id") if hits else None,
                 "push_9_props_sur_doc_c_2_props": cap.status,
                 "total_fusionne_compte_dans_plafond": (cap.status == 422) if merged else None}
    constats = {"push_64_statut": push.status, "push_64_corps_type": type(body).__name__,
                "push_64_corps_cles": sorted(body) if isinstance(body, dict) else None,
                "ids_chunks_renvoyes": ids_returned, "chunks_apres_push": count, "numerotation": numbering,
                "limites": limits_views, "fusion": merge}
    resume = "push 64 → %s (ids renvoyés : %s), chunks %s, atomicité %s" % (
        push.status, ids_returned, count, (limits_views.get("atomicite") or {}).get("atomique"))
    return _result(a, resume, constats)


def probe_p17(ctx: ProbeContext) -> Dict[str, Any]:
    """P17 : filtres de métadonnées, query nulle, hybrid et seuil, rff_k/rrf_k, méthode exact."""
    a: List[Dict[str, Any]] = []
    cid = _collection_id(ctx)
    hexes = ctx.state.get("hexes")
    if cid is None or not hexes:
        return _result(a, "données de P16 absentes", {"collection": cid}, inconclusive=True)

    def eq(key: str, value: Any) -> Dict[str, Any]:
        """Filtre de comparaison eq."""
        return {"key": key, "type": "eq", "value": value}

    base = {"query": "txt", "collection_ids": [cid], "method": "semantic", "limit": 10}
    deadline = ctx.clock() + POLL_MAX_S
    ex = None
    while True:
        ex = _search(ctx, "P17", "filter_hash_eq", dict(base, metadata_filters=eq("content_hash", hexes[7])),
                     record=False)
        if _hits(ex) or ctx.clock() >= deadline or ex.status not in (200, None):
            break
        ctx.sleep(1.0)
    ctx.record_fixture(ex)
    hits = _hits(ex)
    first_meta = (hits[0].get("chunk") or {}).get("metadata") if hits else {}
    _check(a, "(a) filtre content_hash : exactement 1 résultat", len(hits) == 1)
    _check(a, "(a) content_hash du résultat == filtre",
           isinstance(first_meta, dict) and first_meta.get("content_hash") == hexes[7])
    ex_b = _search(ctx, "P17", "filter_int_eq", dict(base, limit=100, metadata_filters=eq("year", 2026)))
    or4 = {"operator": "or", "filters": [eq("content_hash", h) for h in hexes[:4]]}
    or5 = {"operator": "or", "filters": [eq("content_hash", h) for h in hexes[:5]]}
    ex_c4 = _search(ctx, "P17", "filter_or_4", dict(base, metadata_filters=or4))
    ex_c5 = _search(ctx, "P17", "filter_or_5", dict(base, metadata_filters=or5), retries=0)
    _check(a, "(c) or à 4 filtres : 4 résultats", len(_hits(ex_c4)) == 4)
    _check(a, "(c) or à 5 filtres : 422", ex_c5.status == 422)
    ex_d = _search(ctx, "P17", "query_null", dict(base, query=None, metadata_filters=eq("content_hash", hexes[3])),
                   retries=0)
    ex_e = _search(ctx, "P17", "hybrid_score_threshold", dict(base, method="hybrid", score_threshold=0.5), retries=0)
    _check(a, "(e) hybrid + score_threshold : 422", ex_e.status == 422)
    ex_f1 = _search(ctx, "P17", "hybrid_rff_k", dict(base, method="hybrid", rff_k=20), retries=0)
    ex_f2 = _search(ctx, "P17", "hybrid_rrf_k", dict(base, method="hybrid", rrf_k=20), retries=0)
    _check(a, "(f) rff_k avec hybrid : 200", ex_f1.status == 200)
    ex_g = _search(ctx, "P17", "method_exact", dict(base, method="exact"), retries=0)
    _check(a, "(g) method exact : 422", ex_g.status == 422)
    constats = {
        "a_filtre_hash": {"statut": ex.status, "n": len(hits)},
        "b_filtre_int": {"statut": ex_b.status, "n": len(_hits(ex_b))},
        "c_or_4": {"statut": ex_c4.status, "n": len(_hits(ex_c4))},
        "c_or_5": {"statut": ex_c5.status},
        "d_query_null": {"statut": ex_d.status, "n": len(_hits(ex_d)), "extrait": (ex_d.text or "")[:300]
                         if not ex_d.ok() else None},
        "e_hybrid_seuil": {"statut": ex_e.status},
        "f_rff_k": {"statut": ex_f1.status},
        "f_rrf_k": {"statut": ex_f2.status},
        "g_exact": {"statut": ex_g.status},
        "query_null_ok": ex_d.status == 200,
        "rrf_k_comportement": "422" if ex_f2.status == 422 else ("ignoré" if ex_f2.status == 200 else None),
        "usage_search": _usage_presence(ex),
    }
    resume = "hash eq %d résultat(s), or4 %d, query nulle %s, rrf_k %s" % (
        len(hits), len(_hits(ex_c4)), ex_d.status, constats["rrf_k_comportement"])
    return _result(a, resume, constats)


def probe_p18(ctx: ProbeContext) -> Dict[str, Any]:
    """P18 : latence entre l'envoi d'un chunk à jeton unique et sa première apparition en recherche."""
    a: List[Dict[str, Any]] = []
    cid = _collection_id(ctx)
    if cid is None:
        return _no_collection("P18")
    _, doc = _create_document(ctx, "P18", "setup_document", "probe-doc-index", cid)
    if doc is None:
        return _result(a, "document de sonde absent", {}, inconclusive=True)
    token = "ZXQ%d" % int(ctx.wall())
    push = _push(ctx, "P18", "push_token", doc,
                 [{"content": "Jeton unique %s pour la mesure d'indexation." % token, "metadata": {"sonde": "p18"}}])
    start = ctx.clock()
    latency = None
    polls = 0
    ex = None
    while push.ok():
        ex = _search(ctx, "P18", "search_token", {"query": token, "collection_ids": [cid], "method": "lexical",
                                                 "limit": 1}, record=False)
        polls += 1
        if _hits(ex):
            latency = round(ctx.clock() - start, 2)
            break
        if ctx.clock() - start >= POLL_MAX_S or ex.status not in (200, None):
            break
        ctx.sleep(1.0)
    if ex is not None:
        ctx.record_fixture(ex)
    _check(a, "chunk retrouvé en moins de 60 s", latency is not None)
    constats = {"statut_push": push.status, "latence_indexation_s": latency, "interrogations": polls}
    resume = "latence d'indexation : %s s (%d interrogations)" % (latency, polls)
    return _result(a, resume, constats)


def _usage_totals(ctx: ProbeContext, slug: str, endpoint: str) -> Dict[str, Any]:
    """Somme requests et prompt_tokens de /v1/usage pour un endpoint, depuis minuit UTC."""
    now = int(ctx.wall())
    day_start = now - now % 86400
    ex = ctx.call("P19", slug, "GET", "/usage", kind="default",
                  params={"start_time": day_start, "end_time": now + 60, "endpoint": endpoint, "limit": 100})
    data = ex.dig("data", default=None)
    buckets = data if isinstance(data, list) else ([ex.body] if isinstance(ex.body, dict) else [])
    requests_total = sum(b.get("requests") or 0 for b in buckets if isinstance(b, dict))
    tokens_total = sum(b.get("prompt_tokens") or 0 for b in buckets if isinstance(b, dict))
    return {"statut": ex.status, "requests": requests_total, "prompt_tokens": tokens_total}


def probe_p19(ctx: ProbeContext) -> Dict[str, Any]:
    """P19 : l'envoi de 64 chunks consomme-t-il le quota bge-m3 (delta de /v1/usage) ?"""
    a: List[Dict[str, Any]] = []
    cid = _collection_id(ctx)
    if cid is None:
        return _no_collection("P19")
    before_e = _usage_totals(ctx, "usage_embeddings_before", "/v1/embeddings")
    before_s = _usage_totals(ctx, "usage_search_before", "/v1/search")
    _, doc = _create_document(ctx, "P19", "setup_document", "probe-doc-usage", cid)
    push = None
    if doc is not None:
        chunks = [{"content": "Quota %d : phrase distincte pour la mesure d'usage." % i, "metadata": {"i": i}}
                  for i in range(64)]
        push = _push(ctx, "P19", "push_64", doc, chunks)
        _search(ctx, "P19", "search_once", {"query": "quota", "collection_ids": [cid], "method": "lexical",
                                            "limit": 1})
    after_e = after_s = None
    deadline = ctx.clock() + 30.0
    while True:
        after_e = _usage_totals(ctx, "usage_embeddings_after", "/v1/embeddings")
        after_s = _usage_totals(ctx, "usage_search_after", "/v1/search")
        if after_e["requests"] != before_e["requests"] or ctx.clock() >= deadline:
            break
        ctx.sleep(5.0)

    def delta(after: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, Any]:
        """Écart entre deux relevés d'usage."""
        return {"requests": after["requests"] - before["requests"],
                "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"]}

    usage_ok = before_e["statut"] == 200 and after_e["statut"] == 200
    _check(a, "/v1/usage répond 200", usage_ok)
    constats = {"statut_push": push.status if push else None, "usage_ok": usage_ok,
                "delta_embeddings": delta(after_e, before_e), "delta_search": delta(after_s, before_s),
                "releves": {"avant": [before_e, before_s], "apres": [after_e, after_s]}}
    resume = "delta embeddings : %s requêtes, %s tokens" % (constats["delta_embeddings"]["requests"],
                                                            constats["delta_embeddings"]["prompt_tokens"])
    return _result(a, resume, constats, inconclusive=not usage_ok)


SIMPLE_PROBES: Dict[str, Callable[[ProbeContext], Dict[str, Any]]] = {
    "P1": probe_p1, "P2": probe_p2, "P3": probe_p3, "P4": probe_p4, "P5": probe_p5, "P6": probe_p6,
    "P7": probe_p7, "P8": probe_p8, "P9": probe_p9, "P10": probe_p10, "P11": probe_p11, "P12": probe_p12,
    "P13": probe_p13, "P20": probe_p20, "P21": probe_p21,
}
COLLECTION_PROBES: Dict[str, Callable[[ProbeContext], Dict[str, Any]]] = {
    "P14": probe_p14, "P15": probe_p15, "P16": probe_p16, "P17": probe_p17, "P18": probe_p18, "P19": probe_p19,
}


MAIN_DELETE_LABEL = "suppression de la collection principale : 204"
MAIN_GET_LABEL = "GET après suppression de la collection principale : 404"
SWEEP_MAX_PAGES = 50


def _set_assertion(assertions: List[Dict[str, Any]], label: str, ok: Any) -> None:
    """Ajoute une assertion, ou met à jour celle qui porte déjà ce libellé."""
    for item in assertions:
        if item.get("attendu") == label:
            item["ok"] = bool(ok)
            return
    assertions.append({"attendu": label, "ok": bool(ok)})


def _assert_main_teardown(ctx: ProbeContext, delete_status: Optional[int], get_status: Optional[int]) -> None:
    """Ajoute à P14 les assertions DELETE 204 puis GET 404 de la collection principale ; ok devient écart."""
    res = ctx.results.get("P14")
    if not isinstance(res, dict):
        return
    assertions = res.setdefault("assertions", [])
    _set_assertion(assertions, MAIN_DELETE_LABEL, delete_status == 204)
    _set_assertion(assertions, MAIN_GET_LABEL, get_status == 404)
    if res.get("statut") == "ok" and not all(item.get("ok") for item in assertions):
        res["statut"] = "écart"


def _record_suppression(ctx: ProbeContext, entry: Dict[str, Any]) -> None:
    """Mémorise le résultat d'une suppression de collection (rapport et constats de P14)."""
    ctx.state.setdefault("suppressions", []).append(entry)


def _publish_suppressions(ctx: ProbeContext) -> None:
    """Recopie suppressions et balayage dans les constats de P14 (si P14 a été exécutée)."""
    if "P14" not in ctx.results:
        return
    constats = ctx.results["P14"].setdefault("constats", {})
    if ctx.state.get("suppressions"):
        constats["suppressions"] = ctx.sanitize(ctx.state["suppressions"])
    if ctx.state.get("balayage"):
        constats["balayage"] = ctx.sanitize(ctx.state["balayage"])


def cleanup_created_collections(ctx: ProbeContext) -> None:
    """Supprime toutes les collections créées pendant le run (idempotent, jamais d'exception).

    Pour la collection principale, ajoute à P14 (si exécutée) les assertions
    DELETE 204 puis GET 404, et passe P14 de « ok » à « écart » si elles
    échouent.
    """
    main_id = _collection_id(ctx)
    for cid in list(ctx.created_collections):
        is_main = cid == main_id
        slug = "delete_main" if is_main else "delete_extra_%d" % cid
        try:
            d = ctx.call("P14", slug, "DELETE", "/collections/%d" % cid, kind="collections", retries=2,
                         expect_auth_error=True)
            g = ctx.call("P14", "get_deleted_main" if is_main else "get_deleted_extra_%d" % cid, "GET",
                         "/collections/%d" % cid, kind="collections", expect_auth_error=True)
            _record_suppression(ctx, {"collection": cid, "delete": d.status, "get_apres": g.status})
            if d.status in (200, 204, 404):
                ctx.created_collections.remove(cid)
            if is_main:
                _assert_main_teardown(ctx, d.status, g.status)
        except Exception as exc:  # le nettoyage ne doit jamais masquer l'erreur d'origine
            _record_suppression(ctx, {"collection": cid, "erreur": type(exc).__name__})
            if is_main:
                _assert_main_teardown(ctx, None, None)
    _publish_suppressions(ctx)


def _list_items(ex: Exchange) -> List[Dict[str, Any]]:
    """Éléments d'une réponse de liste (``{"data": [...]}`` ou liste nue)."""
    body = ex.body
    items = body.get("data") if isinstance(body, dict) else body
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _find_by_exact_name(ctx: ProbeContext, name: str, skip: set) -> Tuple[List[int], Optional[str]]:
    """Ids des collections privées nommées exactement ``name`` (paginé) ; renvoie (ids, erreur)."""
    found: List[int] = []
    seen: set = set()
    offset = 0
    use_visibility = True
    for _ in range(SWEEP_MAX_PAGES):
        params: Dict[str, Any] = {"name": name, "limit": 100, "offset": offset}
        if use_visibility:
            params["visibility"] = "private"
        ex = ctx.call("P14", "sweep_list", "GET", "/collections", params=params, kind="collections", record=False,
                      expect_auth_error=True)
        if ex.status == 422 and use_visibility:
            use_visibility = False
            continue
        if ex.status != 200:
            return found, "liste impossible (%s)" % (ex.error or "HTTP %s" % ex.status)
        items = _list_items(ex)
        new_ids = {str(item.get("id")) for item in items} - seen
        if items and not new_ids:
            break
        seen |= new_ids
        for item in items:
            cid = _as_int_id(item.get("id"))
            if item.get("name") == name and item.get("visibility") in (None, "private") and cid is not None \
                    and cid not in skip and cid not in found:
                found.append(cid)
        if len(items) < 100:
            break
        offset += 100
    return found, None


def sweep_collections_by_name(ctx: ProbeContext, name: str) -> None:
    """Supprime toute collection privée nommée exactement ``name`` qui a échappé au suivi par id.

    Couvre une création appliquée côté serveur dont la réponse s'est perdue
    (délai, coupure) ou dont l'id est inexploitable, pour la collection
    principale comme pour le doublon de P14 (même nom). Jamais d'exception.
    """
    deleted = {s.get("collection") for s in ctx.state.get("suppressions") or []
               if s.get("delete") in (200, 204, 404)}
    sweep: Dict[str, Any] = {"nom": name, "supprimees": [], "erreur": None}
    ctx.state["balayage"] = sweep
    try:
        found, error = _find_by_exact_name(ctx, name, deleted)
        sweep["erreur"] = error
        for cid in found:
            try:
                d = ctx.call("P14", "sweep_delete", "DELETE", "/collections/%d" % cid, kind="collections",
                             record=False, expect_auth_error=True)
                _record_suppression(ctx, {"collection": cid, "delete": d.status, "balayage": True})
                if d.status in (200, 204, 404):
                    sweep["supprimees"].append(cid)
                    if cid in ctx.created_collections:
                        ctx.created_collections.remove(cid)
            except Exception as exc:  # comme le nettoyage : on continue avec les suivantes
                _record_suppression(ctx, {"collection": cid, "erreur": type(exc).__name__, "balayage": True})
    except Exception as exc:  # le balayage ne doit jamais masquer l'erreur d'origine
        sweep["erreur"] = type(exc).__name__
    _publish_suppressions(ctx)


def run_collection_group(ctx: ProbeContext, selected: Sequence[str]) -> None:
    """Crée la collection privée ``ragpy-probe-<ts>``, lance P14 à P19 choisies, supprime tout en finally.

    Le ``finally`` supprime les ids suivis, puis balaie par nom exact pour
    rattraper une création dont la réponse a été perdue.
    """
    name = "%s%d" % (PROBE_PREFIX, int(ctx.wall()))
    coll: Dict[str, Any] = {"name": name, "id": None, "create_status": None, "create_attempted": False}
    ctx.state["collection"] = coll
    try:
        coll["create_attempted"] = True
        ex = ctx.call("P14", "create", "POST", "/collections", kind="collections",
                      json_body={"name": name, "description": "probe", "visibility": "private"})
        coll["create_status"] = ex.status
        raw_id = ex.dig("id")
        coll["id_type"] = type(raw_id).__name__
        cid = _as_int_id(raw_id)
        if ex.ok() and cid is not None:
            coll["id"] = cid
            ctx.created_collections.append(cid)
        for pid in selected:
            ctx.run_probe(pid, COLLECTION_PROBES[pid])
    finally:
        cleanup_created_collections(ctx)
        if coll["create_attempted"]:
            sweep_collections_by_name(ctx, name)


# ---------------------------------------------------------------------------
# Nettoyage (--cleanup)
# ---------------------------------------------------------------------------


def list_probe_collections(client: httpx.Client) -> List[Dict[str, Any]]:
    """Liste les collections privées dont le nom commence par ``ragpy-probe-`` (paginé)."""
    found: List[Dict[str, Any]] = []
    seen_ids: set = set()
    offset = 0
    use_visibility = True
    for _ in range(200):
        params: Dict[str, Any] = {"limit": 100, "offset": offset}
        if use_visibility:
            params["visibility"] = "private"
        resp = client.get("/collections", params=params, timeout=TIMEOUTS["collections"])
        message = account_error_message(resp.status_code, resp.text)
        if message:
            raise AccountLevelError(message)
        if resp.status_code == 422 and use_visibility:
            use_visibility = False
            continue
        if resp.status_code != 200:
            raise RuntimeError("liste des collections impossible (HTTP %d)" % resp.status_code)
        body = resp.json()
        items = body.get("data", []) if isinstance(body, dict) else (body if isinstance(body, list) else [])
        new_ids = {item.get("id") for item in items if isinstance(item, dict)} - seen_ids
        if items and not new_ids:
            break
        seen_ids |= new_ids
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name.startswith(PROBE_PREFIX) and item.get("visibility") in (None, "private"):
                found.append({"id": item.get("id"), "name": name})
        if len(items) < 100:
            break
        offset += 100
    return found


def run_cleanup(client: httpx.Client, dry_run: bool, log: Callable[[str], None] = print) -> int:
    """Supprime (ou liste en --dry-run) les collections de sonde ; renvoie le nombre restant."""
    found = list_probe_collections(client)
    for item in found:
        log("%s collection %s (%s)" % ("trouvée" if dry_run else "suppression de la", item["id"], item["name"]))
    if dry_run:
        remaining = len(found)
    else:
        for item in found:
            if isinstance(item["id"], int):
                resp = client.delete("/collections/%d" % item["id"], timeout=TIMEOUTS["collections"])
                message = account_error_message(resp.status_code, resp.text)
                if message:
                    raise AccountLevelError(message)
                log("  → HTTP %d" % resp.status_code)
        remaining = len(list_probe_collections(client))
    log("ragpy-probe restantes: %d" % remaining)
    return remaining


# ---------------------------------------------------------------------------
# Décisions D1 à D22
# ---------------------------------------------------------------------------

DECISION_TABLE: List[Tuple[str, Optional[str], str]] = [
    ("D1", "P1", "L1 config, preflight, message 403 des collections"),
    ("D2", "P2", "L1 catalog"),
    ("D3", "P3", "défaut ALBERT_OCR_MODE (L4), saut du test live"),
    ("D4", "P4", "ALBERT_OCR_PART_MB, ALBERT_OCR_PART_PAGES, ALBERT_TIMEOUT_OCR_DOC"),
    ("D5", "P5", "ALBERT_OCR_CONCURRENCY, ALBERT_TIMEOUT_OCR_PAGE"),
    ("D6", "P6", "errors.py"),
    ("D7", "P7", "champs du ledger"),
    ("D8", "P8", "client.chat"),
    ("D9", "P9", "marge de raisonnement, raisonnement jamais pris comme contenu"),
    ("D10", "P10", "filtre de citations, phase 1 des fiches"),
    ("D11", "P11", "mode harden (L3) : seed envoyé seulement si déterministe"),
    ("D12", "P12", "ALBERT_EMBED_L2_NORMALIZE, norm dans les paramètres d'espace et la clé de cache"),
    ("D13", "P13", "retry.py, quota épuisé"),
    ("D14", "P14", "L5 : correspondance exacte ; plusieurs correspondances = erreur listant les ids"),
    ("D15", "P15", "create_document"),
    ("D16", "P16", "contenu du manifeste et rollback (suppression du document dans tous les cas)"),
    ("D17", "P17", "client.search"),
    ("D18", "P18", "attentes des tests live, note de cohérence du preload"),
    ("D19", "P19", "ALBERT_PUSH_CONCURRENCY, rôle embed du limiteur pendant l'envoi"),
    ("D20", "P20", "informative : le preflight utilise /v1/models, le repli 503 reste sur les statuts HTTP"),
    ("D21", None, "ALBERT_OCR_SKIP_RECODE (défaut 0), confirmé au L8 avec l'utilisateur"),
    ("D22", "P21", "ALBERT_RECODE_CONCURRENCY, ALBERT_NOTES_CONCURRENCY, ALBERT_EMBED_CONCURRENCY"),
]

DECISION_OBJECTS = {
    "D1": "RPM/RPD par modèle, budget, expiration, permissions", "D2": "ids, alias, types, contexte",
    "D3": "accès à /v1/ocr, pages à partir de 0", "D4": "taille des parts et latence",
    "D5": "LightOnOCR accepté en chat, latence", "D6": "forme des erreurs", "D7": "présence de cost et impacts",
    "D8": "max_tokens ou max_completion_tokens", "D9": "champ de raisonnement, budget", "D10": "modes JSON",
    "D11": "déterminisme de seed", "D12": "plafond de lot, dimension, norme",
    "D13": "Retry-After, corps distinguant RPM, RPD et TPM", "D14": "filtre de nom, noms en double",
    "D15": "multipart ou urlencoded", "D16": "limites d'envoi, fusion des métadonnées, ids, atomicité",
    "D17": "query=null, rff_k, seuil", "D18": "latence d'indexation", "D19": "quota bge-m3 consommé par les envois",
    "D20": "/health et /health/models", "D21": "recodage des textes LightOnOCR",
    "D22": "débit p50/p95, ceil(0.9 × RPM × p50 / 60)",
}

DECISION_DEFAULTS: Dict[str, Any] = {
    "D1": {"ALBERT_RECODE_RPM": 60, "ALBERT_NOTES_RPM": 30, "ALBERT_OCR_RPM": 30, "ALBERT_EMBED_RPM": 600,
           "ALBERT_CHAT_TPM": 200000},
    "D2": {"models": {
        NOTES_MODEL: {"type": "text-generation", "max_context_length": 131072,
                      "aliases": ["openai/gpt-oss-120b", "openweight-large"]},
        RECODE_MODEL: {"type": "image-text-to-text", "max_context_length": 262144,
                       "aliases": ["mistralai/Ministral-3-8B-Instruct-2512", "openweight-small"]},
        OCR_CHAT_MODEL: {"type": "image-text-to-text", "max_context_length": 16384,
                         "aliases": ["lighton/LightOn-OCR-2-1B", "openweight-ocr"]},
        EMBED_MODEL: {"type": "text-embeddings-inference", "max_context_length": 8192,
                      "aliases": ["BAAI/bge-m3", "openweight-embeddings"]},
        OCR_DOC_MODEL: {"type": "image-to-text", "max_context_length": 16384, "aliases": []},
    }, "mistral_ocr_2512_listed": False},
    "D3": {"ALBERT_OCR_MODE": "auto", "ocr_access": False, "pages_zero_based": True},
    "D4": {"ALBERT_OCR_PART_MB": 15, "ALBERT_OCR_PART_PAGES": 100, "ALBERT_TIMEOUT_OCR_DOC": 300},
    "D5": {"ALBERT_OCR_CONCURRENCY": 2, "ALBERT_TIMEOUT_OCR_PAGE": 120},
    "D6": {"classify_on_status": True, "status_wrong_model_type": 422, "status_unknown_model": 404,
           "status_invalid_key": 401, "detail_422": "list"},
    "D7": {"ledger_cost": False, "ledger_impacts": False},
    "D8": {"chat_max_param": "max_tokens"},
    "D9": {"reasoning_field": None, "ALBERT_REASONING_HEADROOM": 2048, "ALBERT_REASONING_EFFORT": "medium"},
    "D10": {"citation_filter_response_format": None, "book_structure_response_format": None},
    "D11": {"send_seed": False},
    "D12": {"ALBERT_EMBED_BATCH": 64, "embedding_dim": 1024, "ALBERT_EMBED_L2_NORMALIZE": 1},
    "D13": {"retry_after_format": "both", "ALBERT_RETRY_AFTER_MAX": 120},
    "D14": {"name_filter": "substring", "duplicates_allowed": True,
            "policy": "correspondance exacte côté client ; plusieurs correspondances = erreur listant les ids"},
    "D15": {"create_document_encoding": "multipart"},
    "D16": {"chunk_ids_returned": False, "ALBERT_CHUNKS_PER_POST": 64, "ALBERT_METADATA_MAX_PROPS": 10,
            "atomic_post": None, "rollback": "delete_document"},
    "D17": {"query_required": True, "send_rff_k": True, "score_threshold_semantic_only": True, "or_filter_max": 4},
    "D18": {"live_test_index_wait_s": 60},
    "D19": {"ALBERT_PUSH_CONCURRENCY": 1, "push_uses_embed_limiter": True},
    "D20": {"preflight": "/v1/models", "fallback_503_on_status": True},
    "D21": {"ALBERT_OCR_SKIP_RECODE": 0},
    "D22": {"ALBERT_RECODE_CONCURRENCY": 4, "ALBERT_NOTES_CONCURRENCY": 4, "ALBERT_EMBED_CONCURRENCY": 4},
}


def _c(results: Dict[str, Any], pid: str) -> Dict[str, Any]:
    """Constats d'une sonde (dict vide si absente)."""
    return (results.get(pid) or {}).get("constats") or {}


def _rpm(results: Dict[str, Any], model: str) -> Optional[int]:
    """RPM d'un modèle lu par P1 (None si inconnu ou illimité)."""
    quotas = (_c(results, "P1").get("quotas_par_modele") or {}).get(model) or {}
    value = quotas.get("rpm")
    return value if isinstance(value, int) and value > 0 else None


def _d1(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D1 : plafonds RPM (90 % du quota) et faits du compte."""
    c = _c(r, "P1")
    if not c.get("me_ok"):
        return None
    d = DECISION_DEFAULTS["D1"]
    tpm = ((c.get("quotas_par_modele") or {}).get(RECODE_MODEL) or {}).get("tpm")

    def cap(model: str, fallback: int) -> int:
        """90 % du RPM mesuré, sinon le repli."""
        rpm = _rpm(r, model)
        return max(1, int(0.9 * rpm)) if rpm else fallback

    return {"ALBERT_RECODE_RPM": cap(RECODE_MODEL, d["ALBERT_RECODE_RPM"]),
            "ALBERT_NOTES_RPM": cap(NOTES_MODEL, d["ALBERT_NOTES_RPM"]),
            "ALBERT_OCR_RPM": cap(OCR_CHAT_MODEL, d["ALBERT_OCR_RPM"]),
            "ALBERT_EMBED_RPM": cap(EMBED_MODEL, d["ALBERT_EMBED_RPM"]),
            "ALBERT_CHAT_TPM": int(0.9 * tpm) if isinstance(tpm, int) and tpm > 0 else d["ALBERT_CHAT_TPM"],
            "rpm_origine": "probe" if c.get("correspondance_router_possible") else "repli",
            "budget": c.get("budget"), "expires": c.get("expires"), "expires_ok": c.get("expires_ok"),
            "permissions": c.get("permissions"), "limits": c.get("limits")}


def _d2(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D2 : catalogue mesuré."""
    c = _c(r, "P2")
    if not c.get("n_modeles"):
        return None
    models = {mid: {"type": v.get("type"), "max_context_length": v.get("max_context_length"),
                    "aliases": v.get("aliases"), "present": v.get("present")}
              for mid, v in (c.get("modeles_attendus") or {}).items()}
    return {"models": models, "aliases_missing": c.get("alias_manquants"),
            "mistral_ocr_2512_listed": c.get("mistral_ocr_2512_liste")}


def _d3(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D3 : accès /v1/ocr."""
    c = _c(r, "P3")
    if c.get("acces") is None:
        return None
    return {"ALBERT_OCR_MODE": "auto", "ocr_access": c["acces"], "pages_zero_based": c.get("pages_zero_based"),
            "skip_live_ocr_doc_test": not c["acces"], "no_access_signature": c.get("signature_sans_acces")}


def _d4(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D4 : taille des parts et délai de /v1/ocr."""
    c = _c(r, "P4")
    largest = c.get("plus_grand_ok_mo")
    if not largest:
        return None
    durations = [e.get("duree_s") for e in c.get("essais") or [] if isinstance(e.get("duree_s"), (int, float))]
    cent = c.get("cent_pages") or {}
    if isinstance(cent.get("duree_s"), (int, float)):
        durations.append(cent["duree_s"])
    return {"ALBERT_OCR_PART_MB": max(1, int(0.6 * largest)),
            "ALBERT_OCR_PART_PAGES": 100 if cent.get("statut") == 200 else 50,
            "ALBERT_TIMEOUT_OCR_DOC": max(300, int(math.ceil(2 * max(durations)))) if durations else 300,
            "largest_ok_mb": largest, "first_failure": c.get("premier_echec")}


def _d5(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D5 : concurrence et délai de l'OCR par chat."""
    c = _c(r, "P5")
    lat = c.get("latence") or {}
    p50, p95 = lat.get("p50_s"), lat.get("p95_s")
    if not c.get("chat_accepte") or p50 is None:
        return None
    rpm = _rpm(r, OCR_CHAT_MODEL) or DECISION_DEFAULTS["D1"]["ALBERT_OCR_RPM"]
    conc = min(8, max(1, math.ceil(0.9 * rpm * p50 / 60.0)))
    return {"ALBERT_OCR_CONCURRENCY": conc, "ALBERT_TIMEOUT_OCR_PAGE": max(120, int(math.ceil(3 * (p95 or p50)))),
            "p50_s": p50, "p95_s": p95, "rpm_utilise": rpm, "echantillon": lat.get("echantillon")}


def _d6(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D6 : forme des erreurs."""
    c = _c(r, "P6")
    if not c:
        return None
    return {"classify_on_status": True,
            "status_wrong_model_type": (c.get("wrong_model_type") or {}).get("statut"),
            "status_unknown_model": (c.get("unknown_model") or {}).get("statut"),
            "status_invalid_key": (c.get("invalid_key") or {}).get("statut"),
            "detail_422": (c.get("wrong_model_type") or {}).get("type_detail"),
            "detail_404": (c.get("unknown_model") or {}).get("type_detail"),
            "detail_401": (c.get("invalid_key") or {}).get("type_detail"),
            "wrong_model_type_in_body": (c.get("wrong_model_type") or {}).get("contient_wrong_model_type")}


def _d7(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D7 : cost et impacts dans usage."""
    views = {k: v for k, v in (_c(r, "P7").get("par_endpoint") or {}).items() if v.get("statut") == 200}
    if not views:
        return None
    return {"ledger_cost": all(v.get("cost") for v in views.values()),
            "ledger_impacts": all(v.get("impacts") for v in views.values()),
            "par_endpoint": {k: {"cost": v.get("cost"), "impacts": v.get("impacts")} for k, v in views.items()}}


def _d8(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D8 : paramètre de borne de génération."""
    c = _c(r, "P8")
    if c.get("max_tokens_respecte"):
        param = "max_tokens"
    elif c.get("max_completion_tokens_respecte"):
        param = "max_completion_tokens"
    else:
        return None
    return {"chat_max_param": param, "both_winner": c.get("gagnant_les_deux")}


def _d9(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D9 : champ de raisonnement et marge."""
    c = _c(r, "P9")
    if not c:
        return None
    return {"reasoning_field": c.get("champ_raisonnement"), "ALBERT_REASONING_HEADROOM": 2048,
            "ALBERT_REASONING_EFFORT": "medium", "trap_confirmed": c.get("piege_confirme"),
            "completion_includes_reasoning": c.get("completion_inclut_raisonnement"),
            "reasoning_effort_effect": c.get("effet_reasoning_effort")}


def _d10(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D10 : format JSON à demander au modèle."""
    c = _c(r, "P10")
    if not c:
        return None
    fmt = "json_schema" if c.get("json_schema_ministral") else ("json_object" if c.get("json_object_ministral")
                                                               else None)
    return {"citation_filter_response_format": fmt, "book_structure_response_format": fmt,
            "gpt_oss_json_schema": c.get("json_schema_gpt_oss")}


def _d11(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D11 : envoi de seed."""
    c = _c(r, "P11")
    if not c or c.get("statuts") != [200, 200]:
        return None
    return {"send_seed": bool(c.get("identiques") and c.get("modele_stable")),
            "fingerprint_stable": c.get("fingerprint_stable")}


def _d12(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D12 : lot, dimension, norme."""
    c = _c(r, "P12")
    if not c.get("batch_64_ok"):
        return None
    dims = c.get("dimensions") or []
    nmin, nmax = c.get("norme_min"), c.get("norme_max")
    normalized = isinstance(nmin, (int, float)) and isinstance(nmax, (int, float)) and \
        abs(nmin - 1) < 1e-3 and abs(nmax - 1) < 1e-3
    return {"ALBERT_EMBED_BATCH": 64, "embedding_dim": dims[0] if len(dims) == 1 else None,
            "ALBERT_EMBED_L2_NORMALIZE": 1, "already_normalized": normalized, "norm_range": [nmin, nmax],
            "status_65": c.get("statut_65"), "status_empty_string": c.get("statut_vide"),
            "alias_model": c.get("alias_modele_servi")}


def _d13(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D13 : format de Retry-After et en-têtes de limitation."""
    c = _c(r, "P13")
    if not c:
        return None
    formats = c.get("formats_retry_after") or []
    return {"retry_after_format": formats[0] if len(formats) == 1 else "both", "ALBERT_RETRY_AFTER_MAX": 120,
            "ratelimit_headers": c.get("entetes_ratelimit_vus"), "natural_429_seen": c.get("compte_429", 0) > 0,
            "first_429": c.get("premier_429")}


def _d14(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D14 : filtre de nom et doublons."""
    c = _c(r, "P14")
    if c.get("doublon_autorise") is None and c.get("filtre_nom") is None:
        return None
    return {"name_filter": c.get("filtre_nom") or "substring",
            "duplicates_allowed": c.get("doublon_autorise") if c.get("doublon_autorise") is not None else True,
            "policy": DECISION_DEFAULTS["D14"]["policy"], "limit_101_status": c.get("statut_limit_101")}


def _d15(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D15 : encodage de création de document."""
    c = _c(r, "P15")
    if not c:
        return None
    enc = "multipart" if c.get("multipart_sans_fichier") or not c.get("urlencoded_accepte") else "urlencoded"
    return {"create_document_encoding": enc, "multipart_no_file": c.get("multipart_sans_fichier"),
            "urlencoded_accepted": c.get("urlencoded_accepte"),
            "document_metadata_accepted": c.get("metadata_document_acceptee")}


def _d16(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D16 : limites d'envoi, ids et atomicité."""
    c = _c(r, "P16")
    if c.get("push_64_statut") is None:
        return None
    lim = c.get("limites") or {}

    def st(slug: str) -> Optional[int]:
        """Statut d'une tentative de limite."""
        return (lim.get(slug) or {}).get("statut")

    fusion = c.get("fusion") or {}
    return {"chunk_ids_returned": c.get("ids_chunks_renvoyes"),
            "ALBERT_CHUNKS_PER_POST": 64 if 200 <= (c.get("push_64_statut") or 0) < 300 else None,
            "ALBERT_METADATA_MAX_PROPS": 10, "status_65": st("push_65"), "status_11_props": st("meta_11_props"),
            "status_str_256": st("meta_str_256"), "status_empty_string": st("meta_empty_string"),
            "status_none": st("meta_none"), "status_list": st("meta_list"), "status_key_256": st("meta_key_256"),
            "atomic_post": (lim.get("atomicite") or {}).get("atomique"),
            "doc_metadata_merged": fusion.get("fusion"),
            "merged_counts_toward_cap": fusion.get("total_fusionne_compte_dans_plafond"),
            "chunk_id_numbering": c.get("numerotation"), "rollback": "delete_document"}


def _d17(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D17 : paramètres de recherche."""
    c = _c(r, "P17")
    if not c:
        return None
    return {"query_required": not c.get("query_null_ok"), "send_rff_k": True,
            "rrf_k_behaviour": c.get("rrf_k_comportement"),
            "score_threshold_semantic_only": (c.get("e_hybrid_seuil") or {}).get("statut") == 422,
            "or_filter_max": 4 if (c.get("c_or_5") or {}).get("statut") == 422 else None,
            "method_exact_rejected": (c.get("g_exact") or {}).get("statut") == 422}


def _d18(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D18 : attente d'indexation pour les tests live."""
    latency = _c(r, "P18").get("latence_indexation_s")
    if not isinstance(latency, (int, float)):
        return None
    return {"index_latency_s": latency, "live_test_index_wait_s": max(10, int(math.ceil(3 * latency)))}


def _d19(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D19 : consommation du quota bge-m3 par les envois."""
    c = _c(r, "P19")
    if not c.get("usage_ok"):
        return None
    delta = (c.get("delta_embeddings") or {}).get("requests") or 0
    return {"push_counts_embed_quota": delta > 0, "embed_requests_per_push_64": delta,
            "embed_prompt_tokens_per_push_64": (c.get("delta_embeddings") or {}).get("prompt_tokens"),
            "search_delta": c.get("delta_search"), "ALBERT_PUSH_CONCURRENCY": 1, "push_uses_embed_limiter": delta > 0}


def _d20(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D20 : santé (informatif)."""
    c = _c(r, "P20")
    if not c:
        return None
    return {"preflight": "/v1/models", "fallback_503_on_status": True, "health_status": c.get("health"),
            "health_models_status": c.get("health_models"), "v1_health_status": c.get("v1_health"),
            "health_models_shape": c.get("forme_health_models")}


def _d22(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """D22 : concurrences par ceil(0.9 × RPM × p50 / 60)."""
    series = _c(r, "P21").get("series") or {}
    c1 = series.get("c1") or {}
    p50_recode = (c1.get("recode") or {}).get("p50_s")
    p50_notes = (c1.get("notes") or {}).get("p50_s")
    if p50_recode is None and p50_notes is None:
        return None
    d = DECISION_DEFAULTS["D22"]
    rpm_recode = _rpm(r, RECODE_MODEL) or DOCUMENTED_RPM[RECODE_MODEL]
    rpm_notes = _rpm(r, NOTES_MODEL) or DOCUMENTED_RPM[NOTES_MODEL]

    def conc(rpm: int, p50: Optional[float], fallback: int, cap: int) -> int:
        """Formule de concurrence bornée."""
        if p50 is None:
            return fallback
        return min(cap, max(1, math.ceil(0.9 * rpm * p50 / 60.0)))

    return {"ALBERT_RECODE_CONCURRENCY": conc(rpm_recode, p50_recode, d["ALBERT_RECODE_CONCURRENCY"], 16),
            "ALBERT_NOTES_CONCURRENCY": conc(rpm_notes, p50_notes, d["ALBERT_NOTES_CONCURRENCY"], 8),
            "ALBERT_EMBED_CONCURRENCY": d["ALBERT_EMBED_CONCURRENCY"],
            "p50_recode_s": p50_recode, "p50_notes_s": p50_notes, "rpm_recode": rpm_recode, "rpm_notes": rpm_notes,
            "p95": {k: {"recode": (v.get("recode") or {}).get("p95_s"), "notes": (v.get("notes") or {}).get("p95_s")}
                    for k, v in series.items()}}


DECISION_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
    "D1": _d1, "D2": _d2, "D3": _d3, "D4": _d4, "D5": _d5, "D6": _d6, "D7": _d7, "D8": _d8, "D9": _d9,
    "D10": _d10, "D11": _d11, "D12": _d12, "D13": _d13, "D14": _d14, "D15": _d15, "D16": _d16, "D17": _d17,
    "D18": _d18, "D19": _d19, "D20": _d20, "D22": _d22,
}


def build_decisions(results: Dict[str, Any], executed: Iterable[str] = (), previous: Optional[Dict[str, Any]] = None,
                    on_error: Optional[Callable[[str, Exception], None]] = None) -> Dict[str, Any]:
    """Construit D1 à D22 ({value, default, source, consumer}) avec défaut prudent si non concluant.

    Une décision dont la sonde n'a pas été exécutée dans ce run reprend
    l'entrée mesurée d'un ``decisions.json`` précédent (``previous``).
    """
    executed = set(executed)
    out: Dict[str, Any] = {}
    for did, pid, consumer in DECISION_TABLE:
        default = copy.deepcopy(DECISION_DEFAULTS[did])
        value = None
        builder = DECISION_BUILDERS.get(did)
        statut = (results.get(pid) or {}).get("statut") if pid else None
        if builder and pid in results and statut not in ("erreur", "sauté", "interrompu"):
            try:
                value = builder(results)
            except Exception as exc:  # une décision défaillante retombe sur le défaut
                value = None
                if on_error:
                    on_error(did, exc)
        prev = (previous or {}).get(did)
        if value is not None:
            entry = {"value": value, "default": default, "source": "probe", "consumer": consumer}
        elif pid and pid not in executed and isinstance(prev, dict) and prev.get("source") == "probe":
            entry = {"value": prev.get("value"), "default": default, "source": "probe", "consumer": consumer}
        else:
            entry = {"value": copy.deepcopy(default), "default": default, "source": "default", "consumer": consumer}
        out[did] = entry
    return out


# ---------------------------------------------------------------------------
# Fixtures synthétiques, rapport, comparaison
# ---------------------------------------------------------------------------

_USAGE_DOC = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
              "impacts": {"kWh": 0.0, "kgCO2eq": 0.0}, "requests": 1}

SYNTHETIC_TEMPLATES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("P1", "me"): {
        "request": {"method": "GET", "path": "/v1/me", "body_summary": None}, "status": 200,
        "body": {"object": "userInfo", "id": MASK, "email": MASK, "name": MASK, "organization_id": None,
                 "budget": None, "permissions": [], "limits": [{"router_id": 7, "type": "rpm", "value": 50},
                                                               {"router_id": 7, "type": "tpd", "value": None}],
                 "expires": None},
        "source": "référence API Albert §12.1 (valeurs illustratives)"},
    ("P2", "models"): {
        "request": {"method": "GET", "path": "/v1/models", "body_summary": None}, "status": 200,
        "body": {"object": "list", "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "<non documenté>", "type": spec["type"],
             "aliases": spec["aliases"], "max_context_length": spec["max_context_length"],
             "costs": {"prompt_tokens": 0.0, "completion_tokens": 0.0}}
            for mid, spec in (
                (NOTES_MODEL, {"type": "text-generation", "max_context_length": 131072,
                               "aliases": ["openai/gpt-oss-120b", "openweight-large"]}),
                (RECODE_MODEL, {"type": "image-text-to-text", "max_context_length": 262144,
                                "aliases": ["mistralai/Ministral-3-8B-Instruct-2512", "openweight-small"]}),
                (OCR_CHAT_MODEL, {"type": "image-text-to-text", "max_context_length": 16384,
                                  "aliases": ["lighton/LightOn-OCR-2-1B", "openweight-ocr"]}),
                (OCR_DOC_MODEL, {"type": "image-to-text", "max_context_length": 16384,
                                 "aliases": [OCR_DOC_MODEL]}),
                (EMBED_MODEL, {"type": "text-embeddings-inference", "max_context_length": 8192,
                               "aliases": ["BAAI/bge-m3", "openweight-embeddings"]}),
            )]},
        "note": "created, owned_by et costs illustratifs (valeurs non documentées)",
        "source": "référence API Albert §4.2 (catalogue) et §4.4 (objet Model), enveloppe de liste §3.1"},
    ("P3", "ocr_page0"): {
        "request": {"method": "POST", "path": "/v1/ocr", "body_summary": {
            "model": OCR_DOC_MODEL, "document": {"type": "document_url",
                                                 "document_url": "data:application/pdf;base64,<PDF de 2 pages>"},
            "pages": [0], "include_image_base64": False}}, "status": 200,
        "body": {"id": "synthetic-ocr-0001", "model": OCR_DOC_MODEL,
                 "pages": [{"index": 0, "markdown": "PAGE_ONE_7Q", "images": [],
                            "dimensions": {"dpi": 200, "height": 2200, "width": 1700}}],
                 "usage_info": {"pages_processed": 1, "doc_size_bytes": 1500}, "usage": _USAGE_DOC},
        "source": "référence API Albert §9.2 (réponse /v1/ocr) et §3.3 (usage)"},
    ("P5", "ocr_chat_alias"): {
        "request": {"method": "POST", "path": "/v1/chat/completions", "body_summary": {
            "model": OCR_CHAT_ALIAS, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<page 1>"}}]}],
            "max_tokens": 4096, "temperature": 0.2, "top_p": 0.9}}, "status": 200,
        "body": {"id": "synthetic-chat-0001", "object": "chat.completion", "created": 0, "model": OCR_CHAT_MODEL,
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "PAGE_ONE_7Q"},
                              "finish_reason": "stop"}], "usage": _USAGE_DOC},
        "source": "référence API Albert §9.1 et §6.8"},
    ("P12", "batch_64"): {
        "request": {"method": "POST", "path": "/v1/embeddings", "body_summary": {
            "model": EMBED_MODEL, "input": ["<64 textes>"], "encoding_format": "float"}}, "status": 200,
        "body": {"object": "list", "model": EMBED_MODEL, "id": "synthetic-embed-0001",
                 "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * FIXTURE_FLOAT_KEEP}],
                 "usage": _USAGE_DOC},
        "truncated": [{"chemin": "body.data[].embedding", "longueur_originale": 1024}],
        "source": "référence API Albert §7"},
    ("P14", "create"): {
        "request": {"method": "POST", "path": "/v1/collections", "body_summary": {
            "name": PROBE_PREFIX + "0", "description": "probe", "visibility": "private"}}, "status": 201,
        "body": {"id": 1}, "source": "référence API Albert §11.2"},
    ("P15", "multipart_no_file"): {
        "request": {"method": "POST", "path": "/v1/documents", "body_summary": {
            "multipart": {"name": "probe-doc", "collection_id": "1"}}}, "status": 201,
        "body": {"id": 1}, "source": "référence API Albert §11.4"},
    ("P16", "push_64"): {
        "request": {"method": "POST", "path": "/v1/documents/1/chunks", "body_summary": {"chunks": [
            {"content": "txt 0 : chunk de sonde RAGpy numéro 0.", "metadata": {"content_hash": "<64 hex>"}},
            {"content": "txt 1 : chunk de sonde RAGpy numéro 1.", "metadata": {"content_hash": "<64 hex>"}},
            {"content": "txt 2 : chunk de sonde RAGpy numéro 2.", "metadata": {"content_hash": "<64 hex>"}},
            "… (+61 éléments, 64 au total)"]}}, "status": 201,
        "body": None,
        "forme_corps": "inconnue (D16) : ne pas en déduire d'ids de chunks",
        "source": "référence API Albert §11.4 (POST /v1/documents/{id}/chunks, 1 à 64 chunks)"},
    ("P19", "usage_embeddings_before"): {
        "request": {"method": "GET", "path": "/v1/usage?start_time=0&end_time=86460&endpoint=%2Fv1%2Fembeddings"
                                             "&limit=100", "body_summary": None}, "status": 200,
        "body": {"object": "list", "data": [
            {"object": "usage.bucket", "start_time": 0, "end_time": 86400, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "requests": 0,
             "impacts": {"kWh": 0.0, "kgCO2eq": 0.0}}]},
        "source": "référence API Albert §12.3 (usage.bucket), enveloppe de liste §3.1"},
    ("P20", "health_models"): {
        "request": {"method": "GET", "path": "/health/models", "body_summary": None}, "status": 200,
        "body": None,
        "forme_corps": "inconnue (non documentée)",
        "source": "référence API Albert §12.4 (GET /health/models, authentifié)"},
    ("P17", "filter_hash_eq"): {
        "request": {"method": "POST", "path": "/v1/search", "body_summary": {
            "query": "txt", "collection_ids": [1], "method": "semantic", "limit": 10,
            "metadata_filters": {"key": "content_hash", "type": "eq", "value": "<64 hex>"}}}, "status": 200,
        "body": {"object": "list", "data": [{"method": "semantic", "score": 0.83, "chunk": {
            "id": 7, "collection_id": 1, "document_id": 1, "content": "txt 7", "metadata": {"content_hash": "<64 hex>"},
            "created": 0}}], "usage": _USAGE_DOC},
        "source": "référence API Albert §11.5"},
}


def write_synthetic_fixtures(ctx: ProbeContext) -> List[str]:
    """Écrit une fixture synthétique pour chaque succès documenté non obtenu par une sonde exécutée."""
    written = []
    for (pid, slug), template in SYNTHETIC_TEMPLATES.items():
        if pid not in ctx.executed_ids:
            continue
        real = ctx.fixtures.get("%s_%s.json" % (pid, slug))
        if real and isinstance(real.get("status"), int) and 200 <= real["status"] < 300:
            continue
        fixture = copy.deepcopy(template)
        fixture.setdefault("headers", {"content-type": "application/json"})
        fixture["synthetic"] = True
        written.append(ctx.write_fixture("%s_%s_synthetic.json" % (pid, slug), fixture))
    return written


def prune_stale_fixtures(fixtures_dir: Path, pids: Iterable[str], keep: Iterable[str]) -> List[str]:
    """Supprime les fixtures ``<pid>_*.json`` laissées par un run précédent pour les sondes données.

    Les fichiers écrits par ce run (``keep``), ``decisions.json`` et les
    sous-dossiers (``golden_off/`` par exemple) ne sont jamais touchés ;
    ``P13_*.json`` couvre ``P13_natural_429.json``. Renvoie les noms supprimés.
    """
    fixtures_dir = Path(fixtures_dir)
    keep = set(keep)
    removed: List[str] = []
    if not fixtures_dir.is_dir():
        return removed
    for pid in sorted(set(pids) & set(PROBE_ORDER)):
        for path in sorted(fixtures_dir.glob("%s_*.json" % pid)):
            if path.is_file() and path.name not in keep:
                path.unlink()
                removed.append(path.name)
    return removed


def _rel(path: Path) -> str:
    """Chemin relatif au répertoire courant (jamais de chemin absolu dans les sorties)."""
    try:
        return os.path.relpath(str(path))
    except ValueError:
        return Path(path).name


def build_report(ctx: ProbeContext, decisions: Dict[str, Any], aborted: Optional[str],
                 synthetic: Sequence[str]) -> str:
    """Rapport Markdown en français (assaini par l'appelant)."""
    date = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ctx.wall()))
    lines = ["# Rapport des sondes de l'API Albert (lot 0)", "",
             "- Date : %s" % date, "- Base : %s" % BASE_URL,
             "- Sondes exécutées : %d ; erreurs de script : %d" % (ctx.executed, ctx.script_errors),
             "- Fixtures : `%s` ; bruts : `%s`" % (_rel(ctx.fixtures_dir), _rel(ctx.raw_dir))]
    if aborted:
        lines.append("- **Arrêt sur erreur de compte** : %s" % aborted)
    lines += ["", "## Synthèse par sonde", "", "| Sonde | Objet | Statut | Résumé |", "|---|---|---|---|"]
    for pid in PROBE_ORDER:
        res = ctx.results.get(pid)
        if res is None:
            continue
        lines.append("| %s | %s | %s | %s |" % (pid, PROBE_TITLES[pid], res.get("statut"),
                                               str(res.get("résumé", "")).replace("|", "/")))
    lines += ["", "## Détails", ""]
    for pid in PROBE_ORDER:
        res = ctx.results.get(pid)
        if res is None:
            continue
        lines += ["### %s — %s" % (pid, PROBE_TITLES[pid]), ""]
        for assertion in res.get("assertions") or []:
            lines.append("- [%s] %s" % ("x" if assertion.get("ok") else " ", assertion.get("attendu")))
        if res.get("erreur"):
            lines.append("- Erreur : `%s`" % res["erreur"])
        lines += ["", "```json", json.dumps(res.get("constats"), ensure_ascii=False, indent=2)[:12000], "```", ""]
    lines += ["## Décisions D1 à D22", "", "| Décision | Objet | Source | Consommateur |", "|---|---|---|---|"]
    for did, _, consumer in DECISION_TABLE:
        entry = decisions.get(did) or {}
        lines.append("| %s | %s | %s | %s |" % (did, DECISION_OBJECTS[did], entry.get("source"), consumer))
    lines += ["", "Valeurs complètes : `decisions.json` dans le dossier des fixtures.", "",
              "## Fixtures synthétiques", ""]
    lines += ["- `%s` (aucun succès réel obtenu)" % name for name in synthetic] or ["- aucune"]
    lines += ["", "## Nettoyage des collections", ""]
    coll = ctx.state.get("collection") or {}
    suppressions = ctx.state.get("suppressions") or []
    if not coll.get("create_attempted"):
        lines.append("- aucune collection créée")
    else:
        status = coll.get("create_status")
        lines.append("- collection `%s` : création tentée (%s)" % (
            coll.get("name"), "HTTP %s" % status if status is not None else "sans réponse"))
        for s in suppressions:
            origin = " (balayage par nom)" if s.get("balayage") else ""
            if s.get("erreur"):
                lines.append("- collection %s%s : erreur %s" % (s.get("collection"), origin, s.get("erreur")))
            else:
                lines.append("- collection %s%s : DELETE %s, GET ensuite %s" % (
                    s.get("collection"), origin, s.get("delete"), s.get("get_apres", "non demandé")))
        sweep = ctx.state.get("balayage") or {}
        if sweep:
            lines.append("- balayage par nom exact : %d collection(s) supprimée(s)%s" % (
                len(sweep.get("supprimees") or []), " ; erreur : %s" % sweep["erreur"] if sweep.get("erreur") else ""))
        if not suppressions:
            lines.append("- aucune suppression enregistrée : vérifier avec --cleanup --dry-run")
    if ctx.created_collections:
        lines.append("- **Collections non supprimées** : %s (lancer --cleanup)" % ctx.created_collections)
    return "\n".join(lines) + "\n"


def _signature(obj: Any, prefix: str = "", depth: int = 0, out: Optional[set] = None) -> set:
    """Ensemble des chemins de clés d'un corps JSON (profondeur 3), pour comparer des formes."""
    out = set() if out is None else out
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            out.add(path)
            _signature(value, path, depth + 1, out)
    elif isinstance(obj, list) and obj:
        _signature(obj[0], prefix + "[]", depth + 1, out)
    return out


def diff_against(fresh: Dict[str, Dict[str, Any]], decisions: Dict[str, Any], committed_dir: Path,
                 executed: Iterable[str]) -> List[str]:
    """Compare les fixtures et décisions fraîches aux fixtures versionnées (lecture seule)."""
    lines: List[str] = []
    committed_dir = Path(committed_dir)
    executed = set(executed)
    for name, fixture in sorted(fresh.items()):
        path = committed_dir / name
        if not path.exists():
            lines.append("NOUVELLE fixture : %s" % name)
            continue
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lines.append("illisible : %s" % name)
            continue
        if old.get("status") != fixture.get("status"):
            lines.append("%s : statut %s -> %s" % (name, old.get("status"), fixture.get("status")))
        before, after = _signature(old.get("body")), _signature(fixture.get("body"))
        if before - after:
            lines.append("%s : champs disparus %s" % (name, sorted(before - after)[:10]))
        if after - before:
            lines.append("%s : champs nouveaux %s" % (name, sorted(after - before)[:10]))
        old_ids = {d.get("id") for d in (old.get("body") or {}).get("data", []) or [] if isinstance(d, dict)} \
            if isinstance(old.get("body"), dict) else set()
        new_ids = {d.get("id") for d in (fixture.get("body") or {}).get("data", []) or [] if isinstance(d, dict)} \
            if isinstance(fixture.get("body"), dict) else set()
        if name.startswith("P2_") and old_ids != new_ids:
            lines.append("%s : ids retirés %s, ids ajoutés %s" % (name, sorted(map(str, old_ids - new_ids)),
                                                                  sorted(map(str, new_ids - old_ids))))
    for path in sorted(committed_dir.glob("P*_*.json")):
        pid = path.name.split("_", 1)[0]
        if pid in executed and path.name not in fresh and not path.name.endswith("_synthetic.json"):
            lines.append("fixture absente de ce run : %s" % path.name)
    old_decisions_path = committed_dir / "decisions.json"
    if old_decisions_path.exists():
        try:
            old_decisions = json.loads(old_decisions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_decisions = {}
        for did, entry in decisions.items():
            old_entry = old_decisions.get(did) or {}
            if entry.get("source") == "probe" and json.dumps(old_entry.get("value"), sort_keys=True) != \
                    json.dumps(entry.get("value"), sort_keys=True):
                lines.append("décision %s : valeur modifiée" % did)
    return lines


# ---------------------------------------------------------------------------
# Orchestration et CLI
# ---------------------------------------------------------------------------


def root_url(base_url: str) -> str:
    """Racine du service (base sans le suffixe /v1) pour /health."""
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _dotenv_values(env_file: str) -> Dict[str, Optional[str]]:
    """Lit un fichier .env sans l'afficher (dict vide s'il manque)."""
    if not os.path.isfile(env_file):
        return {}
    from dotenv import dotenv_values

    return dict(dotenv_values(env_file))


def live_gate_refusal(environ: Dict[str, str], env_file: str = ENV_FILE) -> Optional[str]:
    """Message de refus si le run live n'est pas explicitement autorisé, sinon None."""
    if environ.get("ALBERT_LIVE") != "1":
        return ("Refus : les sondes appellent l'API Albert de production. Lancer avec ALBERT_LIVE=1 "
                "dans le shell (jamais dans .env).")
    if "ALBERT_LIVE" in _dotenv_values(env_file):
        return ("Refus : ALBERT_LIVE figure dans %s. Cette variable doit venir du shell uniquement ; "
                "la retirer du fichier." % env_file)
    return None


def load_api_key(environ: Dict[str, str], env_file: str = ENV_FILE) -> Tuple[Optional[str], Optional[str]]:
    """Clé Albert : ``dotenv_values(.env)['ALBERT_API_KEY']``, sinon l'environnement (jamais affichée).

    Renvoie ``(clé, None)`` ou ``(None, message de refus)``. Une clé qui n'est
    pas faite d'au moins 8 caractères ASCII imprimables sans espace est
    refusée sans jamais être répétée : un caractère de contrôle dans l'en-tête
    ``Authorization`` ferait afficher la clé par la pile HTTP.
    """
    value = _dotenv_values(env_file).get("ALBERT_API_KEY") or environ.get("ALBERT_API_KEY")
    key = value.strip() if isinstance(value, str) else ""
    if not key:
        return None, "Refus : ALBERT_API_KEY absente de .env et de l'environnement."
    if not API_KEY_RE.fullmatch(key):
        return None, "Refus : ALBERT_API_KEY mal formée (caractères non imprimables)."
    return key, None


def build_client(api_key: str, transport: Optional[httpx.BaseTransport] = None,
                 base_url: str = BASE_URL) -> httpx.Client:
    """Client httpx (Bearer, User-Agent, sans redirection, délai par défaut 60 s)."""
    return httpx.Client(base_url=base_url, transport=transport, follow_redirects=False,
                        timeout=httpx.Timeout(TIMEOUTS["default"]),
                        headers={"Authorization": "Bearer %s" % api_key, "User-Agent": USER_AGENT})


def run_probes(ctx: ProbeContext, selected: Sequence[str]) -> Optional[str]:
    """Exécute les sondes choisies dans l'ordre P1…P21 ; renvoie le message d'arrêt éventuel."""
    ordered = [pid for pid in PROBE_ORDER if pid in set(selected)]
    group = [pid for pid in ordered if pid in COLLECTION_PROBES]
    group_done = False
    try:
        if "P1" not in ordered:
            # Identité du compte connue avant toute écriture, pour l'effacer partout.
            ctx.call("P0", "identity_prefetch", "GET", "/me", record=False)
        for pid in ordered:
            if pid in COLLECTION_PROBES:
                if not group_done:
                    group_done = True
                    run_collection_group(ctx, group)
                continue
            ctx.run_probe(pid, SIMPLE_PROBES[pid])
    except AccountLevelError as exc:
        return str(exc)
    finally:
        cleanup_created_collections(ctx)
    return None


def parse_only(value: str) -> List[str]:
    """Analyse ``--only P2,P5`` en liste d'identifiants valides."""
    ids = [part.strip().upper() for part in value.split(",") if part.strip()]
    unknown = [pid for pid in ids if pid not in PROBE_ORDER]
    if unknown or not ids:
        raise argparse.ArgumentTypeError("sondes inconnues : %s (attendu P1 à P21)" % ", ".join(unknown or [value]))
    return ids


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(prog="albert_probe.py",
                                     description="Sondes live de l'API Albert (exige ALBERT_LIVE=1 dans le shell).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="exécuter P1 à P21")
    mode.add_argument("--only", type=parse_only, help="liste de sondes, par exemple P2,P5")
    mode.add_argument("--cleanup", action="store_true", help="supprimer les collections ragpy-probe-*")
    parser.add_argument("--dry-run", action="store_true", help="avec --cleanup : lister sans supprimer")
    parser.add_argument("--out", default=DEFAULT_OUT, help="dossier des runs (défaut data/albert_probe)")
    parser.add_argument("--report", help="chemin du rapport Markdown")
    parser.add_argument("--fixtures-dir", help="dossier des fixtures assainies (défaut <out>/<ts>/fixtures)")
    parser.add_argument("--diff-against", help="dossier de fixtures versionnées à comparer (jamais réécrites)")
    parser.add_argument("--ocr-sample", help="PDF numérisé réel pour la latence de P5 (10 premières pages)")
    args = parser.parse_args(argv)
    if args.dry_run and not args.cleanup:
        parser.error("--dry-run exige --cleanup")
    return args


def main(argv: Optional[Sequence[str]] = None, *, transport: Optional[httpx.BaseTransport] = None,
         environ: Optional[Dict[str, str]] = None, clock: Optional[Callable[[], float]] = None,
         sleep: Optional[Callable[[float], None]] = None, wall: Optional[Callable[[], float]] = None) -> int:
    """Point d'entrée : garde live, clé, puis sondes ou nettoyage. Codes : 0, 1 (erreurs), 2 (refus), 3 (compte)."""
    args = parse_args(argv)
    env = dict(os.environ) if environ is None else dict(environ)
    refusal = live_gate_refusal(env)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    if args.diff_against and args.fixtures_dir and \
            Path(args.fixtures_dir).resolve() == Path(args.diff_against).resolve():
        print("Refus : --fixtures-dir et --diff-against désignent le même dossier ; les fixtures versionnées "
              "ne sont jamais réécrites.", file=sys.stderr)
        return 2
    api_key, key_refusal = load_api_key(env)
    if key_refusal or not api_key:
        print(key_refusal or "Refus : ALBERT_API_KEY absente.", file=sys.stderr)
        return 2
    wall = wall or time.time
    client = build_client(api_key, transport)
    try:
        if args.cleanup:
            try:
                remaining = run_cleanup(client, args.dry_run, log=lambda m: print(
                    sanitize_payload(m, secrets=[api_key]), flush=True))
            except AccountLevelError as exc:
                print("Arrêt : %s" % exc, file=sys.stderr)
                return 3
            except Exception as exc:  # jamais de trace brute : elle pourrait contenir l'en-tête Authorization
                print("Arrêt : %s" % type(exc).__name__, file=sys.stderr)
                return 1
            return 0 if args.dry_run or remaining == 0 else 1
        selected = PROBE_ORDER if args.all else args.only
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(wall()))
        run_dir = Path(args.out) / stamp
        fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else run_dir / "fixtures"
        ctx = ProbeContext(client, api_key, run_dir, fixtures_dir, clock=clock or time.monotonic,
                           sleep=sleep or time.sleep, wall=wall, ocr_sample=args.ocr_sample,
                           log=lambda m: print(m, flush=True))
        aborted = run_probes(ctx, selected)
        if aborted:
            ctx.log("ARRÊT : %s" % aborted)
        if "P13" in ctx.results:
            ctx.results["P13"].setdefault("constats", {}).update(ctx.sanitize(rate_info_view(ctx)))
        synthetic = write_synthetic_fixtures(ctx)
        diff_dir = Path(args.diff_against).resolve() if args.diff_against else None
        if diff_dir is None or diff_dir != fixtures_dir.resolve():
            # Après le run (et non avant) : un arrêt précoce sur erreur de compte ne vide pas les
            # fixtures des sondes qui n'ont pas tourné.
            removed = prune_stale_fixtures(fixtures_dir, ctx.executed_ids, ctx.fixtures)
            ctx.log("Fixtures périmées supprimées : %d" % len(removed))
        previous = None
        prev_path = fixtures_dir / "decisions.json"
        if prev_path.exists():
            try:
                previous = json.loads(prev_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = None

        def on_error(did: str, exc: Exception) -> None:
            """Compte une décision défaillante comme erreur de script."""
            ctx.script_errors += 1
            ctx.log("[%s] erreur de calcul : %s" % (did, type(exc).__name__))

        decisions = ctx.sanitize(build_decisions(ctx.results, ctx.executed_ids, previous, on_error))
        ctx._write_json(fixtures_dir / "decisions.json", decisions)
        ctx._write_json(run_dir / "decisions.json", decisions)
        ctx._write_json(run_dir / "results.json", ctx.results)
        report_path = Path(args.report) if args.report else run_dir / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(ctx.sanitize(build_report(ctx, decisions, aborted, synthetic)), encoding="utf-8")
        ctx.log("Rapport : %s" % _rel(report_path))
        if args.diff_against:
            fresh = {name: fx for name, fx in ctx.fixtures.items()}
            diffs = diff_against(fresh, decisions, Path(args.diff_against), ctx.executed_ids)
            for line in diffs:
                ctx.log("DIFF: %s" % line)
            ctx.log("DIFFÉRENCES: %d" % len(diffs))
        print("SONDES: %d exécutées, %d erreur de script" % (ctx.executed, ctx.script_errors), flush=True)
        if aborted:
            return 3
        return 1 if ctx.script_errors else 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
