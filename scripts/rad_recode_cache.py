"""scripts/rad_recode_cache.py — Déterminisme du recodage GPT (Lot 7).

Réduit à l'extrême le non-déterminisme du texte **stocké/embeddé** par le recodage
GPT (`gpt_recode_batch` tourne à `temperature>0` → texte recodé non déterministe →
jitter du cross-check Jaccard et de la similarité d'embedding). Trois leviers, tous
**OFF par défaut** (sortie byte-identique à aujourd'hui) :

  - **Durcissement décodage** (``RECODE_HARDEN_ENABLED``) — temp 0 / top_p 1 / seed /
    snapshot daté : réduit la volatilité au 1ᵉʳ appel.
  - **Cache de recodage** (``RECODE_CACHE_ENABLED``) — SQLite clé par ``content_hash`` :
    la vraie garantie (« ne JAMAIS refaire l'appel » → texte byte-identique entre runs).
    Le résidu serveur (batch-invariance, irréductible côté client) est ainsi payé 1×
    par clé puis gelé.
  - **Cache d'embedding** (``RECODE_EMBED_CACHE_ENABLED``) — ferme l'axe vecteur.

ADVISORY : toute erreur (I/O, schéma incompatible) → repli silencieux sur l'appel LLM
normal, **jamais fatal**. On ne cache QUE les succès réels. Le ``content_hash`` (clé)
est celui de rad_dedup (texte BRUT pré-recodage) : ce module ne change que QUEL texte
est stocké, **jamais** la clé de décision dédup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass

logger = logging.getLogger("rad_recode_cache")

# Schéma du cache. Bump → les caches d'une version antérieure sont ignorés.
_USER_VERSION = 1

# Verrou DÉDIÉ aux écritures cache (≠ SAVE_LOCK global de rad_chunk) : il n'est
# JAMAIS tenu pendant le découpage/sauvegarde des chunks, seulement le temps d'un
# INSERT+commit SQLite. Sérialise les writes in-process pour éviter le churn
# « database is locked » (WAL + busy_timeout gèrent le reste).
_CACHE_LOCK = threading.Lock()
_local = threading.local()


# ---------------------------------------------------------------------------
# Parsers d'environnement (stdlib, self-contained)
# ---------------------------------------------------------------------------
def _b(v, d):
    """Booléen d'environnement (``1/true/yes/on``, insensible à la casse) ; ``d`` si
    ``None``."""
    if v is None:
        return d
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _i(v, d):
    """Entier d'environnement ; ``d`` si absent ou invalide."""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return d


def _f(v, d):
    """Flottant d'environnement ; ``d`` si absent ou invalide."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return d


@dataclass
class RecodeConfig:
    cache_enabled: bool = False
    harden_enabled: bool = False
    embed_cache_enabled: bool = False
    model: str = "gpt-4o-mini-2024-07-18"
    prefer_openai: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 1600
    openrouter_provider: str = ""
    cache_path: str = "data/recode_cache.sqlite"

    @classmethod
    def from_env(cls, env=None):
        """Construit la configuration depuis les variables ``RECODE_*`` de ``env``
        (``os.environ`` par défaut) ; toute valeur absente ou invalide retombe sur le
        défaut du champ."""
        env = os.environ if env is None else env
        return cls(
            cache_enabled=_b(env.get("RECODE_CACHE_ENABLED"), False),
            harden_enabled=_b(env.get("RECODE_HARDEN_ENABLED"), False),
            embed_cache_enabled=_b(env.get("RECODE_EMBED_CACHE_ENABLED"), False),
            model=env.get("RECODE_MODEL", "gpt-4o-mini-2024-07-18"),
            prefer_openai=_b(env.get("RECODE_PREFER_OPENAI"), True),
            temperature=_f(env.get("RECODE_TEMPERATURE"), 0.0),
            top_p=_f(env.get("RECODE_TOP_P"), 1.0),
            seed=_i(env.get("RECODE_SEED"), 0),
            max_tokens=_i(env.get("RECODE_MAX_TOKENS"), 1600),
            openrouter_provider=env.get("RECODE_OPENROUTER_PROVIDER", "") or "",
            cache_path=env.get("RECODE_CACHE_PATH", "data/recode_cache.sqlite"),
        )

    def decode_params_json(self) -> str:
        """Composante de la clé de cache : params de décodage triés."""
        return json.dumps(
            {"temperature": self.temperature, "top_p": self.top_p, "seed": self.seed, "max_tokens": self.max_tokens},
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Composition des clés
# ---------------------------------------------------------------------------
def prompt_fingerprint(system: str, template: str, instruction: str) -> str:
    """``PROMPT_VERSION`` dérivé : toute édition d'un octet de system/template/
    instruction → fingerprint différent → MISS automatique (invalidation par clé)."""
    h = hashlib.sha256()
    for part in (system or "", template or "", instruction or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def recode_key(content_hash, model, provider, prompt_version, decode_params_json) -> str:
    """Clé de cache recodage : sha256 sur (content_hash · model · provider ·
    PROMPT_VERSION · decode_params), joints par U+0000. SANS doc_id/id aléatoires."""
    raw = "\x00".join(
        [content_hash or "", model or "", provider or "", prompt_version or "", decode_params_json or ""]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embed_key(text, embed_model, embed_params_json) -> str:
    """Clé de cache embedding : sha256 sur (sha256(texte recodé) · model · params)."""
    text_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    raw = "\x00".join([text_hash, embed_model or "", embed_params_json or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Connexion SQLite paresseuse par (process, thread) — fork-safe Celery
# ---------------------------------------------------------------------------
def _get_conn(path):
    """Connexion paresseuse par-process + thread-local : JAMAIS ouverte à l'import
    (fork-safe Celery). WAL + synchronous=NORMAL + busy_timeout. PRAGMA user_version
    vérifié → mismatch lève (cache ignoré par l'appelant)."""
    pid = os.getpid()
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "pid", None) == pid and getattr(_local, "path", None) == path:
        return conn
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if ver not in (0, _USER_VERSION):
        conn.close()
        raise RuntimeError(f"recode cache user_version {ver} != {_USER_VERSION} (cache ignoré)")
    conn.execute("CREATE TABLE IF NOT EXISTS recode_cache (key TEXT PRIMARY KEY, text TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
    if ver == 0:
        conn.execute(f"PRAGMA user_version={_USER_VERSION}")
    conn.commit()
    _local.conn = conn
    _local.pid = pid
    _local.path = path
    return conn


def get_recode(cfg: RecodeConfig, key):
    """Lit le texte recodé en cache pour ``key``. Renvoie ``None`` (MISS) si la
    clé est vide, absente du cache ou en cas d'erreur (advisory)."""
    if not key:
        return None
    try:
        row = _get_conn(cfg.cache_path).execute(
            "SELECT text FROM recode_cache WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
    except Exception as exc:  # advisory : MISS sur erreur
        logger.debug("recode cache GET failed: %s", exc)
        return None


def put_recode(cfg: RecodeConfig, key, text):
    """``INSERT OR IGNORE`` PUIS ``SELECT`` la valeur canonique → deux MISS
    concurrents de même clé convergent vers une seule valeur (in-run et cross-run).
    Renvoie la valeur canonique (à stocker dans le chunk), ou ``text`` en repli."""
    if not key or text is None:
        return text
    try:
        with _CACHE_LOCK:
            conn = _get_conn(cfg.cache_path)
            conn.execute("INSERT OR IGNORE INTO recode_cache(key, text) VALUES(?, ?)", (key, text))
            conn.commit()
            row = conn.execute("SELECT text FROM recode_cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else text
    except Exception as exc:  # advisory : ne jamais empêcher le run
        logger.debug("recode cache PUT failed: %s", exc)
        return text


def get_embed(cfg: RecodeConfig, key):
    """Lit le vecteur d'embedding en cache pour ``key`` (désérialisé depuis JSON).
    Renvoie ``None`` (MISS) si la clé est vide, absente du cache ou en cas d'erreur."""
    if not key:
        return None
    try:
        row = _get_conn(cfg.cache_path).execute(
            "SELECT vector FROM embed_cache WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception as exc:  # advisory
        logger.debug("embed cache GET failed: %s", exc)
        return None


def put_embed(cfg: RecodeConfig, key, vector):
    """Stocke ``vector`` (sérialisé JSON) par ``INSERT OR IGNORE`` puis relit la
    valeur canonique, comme ``put_recode``. Renvoie le vecteur canonique, ou
    ``vector`` tel quel si la clé est vide, le vecteur ``None`` ou en cas d'erreur."""
    if not key or vector is None:
        return vector
    try:
        payload = json.dumps(vector)
        with _CACHE_LOCK:
            conn = _get_conn(cfg.cache_path)
            conn.execute("INSERT OR IGNORE INTO embed_cache(key, vector) VALUES(?, ?)", (key, payload))
            conn.commit()
            row = conn.execute("SELECT vector FROM embed_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else vector
    except Exception as exc:  # advisory
        logger.debug("embed cache PUT failed: %s", exc)
        return vector
