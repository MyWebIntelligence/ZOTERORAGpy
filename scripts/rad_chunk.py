import os
import json
import random
import time
import threading
import datetime
import dataclasses
import importlib
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, RateLimitError
import spacy
from collections import Counter
import subprocess # Added for spacy download subprocess
import logging

# Metrics tracking (optional - works with or without prometheus)
try:
    from scripts.metrics_helper import (
        track_embedding_batch, track_chunking, track_error,
        log_metrics_summary, _collector as metrics_collector
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    track_embedding_batch = None
    track_chunking = None
    track_error = None
    log_metrics_summary = None
    metrics_collector = None

# Module de déduplication (Lot 1+). Stdlib uniquement : write-path (ici) et
# read-path (rad_vectordb) partagent la MÊME normalisation/hash. Import robuste
# au contexte package (Celery/tests) et CLI (python scripts/rad_chunk.py).
try:
    from scripts import rad_dedup
except ImportError:
    import rad_dedup

# Cache/déterminisme du recodage (Lot 7). OFF par défaut → byte-identique.
try:
    from scripts import rad_recode_cache
except ImportError:
    import rad_recode_cache

# Chargement du .env gardé (RAGPY_DOTENV_DENY : secrets non rechargés pour un
# sous-processus non-admin) et fournisseur historique d'un nom de modèle.
# Modules stdlib, même import double que ci-dessus.
try:
    from scripts.rad_env import load_dotenv_guarded
except ImportError:
    from rad_env import load_dotenv_guarded
try:
    from scripts.rad_providers import (
        ALBERT_PREFIX, PROVIDER_ALBERT, PROVIDER_OPENROUTER,
        EmbeddingConfig, legacy_cache_provider_label, legacy_provider, resolve_llm_provider,
    )
except ImportError:
    from rad_providers import (
        ALBERT_PREFIX, PROVIDER_ALBERT, PROVIDER_OPENROUTER,
        EmbeddingConfig, legacy_cache_provider_label, legacy_provider, resolve_llm_provider,
    )

# Socle Albert (DINUM) : à l'import, seul le paquet léger (``config`` et
# ``errors``, stdlib) est chargé, depuis une SEULE racine (``scripts.rad_albert``
# ou ``rad_albert``). Les sous-modules ``catalog``, ``client`` et ``preflight``
# (httpx, limiteur, retry, ledger) ne sont chargés qu'à la première utilisation
# d'Albert, depuis la même racine (``_albert_module``) : les classes d'erreur
# interceptées ici sont donc celles que lève le client.
try:
    from scripts import rad_albert as _rad_albert
except ImportError:
    import rad_albert as _rad_albert

# Attempt to import RecursiveCharacterTextSplitter from langchain_text_splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: langchain_text_splitters not found. TEXT_SPLITTER will not be initialized.")
    print("Please install it via 'pip install langchain-text-splitters'")
    RecursiveCharacterTextSplitter = None

# ----------------------------------------------------------------------
# Global Configuration and Initializations
# ----------------------------------------------------------------------
SAVE_LOCK = threading.Lock()

# Load environment variables from .env file (secrets refusés non rechargés)
load_dotenv_guarded()

# OpenAI API Client Initialization — jamais de saisie interactive : sans clé,
# le client vaut None et la CLI s'arrête (exit 1) si la phase demandée l'exige.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("OPENAI_API_KEY not found in environment variables or .env file.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# OpenRouter Client Initialization (optional, for cost-effective alternatives)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    print("OpenRouter client initialized successfully.")
else:
    print("OpenRouter API key not found. Will use OpenAI for all LLM calls.")

# Text Splitter Initialization
if RecursiveCharacterTextSplitter:
    TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-large",  # This model is for token counting for the splitter
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "#", "##", "\n", " ", ""] # Ajout des nouveaux séparateurs avec priorité
    )
else:
    TEXT_SPLITTER = None # Fallback or error if not available

# spaCy Model Initialization
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    print("Le modèle spaCy 'fr_core_news_md' n'est pas trouvé. Tentative de téléchargement...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"], check=True)
        nlp = spacy.load("fr_core_news_md")
        print("Modèle spaCy 'fr_core_news_md' téléchargé et chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle spaCy : {e}")
        print("Veuillez installer le modèle manuellement : python -m spacy download fr_core_news_md")
        nlp = None # Fallback or error

# ----------------------------------------------------------------------
# Environment variable helper with validation
# ----------------------------------------------------------------------
def get_env_int(key: str, default: int, min_val: int = 1) -> int:
    """Get integer from environment with validation and fallback."""
    try:
        value = int(os.getenv(key, default))
        return max(min_val, value)
    except (ValueError, TypeError):
        logging.warning(f"Invalid {key}, using default {default}")
        return default

# Default constants from master code (configurable via .env)
DEFAULT_JSON_FILE_CHUNKS = "df_chunks.json"
_cpu_count = os.cpu_count() or 2
DEFAULT_MAX_WORKERS = get_env_int('DEFAULT_MAX_WORKERS', max(1, _cpu_count - 1))
DEFAULT_BATCH_SIZE_GPT = get_env_int('DEFAULT_BATCH_SIZE_GPT', 5)
DEFAULT_EMBEDDING_BATCH_SIZE = get_env_int('DEFAULT_EMBEDDING_BATCH_SIZE', 32)
DEFAULT_DOC_WORKERS = get_env_int('DEFAULT_DOC_WORKERS', 3)
DEFAULT_INPUT_JSON_WITH_EMBEDDINGS = "df_chunks_with_embeddings.json"
DEFAULT_OUTPUT_JSON_SPARSE = "df_chunks_with_embeddings_sparse.json"

# ----------------------------------------------------------------------
# Albert (DINUM) — fournisseur de chat souverain, opt-in (ALBERT_ENABLED=1)
# ----------------------------------------------------------------------
# Constantes de script CLI lues à l'import (décision 11), patchables ; Albert
# désactivé, aucune n'a d'effet (ni appel, ni sortie, ni champ JSON nouveau).
# Le recodage n'emprunte Albert que sur sélection explicite d'un modèle
# ``albert/<id>`` (``--model`` ou ``RECODE_MODEL`` en mode durci), sans jamais
# replier sur OpenAI ni OpenRouter.
def _albert_import_config():
    """Configuration Albert lue à l'import (constantes du module).

    Une ``ALBERT_BASE_URL`` refusée alors qu'Albert est activé ne bloque pas
    l'import (chemins OpenAI/OpenRouter intacts) : les défauts sont gardés ici et
    l'erreur explicite est relevée à la première utilisation d'Albert
    (``_albert_config``).
    """
    try:
        return _rad_albert.AlbertConfig.from_env()
    except ValueError:
        return _rad_albert.AlbertConfig(enabled=True)


_ALBERT_IMPORT_CFG = _albert_import_config()
ALBERT_ENABLED = _ALBERT_IMPORT_CFG.enabled
# Frontière du script : la clé est lue ici (jamais par la bibliothèque rad_albert).
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY") or None
# Plafond de recodage Albert PAR PROCESSUS (D22), partagé par tous les workers
# de documents et de chunks (motif MISTRAL_SEMAPHORE), tenu pendant l'envoi seul.
ALBERT_RECODE_CONCURRENCY = _ALBERT_IMPORT_CFG.recode_concurrency
ALBERT_RECODE_SEMAPHORE = threading.Semaphore(ALBERT_RECODE_CONCURRENCY)
# D21 : texte LightOnOCR recodé par défaut (ALBERT_OCR_SKIP_RECODE=0).
ALBERT_OCR_SKIP_RECODE = _ALBERT_IMPORT_CFG.ocr_skip_recode
# D11 : seed déterministe sur Albert (empreinte stable) → envoyé en mode durci et
# présent dans les paramètres de la clé de cache ; faux → ni envoyé ni en clé.
ALBERT_SEED_DETERMINISTIC = True
# Garde de longueur (Albert seulement) : un texte recodé hors de
# [0.3, 1.5] × longueur brute est suspect → texte brut conservé, jamais caché.
ALBERT_RECODE_MIN_RATIO = 0.3
ALBERT_RECODE_MAX_RATIO = 1.5
# Paramètres historiques de ``gpt_recode_batch`` hors durcissement.
_RECODE_DEFAULT_TEMPERATURE = 0.3
_RECODE_DEFAULT_MAX_TOKENS = 8000

# État Albert du processus : client paresseux (ledger d'usage compris) et
# première erreur fatale (compte, quota, modèle introuvable) qui arrête le job
# (``None`` tant qu'aucun arrêt ; remis à zéro par ``reset_albert_state``).
_ALBERT_CLIENT = None
_ALBERT_CLIENT_LOCK = threading.Lock()
_ALBERT_RESOLVE_LOCK = threading.Lock()
_ALBERT_ABORT = None
_ALBERT_ABORT_LOCK = threading.Lock()
# Première erreur de compte (clé, compte expiré, budget, quota) rencontrée par
# les embeddings Albert : les lots suivants restent sans vecteur, sans appel, et
# la phase dense sort en 2 (décision 22). Remise à zéro par ``reset_albert_state``.
_ALBERT_EMBED_ABORT = None
_ALBERT_EMBED_ABORT_LOCK = threading.Lock()

# Embeddings denses : champs d'espace écrits sur chaque chunk seulement hors
# espace par défaut (OpenAI text-embedding-3-large, fichier inchangé à l'octet),
# et plafond d'un lot Albert (D12 : 65 textes donnent un 413).
EMBEDDING_SPACE_FIELDS = ("embedding_provider", "embedding_model", "embedding_dim")
ALBERT_EMBED_BATCH_MAX = 64


def recode_skip_providers(skip_lightonocr):
    """Fournisseurs d'OCR dont le texte n'est pas recodé.

    Mistral, CSV et les OCR locaux (markdown déjà propre), plus
    ``albert_mistral_ocr`` (même moteur Mistral OCR servi par Albert).
    ``albert_lightonocr`` n'y figure que si ``ALBERT_OCR_SKIP_RECODE=1`` (D21 :
    qualité non évaluée, recodage par défaut).

    Args:
        skip_lightonocr: valeur de ``ALBERT_OCR_SKIP_RECODE``.

    Returns:
        Tuple des valeurs de ``texteocr_provider`` exclues du recodage.
    """
    providers = ("mistral", "csv", "docling", "mineru", "marker", "albert_mistral_ocr")
    if skip_lightonocr:
        providers += ("albert_lightonocr",)
    return providers


# Providers dont l'OCR produit déjà un markdown propre → recodage GPT inutile
# (économie de coût). Mistral et CSV historiquement ; Lot 4 ajoute les moteurs
# d'OCR LOCAL (Docling/MinerU/Marker) qui sortent aussi du markdown structuré ;
# Albert ajoute Mistral OCR servi par /v1/ocr (et LightOnOCR sur option, D21).
RECODE_SKIP_PROVIDERS = recode_skip_providers(ALBERT_OCR_SKIP_RECODE)

# ----------------------------------------------------------------------
# PART 1: Découpage en CHUNKs assisté par gpt_recode
# ----------------------------------------------------------------------

# Prompt de recodage extrait en constantes : le fingerprint du prompt (Lot 7) en
# dérive, donc toute édition d'un octet du system/template/instruction invalide
# automatiquement le cache (MISS). NE PAS reformuler sans intention.
_RECODE_SYSTEM = "Assistant spécialisé en recodage de textes académiques."
_RECODE_TEMPLATE = "Instructions : {instructions}\n\nTexte à recoder :\n{chunk}\n\nTexte recodé :"
_RECODE_INSTRUCTIONS = (
    "ce chunk est issu d'un ocr brut qui laisse beaucoup de blocs de texte inutiles comme des titres de pages, "
    "des numeros, etc. Nettoie ce chunk pour en faire un texte propre qui commence par une phrase complète et se "
    "termine par un point. Supprime le bruit d'OCR et les imperfections en conservant le sens original. Ne echange "
    "ni ajoute aucun mot du texte d'origine. C'est une correction et un nettoyage de texte (suppression des erreurs) "
    "pas une réécriture"
)


# ----------------------------------------------------------------------
# Albert : sélection, client, preflight, recodage
# ----------------------------------------------------------------------
def _selects_albert(model):
    """Vrai si le nom de modèle porte le préfixe ``albert/`` (casse ignorée)."""
    return str(model or "")[: len(ALBERT_PREFIX)].lower() == ALBERT_PREFIX


def effective_recode_model(cli_model, cfg):
    """Modèle de recodage effectif : source unique du routage ET de la clé de cache.

    Un modèle ``albert/…`` passé en ligne de commande l'emporte (``prefer_openai``
    est alors sans objet) ; sinon la sémantique durcie historique s'applique à
    l'identique : ``RECODE_MODEL`` (``cfg.model``) si ``RECODE_HARDEN_ENABLED``,
    le modèle de la ligne de commande sinon.

    Args:
        cli_model: modèle passé par ``--model`` (ou par l'appelant).
        cfg: ``RecodeConfig`` (ou ``None`` : pas de durcissement).

    Returns:
        Le nom de modèle effectif (``albert/<id>`` compris).
    """
    if _selects_albert(cli_model):
        return cli_model
    if cfg is not None and cfg.harden_enabled:
        return cfg.model
    return cli_model


def _albert_module(name):
    """Sous-module ``name`` du socle Albert, importé à la demande depuis la racine de ``_rad_albert``.

    Même racine que les classes d'erreur importées au chargement du module
    (``scripts.rad_albert.<name>`` ou ``rad_albert.<name>``) ; n'est appelé que
    sur un chemin Albert, jamais Albert désactivé.

    Args:
        name: ``catalog``, ``client`` ou ``preflight``.

    Returns:
        Le module importé (mis en cache par ``sys.modules``).
    """
    return importlib.import_module(f"{_rad_albert.__name__}.{name}")


def _albert_config():
    """Configuration Albert lue à l'appel (lève ``ValueError`` si l'URL de base est refusée)."""
    return _rad_albert.AlbertConfig.from_env()


def _albert_enabled():
    """Vrai si Albert est activé : la constante de module ``ALBERT_ENABLED`` fait foi (décision 11).

    Lue à l'import (sous-processus CLI) et patchable ; l'environnement courant
    n'est jamais relu ici. Une ``ALBERT_BASE_URL`` refusée alors qu'Albert est
    activé laisse la constante vraie : l'erreur explicite est relevée par
    ``_albert_config`` à la première utilisation.
    """
    return bool(ALBERT_ENABLED)


def _get_albert_client():
    """Client Albert du processus, créé à la première utilisation (jamais à l'import).

    La clé vient de la constante ``ALBERT_API_KEY`` (frontière du script) ; le
    client porte le ledger d'usage du processus. ``_ALBERT_CLIENT`` peut être
    remplacé par un client construit sur un transport factice (tests).

    Raises:
        AlbertAuthError: clé absente (``AlbertMissingKeyError``).
        ValueError: ``ALBERT_BASE_URL`` refusée.
    """
    global _ALBERT_CLIENT
    with _ALBERT_CLIENT_LOCK:
        if _ALBERT_CLIENT is None:
            _ALBERT_CLIENT = _albert_module("client").AlbertClient(_albert_config(), ALBERT_API_KEY)
        return _ALBERT_CLIENT


def reset_albert_state():
    """Oublie le client Albert et les arrêts mémorisés du processus (recodage et
    embeddings ; tests, nouveau job en processus)."""
    global _ALBERT_CLIENT, _ALBERT_ABORT, _ALBERT_EMBED_ABORT
    with _ALBERT_CLIENT_LOCK:
        _ALBERT_CLIENT = None
    with _ALBERT_ABORT_LOCK:
        _ALBERT_ABORT = None
    with _ALBERT_EMBED_ABORT_LOCK:
        _ALBERT_EMBED_ABORT = None


def _albert_set_abort(exc):
    """Mémorise la première erreur fatale Albert du processus et l'annonce une seule fois.

    Erreurs de compte (clé, compte expiré, budget, quota) et modèle introuvable
    ou incompatible : les lots suivants passent en ``fallback_raw`` sans appel et
    la CLI sort en 2, après la ligne ``albert_abort_marker`` qui distingue une
    erreur de compte (``kind=account``) d'une erreur de modèle (``kind=model``).
    """
    global _ALBERT_ABORT
    with _ALBERT_ABORT_LOCK:
        first = _ALBERT_ABORT is None
        if first:
            _ALBERT_ABORT = exc
    if first:
        print(f"Albert : recodage arrêté — {exc} "
              "Lots restants conservés en texte brut (fallback_raw), sans appel.")


class _AlbertAborted(Exception):
    """Envoi refusé : un arrêt Albert a été mémorisé pendant l'attente du sémaphore."""


class _AlbertAbortGate:
    """Porte tenue juste après ``ALBERT_RECODE_SEMAPHORE`` : refuse l'envoi après un arrêt.

    Un worker qui attendait le sémaphore quand l'arrêt a été mémorisé n'envoie
    donc pas sa requête (les lots restants partent en ``fallback_raw`` sans appel).
    """

    def acquire(self):
        """Laisse passer l'envoi, ou lève ``_AlbertAborted`` si l'arrêt est mémorisé."""
        if _ALBERT_ABORT is not None:
            raise _AlbertAborted()
        return True

    def release(self):
        """Rien à libérer (la porte ne compte aucun jeton)."""
        return None


_ALBERT_ABORT_GATE = _AlbertAbortGate()


def _albert_prepare(model):
    """Client Albert et id épinglé du modèle ``albert/…`` (résolu par ``/v1/models``).

    L'id résolu est celui envoyé sur le fil et celui de la clé de cache : un
    alias (``openweight-small``) n'est jamais utilisé tel quel. Aucun repli.

    Raises:
        AlbertDisabledError: Albert désactivé.
        ValueError: modèle vide après ``albert/`` ou URL de base refusée.
        AlbertAuthError: clé absente ou refusée, compte expiré, quota épuisé.
        AlbertPermanentError: modèle absent de ``/v1/models`` ou de type incompatible.
        AlbertTransientError: ``/v1/models`` injoignable après réessais.
    """
    wire = resolve_llm_provider(model, albert_enabled=_albert_enabled()).wire_model
    client = _get_albert_client()
    with _ALBERT_RESOLVE_LOCK:
        model_id = client.resolve_model_id(wire)
    _albert_module("catalog").check_endpoint_type(model_id, "chat")
    return client, model_id


def _albert_prepare_or_abort(model):
    """``_albert_prepare`` pour un lot ; ``None`` si le lot doit rester en texte brut.

    Arrêt déjà mémorisé → ``None`` sans appel. Erreur de compte ou modèle
    introuvable/incompatible → arrêt mémorisé, ``None``. Erreur transitoire
    (``/v1/models`` injoignable) → ``None`` pour ce lot seulement. Une erreur de
    configuration (Albert désactivé, URL refusée) est relevée : jamais de repli.
    """
    if _ALBERT_ABORT is not None:
        return None
    try:
        return _albert_prepare(model)
    except (_rad_albert.AlbertAuthError, _rad_albert.AlbertPermanentError) as exc:
        _albert_set_abort(exc)
        return None
    except _rad_albert.AlbertError as exc:
        print(f"Albert indisponible pour ce lot (texte brut conservé) : {exc}")
        return None


def _albert_decode_params(recode_cfg, model_id, *, temperature, max_tokens):
    """Paramètres de décodage envoyés à Albert pour le recodage.

    Mode durci : ceux de ``RecodeConfig`` (température, ``top_p``, ``max_tokens``),
    plus ``seed`` si le seed est déterministe sur Albert (D11). Sinon : la
    température recommandée par le catalogue pour le modèle (à défaut
    ``temperature``) et ``max_tokens`` ; ni ``top_p`` ni ``seed``.
    """
    if recode_cfg is not None and recode_cfg.harden_enabled:
        params = {
            "temperature": recode_cfg.temperature,
            "top_p": recode_cfg.top_p,
            "max_tokens": recode_cfg.max_tokens,
        }
        if ALBERT_SEED_DETERMINISTIC:
            params["seed"] = recode_cfg.seed
        return params
    recommended = _albert_module("catalog").recommended_temperature(model_id)
    return {
        "temperature": temperature if recommended is None else recommended,
        "max_tokens": max_tokens,
    }


def _albert_decode_params_json(params, model_id, albert_cfg):
    """Composante ``decode_params`` de la clé de cache Albert.

    Mêmes champs que ``RecodeConfig.decode_params_json()`` avec les valeurs
    réellement envoyées (``seed`` seulement s'il est envoyé), enrichis des
    champs Albert : ``provider`` et, pour un modèle à raisonnement, l'effort et
    la marge de raisonnement.
    """
    doc = dict(params)
    doc["provider"] = PROVIDER_ALBERT
    if _albert_module("catalog").is_reasoning(model_id):
        doc["reasoning_effort"] = albert_cfg.reasoning_effort
        doc["reasoning_headroom"] = albert_cfg.reasoning_headroom
    return json.dumps(doc, sort_keys=True)


def _albert_length_ratio_ok(raw, text):
    """Vrai si ``len(text) / len(raw)`` est dans [ALBERT_RECODE_MIN_RATIO, ALBERT_RECODE_MAX_RATIO]."""
    ratio = len(text) / max(1, len(str(raw).strip()))
    return ALBERT_RECODE_MIN_RATIO <= ratio <= ALBERT_RECODE_MAX_RATIO


def _albert_extract(result, raw):
    """Statut d'une réponse Albert (même contrat que ``_extract``, plus la garde de longueur).

    ``length`` / ``content_filter`` → ``fallback_truncated`` ; contenu vide →
    ``fallback_raw`` (le champ ``reasoning`` n'est jamais repris) ; longueur
    hors bornes → ``fallback_suspicious`` (statut interne, enregistré
    ``fallback_raw`` par l'appelant) ; sinon ``recoded``.

    Returns:
        ``(texte | None, statut)``.
    """
    finish = getattr(result, "finish_reason", None)
    text = str(getattr(result, "content", result) or "").strip()
    if finish in ("length", "content_filter"):
        return None, "fallback_truncated"
    if not text:
        return None, "fallback_raw"
    if not _albert_length_ratio_ok(raw, text):
        return None, "fallback_suspicious"
    return text, "recoded"


def _albert_recode_core(chunks, instructions, client, model_id, params):
    """Recode les chunks par Albert (rôle ``recode``), en un seul passage.

    Une seule couche de réessai (celle du client : backoff hors sémaphore, repli
    de rôle sur 503 répétés) ; pas de 2ᵉ passage séquentiel. Le sémaphore de
    module ``ALBERT_RECODE_SEMAPHORE`` n'est tenu que pendant l'envoi, suivi de
    la porte d'arrêt (aucun envoi après un arrêt mémorisé). Aucun appel à
    ``client`` ni à ``openrouter_client`` : tout échec garde le texte brut.

    Returns:
        ``(textes, statuts, modèles_servis)`` index-alignés à ``chunks``
        (``modèles_servis[i]`` : id qui a servi la réponse, ou ``None``).
    """
    n = len(chunks)
    texts = list(chunks)
    statuses = ["fallback_raw"] * n
    served = [None] * n
    messages_list = [
        [
            {"role": "system", "content": _RECODE_SYSTEM},
            {"role": "user", "content": _RECODE_TEMPLATE.format(instructions=instructions, chunk=chunk)},
        ]
        for chunk in chunks
    ]
    print(f"Using Albert with model: {model_id}")

    def _one(idx):
        """Recode le chunk ``idx`` ; renvoie ``(idx, texte | None, statut, modèle servi | None)``."""
        if _ALBERT_ABORT is not None:
            return idx, None, "fallback_raw", None
        try:
            result = client.chat(
                messages_list[idx], model_id, role="recode",
                semaphore=(ALBERT_RECODE_SEMAPHORE, _ALBERT_ABORT_GATE), **params
            )
        except _AlbertAborted:
            return idx, None, "fallback_raw", None
        except _rad_albert.AlbertTruncatedError:
            return idx, None, "fallback_truncated", None
        except _rad_albert.AlbertAuthError as exc:
            _albert_set_abort(exc)
            return idx, None, "fallback_raw", None
        except _rad_albert.AlbertPermanentError as exc:
            if exc.reason == "not_found":
                _albert_set_abort(exc)
            else:
                print(f"Erreur chunk #{idx+1} (Albert) : {exc}")
            return idx, None, "fallback_raw", None
        except Exception as exc:
            print(f"Erreur chunk #{idx+1} (Albert) : {exc}")
            return idx, None, "fallback_raw", None
        text, status = _albert_extract(result, chunks[idx])
        if status == "fallback_suspicious":
            print(f"Chunk #{idx+1} : longueur recodée hors de "
                  f"[{ALBERT_RECODE_MIN_RATIO}, {ALBERT_RECODE_MAX_RATIO}] × brut → texte brut conservé.")
            status = "fallback_raw"
        return idx, text, status, getattr(result, "model", None)

    with ThreadPoolExecutor(max_workers=max(1, min(DEFAULT_MAX_WORKERS, n))) as executor:
        futures = [executor.submit(_one, idx) for idx in range(n)]
        for future in as_completed(futures):
            idx, text, status, model_served = future.result()
            statuses[idx] = status
            served[idx] = model_served
            if status == "recoded" and text is not None:
                texts[idx] = text

    return texts, statuses, served


def _fill_models_out(models_out, values):
    """Remplit la liste de sortie ``models_out`` (si fournie) avec ``values``."""
    if models_out is not None:
        models_out[:] = list(values)


def _albert_recode_batch(chunks, instructions, model, *, recode_cfg=None,
                         temperature=_RECODE_DEFAULT_TEMPERATURE,
                         max_tokens=_RECODE_DEFAULT_MAX_TOKENS, models_out=None):
    """Recodage d'un lot par Albert (retour anticipé de ``gpt_recode_batch``).

    Même contrat que ``gpt_recode_batch`` : ``(textes, statuts)`` index-alignés,
    texte brut pour tout repli. Albert désactivé → ``AlbertDisabledError`` (jamais
    de routage vers OpenRouter d'un nom ``albert/…``).

    Args:
        chunks: textes bruts.
        instructions: consigne de recodage.
        model: modèle effectif ``albert/<id ou alias>``.
        recode_cfg: ``RecodeConfig`` (durcissement éventuel).
        temperature: température hors durcissement, à défaut de celle du catalogue.
        max_tokens: budget de réponse hors durcissement.
        models_out: liste remplie des ids servis, index-alignés (``None`` si
            aucun appel n'a abouti ou si le lot est resté en texte brut).

    Returns:
        ``(textes, statuts)``.
    """
    n = len(chunks)
    prepared = _albert_prepare_or_abort(model)
    if prepared is None:
        _fill_models_out(models_out, [None] * n)
        return list(chunks), ["fallback_raw"] * n
    client, model_id = prepared
    params = _albert_decode_params(recode_cfg, model_id, temperature=temperature, max_tokens=max_tokens)
    texts, statuses, served = _albert_recode_core(chunks, instructions, client, model_id, params)
    _fill_models_out(models_out, [m or model_id for m in served])
    return texts, statuses


def _albert_recode_batch_cached(raw_batch, instructions, model, cfg, models_out):
    """Branche Albert de ``recode_batch_cached`` (cache ou recodage direct).

    Clé : fournisseur ``albert``, modèle ``albert/<id résolu>`` (jamais l'alias),
    ``decode_params`` Albert. Seuls les ``recoded`` sont mis en cache ; une réponse
    servie par un modèle de repli (503) est rangée sous ``albert/<id servi>``.
    """
    n = len(raw_batch)
    if not cfg.cache_enabled:
        return _albert_recode_batch(raw_batch, instructions, model, recode_cfg=cfg, models_out=models_out)
    prepared = _albert_prepare_or_abort(model)
    if prepared is None:
        _fill_models_out(models_out, [None] * n)
        return list(raw_batch), ["fallback_raw"] * n
    client, model_id = prepared
    params = _albert_decode_params(
        cfg, model_id, temperature=_RECODE_DEFAULT_TEMPERATURE, max_tokens=_RECODE_DEFAULT_MAX_TOKENS
    )
    pv = rad_recode_cache.prompt_fingerprint(_RECODE_SYSTEM, _RECODE_TEMPLATE, instructions)
    dp = _albert_decode_params_json(params, model_id, client.cfg)

    def _key(raw, served_id):
        """Clé de cache du texte brut ``raw`` pour le modèle ``served_id`` (``None`` si non cacheable)."""
        if not rad_dedup.normalize_text_for_hash(raw):
            return None
        return rad_recode_cache.recode_key(
            rad_dedup.content_hash(raw), f"{ALBERT_PREFIX}{served_id}", PROVIDER_ALBERT, pv, dp
        )

    keys = [_key(raw, model_id) for raw in raw_batch]
    texts = [None] * n
    statuses = [None] * n
    served_ids = [model_id] * n
    miss_idx = []
    for i, key in enumerate(keys):
        hit = rad_recode_cache.get_recode(cfg, key) if key else None
        if hit is not None:
            texts[i], statuses[i] = hit, "cached"
        else:
            miss_idx.append(i)

    if miss_idx:
        sub_texts, sub_statuses, sub_served = _albert_recode_core(
            [raw_batch[i] for i in miss_idx], instructions, client, model_id, params
        )
        for j, i in enumerate(miss_idx):
            txt, st, srv = sub_texts[j], sub_statuses[j], sub_served[j] or model_id
            statuses[i] = st
            served_ids[i] = srv
            key = keys[i] if srv == model_id else _key(raw_batch[i], srv)
            if st == "recoded" and key:
                txt = rad_recode_cache.put_recode(cfg, key, txt)
            texts[i] = txt

    _fill_models_out(models_out, served_ids)
    return texts, statuses


def _albert_known_id(model):
    """Id épinglé d'un modèle ``albert/…`` s'il est déjà connu du client, sans requête."""
    wire = str(model or "")[len(ALBERT_PREFIX):].strip()
    client = _ALBERT_CLIENT
    if client is None or not wire:
        return wire
    try:
        return client.wire_model(wire)
    except Exception:
        return wire


def _albert_recode_model_label(served_id, model):
    """Valeur du champ ``recode_model`` pour Albert : ``albert/<id servi>`` (ou id connu du modèle)."""
    return f"{ALBERT_PREFIX}{served_id or _albert_known_id(model)}"


def albert_recode_startup_exception(model, recode_cfg=None):
    """Erreur qui empêche le recodage Albert sélectionné de démarrer, sinon ``None``.

    Contrôles locaux, sans appel réseau : Albert activé, modèle non vide après
    ``albert/``, URL de base admise, clé présente. ``None`` aussi quand le modèle
    effectif n'est pas un modèle Albert (aucun import du socle Albert alors).

    Args:
        model: modèle de ``--model``.
        recode_cfg: ``RecodeConfig`` (défaut : lu dans l'environnement).

    Returns:
        ``AlbertDisabledError`` (Albert désactivé), ``ValueError`` (modèle vide
        ou URL de base refusée), ``AlbertMissingKeyError`` (clé absente, erreur
        de compte) ou ``None``.
    """
    cfg = recode_cfg if recode_cfg is not None else rad_recode_cache.RecodeConfig.from_env()
    eff_model = effective_recode_model(model, cfg)
    if not _selects_albert(eff_model):
        return None
    try:
        resolve_llm_provider(eff_model, albert_enabled=_albert_enabled())
        _albert_config()
    except ValueError as exc:
        return exc
    if not ALBERT_API_KEY:
        client_mod = _albert_module("client")
        return client_mod.AlbertMissingKeyError(client_mod.MISSING_KEY_MESSAGE)
    return None


def albert_recode_startup_error(model, recode_cfg=None):
    """Message d'erreur si le recodage Albert sélectionné ne peut pas démarrer, sinon ``None``.

    Texte de ``albert_recode_startup_exception`` (mêmes contrôles locaux, sans
    appel réseau).

    Args:
        model: modèle de ``--model``.
        recode_cfg: ``RecodeConfig`` (défaut : lu dans l'environnement).
    """
    exc = albert_recode_startup_exception(model, recode_cfg)
    return None if exc is None else str(exc)


# Ligne de sortie stable émise juste avant chaque ``exit(2)`` Albert de la CLI
# (contrat lu par les routes et le runner Celery, sur stdout, ancrée en début de
# ligne) : ``Albert abort: kind=<kind> reason=<reason>`` suivie, pour une erreur
# de compte seulement, de `` credential_required=albert_api_key``.
ALBERT_ABORT_MARKER = "Albert abort:"
_ALBERT_MODEL_REASONS = ("not_found", "validation")


def albert_abort_marker(exc):
    """Ligne ``Albert abort: …`` qui qualifie l'arrêt Albert ``exc`` pour l'appelant du sous-processus.

    Motif lisible par ``^Albert abort: kind=(\\w+) reason=(\\w+)(?: credential_required=(\\w+))?$`` :

    * ``kind=account`` : ``AlbertAuthError`` (clé absente ou refusée, compte
      expiré, budget ou quota épuisé), avec ``credential_required=albert_api_key``
      → événement SSE ``{type: error, credential_required}`` ;
    * ``kind=model`` : modèle absent de ``/v1/models`` ou 404
      (``reason=not_found``), type de modèle incompatible (``reason=validation``) ;
    * ``kind=request`` : autre erreur permanente (``bad_request``, ``no_access``…) ;
    * ``kind=service`` : service injoignable au démarrage (erreur transitoire
      épuisée) ou autre erreur de l'API ;
    * ``kind=config`` : Albert désactivé (``reason=disabled``), modèle vide ou
      URL de base refusée (``reason=invalid_config``) ;
    * ``kind=error`` : toute autre exception (``reason=unexpected``).

    Seuls ``kind=account`` portent ``credential_required`` : une erreur de modèle
    ou de configuration n'est jamais présentée comme une clé invalide.

    Args:
        exc: erreur qui arrête le recodage (arrêt mémorisé, preflight, démarrage).

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


def _albert_explicit_preflight_result(result, wire, model_id, day):
    """Preflight d'un modèle ``albert/…`` explicite autre que le primaire du rôle ``recode``.

    Complète un preflight de compte seul (``roles=()`` : ``/v1/me`` et
    ``/v1/models``) : la chaîne du rôle ne sert plus qu'aux replis sur 503
    répétés, donc un de ses modèles absent de ``/v1/models`` n'est qu'un
    avertissement (jamais une erreur). Le résultat renvoyé est une copie : le
    résultat mis en cache par le preflight n'est pas modifié.

    Args:
        result: ``PreflightResult`` du preflight de compte.
        wire: nom demandé (après ``albert/``).
        model_id: id épinglé résolu du modèle demandé.
        day: date de référence de la chaîne de repli.

    Returns:
        ``PreflightResult`` du rôle ``recode`` (modèle demandé en tête de chaîne).
    """
    catalog = _albert_module("catalog")
    names = result.names or {}
    chain = [m for m in catalog.fallback_chain("recode", today=day) if m != model_id]
    available, missing = [], []
    for name in chain:
        ident = names.get(name) or names.get(name.lower())
        if not ident:
            missing.append(name)
        elif ident != model_id and ident not in available:
            available.append(ident)
    warnings = list(result.warnings)
    added = [f"Rôle Albert recode : modèle de repli {name} absent de /v1/models." for name in missing]
    added += [w for w in catalog.lifecycle_warnings([model_id] + available, today=day)
              if w not in warnings and w not in added]
    for message in added:
        logging.getLogger(__name__).warning("%s", message)
    resolved = dict(result.resolved)
    resolved[wire] = model_id
    return dataclasses.replace(
        result,
        roles=("recode",),
        requested={"recode": [wire] + chain},
        chains={"recode": [model_id] + available},
        resolved=resolved,
        unavailable={"recode": missing} if missing else {},
        warnings=warnings + added,
    )


def run_albert_recode_preflight(model, recode_cfg=None, *, today=None):
    """Preflight Albert du recodage au début de la phase initial.

    ``/v1/me`` (compte expiré → erreur) et ``/v1/models`` (ids mémorisés par le
    client), puis résolution du modèle demandé : un id épinglé absent ou de type
    incompatible est une erreur explicite, sans repli. Si le modèle demandé est
    le primaire du rôle ``recode``, toute la chaîne du rôle est contrôlée par le
    preflight (primaire absent = erreur) ; pour un autre modèle explicite, seul
    le compte est contrôlé par le preflight et les modèles absents de la chaîne
    (replis sur 503) ne sont que des avertissements. Sans effet (``None``)
    quand le modèle effectif n'est pas un modèle Albert.

    Args:
        model: modèle de ``--model``.
        recode_cfg: ``RecodeConfig`` (défaut : lu dans l'environnement).
        today: date de référence des chaînes de repli (défaut : aujourd'hui).

    Returns:
        ``PreflightResult`` ou ``None``.

    Raises:
        AlbertDisabledError, ValueError: configuration refusée.
        AlbertError: compte, modèle ou service indisponible.
    """
    cfg = recode_cfg if recode_cfg is not None else rad_recode_cache.RecodeConfig.from_env()
    eff_model = effective_recode_model(model, cfg)
    if not _selects_albert(eff_model):
        return None
    wire = resolve_llm_provider(eff_model, albert_enabled=_albert_enabled()).wire_model
    day = today or datetime.date.today()
    catalog = _albert_module("catalog")
    client = _get_albert_client()
    role_primary = catalog.primary_model("recode", today=day)
    explicit = role_primary is None or catalog.canonical_id(wire) != role_primary
    result = _albert_module("preflight").run_preflight(
        client, roles=() if explicit else ("recode",), today=day
    )
    with _ALBERT_RESOLVE_LOCK:
        model_id = client.resolve_model_id(wire)
    catalog.check_endpoint_type(model_id, "chat")
    if explicit and not result.skipped:
        result = _albert_explicit_preflight_result(result, wire, model_id, day)
    return result


def report_albert_usage(output_dir):
    """Écrit ``albert_usage.jsonl`` et affiche la synthèse, seulement si Albert a été appelé.

    Args:
        output_dir: dossier de sortie de la session.

    Returns:
        Chemin du journal écrit, ou ``None`` (Albert non appelé, journal désactivé
        par ``ALBERT_USAGE_LOG=0`` ou erreur d'écriture).
    """
    client = _ALBERT_CLIENT
    if client is None or not client.ledger.called:
        return None
    path = None
    if client.cfg.usage_log:
        try:
            path = client.ledger.write_jsonl(output_dir)
        except OSError as exc:
            print(f"Avertissement : journal d'usage Albert non écrit ({exc}).")
    print(client.ledger.summary_line())
    if path:
        print(f"Journal d'usage Albert : {path}")
    return path


def gpt_recode_batch(chunks, instructions, model="gpt-4o-mini", temperature=0.3, max_tokens=8000, recode_cfg=None,
                     models_out=None):
    """
    Recoder un lot de textes selon des instructions précises en parallèle,
    puis retenter séquentiellement en cas d'erreur.

    Args:
        model: Nom du modèle (ex: "gpt-4o-mini" pour OpenAI, "google/gemini-2.5-flash" pour OpenRouter,
               "albert/ministral-3-8b-instruct-2512" pour Albert). Préfixe ``albert/`` → Albert
               (``_albert_recode_batch``, sans 2ᵉ passage ni repli OpenAI) ; sinon, si le modèle
               contient "/" → OpenRouter, sinon OpenAI.
        recode_cfg: RecodeConfig optionnel (Lot 7). Si ``harden_enabled``, force
               temp/top_p/seed/max_tokens/modèle snapshot et le routage OpenAI vs
               OpenRouter pinné. Sinon comportement historique (temp 0.3).
        models_out: liste optionnelle remplie, pour Albert seulement, des ids de
               modèle servis (index-alignés) ; inchangée pour OpenAI/OpenRouter.

    Returns:
        tuple[list[str], list[str]]: ``(recoded_texts, statuses)`` index-alignés à
        ``chunks``. ``status`` ∈ {``'recoded'`` (succès finish_reason=stop, non vide),
        ``'fallback_truncated'`` (length/content_filter → texte RAW), ``'fallback_raw'``
        (exception/vide → texte RAW)}. Les deux fallbacks NE sont jamais cachés ni
        (par contrat Lot 7.d) upsertés.
    """
    # Source unique du modèle effectif (routage et clé de cache).
    eff_model = effective_recode_model(model, recode_cfg)
    if _selects_albert(eff_model):
        return _albert_recode_batch(
            chunks, instructions, eff_model, recode_cfg=recode_cfg,
            temperature=temperature, max_tokens=max_tokens, models_out=models_out,
        )

    # --- Durcissement décodage optionnel (Lot 7.a) ---
    top_p = None
    seed = None
    if recode_cfg is not None and recode_cfg.harden_enabled:
        model = eff_model  # == recode_cfg.model
        temperature = recode_cfg.temperature
        max_tokens = recode_cfg.max_tokens
        top_p = recode_cfg.top_p
        seed = recode_cfg.seed
        # OpenAI direct (seed honoré) sauf si OpenRouter explicitement préféré ET pinné.
        if recode_cfg.prefer_openai or not recode_cfg.openrouter_provider:
            use_openrouter = False
        else:
            use_openrouter = (legacy_provider(model) == PROVIDER_OPENROUTER) and (openrouter_client is not None)
    else:
        # OpenRouter models have format "provider/model"
        use_openrouter = legacy_provider(model) == PROVIDER_OPENROUTER

    active_client = openrouter_client if (use_openrouter and openrouter_client) else client

    if use_openrouter and not openrouter_client:
        print(f"Warning: OpenRouter model '{model}' requested but OpenRouter client not initialized.")
        print("Falling back to OpenAI gpt-4o-mini")
        model = "gpt-4o-mini"
        active_client = client
        use_openrouter = False

    print(f"Using {'OpenRouter' if (use_openrouter and openrouter_client) else 'OpenAI'} with model: {model}")

    def _create(msgs, mdl=model):
        kwargs = {"model": mdl, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None and not use_openrouter:  # seed honoré côté OpenAI seul
            kwargs["seed"] = seed
        if use_openrouter and recode_cfg is not None and recode_cfg.openrouter_provider:
            kwargs["extra_body"] = {"provider": {"order": [recode_cfg.openrouter_provider], "allow_fallbacks": False}}
        return active_client.chat.completions.create(**kwargs)

    def _extract(resp):
        """Renvoie (text|None, status). Truncation/échec → texte RAW en aval."""
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        if finish in ("length", "content_filter"):
            return None, "fallback_truncated"  # jamais caché (Lot 7.c), RAW stocké
        if not text:
            return None, "fallback_raw"
        return text, "recoded"

    messages_list = []
    for chunk in chunks:
        prompt = _RECODE_TEMPLATE.format(instructions=instructions, chunk=chunk)
        messages_list.append([
            {"role": "system", "content": _RECODE_SYSTEM},
            {"role": "user",   "content": prompt}
        ])

    recoded = [None] * len(chunks)
    statuses = [None] * len(chunks)
    # Use DEFAULT_MAX_WORKERS for the ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = {executor.submit(_create, msgs): idx for idx, msgs in enumerate(messages_list)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                recoded[idx], statuses[idx] = _extract(future.result())
            except Exception as e:
                print(f"Erreur chunk #{idx+1} (1ʳᵉ passe) : {e}")
                statuses[idx] = None  # retry en 2ᵉ passe

    # Deuxième passe séquentielle pour les chunks ayant levé en 1ʳᵉ passe (status None).
    # Une troncature (fallback_truncated) n'est PAS réessayée dans le run (retentée au
    # run suivant) — c'est un échec déterministe, pas un glitch transitoire.
    for i in range(len(chunks)):
        if statuses[i] is None:
            print(f"Tentative de 2ᵉ passe pour le chunk #{i+1}")
            try:
                time.sleep(2)  # Wait before retrying
                recoded[i], statuses[i] = _extract(_create(messages_list[i]))
                print(f"Chunk #{i+1} : 2ᵉ passe statut={statuses[i]}.")
            except Exception as e:
                print(f"Échec chunk #{i+1} après 2ᵉ passe : {e}")
                statuses[i] = "fallback_raw"

    # Matérialiser le texte RAW pour tout fallback (et garde-fou final).
    for i in range(len(chunks)):
        if statuses[i] in ("fallback_raw", "fallback_truncated") or recoded[i] is None:
            recoded[i] = chunks[i]
            if statuses[i] not in ("fallback_raw", "fallback_truncated"):
                statuses[i] = "fallback_raw"

    return recoded, statuses


def recode_batch_cached(raw_batch, instructions, model, models_out=None):
    """Wrapper cache (Lot 7.b) autour de ``gpt_recode_batch``. Renvoie
    ``(texts, statuses)`` (statuts incluant ``'cached'``).

    Clés sur le ``content_hash`` du texte BRUT (calculé inconditionnellement de
    ``DEDUP_ENABLED`` ; un chunk normalisé vide est **non-cacheable**). On ne cache
    QUE les succès (``status == 'recoded'``). HIT → 0 appel LLM, valeur canonique
    réutilisée → texte stocké byte-identique entre runs.

    Modèle effectif ``albert/…`` (``effective_recode_model``) : fournisseur
    ``albert``, modèle ``albert/<id résolu>`` dans la clé ; ``models_out`` reçoit
    alors les ids servis (index-alignés). Autres fournisseurs : inchangé.
    """
    cfg = rad_recode_cache.RecodeConfig.from_env()
    n = len(raw_batch)
    eff_model = effective_recode_model(model, cfg)
    if _selects_albert(eff_model):
        return _albert_recode_batch_cached(raw_batch, instructions, eff_model, cfg, models_out)
    if not cfg.cache_enabled:
        # Cache off : recode direct (durcissement éventuel si harden_enabled).
        return gpt_recode_batch(raw_batch, instructions, model=model, recode_cfg=cfg)

    provider = legacy_cache_provider_label(eff_model, prefer_openai=cfg.prefer_openai)
    pv = rad_recode_cache.prompt_fingerprint(_RECODE_SYSTEM, _RECODE_TEMPLATE, instructions)
    dp = cfg.decode_params_json()

    keys = []
    for raw in raw_batch:
        norm = rad_dedup.normalize_text_for_hash(raw)
        if not norm:  # normalisé vide → non-cacheable (sinon collapse de clé)
            keys.append(None)
        else:
            keys.append(rad_recode_cache.recode_key(rad_dedup.content_hash(raw), eff_model, provider, pv, dp))

    texts = [None] * n
    statuses = [None] * n
    miss_idx = []
    for i, key in enumerate(keys):
        hit = rad_recode_cache.get_recode(cfg, key) if key else None
        if hit is not None:
            texts[i], statuses[i] = hit, "cached"
        else:
            miss_idx.append(i)

    if miss_idx:
        sub_texts, sub_statuses = gpt_recode_batch(
            [raw_batch[i] for i in miss_idx], instructions, model=model, recode_cfg=cfg
        )
        for j, i in enumerate(miss_idx):
            txt, st = sub_texts[j], sub_statuses[j]
            statuses[i] = st
            if st == "recoded" and keys[i]:
                # PUT puis valeur canonique (convergence de MISS concurrents).
                txt = rad_recode_cache.put_recode(cfg, keys[i], txt)
            texts[i] = txt

    return texts, statuses

def save_raw_chunks_to_json_incrementally(chunks_to_add, json_file):
    """
    Sauvegarde les nouveaux chunks dans `json_file` de manière incrémentale et thread-safe.

    Quand ``DEDUP_ENABLED`` est actif, le merge est **dédupliqué par ``id``** : les
    ids éligibles étant adressés par contenu (``content_hash``), un même texte
    ré-ingéré ne s'empile plus (idempotence cross-run/cross-session). Les chunks
    inéligibles (ids aléatoires uniques) ne collisionnent jamais → tous conservés.
    Quand OFF : concaténation simple, sortie **byte-identique** à avant.
    """
    with SAVE_LOCK:
        existing_chunks = []
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)
            except json.JSONDecodeError:
                print(f"Fichier JSON '{json_file}' corrompu ou vide. On repart d'une liste vide.")
                existing_chunks = []

        if rad_dedup.DedupConfig.from_env().enabled:
            merged_chunks = []
            seen_ids = set()
            for chunk in existing_chunks + chunks_to_add:
                cid = chunk.get("id")
                # 1re occurrence gagne (existing avant nouveaux) ; idempotent au ré-ingest.
                if cid is not None and cid in seen_ids:
                    continue
                if cid is not None:
                    seen_ids.add(cid)
                merged_chunks.append(chunk)
        else:
            merged_chunks = existing_chunks + chunks_to_add

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(merged_chunks, f, ensure_ascii=False, indent=2)

def process_document_chunks(row_data, json_file=DEFAULT_JSON_FILE_CHUNKS, model="gpt-4o-mini"):
    """
    Traite un document (représenté par row_data, ex: une ligne de DataFrame).
    1. Extraction et nettoyage du texte (implicite par TEXT_SPLITTER)
    2. Découpage en chunks avec TEXT_SPLITTER
    3. Recodage par batch avec gpt_recode_batch
    4. Sauvegarde des chunks avec save_raw_chunks_to_json_incrementally

    Args:
        model: Modèle LLM pour le recodage (ex: "gpt-4o-mini" ou "google/gemini-2.5-flash")
    """
    if TEXT_SPLITTER is None:
        print("Erreur: TEXT_SPLITTER n'est pas initialisé. Impossible de traiter le document.")
        return []

    text = row_data.get("texteocr", "").strip()
    if not text:
        print(f"Document ignoré (texte vide) : {row_data.get('filename', 'Nom de fichier inconnu')}")
        return []

    provider_raw = row_data.get("texteocr_provider", "")
    if isinstance(provider_raw, float) and pd.isna(provider_raw):
        provider_raw = ""
    provider = str(provider_raw).strip().lower()
    # Skip recodage GPT si l'OCR produit déjà du Markdown propre (Mistral, OCR
    # local Docling/MinerU/Marker) ou source CSV déjà propre.
    recode_required = provider not in RECODE_SKIP_PROVIDERS

    # Lot 1 — config dédup lue par document (respecte un DEDUP_ENABLED posé avant
    # l'appel). Quand OFF : aucun champ content_hash écrit, id aléatoire conservé
    # → JSON byte-identique à avant.
    dedup_cfg = rad_dedup.DedupConfig.from_env()
    # Lot 7 — observabilité recodage : on ne pose le champ recode_status QUE si le
    # cache ou le durcissement est actif (sinon JSON byte-identique à avant).
    recode_cfg = rad_recode_cache.RecodeConfig.from_env()
    # Albert : recode_status et recode_model=albert/<id servi> TOUJOURS écrits
    # (reproductibilité), même sans cache ni durcissement.
    eff_model = effective_recode_model(model, recode_cfg)
    albert_recode = _selects_albert(eff_model)
    _emit_recode_status = recode_cfg.cache_enabled or recode_cfg.harden_enabled or albert_recode

    # doc_id reste ALÉATOIRE (traçabilité au journal uniquement) ; il n'est JAMAIS
    # une clé de dédup. La clé est l'id adressé par contenu (cf. boucle ci-dessous).
    doc_id = str(random.randint(10**11, 10**12 - 1))
    text_chunks = TEXT_SPLITTER.split_text(text)
    
    filename = row_data.get('filename', f'doc_{doc_id}')
    print(f"Traitement de '{filename}': {len(text_chunks)} chunks bruts générés.")

    all_processed_chunks = []
    for start_index in range(0, len(text_chunks), DEFAULT_BATCH_SIZE_GPT):
        batch_to_recode = text_chunks[start_index : start_index + DEFAULT_BATCH_SIZE_GPT]
        total_batches = ((len(text_chunks) - 1) // DEFAULT_BATCH_SIZE_GPT) + 1
        batch_models = []  # Albert seulement : ids servis, index-alignés

        if recode_required:
            print(f"  Lot {start_index // DEFAULT_BATCH_SIZE_GPT + 1} / {total_batches} en recodage...")
            # Lot 7 : passe par le wrapper cache (no-op si RECODE_CACHE_ENABLED=0 →
            # recode direct, durcissement éventuel). Renvoie (textes, statuts).
            if albert_recode:
                cleaned_batch, recode_statuses = recode_batch_cached(
                    batch_to_recode,
                    instructions=_RECODE_INSTRUCTIONS,
                    model=model,
                    models_out=batch_models,
                )
            else:
                cleaned_batch, recode_statuses = recode_batch_cached(
                    batch_to_recode,
                    instructions=_RECODE_INSTRUCTIONS,
                    model=model,
                )
        else:
            if start_index == 0:
                print("  OCR Mistral détecté → recodage GPT sauté (chunks utilisés tels quels).")
            cleaned_batch = batch_to_recode
            recode_statuses = ["skipped"] * len(batch_to_recode)

        for i, cleaned_text in enumerate(cleaned_batch):
            original_chunk_index = start_index + i + 1
            # Texte BRUT pré-recodage (clé de hash) : aligné index-à-index avec
            # cleaned_batch. Repli défensif si gpt_recode_batch désalignait la liste.
            raw_chunk_text = batch_to_recode[i] if i < len(batch_to_recode) else cleaned_text

            # Sanitize metadata values, especially for NaN or other non-JSON-friendly types
            # Convert pandas.NA or numpy.nan to empty strings or None
            # Pinecone metadata values should be string, number, boolean, or list of strings.

            def sanitize_metadata_value(value, default=""):
                if pd.isna(value):
                    return default
                # Ensure it's a basic type suitable for JSON and Pinecone metadata
                if isinstance(value, (str, int, float, bool)):
                    return value
                return str(value) # Fallback to string representation

            # Construction dynamique des métadonnées
            # Injecte TOUTES les colonnes du row_data (compatibilité CSV)
            chunk_metadata = {
                "id":           f"{doc_id}_{original_chunk_index}",
                "doc_id":       doc_id,
                "chunk_index":  original_chunk_index,
                "total_chunks": len(text_chunks),
                "text":         cleaned_text,
            }

            # Injecter toutes les métadonnées source (sauf texteocr qui est dans "text")
            for key, value in row_data.items():
                # Exclure les champs déjà gérés ou trop volumineux (content_hash /
                # dedup_eligible protégés : ils sont posés par la dédup ci-dessous).
                if key not in ("texteocr", "text", "id", "doc_id", "chunk_index",
                               "total_chunks", "content_hash", "dedup_eligible"):
                    chunk_metadata[key] = sanitize_metadata_value(value, "")

            # S'assurer que ocr_provider est présent (backward compatibility)
            if "ocr_provider" not in chunk_metadata and "texteocr_provider" not in chunk_metadata:
                chunk_metadata["ocr_provider"] = sanitize_metadata_value(provider, "")

            # Lot 1 — Tier 0 : content_hash sur le texte BRUT pré-recodage. Raison
            # décisive : gpt_recode_batch tourne à temperature>0 → texte recodé NON
            # déterministe ; hacher le brut couvre TOUT le corpus (recodé inclus).
            # Posé en DERNIER pour ne pas être écrasé par l'injection row_data.
            if dedup_cfg.enabled:
                chash, eligible, cid = rad_dedup.compute_dedup_fields(
                    raw_chunk_text, original_chunk_index, dedup_cfg.min_chars
                )
                chunk_metadata["content_hash"] = chash
                chunk_metadata["dedup_eligible"] = eligible
                # ID adressé par contenu SEULEMENT si éligible : un chunk inéligible
                # (court : '', 'Références', bruit OCR) garde son id aléatoire unique
                # pour éviter une collision inter-documents au même chunk_index.
                if eligible:
                    chunk_metadata["id"] = cid

            # Lot 7.f — statut de recodage (cached|recoded|fallback_raw|
            # fallback_truncated|skipped) pour l'observabilité + le contrat 7.d
            # (les fallbacks ne sont pas upsertés). Gated → byte-identique quand OFF.
            if _emit_recode_status:
                chunk_metadata["recode_status"] = (
                    recode_statuses[i] if i < len(recode_statuses) else "recoded"
                )
                if albert_recode:
                    chunk_metadata["recode_model"] = _albert_recode_model_label(
                        batch_models[i] if i < len(batch_models) else None, eff_model
                    )
                else:
                    chunk_metadata["recode_model"] = recode_cfg.model if recode_cfg.harden_enabled else model

            all_processed_chunks.append(chunk_metadata)

    if all_processed_chunks:
        save_raw_chunks_to_json_incrementally(all_processed_chunks, json_file)
        print(f"→ {len(all_processed_chunks)} chunks traités et sauvegardés pour le document '{filename}' (doc_id={doc_id}) dans '{json_file}'.")
        # Ajout d'un log pour chaque chunk individuel ajouté (peut être verbeux)
        # for idx, chunk_data in enumerate(all_processed_chunks):
        #     print(f"    Chunk {idx+1}/{len(all_processed_chunks)} (ID: {chunk_data.get('id')}) ajouté au fichier JSON.")
    
    return all_processed_chunks

def process_all_documents(df, json_file=DEFAULT_JSON_FILE_CHUNKS, model="gpt-4o-mini"):
    """
    Lance le traitement de tous les documents d'un DataFrame en parallèle.
    Utilise un nombre limité de workers pour process_document_chunks pour éviter la surcharge API.

    Args:
        model: Modèle LLM pour le recodage (ex: "gpt-4o-mini" ou "google/gemini-2.5-flash")
    """
    # Max 3 documents processed in parallel for their chunking/recoding stages
    num_doc_workers = min(DEFAULT_DOC_WORKERS, DEFAULT_MAX_WORKERS)

    total_docs = len(df)
    total_chunks_generated = 0
    chunking_start_time = time.time()

    # Emit init event for SSE progress tracking
    print(f"PROGRESS|init|{total_docs}|Found {total_docs} documents to chunk", flush=True)

    with ThreadPoolExecutor(max_workers=num_doc_workers) as executor:
        # df.iterrows() returns (index, Series)
        futures = {
            executor.submit(process_document_chunks, row_series, json_file, model): idx
            for idx, row_series in df.iterrows()
        }

        completed_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Traitement des Documents (Chunking)"):
            doc_idx = futures[future]
            completed_count += 1
            try:
                result_chunks = future.result()
                chunk_count = len(result_chunks) if result_chunks else 0
                total_chunks_generated += chunk_count
                # Emit row-level progress
                print(f"PROGRESS|row|{completed_count}/{total_docs}|Document #{doc_idx}: {chunk_count} chunks", flush=True)
                print(f"Document #{doc_idx} traité, {chunk_count} chunks produits.")
            except Exception as e:
                print(f"PROGRESS|row|{completed_count}/{total_docs}|Document #{doc_idx}: error", flush=True)
                print(f"Erreur lors du traitement du document #{doc_idx}: {e}")
                # Track error
                if METRICS_AVAILABLE and track_error:
                    track_error('chunking', type(e).__name__)

    # Record chunking metrics
    chunking_elapsed = time.time() - chunking_start_time
    if METRICS_AVAILABLE and metrics_collector:
        try:
            metrics_collector.increment_counter(
                'chunks_generated',
                total_chunks_generated,
                phase='initial'
            )
            metrics_collector.observe_histogram(
                'chunking_duration',
                chunking_elapsed
            )
        except Exception:
            pass  # Don't fail on metrics errors

    logging.info(f"Chunking complete: {total_chunks_generated} chunks from {total_docs} docs in {chunking_elapsed:.1f}s")

# ----------------------------------------------------------------------
# PART 2: Chunk Embedding (Dense)
# ----------------------------------------------------------------------

# Espace d'embeddings (EMBEDDING_PROVIDER) : OpenAI par défaut, chemin historique
# inchangé à l'octet ; ``albert`` = bge-m3 (1024 dimensions) servi par Albert,
# espace séparé, sans jamais de repli vers OpenAI ni de vecteur nul.
def _embedding_provider_requested(env=None):
    """Valeur normalisée de ``EMBEDDING_PROVIDER`` (``''`` si absente), sans validation."""
    source = os.environ if env is None else env
    return (source.get("EMBEDDING_PROVIDER") or "").strip().lower()


def resolve_embedding_config(env=None):
    """Configuration d'embeddings du run (``EmbeddingConfig.from_env``).

    L'interrupteur Albert est celui du module (``ALBERT_ENABLED``, lu à l'import
    et patchable, décision 11) : il remplace la valeur de ``ALBERT_ENABLED`` de
    l'environnement lu, comme pour le recodage (``_albert_enabled``).

    Args:
        env: environnement lu (défaut : ``os.environ``), jamais modifié.

    Returns:
        ``EmbeddingConfig`` (espace OpenAI par défaut si ``EMBEDDING_PROVIDER``
        est vide, absent ou ``openai``).

    Raises:
        AlbertDisabledError: ``albert`` demandé alors qu'Albert est désactivé.
        ValueError: valeur inconnue, modèle Albert non pris en charge ou URL de
            base Albert refusée.
    """
    source = os.environ if env is None else env
    overlay = dict(source)
    overlay["ALBERT_ENABLED"] = "1" if _albert_enabled() else "0"
    return EmbeddingConfig.from_env(overlay)


def albert_embed_startup_exception():
    """Erreur qui empêche la phase dense de démarrer avec l'espace demandé, sinon ``None``.

    Contrôles locaux, sans appel réseau : ``EMBEDDING_PROVIDER`` connu, Albert
    activé pour ``albert``, URL de base admise et clé présente. ``None`` pour
    l'espace OpenAI (le socle Albert n'est alors pas chargé).

    Returns:
        ``AlbertDisabledError`` ou ``ValueError`` (configuration refusée),
        ``AlbertMissingKeyError`` (clé Albert absente, erreur de compte) ou ``None``.
    """
    try:
        cfg = resolve_embedding_config()
    except ValueError as exc:
        return exc
    if cfg.space.provider != PROVIDER_ALBERT:
        return None
    try:
        _albert_config()
    except ValueError as exc:
        return exc
    if not ALBERT_API_KEY:
        client_mod = _albert_module("client")
        return client_mod.AlbertMissingKeyError(client_mod.MISSING_KEY_MESSAGE)
    return None


def abort_embeddings(exc, code=2):
    """Annonce l'arrêt de la phase dense puis lève ``SystemExit(code)``.

    Affiche un message français et, si Albert est en cause
    (``EMBEDDING_PROVIDER=albert`` ou erreur Albert), la ligne stable
    ``albert_abort_marker`` lue par les routes et le runner Celery. Une valeur
    inconnue de ``EMBEDDING_PROVIDER`` n'émet pas de ligne Albert.

    Args:
        exc: erreur de configuration, de compte ou de service.
        code: code de sortie (2 par défaut).

    Raises:
        SystemExit: toujours.
    """
    message = f"Erreur critique (embeddings) : {exc} Arrêt."
    print(message)
    logging.error(message)
    albert_related = (
        _embedding_provider_requested() == PROVIDER_ALBERT
        or isinstance(exc, (_rad_albert.AlbertError, _rad_albert.AlbertDisabledError))
    )
    if albert_related:
        print(albert_abort_marker(exc))
    raise SystemExit(code)


def run_albert_embed_preflight(space, *, today=None):
    """Preflight Albert du rôle ``embed`` avant la phase dense.

    ``/v1/me`` (compte expiré → erreur) puis ``/v1/models`` : le modèle
    d'embedding épinglé (``ALBERT_EMBED_MODEL``, ``bge-m3``) doit exister et être
    de type embeddings ; les ids résolus sont mémorisés par le client, qui
    envoie ensuite l'id épinglé (jamais un alias). Aucun repli de modèle.

    Args:
        space: espace Albert du run.
        today: date de référence (défaut : aujourd'hui).

    Returns:
        ``PreflightResult`` (``skipped`` si ``ALBERT_PREFLIGHT=0``).

    Raises:
        AlbertAuthError: clé absente ou refusée, compte expiré, quota épuisé.
        AlbertPermanentError: modèle absent de ``/v1/models`` ou de type incompatible.
        AlbertTransientError: service injoignable après réessais.
        ValueError: URL de base refusée.
    """
    client = _get_albert_client()
    day = today or datetime.date.today()
    result = _albert_module("preflight").run_preflight(client, roles=("embed",), today=day)
    if not result.skipped:
        catalog = _albert_module("catalog")
        served = result.primary("embed")
        dim = catalog.embedding_dim(served) if served else None
        if dim is not None and int(dim) != int(space.dim):
            raise _rad_albert.AlbertPermanentError(
                reason="validation",
                endpoint="/v1/models",
                detail=f"modèle d'embedding {served} en {dim} dimensions, espace attendu {space.dim}",
            )
    return result


def _start_albert_embeddings(space):
    """Configuration, clé et preflight Albert de la phase dense ; ``exit 2`` sinon.

    Returns:
        ``AlbertConfig`` du run (ratio de manques toléré, concurrence).

    Raises:
        SystemExit: code 2 (configuration refusée, erreur de compte, modèle ou
            service indisponible), après la ligne ``albert_abort_marker``.
    """
    try:
        albert_cfg = _albert_config()
        result = run_albert_embed_preflight(space)
    except (_rad_albert.AlbertError, ValueError) as exc:
        abort_embeddings(exc, code=2)
    print(result.summary_line())
    return albert_cfg


def _albert_set_embed_abort(exc):
    """Mémorise la première erreur de compte des embeddings Albert et l'annonce une fois.

    Les lots suivants restent sans vecteur, sans appel ; la phase dense sort en 2
    après l'écriture du fichier (``_finish_albert_embeddings``).
    """
    global _ALBERT_EMBED_ABORT
    with _ALBERT_EMBED_ABORT_LOCK:
        first = _ALBERT_EMBED_ABORT is None
        if first:
            _ALBERT_EMBED_ABORT = exc
    if first:
        print(f"Albert : embeddings arrêtés — {exc} Lots restants sans vecteur, sans appel.")


def _valid_space_vector(vec, space):
    """Vrai si ``vec`` est un vecteur exploitable de l'espace ``space``.

    Liste de nombres de la dimension de l'espace, non entièrement nulle : un
    vecteur nul ou de mauvaise dimension n'est jamais accepté hors espace par
    défaut.
    """
    if not isinstance(vec, (list, tuple)) or len(vec) != space.dim:
        return False
    try:
        return any(float(x) != 0.0 for x in vec)
    except (TypeError, ValueError):
        return False


def _is_blank_chunk_text(chunk):
    """Vrai si le texte du chunk (champ ``text``) est absent, vide ou blanc.

    Même règle que ``_albert_embed_batch`` : un tel texte n'est jamais envoyé
    aux embeddings Albert. ``rad_vectordb._is_blank_chunk_text`` applique la
    même règle côté envoi.
    """
    text = chunk.get("text") if isinstance(chunk, dict) else None
    return not (isinstance(text, str) and text.strip())


def _albert_embed_batch(texts, space):
    """Embeddings Albert d'un lot de textes : un vecteur par texte, ``None`` sinon.

    Les textes vides ne sont jamais envoyés (``None``). ``AlbertClient.embed``
    découpe en tranches de 64 au plus, trie par index, contrôle la dimension et
    normalise (D12) ; l'id épinglé ``bge-m3`` est envoyé. Un échec donne
    ``None`` pour les textes du lot : jamais de vecteur nul, jamais d'appel
    OpenAI. Une erreur de compte ou de quota arrête tous les lots suivants.

    Args:
        texts: textes du lot.
        space: espace Albert (modèle et dimension attendus).

    Returns:
        Liste de même longueur que ``texts`` : vecteurs ou ``None``.
    """
    out = [None] * len(texts)
    positions = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
    if not positions or _ALBERT_EMBED_ABORT is not None:
        return out
    try:
        vectors = _get_albert_client().embed([texts[i] for i in positions], model=space.model)
    except _rad_albert.AlbertAuthError as exc:
        _albert_set_embed_abort(exc)
        return out
    except (_rad_albert.AlbertError, ValueError) as exc:
        logging.error(f"Embeddings Albert en échec pour un lot de {len(positions)} texte(s) : {exc}")
        return out
    for j, i in enumerate(positions):
        vec = vectors[j] if j < len(vectors) else None
        if _valid_space_vector(vec, space):
            out[i] = vec
    return out


def get_embeddings_batch(texts, model="text-embedding-3-large", retry_count=0, max_retries=3, space=None):
    """
    Generate embeddings avec retry exponentiel et adaptive batching.

    Cette fonction calcule les embeddings pour un lot de textes avec une gestion
    robuste des erreurs incluant:
    - Retry avec backoff exponentiel en cas de rate limit
    - Adaptive batching (split du batch si échecs répétés)
    - Fallback individuel en cas d'erreur persistante

    Args:
        texts: Liste de textes pour lesquels générer les embeddings.
        model: Modèle d'embedding OpenAI à utiliser.
        retry_count: Compteur de tentatives (usage interne pour récursion).
        max_retries: Nombre maximum de tentatives avant échec.
        space: espace d'embeddings. ``None`` ou espace par défaut : chemin OpenAI
            historique, strictement inchangé. Espace Albert : ``AlbertClient.embed``
            (``_albert_embed_batch``), ``None`` pour un texte vide ou en échec,
            jamais de vecteur nul ni d'appel OpenAI.

    Returns:
        Liste d'embeddings (vecteurs) correspondant aux textes d'entrée.
        En cas d'échec total, retourne des vecteurs nuls de dimension 3072
        (chemin OpenAI seulement).
    """
    if space is not None and space.provider == PROVIDER_ALBERT:
        return _albert_embed_batch(texts, space)
    batch_size = len(texts)

    try:
        response = client.embeddings.create(
            input=texts,
            model=model,
            timeout=60.0  # Timeout explicite
        )
        return [item.embedding for item in response.data]

    except RateLimitError as e:
        if retry_count >= max_retries:
            logging.error(f"Rate limit after {max_retries} retries, batch size {batch_size}")
            raise

        # Exponential backoff
        wait_time = 2 ** retry_count
        logging.warning(f"Rate limit hit, waiting {wait_time}s (retry {retry_count + 1}/{max_retries})")
        time.sleep(wait_time)

        # Réduire batch size si échec répété
        if retry_count > 0 and batch_size > 10:
            # Split batch en deux
            mid = batch_size // 2
            logging.info(f"Splitting batch {batch_size} → {mid} + {batch_size - mid}")

            emb1 = get_embeddings_batch(texts[:mid], model, retry_count + 1, max_retries)
            emb2 = get_embeddings_batch(texts[mid:], model, retry_count + 1, max_retries)
            return emb1 + emb2
        else:
            return get_embeddings_batch(texts, model, retry_count + 1, max_retries)

    except Exception as e:
        logging.error(f"Embedding error (batch {batch_size}): {e}")
        # Fallback: process un par un
        if batch_size > 1:
            logging.info("Fallback: processing batch individually")
            embeddings = []
            for text in texts:
                try:
                    resp = client.embeddings.create(input=[text], model=model)
                    embeddings.append(resp.data[0].embedding)
                except Exception as e2:
                    logging.error(f"Failed individual embedding: {e2}")
                    embeddings.append([0.0] * 3072)  # Zero vector fallback
            return embeddings
        else:
            # Single text failed, return zero vector
            logging.error(f"Single embedding failed, returning zero vector")
            return [[0.0] * 3072]

def _embed_with_cache_space(texts, recode_cfg, space):
    """Cache de vecteurs denses d'un espace hors défaut (Albert bge-m3).

    Clé cloisonnée : ``embed_key(texte, space.cache_model, space.params_json())``,
    soit ``'albert:bge-m3'`` et ``{dim, model, norm, provider}``. Un HIT de
    mauvaise dimension (ou nul) compte comme MISS ; ``None`` n'est jamais mis en
    cache ; un MISS passe par ``get_embeddings_batch(..., space=space)`` (jamais
    OpenAI).

    Args:
        texts: textes du lot.
        recode_cfg: ``RecodeConfig`` (chemin du cache).
        space: espace d'embeddings hors défaut.

    Returns:
        Liste de vecteurs (ou ``None``), dans l'ordre des textes.
    """
    cache_model = space.cache_model
    params = space.params_json()
    keys = [rad_recode_cache.embed_key(t, cache_model, params) if t else None for t in texts]
    out = [None] * len(texts)
    miss_idx = []
    for i, key in enumerate(keys):
        hit = rad_recode_cache.get_embed(recode_cfg, key) if key else None
        if _valid_space_vector(hit, space):
            out[i] = hit
        else:
            miss_idx.append(i)
    if miss_idx:
        sub = get_embeddings_batch([texts[i] for i in miss_idx], model=space.model, space=space)
        for j, i in enumerate(miss_idx):
            vec = sub[j] if j < len(sub) else None
            if not _valid_space_vector(vec, space):
                continue
            out[i] = vec
            if keys[i]:
                rad_recode_cache.put_embed(recode_cfg, keys[i], vec)
    return out


def _embed_with_cache(texts, recode_cfg, model="text-embedding-3-large", space=None):
    """Lot 7.e — cache de vecteurs denses clé par sha256(texte recodé)·model·params.
    HIT → 0 appel embedding. Ferme l'axe vecteur (texte identique → vecteur
    byte-identique). MISS → appel groupé puis PUT. Advisory (erreur cache → recalcul).

    ``space`` hors défaut (Albert) : clé cloisonnée par ``_embed_with_cache_space`` ;
    ``None`` ou espace par défaut : clés et appels historiques inchangés.
    """
    if space is not None and not space.is_default:
        return _embed_with_cache_space(texts, recode_cfg, space)
    embed_params = json.dumps({"model": model}, sort_keys=True)
    keys = [rad_recode_cache.embed_key(t, model, embed_params) if t else None for t in texts]
    out = [None] * len(texts)
    miss_idx = []
    for i, key in enumerate(keys):
        hit = rad_recode_cache.get_embed(recode_cfg, key) if key else None
        if hit is not None:
            out[i] = hit
        else:
            miss_idx.append(i)
    if miss_idx:
        sub = get_embeddings_batch([texts[i] for i in miss_idx], model=model)
        for j, i in enumerate(miss_idx):
            vec = sub[j]
            out[i] = vec
            if vec is not None and keys[i]:
                rad_recode_cache.put_embed(recode_cfg, keys[i], vec)
    return out


def _apply_space_fields(chunk, space):
    """Écrit (hors défaut) ou retire (défaut) les champs d'espace d'un chunk.

    Espace hors défaut : ``embedding_provider``, ``embedding_model`` et
    ``embedding_dim``. Espace par défaut (``None`` ou OpenAI) : ces champs sont
    retirés s'ils existent (fichier ré-embeddé), sans rien changer sinon.
    """
    if space is None or space.is_default:
        for key in EMBEDDING_SPACE_FIELDS:
            if key in chunk:
                del chunk[key]
        return
    chunk["embedding_provider"] = space.provider
    chunk["embedding_model"] = space.model
    chunk["embedding_dim"] = space.dim


def _failed_batch_embedding(space):
    """Vecteur d'un lot en échec : zéros à 3072 pour OpenAI (historique), ``None`` sinon."""
    if space is None or space.is_default:
        return [0.0] * 3072
    return None


def process_chunks_for_embedding(chunks_batch, space=None):
    """
    Traite un lot de chunks pour y ajouter les embeddings denses.
    Modifie les dictionnaires de chunks en place.

    ``space`` ``None`` (ou espace par défaut) : chemin OpenAI historique inchangé.
    Espace hors défaut (Albert) : vecteurs de cet espace (``None`` si absent) et
    champs d'espace écrits sur chaque chunk du lot.
    """
    texts_to_embed = [chunk.get("text", "") for chunk in chunks_batch]
    recode_cfg = rad_recode_cache.RecodeConfig.from_env()
    if space is not None and not space.is_default:
        if recode_cfg.embed_cache_enabled:
            embeddings = _embed_with_cache(texts_to_embed, recode_cfg, model=space.model, space=space)
        else:
            embeddings = get_embeddings_batch(texts_to_embed, model=space.model, space=space)
    elif recode_cfg.embed_cache_enabled:
        embeddings = _embed_with_cache(texts_to_embed, recode_cfg)
    else:
        embeddings = get_embeddings_batch(texts_to_embed)

    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            chunks_batch[i]["embedding"] = embedding
        else:
            # Marquer l'échec ou laisser vide, selon la stratégie souhaitée
            chunks_batch[i]["embedding"] = None
            print(f"Avertissement: Embedding non généré pour le chunk ID {chunks_batch[i].get('id', 'Inconnu')}")
        _apply_space_fields(chunks_batch[i], space)
    return chunks_batch # Retourne le lot modifié

def save_processed_chunks_to_json_overwrite(all_chunks, json_file):
    """
    Sauvegarde la liste complète des chunks (avec embeddings) dans un fichier JSON, en écrasant le contenu existant.
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Tous les chunks ({len(all_chunks)}) ont été sauvegardés dans {json_file}")

def generate_and_save_embeddings(input_json_file, output_json_file=None):
    """
    Charge les chunks depuis `input_json_file`, génère les embeddings denses,
    et les sauvegarde dans `output_json_file`.

    Cette fonction utilise une parallélisation optimisée avec:
    - ThreadPoolExecutor limité à 4 workers pour éviter les rate limits
    - Monitoring du throughput (embeddings/seconde)
    - Tracking des rate limit hits

    Espace d'embeddings (``EMBEDDING_PROVIDER``, résolu une seule fois) :

    - OpenAI (défaut) : requêtes, fichier et sorties strictement inchangés ;
    - Albert (``albert``, bge-m3, 1024 dimensions) : preflight du rôle ``embed``,
      lots de ``min(DEFAULT_EMBEDDING_BATCH_SIZE, 64)`` textes, jamais de vecteur
      nul (``None`` si absent), champs d'espace sur chaque chunk, journal d'usage
      si Albert a été appelé. Au-delà de ``ALBERT_EMBED_MAX_MISSING_RATIO``
      embeddings manquants (part calculée sur les chunks au texte non vide, les
      chunks vides n'étant jamais envoyés), le fichier est écrit puis la phase
      sort en 1 ; une erreur de compte la fait sortir en 2.

    ``albert`` avec Albert désactivé, ou une valeur inconnue : message français
    et sortie en 2, avant toute lecture.

    Args:
        input_json_file: Chemin vers le fichier JSON contenant les chunks.
        output_json_file: Chemin de sortie (auto-généré si None).

    Returns:
        Chemin du fichier de sortie ou None en cas d'erreur.

    Raises:
        SystemExit: 2 (configuration refusée, erreur de compte ou preflight
            Albert), 1 (embeddings Albert manquants au-delà du seuil, fichier écrit).
    """
    if output_json_file is None:
        base_name = os.path.splitext(input_json_file)[0]
        output_json_file = f"{base_name}_with_embeddings.json"

    try:
        embed_cfg = resolve_embedding_config()
    except ValueError as exc:
        abort_embeddings(exc, code=2)
    space = embed_cfg.space
    use_albert = space.provider == PROVIDER_ALBERT

    if not os.path.exists(input_json_file):
        print(f"Le fichier d'entrée '{input_json_file}' n'existe pas.")
        return None

    albert_cfg = _start_albert_embeddings(space) if use_albert else None

    with open(input_json_file, 'r', encoding='utf-8') as f:
        all_chunks_from_file = json.load(f)

    total_chunks = len(all_chunks_from_file)
    print(f"Chargement de {total_chunks} chunks depuis '{input_json_file}' pour génération d'embeddings.")

    # Emit init event for SSE progress tracking (chunks only - documents were processed in step 3.1)
    print(f"PROGRESS|init|{total_chunks}|Generating embeddings for {total_chunks} chunks", flush=True)

    # Taille de lot : inchangée pour OpenAI ; plafonnée à 64 textes pour Albert (D12).
    batch_size = DEFAULT_EMBEDDING_BATCH_SIZE
    if use_albert:
        batch_size = min(DEFAULT_EMBEDDING_BATCH_SIZE, space.batch_max or ALBERT_EMBED_BATCH_MAX,
                         ALBERT_EMBED_BATCH_MAX)

    # Créer tous les batches à traiter (flat list, pas groupés par doc)
    all_batches = []
    batch_to_indices = {}  # Map batch index to chunk indices in original list

    for i in range(0, total_chunks, batch_size):
        batch = all_chunks_from_file[i : i + batch_size]
        batch_idx = len(all_batches)
        all_batches.append(batch)
        batch_to_indices[batch_idx] = list(range(i, min(i + batch_size, total_chunks)))

    total_batches = len(all_batches)
    print(f"Préparation de {total_batches} batches (taille max: {batch_size})")

    # Monitoring variables
    batch_start = time.time()
    embeddings_generated = 0
    rate_limit_hits = 0
    results = [None] * total_batches  # Pre-allocate results array

    # Max 4 workers pour éviter rate limits OpenAI (Albert : ALBERT_EMBED_CONCURRENCY)
    if use_albert:
        max_embedding_workers = max(1, min(DEFAULT_MAX_WORKERS, albert_cfg.embed_concurrency))
    else:
        max_embedding_workers = min(DEFAULT_MAX_WORKERS, 4)
    logging.info(f"Using {max_embedding_workers} workers for embedding generation")

    with ThreadPoolExecutor(max_workers=max_embedding_workers) as executor:
        if use_albert:
            futures = {
                executor.submit(process_chunks_for_embedding, batch, space): batch_idx
                for batch_idx, batch in enumerate(all_batches)
            }
        else:
            futures = {
                executor.submit(process_chunks_for_embedding, batch): batch_idx
                for batch_idx, batch in enumerate(all_batches)
            }

        for future in tqdm(as_completed(futures), total=total_batches, desc="Generating embeddings"):
            batch_idx = futures[future]
            try:
                batch_embeddings = future.result()
                results[batch_idx] = batch_embeddings
                embeddings_generated += len(batch_embeddings)

                # Emit chunk-level progress
                print(f"PROGRESS|chunk|{embeddings_generated}/{total_chunks}|Chunk {embeddings_generated}/{total_chunks}", flush=True)

            except RateLimitError:
                rate_limit_hits += 1
                logging.error(f"Batch {batch_idx} failed after retries (rate limit)")
                # Mark as failed - assign zero vectors (OpenAI only; never on Albert)
                failed_batch = all_batches[batch_idx]
                for chunk in failed_batch:
                    chunk["embedding"] = _failed_batch_embedding(space)
                    _apply_space_fields(chunk, space)
                results[batch_idx] = failed_batch
                embeddings_generated += len(failed_batch)

            except Exception as e:
                logging.error(f"Batch {batch_idx} failed with error: {e}")
                failed_batch = all_batches[batch_idx]
                for chunk in failed_batch:
                    chunk["embedding"] = _failed_batch_embedding(space)
                    _apply_space_fields(chunk, space)
                results[batch_idx] = failed_batch
                embeddings_generated += len(failed_batch)

    # Flatten results maintaining order
    all_chunks_with_embeddings = []
    for batch_result in results:
        if batch_result:
            all_chunks_with_embeddings.extend(batch_result)

    # Performance logging
    elapsed = time.time() - batch_start
    throughput = embeddings_generated / elapsed if elapsed > 0 else 0
    logging.info(f"Embeddings: {embeddings_generated} in {elapsed:.1f}s ({throughput:.1f}/s)")
    logging.info(f"Rate limit hits: {rate_limit_hits}")
    print(f"Performance: {embeddings_generated} embeddings en {elapsed:.1f}s ({throughput:.1f} emb/s)")
    if total_batches > 0:
        print(f"Rate limit hits: {rate_limit_hits}/{total_batches} batches ({100*rate_limit_hits/total_batches:.1f}%)")

    # Record metrics
    if METRICS_AVAILABLE and metrics_collector:
        try:
            metrics_collector.increment_counter(
                'embeddings_generated',
                embeddings_generated,
                model=space.model,
                type='dense'
            )
            metrics_collector.observe_histogram(
                'embedding_duration',
                elapsed,
                model=space.model
            )
            if rate_limit_hits > 0:
                metrics_collector.increment_counter(
                    'errors',
                    rate_limit_hits,
                    operation='embedding',
                    error_type='RateLimitError'
                )
        except Exception:
            pass  # Don't fail on metrics errors

    save_processed_chunks_to_json_overwrite(all_chunks_with_embeddings, output_json_file)
    if use_albert:
        _finish_albert_embeddings(all_chunks_with_embeddings, output_json_file, albert_cfg, space)
    print(f"Tous les embeddings denses ont été générés. Total {len(all_chunks_with_embeddings)} chunks sauvegardés dans '{output_json_file}'.")
    return output_json_file


def _finish_albert_embeddings(chunks, output_json_file, albert_cfg, space):
    """Fin de la phase dense Albert, une fois le fichier écrit.

    Journal d'usage (``albert_usage.jsonl`` et synthèse) si Albert a été appelé,
    puis : erreur de compte mémorisée → ``exit 2`` (ligne ``albert_abort_marker``) ;
    part d'embeddings manquants au-delà de ``ALBERT_EMBED_MAX_MISSING_RATIO`` →
    ``exit 1`` avec un message français (jamais d'index silencieusement incomplet).

    Les chunks au texte vide ou blanc (``_is_blank_chunk_text``) ne sont jamais
    envoyés aux embeddings : ils ne comptent pas comme manquants, la part est
    calculée sur les seuls chunks non vides, et une ligne unique donne le nombre
    de chunks vides sans vecteur (ignorés ensuite par les connecteurs). Aucune
    ligne s'il n'y en a pas.

    Args:
        chunks: chunks écrits dans le fichier de sortie.
        output_json_file: fichier de sortie (son dossier reçoit le journal d'usage).
        albert_cfg: ``AlbertConfig`` du run.
        space: espace Albert du run.

    Raises:
        SystemExit: 2 (erreur de compte) ou 1 (manques au-delà du seuil).
    """
    report_albert_usage(os.path.dirname(os.path.abspath(output_json_file)))
    abort = _ALBERT_EMBED_ABORT
    if abort is not None:
        message = f"Erreur critique Albert : {abort} Embeddings arrêtés (code 2)."
        print(message)
        logging.error(message)
        print(albert_abort_marker(abort))
        raise SystemExit(2)
    counted = [chunk for chunk in chunks if not _is_blank_chunk_text(chunk)]
    blank_skipped = sum(1 for chunk in chunks
                        if _is_blank_chunk_text(chunk) and not _valid_space_vector(chunk.get("embedding"), space))
    if blank_skipped:
        print(f"Albert : {blank_skipped} chunk(s) au texte vide ignoré(s) (jamais envoyé(s) aux embeddings, "
              f"non compté(s) comme manquant(s), non envoyé(s) à la base vectorielle).")
    total = len(counted)
    missing = sum(1 for chunk in counted if not _valid_space_vector(chunk.get("embedding"), space))
    ratio = (missing / total) if total else 0.0
    threshold = float(albert_cfg.embed_max_missing_ratio)
    if missing and ratio > threshold:
        message = (
            f"Erreur : {missing}/{total} embedding(s) Albert manquant(s) (part {ratio:.4f}, "
            f"au-delà du seuil ALBERT_EMBED_MAX_MISSING_RATIO={threshold:g}). "
            f"Fichier écrit dans '{output_json_file}', phase dense en échec (code 1)."
        )
        print(message)
        logging.error(message)
        raise SystemExit(1)

# ----------------------------------------------------------------------
# PART 3: Chunk sparse embedding
# ----------------------------------------------------------------------

def extract_sparse_features(text):
    """
    Extrait les lemmes des mots pertinents et crée une représentation sparse.
    Utilise le `nlp` global (modèle spaCy).
    """
    if nlp is None:
        print("Erreur: Modèle spaCy (nlp) non initialisé. Impossible d'extraire les features sparse.")
        return {"indices": [], "values": []}
        
    # Limiter la taille des textes très longs pour la performance de spaCy
    if len(text) > nlp.max_length: # Check against model's max_length
         print(f"Warning: Text too long for spaCy ({len(text)} chars), truncating to {nlp.max_length}")
         text = text[:nlp.max_length]
    elif len(text) > 50000: # Fallback if max_length is very large or not restrictive enough
         print(f"Warning: Text quite long ({len(text)} chars), truncating to 50000 for sparse features")
         text = text[:50000]

    doc = nlp(text)
    relevant_pos = {"NOUN", "PROPN", "ADJ", "VERB"}
    lemmas = [
        token.lemma_.lower() for token in doc 
        if token.pos_ in relevant_pos 
        and not token.is_stop 
        and not token.is_punct 
        and len(token.lemma_) > 1 # Exclure les lemmes d'un seul caractère
    ]
    
    counts = Counter(lemmas)
    sparse_dict = {}
    # Utiliser un simple hachage pour créer un indice unique, limité à 100k dimensions
    # La normalisation (TF) est appliquée ici. IDF nécessiterait une connaissance globale du corpus.
    total_lemmas_in_doc = sum(counts.values())
    if total_lemmas_in_doc > 0:
        for lemma, count in counts.items():
            index = hash(lemma) % 100000  # Dimensionnalité de l'espace sparse
            sparse_dict[str(index)] = count / total_lemmas_in_doc # TF (Term Frequency)

    return {
        "indices": list(sparse_dict.keys()), # Convertir les indices en string comme dans le master code
        "values": list(sparse_dict.values())
    }

def generate_sparse_embeddings(input_json_file=DEFAULT_INPUT_JSON_WITH_EMBEDDINGS, 
                               output_json_file=DEFAULT_OUTPUT_JSON_SPARSE):
    """
    Charge les chunks (qui incluent déjà les embeddings denses) depuis `input_json_file`,
    génère les embeddings sparses pour chaque chunk, et sauvegarde le tout dans `output_json_file`.
    """
    if not os.path.exists(input_json_file):
        print(f"Le fichier d'entrée '{input_json_file}' pour les embeddings sparses n'existe pas.")
        return None
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    total_chunks = len(all_chunks)
    print(f"Chargement de {total_chunks} chunks depuis '{input_json_file}' pour génération d'embeddings sparses.")
    # Emit init event for SSE progress tracking
    print(f"PROGRESS|init|{total_chunks}|Loading {total_chunks} chunks for sparse embedding", flush=True)

    for i, chunk in enumerate(tqdm(all_chunks, desc="Génération Embeddings Sparses")):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            print(f"Chunk ID {chunk.get('id', i)} a un texte vide, embedding sparse sera vide.")
            sparse_embedding = {"indices": [], "values": []}
        else:
            try:
                sparse_embedding = extract_sparse_features(chunk_text)
            except Exception as e:
                print(f"Erreur lors de la génération de l'embedding sparse pour le chunk ID {chunk.get('id', i)}: {e}")
                sparse_embedding = {"indices": [], "values": []}  # Fallback

        all_chunks[i]["sparse_embedding"] = sparse_embedding  # Ajoute/met à jour la clé "sparse_embedding"

        # Emit chunk-level progress every 50 chunks to avoid overwhelming SSE
        if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
            print(f"PROGRESS|chunk|{i + 1}/{total_chunks}|SpaCy processing chunk {i + 1}", flush=True)

    # Sauvegarde finale des chunks (maintenant avec embeddings denses et sparses)
    # Utilise la même fonction de sauvegarde que pour les embeddings denses (overwrite)
    save_processed_chunks_to_json_overwrite(all_chunks, output_json_file)
    
    print(f"Traitement des embeddings sparses terminé. Fichier sauvegardé: {output_json_file}")
    return output_json_file


def _dense_requires_openai():
    """Vrai si la phase dense emprunte l'espace OpenAI (défaut), donc le client OpenAI.

    Une configuration d'embeddings refusée n'exige pas OpenAI : l'erreur est
    rapportée par ``albert_embed_startup_exception`` (sortie en 2).
    """
    try:
        return resolve_embedding_config().space.provider != PROVIDER_ALBERT
    except ValueError:
        return False


def missing_llm_client_for_phase(phase, model):
    """Indique si un client requis par la phase CLI demandée est absent.

    Garde de la CLI (plus de saisie interactive de clé) :

    * ``sparse`` : n'exige aucun client (spaCy seulement) ;
    * ``dense`` : exige le client OpenAI pour l'espace OpenAI seulement
      (``text-embedding-3-large``) ; l'espace Albert (``EMBEDDING_PROVIDER=albert``)
      n'utilise jamais OpenAI, sa clé est contrôlée par
      ``albert_embed_startup_exception`` ;
    * ``initial`` : exige un client pour le fournisseur du modèle de recodage
      (``legacy_provider``). Pour un modèle OpenRouter, le client OpenAI suffit
      aussi : ``gpt_recode_batch`` y replie déjà sur ``gpt-4o-mini`` quand le
      client OpenRouter manque. Modèle effectif ``albert/…``
      (``effective_recode_model``) : exige ``ALBERT_API_KEY`` seulement (aucun
      client OpenAI ni OpenRouter n'est utilisé pour le recodage) ;
    * ``all`` : cumule ``initial`` et ``dense``.

    Args:
        phase: ``'initial'``, ``'dense'``, ``'sparse'`` ou ``'all'``.
        model: modèle de recodage passé par ``--model``.

    Returns:
        ``True`` si la phase ne peut pas s'exécuter faute de client.
    """
    if phase in ("dense", "all") and client is None and _dense_requires_openai():
        return True
    if phase in ("initial", "all"):
        if _selects_albert(effective_recode_model(model, rad_recode_cache.RecodeConfig.from_env())):
            return not ALBERT_API_KEY
        if legacy_provider(model) == PROVIDER_OPENROUTER:
            return openrouter_client is None and client is None
        return client is None
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process text data through chunking and embedding phases.")
    parser.add_argument("--input", required=True, help="Path to the input file (CSV for 'initial' phase, JSON for 'dense' and 'sparse' phases).")
    parser.add_argument("--output", required=True, help="Directory to save the output JSON files.")
    parser.add_argument("--phase", choices=['initial', 'dense', 'sparse', 'all'], default='all',
                        help="Specify processing phase: 'initial' (chunking), 'dense' (dense embeddings), 'sparse' (sparse embeddings), or 'all'.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model for text recoding. Use 'gpt-4o-mini' (OpenAI), 'google/gemini-2.5-flash' "
                             "(OpenRouter) or 'albert/<id>' (Albert, DINUM: e.g. 'albert/ministral-3-8b-instruct-2512'; "
                             "requires ALBERT_ENABLED=1 and ALBERT_API_KEY, never falls back to OpenAI/OpenRouter). "
                             "Default: gpt-4o-mini")
    parser.add_argument("--embedding-provider", choices=["openai", "albert"], default=None,
                        help="Dense embedding provider for the 'dense' phase: 'openai' (text-embedding-3-large, "
                             "3072 dimensions) or 'albert' (bge-m3 served by Albert, DINUM: separate 1024-dimension "
                             "vector space, requires ALBERT_ENABLED=1 and ALBERT_API_KEY, never falls back to OpenAI). "
                             "Sets EMBEDDING_PROVIDER for this run. Default: EMBEDDING_PROVIDER, else openai")

    args = parser.parse_args()
    if args.embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider

    # Setup logging for chunking phase
    chunking_log_path = os.path.join(args.output, "chunking.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(chunking_log_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("chunking")

    # Ensure output directory exists
    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output, exist_ok=True)
            print(f"Output directory '{args.output}' created.")
            logger.info(f"Output directory '{args.output}' created.")
        except Exception as e:
            print(f"Error creating output directory '{args.output}': {e}")
            logger.error(f"Error creating output directory '{args.output}': {e}")
            exit(1)

    # Determine filenames based on the phase and input.
    base_name_for_outputs = "output" # Consistent with main.py's expectation for intermediate files
    if args.phase == "dense" or args.phase == "sparse":
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        if input_basename.endswith("_chunks_with_embeddings"):
            base_name_for_outputs = input_basename.replace("_chunks_with_embeddings", "")
        elif input_basename.endswith("_chunks"):
            base_name_for_outputs = input_basename.replace("_chunks", "")

    initial_chunks_json = os.path.join(args.output, f"{base_name_for_outputs}_chunks.json")
    chunks_with_dense_json = os.path.join(args.output, f"{base_name_for_outputs}_chunks_with_embeddings.json")
    chunks_with_sparse_json = os.path.join(args.output, f"{base_name_for_outputs}_chunks_with_embeddings_sparse.json")

    print(f"rad_chunk.py - Phase: {args.phase}")
    print(f"  Input File: {args.input}")
    print(f"  Output Directory: {args.output}")
    print(f"  Initial Chunks JSON: {initial_chunks_json}")
    print(f"  Dense Embeddings JSON: {chunks_with_dense_json}")
    print(f"  Sparse Embeddings JSON: {chunks_with_sparse_json}")
    logger.info(f"rad_chunk.py - Phase: {args.phase}")
    logger.info(f"  Input File: {args.input}")
    logger.info(f"  Output Directory: {args.output}")
    logger.info(f"  Initial Chunks JSON: {initial_chunks_json}")
    logger.info(f"  Dense Embeddings JSON: {chunks_with_dense_json}")
    logger.info(f"  Sparse Embeddings JSON: {chunks_with_sparse_json}")
    
    if TEXT_SPLITTER is None:
        print("Erreur critique: TEXT_SPLITTER n'est pas initialisé (langchain_text_splitters manquant?). Arrêt.")
        logger.error("Erreur critique: TEXT_SPLITTER n'est pas initialisé (langchain_text_splitters manquant?). Arrêt.")
        exit(1)
    if nlp is None:
        print("Erreur critique: Modèle spaCy (nlp) n'est pas initialisé. Arrêt.")
        logger.error("Erreur critique: Modèle spaCy (nlp) n'est pas initialisé. Arrêt.")
        exit(1)
    # Albert sélectionné pour le recodage : contrôles locaux avant tout (Albert
    # activé, modèle, clé) ; jamais de repli vers OpenAI/OpenRouter (code 2,
    # précédé de la ligne ``Albert abort: …``, comme tout arrêt Albert).
    if args.phase in ('initial', 'all'):
        albert_startup_exc = albert_recode_startup_exception(args.model)
        if albert_startup_exc is not None:
            print(f"Erreur critique Albert : {albert_startup_exc} Arrêt.")
            logger.error(f"Erreur critique Albert : {albert_startup_exc} Arrêt.")
            print(albert_abort_marker(albert_startup_exc))
            exit(2)
    # Espace d'embeddings de la phase dense (EMBEDDING_PROVIDER) : valeur connue,
    # Albert activé et clé présente pour ``albert`` ; sinon message et code 2,
    # avant tout traitement (jamais de repli vers OpenAI).
    if args.phase in ('dense', 'all'):
        embed_startup_exc = albert_embed_startup_exception()
        if embed_startup_exc is not None:
            abort_embeddings(embed_startup_exc, code=2)
    if missing_llm_client_for_phase(args.phase, args.model):
        print("Erreur critique: Client OpenAI non initialisé (OPENAI_API_KEY manquante?). Arrêt.")
        logger.error("Erreur critique: Client OpenAI non initialisé (OPENAI_API_KEY manquante?). Arrêt.")
        exit(1)

    # Phase-specific execution
    if args.phase == 'initial' or args.phase == 'all':
        print("\n--- Phase 3.1 : Découpage initial (initial chunk) ---")
        logger.info("=== Phase 3.1 : Découpage initial (initial chunk) ===")
        if not args.input.lower().endswith(".csv"):
            print(f"Erreur: La phase 'initial' attend un fichier CSV en entrée, reçu: {args.input}")
            logger.error(f"Erreur: La phase 'initial' attend un fichier CSV en entrée, reçu: {args.input}")
            exit(1)
        try:
            df = pd.read_csv(args.input)
            print(f"Chargement de {len(df)} lignes depuis '{args.input}'.")
            logger.info(f"Chargement de {len(df)} lignes depuis '{args.input}'.")
        except FileNotFoundError:
            print(f"Erreur: Le fichier d'entrée CSV '{args.input}' n'a pas été trouvé.")
            logger.error(f"Erreur: Le fichier d'entrée CSV '{args.input}' n'a pas été trouvé.")
            exit(1)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier CSV '{args.input}': {e}")
            logger.error(f"Erreur lors du chargement du fichier CSV '{args.input}': {e}")
            exit(1)

        # Preflight Albert du rôle recode (compte, modèles épinglés) avant de
        # toucher aux sorties ; sans effet si le modèle n'est pas albert/….
        try:
            albert_preflight = run_albert_recode_preflight(args.model)
        except (_rad_albert.AlbertError, ValueError) as e:
            print(f"Erreur critique Albert (preflight) : {e} Arrêt.")
            logger.error(f"Erreur critique Albert (preflight) : {e} Arrêt.")
            print(albert_abort_marker(e))
            exit(2)
        if albert_preflight is not None:
            print(albert_preflight.summary_line())
            logger.info(albert_preflight.summary_line())

        if os.path.exists(initial_chunks_json):
            print(f"Nettoyage du fichier de chunks existant: {initial_chunks_json}")
            logger.info(f"Nettoyage du fichier de chunks existant: {initial_chunks_json}")
            try:
                os.remove(initial_chunks_json)
            except OSError as e:
                print(f"Avertissement: Impossible de supprimer {initial_chunks_json}: {e}. Le contenu pourrait être ajouté.")
                logger.warning(f"Avertissement: Impossible de supprimer {initial_chunks_json}: {e}. Le contenu pourrait être ajouté.")

        process_all_documents(df, json_file=initial_chunks_json, model=args.model)
        # Ledger d'usage Albert : fichier et synthèse seulement si Albert a été appelé.
        report_albert_usage(args.output)
        if _ALBERT_ABORT is not None:
            print(f"Erreur critique Albert : {_ALBERT_ABORT} Recodage arrêté (code 2).")
            logger.error(f"Erreur critique Albert : {_ALBERT_ABORT} Recodage arrêté (code 2).")
            print(albert_abort_marker(_ALBERT_ABORT))
            exit(2)
        if not os.path.exists(initial_chunks_json) or os.path.getsize(initial_chunks_json) == 0:
            print(f"Erreur: Aucun chunk n'a été généré dans '{initial_chunks_json}'.")
            logger.error(f"Erreur: Aucun chunk n'a été généré dans '{initial_chunks_json}'.")
            exit(1)
        print(f"Phase 'initial' terminée. Output: {initial_chunks_json}")
        logger.info(f"Phase 'initial' terminée. Output: {initial_chunks_json}")

    if args.phase == 'dense' or args.phase == 'all':
        print("\n--- Phase: Dense Embedding Generation ---")
        logger.info("--- Phase: Dense Embedding Generation ---")
        input_for_dense = initial_chunks_json if args.phase == 'all' else args.input
        if not input_for_dense.lower().endswith("_chunks.json") and args.phase != 'all':
             if not input_for_dense.lower().endswith(".json"):
                print(f"Erreur: La phase 'dense' attend un fichier JSON de chunks en entrée (ex: ..._chunks.json), reçu: {input_for_dense}")
                logger.error(f"Erreur: La phase 'dense' attend un fichier JSON de chunks en entrée (ex: ..._chunks.json), reçu: {input_for_dense}")
                exit(1)

        dense_output_file = generate_and_save_embeddings(
            input_json_file=input_for_dense,
            output_json_file=chunks_with_dense_json
        )
        if dense_output_file is None or not os.path.exists(dense_output_file) or os.path.getsize(dense_output_file) == 0:
            print(f"Erreur: Le fichier d'embeddings denses '{chunks_with_dense_json}' n'a pas été généré ou est vide.")
            logger.error(f"Erreur: Le fichier d'embeddings denses '{chunks_with_dense_json}' n'a pas été généré ou est vide.")
            exit(1)
        print(f"Phase 'dense' terminée. Output: {chunks_with_dense_json}")
        logger.info(f"Phase 'dense' terminée. Output: {chunks_with_dense_json}")

    if args.phase == 'sparse' or args.phase == 'all':
        print("\n--- Phase: Sparse Embedding Generation ---")
        logger.info("--- Phase: Sparse Embedding Generation ---")
        input_for_sparse = chunks_with_dense_json if args.phase == 'all' else args.input
        if not input_for_sparse.lower().endswith("_chunks_with_embeddings.json") and args.phase != 'all':
            if not input_for_sparse.lower().endswith(".json"):
                print(f"Erreur: La phase 'sparse' attend un fichier JSON avec embeddings denses (ex: ..._chunks_with_embeddings.json), reçu: {input_for_sparse}")
                logger.error(f"Erreur: La phase 'sparse' attend un fichier JSON avec embeddings denses (ex: ..._chunks_with_embeddings.json), reçu: {input_for_sparse}")
                exit(1)

        sparse_output_file = generate_sparse_embeddings(
            input_json_file=input_for_sparse,
            output_json_file=chunks_with_sparse_json
        )
        if sparse_output_file is None or not os.path.exists(sparse_output_file) or os.path.getsize(sparse_output_file) == 0:
            print(f"Erreur: Le fichier d'embeddings sparses '{chunks_with_sparse_json}' n'a pas été généré ou est vide.")
            logger.error(f"Erreur: Le fichier d'embeddings sparses '{chunks_with_sparse_json}' n'a pas été généré ou est vide.")
            exit(1)
        print(f"Phase 'sparse' terminée. Output: {chunks_with_sparse_json}")
        logger.info(f"Phase 'sparse' terminée. Output: {chunks_with_sparse_json}")
        
    print(f"\nTraitement ({args.phase}) terminé.")
    logger.info(f"Traitement ({args.phase}) terminé.")
