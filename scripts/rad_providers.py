"""scripts/rad_providers.py — Résolution du fournisseur LLM et de l'espace d'embeddings.

Module **stdlib seulement**, sans effet de bord à l'import, utilisable par les
routes, les utilitaires web et les scripts CLI, qu'Albert soit activé ou non.

* ``resolve_llm_provider`` : résolveur unique du fournisseur de chat. Le préfixe
  ``albert/`` (insensible à la casse) est testé **en premier** ; sinon
  l'heuristique historique s'applique à l'identique (``"/"`` dans le nom →
  OpenRouter, sinon OpenAI).
* ``legacy_provider`` / ``legacy_cache_provider_label`` : répliques exactes des
  décisions historiques de ``rad_chunk.py`` (routage de ``gpt_recode_batch`` et
  libellé de fournisseur de la clé de cache de ``recode_batch_cached``).
* ``EmbeddingSpace`` / ``EmbeddingConfig`` : description de l'espace vectoriel
  (fournisseur, modèle, dimension, lot maximal, normalisation) ; l'espace OpenAI
  par défaut garde exactement les paramètres de cache d'aujourd'hui.
* ``check_uniform_space`` / ``target_mismatch_message`` : gardes d'espace
  (fichier mélangé, dimension de la cible), sans aucun appel réseau.
* ``valid_dense_vector`` : validation commune d'un vecteur dense (audit A07),
  partagée par la phase dense (avant cache et écriture) et les connecteurs
  (avant insertion), quel que soit le fournisseur.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, NamedTuple, Optional

try:
    from scripts.rad_albert.errors import AlbertDisabledError
    from scripts.rad_settings.capabilities import declared_choice
    from scripts.rad_settings.chat import OFFICIAL_BASE_URLS, compat_base_url, compat_key_var, parse_compat
except ImportError:
    from rad_albert.errors import AlbertDisabledError
    from rad_settings.capabilities import declared_choice
    from rad_settings.chat import OFFICIAL_BASE_URLS, compat_base_url, compat_key_var, parse_compat


# ---------------------------------------------------------------------------
# Fournisseurs LLM
# ---------------------------------------------------------------------------
ALBERT_PREFIX = "albert/"

PROVIDER_OPENAI = "openai"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_ALBERT = "albert"
PROVIDER_COMPAT_PREFIX = "compat:"
"""Fournisseur ``compat:<serveur>`` : client compatible OpenAI sur l'adresse déclarée
du serveur (forme interne ``@<serveur>:<modèle>``, ``scripts/rad_settings/chat.py``)."""

# Identifiant de credential (``app/core/credentials.py``) propre à chaque fournisseur.
CREDENTIAL_KEYS = {
    PROVIDER_OPENAI: "openai_api_key",
    PROVIDER_OPENROUTER: "openrouter_api_key",
    PROVIDER_ALBERT: "albert_api_key",
}


class ProviderResolution(NamedTuple):
    """Résultat de ``resolve_llm_provider``.

    Attributes:
        provider: ``'openai'``, ``'openrouter'`` ou ``'albert'``.
        wire_model: nom de modèle envoyé à l'API (préfixe ``albert/`` retiré).
        credential_key: identifiant de credential à utiliser pour ce fournisseur.
    """

    provider: str
    wire_model: str
    credential_key: Optional[str]


def is_compat(provider: Optional[str]) -> bool:
    """Vrai pour un fournisseur ``compat:<serveur>``."""
    return bool(provider) and str(provider).startswith(PROVIDER_COMPAT_PREFIX)


def compat_server(provider: str) -> str:
    """Serveur d'un fournisseur ``compat:<serveur>`` (``mistral``…)."""
    return str(provider)[len(PROVIDER_COMPAT_PREFIX):]


def credential_key_for(provider: str) -> Optional[str]:
    """Identifiant de credential d'un fournisseur (``mistral_api_key`` pour ``compat:mistral``)."""
    if is_compat(provider):
        return f"{compat_server(provider)}_api_key"
    return CREDENTIAL_KEYS.get(provider)


def make_compat_client(provider: str, api_key: Optional[str], env: Optional[Mapping[str, str]] = None):
    """Client compatible OpenAI pour ``compat:<serveur>``, sur l'adresse déclarée.

    Args:
        provider: ``compat:<serveur>``.
        api_key: Clé du serveur (personnelle, ou ``.env`` pour un administrateur).
        env: Valeurs où lire l'adresse (``os.environ`` par défaut).

    Returns:
        Un client ``openai.OpenAI`` dont ``base_url`` est l'adresse du serveur.

    Raises:
        ValueError: Clé absente ou adresse non déclarée.
    """
    from openai import OpenAI

    server = compat_server(provider)
    if not api_key:
        raise ValueError(
            f"Clé du serveur {server} absente ({compat_key_var(server)}) : la déclarer au bloc 1 du .env "
            "ou dans Paramètres > Mes Identifiants.")
    return OpenAI(api_key=api_key, base_url=compat_base_url(server, env))


def make_async_compat_client(provider: str, api_key: Optional[str], env: Optional[Mapping[str, str]] = None):
    """Version asynchrone de ``make_compat_client`` (``openai.AsyncOpenAI``)."""
    from openai import AsyncOpenAI

    server = compat_server(provider)
    if not api_key:
        raise ValueError(
            f"Clé du serveur {server} absente ({compat_key_var(server)}) : la déclarer au bloc 1 du .env "
            "ou dans Paramètres > Mes Identifiants.")
    return AsyncOpenAI(api_key=api_key, base_url=compat_base_url(server, env))


def legacy_provider(model: Optional[str]) -> str:
    """Réplique exacte du routage historique ``use_openrouter = "/" in model``.

    Reprend ``gpt_recode_batch`` (``rad_chunk.py``, branche historique et branche
    durcie avec client OpenRouter disponible) et les autres sites qui testent
    ``"/" in model``. ``None`` ou ``''`` donnent ``'openai'``.

    Args:
        model: nom de modèle tel que saisi.

    Returns:
        ``'openrouter'`` si le nom contient ``/``, sinon ``'openai'`` ; pour la
        forme interne ``@<serveur>:<modèle>``, ``compat:<serveur>``.
    """
    compat = parse_compat(model) if (model and str(model).startswith("@")) else None
    if compat:
        return PROVIDER_COMPAT_PREFIX + compat[0]
    return PROVIDER_OPENROUTER if (model and "/" in model) else PROVIDER_OPENAI


def legacy_cache_provider_label(model: Optional[str], *, prefer_openai: bool) -> str:
    """Réplique exacte du libellé de fournisseur de la clé de cache de recodage.

    Reprend ``recode_batch_cached`` (``rad_chunk.py``) :
    ``"openrouter" if (("/" in eff_model) and not cfg.prefer_openai) else "openai"``.

    Args:
        model: modèle effectif (``eff_model``) servant à la clé de cache.
        prefer_openai: valeur de ``RecodeConfig.prefer_openai``.

    Returns:
        ``'openrouter'`` ou ``'openai'``.
    """
    if model and str(model).startswith("@"):
        return legacy_provider(model)
    if (model and "/" in model) and not prefer_openai:
        return PROVIDER_OPENROUTER
    return PROVIDER_OPENAI


def resolve_llm_provider(model: Optional[str], *, albert_enabled: bool) -> ProviderResolution:
    """Résout le fournisseur de chat d'un nom de modèle.

    Le préfixe ``albert/`` est testé en premier, sans tenir compte de la casse
    (``model[:7].lower()``). La découpe se fait au premier ``/`` :
    ``albert/openai/gpt-oss-120b`` donne le modèle ``openai/gpt-oss-120b``. Toute
    autre chaîne est routée exactement comme aujourd'hui (``legacy_provider``) et
    transmise inchangée.

    Args:
        model: nom de modèle saisi (``None`` est traité comme ``''``).
        albert_enabled: valeur de ``AlbertConfig.from_env().enabled``.

    Returns:
        ``ProviderResolution(provider, wire_model, credential_key)``.

    Raises:
        AlbertDisabledError: préfixe ``albert/`` alors qu'Albert est désactivé.
        ValueError: préfixe ``albert/`` sans identifiant de modèle après lui.
    """
    text = "" if model is None else str(model)
    if text[: len(ALBERT_PREFIX)].lower() == ALBERT_PREFIX:
        if not albert_enabled:
            raise AlbertDisabledError(
                f"Le modèle « {text} » exige Albert, désactivé sur ce serveur (ALBERT_ENABLED=1 requis).",
                model=text,
            )
        wire = text.split("/", 1)[1]
        if not wire.strip():
            raise ValueError(
                "Modèle Albert vide : préciser un identifiant après « albert/ » (ex. albert/gpt-oss-120b)."
            )
        return ProviderResolution(PROVIDER_ALBERT, wire, CREDENTIAL_KEYS[PROVIDER_ALBERT])
    compat = parse_compat(text) if text.startswith("@") else None
    if compat:
        provider = PROVIDER_COMPAT_PREFIX + compat[0]
        return ProviderResolution(provider, compat[1], credential_key_for(provider))
    provider = legacy_provider(text)
    return ProviderResolution(provider, text, CREDENTIAL_KEYS[provider])


# ---------------------------------------------------------------------------
# Espaces d'embeddings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingSpace:
    """Espace vectoriel dense : deux espaces différents ne se mélangent jamais.

    Attributes:
        provider: ``'openai'`` ou ``'albert'`` (ou fournisseur lu dans un fichier).
        model: identifiant du modèle d'embedding.
        dim: dimension des vecteurs.
        batch_max: nombre maximal de textes par requête (0 = inconnu).
        is_default: ``True`` pour l'espace historique (aucun champ d'espace écrit).
        norm: normalisation appliquée par RAGpy (``'l2'``, ``'none'``, ``'unknown'``).
    """

    provider: str
    model: str
    dim: int
    batch_max: int
    is_default: bool
    norm: str

    @property
    def cache_model(self) -> str:
        """Nom de modèle utilisé dans la clé du cache d'embeddings.

        L'espace par défaut garde le nom du modèle (clés historiques inchangées) ;
        tout autre espace est cloisonné par ``'<fournisseur>:<modèle>'``
        (ex. ``'albert:bge-m3'``).
        """
        if self.is_default:
            return self.model
        return f"{self.provider}:{self.model}"

    def params_json(self) -> str:
        """Paramètres d'espace sérialisés (composante de la clé de cache).

        Espace par défaut : **exactement** ``json.dumps({"model": model}, sort_keys=True)``
        comme ``rad_chunk._embed_with_cache``. Autres espaces :
        ``{dim, model, norm, provider}`` triés.
        """
        if self.is_default:
            return json.dumps({"model": self.model}, sort_keys=True)
        return json.dumps(
            {"dim": self.dim, "model": self.model, "norm": self.norm, "provider": self.provider},
            sort_keys=True,
        )


OPENAI_DEFAULT = EmbeddingSpace(
    provider=PROVIDER_OPENAI,
    model="text-embedding-3-large",
    dim=3072,
    batch_max=2048,
    is_default=True,
    norm="none",
)

ALBERT_BGE_M3 = EmbeddingSpace(
    provider=PROVIDER_ALBERT,
    model="bge-m3",
    dim=1024,
    batch_max=64,
    is_default=False,
    norm="l2",
)

# Modèles d'embedding Albert pris en charge (id canonique → dimension) et alias
# documentés (``/v1/models``, décision D2).
_ALBERT_EMBED_DIMS = {"bge-m3": 1024}
_ALBERT_EMBED_ALIASES = {"baai/bge-m3": "bge-m3", "openweight-embeddings": "bge-m3"}

# Modèle implicite d'un chunk sans champ ``embedding_model``.
_DEFAULT_MODEL_BY_PROVIDER = {
    PROVIDER_OPENAI: OPENAI_DEFAULT.model,
    PROVIDER_ALBERT: ALBERT_BGE_M3.model,
}

_KNOWN_SPACES = {
    (space.provider, space.model, space.dim): space for space in (OPENAI_DEFAULT, ALBERT_BGE_M3)
}

# Modèles d'embedding OpenAI (adresse officielle) pris en charge par le couple
# EMBEDDING_SERVER + EMBEDDING_MODEL (lot L6) : id → dimension.
_OPENAI_EMBED_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536, "text-embedding-ada-002": 1536}


def _load_albert_config(env: Optional[Mapping[str, str]]) -> Any:
    """Construit ``AlbertConfig.from_env(env)`` (import paresseux, motif double)."""
    try:
        from scripts.rad_albert.config import AlbertConfig
    except ImportError:
        from rad_albert.config import AlbertConfig
    return AlbertConfig.from_env(env)


def _load_policy_module() -> Any:
    """Module ``rad_albert.policy`` (import paresseux, motif double)."""
    try:
        from scripts.rad_albert import policy
    except ImportError:
        from rad_albert import policy
    return policy


def _albert_space(cfg: Any) -> EmbeddingSpace:
    """Espace Albert décrit par la configuration (modèle et normalisation).

    Seuls les réglages qui changent les vecteurs entrent dans l'espace ;
    ``batch_max`` reste le plafond de l'API (64) : la taille de lot effective est
    ``min(cfg.embed_batch, space.batch_max)`` côté appelant.

    Raises:
        ValueError: modèle d'embedding Albert non pris en charge.
    """
    raw_model = str(getattr(cfg, "embed_model", "") or ALBERT_BGE_M3.model).strip()
    model = _ALBERT_EMBED_ALIASES.get(raw_model.lower(), raw_model)
    if model not in _ALBERT_EMBED_DIMS:
        raise ValueError(
            f"ALBERT_EMBED_MODEL={raw_model!r} non pris en charge : seul bge-m3 (1024 dimensions) est accepté."
        )
    norm = "l2" if getattr(cfg, "embed_l2_normalize", True) else "none"
    space = replace(ALBERT_BGE_M3, model=model, dim=_ALBERT_EMBED_DIMS[model], norm=norm)
    return ALBERT_BGE_M3 if space == ALBERT_BGE_M3 else space


@dataclass(frozen=True)
class EmbeddingConfig:
    """Choix du fournisseur d'embeddings denses (couple ``EMBEDDING_*`` ou ``EMBEDDING_PROVIDER``).

    Attributes:
        provider: ``'openai'`` (défaut, vide ou absent), ``'albert'``, ou la clé
            d'un serveur compatible OpenAI déclaré (``'mistral'``, ``'local'``…).
        space: espace vectoriel correspondant ; ``dim`` vaut 0 pour un serveur
            compatible tant que la dimension n'a pas été mesurée (premier appel).
        compat_server: clé du serveur compatible OpenAI (vide pour OpenAI à son
            adresse officielle et pour Albert).
        key_var: variable de la clé de ce serveur (``MISTRAL_API_KEY``…).
    """

    provider: str
    space: EmbeddingSpace
    compat_server: str = ""
    key_var: str = ""

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "EmbeddingConfig":
        """Lit le couple ``EMBEDDING_SERVER`` + ``EMBEDDING_MODEL``, sinon ``EMBEDDING_PROVIDER``.

        Mode unifié avec un couple déclaré (lot L6) : voir ``_from_choice``.
        Sinon, ``EMBEDDING_PROVIDER`` : ``''`` ou absent → OpenAI (espace
        historique, aucune autre variable lue) ; ``albert`` → espace bge-m3,
        décrit par ``AlbertConfig.from_env(env)``.

        Raises:
            AlbertDisabledError: Albert demandé alors qu'``ALBERT_ENABLED`` ≠ 1.
            ValueError: valeur inconnue, modèle non pris en charge, couple
                invalide (``ServiceConfigError``) ou refusé par la politique
                ``albert_only`` (``AlbertPolicyError``).
        """
        env = os.environ if env is None else env
        choice = declared_choice("embedding", env)
        if choice is not None:
            return cls._from_choice(choice, env)
        raw = env.get("EMBEDDING_PROVIDER")
        name = (raw or "").strip().lower()
        if name in ("", PROVIDER_OPENAI):
            return cls(provider=PROVIDER_OPENAI, space=OPENAI_DEFAULT)
        if name == PROVIDER_ALBERT:
            cfg = _load_albert_config(env)
            if not getattr(cfg, "enabled", False):
                raise AlbertDisabledError(
                    "EMBEDDING_PROVIDER=albert exige Albert, désactivé sur ce serveur (ALBERT_ENABLED=1 requis).",
                    model="embedding_provider=albert",
                )
            return cls(provider=PROVIDER_ALBERT, space=_albert_space(cfg))
        raise ValueError(
            f"EMBEDDING_PROVIDER={raw!r} inconnu : valeurs admises « openai » ou « albert »."
        )

    @classmethod
    def _from_choice(cls, choice: Any, env: Mapping[str, str]) -> "EmbeddingConfig":
        """Configuration d'un couple déclaré (``rad_settings.models.ServiceChoice``).

        * Albert : espace décrit par ``AlbertConfig.from_env(env)``, dont
          ``ALBERT_EMBED_MODEL`` vaut le modèle du couple (superposition) ;
        * OpenAI à son adresse officielle : modèle de ``_OPENAI_EMBED_DIMS`` ;
          ``text-embedding-3-large`` redonne l'espace historique à l'identique ;
        * autre serveur : client compatible OpenAI sur l'adresse déclarée,
          espace ``<serveur>`` + modèle, dimension mesurée au premier appel.

        Raises:
            AlbertDisabledError: Albert désactivé.
            ValueError: modèle OpenAI inconnu, ou politique ``albert_only`` et
                serveur autre qu'Albert.
        """
        server = choice.server.key
        cfg = _load_albert_config(env)
        _load_policy_module().check_embedding_provider(
            PROVIDER_ALBERT if server == PROVIDER_ALBERT else PROVIDER_OPENAI, cfg)
        if server == PROVIDER_ALBERT:
            if not getattr(cfg, "enabled", False):
                raise AlbertDisabledError(
                    "EMBEDDING_SERVER désigne Albert, désactivé sur ce serveur (ALBERT_ENABLED=1 requis).",
                    model="EMBEDDING_SERVER=albert",
                )
            return cls(provider=PROVIDER_ALBERT, space=_albert_space(cfg))
        if server == PROVIDER_OPENAI and choice.server_url == OFFICIAL_BASE_URLS[PROVIDER_OPENAI]:
            dim = _OPENAI_EMBED_DIMS.get(choice.model)
            if dim is None:
                raise ValueError(
                    f"EMBEDDING_MODEL={choice.model!r} : modèle OpenAI non pris en charge "
                    f"(admis : {', '.join(sorted(_OPENAI_EMBED_DIMS))})."
                )
            if choice.model == OPENAI_DEFAULT.model:
                return cls(provider=PROVIDER_OPENAI, space=OPENAI_DEFAULT)
            space = replace(OPENAI_DEFAULT, model=choice.model, dim=dim, is_default=False)
            return cls(provider=PROVIDER_OPENAI, space=space)
        space = EmbeddingSpace(provider=server, model=choice.model, dim=0, batch_max=0, is_default=False, norm="none")
        return cls(provider=server, space=space, compat_server=server, key_var=choice.server.key_var)

    def embed_params_json(self) -> str:
        """Paramètres d'espace de la clé du cache d'embeddings.

        Pour OpenAI, **exactement** ``json.dumps({"model": "text-embedding-3-large"},
        sort_keys=True)`` (clés historiques inchangées).
        """
        return self.space.params_json()


# ---------------------------------------------------------------------------
# Gardes d'espace
# ---------------------------------------------------------------------------
def valid_dense_vector(vec: Any, dim: Optional[int] = None) -> bool:
    """Vrai si ``vec`` est un vecteur dense exploitable (audit A07).

    Liste (ou tuple) non vide de nombres **finis** (ni NaN ni infini), de norme
    non nulle, et de longueur ``dim`` quand elle est donnée. Un vecteur nul
    (ancien repli d'échec OpenAI), vide, de mauvaise dimension ou contenant une
    valeur non numérique n'est jamais mis en cache, écrit comme succès ni inséré.

    Args:
        vec: vecteur candidat.
        dim: dimension attendue, ou ``None`` pour ne pas la contrôler.

    Returns:
        ``True`` si le vecteur est exploitable.
    """
    if isinstance(vec, (str, bytes)) or not isinstance(vec, (list, tuple)) or not vec:
        return False
    if dim is not None and len(vec) != dim:
        return False
    # Boucles C (sum, any) plutôt qu'une boucle Python sur 3072 valeurs : la somme
    # propage NaN et l'infini (inf - inf donne NaN), déborde vers l'infini et refuse
    # un non-nombre (TypeError).
    try:
        total = sum(vec)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(total) and any(vec)


def _real_vector(vec: Any) -> bool:
    """Vrai si ``vec`` est une liste de nombres non vide et non entièrement nulle."""
    if not isinstance(vec, (list, tuple)) or not vec:
        return False
    return any(x != 0 for x in vec)


def _is_int(value: Any) -> bool:
    """Vrai pour un entier (``bool`` exclu), numpy compris."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _space_for(provider: str, model: str, dim: int) -> EmbeddingSpace:
    """Espace connu correspondant au triplet, ou espace ad hoc hors défaut."""
    known = _KNOWN_SPACES.get((provider, model, dim))
    if known is not None:
        return known
    return EmbeddingSpace(provider=provider, model=model, dim=dim, batch_max=0, is_default=False, norm="unknown")


def check_uniform_space(chunks: Optional[Iterable[Any]]) -> Optional[EmbeddingSpace]:
    """Vérifie que tous les vecteurs denses d'un fichier partagent un seul espace.

    Chaque chunk est lu via ``embedding`` et les champs hors défaut
    ``embedding_provider``, ``embedding_model``, ``embedding_dim`` (absents = espace
    OpenAI historique). Les vecteurs ``None``, vides ou entièrement nuls (repli
    OpenAI) et les entrées qui ne sont pas des dictionnaires sont ignorés.

    Args:
        chunks: chunks du fichier (``*_chunks_with_embeddings*.json``).

    Returns:
        L'espace commun, ou ``None`` si aucun vecteur exploitable.

    Raises:
        ValueError: espaces mélangés (dimensions ou modèles différents), ou
            ``embedding_dim`` déclaré différent de la longueur réelle.
    """
    found = {}
    for chunk in chunks or ():
        if not isinstance(chunk, Mapping):
            continue
        vec = chunk.get("embedding")
        if not _real_vector(vec):
            continue
        provider = str(chunk.get("embedding_provider") or PROVIDER_OPENAI).strip().lower()
        model = str(chunk.get("embedding_model") or _DEFAULT_MODEL_BY_PROVIDER.get(provider, ""))
        dim = len(vec)
        declared = chunk.get("embedding_dim")
        if _is_int(declared) and int(declared) != dim:
            raise ValueError(
                f"Chunk {chunk.get('id', '?')} : embedding_dim déclaré {int(declared)} "
                f"≠ longueur réelle du vecteur {dim}."
            )
        key = (provider, model, dim)
        found[key] = found.get(key, 0) + 1
    if not found:
        return None
    if len(found) > 1:
        detail = " ; ".join(
            f"{p}/{m} ({d} dimensions) : {n} chunk(s)" for (p, m, d), n in found.items()
        )
        raise ValueError(
            f"Espaces d'embeddings mélangés dans le fichier ({detail}). "
            "Régénérer les embeddings avec un seul fournisseur."
        )
    (provider, model, dim), = found
    return _space_for(provider, model, dim)


def target_mismatch_message(target_dim: Any, space: Optional[EmbeddingSpace]) -> Optional[str]:
    """Message français si la dimension de la cible ne correspond pas à l'espace.

    Une dimension de cible qui n'est pas un entier (``None``, chaîne, flottant,
    booléen, objet factice) est ignorée : aucune garde, aucun appel réseau.

    Args:
        target_dim: dimension lue sur l'index ou la collection cible.
        space: espace du fichier (``check_uniform_space``) ou ``None``.

    Returns:
        Le message d'erreur, ou ``None`` si rien à signaler.
    """
    if space is None or not _is_int(target_dim):
        return None
    target = int(target_dim)
    if target == space.dim:
        return None
    return (
        f"Dimension incompatible : la cible attend des vecteurs de {target} dimensions, "
        f"mais les embeddings du fichier ({space.provider}/{space.model}) en ont {space.dim}. "
        "Choisir une cible créée pour cet espace ou régénérer les embeddings."
    )
