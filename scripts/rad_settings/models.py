"""Règle serveur + modèle par service (sprint « configuration unifiée », lot L1).

Règle décidée par Amar le 2026-10-03 :

* un serveur par défaut (``LLM_DEFAULT_SERVER``) et un modèle par défaut
  (``LLM_DEFAULT_MODEL``) ;
* chaque service déclare un couple ``<SERVICE>_SERVER`` + ``<SERVICE>_MODEL`` ;
  un service qui ne donne qu'un modèle l'envoie au serveur par défaut, un
  service qui déclare son serveur l'utilise à la place du serveur par défaut ;
* **le serveur est l'adresse de l'API** (``https://openrouter.ai/api/v1``) et
  **le nom du modèle est envoyé tel quel**, dans la nomenclature du serveur ;
* l'adresse doit être celle d'un serveur déclaré au bloc 1 du ``.env``
  (``<FOURNISSEUR>_API_BASE_URL``), qui fournit la clé et le protocole : une
  clé ne part jamais vers une adresse qui n'est pas la sienne ;
* seul le mot réservé ``local``, sans adresse, désigne les moteurs OCR
  internes (``docling``, ``pymupdf``).

Fonctions pures, de la bibliothèque standard : elles reçoivent les valeurs
(``Mapping`` nom → valeur) et ne lisent jamais l'environnement elles-mêmes.
Elles sont branchées sur le code au lot L4 (modèles de langage), L5 (OCR) et
L6 (embeddings, audio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple
from urllib.parse import urlsplit

CHAT = "chat"
OCR = "ocr"
EMBEDDING = "embedding"
AUDIO = "audio"

LOCAL_KEYWORD = "local"
"""Mot réservé de ``OCR_SERVER`` pour les moteurs OCR internes, sans adresse."""

LOCAL_ENGINES = ("docling", "pymupdf")
"""Modèles admis avec le mot réservé ``local``."""


@dataclass(frozen=True)
class ServerDef:
    """Un serveur d'API déclaré au bloc 1 du ``.env``.

    Attributes:
        key: Identifiant court (``openrouter``).
        label: Nom affiché (``OpenRouter``).
        base_url_vars: Variables qui portent l'adresse, la première prioritaire.
        key_var: Variable de la clé (``OPENROUTER_API_KEY``).
        protocol: ``openrouter``, ``mistral``, ``albert``, ``openai_compatible``
            ou ``local_engine``.
        families: Services que ce serveur sait rendre.
    """

    key: str
    label: str
    base_url_vars: Tuple[str, ...]
    key_var: str
    protocol: str
    families: frozenset


SERVER_DEFS: Tuple[ServerDef, ...] = (
    ServerDef("openrouter", "OpenRouter", ("OPENROUTER_API_BASE_URL",), "OPENROUTER_API_KEY", "openrouter",
              frozenset({CHAT, EMBEDDING})),
    ServerDef("mistral", "Mistral", ("MISTRAL_API_BASE_URL",), "MISTRAL_API_KEY", "mistral",
              frozenset({CHAT, OCR, EMBEDDING})),
    ServerDef("openai", "OpenAI", ("OPENAI_API_BASE_URL",), "OPENAI_API_KEY", "openai_compatible",
              frozenset({CHAT, EMBEDDING})),
    ServerDef("albert", "Albert (DINUM)", ("ALBERT_BASE_URL",), "ALBERT_API_KEY", "albert",
              frozenset({CHAT, OCR, EMBEDDING, AUDIO})),
    ServerDef("anthropic", "Anthropic", ("ANTHROPIC_API_BASE_URL",), "ANTHROPIC_API_KEY", "openai_compatible",
              frozenset({CHAT})),
    ServerDef("google", "Google Gemini", ("GOOGLE_API_BASE_URL",), "GOOGLE_API_KEY", "openai_compatible",
              frozenset({CHAT, EMBEDDING})),
    ServerDef("deepseek", "DeepSeek", ("DEEPSEEK_API_BASE_URL",), "DEEPSEEK_API_KEY", "openai_compatible",
              frozenset({CHAT})),
    ServerDef("qwen", "Qwen (Alibaba Cloud)", ("QWEN_API_BASE_URL",), "QWEN_API_KEY", "openai_compatible",
              frozenset({CHAT, EMBEDDING})),
    ServerDef("glm", "GLM (Z.ai)", ("GLM_API_BASE_URL",), "GLM_API_KEY", "openai_compatible",
              frozenset({CHAT})),
    ServerDef("local", "Serveur local", ("LOCAL_API_BASE_URL",), "LOCAL_API_KEY", "openai_compatible",
              frozenset({CHAT, OCR, EMBEDDING, AUDIO})),
)
"""Serveurs connus, dans l'ordre du bloc 1. OpenAI et OpenRouter ne font pas
d'OCR (décision D7 : pas d'OCR par OpenAI Vision)."""

LOCAL_ENGINE_DEF = ServerDef("local_engine", "Moteurs OCR internes", (), "", "local_engine", frozenset({OCR}))
"""Pseudo-serveur du mot réservé ``local`` (Docling, PyMuPDF) : aucune clé."""


@dataclass(frozen=True)
class ServiceDef:
    """Un service et ses deux variables (serveur, modèle)."""

    key: str
    label: str
    server_var: str
    model_var: str
    family: str


SERVICES: Tuple[ServiceDef, ...] = (
    ServiceDef("recode", "Recodage des chunks", "LLM_RECODE_SERVER", "LLM_RECODE_MODEL", CHAT),
    ServiceDef("notes", "Notes Zotero", "LLM_NOTES_SERVER", "LLM_NOTES_MODEL", CHAT),
    ServiceDef("book", "Fiches de lecture", "LLM_BOOK_SERVER", "LLM_BOOK_MODEL", CHAT),
    ServiceDef("citations", "Filtre des citations", "LLM_CITATIONS_SERVER", "LLM_CITATIONS_MODEL", CHAT),
    ServiceDef("ocr", "OCR", "OCR_SERVER", "OCR_MODEL", OCR),
    ServiceDef("ocr_fallback", "OCR de repli", "OCR_SERVER_FALLBACK", "OCR_MODEL_FALLBACK", OCR),
    ServiceDef("embedding", "Embeddings", "EMBEDDING_SERVER", "EMBEDDING_MODEL", EMBEDDING),
    ServiceDef("audio", "Transcription audio", "AUDIO_SERVER", "AUDIO_MODEL", AUDIO),
)
SERVICES_BY_KEY = {s.key: s for s in SERVICES}

DEFAULT_SERVER_VAR = "LLM_DEFAULT_SERVER"
DEFAULT_MODEL_VAR = "LLM_DEFAULT_MODEL"


class ServiceConfigError(ValueError):
    """Configuration d'un service incomplète ou incohérente.

    Attributes:
        service: Clé du service (``recode``…).
        variables: Variables à déclarer ou à corriger, pour le message et
            l'interface.
    """

    def __init__(self, service: str, message: str, variables: Tuple[str, ...] = ()):
        super().__init__(message)
        self.service = service
        self.variables = variables


@dataclass(frozen=True)
class ServiceChoice:
    """Couple résolu pour un service.

    Attributes:
        service: Clé du service.
        server_url: Adresse normalisée du serveur (``local`` pour les moteurs
            internes).
        model: Nom du modèle, envoyé tel quel au serveur.
        server: Définition du serveur reconnu (clé, protocole).
        model_from_default: Vrai si le modèle vient de ``LLM_DEFAULT_MODEL``
            alors que le serveur a été choisi pour le service : les noms de
            modèle diffèrent d'un serveur à l'autre, l'interface avertit.
    """

    service: str
    server_url: str
    model: str
    server: ServerDef
    model_from_default: bool = False

    @property
    def key_var(self) -> str:
        """Variable de la clé à envoyer (``""`` pour les moteurs internes)."""
        return self.server.key_var


def _has_bad_chars(value: str) -> bool:
    """Vrai si ``value`` contient un blanc, un caractère de contrôle ou un antislash."""
    return any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 or ch == "\\" for ch in value)


def normalize_url(value: str) -> str:
    """Normalise une adresse de serveur pour la comparer aux adresses déclarées.

    Schéma et hôte en minuscules, ``/`` final retiré ; le chemin garde sa casse.

    Args:
        value: Adresse saisie (``https://OpenRouter.ai/api/v1/``).

    Returns:
        L'adresse normalisée (``https://openrouter.ai/api/v1``).

    Raises:
        ValueError: Adresse vide, avec blanc ou caractère de contrôle, schéma
            autre que http(s), ou sans hôte.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError("adresse vide")
    if _has_bad_chars(text):
        raise ValueError("adresse invalide : blanc, caractère de contrôle ou antislash")
    parts = urlsplit(text)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError(f"adresse invalide (http ou https attendu) : {text}")
    if parts.query or parts.fragment:
        raise ValueError(f"adresse invalide (paramètres ou ancre interdits) : {text}")
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"


def declared_servers(values: Mapping[str, str]) -> Tuple[Tuple[str, ServerDef], ...]:
    """Adresses déclarées au bloc 1, normalisées : ``((adresse, serveur), …)``.

    Une adresse vide est ignorée (serveur non déclaré) ; le premier nom non vide
    de ``base_url_vars`` l'emporte.

    Raises:
        ValueError: Deux serveurs déclarent la même adresse, ou une adresse
            déclarée est invalide.
    """
    found = []
    owners = {}
    for sdef in SERVER_DEFS:
        raw = next((values.get(var, "") for var in sdef.base_url_vars if (values.get(var) or "").strip()), "")
        if not raw.strip():
            continue
        try:
            url = normalize_url(raw)
        except ValueError as exc:
            raise ValueError(f"{sdef.base_url_vars[0]} : {exc}") from exc
        if url in owners:
            raise ValueError(f"adresse déclarée deux fois ({owners[url]}, {sdef.label}) : {url}")
        owners[url] = sdef.label
        found.append((url, sdef))
    return tuple(found)


def _without_v1(url: str) -> str:
    """Adresse normalisée sans le suffixe ``/v1``."""
    return url[: -len("/v1")] if url.endswith("/v1") else url


def match_server(value: str, values: Mapping[str, str]) -> Tuple[str, ServerDef]:
    """Reconnaît le serveur désigné par ``value`` parmi ceux du bloc 1.

    Mistral s'écrit avec ou sans ``/v1``, comme ``MISTRAL_API_BASE_URL`` ; les
    autres serveurs exigent l'adresse exacte.

    Args:
        value: Adresse du serveur, ou le mot réservé ``local``.
        values: Valeurs des variables (``.env`` effectif).

    Returns:
        ``(adresse déclarée au bloc 1, normalisée ; serveur)``.

    Raises:
        ValueError: Adresse invalide ou non déclarée au bloc 1.
    """
    if (value or "").strip().lower() == LOCAL_KEYWORD:
        return LOCAL_KEYWORD, LOCAL_ENGINE_DEF
    url = normalize_url(value)
    for declared, sdef in declared_servers(values):
        if declared == url or (sdef.protocol == "mistral" and _without_v1(declared) == _without_v1(url)):
            return declared, sdef
    raise ValueError(
        f"serveur inconnu : {url}. Déclarer cette adresse au bloc 1 du .env "
        f"(<FOURNISSEUR>_API_BASE_URL), avec sa clé.")


def _first(*candidates: Optional[str]) -> Tuple[str, int]:
    """Première valeur non vide (après ``strip``) et son rang (``-1`` si aucune)."""
    for rank, candidate in enumerate(candidates):
        text = (candidate or "").strip()
        if text:
            return text, rank
    return "", -1


def resolve_service(
    service: str,
    values: Mapping[str, str],
    *,
    override_server: Optional[str] = None,
    override_model: Optional[str] = None,
    user_values: Optional[Mapping[str, str]] = None,
) -> ServiceChoice:
    """Résout le couple serveur + modèle d'un service.

    Serveur et modèle se résolvent indépendamment, du plus précis au plus
    général : champ de l'étape, choix personnel, variable du service, valeur
    par défaut. Le modèle par défaut ne vaut que pour les services de chat.

    Args:
        service: Clé du service (``recode``, ``ocr``…).
        values: Valeurs du ``.env`` effectif.
        override_server: Serveur choisi pour cette exécution (champ de l'étape).
        override_model: Modèle choisi pour cette exécution.
        user_values: Choix personnels de l'utilisateur (mêmes noms de variables).

    Returns:
        Le couple résolu.

    Raises:
        ServiceConfigError: Service inconnu, serveur ou modèle absent, serveur
            non déclaré ou incapable de rendre le service, modèle interne
            inconnu.
    """
    sdef = SERVICES_BY_KEY.get(service)
    if sdef is None:
        raise ServiceConfigError(service, f"service inconnu : {service}")
    user = user_values or {}
    server, server_rank = _first(override_server, user.get(sdef.server_var), values.get(sdef.server_var),
                                 values.get(DEFAULT_SERVER_VAR))
    if not server:
        raise ServiceConfigError(
            service, f"{sdef.label} : aucun serveur déclaré ({sdef.server_var} ou {DEFAULT_SERVER_VAR}).",
            (sdef.server_var, DEFAULT_SERVER_VAR))
    default_model = values.get(DEFAULT_MODEL_VAR) if sdef.family == CHAT else None
    model, model_rank = _first(override_model, user.get(sdef.model_var), values.get(sdef.model_var), default_model)
    if not model:
        wanted = (sdef.model_var, DEFAULT_MODEL_VAR) if sdef.family == CHAT else (sdef.model_var,)
        raise ServiceConfigError(service, f"{sdef.label} : aucun modèle déclaré ({' ou '.join(wanted)}).", wanted)
    if _has_bad_chars(model):
        raise ServiceConfigError(service, f"{sdef.label} : nom de modèle invalide.", (sdef.model_var,))
    try:
        url, server_def = match_server(server, values)
    except ValueError as exc:
        raise ServiceConfigError(service, f"{sdef.label} : {exc}", (sdef.server_var,)) from exc
    if sdef.family not in server_def.families:
        raise ServiceConfigError(
            service, f"{sdef.label} : {server_def.label} ne sait pas rendre ce service ; déclarer {sdef.server_var}.",
            (sdef.server_var,))
    if server_def is LOCAL_ENGINE_DEF and model not in LOCAL_ENGINES:
        raise ServiceConfigError(
            service, f"{sdef.label} : moteur interne inconnu « {model} » (attendu : {', '.join(LOCAL_ENGINES)}).",
            (sdef.model_var,))
    server_is_specific = server_rank < 3
    model_from_default = sdef.family == CHAT and model_rank == 3 and server_is_specific
    return ServiceChoice(service, url, model, server_def, model_from_default)


def split_legacy(value: Optional[str], *, openai_url: str, openrouter_url: str, albert_url: str) -> Tuple[str, str]:
    """Sépare une ancienne valeur de modèle en ``(serveur, modèle)`` (décision D2).

    Conserve l'acheminement d'avant le sprint :

    * ``albert/<id>`` (préfixe sans casse) → Albert, ``<id>`` ;
    * un nom contenant ``/`` (``google/gemini-2.5-flash``, ``openai/gpt-4o``,
      ``mistralai/…``) → OpenRouter, nom inchangé ;
    * un nom nu (``gpt-4o-mini``) → OpenAI, nom inchangé ;
    * vide → ``("", "")``.

    Args:
        value: Ancienne valeur (``OPENROUTER_DEFAULT_MODEL``, champ de session…).
        openai_url: Adresse OpenAI à écrire.
        openrouter_url: Adresse OpenRouter à écrire.
        albert_url: Adresse Albert à écrire.

    Returns:
        ``(adresse du serveur, nom du modèle)``.

    Raises:
        ValueError: ``albert/`` sans identifiant.
    """
    text = (value or "").strip()
    if not text:
        return "", ""
    if text[:7].lower() == "albert/":
        model = text[7:].strip()
        if not model:
            raise ValueError("modèle Albert vide après « albert/ »")
        return albert_url, model
    if "/" in text:
        return openrouter_url, text
    return openai_url, text
