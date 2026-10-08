"""Couple serveur + modèle d'un service → modèle routé compris par le code (lot L4).

La règle d'Amar (``models.py``) donne, pour chaque service, une adresse de
serveur et un nom de modèle envoyé tel quel. Le reste du code (routes,
``rad_chunk``, notes, fiches, citations, politique Albert, caches) sait déjà
router une chaîne de modèle :

* ``albert/<modèle>`` → Albert ;
* ``<fournisseur>/<modèle>`` → OpenRouter (adresse officielle) ;
* ``<modèle>`` sans ``/`` → OpenAI (adresse officielle).

``route_model`` traduit donc le couple résolu vers ces formes quand elles
suffisent (rien ne change pour OpenAI, OpenRouter et Albert à leurs adresses
officielles) et vers une forme interne ``@<serveur>:<modèle>`` pour tous les
autres serveurs (Mistral, Anthropic, Google, DeepSeek, Qwen, GLM, serveur
local, ou OpenAI/OpenRouter à une autre adresse) : ``rad_providers`` la résout
en client compatible OpenAI sur l'adresse déclarée au bloc 1, avec la clé du
même sous-bloc.

**Transition (décision D2).** Tant que le ``.env`` ne déclare pas
``LLM_DEFAULT_SERVER`` (« mode historique »), ``routed_model_for`` ne traduit
rien et le comportement d'avant le sprint s'applique à l'identique (modèle du
champ, ``OPENROUTER_DEFAULT_MODEL``, défaut ``gpt-4o-mini`` du recodage). Le
lot L10 retire ce mode après la migration des ``.env``.

Bibliothèque standard seulement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .models import (
    DEFAULT_SERVER_VAR,
    LOCAL_KEYWORD,
    SERVER_DEFS,
    ServiceChoice,
    ServiceConfigError,
    normalize_url,
    resolve_service,
)

COMPAT_PREFIX = "@"
"""Préfixe de la forme interne ``@<serveur>:<modèle>``."""

ALBERT_ROUTE_PREFIX = "albert/"

OFFICIAL_BASE_URLS: Mapping[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
"""Adresses officielles pour lesquelles la forme historique du modèle suffit."""

ALBERT_DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
"""Défaut documenté d'``ALBERT_BASE_URL`` (registre, ``AlbertConfig``)."""


def legacy_mode(env: Optional[Mapping[str, str]] = None) -> bool:
    """Vrai tant que le ``.env`` ne déclare pas ``LLM_DEFAULT_SERVER`` (D2)."""
    values = os.environ if env is None else env
    return not (values.get(DEFAULT_SERVER_VAR) or "").strip()


def mistral_api_root(value: Optional[str]) -> str:
    """Racine de l'API Mistral **sans** ``/v1`` (forme attendue par l'OCR historique).

    Accepte les deux écritures de ``MISTRAL_API_BASE_URL`` : ``https://api.mistral.ai``
    (avant le lot L4) et ``https://api.mistral.ai/v1`` (règle des serveurs).
    """
    text = (value or "").strip().rstrip("/")
    if text.endswith("/v1"):
        text = text[: -len("/v1")]
    return text


def mistral_base_url(value: Optional[str]) -> str:
    """Adresse de l'API Mistral **avec** ``/v1`` (forme de la règle des serveurs)."""
    root = mistral_api_root(value)
    return f"{root}/v1" if root else ""


def server_values(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Valeurs à donner à ``models.resolve_service`` : l'environnement, plus deux
    normalisations sans effet sur les valeurs déclarées.

    * ``MISTRAL_API_BASE_URL`` écrit sans ``/v1`` est compris avec ``/v1`` ;
    * Albert activé sans ``ALBERT_BASE_URL`` : adresse par défaut documentée du
      registre.
    """
    values: Dict[str, str] = dict(os.environ if env is None else env)
    if (values.get("MISTRAL_API_BASE_URL") or "").strip():
        values["MISTRAL_API_BASE_URL"] = mistral_base_url(values["MISTRAL_API_BASE_URL"])
    albert_on = (values.get("ALBERT_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")
    if albert_on and not (values.get("ALBERT_BASE_URL") or "").strip():
        values["ALBERT_BASE_URL"] = ALBERT_DEFAULT_BASE_URL
    return values


def route_model(choice: ServiceChoice) -> str:
    """Traduit un couple résolu dans la forme de modèle comprise par le code.

    Args:
        choice: Couple résolu par ``models.resolve_service``.

    Returns:
        ``albert/<modèle>``, la forme historique OpenAI ou OpenRouter (adresse
        officielle), ou ``@<serveur>:<modèle>``.

    Raises:
        ServiceConfigError: Mot réservé ``local`` pour un service de langage.
    """
    key, model, url = choice.server.key, choice.model, choice.server_url
    if url == LOCAL_KEYWORD:
        raise ServiceConfigError(choice.service, "le mot réservé « local » ne vaut que pour l'OCR.")
    if key == "albert":
        return ALBERT_ROUTE_PREFIX + model
    official = OFFICIAL_BASE_URLS.get(key)
    if official and url == normalize_url(official):
        if key == "openai" and "/" not in model and not model.startswith(COMPAT_PREFIX):
            return model
        if key == "openrouter" and "/" in model and model[: len(ALBERT_ROUTE_PREFIX)].lower() != ALBERT_ROUTE_PREFIX:
            return model
    return f"{COMPAT_PREFIX}{key}:{model}"


def parse_compat(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """Décompose ``@<serveur>:<modèle>`` en ``(serveur, modèle)`` ; ``None`` sinon.

    Raises:
        ValueError: Forme ``@`` mal écrite, serveur inconnu ou modèle vide.
    """
    value = "" if text is None else str(text)
    if not value.startswith(COMPAT_PREFIX):
        return None
    key, sep, model = value[len(COMPAT_PREFIX):].partition(":")
    known = {sdef.key for sdef in SERVER_DEFS}
    if not sep or key not in known or not model.strip():
        raise ValueError(f"modèle routé invalide : {value!r} (attendu @<serveur>:<modèle>)")
    return key, model


def compat_base_url(server_key: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Adresse déclarée du serveur ``server_key`` (normalisée).

    Raises:
        ValueError: Serveur inconnu ou adresse non déclarée.
    """
    values = server_values(env)
    sdef = next((s for s in SERVER_DEFS if s.key == server_key), None)
    if sdef is None:
        raise ValueError(f"serveur inconnu : {server_key}")
    raw = next((values.get(var, "") for var in sdef.base_url_vars if (values.get(var) or "").strip()), "")
    if not raw.strip():
        raise ValueError(f"adresse du serveur {sdef.label} non déclarée ({sdef.base_url_vars[0]}).")
    return normalize_url(raw)


def compat_key_var(server_key: str) -> str:
    """Variable de la clé du serveur ``server_key`` (``MISTRAL_API_KEY``…)."""
    sdef = next((s for s in SERVER_DEFS if s.key == server_key), None)
    if sdef is None:
        raise ValueError(f"serveur inconnu : {server_key}")
    return sdef.key_var


@dataclass(frozen=True)
class RoutedChoice:
    """Résultat de ``routed_model_for`` : couple résolu et forme routée."""

    routed_model: str
    choice: ServiceChoice

    @property
    def description(self) -> str:
        """Libellé lisible (« OpenRouter — google/gemini-3.8-flash »)."""
        return f"{self.choice.server.label} — {self.choice.model}"


def routed_model_for(
    service: str,
    model: Optional[str] = None,
    server: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    user_values: Optional[Mapping[str, str]] = None,
) -> Optional[RoutedChoice]:
    """Résout le modèle routé d'un service, ou ``None`` en mode historique.

    Un modèle saisi ``albert/<id>`` sans serveur désigne Albert (alias de
    transition, D2).

    Args:
        service: Clé du service (``recode``, ``notes``, ``book``, ``citations``, ``rag``).
        model: Modèle choisi pour cette exécution (champ de l'étape), ou vide.
        server: Adresse choisie pour cette exécution, ou vide.
        env: Valeurs (``os.environ`` par défaut).
        user_values: Choix personnels de l'utilisateur.

    Returns:
        ``RoutedChoice``, ou ``None`` en mode historique sans serveur imposé.

    Raises:
        ServiceConfigError: Configuration incomplète ou incohérente.
    """
    values = server_values(env)
    if legacy_mode(values) and not (server or "").strip():
        return None
    model_text = (model or "").strip()
    server_text = (server or "").strip()
    if not server_text and model_text[: len(ALBERT_ROUTE_PREFIX)].lower() == ALBERT_ROUTE_PREFIX:
        albert_url = values.get("ALBERT_BASE_URL") or ALBERT_DEFAULT_BASE_URL
        server_text, model_text = albert_url, model_text[len(ALBERT_ROUTE_PREFIX):].strip()
    choice = resolve_service(service, values, override_server=server_text or None,
                             override_model=model_text or None, user_values=user_values)
    return RoutedChoice(route_model(choice), choice)
