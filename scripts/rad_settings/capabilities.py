"""Embeddings et transcription : couple serveur + modèle (sprint « configuration unifiée », lot L6).

Même règle que les services de langage (``models.resolve_service``) : un
modèle seul utilise ``LLM_DEFAULT_SERVER``, un serveur propre au service
l'emporte, l'adresse doit être déclarée au bloc 1 et le serveur capable du
service. Il n'y a pas de modèle par défaut : **un modèle déclaré active la
fonction** (décision D11) ; sans aucune des deux variables, le comportement
historique s'applique (``EMBEDDING_PROVIDER``,
``ALBERT_AUDIO_*``), de même en mode historique (pas de ``LLM_DEFAULT_SERVER``).

* Embeddings : OpenAI (adresse officielle), Albert, ou tout serveur
  compatible OpenAI déclaré (Mistral, Google, Qwen, OpenRouter, serveur local) ;
  voir ``rad_providers.EmbeddingConfig.from_env``.
* Transcription : servie par Albert seulement (``/v1/audio/transcriptions``) ;
  un autre serveur est refusé.

Les réponses sourcées et le rerank ont été retirés le 2026-10-03 (RAGpy
prépare les corpus, les questions sont posées par d'autres outils).

``albert_overlay`` traduit ces couples en variables ``ALBERT_*`` lues par
``AlbertConfig.from_env`` (point de passage unique des interrupteurs).
Bibliothèque standard seulement.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from .chat import legacy_mode, server_values
from .models import SERVICES_BY_KEY, ServiceChoice, ServiceConfigError, resolve_service

ALBERT_SERVER_KEY = "albert"

ALBERT_ONLY_SERVICES = {
    "audio": ("ALBERT_AUDIO_MODEL", "ALBERT_AUDIO_ENABLED"),
}
"""Services rendus par Albert seulement : variable du modèle et interrupteur ``ALBERT_*``."""


def declared_choice(
    service: str,
    env: Optional[Mapping[str, str]] = None,
    user_values: Optional[Mapping[str, str]] = None,
    *,
    override_server: Optional[str] = None,
    override_model: Optional[str] = None,
) -> Optional[ServiceChoice]:
    """Couple déclaré d'un service à modèle facultatif, ou ``None`` (comportement historique).

    Args:
        service: ``embedding`` ou ``audio``.
        env: Valeurs lues (``os.environ`` par défaut).
        user_values: Choix personnels (mêmes noms de variables).
        override_server: Serveur du champ de l'étape.
        override_model: Modèle du champ de l'étape.

    Returns:
        Le couple résolu ; ``None`` en mode historique ou si rien n'est déclaré
        (ni variable du service, ni choix personnel, ni champ de l'étape).

    Raises:
        ServiceConfigError: Couple incomplet, serveur non déclaré ou incapable.
    """
    values = server_values(env)
    if legacy_mode(values):
        return None
    sdef = SERVICES_BY_KEY[service]
    personal = {k: v for k, v in (user_values or {}).items() if v}
    declared = [override_server, override_model] + [
        source.get(var) for source in (values, personal) for var in (sdef.server_var, sdef.model_var)]
    if not any((value or "").strip() for value in declared):
        return None
    return resolve_service(service, values, override_server=override_server, override_model=override_model,
                           user_values=personal)


def albert_overlay(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Variables ``ALBERT_*`` imposées par les couples déclarés (mode unifié).

    * embeddings sur Albert : ``ALBERT_EMBED_MODEL`` ;
    * transcription : modèle et interrupteur (``ALBERT_AUDIO_ENABLED=1``).

    Args:
        env: Valeurs lues (``os.environ`` par défaut).

    Returns:
        Les variables à superposer (vide en mode historique ou sans couple).

    Raises:
        ServiceConfigError: Couple invalide, ou transcription sur un
            autre serveur qu'Albert.
    """
    out: Dict[str, str] = {}
    if legacy_mode(env):
        return out
    embedding = declared_choice("embedding", env)
    if embedding is not None and embedding.server.key == ALBERT_SERVER_KEY:
        out["ALBERT_EMBED_MODEL"] = embedding.model
    for service, (model_var, switch_var) in ALBERT_ONLY_SERVICES.items():
        choice = declared_choice(service, env)
        if choice is None:
            continue
        sdef = SERVICES_BY_KEY[service]
        if choice.server.key != ALBERT_SERVER_KEY:
            raise ServiceConfigError(
                service, f"{sdef.label} : seul le serveur Albert est pris en charge ({choice.server.label} refusé) ; "
                f"déclarer {sdef.server_var} à l'adresse d'Albert.", (sdef.server_var,))
        if choice.model_from_default:
            raise ServiceConfigError(
                service, f"{sdef.label} : déclarer {sdef.model_var} (modèle Albert).", (sdef.model_var,))
        out[model_var] = choice.model
        out[switch_var] = "1"
    return out


__all__ = ["ALBERT_ONLY_SERVICES", "albert_overlay", "declared_choice"]
