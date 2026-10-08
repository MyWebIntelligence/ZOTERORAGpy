"""Couple des embeddings (``EMBEDDING_SERVER`` + ``EMBEDDING_MODEL``) côté web et Celery (lot L6).

Sprint « configuration unifiée » : quand le couple est déclaré (``.env`` ou
champs de l'étape), la phase dense utilise ce seul serveur, sans repli ; la
route n'exige que la clé de ce serveur (aucune pour le serveur local, dont la
clé est facultative) et transmet le couple au script, qui contrôle le modèle
et mesure la dimension. Sans couple (ou en mode historique), ces fonctions
renvoient ``None`` et le champ ``embedding_provider`` garde sa règle.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from app.core.albert_policy import check_configuration, restrict_subprocess_env, strict_policy
from app.core.credentials import build_subprocess_env
from app.models.user import User

try:
    from scripts.rad_albert.policy import AlbertPolicyError
    from scripts.rad_settings.capabilities import declared_choice
    from scripts.rad_settings.models import ServiceChoice
except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
    from rad_albert.policy import AlbertPolicyError
    from rad_settings.capabilities import declared_choice
    from rad_settings.models import ServiceChoice

EMBEDDING_SERVER_ENV = "EMBEDDING_SERVER"
EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"


def explicit_embedding_target(server: Optional[str] = None, model: Optional[str] = None) -> Optional[ServiceChoice]:
    """Couple des embeddings (champs de l'étape, sinon ``.env``), ou ``None`` (règle historique).

    Args:
        server: Champ serveur de l'étape (adresse déclarée au bloc 1).
        model: Champ modèle de l'étape.

    Raises:
        ServiceConfigError: Couple incomplet, serveur non déclaré ou incapable d'embeddings.
    """
    return declared_choice("embedding", os.environ, override_server=(server or "").strip() or None,
                           override_model=(model or "").strip() or None)


def explicit_embedding_required_keys(choice: ServiceChoice) -> List[str]:
    """Clé exigée pour ``choice`` : celle de son serveur, aucune pour le serveur local."""
    key = choice.server.key
    if key == "local":
        return []
    return [f"{key}_api_key"]


def explicit_embedding_env(user: User, choice: ServiceChoice) -> Dict[str, str]:
    """Environnement de la phase dense pour ``user`` avec le couple ``choice``.

    Raises:
        CredentialMissingError: Clé du serveur absente pour cet utilisateur.
        AlbertPolicyError: ``ALBERT_DATA_POLICY=albert_only`` avec un autre
            serveur qu'Albert (``ValueError``), ou Albert désactivé.
    """
    required = explicit_embedding_required_keys(choice)
    if strict_policy():
        check_configuration()
        if choice.server.key != "albert":
            raise AlbertPolicyError(
                f"ALBERT_DATA_POLICY=albert_only : embeddings sur {choice.server.label} refusés "
                "(EMBEDDING_SERVER doit désigner Albert).", capability="embeddings", reason="policy_violation")
        env = restrict_subprocess_env(build_subprocess_env(user, required_keys=required))
    else:
        env = build_subprocess_env(user, required_keys=required)
    env[EMBEDDING_SERVER_ENV] = choice.server_url
    env[EMBEDDING_MODEL_ENV] = choice.model
    return env


__all__ = [
    "EMBEDDING_MODEL_ENV", "EMBEDDING_SERVER_ENV", "explicit_embedding_env",
    "explicit_embedding_required_keys", "explicit_embedding_target",
]
