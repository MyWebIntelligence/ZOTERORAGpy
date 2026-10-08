"""Moteur OCR déclaré (``OCR_SERVER`` + ``OCR_MODEL``) côté web et Celery (lot L5).

Sprint « configuration unifiée » : quand ``OCR_MODEL`` est déclaré, un seul
moteur sert, sans aucun repli ; le script ``rad_dataframe.py`` le contrôle avant
le premier document. Les routes d'OCR et les tâches Celery n'exigent alors que
la clé de son serveur (aucune pour les moteurs internes ``local`` + ``docling``
ou ``pymupdf`` ni pour le serveur local, dont la clé est facultative). Sans
``OCR_MODEL``, ces fonctions renvoient ``None`` et la cascade historique des
clés s'applique.

Repli déclaré (``OCR_SERVER_FALLBACK`` + ``OCR_MODEL_FALLBACK``, configuration
du serveur) : sa clé est exigée aussi, il doit désigner un autre serveur que le
moteur principal, et la politique ``albert_only`` s'applique à lui comme au
moteur principal.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from app.core.albert_policy import check_configuration, restrict_subprocess_env, strict_policy
from app.core.credentials import build_subprocess_env
from app.models.user import User

try:
    from scripts.rad_albert.policy import AlbertPolicyError
    from scripts.rad_settings.chat import server_values
    from scripts.rad_settings.models import LOCAL_KEYWORD, ServiceChoice, ServiceConfigError, resolve_service
except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
    from rad_albert.policy import AlbertPolicyError
    from rad_settings.chat import server_values
    from rad_settings.models import LOCAL_KEYWORD, ServiceChoice, ServiceConfigError, resolve_service

FALLBACK_VARS = ("OCR_SERVER_FALLBACK", "OCR_MODEL_FALLBACK")

OCR_SERVER_CREDENTIALS = {"mistral": "mistral_api_key", "albert": "albert_api_key"}
"""Identifiant de clé exigé par serveur d'OCR déclaré."""


def explicit_ocr_target(user_values: Optional[Dict[str, str]] = None) -> Optional[ServiceChoice]:
    """Couple OCR déclaré (serveur ou choix personnel, lot L9), ou ``None`` (chaîne historique).

    Args:
        user_values: Choix personnels de l'utilisateur (``OCR_SERVER``, ``OCR_MODEL``).

    Raises:
        ServiceConfigError: Couple incomplet, serveur inconnu ou incapable d'OCR.
    """
    values = server_values(os.environ)
    personal = {k: v for k, v in (user_values or {}).items() if v}
    if not (personal.get("OCR_MODEL") or values.get("OCR_MODEL") or "").strip():
        return None
    choice = resolve_service("ocr", values, user_values=personal)
    explicit_ocr_fallback(choice)  # repli invalide : refus avant tout lancement
    return choice


def explicit_ocr_fallback(choice: ServiceChoice) -> Optional[ServiceChoice]:
    """Repli déclaré (``OCR_SERVER_FALLBACK`` + ``OCR_MODEL_FALLBACK``) du moteur ``choice``, ou ``None``.

    Raises:
        ServiceConfigError: Repli incomplet, serveur incapable d'OCR, ou même
            serveur que le moteur principal (avec ``local`` : même moteur interne).
    """
    values = server_values(os.environ)
    if not any((values.get(name) or "").strip() for name in FALLBACK_VARS):
        return None
    fallback = resolve_service("ocr_fallback", values)
    if fallback.server_url == choice.server_url and (
            choice.server_url != LOCAL_KEYWORD or fallback.model == choice.model):
        raise ServiceConfigError(
            "ocr_fallback", "OCR de repli : déclarer un autre serveur que celui du moteur principal "
            "(ou, avec local, un autre moteur interne).", ("OCR_SERVER_FALLBACK",))
    return fallback


def explicit_ocr_uses_albert(choice: ServiceChoice) -> bool:
    """Vrai si le moteur principal ou son repli déclaré est Albert (délai long des traitements Albert)."""
    fallback = explicit_ocr_fallback(choice)
    return choice.server.key == "albert" or (fallback is not None and fallback.server.key == "albert")


def explicit_ocr_required_keys(choice: ServiceChoice) -> List[str]:
    """Clés exigées pour ``choice`` : celle de son serveur, aucune pour un moteur local."""
    if choice.server_url == "local":
        return []
    key = OCR_SERVER_CREDENTIALS.get(choice.server.key)
    return [key] if key else []


def explicit_ocr_env(user: User, choice: ServiceChoice) -> Dict[str, str]:
    """Environnement du script d'OCR explicite pour ``user``.

    Raises:
        CredentialMissingError: Clé du serveur absente pour cet utilisateur.
        AlbertPolicyError: ``ALBERT_DATA_POLICY=albert_only`` avec un serveur autre
            qu'Albert ou un moteur local (``ValueError``), ou Albert désactivé.
    """
    fallback = explicit_ocr_fallback(choice)
    engines = [choice] + ([fallback] if fallback is not None else [])
    required = list(dict.fromkeys(key for engine in engines for key in explicit_ocr_required_keys(engine)))
    if strict_policy():
        check_configuration()
        for engine in engines:
            if engine.server.key not in ("albert", "local") and engine.server_url != "local":
                raise AlbertPolicyError(
                    f"ALBERT_DATA_POLICY=albert_only : OCR sur {engine.server.label} refusé "
                    "(OCR_SERVER et OCR_SERVER_FALLBACK doivent désigner Albert ou un moteur local).",
                    capability="ocr", reason="policy_violation")
        env = restrict_subprocess_env(build_subprocess_env(user, required_keys=required))
    else:
        env = build_subprocess_env(user, required_keys=required)
    # Le script lit le couple dans son environnement : celui de l'utilisateur s'il en a choisi un.
    env["OCR_SERVER"] = choice.server_url
    env["OCR_MODEL"] = choice.model
    return env
