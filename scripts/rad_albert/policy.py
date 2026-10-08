"""Politique d'inférence Albert (``ALBERT_DATA_POLICY``) — stdlib seulement.

Deux politiques, réglage **serveur** (jamais assoupli par une requête) :

* ``compatible`` (défaut) : les fournisseurs historiques (OpenAI, OpenRouter,
  Mistral direct) restent disponibles à côté d'Albert ; comportement
  inchangé.
* ``albert_only`` : l'inférence passe **uniquement** par Albert ; le traitement
  local (OCR Docling, extraction PyMuPDF, spaCy, clustering) reste autorisé.
  Un modèle de chat sans préfixe ``albert/`` est refusé ; un modèle vide prend
  le défaut Albert du rôle ; des embeddings OpenAI sont refusés ; les maillons
  OCR Mistral et OpenAI sont sautés. Exiger ``albert_only`` alors qu'Albert est
  désactivé est une erreur de configuration explicite, jamais un retour
  silencieux au fournisseur historique.

Les exports vers Zotero ou une base vectorielle externe ne relèvent pas de
cette politique (ce sont des destinations de stockage, pas de l'inférence) :
l'interface doit le dire, sans promettre que « tout reste chez Albert ».

Les fonctions reçoivent la configuration (``AlbertConfig`` ou tout objet à
attributs ``enabled`` et ``data_policy``) ; aucune ne lit l'environnement.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

POLICY_COMPATIBLE = "compatible"
POLICY_ALBERT_ONLY = "albert_only"

ALBERT_PREFIX = "albert/"

BLOCKED_PROVIDERS = ("openai", "openrouter", "mistral")
"""Fournisseurs d'inférence refusés en ``albert_only``."""

ROLE_DEFAULTS = {
    "recode": "recode",
    "citation": "citation",
    "notes": "notes",
    "book_structure": "book_structure",
}
"""Rôle de chat → rôle du catalogue dont le modèle primaire sert de défaut."""


class AlbertPolicyError(ValueError):
    """Requête refusée par la politique ``albert_only`` (ou configuration incohérente).

    Attributes:
        capability: capacité concernée (``chat``, ``embeddings``, ``ocr``, ``config``).
        reason: motif court (``policy_violation`` ou ``policy_requires_albert``).
    """

    def __init__(self, message: str, *, capability: str, reason: str = "policy_violation") -> None:
        """Construit l'erreur.

        Args:
            message: message français destiné à l'utilisateur.
            capability: capacité concernée.
            reason: motif court (``policy_violation`` ou ``policy_requires_albert``).
        """
        super().__init__(message)
        self.capability = capability
        self.reason = reason


def effective_policy(cfg: Any) -> str:
    """Politique effective de ``cfg`` (``compatible`` si absente ou inconnue).

    Args:
        cfg: configuration Albert (attribut ``data_policy``).

    Returns:
        ``compatible`` ou ``albert_only``.
    """
    value = str(getattr(cfg, "data_policy", POLICY_COMPATIBLE) or POLICY_COMPATIBLE).strip().lower()
    return value if value in (POLICY_COMPATIBLE, POLICY_ALBERT_ONLY) else POLICY_COMPATIBLE


def is_strict(cfg: Any) -> bool:
    """Vrai pour la politique ``albert_only``."""
    return effective_policy(cfg) == POLICY_ALBERT_ONLY


def check_configuration(cfg: Any) -> None:
    """Refuse une politique ``albert_only`` posée alors qu'Albert est désactivé.

    Args:
        cfg: configuration Albert (``enabled``, ``data_policy``).

    Raises:
        AlbertPolicyError: ``albert_only`` avec ``ALBERT_ENABLED=0`` (motif
            ``policy_requires_albert``) : aucun traitement n'est lancé.
    """
    if is_strict(cfg) and not bool(getattr(cfg, "enabled", False)):
        raise AlbertPolicyError(
            "ALBERT_DATA_POLICY=albert_only exige Albert (ALBERT_ENABLED=1) : aucun traitement "
            "n'est lancé, et aucun retour automatique vers OpenAI, OpenRouter ou Mistral n'a lieu.",
            capability="config",
            reason="policy_requires_albert",
        )


def is_albert_model(model: Any) -> bool:
    """Vrai si ``model`` porte le préfixe ``albert/`` (casse ignorée)."""
    text = "" if model is None else str(model).strip()
    return text[: len(ALBERT_PREFIX)].lower() == ALBERT_PREFIX


def default_albert_model(role: str, cfg: Any, *, today: Optional[date] = None) -> str:
    """Modèle Albert par défaut d'un rôle, sous la forme ``albert/<id>``.

    Args:
        role: rôle de chat (``recode``, ``citation``, ``notes``, ``book_structure``).
        cfg: configuration Albert (réservé : le défaut vient du catalogue).
        today: date de référence des chaînes de repli (défaut : aujourd'hui).

    Returns:
        Par exemple ``albert/ministral-3-8b-instruct-2512``.
    """
    from . import catalog  # import paresseux : catalogue stdlib

    catalog_role = ROLE_DEFAULTS.get(role, "recode")
    primary = catalog.primary_model(catalog_role, today=today or date.today())
    return ALBERT_PREFIX + (primary or "ministral-3-8b-instruct-2512")


def check_chat_model(model: Any, cfg: Any, *, role: str = "recode", today: Optional[date] = None) -> Optional[str]:
    """Modèle de chat autorisé par la politique.

    ``compatible`` : la valeur est renvoyée telle quelle (``None`` compris).
    ``albert_only`` : une valeur vide devient le défaut Albert du rôle ; une
    valeur sans préfixe ``albert/`` est refusée.

    Args:
        model: modèle saisi (``None`` ou vide = défaut).
        cfg: configuration Albert.
        role: rôle de chat (choix du défaut).
        today: date de référence (tests).

    Returns:
        Le modèle à utiliser.

    Raises:
        AlbertPolicyError: configuration incohérente, ou modèle non Albert en ``albert_only``.
    """
    if not is_strict(cfg):
        return model
    check_configuration(cfg)
    text = "" if model is None else str(model).strip()
    if not text:
        return default_albert_model(role, cfg, today=today)
    if is_albert_model(text):
        return text
    raise AlbertPolicyError(
        f"Politique albert_only : le modèle « {text} » n'est pas servi par Albert ; choisir un "
        f"modèle « albert/… » (par exemple {default_albert_model(role, cfg, today=today)}) ou "
        "laisser le champ vide.",
        capability="chat",
    )


def check_embedding_provider(provider: Any, cfg: Any) -> Optional[str]:
    """Fournisseur d'embeddings autorisé par la politique.

    ``compatible`` : valeur renvoyée telle quelle. ``albert_only`` : vide →
    ``albert`` ; ``openai`` refusé.

    Args:
        provider: ``openai``, ``albert``, vide ou ``None``.
        cfg: configuration Albert.

    Returns:
        Le fournisseur à utiliser.

    Raises:
        AlbertPolicyError: ``openai`` (ou autre) en ``albert_only``.
    """
    if not is_strict(cfg):
        return provider
    check_configuration(cfg)
    text = "" if provider is None else str(provider).strip().lower()
    if text in ("", "albert"):
        return "albert"
    raise AlbertPolicyError(
        "Politique albert_only : embeddings OpenAI refusés ; utiliser bge-m3 (Albert, 1024 dimensions).",
        capability="embeddings",
    )


def ocr_external_allowed(cfg: Any) -> bool:
    """Vrai si les maillons OCR externes (Mistral direct, OpenAI Vision) peuvent être appelés."""
    return not is_strict(cfg)


def describe(cfg: Any) -> Dict[str, Any]:
    """Résumé de la politique pour l'interface et les journaux (aucun secret).

    Args:
        cfg: configuration Albert.

    Returns:
        ``{policy, strict, blocked_providers, local_allowed, note}``.
    """
    strict = is_strict(cfg)
    blocked: List[str] = list(BLOCKED_PROVIDERS) if strict else []
    note = (
        "Inférence par Albert uniquement ; traitements locaux autorisés. Les exports vers Zotero "
        "ou une base vectorielle externe restent des destinations distinctes."
        if strict
        else "Fournisseurs historiques disponibles à côté d'Albert."
    )
    return {
        "policy": effective_policy(cfg),
        "strict": strict,
        "blocked_providers": blocked,
        "local_allowed": True,
        "note": note,
    }


__all__ = [
    "ALBERT_PREFIX",
    "BLOCKED_PROVIDERS",
    "POLICY_ALBERT_ONLY",
    "POLICY_COMPATIBLE",
    "AlbertPolicyError",
    "check_chat_model",
    "check_configuration",
    "check_embedding_provider",
    "default_albert_model",
    "describe",
    "effective_policy",
    "is_albert_model",
    "is_strict",
    "ocr_external_allowed",
]
