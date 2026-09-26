"""Contrôle préalable (preflight) du compte et des modèles Albert avant un traitement long.

``run_preflight(client, *, roles, today, margin_days=7)`` :

1. ``/v1/me`` : compte expiré → ``AlbertAuthError(account_expired)`` ;
   avertissement à moins de 30 jours de l'expiration ; régime du compte (D1) :
   en expérimentation, plafond de 1 000 requêtes par jour et par modèle de
   chat, signalé par un avertissement ; budget et permissions relevés.
2. ``/v1/models`` : chaque modèle des chaînes des rôles demandés est résolu
   (alias → id) et son type vérifié ; un primaire absent est une erreur
   explicite (sauf ``ocr_doc``, d'accès restreint : bascule sur LightOnOCR) ;
   échéances de retrait ou de réexamen proches signalées.

Les résolutions sont transmises au client (``remember_models``) : les noms
demandés, et aussi **tous** les ids et alias de la liste du compte, pour que
le client n'envoie jamais un alias dont l'id est connu (clés de cache
fondées sur l'id, décision 6). Le résultat
est mis en cache par empreinte de clé (``sha256(clé)[:12]``), URL de base
(``client.cache_scope``, propre au client si son transport est injecté),
rôles et date, pendant 600 s. Le module n'importe pas httpx : il n'utilise
que les méthodes ``me()`` et ``models()`` du client.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import catalog
from .errors import AlbertAuthError

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 600.0
"""Durée de vie (secondes) d'un preflight réussi en cache."""

EXPIRY_WARNING_DAYS = 30
"""Avertissement quand le compte expire dans moins de ce nombre de jours."""

EXPERIMENTATION_DAILY_CAP = 1000
"""Plafond journalier de requêtes par modèle de chat en régime d'expérimentation (D1)."""


@dataclass
class PreflightResult:
    """Résultat d'un preflight.

    Attributes:
        roles: rôles contrôlés.
        requested: rôle → modèles demandés, dans l'ordre (configuration comprise).
        chains: rôle → ids disponibles dans l'ordre d'essai (primaire en tête).
        resolved: nom demandé (id ou alias) → id réel.
        names: tous les ids et alias de ``/v1/models`` → id réel
            (``catalog.listing_name_map``), transmis au client.
        unavailable: rôle → modèles de la chaîne absents de ``/v1/models``.
        warnings: avertissements (français), déjà journalisés.
        regime: ``experimentation``, ``production`` ou ``inconnu``.
        daily_request_cap: plafond journalier le plus bas relevé (ou ``None``).
        expires: expiration du compte (timestamp Unix) ou ``None``.
        budget: budget du compte (``None`` = illimité).
        permissions: permissions du compte.
        limits: limites brutes de ``/v1/me``.
        fingerprint: empreinte de la clé (``sha256[:12]``).
        checked_at: horodatage du contrôle.
        cached: vrai si le résultat vient du cache.
        skipped: vrai si ``ALBERT_PREFLIGHT=0``.
    """

    roles: Tuple[str, ...] = ()
    requested: Dict[str, List[str]] = field(default_factory=dict)
    chains: Dict[str, List[str]] = field(default_factory=dict)
    resolved: Dict[str, str] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    unavailable: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    regime: str = "inconnu"
    daily_request_cap: Optional[int] = None
    expires: Optional[float] = None
    budget: Any = None
    permissions: List[str] = field(default_factory=list)
    limits: List[Dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""
    checked_at: float = 0.0
    cached: bool = False
    skipped: bool = False

    def primary(self, role: str) -> Optional[str]:
        """Premier modèle disponible du rôle (``None`` si aucun ou rôle non contrôlé)."""
        chain = self.chains.get(role) or []
        return chain[0] if chain else None

    def available(self, role: str) -> bool:
        """Vrai si le modèle demandé en tête pour le rôle est disponible sur le compte."""
        wanted = self.requested.get(role) or []
        return bool(wanted) and wanted[0] not in (self.unavailable.get(role) or [])

    def summary_line(self) -> str:
        """Ligne de synthèse en français."""
        if self.skipped:
            return "Preflight Albert désactivé (ALBERT_PREFLIGHT=0)."
        parts = [f"{role} → {self.primary(role) or 'aucun modèle'}" for role in self.roles]
        line = f"Preflight Albert : régime {self.regime}"
        if self.daily_request_cap is not None:
            line += f", {self.daily_request_cap} requêtes/jour au plus par modèle"
        if parts:
            line += " ; " + ", ".join(parts)
        if self.warnings:
            line += f" ; {len(self.warnings)} avertissement(s)"
        return line


_CACHE: Dict[Tuple[Any, ...], Tuple[float, PreflightResult]] = {}
_CACHE_LOCK = threading.Lock()


def clear_preflight_cache() -> None:
    """Vide le cache des preflights (tests, rotation de clé)."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Compte
# ---------------------------------------------------------------------------
def _timestamp(value: Any) -> Optional[float]:
    """Timestamp Unix d'une expiration (nombre, chaîne numérique ou date ISO)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _iso(ts: float) -> str:
    """Date ISO (UTC) d'un timestamp."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _check_account(me: Any, now: float, result: PreflightResult) -> None:
    """Contrôle ``/v1/me`` : expiration, budget, régime et plafond journalier (D1).

    Raises:
        AlbertAuthError: compte expiré (motif ``account_expired``).
    """
    me = me if isinstance(me, dict) else {}
    expires = _timestamp(me.get("expires"))
    result.expires = expires
    if expires is not None:
        if expires <= now:
            raise AlbertAuthError(
                reason="account_expired",
                endpoint="/v1/me",
                detail=f"compte expiré le {_iso(expires)}",
            )
        days_left = (expires - now) / 86400.0
        if days_left < EXPIRY_WARNING_DAYS:
            result.warnings.append(
                f"Le compte Albert expire le {_iso(expires)} (dans {int(days_left)} jour(s)) : "
                "demander son renouvellement à albert.api@numerique.gouv.fr."
            )
    result.budget = me.get("budget")
    if isinstance(result.budget, (int, float)) and not isinstance(result.budget, bool) and result.budget <= 0:
        result.warnings.append(
            "Budget Albert nul ou épuisé d'après /v1/me : les appels risquent d'être refusés (HTTP 400)."
        )
    permissions = me.get("permissions")
    result.permissions = [str(p) for p in permissions] if isinstance(permissions, list) else []
    limits = me.get("limits")
    result.limits = [dict(item) for item in limits if isinstance(item, dict)] if isinstance(limits, list) else []
    daily = []
    for item in result.limits:
        value = item.get("value")
        if item.get("type") == "rpd" and isinstance(value, (int, float)) and not isinstance(value, bool):
            daily.append(int(value))
    if daily:
        result.daily_request_cap = min(daily)
        if result.daily_request_cap <= EXPERIMENTATION_DAILY_CAP:
            result.regime = "experimentation"
            result.warnings.append(
                "Compte Albert en régime d'expérimentation : "
                f"{result.daily_request_cap} requêtes par jour au plus par modèle de chat "
                "(un recodage de masse ou l'OCR d'un gros livre peut l'atteindre) ; "
                "le passage en production se demande à albert.api@numerique.gouv.fr."
            )
        else:
            result.regime = "production"


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def _role_models(role: str, cfg: Any, day: date) -> List[str]:
    """Modèles essayés pour un rôle (surcharges de configuration comprises)."""
    chain = catalog.fallback_chain(role, today=day)
    if cfg is None:
        return chain
    if role == "embed":
        model = getattr(cfg, "embed_model", None) or (chain[0] if chain else None)
        return [model] if model else []
    if role == "ocr_chat":
        return [getattr(cfg, "ocr_chat_model", None) or (chain[0] if chain else "lightonocr-2-1b")]
    if role == "ocr_doc":
        doc_model = getattr(cfg, "ocr_doc_model", None) or (chain[0] if chain else None)
        chat_model = getattr(cfg, "ocr_chat_model", None) or (chain[-1] if chain else None)
        return [m for m in dict.fromkeys([doc_model, chat_model]) if m]
    return chain


def _check_role(role: str, listing: Any, cfg: Any, day: date, margin_days: int, result: PreflightResult) -> None:
    """Résout et vérifie la chaîne d'un rôle.

    Raises:
        catalog.ModelNotFoundError: primaire absent (rôle non optionnel).
        AlbertPermanentError: type de modèle incompatible avec le point d'accès.
    """
    spec = catalog.role_spec(role)
    requested = _role_models(spec.name, cfg, day)
    result.requested[spec.name] = list(requested)
    available: List[str] = []
    missing: List[str] = []
    for name in requested:
        try:
            ident = catalog.resolve_model(name, listing)
        except catalog.ModelNotFoundError:
            missing.append(name)
            continue
        endpoint = "chat" if (spec.name == "ocr_doc" and name != requested[0]) else spec.endpoint
        catalog.check_endpoint_type(ident, endpoint, listing=listing)
        result.resolved[name] = ident
        if ident not in available:
            available.append(ident)
    result.chains[spec.name] = available
    if missing:
        result.unavailable[spec.name] = missing
    primary_missing = bool(requested) and requested[0] in missing
    if primary_missing and not spec.optional:
        raise catalog.ModelNotFoundError(
            f"Modèle Albert épinglé introuvable pour le rôle {spec.name} : {requested[0]!r} "
            "n'apparaît pas dans /v1/models (retrait ou accès manquant). Aucun repli silencieux : "
            "vérifier la liste des modèles du compte et ajuster la configuration.",
            reason="not_found",
            endpoint="/v1/models",
        )
    if not available:
        raise catalog.ModelNotFoundError(
            f"Aucun modèle Albert disponible pour le rôle {spec.name} "
            f"({', '.join(requested) or 'chaîne vide'}) d'après /v1/models.",
            reason="not_found",
            endpoint="/v1/models",
        )
    if primary_missing:
        result.warnings.append(
            f"Rôle Albert {spec.name} : {requested[0]} absent de /v1/models (accès restreint) ; "
            f"bascule sur {available[0]}."
        )
    for name in missing[1:] if primary_missing else missing:
        result.warnings.append(f"Rôle Albert {spec.name} : modèle de repli {name} absent de /v1/models.")
    result.warnings.extend(
        w for w in catalog.lifecycle_warnings(available, today=day, margin_days=margin_days)
        if w not in result.warnings
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def _normalise_roles(roles: Any) -> Tuple[str, ...]:
    """Rôles validés, sans doublon, dans l'ordre donné."""
    if isinstance(roles, str):
        roles = [roles]
    names = [catalog.role_spec(role).name for role in (roles or ())]
    return tuple(dict.fromkeys(names))


def run_preflight(
    client: Any,
    *,
    roles: Sequence[str],
    today: Any,
    margin_days: int = 7,
    ttl: float = CACHE_TTL_SECONDS,
    use_cache: bool = True,
    clock: Callable[[], float] = time.time,
) -> PreflightResult:
    """Contrôle le compte (``/v1/me``) et les modèles (``/v1/models``) avant un traitement.

    Args:
        client: ``AlbertClient`` (méthodes ``me``, ``models``, ``remember_models``
            et propriété ``key_fingerprint``).
        roles: rôles à contrôler (``recode``, ``notes``, ``embed``…).
        today: date de référence (chaînes de repli, échéances).
        margin_days: horizon (jours) des avertissements de retrait ou de réexamen.
        ttl: durée de vie du cache, en secondes.
        use_cache: réutilise un preflight réussi récent.
        clock: horloge murale (expiration du compte, cache).

    Returns:
        ``PreflightResult`` (avertissements déjà journalisés).

    Raises:
        AlbertAuthError: compte expiré, clé invalide, budget épuisé (erreurs HTTP classées).
        catalog.ModelNotFoundError: modèle épinglé absent du compte.
        AlbertPermanentError: type de modèle incompatible.
        catalog.UnknownRoleError: rôle inconnu.
    """
    role_names = _normalise_roles(roles)
    day = catalog.as_date(today)
    cfg = getattr(client, "cfg", None)
    if cfg is not None and not getattr(cfg, "preflight", True):
        return PreflightResult(roles=role_names, skipped=True)
    fingerprint = str(getattr(client, "key_fingerprint", "") or "")
    scope = getattr(client, "cache_scope", None)
    if not isinstance(scope, tuple):
        scope = (fingerprint, str(getattr(client, "base_url", "") or ""), None)
    cache_key = (scope, role_names, day.isoformat(), int(margin_days))
    now = float(clock())
    if use_cache and fingerprint:
        with _CACHE_LOCK:
            hit = _CACHE.get(cache_key)
        if hit is not None and now - hit[0] < ttl:
            cached = replace(hit[1], cached=True)
            _remember(client, cached.names, cached.resolved)
            return cached

    result = PreflightResult(roles=role_names, fingerprint=fingerprint, checked_at=now)
    _check_account(client.me(), now, result)
    listing = client.models()
    result.names = catalog.listing_name_map(listing)
    for role in role_names:
        _check_role(role, listing, cfg, day, int(margin_days), result)
    _remember(client, result.names, result.resolved)
    for message in result.warnings:
        logger.warning("%s", message)
    if use_cache and fingerprint:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (now, result)
    return result


def _remember(client: Any, *mappings: Dict[str, str]) -> None:
    """Transmet les résolutions (nom → id) au client s'il sait les mémoriser, dans l'ordre."""
    remember = getattr(client, "remember_models", None)
    if not callable(remember):
        return
    for mapping in mappings:
        if mapping:
            remember(mapping)


__all__ = [
    "CACHE_TTL_SECONDS",
    "EXPERIMENTATION_DAILY_CAP",
    "EXPIRY_WARNING_DAYS",
    "PreflightResult",
    "clear_preflight_cache",
    "run_preflight",
]
