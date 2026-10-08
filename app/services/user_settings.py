"""Choix personnels de serveur et de modèle (sprint « configuration unifiée », lot L9).

Chaque utilisateur peut déclarer, pour chaque service, un serveur (adresse
d'un serveur du bloc 1 du ``.env``) et un modèle ; une valeur vide hérite du
``.env``. La résolution suit la règle d'Amar (``scripts/rad_settings/models.py``) :
champ de l'étape, puis choix personnel, puis variable du service, puis défaut.

Les clés ne sont jamais stockées ici ; un serveur personnel doit être déclaré
par l'administrateur au bloc 1 (aucune adresse libre : pas de clé envoyée
ailleurs, pas de requête vers une adresse arbitraire).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from sqlalchemy.orm import Session

from app.models.user import User
from app.models.user_setting import UserSetting

try:
    from scripts.rad_settings import registry as reg
    from scripts.rad_settings.chat import legacy_mode, routed_model_for, server_values
    from scripts.rad_settings.models import (
        LOCAL_KEYWORD, SERVICES, SERVICES_BY_KEY, ServiceConfigError, match_server, resolve_service,
    )
except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
    from rad_settings import registry as reg
    from rad_settings.chat import legacy_mode, routed_model_for, server_values
    from rad_settings.models import (
        LOCAL_KEYWORD, SERVICES, SERVICES_BY_KEY, ServiceConfigError, match_server, resolve_service,
    )

PERSONAL_SERVICES = ("recode", "notes", "book", "citations", "ocr")
"""Services dont le serveur et le modèle sont déclarables par chaque utilisateur."""

PERSONAL_SETTING_NAMES: Tuple[str, ...] = tuple(
    name
    for service in PERSONAL_SERVICES
    for name in (SERVICES_BY_KEY[service].server_var, SERVICES_BY_KEY[service].model_var)
)
"""Variables personnelles, dans l'ordre du registre."""

MAX_VALUE_LENGTH = 300


def get_user_settings(db: Session, user: User) -> Dict[str, str]:
    """Choix personnels non vides de ``user`` (nom → valeur)."""
    rows = db.query(UserSetting).filter(UserSetting.user_id == user.id).all()
    return {row.name: row.value for row in rows if row.name in PERSONAL_SETTING_NAMES and row.value}


def validate_user_settings(changes: Mapping[str, Optional[str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Valide des choix personnels : ``(retenus, erreurs)``.

    Un serveur doit être une adresse déclarée au bloc 1 (ou ``local`` pour
    l'OCR), capable du service (OCR, chat). Un modèle est un nom sans blanc ni
    caractère de contrôle. Une valeur vide efface le choix (héritage du ``.env``).
    """
    values = server_values(os.environ)
    server_services = {SERVICES_BY_KEY[s].server_var: SERVICES_BY_KEY[s] for s in PERSONAL_SERVICES}
    accepted: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for name, raw in changes.items():
        if name not in PERSONAL_SETTING_NAMES:
            errors[name] = "choix personnel inconnu"
            continue
        value = ("" if raw is None else str(raw)).strip()
        if len(value) > MAX_VALUE_LENGTH or any(ch.isspace() or ord(ch) < 32 for ch in value):
            errors[name] = "valeur invalide (blanc, caractère de contrôle ou trop longue)"
            continue
        service = server_services.get(name)
        if value and service is not None:
            if service.key == "ocr" and value.lower() == LOCAL_KEYWORD:
                value = LOCAL_KEYWORD
            else:
                try:
                    value, sdef = match_server(value, values)
                except ValueError as exc:
                    errors[name] = str(exc)
                    continue
                if service.family not in sdef.families:
                    errors[name] = f"{sdef.label} ne sait pas rendre le service « {service.label} »"
                    continue
        accepted[name] = value
    return accepted, errors


def set_user_settings(db: Session, user: User, accepted: Mapping[str, str]) -> List[str]:
    """Écrit des choix déjà validés ; une valeur vide supprime le choix. Renvoie les noms touchés."""
    existing = {row.name: row for row in db.query(UserSetting).filter(UserSetting.user_id == user.id).all()}
    for name, value in accepted.items():
        row = existing.get(name)
        if not value:
            if row is not None:
                db.delete(row)
            continue
        if row is None:
            db.add(UserSetting(user_id=user.id, name=name, value=value))
        else:
            row.value = value
    db.commit()
    return sorted(accepted)


def effective_services(user_values: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """Couple effectif de chaque service personnel pour un utilisateur (affichage des étapes).

    Returns:
        Service → ``{"server", "server_label", "model", "legacy", "error"}`` ;
        ``legacy`` vrai en mode historique (pas de ``LLM_DEFAULT_SERVER``) et
        pour l'OCR sans modèle déclaré (chaîne historique), ``error``
        renseignée si la configuration est incomplète.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for service in PERSONAL_SERVICES:
        sdef = SERVICES_BY_KEY[service]
        entry: Dict[str, Any] = {"server": "", "server_label": "", "model": "", "legacy": False, "error": None}
        try:
            if service == "ocr":
                values = server_values(os.environ)
                merged = dict(values)
                merged.update({k: v for k, v in user_values.items() if v})
                if not (merged.get(sdef.model_var) or "").strip():
                    entry["legacy"] = True
                else:
                    choice = resolve_service("ocr", values, user_values=user_values)
                    entry.update(server=choice.server_url, model=choice.model,
                                 server_label="moteur interne" if choice.server_url == LOCAL_KEYWORD
                                 else choice.server.label)
            else:
                routed = routed_model_for(service, user_values=user_values)
                if routed is None:
                    entry["legacy"] = True
                else:
                    entry.update(server=routed.choice.server_url, model=routed.choice.model,
                                 server_label=routed.choice.server.label)
        except ServiceConfigError as exc:
            entry["error"] = str(exc)
        out[service] = entry
    out["embedding"] = _effective_embedding()
    return out


def _effective_embedding() -> Dict[str, Any]:
    """Couple des embeddings du serveur (lot L6, affichage de l'étape dense ; pas de choix personnel)."""
    from scripts.rad_settings.capabilities import declared_choice

    entry: Dict[str, Any] = {"server": "", "server_label": "", "model": "", "legacy": False, "error": None}
    try:
        choice = declared_choice("embedding", os.environ)
    except ServiceConfigError as exc:
        entry["error"] = str(exc)
        return entry
    if choice is None:
        entry["legacy"] = True
    else:
        entry.update(server=choice.server_url, model=choice.model, server_label=choice.server.label)
    return entry


def declared_server_choices() -> List[Dict[str, Any]]:
    """Serveurs déclarés au bloc 1 (pour les listes de l'interface), avec les services qu'ils rendent."""
    from scripts.rad_settings.models import declared_servers

    try:
        return [{"key": sdef.key, "label": sdef.label, "url": url, "families": sorted(sdef.families)}
                for url, sdef in declared_servers(server_values(os.environ))]
    except ValueError:
        return []


def service_labels() -> Dict[str, str]:
    """Libellés des services personnels."""
    return {s.key: s.label for s in SERVICES if s.key in PERSONAL_SERVICES}


def unified_mode() -> bool:
    """Vrai si le ``.env`` déclare ``LLM_DEFAULT_SERVER`` (règle serveur + modèle active)."""
    return not legacy_mode()


__all__ = [
    "PERSONAL_SERVICES", "PERSONAL_SETTING_NAMES", "declared_server_choices", "effective_services",
    "get_user_settings", "reg", "service_labels", "set_user_settings", "unified_mode", "validate_user_settings",
]
