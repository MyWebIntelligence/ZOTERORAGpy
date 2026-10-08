"""Choix personnels de serveur et de modèle (sprint « configuration unifiée », lot L9).

* ``GET /users/me/settings`` : choix de l'utilisateur, couple effectif de chaque
  service (après héritage du ``.env``), serveurs déclarés ;
* ``PUT /users/me/settings`` : choix validés (serveur déclaré au bloc 1, modèle
  sans blanc), écrits dans ``user_settings`` ; une valeur vide hérite du ``.env``.

Aucune clé ne transite ici ; les clés restent dans « Mes identifiants ».
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.middleware.auth import get_current_active_user
from app.models.audit import AuditAction, create_audit_log
from app.models.user import User
from app.services.user_settings import (
    PERSONAL_SETTING_NAMES, declared_server_choices, effective_services, get_user_settings, service_labels,
    set_user_settings, unified_mode, validate_user_settings,
)

router = APIRouter(prefix="/users/me/settings", tags=["user-settings"])


def _payload(db: Session, user: User) -> Dict[str, Any]:
    """État complet des choix de l'utilisateur pour la page et les étapes."""
    prefs = get_user_settings(db, user)
    return {
        "unified": unified_mode(),
        "settings": {name: prefs.get(name, "") for name in PERSONAL_SETTING_NAMES},
        "services": effective_services(prefs),
        "labels": service_labels(),
        "servers": declared_server_choices(),
    }


@router.get("")
def get_my_settings(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Choix personnels, couples effectifs et serveurs déclarés."""
    return _payload(db, current_user)


class MySettingsUpdate(BaseModel):
    """Choix modifiés : nom (``LLM_NOTES_MODEL``…) → valeur (vide = héritage du ``.env``)."""

    changes: Dict[str, Optional[str]]


@router.put("")
def put_my_settings(
    payload: MySettingsUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Valide puis enregistre les choix ; refus global (400) si l'un est invalide."""
    accepted, errors = validate_user_settings(payload.changes)
    if errors:
        return JSONResponse(status_code=400, content={"errors": errors})
    names = set_user_settings(db, current_user, accepted)
    create_audit_log(
        db=db, action=AuditAction.USER_UPDATE, user_id=current_user.id, resource_type="user_settings",
        resource_id=current_user.id, details={"updated_settings": names},
        ip_address=request.client.host if request.client else None,
    )
    return dict(_payload(db, current_user), changed=names)
