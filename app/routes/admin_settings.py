"""API de la page administrateur « Paramètres » (sprint « configuration unifiée », lot L8).

La page déclare **toutes** les variables du registre (``scripts/rad_settings/registry.py``),
dans l'ordre du ``.env`` :

* ``GET /api/admin/settings`` : blocs, sous-blocs et variables actives, avec
  leur valeur (secrets masqués : ``••••`` + quatre derniers caractères), leur
  état (déclarée, invalide, masquée par l'environnement du processus, en
  attente de redémarrage) et la liste des serveurs déclarés ;
* ``PUT /api/admin/settings`` : modifications validées toutes ensemble (nature,
  bornes, adresses de serveur déclarées, retours à la ligne refusés), écrites
  **sur place** dans le ``.env`` (commentaires et ordre conservés, sauvegarde
  ``data/config_backups/``, verrou), relues aussitôt, journalisées (noms
  seulement) ; une variable verrouillée n'est jamais modifiable d'ici ;
* ``POST /api/admin/settings/test-server`` : ``GET {adresse}/models`` d'un
  serveur du bloc 1 avec sa clé du ``.env`` (jamais renvoyée) ;
* ``GET /api/admin/settings/models`` : identifiants de modèles d'un serveur
  déclaré (suggestions des champs « modèle »).

Réservé aux administrateurs. Aucune valeur de secret ne quitte jamais le serveur.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.middleware.auth import require_admin
from app.models.audit import create_audit_log
from app.models.user import User
from scripts.rad_settings import access, dotenv_file
from scripts.rad_settings import registry as reg
from scripts.rad_settings.chat import mistral_base_url, server_values
from scripts.rad_settings.capabilities import ALBERT_ONLY_SERVICES, albert_overlay
from scripts.rad_settings.models import (
    LOCAL_KEYWORD, SERVER_DEFS, SERVICES_BY_KEY, ServiceConfigError, declared_servers, match_server,
)
from scripts.rad_settings.render import TIDY_TITLE, tidy_layout, validate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/settings", tags=["admin-settings"])

MASK_PREFIX = "••••"
"""Préfixe du masque d'un secret (``••••`` + quatre derniers caractères)."""

AUDIT_ACTION = "SETTINGS_UPDATE"
BACKUP_DIR = access.REPO / "data" / "config_backups"
LOCK_PATH = access.REPO / "data" / "locks" / "env.lock"
HTTP_TIMEOUT = 15


def mask(value: Optional[str]) -> str:
    """Masque d'un secret : vide, ou ``••••`` suivi des quatre derniers caractères."""
    if not value:
        return ""
    return MASK_PREFIX + (value[-4:] if len(value) >= 8 else "")


def _setting_view(setting: reg.Setting, file_values: Dict[str, str], real_env: Dict[str, str]) -> Dict[str, Any]:
    """Vue d'une variable pour la page (jamais la valeur d'un secret)."""
    name = setting.name
    declared = name in file_values
    value = file_values.get(name, "")
    from_process = name in real_env and real_env.get(name) != value
    effective = os.environ.get(name)
    pending_restart = (setting.apply != reg.HOT and declared and not from_process
                       and (effective or "") != value)
    level, reason = validate(setting, value) if declared else ("ok", "")
    return {
        "name": name,
        "kind": setting.kind,
        "scope": setting.scope,
        "apply": setting.apply,
        "family": setting.family,
        "choices": list(setting.choices),
        "bool_style": setting.bool_style,
        "doc": setting.doc,
        "default": "" if setting.is_secret else setting.default,
        "default_rule": setting.default_rule,
        "example": "" if setting.is_secret else setting.example_value,
        "locked": setting.locked,
        "optional": setting.render != "active",
        "declared": declared,
        "value": mask(value) if setting.is_secret else value,
        "has_value": bool(value),
        "from_process": from_process,
        "pending_restart": pending_restart,
        "level": level,
        "reason": reason,
    }


@router.get("")
def get_settings(admin: User = Depends(require_admin)) -> Dict[str, Any]:
    """Blocs, sous-blocs et variables actives du registre, dans l'ordre du ``.env``."""
    access.refresh_environ()
    path = access.env_file_path()
    file_values = access.read_env_file(path)
    real_env: Dict[str, str] = dict(access._STATE.get("base") or {})
    blocks: List[Dict[str, Any]] = []
    for block in reg.LAYOUT:
        subblocks = []
        for sub in block.subblocks:
            items = [_setting_view(reg.BY_NAME[s.name], file_values, real_env)
                     for s in sub.settings if reg.BY_NAME[s.name].status == reg.ACTIVE]
            if items:
                subblocks.append({"key": sub.key, "title": sub.title, "doc": sub.doc, "settings": items})
        if subblocks:
            blocks.append({"key": block.key, "title": block.title, "doc": block.doc, "subblocks": subblocks})
    try:
        servers = [{"key": sdef.key, "label": sdef.label, "url": url}
                   for url, sdef in declared_servers(server_values(os.environ))]
    except ValueError as exc:
        servers = []
        logger.warning("Paramètres : serveurs déclarés illisibles (%s)", exc)
    return {"file": path.name, "file_exists": path.exists(), "blocks": blocks, "servers": servers,
            "server_defs": [{"key": s.key, "label": s.label, "url_var": s.base_url_vars[0], "key_var": s.key_var}
                            for s in SERVER_DEFS]}


class SettingsUpdate(BaseModel):
    """Modifications demandées : nom → nouvelle valeur (un masque laisse un secret inchangé)."""

    changes: Dict[str, Optional[str]]


def _check_changes(changes: Dict[str, Optional[str]]) -> tuple:
    """Valide les modifications : ``(retenues, erreurs)``."""
    applied: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for name, raw in changes.items():
        setting = reg.BY_NAME.get(name)
        if setting is None or setting.status != reg.ACTIVE:
            errors[name] = "variable inconnue du registre"
            continue
        if setting.locked:
            errors[name] = f"verrouillée : {setting.locked} (modifiable dans le fichier seulement)"
            continue
        value = "" if raw is None else str(raw)
        if setting.is_secret and value.startswith(MASK_PREFIX):
            continue
        if not setting.is_secret:
            value = value.strip()
        try:
            dotenv_file.format_assignment(name, value)
        except ValueError as exc:
            errors[name] = str(exc)
            continue
        level, reason = validate(setting, value)
        if level == "error":
            errors[name] = reason
            continue
        applied[name] = value
    merged = dict(access.read_env_file())
    merged.update(applied)
    values = server_values(merged)
    for name, value in applied.items():
        if reg.BY_NAME[name].kind != reg.SERVER or not value:
            continue
        if name == "OCR_SERVER" and value.lower() == LOCAL_KEYWORD:
            continue
        try:
            _url, sdef = match_server(value, values)
        except ValueError as exc:
            errors[name] = str(exc)
            continue
        # Lot L6 : le serveur doit savoir rendre le service de la variable
        # (pas de rerank sur OpenRouter, pas d'OCR sur OpenAI…).
        family = reg.BY_NAME[name].family
        if family and family not in sdef.families:
            errors[name] = f"{sdef.label} ne sait pas rendre ce service ({family})."
    # Couples réservés à Albert (réponses sourcées, rerank, transcription) : même
    # contrôle qu'au chargement de la configuration (AlbertConfig.from_env).
    albert_couples = {var for key in ALBERT_ONLY_SERVICES for var in (SERVICES_BY_KEY[key].server_var,
                                                                        SERVICES_BY_KEY[key].model_var)}
    if not errors and albert_couples & set(applied):
        try:
            albert_overlay(values)
        except ServiceConfigError as exc:
            for var in exc.variables or (sorted(albert_couples & set(applied))[0],):
                errors[var] = str(exc)
    return applied, errors


def _write_changes(applied: Dict[str, str]) -> Optional[str]:
    """Écrit les valeurs sur place (lignes remplacées, nouvelles variables rangées). Renvoie la sauvegarde."""
    path = access.env_file_path()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = dotenv_file.parse(text)
    present = {line.name for line in lines if line.kind == dotenv_file.ASSIGN}
    out: List[str] = []
    for line in lines:
        if line.kind == dotenv_file.ASSIGN and line.name in applied:
            out.append(dotenv_file.format_assignment(line.name, applied[line.name]))
        else:
            out.append(line.raw)
    new_names = [name for name in reg.names() if name in applied and name not in present]
    new_text = "\n".join(out).rstrip("\n") + "\n" if out else ""
    if new_names:
        new_text += "\n".join(dotenv_file.format_assignment(n, applied[n]) for n in new_names) + "\n"
        new_text, _report = dotenv_file.tidy(dotenv_file.parse(new_text), tidy_layout(), reg.classify, TIDY_TITLE)
    backup = dotenv_file.write_in_place(path, new_text, backup_dir=BACKUP_DIR, lock_path=LOCK_PATH)
    return backup.name if backup is not None else None


@router.put("")
def put_settings(
    payload: SettingsUpdate,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Valide puis écrit les modifications ; refus global (400) si une seule est invalide."""
    applied, errors = _check_changes(payload.changes)
    if errors:
        return JSONResponse(status_code=400, content={"errors": errors})
    if not applied:
        return {"changed": [], "restart_required": [], "backup": None}
    backup = _write_changes(applied)
    access.refresh_environ()
    names = sorted(applied)
    create_audit_log(
        db, AUDIT_ACTION, user_id=admin.id, resource_type="settings",
        details={"variables": names},
        ip_address=request.client.host if request.client else None,
    )
    logger.info("Paramètres modifiés par l'administrateur %s : %s", admin.id, ", ".join(names))
    restart = [n for n in names if reg.BY_NAME[n].apply != reg.HOT]
    return {"changed": names, "restart_required": restart, "backup": backup}


def _server_url_and_key(server_key: str) -> tuple:
    """Adresse ``…/v1`` et clé (``.env``) d'un serveur du bloc 1."""
    sdef = next((s for s in SERVER_DEFS if s.key == server_key), None)
    if sdef is None:
        raise HTTPException(status_code=400, detail="Serveur inconnu.")
    values = server_values(os.environ)
    raw = next((values.get(var, "") for var in sdef.base_url_vars if (values.get(var) or "").strip()), "")
    if not raw:
        raise HTTPException(status_code=400, detail=f"Adresse non déclarée ({sdef.base_url_vars[0]}).")
    url = mistral_base_url(raw) if server_key == "mistral" else raw.rstrip("/")
    return url, os.environ.get(sdef.key_var) or ""


def _list_models(url: str, key: str) -> List[str]:
    """Identifiants (et alias) de ``GET {url}/models``."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = requests.get(f"{url}/models", headers=headers, timeout=HTTP_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}")
    payload = resp.json()
    entries = payload.get("data", []) if isinstance(payload, dict) else payload
    ids = set()
    for entry in entries or []:
        if isinstance(entry, dict):
            ids.add(str(entry.get("id") or entry.get("name") or ""))
            ids.update(str(a) for a in (entry.get("aliases") or []))
    ids.discard("")
    return sorted(ids)


class ServerTest(BaseModel):
    """Serveur du bloc 1 à éprouver (clé courte : ``mistral``, ``openrouter``…)."""

    server: str


@router.post("/test-server")
def test_server(payload: ServerTest, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    """Interroge ``GET {adresse}/models`` avec la clé du ``.env`` (jamais renvoyée)."""
    url, key = _server_url_and_key(payload.server)
    try:
        models = _list_models(url, key)
    except Exception as exc:  # réponse lisible, sans détail de clé
        return {"ok": False, "url": url, "error": f"{type(exc).__name__} : {exc}"[:300], "has_key": bool(key)}
    return {"ok": True, "url": url, "models": len(models), "has_key": bool(key)}


@router.get("/models")
def server_models(server: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    """Modèles d'un serveur déclaré (``server`` = adresse) : suggestions des champs « modèle »."""
    try:
        url, sdef = match_server(server, server_values(os.environ))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if sdef.key == "local_engine":
        return {"server": url, "models": ["docling", "pymupdf"]}
    url, key = _server_url_and_key(sdef.key)
    try:
        models = _list_models(url, key)
    except Exception as exc:
        return {"server": url, "models": [], "error": f"{type(exc).__name__} : {exc}"[:300]}
    return {"server": url, "models": models[:1000]}
