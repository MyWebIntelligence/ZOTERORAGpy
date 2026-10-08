"""Accès aux réglages : le ``.env`` fait foi, relu quand il change (lot L3).

Avant ce lot, l'application chargeait le ``.env`` une fois (``load_dotenv`` à
l'import de ``app/config.py``) : une modification n'était vue qu'après
redémarrage. Désormais :

* ``load_into_environ()`` remplace ce ``load_dotenv`` : il photographie d'abord
  l'environnement **réel** du processus (shell, ``environment:`` de Compose,
  tests), puis y ajoute les valeurs du fichier ;
* ``refresh_environ()`` (appelé à chaque requête HTTP, avant chaque
  sous-processus et avant chaque tâche Celery) relit le fichier s'il a changé
  (date de modification) et resynchronise ``os.environ`` pour les variables du
  registre de portée ``user`` ou ``server`` : une valeur ajoutée ou modifiée
  est prise, une valeur retirée du fichier est retirée de l'environnement (le
  défaut du code s'applique, comme au démarrage) ;
* une variable présente dans l'environnement réel l'emporte toujours sur le
  fichier, et les variables de déploiement (``deploy``, ``build``) ne sont
  jamais modifiées en cours de route : elles exigent un redémarrage.

Le code existant continue donc de lire ``os.getenv`` ; une valeur lue à
l'import d'un module reste figée jusqu'au redémarrage (attribut ``apply`` du
registre). ``missing_required()`` sert le démarrage strict (décision D3).

Bibliothèque standard seulement.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

from . import dotenv_file
from . import registry as reg

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ENV_FILE_VAR = "RAGPY_ENV_FILE"
"""Variable interne : chemin d'un autre fichier ``.env`` (tests, conteneurs)."""
STRICT_VAR = "RAGPY_SETTINGS_STRICT"
"""Variable interne : ``0`` désactive le démarrage strict (tests, dépannage)."""

_LOCK = threading.Lock()
_STATE: Dict[str, object] = {
    "base": None,      # environnement réel photographié au premier chargement
    "path": None,      # fichier chargé
    "mtime": None,     # (mtime_ns, taille) au dernier chargement
    "values": {},      # valeurs du fichier au dernier chargement
}


def env_file_path() -> Path:
    """Chemin du ``.env`` : ``RAGPY_ENV_FILE`` s'il est défini, sinon la racine du dépôt."""
    override = os.environ.get(ENV_FILE_VAR, "").strip()
    return Path(override) if override else REPO / ".env"


def _stamp(path: Path) -> Optional[Tuple[int, int]]:
    """Date de modification et taille du fichier (``None`` s'il est absent)."""
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


def read_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Valeurs du ``.env`` (``{}`` s'il est absent ou illisible).

    Lecture par python-dotenv quand il est installé (même sémantique que
    l'ancien ``load_dotenv``, interpolation ``${…}`` comprise), sinon par
    l'analyseur de ``dotenv_file``.
    """
    target = path or env_file_path()
    if not target.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - dépendance du dépôt
        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            return {}
        return dotenv_file.values(dotenv_file.parse(text))
    return {name: (value or "") for name, value in dotenv_values(target).items()}


def _managed(name: str) -> bool:
    """Vrai si ``refresh_environ`` peut modifier ``name`` en cours de route."""
    setting = reg.BY_NAME.get(name)
    return setting is not None and setting.scope in (reg.USER, reg.SERVER_SCOPE)


def load_into_environ(environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, str]:
    """Charge le ``.env`` dans l'environnement, sans écraser l'environnement réel.

    Équivalent de ``load_dotenv()`` (non-override) au premier appel ; photographie
    l'environnement réel pour que ``refresh_environ`` sache ce qui vient du
    fichier. Les appels suivants se contentent de rafraîchir.

    Args:
        environ: Environnement à remplir (``os.environ`` par défaut).

    Returns:
        Les valeurs lues dans le fichier.
    """
    env = os.environ if environ is None else environ
    with _LOCK:
        if _STATE["base"] is None:
            _STATE["base"] = dict(env)
    return refresh_environ(env, force=True)


def refresh_environ(environ: Optional[MutableMapping[str, str]] = None, *, force: bool = False) -> Dict[str, str]:
    """Resynchronise l'environnement avec le ``.env`` s'il a changé.

    Coût d'un appel sans changement : un ``stat`` du fichier.

    Args:
        environ: Environnement à mettre à jour (``os.environ`` par défaut).
        force: Relire même si la date de modification n'a pas changé.

    Returns:
        Les valeurs du fichier au dernier chargement.
    """
    env = os.environ if environ is None else environ
    path = env_file_path()
    stamp = _stamp(path)
    with _LOCK:
        if _STATE["base"] is None:
            _STATE["base"] = dict(env)
        unchanged = (not force and _STATE["path"] == path and _STATE["mtime"] == stamp)
        if unchanged:
            return dict(_STATE["values"])  # type: ignore[arg-type]
        first_load = _STATE["path"] is None
        previous: Dict[str, str] = dict(_STATE["values"])  # type: ignore[arg-type]
        values = read_env_file(path) if stamp is not None else {}
        base: Dict[str, str] = _STATE["base"]  # type: ignore[assignment]
        changed: List[str] = []
        for name, value in values.items():
            if name in base:
                continue  # l'environnement réel l'emporte toujours
            if first_load:
                if name not in env:
                    env[name] = value
                continue
            if not _managed(name):
                continue  # déploiement : redémarrage nécessaire
            if env.get(name) != value:
                env[name] = value
                changed.append(name)
        if not first_load:
            for name in previous:
                if name not in values and name not in base and _managed(name) and name in env:
                    del env[name]
                    changed.append(name)
        _STATE.update(path=path, mtime=stamp, values=values)
    if changed:
        logger.info("Réglages relus depuis %s : %s", path.name, ", ".join(sorted(changed)))
    return dict(values)


def effective(name: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Valeur effective d'une variable après rafraîchissement (``None`` si absente)."""
    env = os.environ if environ is None else environ
    if environ is None:
        refresh_environ()
    return env.get(name)


def required_names() -> Tuple[str, ...]:
    """Variables à déclarer pour le démarrage strict (D3).

    Ce sont les variables actives montrées sans commentaire dans
    ``.env.example`` ; une variable en commentaire y est facultative.
    """
    return tuple(s.name for s in reg.SETTINGS if s.status == reg.ACTIVE and s.render == "active")


def missing_required(environ: Optional[Mapping[str, str]] = None, file_values: Optional[Mapping[str, str]] = None
                     ) -> List[str]:
    """Variables obligatoires déclarées ni dans l'environnement réel ni dans le ``.env``."""
    env = os.environ if environ is None else environ
    values = read_env_file() if file_values is None else file_values
    return [name for name in required_names() if name not in env and name not in values]


def strict_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Démarrage strict actif sauf ``RAGPY_SETTINGS_STRICT=0``."""
    env = os.environ if environ is None else environ
    return env.get(STRICT_VAR, "1").strip() != "0"


class SettingsIncompleteError(RuntimeError):
    """Démarrage refusé : des variables obligatoires ne sont pas déclarées."""

    def __init__(self, missing: List[str]):
        self.missing = missing
        shown = ", ".join(missing[:12]) + (f" … (+{len(missing) - 12})" if len(missing) > 12 else "")
        super().__init__(
            f"{len(missing)} variable(s) non déclarée(s) dans le .env : {shown}. "
            "Les écrire avec leur valeur actuelle : .venv/bin/python scripts/env_tool.py sync --apply")


def enforce_complete(environ: Optional[Mapping[str, str]] = None) -> None:
    """Refuse le démarrage si une variable obligatoire manque (sauf strict désactivé).

    Raises:
        SettingsIncompleteError: Variables obligatoires absentes.
    """
    if not strict_enabled(environ):
        return
    missing = missing_required(environ)
    if missing:
        raise SettingsIncompleteError(missing)


def _reset_for_tests() -> None:
    """Oublie la photographie et le dernier chargement (tests seulement)."""
    with _LOCK:
        _STATE.update(base=None, path=None, mtime=None, values={})
