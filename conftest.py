"""Configuration pytest commune aux trois racines de tests (``tests/``, ``scripts/``, ``app/``).

Ce fichier n'est pas une configuration pytest (aucun ``pytest.ini``) : c'est le
``conftest.py`` de la racine du dépôt, chargé pour toute collecte lancée depuis
la racine. Il garantit qu'aucune session de test n'appelle l'API Albert réelle :

* ``ALBERT_LIVE`` est capturé **à l'import**, avant tout import de l'application,
  et refusé (``pytest.UsageError``) s'il figure dans un fichier ``.env`` ;
* hors mode live, ``pytest_configure`` neutralise l'environnement Albert
  (``ALBERT_ENABLED=0``, ``OCR_ENABLE_ALBERT=0``, ``ALBERT_API_KEY=''``,
  ``EMBEDDING_PROVIDER=''``, ``ALBERT_LIMITER_BACKEND=local``) : un
  ``load_dotenv()`` non-override ne peut plus injecter la clé de production ;
* une garde réseau de session intercepte ``httpx`` (sync et async) et
  ``requests`` : les hôtes locaux passent, ``*.etalab.gouv.fr`` n'est permis
  qu'aux tests ``albert_live`` en mode live, les autres hôtes sont bloqués pour
  les nouveaux tests et seulement comptés pour les tests historiques
  (``data/albert_gates/net_legacy.json``) ;
* une fixture autouse retire les variables Albert, force à OFF les constantes
  Albert des modules déjà importés et fait échouer le test si une requête a été
  bloquée, même quand le code a avalé l'exception ;
* ``anyio_backend`` vaut ``'asyncio'`` et toute variante ``[trio]`` est refusée ;
  les tests asynchrones sont marqués ``@pytest.mark.anyio`` (plugin d'anyio,
  déjà installé avec Starlette) : ``pytest-asyncio`` n'est pas une dépendance ;
* les verrous de ``app.services.job_control`` (``RAGPY_LOCK_DIR``) vont dans un
  dossier temporaire par test ;
* les tests marqués ``albert_live`` sont sautés sauf si ``ALBERT_LIVE=1`` ;
* sans ``TIKTOKEN_CACHE_DIR`` (ni l'ancien ``DATA_GYM_CACHE_DIR``), le cache
  tiktoken est placé **à l'import** dans ``data/tiktoken_cache`` du dépôt
  (persistant, ignoré par git) : le cache par défaut, sous le dossier
  temporaire du système, peut être purgé, et les tests nouveaux échouent alors
  faute de pouvoir le retélécharger. Rien n'est téléchargé ici (tiktoken n'est
  pas importé) ; le préchauffage se fait une fois, hors pytest
  (``.claude/docs/albert.md``, section 10).

Les transports ``httpx.MockTransport`` (``tests/albert_fakes.FakeAlbert``) et le
``TestClient`` de Starlette ne passent pas par ``HTTPTransport`` : la garde ne
les voit pas.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Capture à l'import (avant tout import de ``app``)
# ---------------------------------------------------------------------------
LIVE = os.environ.get("ALBERT_LIVE") == "1"


def _env_files_declaring_live() -> List[str]:
    """Fichiers ``.env`` (racine du dépôt, répertoire courant) qui déclarent ``ALBERT_LIVE``.

    Seuls les **noms** de variables sont examinés ; aucune valeur n'est affichée.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:  # python-dotenv absent : rien à contrôler
        return []
    candidates = []
    for path in (os.path.join(ROOT, ".env"), os.path.join(os.getcwd(), ".env")):
        path = os.path.abspath(path)
        if path not in candidates and os.path.isfile(path):
            candidates.append(path)
    found = []
    for path in candidates:
        try:
            names = dotenv_values(path)
        except Exception:  # fichier illisible : il ne peut rien injecter non plus
            continue
        if "ALBERT_LIVE" in names:
            found.append(path)
    return found


_LIVE_ENV_FILES = _env_files_declaring_live()
if _LIVE_ENV_FILES:
    raise pytest.UsageError(
        "ALBERT_LIVE ne doit jamais figurer dans un fichier .env (trouvé dans : "
        + ", ".join(_LIVE_ENV_FILES)
        + "). Le retirer ; les tests live s'activent uniquement par ALBERT_LIVE=1 exporté dans le shell."
    )

TIKTOKEN_CACHE_ENV = "TIKTOKEN_CACHE_DIR"
TIKTOKEN_LEGACY_CACHE_ENV = "DATA_GYM_CACHE_DIR"
TIKTOKEN_CACHE_DIR = os.path.join(ROOT, "data", "tiktoken_cache")


def _default_tiktoken_cache_dir() -> Optional[str]:
    """Pose ``TIKTOKEN_CACHE_DIR`` sur le cache persistant du dépôt s'il n'est pas choisi.

    Un emplacement déjà choisi (``TIKTOKEN_CACHE_DIR``, même vide, ou l'ancien
    ``DATA_GYM_CACHE_DIR``, que tiktoken lit aussi) est laissé tel quel. Ni
    dossier créé ni tiktoken importé : aucun accès réseau possible ici.

    Returns:
        Le chemin posé, ou ``None`` si l'appelant avait déjà choisi.
    """
    if TIKTOKEN_CACHE_ENV in os.environ or TIKTOKEN_LEGACY_CACHE_ENV in os.environ:
        return None
    os.environ[TIKTOKEN_CACHE_ENV] = TIKTOKEN_CACHE_DIR
    return TIKTOKEN_CACHE_DIR


_default_tiktoken_cache_dir()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LIVE_MARKER = "albert_live"

# Valeurs posées au ``pytest_configure`` hors mode live.
OFF_ENV = {
    "ALBERT_ENABLED": "0",
    "OCR_ENABLE_ALBERT": "0",
    "ALBERT_API_KEY": "",
    "EMBEDDING_PROVIDER": "",
    "ALBERT_LIMITER_BACKEND": "local",
}

# Variables retirées par la fixture autouse (en plus du préfixe ``ALBERT_``).
ALBERT_ENV_PREFIX = "ALBERT_"
ALBERT_ENV_EXTRA = ("OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER")

# Constantes de module (lues à l'import par les scripts CLI) forcées à OFF.
FORCED_MODULE_CONSTANTS = (("OCR_ENABLE_ALBERT", False), ("ALBERT_API_KEY", None))

# Fichiers de tests « nouveaux » : tout hôte externe y est bloqué.
NEW_TEST_PREFIX = "test_albert_"
NEW_TEST_STEMS = frozenset({
    "test_env_isolation",
    "test_debt_regressions",
    "test_celery_tasks",
    "test_vector_space_guards",
    "test_rad_providers",
    "test_gate_tools",
})

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})
ALBERT_DOMAIN = "etalab.gouv.fr"

NET_LEGACY_PATH = os.path.join(ROOT, "data", "albert_gates", "net_legacy.json")

_TRIO_ID_RE = re.compile(r"[\[\-]trio[\]\-]")


# ---------------------------------------------------------------------------
# Garde réseau
# ---------------------------------------------------------------------------
class NetworkGuardError(RuntimeError):
    """Requête réseau refusée par la garde de la session de test."""


@dataclass
class NetRecord:
    """Requête réseau non locale vue par la garde.

    Attributes:
        nodeid: test courant (``None`` hors d'un test : collecte, fixtures de session).
        host: hôte visé (jamais l'URL complète, qui pourrait porter un secret).
        library: ``'httpx'``, ``'httpx-async'`` ou ``'requests'``.
        blocked: ``True`` si la requête a été refusée (exception levée).
        reason: ``'albert'`` (``*.etalab.gouv.fr``) ou ``'external'``.
        consumed: ``True`` une fois acquittée par ``network_guard.consume()``.
    """

    nodeid: Optional[str]
    host: str
    library: str
    blocked: bool
    reason: str
    consumed: bool = False


@dataclass(frozen=True)
class _TestContext:
    """Contexte du test en cours pour la garde réseau."""

    nodeid: str
    new_style: bool
    live_allowed: bool


def _is_local_host(host: str) -> bool:
    """Vrai pour ``localhost``, ``testserver`` et toute adresse de bouclage."""
    if host in LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_albert_host(host: str) -> bool:
    """Vrai pour ``etalab.gouv.fr`` et ses sous-domaines."""
    return host == ALBERT_DOMAIN or host.endswith("." + ALBERT_DOMAIN)


def _normalise_host(host: Any) -> str:
    """Hôte en minuscules, sans crochets IPv6 ni point final."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return str(host or "").strip().strip("[]").rstrip(".").lower()


class NetworkGuard:
    """Garde réseau de session : patch des transports httpx et de l'adaptateur requests."""

    def __init__(self) -> None:
        """Crée une garde non installée, sans enregistrement."""
        self._lock = threading.Lock()
        self._records: List[NetRecord] = []
        self._originals: Dict[Tuple[Any, str], Any] = {}
        self.current: Optional[_TestContext] = None
        self.installed = False

    # --- décision -----------------------------------------------------------
    def check(self, host: Any, library: str) -> None:
        """Autorise, enregistre ou refuse une requête vers ``host``.

        Raises:
            NetworkGuardError: hôte Albert hors test live en mode live, ou hôte
                externe depuis un nouveau test.
        """
        name = _normalise_host(host)
        if _is_local_host(name):
            return
        ctx = self.current
        if _is_albert_host(name):
            if LIVE and ctx is not None and ctx.live_allowed:
                return
            blocked, reason = True, "albert"
        else:
            blocked, reason = bool(ctx is not None and ctx.new_style), "external"
        record = NetRecord(nodeid=ctx.nodeid if ctx else None, host=name, library=library,
                           blocked=blocked, reason=reason)
        with self._lock:
            self._records.append(record)
        if blocked:
            where = ctx.nodeid if ctx else "hors test"
            raise NetworkGuardError(
                f"Garde réseau : requête {library} vers {name} refusée ({where}). "
                "Aucun appel réseau externe en test : utiliser tests/albert_fakes.FakeAlbert ou un MockTransport."
            )

    # --- journal -----------------------------------------------------------
    def records_for(self, nodeid: Optional[str]) -> List[NetRecord]:
        """Enregistrements (bloqués ou non) attribués au test ``nodeid``."""
        with self._lock:
            return [r for r in self._records if r.nodeid == nodeid]

    def consume(self, nodeid: Optional[str]) -> List[NetRecord]:
        """Acquitte et renvoie les enregistrements bloquants non acquittés du test."""
        with self._lock:
            pending = [r for r in self._records if r.nodeid == nodeid and r.blocked and not r.consumed]
            for record in pending:
                record.consumed = True
        return pending

    def legacy_count(self) -> int:
        """Nombre de requêtes externes seulement enregistrées (tests historiques)."""
        with self._lock:
            return sum(1 for r in self._records if not r.blocked)

    def unattributed_blocked(self) -> List[NetRecord]:
        """Requêtes bloquées hors de tout test (collecte, fixtures de session)."""
        with self._lock:
            return [r for r in self._records if r.nodeid is None and r.blocked]

    # --- installation -------------------------------------------------------
    def _patch(self, owner: Any, attr: str, wrapper: Any) -> None:
        """Remplace ``owner.attr`` par ``wrapper`` en mémorisant l'original."""
        self._originals[(owner, attr)] = owner.__dict__[attr]
        setattr(owner, attr, wrapper)

    def install(self) -> None:
        """Installe les patchs sur httpx (sync, async) et requests, s'ils sont importables."""
        if self.installed:
            return
        guard = self
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx est une dépendance du dépôt
            httpx = None
        if httpx is not None:
            sync_original = httpx.HTTPTransport.handle_request
            async_original = httpx.AsyncHTTPTransport.handle_async_request

            def guarded_handle_request(transport, request):
                """``HTTPTransport.handle_request`` précédé du contrôle de la garde."""
                guard.check(request.url.host, "httpx")
                return sync_original(transport, request)

            async def guarded_handle_async_request(transport, request):
                """``AsyncHTTPTransport.handle_async_request`` précédé du contrôle de la garde."""
                guard.check(request.url.host, "httpx-async")
                return await async_original(transport, request)

            self._patch(httpx.HTTPTransport, "handle_request", guarded_handle_request)
            self._patch(httpx.AsyncHTTPTransport, "handle_async_request", guarded_handle_async_request)
        try:
            import requests.adapters
        except ImportError:  # pragma: no cover - requests est une dépendance du dépôt
            requests = None
        if requests is not None:
            send_original = requests.adapters.HTTPAdapter.send

            def guarded_send(adapter, request, *args, **kwargs):
                """``HTTPAdapter.send`` précédé du contrôle de la garde."""
                guard.check(urlsplit(request.url or "").hostname, "requests")
                return send_original(adapter, request, *args, **kwargs)

            self._patch(requests.adapters.HTTPAdapter, "send", guarded_send)
        self.installed = True

    def uninstall(self) -> None:
        """Restaure les méthodes d'origine."""
        for (owner, attr), original in self._originals.items():
            setattr(owner, attr, original)
        self._originals.clear()
        self.installed = False


GUARD = NetworkGuard()


class NetworkGuardHandle:
    """Vue de la garde réseau limitée au test courant (fixture ``network_guard``)."""

    error_class = NetworkGuardError

    def __init__(self, guard: NetworkGuard, nodeid: str) -> None:
        """Associe la garde de session au test ``nodeid``."""
        self._guard = guard
        self.nodeid = nodeid

    def records(self) -> List[NetRecord]:
        """Toutes les requêtes non locales vues pendant ce test."""
        return self._guard.records_for(self.nodeid)

    def blocked(self) -> List[NetRecord]:
        """Requêtes bloquées et non encore acquittées pendant ce test."""
        return [r for r in self.records() if r.blocked and not r.consumed]

    def consume(self) -> List[NetRecord]:
        """Acquitte les requêtes bloquées attendues : le test n'échoue plus au teardown."""
        return self._guard.consume(self.nodeid)


# ---------------------------------------------------------------------------
# Classement des tests
# ---------------------------------------------------------------------------
def _item_basename(item: Any) -> str:
    """Nom de fichier (sans chemin) du test ``item``."""
    path = getattr(item, "path", None) or getattr(item, "fspath", None)
    return os.path.basename(str(path or ""))


def is_new_style_test(item: Any) -> bool:
    """Vrai pour les fichiers de tests du sprint Albert (garde réseau stricte)."""
    name = _item_basename(item)
    stem = name[:-3] if name.endswith(".py") else name
    return stem.startswith(NEW_TEST_PREFIX) or stem in NEW_TEST_STEMS


def _live_allowed(item: Any) -> bool:
    """Vrai si ``item`` est un test ``albert_live`` et que la session est en mode live."""
    return LIVE and item.get_closest_marker(LIVE_MARKER) is not None


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
_SAVED_ENV: Dict[str, Optional[str]] = {}


def pytest_configure(config):
    """Enregistre ``albert_live``, neutralise l'env Albert (hors live) et installe la garde réseau."""
    config.addinivalue_line(
        "markers",
        f"{LIVE_MARKER}: appelle l'API Albert réelle ; sauté sauf si ALBERT_LIVE=1 est exporté dans le shell",
    )
    if not LIVE:
        for name, value in OFF_ENV.items():
            _SAVED_ENV.setdefault(name, os.environ.get(name))
            os.environ[name] = value
    GUARD.install()


def pytest_unconfigure(config):
    """Désinstalle la garde réseau et restaure les variables modifiées au ``pytest_configure``."""
    GUARD.uninstall()
    for name, value in _SAVED_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _SAVED_ENV.clear()


def pytest_collection_modifyitems(config, items):
    """Refuse toute variante trio ; saute les tests ``albert_live`` hors mode live."""
    trio = [item.nodeid for item in items if "[trio]" in item.nodeid or _TRIO_ID_RE.search(item.nodeid)]
    if trio:
        raise pytest.UsageError(
            "Variante trio interdite (anyio_backend = asyncio seulement) : " + ", ".join(trio[:5])
        )
    if LIVE:
        return
    skip_live = pytest.mark.skip(reason="test live Albert : exporter ALBERT_LIVE=1 dans le shell pour l'activer")
    for item in items:
        if item.get_closest_marker(LIVE_MARKER) is not None:
            item.add_marker(skip_live)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Rattache les requêtes réseau (setup, appel, teardown) au test en cours."""
    GUARD.current = _TestContext(
        nodeid=item.nodeid,
        new_style=is_new_style_test(item),
        live_allowed=_live_allowed(item),
    )
    try:
        yield
    finally:
        GUARD.current = None


def pytest_sessionfinish(session, exitstatus):
    """Écrit le compteur des requêtes externes des tests historiques (``net_legacy.json``)."""
    try:
        os.makedirs(os.path.dirname(NET_LEGACY_PATH), exist_ok=True)
        with open(NET_LEGACY_PATH, "w", encoding="utf-8") as fh:
            json.dump({"count": GUARD.legacy_count()}, fh)
            fh.write("\n")
    except OSError:
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Signale les requêtes bloquées hors de tout test (sinon, rien n'est affiché)."""
    stray = GUARD.unattributed_blocked()
    if stray:
        hosts = ", ".join(sorted({r.host for r in stray}))
        terminalreporter.write_line(f"Garde réseau : {len(stray)} requête(s) bloquée(s) hors test ({hosts}).")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_PROJECT_MODULES: Dict[str, bool] = {}
_ROOT_REAL = os.path.realpath(ROOT)
_EXCLUDED_DIRS = tuple(os.path.join(_ROOT_REAL, d) + os.sep for d in (".venv", "tests"))


def _is_project_module(name: str, module: Any) -> bool:
    """Vrai pour un module du dépôt hors tests (``scripts/``, ``app/``…), résultat mis en cache."""
    cached = _PROJECT_MODULES.get(name)
    if cached is not None:
        return cached
    path = getattr(module, "__file__", None)
    result = False
    if isinstance(path, str):
        path = os.path.realpath(path)
        base = os.path.basename(path)
        result = (
            path.startswith(_ROOT_REAL + os.sep)
            and not path.startswith(_EXCLUDED_DIRS)
            and not base.startswith("test_")
            and base != "conftest.py"
        )
    _PROJECT_MODULES[name] = result
    return result


def _force_module_constants_off(monkeypatch) -> None:
    """Force à OFF les constantes Albert des modules du dépôt déjà importés."""
    for name, module in list(sys.modules.items()):
        if module is None or not _is_project_module(name, module):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for attr, value in FORCED_MODULE_CONSTANTS:
            if attr in namespace and namespace[attr] is not value:
                monkeypatch.setattr(module, attr, value)


@pytest.fixture(autouse=True)
def _albert_test_isolation(request, monkeypatch):
    """Isole chaque test d'Albert et de toute requête réseau non voulue.

    Hors test live en mode live : retire ``ALBERT_*``, ``OCR_ENABLE_ALBERT`` et
    ``EMBEDDING_PROVIDER`` (monkeypatch) et force à OFF les constantes Albert des
    modules déjà importés. Au teardown, échoue si la garde a bloqué une requête
    pendant le test sans que ``network_guard.consume()`` l'ait acquittée, même si
    le code testé a avalé l'exception.
    """
    if not _live_allowed(request.node):
        for name in list(os.environ):
            if name.startswith(ALBERT_ENV_PREFIX) or name in ALBERT_ENV_EXTRA:
                monkeypatch.delenv(name, raising=False)
        _force_module_constants_off(monkeypatch)
    yield
    pending = GUARD.consume(request.node.nodeid)
    if pending:
        hosts = ", ".join(sorted({r.host for r in pending}))
        pytest.fail(
            f"Garde réseau : {len(pending)} requête(s) bloquée(s) pendant ce test ({hosts}), "
            "même si l'exception a été interceptée. Aucun appel réseau externe n'est permis en test.",
            pytrace=False,
        )


@pytest.fixture
def network_guard(request):
    """Journal de la garde réseau pour le test courant (``records``, ``blocked``, ``consume``).

    ``consume()`` acquitte les requêtes bloquées attendues par le test ;
    ``error_class`` est l'exception levée par la garde.
    """
    return NetworkGuardHandle(GUARD, request.node.nodeid)


@pytest.fixture(scope="session")
def anyio_backend():
    """Backend anyio unique de la suite : ``'asyncio'`` (aucune variante trio)."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolated_job_locks(tmp_path_factory, monkeypatch):
    """Verrous de ``app.services.job_control`` dans un dossier temporaire propre à chaque test.

    Jamais ``data/locks`` du dépôt : un test ne peut ni bloquer ni être bloqué
    par un autre, ni par un serveur de développement lancé en parallèle.
    """
    monkeypatch.setenv("RAGPY_LOCK_DIR", str(tmp_path_factory.mktemp("job_locks")))
