"""Audit du 2026-09-27 : files Celery et planning Beat (A04), défauts de déploiement (A09).

A04 — le worker fourni consomme toutes les files où publient les tâches, et les
noms du routage et de Beat sont ceux du registre :

* chaque file déclarée par une tâche de ``app/tasks`` figure dans
  ``task_queues`` (un worker lancé sans ``-Q`` la consomme) ;
* chaque tâche du planning Beat existe dans le registre chargé ;
* chaque motif de ``task_routes`` correspond à au moins une tâche ;
* le nettoyage des sessions expirées n'est plus planifié par Beat (une seule
  responsabilité : l'APScheduler de l'application web) ;
* le seuil des processus orphelins dépasse toute limite de temps d'une tâche
  (une tâche Albert légitime n'est jamais tuée) ;
* la commande Compose du worker ne restreint pas les files.

A09 — défauts sûrs :

* ``RAGPY_ENV=production`` refuse le secret JWT de remplacement ou trop court ;
  hors production, le problème est journalisé sans bloquer ;
* une rotation de ``JWT_SECRET_KEY`` garde lisibles les identifiants chiffrés
  sous l'ancien secret (``JWT_SECRET_KEY_PREVIOUS``), et le script de rotation
  les rechiffre ;
* le middleware CORS applique ``CORS_ORIGINS`` (jamais ``*`` avec cookies) ;
* Compose publie Redis et Flower en local par défaut et Flower exige des
  identifiants (plus de ``admin/admin``).

Aucun réseau, aucun broker : le registre Celery est chargé en mémoire.
"""

from __future__ import annotations

import fnmatch
import importlib
import os
import re

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_MODULES = (
    "app.tasks.extraction",
    "app.tasks.chunking",
    "app.tasks.embeddings",
    "app.tasks.vectordb",
    "app.tasks.cleanup",
    "app.tasks.monitoring",
)


# ======================================================================
# A04 — files, routage et Beat
# ======================================================================
@pytest.fixture(scope="module")
def celery_registry():
    """Application Celery avec toutes les tâches de ``app/tasks`` importées (aucun broker)."""
    from app.celery_app import celery_app

    for name in TASK_MODULES:
        importlib.import_module(name)
    tasks = {name: task for name, task in celery_app.tasks.items() if not name.startswith("celery.")}
    return celery_app, tasks


def test_every_task_queue_is_consumed_by_default(celery_registry):
    """La file de chaque tâche figure dans ``task_queues`` : le worker sans ``-Q`` la consomme."""
    app, tasks = celery_registry
    declared = {queue.name for queue in app.conf.task_queues}
    task_queues = {name: getattr(task, "queue", None) or app.conf.task_default_queue for name, task in tasks.items()}
    assert task_queues, "aucune tâche chargée"
    missing = {name: queue for name, queue in task_queues.items() if queue not in declared}
    assert missing == {}
    assert app.conf.task_default_queue in declared


def test_beat_schedule_names_exist_in_registry(celery_registry):
    """Chaque entrée de Beat vise une tâche du registre, sur la file de cette tâche."""
    app, tasks = celery_registry
    for entry, spec in app.conf.beat_schedule.items():
        assert spec["task"] in tasks, (entry, spec["task"])
        queue = (spec.get("options") or {}).get("queue")
        assert queue == tasks[spec["task"]].queue, (entry, queue)


def test_expired_session_cleanup_not_scheduled_by_beat(celery_registry):
    """Une seule responsabilité : le nettoyage des sessions expirées reste à l'APScheduler web."""
    app, _tasks = celery_registry
    scheduled = {spec["task"] for spec in app.conf.beat_schedule.values()}
    assert "cleanup.cleanup_sessions" not in scheduled


def test_task_routes_match_registered_names(celery_registry):
    """Chaque motif de ``task_routes`` correspond à des tâches du registre, routées vers leur propre file."""
    app, tasks = celery_registry
    for pattern, route in app.conf.task_routes.items():
        matched = [name for name in tasks if fnmatch.fnmatch(name, pattern)]
        assert matched, pattern
        for name in matched:
            assert tasks[name].queue == route["queue"], (pattern, name)


def test_orphan_threshold_above_every_time_limit(celery_registry, monkeypatch):
    """Le seuil des orphelins dépasse la limite dure globale et celle d'une tâche Albert."""
    app, _tasks = celery_registry
    from app.tasks import cleanup, runner

    monkeypatch.delenv("ORPHAN_PROCESS_MAX_RUNTIME", raising=False)
    threshold = cleanup.orphan_max_runtime()
    assert threshold > int(app.conf.task_time_limit)
    assert threshold > int(runner.albert_time_limits()["time_limit"])
    monkeypatch.setenv("ORPHAN_PROCESS_MAX_RUNTIME", "30000")
    assert cleanup.orphan_max_runtime() == 30000


def test_cleanup_sessions_task_delegates_to_service(monkeypatch):
    """``cleanup.cleanup_sessions`` (sur demande) exécute le service partagé de l'APScheduler."""
    from app.services import session_cleanup
    from app.tasks.cleanup import cleanup_sessions_task

    calls = []

    def _fake_cleanup():
        """Service simulé (clés réelles du service) : deux sessions nettoyées, une en échec."""
        calls.append(True)
        return {"sessions_found": 3, "sessions_cleaned": 2, "sessions_failed": 1, "details": []}

    monkeypatch.setattr(session_cleanup, "cleanup_expired_sessions", _fake_cleanup)
    result = cleanup_sessions_task.run()
    assert calls == [True]
    assert result["cleaned_count"] == 2 and result["errors"] == 1

    def _failing_cleanup():
        """Service simulé en échec global (clé ``error``)."""
        return {"sessions_found": 0, "sessions_cleaned": 0, "sessions_failed": 0, "error": "base indisponible"}

    monkeypatch.setattr(session_cleanup, "cleanup_expired_sessions", _failing_cleanup)
    result = cleanup_sessions_task.run()
    assert result["errors"] == 1 and result["error"] == "base indisponible"


def test_cleanup_task_reads_the_service_result_keys():
    """Les clés lues par la tâche sont celles que renvoie réellement le service."""
    import inspect

    from app.services import session_cleanup
    from app.tasks import cleanup

    service_source = inspect.getsource(session_cleanup.cleanup_expired_sessions)
    task_source = inspect.getsource(cleanup.cleanup_sessions_task)
    for key in ("sessions_cleaned", "sessions_failed"):
        assert f'"{key}"' in service_source and f'"{key}"' in task_source


@pytest.fixture(scope="module")
def compose():
    """``docker-compose.yml`` du dépôt, lu comme YAML."""
    with open(os.path.join(REPO_ROOT, "docker-compose.yml"), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_compose_worker_consumes_every_queue(compose):
    """La commande du worker ne restreint pas les files (pas de ``-Q`` partiel)."""
    command = compose["services"]["celery_worker"]["command"]
    assert " -Q" not in command and "--queues" not in command


def test_compose_web_uses_the_worker_broker(compose):
    """L'application web soumet sur le même broker interne que le worker."""
    web = compose["services"]["ragpy"]["environment"]
    worker = compose["services"]["celery_worker"]["environment"]
    for name in ("CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
        assert f"{name}=redis://redis:6379/0" in web
        assert f"{name}=redis://redis:6379/0" in worker


# ======================================================================
# A09 — Compose : Redis et Flower
# ======================================================================
def test_compose_redis_bound_to_localhost_by_default(compose):
    """Redis (sans authentification) n'est publié que sur 127.0.0.1 par défaut."""
    ports = compose["services"]["redis"]["ports"]
    assert ports == ["${REDIS_BIND:-127.0.0.1}:6379:6379"]


def test_compose_flower_requires_credentials(compose):
    """Flower : plus de valeurs par défaut ``admin/admin``, refus sans identifiants, publication locale."""
    flower = compose["services"]["flower"]
    command = flower["command"]
    assert "FLOWER_USER:-admin" not in command and "FLOWER_PASSWORD:-admin" not in command
    assert 'exit 1' in command and '"$$FLOWER_PASSWORD" = "admin"' in command
    assert flower["ports"] == ["${FLOWER_BIND:-127.0.0.1}:5555:5555"]


# ======================================================================
# A09 — secret JWT et rotation des identifiants chiffrés
# ======================================================================
def _settings(**overrides):
    """Copie de ``Settings`` avec des attributs remplacés (le module ``settings`` reste intact)."""
    from app.config import Settings

    config = Settings()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize("secret", ["change-this-secret-key-in-production-min-32-chars", "short-secret"])
def test_production_refuses_placeholder_or_short_secret(secret):
    """``RAGPY_ENV=production`` : démarrage refusé avec le secret public ou un secret court."""
    from app.config import enforce_secure_settings

    with pytest.raises(RuntimeError, match="Démarrage refusé"):
        enforce_secure_settings(_settings(RAGPY_ENV="production", JWT_SECRET_KEY=secret))


def test_production_accepts_a_strong_secret():
    """Secret aléatoire de 48 caractères en production : aucun problème, aucun refus."""
    from app.config import enforce_secure_settings

    problems = enforce_secure_settings(_settings(
        RAGPY_ENV="production", JWT_SECRET_KEY="x" * 48, CORS_ORIGINS=["https://ragpy.example.org"],
    ))
    assert problems == []


def test_development_logs_placeholder_without_refusing(caplog):
    """Hors production : le secret de remplacement est signalé (ERROR) sans bloquer le démarrage."""
    from app.config import DEFAULT_JWT_SECRET_KEY, enforce_secure_settings

    with caplog.at_level("ERROR"):
        problems = enforce_secure_settings(_settings(RAGPY_ENV="development", JWT_SECRET_KEY=DEFAULT_JWT_SECRET_KEY))
    assert problems and "JWT_SECRET_KEY" in problems[0]
    assert any("Configuration non sûre" in record.getMessage() for record in caplog.records)


def test_empty_jwt_secret_falls_back_to_the_flagged_placeholder():
    """``JWT_SECRET_KEY=`` (vide, copié de ``.env.example``) ne signe jamais avec une clé vide."""
    from app.config import DEFAULT_JWT_SECRET_KEY, jwt_secret_from_env

    assert jwt_secret_from_env({"JWT_SECRET_KEY": ""}) == DEFAULT_JWT_SECRET_KEY
    assert jwt_secret_from_env({}) == DEFAULT_JWT_SECRET_KEY
    assert jwt_secret_from_env({"JWT_SECRET_KEY": "s" * 40}) == "s" * 40


def test_rotation_keeps_credentials_readable(monkeypatch):
    """Nouveau secret + ancien en ``JWT_SECRET_KEY_PREVIOUS`` : les identifiants restent lisibles."""
    from app.core import credentials

    settings = credentials.settings  # l'objet que lit le module de chiffrement

    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "old-secret-" + "o" * 40)
    monkeypatch.setattr(settings, "JWT_SECRET_KEY_PREVIOUS", None)
    token = credentials.encrypt_credentials({"openai_api_key": "fake-openai-rotation-0001"})

    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "new-secret-" + "n" * 40)
    assert credentials.decrypt_credentials(token) == {}  # sans l'ancien secret : illisible
    monkeypatch.setattr(settings, "JWT_SECRET_KEY_PREVIOUS", "old-secret-" + "o" * 40)
    assert credentials.decrypt_credentials(token) == {"openai_api_key": "fake-openai-rotation-0001"}


def test_rotation_script_reencrypts_under_the_new_secret(monkeypatch, tmp_path):
    """``rotate_credentials`` rechiffre chaque ligne : lisible ensuite avec le seul nouveau secret."""
    from app.core import credentials

    settings = credentials.settings  # l'objet que lit le module de chiffrement
    from app.database.base import Base
    from app.models import audit, background_task, pipeline_session, project  # noqa: F401
    from app.models.user import User
    from scripts.rotate_credentials_key import rotate_credentials

    engine = create_engine(f"sqlite:///{tmp_path / 'rotation.db'}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    try:
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", "old-secret-" + "o" * 40)
        monkeypatch.setattr(settings, "JWT_SECRET_KEY_PREVIOUS", None)
        db.add(User(email="a@example.org", hashed_password="x", roles=["USER"],
                    api_credentials=credentials.encrypt_credentials({"openai_api_key": "fake-openai-a-0001"})))
        db.add(User(email="b@example.org", hashed_password="x", roles=["USER"], api_credentials=None))
        db.commit()

        monkeypatch.setattr(settings, "JWT_SECRET_KEY", "new-secret-" + "n" * 40)
        monkeypatch.setattr(settings, "JWT_SECRET_KEY_PREVIOUS", "old-secret-" + "o" * 40)
        assert rotate_credentials(db, dry_run=True) == {"rotated": 1, "unreadable": 0, "empty": 1}
        assert rotate_credentials(db) == {"rotated": 1, "unreadable": 0, "empty": 1}

        monkeypatch.setattr(settings, "JWT_SECRET_KEY_PREVIOUS", None)
        stored = db.query(User).filter(User.email == "a@example.org").one().api_credentials
        assert credentials.decrypt_credentials(stored) == {"openai_api_key": "fake-openai-a-0001"}
    finally:
        db.close()
        engine.dispose()


def test_previous_secret_is_a_stripped_server_secret():
    """``JWT_SECRET_KEY_PREVIOUS`` est un secret serveur : jamais transmis aux sous-processus non-admin."""
    from app.core import credentials
    from scripts import rad_env

    assert "JWT_SECRET_KEY_PREVIOUS" in credentials.SERVER_SECRET_ENV_VARS
    assert tuple(rad_env.SERVER_SECRET_ENV_VARS) == tuple(credentials.SERVER_SECRET_ENV_VARS)


# ======================================================================
# A09 — CORS
# ======================================================================
def _cors_options():
    """Options du ``CORSMiddleware`` enregistré sur l'application."""
    from fastapi.middleware.cors import CORSMiddleware

    from app.main import app

    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            return middleware.kwargs if hasattr(middleware, "kwargs") else middleware.options
    raise AssertionError("CORSMiddleware absent")


def test_cors_uses_configured_origins_with_credentials():
    """Origines de ``CORS_ORIGINS`` (jamais ``*``) avec cookies autorisés."""
    from app.config import settings

    options = _cors_options()
    if "*" in settings.CORS_ORIGINS:
        assert options["allow_origins"] == ["*"] and options["allow_credentials"] is False
    else:
        assert options["allow_origins"] == list(settings.CORS_ORIGINS)
        assert "*" not in options["allow_origins"]
        assert options["allow_credentials"] is True


def test_cors_foreign_origin_gets_no_allow_header():
    """Une origine étrangère ne reçoit pas ``Access-Control-Allow-Origin`` (pré-vol refusé)."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.options("/health", headers={
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "GET",
    })
    assert resp.headers.get("access-control-allow-origin") is None


def test_health_detailed_does_not_sleep_on_the_event_loop():
    """``/health/detailed`` : échantillonnage CPU sans pause (``interval=None``) dans un thread."""
    import inspect

    from app import main

    source = inspect.getsource(main._collect_health_data)
    assert "cpu_percent(interval=None)" in source
    assert "asyncio.to_thread(_collect_health_data)" in inspect.getsource(main.health_detailed)
    assert re.search(r"interval=0\.\d", source) is None
