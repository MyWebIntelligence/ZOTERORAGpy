"""Audit A12 (2026-09-27) : exclusion des traitements par session et admission bornée.

* deux traitements du même groupe sur la même session : le second reçoit 409
  (JSON ou unique événement SSE), sans lancement ; après la fin du premier, la
  session est de nouveau disponible (y compris après un flux SSE) ;
* les groupes sont indépendants (notes pendant le clustering) ;
* admission : au-delà de ``MAX_ACTIVE_JOBS_PER_USER`` (ou ``MAX_ACTIVE_JOBS``),
  429 ; un autre utilisateur n'est pas concerné par la limite par utilisateur ;
* verrou inter-processus : un autre processus qui tient le verrou bloque la
  session ; il est libéré par le noyau si ce processus meurt ;
* upload d'un fichier d'étape pendant une étape en cours : 409, fichier intact ;
* Celery : soumission refusée (409) pendant une étape web, et le worker tient
  le même verrou (``runner.run_script(session_dir=...)``).

Harnais : ``golden_app`` des goldens (lanceurs remplacés par des enregistreurs).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import app.routes.ingestion as ingestion_routes
from app.services import job_control
from app.tasks import runner
from tests import test_albert_off_golden_routes as golden
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401

OWNER = "member_nokeys"      # propriétaire du projet de gsess-full
REGISTERED = "gsess-full"
BUSY_PIPELINE = job_control.SESSION_BUSY_MESSAGE.format(label="étapes du pipeline")


def _post(env, persona, url, form):
    """POST de formulaire : (réponse, appels enregistrés des lanceurs)."""
    resp = env.client.post(url, data=form, headers=env.headers[persona])
    golden._take_server_errors(env)
    return resp, env.recorder.take()


def _hold(session_key, group=job_control.GROUP_PIPELINE, user_id=None, admission=False):
    """Ticket tenu comme par un traitement en cours."""
    ticket, busy = job_control.acquire_job(session_key, group, user_id, admission=admission)
    assert busy is None
    return ticket


def test_json_route_refused_while_session_busy(golden_app):
    """Étape JSON pendant une étape en cours sur la session : 409, aucun lancement ; libre ensuite."""
    env = golden_app
    ticket = _hold(REGISTERED)
    try:
        resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": REGISTERED})
        assert resp.status_code == 409 and resp.json() == {"error": BUSY_PIPELINE}
        assert calls == []
    finally:
        ticket.release()
    resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": REGISTERED})
    assert resp.status_code == 200 and len(calls) == 1


def test_sse_route_refused_while_busy_and_released_after_stream(golden_app):
    """Route SSE : 409 en un unique événement ; le ticket d'un flux terminé est rendu."""
    env = golden_app
    ticket = _hold(REGISTERED)
    try:
        resp, calls = _post(env, "admin", "/sparse_embedding_generation_sse", {"path": REGISTERED})
        assert resp.status_code == 409 and calls == []
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert BUSY_PIPELINE in resp.text
    finally:
        ticket.release()
    for _ in range(2):  # deux flux complets successifs : aucun ticket résiduel
        resp, calls = _post(env, "admin", "/sparse_embedding_generation_sse", {"path": REGISTERED})
        assert resp.status_code == 200 and len(calls) == 1
    assert not job_control.session_busy(REGISTERED, job_control.GROUP_PIPELINE)


def test_other_spelling_hits_the_same_lock(golden_app):
    """``./gsess-full`` pendant une étape sur ``gsess-full`` : même verrou (clé canonique)."""
    env = golden_app
    ticket = _hold(REGISTERED)
    try:
        resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": "./gsess-full"})
        assert resp.status_code == 409 and calls == []
    finally:
        ticket.release()


def test_groups_are_independent(golden_app):
    """Un clustering en cours ne bloque pas une étape du pipeline de la même session."""
    env = golden_app
    ticket = _hold(REGISTERED, group=job_control.GROUP_CLUSTERING)
    try:
        resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": REGISTERED})
        assert resp.status_code == 200 and len(calls) == 1
    finally:
        ticket.release()


def test_per_user_admission_limit(golden_app, monkeypatch):
    """``MAX_ACTIVE_JOBS_PER_USER=1`` : 429 pour le même utilisateur sur une autre session, pas pour un autre."""
    env = golden_app
    monkeypatch.setenv("MAX_ACTIVE_JOBS_PER_USER", "1")
    admin_id = env.users["admin"].id
    ticket = _hold("another-session", user_id=admin_id, admission=True)
    try:
        resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": REGISTERED})
        assert resp.status_code == 429 and calls == []
        assert "pour votre compte" in resp.json()["error"]
        resp, calls = _post(env, OWNER, "/sparse_embedding_generation", {"path": REGISTERED})
        assert resp.status_code != 429
    finally:
        ticket.release()
    resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": REGISTERED})
    assert resp.status_code == 200


def test_global_admission_limit(golden_app, monkeypatch):
    """``MAX_ACTIVE_JOBS=1`` : un traitement en cours ailleurs suffit à refuser (429)."""
    env = golden_app
    monkeypatch.setenv("MAX_ACTIVE_JOBS", "1")
    ticket = _hold("another-session", user_id=999, admission=True)
    try:
        resp, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": REGISTERED})
        assert resp.status_code == 429 and calls == [] and "sur le serveur" in resp.json()["error"]
    finally:
        ticket.release()


def test_lock_held_by_another_process_blocks_until_it_dies(tmp_path):
    """Verrou tenu par un autre processus : session occupée ; libérée quand ce processus meurt."""
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {os.getcwd()!r})
        from app.services import job_control
        ticket, busy = job_control.acquire_job("cross-process", job_control.GROUP_PIPELINE, None, admission=False)
        print("held" if busy is None else "busy", flush=True)
        time.sleep(60)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                            env=dict(os.environ), stdin=subprocess.DEVNULL)
    try:
        assert proc.stdout.readline().strip() == "held"
        assert job_control.session_busy("cross-process", job_control.GROUP_PIPELINE)
        ticket, busy = job_control.acquire_job("cross-process", job_control.GROUP_PIPELINE, None)
        assert ticket is None and busy[0] == 409
    finally:
        proc.kill()
        proc.wait(timeout=10)
        proc.stdout.close()
    assert not job_control.session_busy("cross-process", job_control.GROUP_PIPELINE)


def test_stage_upload_refused_while_pipeline_runs(golden_app, monkeypatch):
    """Upload d'un fichier d'étape pendant une étape en cours : 409, artefact intact."""
    env = golden_app
    monkeypatch.setattr(ingestion_routes, "UPLOAD_DIR", str(env.uploads))
    target = env.uploads / REGISTERED / "output.csv"
    before = target.read_bytes()
    ticket = _hold(REGISTERED)
    try:
        resp = env.client.post("/upload_stage_file/initial", data={"path": REGISTERED},
                               files={"file": ("mine.csv", b"title,texteocr\nX,y\n")},
                               headers=env.headers[OWNER])
        assert resp.status_code == 409 and resp.json() == {"error": BUSY_PIPELINE}
    finally:
        ticket.release()
    assert target.read_bytes() == before


def test_worker_run_script_holds_the_session_lock(tmp_path, monkeypatch):
    """Celery : ``run_script(session_dir=...)`` échoue tout de suite si la session est occupée, sinon tient le verrou."""
    session_dir = tmp_path / "sess-celery"
    session_dir.mkdir()
    key = runner.session_lock_key(str(session_dir))
    ticket = _hold(key)
    try:
        with pytest.raises(runner.ScriptFailedError, match="déjà en cours"):
            runner.run_script([sys.executable, "-c", "print('never')"], dict(os.environ),
                              session_dir=str(session_dir), timeout=30)
    finally:
        ticket.release()
    code = (
        "import sys; sys.path.insert(0, %r); from app.services import job_control; "
        "print(job_control.session_busy(%r, job_control.GROUP_PIPELINE))" % (os.getcwd(), key)
    )
    result = runner.run_script([sys.executable, "-c", code], dict(os.environ),
                               session_dir=str(session_dir), timeout=60)
    assert result.returncode == 0 and result.stdout.strip() == "True"  # tenu pendant le script
    assert not job_control.session_busy(key, job_control.GROUP_PIPELINE)  # rendu après


def test_celery_submission_refused_while_session_busy(golden_app, monkeypatch):
    """Soumission Celery pendant une étape web sur la session : 409 avant toute mise en file."""
    from fastapi import HTTPException

    import app.routes.celery_tasks as celery_routes
    from app.models.pipeline_session import PipelineSession

    env = golden_app
    monkeypatch.setattr(celery_routes, "UPLOAD_DIR", str(env.uploads))
    session_id = env.db.query(PipelineSession).filter(PipelineSession.session_folder == REGISTERED).one().id
    owner = env.users[OWNER]
    directory = celery_routes._session_directory(env.db, REGISTERED, session_id, owner)
    assert os.path.basename(directory) == REGISTERED
    ticket = _hold(REGISTERED)
    try:
        with pytest.raises(HTTPException) as exc:
            celery_routes._session_directory(env.db, REGISTERED, session_id, owner)
        assert exc.value.status_code == 409
    finally:
        ticket.release()
