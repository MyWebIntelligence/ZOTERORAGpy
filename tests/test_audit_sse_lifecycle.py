"""Audit A08 (2026-09-27) : cycle de vie des scripts lancés par les routes web.

``run_subprocess_with_sse`` et ``run_tracked_subprocess`` lancent de vrais
processus Python locaux (``python -c``, aucun réseau) :

* silence : un script muet est arrêté à l'échéance, sans attendre sa fin ;
* flot continu : un script qui écrit sans fin est arrêté à l'échéance (avant,
  le délai n'était appliqué qu'après la fermeture des flux) ;
* débit élevé et ligne très longue : tout est lu, aucun blocage ;
* déconnexion (décision explicite) : le script continue sous son superviseur,
  sa sortie est drainée, son échéance s'applique toujours, il reste arrêtable
  (PID enregistré) et garde le verrou de session jusqu'à sa vraie fin ;
* client qui ne lit plus sans se déconnecter : l'échéance tue quand même le
  groupe (superviseur indépendant du flux) ;
* délai et annulation de ``run_tracked_subprocess`` : groupe de processus arrêté ;
* arrêt par le registre : ``stop_session`` atteint les enfants du script.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time

import pytest

from app.routes import processing
from app.services.process_manager import process_manager
from app.utils import sse_helpers

GRACE = 0.5


def _alive(pid: int) -> bool:
    """Vrai si ``pid`` existe et n'est pas un zombie (``os.kill(pid, 0)`` puis l'état lu par ``ps``)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_dead(pids, timeout=5.0) -> bool:
    """Attend que tous les ``pids`` soient morts (ou zombies) ; vrai si c'est le cas avant ``timeout``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in pids):
            return True
        time.sleep(0.05)
    return not any(_alive(pid) for pid in pids)


def _every_line(line):
    """Analyseur de test : chaque ligne devient un événement ``progress``."""
    return {"type": "progress", "message": line}


# Script qui lance un enfant dormant, affiche les deux PID, puis dort.
_WITH_CHILD = (
    "import os, subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "print(f'PIDS {os.getpid()} {child.pid}', flush=True)\n"
    "time.sleep(60)\n"
)


async def _collect(gen, stop_when=None, limit=100000):
    """Lit les événements du générateur ; s'arrête (sans le fermer) quand ``stop_when(event)`` est vrai."""
    events = []
    async for event in gen:
        events.append(event)
        if stop_when is not None and stop_when(event):
            break
        if len(events) >= limit:
            break
    return events


def _pids_from(events):
    """PID du script et de son enfant, lus dans la ligne ``PIDS a b``."""
    for event in events:
        if "PIDS " in event:
            parts = event.split("PIDS ", 1)[1].split('"')[0].split()
            return int(parts[0]), int(parts[1])
    raise AssertionError(f"ligne PIDS absente : {events[:3]}")


def test_silent_process_stopped_at_deadline():
    """Script muet de 30 s, délai 0,5 s : erreur « Process timed out » bien avant sa fin."""
    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        started = time.monotonic()
        events = await _collect(sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", "import time; time.sleep(30)"], _every_line,
            timeout=0.5, kill_grace=GRACE))
        return events, time.monotonic() - started

    events, elapsed = asyncio.run(scenario())
    assert events[-1] == 'data: {"type": "error", "message": "Process timed out"}\n\n'
    assert elapsed < 0.5 + 2 * GRACE + 2.0


def test_endless_output_stopped_at_deadline():
    """Script qui écrit sans fin : l'échéance encadre la lecture des flux, pas seulement l'attente de sortie."""
    code = "import time\nwhile True:\n    print('tick', flush=True)\n    time.sleep(0.05)\n"

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        started = time.monotonic()
        events = await _collect(sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", code], _every_line, timeout=1, kill_grace=GRACE))
        return events, time.monotonic() - started

    events, elapsed = asyncio.run(scenario())
    assert any('"tick"' in event for event in events)
    assert events[-1] == 'data: {"type": "error", "message": "Process timed out"}\n\n'
    assert elapsed < 1 + 2 * GRACE + 2.0


def test_high_throughput_and_long_line_complete():
    """20 000 lignes et une ligne de 200 Ko : lecture complète, événement ``complete``, aucun blocage."""
    code = (
        "import sys\n"
        "for i in range(20000):\n"
        "    print(f'line {i}')\n"
        "print('x' * 200000)\n"
        "print('last line', flush=True)\n"
    )

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        return await _collect(sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", code], _every_line, timeout=60, kill_grace=GRACE))

    events = asyncio.run(scenario())
    lines = [event for event in events if '"line ' in event]
    assert len(lines) == 20000
    assert any('"last line"' in event for event in events)
    assert events[-1] == 'data: {"type": "complete", "message": "Process completed successfully"}\n\n'


def test_disconnect_keeps_script_running_then_deadline_stops_it():
    """Déconnexion : le script et son enfant continuent (PID enregistré) ; l'échéance les arrête ensuite."""
    session = "audit-a08-disconnect"

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        gen = sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", _WITH_CHILD], _every_line,
            session_folder=session, timeout=3, kill_grace=GRACE)
        events = await _collect(gen, stop_when=lambda event: "PIDS " in event)
        pids = _pids_from(events)
        await gen.aclose()  # ce que fait Starlette quand le client se déconnecte
        await asyncio.sleep(0.5)
        still_running = all(_alive(pid) for pid in pids) and process_manager.get_pids(session) == [pids[0]]
        for _ in range(int((3 + 2 * GRACE + 3) / 0.05)):
            if not any(_alive(pid) for pid in pids) and not process_manager.get_pids(session):
                break
            await asyncio.sleep(0.05)
        return pids, still_running

    pids, still_running = asyncio.run(scenario())
    assert still_running
    assert _wait_dead(pids)
    assert process_manager.get_pids(session) == []


def test_disconnected_chatty_script_is_drained_to_completion():
    """Après déconnexion, un script bavard n'est jamais bloqué sur un tube plein : il se termine normalement."""
    code = ("import os, sys, time\n"
            "print(f'PIDS {os.getpid()} 0', flush=True)\n"
            "time.sleep(0.3)\n"
            "for i in range(50000):\n"
            "    print(f'ligne {i} ' + 'x' * 80)\n")

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        gen = sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", code], _every_line, timeout=60, kill_grace=GRACE)
        events = await _collect(gen, stop_when=lambda event: "PIDS " in event)
        leader = _pids_from(events)[0]
        await gen.aclose()
        for _ in range(400):
            if not _alive(leader):
                break
            await asyncio.sleep(0.05)
        return leader

    leader = asyncio.run(scenario())
    assert _wait_dead([leader], timeout=1)


def test_stalled_reader_does_not_suspend_the_deadline():
    """Client qui ne lit plus (générateur en attente) : le superviseur tue le groupe à l'échéance."""
    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        gen = sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", _WITH_CHILD], _every_line, timeout=1, kill_grace=GRACE)
        events = await _collect(gen, stop_when=lambda event: "PIDS " in event)
        pids = _pids_from(events)
        await asyncio.sleep(1 + 2 * GRACE + 1)  # le générateur reste suspendu, jamais relu
        dead = not any(_alive(pid) for pid in pids)
        await gen.aclose()
        return pids, dead

    pids, dead = asyncio.run(scenario())
    assert dead or _wait_dead(pids, timeout=1)


def test_job_ticket_held_until_the_script_really_ends():
    """Le verrou de session reste pris après la fin du flux, jusqu'à la sortie réelle du script."""
    from app.services import job_control

    key = "audit-a08-ticket"
    code = "import time; print('PIDS 1 0', flush=True); time.sleep(1.0)"

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        ticket, busy = job_control.acquire_job(key, job_control.GROUP_PIPELINE, None, admission=False)
        assert busy is None
        gen = sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", code], _every_line, timeout=30, kill_grace=GRACE, job_ticket=ticket)
        await _collect(gen, stop_when=lambda event: "PIDS " in event)
        await gen.aclose()
        ticket.release()  # fin du flux HTTP (``_release_after_stream``)
        busy_after_stream = job_control.session_busy(key, job_control.GROUP_PIPELINE)
        for _ in range(100):
            if not job_control.session_busy(key, job_control.GROUP_PIPELINE):
                break
            await asyncio.sleep(0.05)
        return busy_after_stream, job_control.session_busy(key, job_control.GROUP_PIPELINE)

    busy_after_stream, busy_at_end = asyncio.run(scenario())
    assert busy_after_stream is True
    assert busy_at_end is False


def test_timeout_stops_children_too():
    """Échéance dépassée : l'enfant du script est arrêté avec lui (groupe de processus)."""
    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        return await _collect(sse_helpers.run_subprocess_with_sse(
            [sys.executable, "-c", _WITH_CHILD], _every_line, timeout=1, kill_grace=GRACE))

    events = asyncio.run(scenario())
    assert events[-1] == 'data: {"type": "error", "message": "Process timed out"}\n\n'
    assert _wait_dead(_pids_from(events))


def test_tracked_subprocess_timeout_stops_the_group():
    """``run_tracked_subprocess`` : délai dépassé -> ``TimeoutExpired`` et groupe arrêté, PID désenregistré."""
    session = "audit-a08-tracked-timeout"
    code = _WITH_CHILD.replace("print(f'PIDS", "open(sys.argv[1], 'w').write(f'{os.getpid()} {child.pid}')\nprint(f'PIDS")
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"{session}.pids")
    pid_file = os.path.abspath(pid_file)

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        with pytest.raises(subprocess.TimeoutExpired):
            await processing.run_tracked_subprocess([sys.executable, "-c", code, pid_file], session, timeout=1)

    try:
        asyncio.run(scenario())
        with open(pid_file, encoding="utf-8") as handle:
            pids = [int(value) for value in handle.read().split()]
        assert _wait_dead(pids)
        assert process_manager.get_pids(session) == []
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)


def test_tracked_subprocess_cancellation_stops_the_group():
    """Annulation de la requête (arrêt du serveur) : le script lancé par ``run_tracked_subprocess`` est arrêté."""
    session = "audit-a08-tracked-cancel"

    async def scenario():
        """Scénario asynchrone du test, exécuté par ``asyncio.run``."""
        task = asyncio.create_task(processing.run_tracked_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)"], session, timeout=60))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if process_manager.get_pids(session):
                break
        pids = process_manager.get_pids(session)
        assert len(pids) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(int((2 * sse_helpers.SSE_KILL_GRACE + 3) / 0.05)):
            if not _alive(pids[0]) and not process_manager.get_pids(session):
                break
            await asyncio.sleep(0.05)
        return pids

    pids = asyncio.run(scenario())
    assert _wait_dead(pids)
    assert process_manager.get_pids(session) == []


def test_stop_session_reaches_children():
    """``process_manager.stop_session`` : le groupe entier du script enregistré est arrêté."""
    session = "audit-a08-stop"
    proc = subprocess.Popen(
        [sys.executable, "-c", _WITH_CHILD], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
        start_new_session=True, text=True,
    )
    try:
        line = proc.stdout.readline()
        leader, child = (int(value) for value in line.split()[1:3])
        # Le serveur récolte ses enfants (boucle asyncio) : même chose ici, sans zombie.
        threading.Thread(target=proc.wait, daemon=True).start()
        process_manager.register(session, leader)
        result = process_manager.stop_session(session, timeout=2)
        assert result["action_taken"] is True
        proc.wait(timeout=5)
        assert _wait_dead([leader, child])
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        for pid in process_manager.get_pids(session):
            process_manager.unregister(session, pid)
