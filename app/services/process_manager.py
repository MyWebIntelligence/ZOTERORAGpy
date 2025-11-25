"""
Process Manager
===============

This module provides a thread-safe singleton `ProcessManager` to track and control
background subprocesses spawned by the application. It ensures that processes are
isolated by user session, allowing for targeted termination without affecting other users.

Key Features:
- Session Isolation: Tracks PIDs associated with specific session IDs.
- Thread Safety: Uses locks to ensure safe concurrent access.
- Graceful Shutdown: Attempts SIGTERM before resorting to SIGKILL.
- Cleanup: Automatically removes dead processes from the registry.
"""

import os
import signal
import threading
import logging
import time
from typing import Dict, Set, List, Optional

logger = logging.getLogger(__name__)


class ProcessManager:
    """
    Singleton thread-safe pour tracker les PIDs par session.

    Usage:
        from app.services.process_manager import process_manager

        # Enregistrer un processus
        process_manager.register("session_folder", pid)

        # Arrêter tous les processus d'une session
        result = process_manager.stop_session("session_folder")

        # Désenregistrer (appelé automatiquement quand le process termine)
        process_manager.unregister("session_folder", pid)
    """

    _instance: Optional['ProcessManager'] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> 'ProcessManager':
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._processes: Dict[str, Set[int]] = {}  # session_folder → {pid1, pid2, ...}
        self._initialized = True
        logger.info("ProcessManager initialized")

    def register(self, session_folder: str, pid: int) -> None:
        """
        Enregistre un PID pour une session donnée.

        Args:
            session_folder: Identifiant unique de la session (ex: "uuid_filename")
            pid: Process ID à tracker
        """
        with self._lock:
            if session_folder not in self._processes:
                self._processes[session_folder] = set()
            self._processes[session_folder].add(pid)
            logger.info(f"Registered PID {pid} for session '{session_folder}'")

    def unregister(self, session_folder: str, pid: int) -> None:
        """
        Désenregistre un PID d'une session (appelé quand le process termine).

        Args:
            session_folder: Identifiant de la session
            pid: Process ID à retirer
        """
        with self._lock:
            if session_folder in self._processes:
                self._processes[session_folder].discard(pid)
                if not self._processes[session_folder]:
                    del self._processes[session_folder]
                logger.info(f"Unregistered PID {pid} from session '{session_folder}'")

    def get_pids(self, session_folder: str) -> List[int]:
        """
        Retourne la liste des PIDs actifs pour une session.

        Args:
            session_folder: Identifiant de la session

        Returns:
            Liste des PIDs (peut être vide)
        """
        with self._lock:
            return list(self._processes.get(session_folder, set()))

    def _is_process_alive(self, pid: int) -> bool:
        """Vérifie si un processus est toujours en vie."""
        try:
            os.kill(pid, 0)  # Signal 0 = juste vérifier si le process existe
            return True
        except OSError:
            return False

    def stop_session(self, session_folder: str, timeout: float = 5.0) -> Dict:
        """
        Arrête tous les processus d'une session de manière sécurisée.

        1. Envoie SIGTERM à tous les PIDs
        2. Attend `timeout` secondes
        3. Si certains survivent, envoie SIGKILL

        Args:
            session_folder: Identifiant de la session
            timeout: Délai avant SIGKILL (défaut: 5s)

        Returns:
            Dict avec le résultat de l'opération
        """
        pids = self.get_pids(session_folder)

        if not pids:
            logger.info(f"No processes found for session '{session_folder}'")
            return {
                "status": "No running processes found",
                "action_taken": False,
                "session": session_folder,
                "details": "No matching processes were found running for this session."
            }

        logger.info(f"Stopping {len(pids)} process(es) for session '{session_folder}': {pids}")

        terminated = []
        killed = []
        failed = []
        already_dead = []

        # Phase 1: SIGTERM (arrêt propre)
        for pid in pids:
            if not self._is_process_alive(pid):
                already_dead.append(pid)
                self.unregister(session_folder, pid)
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to PID {pid}")
            except OSError as e:
                logger.warning(f"Failed to send SIGTERM to PID {pid}: {e}")
                failed.append(pid)

        # Attendre que les processus se terminent proprement
        deadline = time.time() + timeout
        remaining_pids = [p for p in pids if p not in already_dead and p not in failed]

        while remaining_pids and time.time() < deadline:
            time.sleep(0.2)
            still_alive = []
            for pid in remaining_pids:
                if self._is_process_alive(pid):
                    still_alive.append(pid)
                else:
                    terminated.append(pid)
                    self.unregister(session_folder, pid)
            remaining_pids = still_alive

        # Phase 2: SIGKILL pour les récalcitrants
        for pid in remaining_pids:
            if self._is_process_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                    self.unregister(session_folder, pid)
                    logger.warning(f"Sent SIGKILL to PID {pid} (did not respond to SIGTERM)")
                except OSError as e:
                    logger.error(f"Failed to kill PID {pid}: {e}")
                    failed.append(pid)
            else:
                terminated.append(pid)
                self.unregister(session_folder, pid)

        # Construire le résultat
        total_stopped = len(terminated) + len(killed) + len(already_dead)

        result = {
            "status": "Stop signal sent to running scripts" if total_stopped > 0 else "No processes stopped",
            "action_taken": total_stopped > 0,
            "session": session_folder,
            "details": {
                "terminated_gracefully": terminated,
                "force_killed": killed,
                "already_dead": already_dead,
                "failed": failed,
                "total_stopped": total_stopped
            }
        }

        logger.info(f"Stop session result: {result}")
        return result

    def cleanup_dead_processes(self) -> int:
        """
        Nettoie les PIDs de processus morts de toutes les sessions.

        Returns:
            Nombre de PIDs nettoyés
        """
        cleaned = 0
        with self._lock:
            for session_folder in list(self._processes.keys()):
                dead_pids = [
                    pid for pid in self._processes[session_folder]
                    if not self._is_process_alive(pid)
                ]
                for pid in dead_pids:
                    self._processes[session_folder].discard(pid)
                    cleaned += 1
                if not self._processes[session_folder]:
                    del self._processes[session_folder]

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} dead process(es)")
        return cleaned

    def get_all_sessions(self) -> Dict[str, List[int]]:
        """
        Retourne toutes les sessions avec leurs PIDs (pour debug/monitoring).

        Returns:
            Dict session_folder → [pids]
        """
        with self._lock:
            return {k: list(v) for k, v in self._processes.items()}


# Instance singleton exportée
process_manager = ProcessManager()
