"""
SSE Helpers
===========

This module provides utilities for implementing Server-Sent Events (SSE) to stream
real-time updates from background processes to the client. It is particularly useful
for long-running tasks like file processing and ingestion.

Key Features:
- Subprocess Streaming: Captures stdout/stderr from subprocesses and yields SSE events.
- Progress Parsing: Includes parsers for various output formats (tqdm, custom logs).
- Event Formatting: Formats data into standard SSE messages (data: ...).
"""

import asyncio
import json
import os
import re
import logging
import signal
from types import SimpleNamespace
from typing import AsyncGenerator, Callable, Optional, Dict, Any, Set

from app.services.process_manager import process_manager

logger = logging.getLogger(__name__)

# Bounded event queue between the stream readers and the SSE consumer: a slow
# client applies back-pressure to the script instead of growing the memory.
SSE_EVENT_QUEUE_MAXSIZE = 1000
# Longest line read at once from the script output (asyncio's default is 64 KiB);
# a longer line is skipped, never stops the reader.
SSE_STREAM_LIMIT = 1024 * 1024
# Seconds granted after SIGTERM before SIGKILL, for the whole process group.
SSE_KILL_GRACE = 5.0

# Reaper tasks of processes stopped by a cancelled request (kept referenced).
_REAPERS: Set["asyncio.Task"] = set()
# Supervisor and drain tasks of streamed scripts (kept referenced until they end).
_SUPERVISORS: Set["asyncio.Task"] = set()

_ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def signal_process_group(pid: Optional[int], sig: int) -> None:
    """
    Send ``sig`` to the process group led by ``pid`` (errors ignored).

    The pipeline scripts are started in their own session
    (``start_new_session=True``), so the group holds the script and every
    child it spawned (local OCR subprocess...). ``pid`` values below 2 are
    refused: ``killpg(0)`` would signal the server's own group.

    Args:
        pid: PID of the group leader (the launched script).
        sig: Signal to send.
    """
    if not isinstance(pid, int) or pid < 2:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, sig)
        else:  # pragma: no cover - non-POSIX platforms
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def terminate_process_group(process: Any, grace: float = SSE_KILL_GRACE) -> None:
    """
    Stop a script and its children: SIGTERM to the group, then SIGKILL after ``grace``.

    Args:
        process: ``asyncio.subprocess.Process`` started with ``start_new_session=True``.
        grace: Seconds to wait for the leader after SIGTERM.
    """
    signal_process_group(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except asyncio.TimeoutError:
        pass
    # Children that ignored SIGTERM (or outlived the leader) are killed too.
    signal_process_group(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except asyncio.TimeoutError:  # pragma: no cover - SIGKILL cannot be ignored
        logger.warning(f"Process {process.pid} did not exit after SIGKILL")


def stop_process_group_later(process: Any, session_folder: Optional[str], grace: float = SSE_KILL_GRACE) -> None:
    """
    Stop a still running script from a cancelled context, without awaiting.

    Used in ``finally`` blocks reached by a cancellation (client disconnect,
    server shutdown), where awaiting is not possible: SIGTERM is sent at
    once, then a separate reaper task waits ``grace``, kills the group with
    SIGKILL, reaps the process and unregisters its PID.

    Args:
        process: The running ``asyncio.subprocess.Process``.
        session_folder: Registry key of the PID, or None.
        grace: Seconds granted after SIGTERM.
    """
    signal_process_group(process.pid, signal.SIGTERM)

    async def _reap() -> None:
        """Wait for the leader, kill the group, then unregister the PID."""
        try:
            await terminate_process_group(process, grace)
        finally:
            if session_folder:
                process_manager.unregister(session_folder, process.pid)

    try:
        task = asyncio.get_running_loop().create_task(_reap())
    except RuntimeError:  # no running loop (garbage collection): kill now
        signal_process_group(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        if session_folder:
            process_manager.unregister(session_folder, process.pid)
        return
    _REAPERS.add(task)
    task.add_done_callback(_REAPERS.discard)


async def _supervise_process(
    process: Any,
    deadline: Optional[float],
    kill_grace: float,
    session_folder: Optional[str],
    hold: Any,
    state: Any,
) -> None:
    """
    Own the lifecycle of a streamed script, independently of its HTTP stream.

    Waits for the script until the deadline, stops its process group when
    the deadline expires (``state.timed_out``), then unregisters its PID and
    releases the job hold. Runs as its own task: a slow, stalled or
    disconnected client never delays the deadline nor leaks the PID or the
    session lock.

    Args:
        process: The running ``asyncio.subprocess.Process``.
        deadline: Loop time of the deadline, or None.
        kill_grace: Seconds granted after SIGTERM before SIGKILL.
        session_folder: Registry key of the PID, or None when not registered.
        hold: ``job_control.JobHold`` of the script, or None.
        state: Shared state object (``timed_out`` attribute set here).
    """
    loop = asyncio.get_running_loop()
    try:
        remaining = None if deadline is None else max(0.0, deadline - loop.time())
        try:
            await asyncio.wait_for(process.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            state.timed_out = True
            logger.warning(f"Subprocess {process.pid} timed out; stopping its process group")
            await terminate_process_group(process, kill_grace)
    finally:
        if session_folder:
            process_manager.unregister(session_folder, process.pid)
            logger.info(f"Unregistered async PID {process.pid} for session '{session_folder}'")
        if hold is not None:
            hold.release()


async def _drain_detached(readers: list, event_queue: "asyncio.Queue") -> None:
    """
    Keep reading a script's output after its stream closed, discarding the events.

    Without it, the bounded queue would fill up and the script would block on
    a full pipe. Ends when both readers reach the end of the output (the
    script exited or was stopped by its supervisor).

    Args:
        readers: The two reader tasks (stdout, stderr).
        event_queue: Their event queue.
    """
    while not all(reader.done() for reader in readers):
        try:
            await asyncio.wait_for(event_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
    while not event_queue.empty():
        event_queue.get_nowait()


async def run_subprocess_with_sse(
    cmd: list[str],
    progress_parser: Callable[[str], Optional[Dict[str, Any]]],
    *,
    session_folder: Optional[str] = None,
    error_keywords: Optional[list[str]] = None,
    timeout: Optional[int] = None,
    heartbeat_interval: int = 15,
    env: Optional[Dict[str, str]] = None,
    kill_grace: float = SSE_KILL_GRACE,
    job_ticket: Any = None,
) -> AsyncGenerator[str, None]:
    """
    Execute a subprocess and stream SSE events by parsing stdout/stderr.

    Lifecycle (audit A08, 2026-09-27):

    * The script runs in its own process group, owned by a supervisor task
      (``_supervise_process``), not by the HTTP stream: ``timeout`` is a
      single deadline (monotonic clock) around the whole run, enforced even
      when the client reads slowly, stalls or has gone; on expiry the whole
      group is stopped (SIGTERM, then SIGKILL after ``kill_grace``) and the
      stream reports ``Process timed out``.
    * A client disconnect does NOT stop the script (explicit decision: a
      page reload must not throw away hours of paid recoding or embedding
      calls). Its output keeps being drained, its deadline still applies,
      it stays stoppable through ``/stop_all_scripts`` (PID registered until
      it ends) and it keeps the session locked (``job_ticket`` hold, audit
      A12) until it really exits.
    * The event queue is bounded (``SSE_EVENT_QUEUE_MAXSIZE``); lines longer
      than ``SSE_STREAM_LIMIT`` are skipped without stopping the reader.

    Args:
        cmd: Command and arguments to execute
        progress_parser: Function that parses log lines and returns event dict or None
        session_folder: Session identifier for PID tracking (enables session-aware stop)
        error_keywords: List of keywords that indicate errors in output
        timeout: Optional timeout in seconds, for the whole run
        heartbeat_interval: Seconds between heartbeat messages to keep connection alive (default: 15)
        env: Optional environment dictionary for subprocess. If None, uses current env.
             Use build_subprocess_env() to create a secure environment with user credentials.
        kill_grace: Seconds granted after SIGTERM before SIGKILL.
        job_ticket: ``job_control.JobTicket`` of the request, or None: the
            script holds it until it exits.

    Yields:
        SSE-formatted strings: "data: {JSON}\\n\\n"

    Event types emitted:
        - init: Initial setup with total count if known
        - progress: Progress updates with current/total
        - heartbeat: Keep-alive message during long operations
        - complete: Successful completion
        - error: Error occurred (including "Process timed out")
    """
    error_keywords = error_keywords or ["error", "failed", "exception", "traceback"]
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + float(timeout)
    process = None
    readers: list = []
    supervisor = None
    event_queue = None
    state = SimpleNamespace(timed_out=False)

    async def read_stream(stream, stream_name):
        """Read from stdout or stderr and parse progress."""
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # Line above SSE_STREAM_LIMIT: asyncio dropped it, keep reading.
                logger.debug(f"[{stream_name}] line above {SSE_STREAM_LIMIT} bytes skipped")
                continue
            except Exception as e:
                logger.error(f"Error reading {stream_name}: {e}")
                break
            if not line:
                break

            decoded = line.decode('utf-8', errors='replace').strip()
            if not decoded:
                continue

            # Strip ANSI codes for clean parsing
            decoded = _ANSI_ESCAPE_RE.sub('', decoded)

            logger.debug(f"[{stream_name}] {decoded}")

            # Log PROGRESS lines at INFO level for debugging
            if decoded.startswith("PROGRESS|"):
                logger.info(f"[{stream_name}] PROGRESS line: {decoded}")

            # Check for error indicators
            if any(keyword in decoded.lower() for keyword in error_keywords):
                # Only emit error if it looks serious (not just a warning)
                if "error" in decoded.lower() and "warning" not in decoded.lower():
                    yield {"type": "error", "message": decoded}
                    continue

            # Try to parse progress
            event = progress_parser(decoded)
            if event:
                logger.info(f"SSE Parsed event: {event.get('type', 'unknown')} - {event}")
                yield event

    try:
        # Create subprocess with both stdout and stderr captured, in its own
        # process group so that a stop reaches its children.
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,  # never inherit the server's stdin
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,  # Pass custom environment if provided
            start_new_session=True,
            limit=SSE_STREAM_LIMIT,
        )

        # Register PID for session-aware process management
        registered = False
        if session_folder:
            process_manager.register(session_folder, process.pid)
            registered = True
            logger.info(f"Registered async PID {process.pid} for session '{session_folder}'")
        hold = job_ticket.hold() if job_ticket is not None else None
        supervisor = loop.create_task(_supervise_process(
            process, deadline, kill_grace, session_folder if registered else None, hold, state,
        ))
        _SUPERVISORS.add(supervisor)
        supervisor.add_done_callback(_SUPERVISORS.discard)

        logger.info(f"Started subprocess: {' '.join(cmd)}")

        # Read both streams concurrently and merge events using a bounded queue
        event_queue = asyncio.Queue(maxsize=SSE_EVENT_QUEUE_MAXSIZE)

        async def read_and_queue(stream, stream_name):
            """Read stream and put events into queue."""
            async for event in read_stream(stream, stream_name):
                await event_queue.put(event)

        readers = [
            asyncio.create_task(read_and_queue(process.stdout, "stdout")),
            asyncio.create_task(read_and_queue(process.stderr, "stderr"))
        ]

        # Stream events as they come from either stream
        last_heartbeat = loop.time()
        while True:
            if all(r.done() for r in readers) and event_queue.empty():
                break
            if state.timed_out:
                break
            try:
                # Wait for event with timeout to check if readers finished
                event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                current_time = loop.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield "data: {\"type\": \"heartbeat\", \"message\": \"Processing...\"}\n\n"
                    last_heartbeat = current_time
                    logger.debug(f"Sent heartbeat after {heartbeat_interval}s of inactivity")
                continue
            yield f"data: {json.dumps(event)}\n\n"
            last_heartbeat = loop.time()  # Reset heartbeat timer on activity

        # Output closed (or deadline hit): the supervisor ends the run. Shielded:
        # a cancelled stream must never cancel the supervisor.
        await asyncio.shield(supervisor)

        if state.timed_out:
            yield "data: {\"type\": \"error\", \"message\": \"Process timed out\"}\n\n"
        elif process.returncode != 0:
            yield f"data: {{\"type\": \"error\", \"message\": \"Process failed with code {process.returncode}\"}}\n\n"
        else:
            yield "data: {\"type\": \"complete\", \"message\": \"Process completed successfully\"}\n\n"

    except Exception as e:
        logger.error(f"Subprocess error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        # Every exit path, cancellation and client disconnect included. Nothing
        # is awaited here: the supervisor owns the process, the PID and the hold.
        if supervisor is not None and not supervisor.done() and readers and event_queue is not None:
            # Stream gone while the script runs: keep draining its output.
            drain = loop.create_task(_drain_detached(readers, event_queue))
            _SUPERVISORS.add(drain)
            drain.add_done_callback(_SUPERVISORS.discard)
            logger.info(f"SSE stream closed; subprocess {process.pid} keeps running under its supervisor")
        else:
            for reader in readers:
                reader.cancel()


# ============================================================================
# Progress Parsers for specific scripts
# ============================================================================

def parse_tqdm_progress(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse tqdm progress bar output from stderr.
    
    Example formats:
        "Processing Zotero items: 45%|████▌     | 45/100 [00:23<00:28,  1.98it/s]"
        "100%|██████████| 100/100 [01:23<00:00,  1.20it/s]"
    """
    # Match tqdm progress bar format
    # Pattern: <desc>: <percentage>%|<bar>| <current>/<total> [...]
    match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)', line)
    if match:
        percent = int(match.group(1))
        current = int(match.group(2))
        total = int(match.group(3))
        
        # Extract description if present
        desc_match = re.match(r'([^:]+):', line)
        message = desc_match.group(1).strip() if desc_match else "Processing"
        
        return {
            "type": "progress",
            "current": current,
            "total": total,
            "percent": percent,
            "message": f"{message}: {current}/{total}"
        }
    
    return None


def parse_dataframe_logs(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse rad_dataframe.py log output for progress events.
    
    Example formats:
        "INFO - Chargement du fichier JSON Zotero depuis : /path/to/file.json"
        "INFO - Detected Zotero JSON format: direct array with 150 items"
        "INFO - Processing document 45/150: document.pdf"
        "INFO - ✓ Item ABC123 saved (45 total)"
    """
    # Check for total items detected
    match = re.search(r'(\d+)\s+items', line, re.IGNORECASE)
    if match and 'detected' in line.lower():
        total = int(match.group(1))
        return {
            "type": "init",
            "total": total,
            "message": f"Found {total} items to process"
        }
    
    # Check for item saved (progress indicator)
    match = re.search(r'✓\s+Item\s+\w+\s+saved\s+\((\d+)\s+total\)', line)
    if match:
        current = int(match.group(1))
        return {
            "type": "progress",
            "current": current,
            "message": f"Processed {current} items"
        }
    
    # Check for resuming message
    match = re.search(r'Resuming:\s+(\d+)/(\d+)\s+items\salready\sprocessed', line)
    if match:
        done = int(match.group(1))
        total = int(match.group(2))
        return {
            "type": "init",
            "total": total,
            "message": f"Resuming: {done}/{total} already done"
        }
    
    return None


def parse_chunking_logs(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse rad_chunk.py log output for progress events.
    
    Example formats:
        "Traitement de 'document.pdf': 15 chunks bruts générés."
        "→ 15 chunks traités et sauvegardés pour le document 'doc' (doc_id=123)"
        "Document #5 traité, 15 chunks produits."
        "Chargement de 450 chunks depuis 'file.json' pour génération d'embeddings."
    """
    # Check for document processing
    match = re.search(r'Document\s+#(\d+)\s+traité.*?(\d+)\s+chunks', line)
    if match:
        doc_num = int(match.group(1))
        chunk_count = int(match.group(2))
        return {
            "type": "progress",
            "current": doc_num,
            "message": f"Document #{doc_num}: {chunk_count} chunks generated"
        }
    
    # Check for embedding generation init
    match = re.search(r'Chargement\s+de\s+(\d+)\s+chunks', line)
    if match:
        total = int(match.group(1))
        return {
            "type": "init",
            "total": total,
            "message": f"Loading {total} chunks for processing"
        }
    
    # Check for phase start
    if "Phase" in line and ("initial" in line.lower() or "dense" in line.lower() or "sparse" in line.lower()):
        return {
            "type": "init",
            "message": line.strip()
        }
    
    return None


def parse_multilevel_progress(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse structured multilevel progress logs.

    Format: PROGRESS|level|current/total|message

    Levels:
        - row: CSV row being processed (primary progress)
        - chunk: Chunk being processed (secondary progress)
        - page: PDF page being processed (secondary progress)
        - embed: Embedding being generated (secondary progress)

    Examples:
        "PROGRESS|row|5/20|Processing: document.pdf"
        "PROGRESS|chunk|150/500|Generating embedding"
        "PROGRESS|page|3/15|OCR page 3"
        "PROGRESS|init|20|Found 20 documents to process"
    """
    if not line.startswith("PROGRESS|"):
        return None

    try:
        # Use simple split
        parts = line.split("|")
        # Rejoin message part if it contained pipes
        if len(parts) > 4:
            parts = parts[:3] + ["|".join(parts[3:])]
        
        if len(parts) < 3:
            return None

        level = parts[1].strip().lower()

        # Handle init event (special case: total only)
        if level == "init":
            try:
                total = int(parts[2].strip())
                message = parts[3].strip() if len(parts) > 3 else f"Found {total} items"
                return {
                    "type": "init",
                    "total": total,
                    "message": message
                }
            except ValueError:
                return None

        # Parse current/total
        counts = parts[2].strip()
        if "/" not in counts:
            return None

        current_str, total_str = counts.split("/", 1)
        current = int(current_str.strip())
        total = int(total_str.strip())

        # Get message (optional)
        message = parts[3].strip() if len(parts) > 3 else f"{level.capitalize()} {current}/{total}"

        # Calculate percentage
        percent = round((current / total) * 100) if total > 0 else 0

        return {
            "type": "progress",
            "level": level,
            "current": current,
            "total": total,
            "percent": percent,
            "message": message
        }
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse multilevel progress: {line} - {e}")
        return None


def create_combined_parser(*parsers: Callable[[str], Optional[Dict[str, Any]]]) -> Callable[[str], Optional[Dict[str, Any]]]:
    """
    Create a combined parser that tries multiple parsers in order.

    Args:
        *parsers: Variable number of parser functions

    Returns:
        A parser function that tries each parser until one returns a result
    """
    def combined(line: str) -> Optional[Dict[str, Any]]:
        for parser in parsers:
            result = parser(line)
            if result:
                return result
        return None

    return combined
