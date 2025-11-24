"""
SSE (Server-Sent Events) helper utilities for streaming subprocess progress.

This module provides async generators for streaming subprocess output as SSE events,
with specialized parsers for different script output formats.
"""

import asyncio
import json
import re
import logging
from typing import AsyncGenerator, Callable, Optional, Dict, Any

from app.services.process_manager import process_manager

logger = logging.getLogger(__name__)


async def run_subprocess_with_sse(
    cmd: list[str],
    progress_parser: Callable[[str], Optional[Dict[str, Any]]],
    *,
    session_folder: Optional[str] = None,
    error_keywords: Optional[list[str]] = None,
    timeout: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    Execute a subprocess and stream SSE events by parsing stdout/stderr.

    Args:
        cmd: Command and arguments to execute
        progress_parser: Function that parses log lines and returns event dict or None
        session_folder: Session identifier for PID tracking (enables session-aware stop)
        error_keywords: List of keywords that indicate errors in output
        timeout: Optional timeout in seconds

    Yields:
        SSE-formatted strings: "data: {JSON}\\n\\n"

    Event types emitted:
        - init: Initial setup with total count if known
        - progress: Progress updates with current/total
        - complete: Successful completion
        - error: Error occurred
    """
    error_keywords = error_keywords or ["error", "failed", "exception", "traceback"]

    try:
        # Create subprocess with both stdout and stderr captured
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Register PID for session-aware process management
        if session_folder:
            process_manager.register(session_folder, process.pid)
            logger.info(f"Registered async PID {process.pid} for session '{session_folder}'")

        logger.info(f"Started subprocess: {' '.join(cmd)}")
        
        async def read_stream(stream, stream_name):
            """Read from stdout or stderr and parse progress."""
            while True:
                try:
                    line = await stream.readline()
                    if not line:
                        break
                        
                    decoded = line.decode('utf-8', errors='replace').strip()
                    if not decoded:
                        continue
                        
                    logger.debug(f"[{stream_name}] {decoded}")
                    
                    # Check for error indicators
                    if any(keyword in decoded.lower() for keyword in error_keywords):
                        # Only emit error if it looks serious (not just a warning)
                        if "error" in decoded.lower() and "warning" not in decoded.lower():
                            yield {"type": "error", "message": decoded}
                            continue
                    
                    # Try to parse progress
                    event = progress_parser(decoded)
                    if event:
                        yield event
                        
                except Exception as e:
                    logger.error(f"Error reading {stream_name}: {e}")
                    break
        
        
        # Read both streams concurrently and merge events using a queue
        event_queue = asyncio.Queue()
        
        async def read_and_queue(stream, stream_name):
            """Read stream and put events into queue."""
            async for event in read_stream(stream, stream_name):
                await event_queue.put(event)
        
        # Start both readers concurrently
        readers = [
            asyncio.create_task(read_and_queue(process.stdout, "stdout")),
            asyncio.create_task(read_and_queue(process.stderr, "stderr"))
        ]
        
        # Stream events as they come from either stream
        async def stream_from_queue():
            while True:
                # Check if both readers are done
                if all(r.done() for r in readers) and event_queue.empty():
                    break
                
                try:
                    # Wait for event with timeout to check if readers finished
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # No event available, check if we should continue
                    if all(r.done() for r in readers):
                        break
                    continue
        
        # Stream all events
        async for sse_event in stream_from_queue():
            yield sse_event
        
        # Wait for both readers to finish
        await asyncio.gather(*readers, return_exceptions=True)
        
        # Wait for process to complete
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            # Cleanup PID tracking on timeout
            if session_folder:
                process_manager.unregister(session_folder, process.pid)
            yield f"data: {{\"type\": \"error\", \"message\": \"Process timed out\"}}\n\n"
            return

        # Cleanup PID tracking on completion
        if session_folder:
            process_manager.unregister(session_folder, process.pid)
            logger.info(f"Unregistered async PID {process.pid} for session '{session_folder}'")

        # Check exit code
        if process.returncode != 0:
            yield f"data: {{\"type\": \"error\", \"message\": \"Process failed with code {process.returncode}\"}}\n\n"
        else:
            yield f"data: {{\"type\": \"complete\", \"message\": \"Process completed successfully\"}}\n\n"

    except Exception as e:
        logger.error(f"Subprocess error: {e}", exc_info=True)
        # Try to cleanup PID tracking on error (process may not exist)
        try:
            if session_folder and 'process' in locals():
                process_manager.unregister(session_folder, process.pid)
        except Exception:
            pass
        yield f"data: {{\"type\": \"error\", \"message\": \"{str(e)}\"}}\n\n"


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
        parts = line.split("|", 3)  # Max 4 parts
        if len(parts) < 3:
            return None

        level = parts[1].strip().lower()

        # Handle init event (special case: total only)
        if level == "init":
            total = int(parts[2].strip())
            message = parts[3].strip() if len(parts) > 3 else f"Found {total} items"
            return {
                "type": "init",
                "total": total,
                "message": message
            }

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
