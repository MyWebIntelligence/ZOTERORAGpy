"""
Upload Safety
=============

Confinement and limits of the files written under ``uploads/`` by the upload
routes (audit A01 of 2026-09-27):

* storage names are generated on the server: the client file name only
  gives a sanitised, human-readable stem (``safe_filename_stem``) and is kept
  as metadata, never joined as a path;
* every destination, temporary files and cleanup targets included, is
  resolved and checked strictly under its root before being opened or
  removed (``confined_path``, ``safe_remove_tree``, ``safe_remove_file``);
* the size of an uploaded file (``UPLOAD_MAX_MB``), the total uncompressed
  size of a ZIP archive (``UPLOAD_MAX_UNZIPPED_MB``) and its number of
  members (``UPLOAD_MAX_ZIP_ENTRIES``) are bounded; the uncompressed size is
  counted while extracting, since ZIP headers may lie.

The ZIP extraction helpers live here so that the ingestion routes and the
project upload routes share them.
"""
import logging
import os
import re
import shutil
import unicodedata
import uuid
import zipfile
from typing import BinaryIO, Optional, Tuple

logger = logging.getLogger(__name__)

# Limits (MB / count), read at call time so that an operator change applies
# at the next request. 0 or a negative value disables a limit.
DEFAULT_UPLOAD_MAX_MB = 4096
DEFAULT_UNZIPPED_MAX_MB = 16384
DEFAULT_ZIP_MAX_ENTRIES = 100000

_STEM_MAX_CHARS = 100
_UNSAFE_STEM_CHARS_RE = re.compile(r"[^\w .,()+=@&'-]+", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")
_COPY_CHUNK_BYTES = 1024 * 1024


class UnsafePathError(ValueError):
    """A destination that does not resolve strictly under its root directory."""


class UploadTooLargeError(ValueError):
    """An upload, or the content of an archive, above the configured limit."""


class UnsafeZipMemberError(zipfile.BadZipFile):
    """A ZIP member whose name would be extracted outside the destination directory.

    Subclass of ``zipfile.BadZipFile``: callers that already refuse invalid
    archives (``/upload_zip`` answers 400) refuse these the same way.
    """


class ZipLimitError(UploadTooLargeError):
    """A ZIP archive with too many members or too large once uncompressed."""


def _limit_bytes(env_name: str, default_mb: int) -> Optional[int]:
    """Limit in bytes from ``env_name`` (MB), ``None`` when disabled (0 or negative)."""
    raw = os.getenv(env_name)
    try:
        value = float(raw) if raw not in (None, "") else float(default_mb)
    except ValueError:
        logger.warning(f"Invalid {env_name}={raw!r}; using {default_mb} MB")
        value = float(default_mb)
    return int(value * 1024 * 1024) if value > 0 else None


def max_upload_bytes() -> Optional[int]:
    """Maximum size of one uploaded file (``UPLOAD_MAX_MB``, default 4096 MB); None when disabled."""
    return _limit_bytes("UPLOAD_MAX_MB", DEFAULT_UPLOAD_MAX_MB)


def max_unzipped_bytes() -> Optional[int]:
    """Maximum uncompressed size of a ZIP archive (``UPLOAD_MAX_UNZIPPED_MB``, default 16384 MB)."""
    return _limit_bytes("UPLOAD_MAX_UNZIPPED_MB", DEFAULT_UNZIPPED_MAX_MB)


def max_zip_entries() -> Optional[int]:
    """Maximum number of members of a ZIP archive (``UPLOAD_MAX_ZIP_ENTRIES``, default 100000)."""
    raw = os.getenv("UPLOAD_MAX_ZIP_ENTRIES")
    try:
        value = int(raw) if raw not in (None, "") else DEFAULT_ZIP_MAX_ENTRIES
    except ValueError:
        logger.warning(f"Invalid UPLOAD_MAX_ZIP_ENTRIES={raw!r}; using {DEFAULT_ZIP_MAX_ENTRIES}")
        value = DEFAULT_ZIP_MAX_ENTRIES
    return value if value > 0 else None


def safe_filename_stem(filename: Optional[str], default: str = "upload") -> str:
    """
    Readable, path-free stem of a client file name.

    Only the last component survives (``/`` and ``\\`` both separate), the
    extension is dropped, control characters and characters outside a
    conservative set are replaced by ``_``, leading dots are removed and the
    result is truncated to 100 characters. The stem never contains a
    separator, ``..`` as a whole or a NUL byte.

    Args:
        filename: File name sent by the client (may be absolute, contain
            ``..`` segments or be empty).
        default: Stem used when nothing usable is left.

    Returns:
        The sanitised stem.
    """
    name = unicodedata.normalize("NFC", str(filename or ""))
    name = name.replace("\x00", "")
    name = re.split(r"[\\/]", name)[-1]
    stem, _ext = os.path.splitext(name)
    stem = "".join(ch if ch.isprintable() else "_" for ch in stem)
    stem = _UNSAFE_STEM_CHARS_RE.sub("_", stem)
    stem = _SPACES_RE.sub(" ", stem).strip(" .")
    stem = stem[:_STEM_MAX_CHARS].rstrip(" .")
    return stem or default


def filename_extension(filename: Optional[str]) -> str:
    """
    Lower-cased extension of the last component of a client file name (``""`` if none).

    Args:
        filename: File name sent by the client.

    Returns:
        The extension with its dot (``".csv"``), lower-cased.
    """
    name = re.split(r"[\\/]", str(filename or "").replace("\x00", ""))[-1]
    return os.path.splitext(name)[1].lower()


def new_session_folder_name(filename: Optional[str]) -> str:
    """
    Server-generated session folder name: ``<8 hex>_<safe stem>``.

    Args:
        filename: File name sent by the client (only its sanitised stem is used).

    Returns:
        A single path component, unique in practice.
    """
    return f"{uuid.uuid4().hex[:8]}_{safe_filename_stem(filename)}"


def confined_path(root: str, *parts: str) -> str:
    """
    Join ``parts`` under ``root`` and check the result stays strictly under ``root``.

    Symbolic links are resolved on both sides.

    Args:
        root: Directory that must contain the result.
        *parts: Path components (server-generated names).

    Returns:
        The joined path (unresolved spelling).

    Raises:
        UnsafePathError: When the resolved path is ``root`` itself, outside
            it, or cannot be resolved.
    """
    target = os.path.join(root, *parts)
    real_root = os.path.realpath(root)
    try:
        resolved = os.path.realpath(target)
        inside = resolved != real_root and os.path.commonpath([real_root, resolved]) == real_root
    except (TypeError, ValueError):
        inside = False
    if not inside:
        raise UnsafePathError(f"Path outside {root!r}: {os.path.join(*parts) if parts else ''!r}")
    return target


def save_upload(source: BinaryIO, destination: str, max_bytes: Optional[int] = None) -> int:
    """
    Copy an uploaded stream to ``destination``, refusing it above ``max_bytes``.

    The partial file is removed when the limit is exceeded or the copy fails.

    Args:
        source: Readable binary stream (``UploadFile.file``).
        destination: Target path (already checked with ``confined_path``).
        max_bytes: Size limit in bytes (``max_upload_bytes()``), None for no limit.

    Returns:
        Number of bytes written.

    Raises:
        UploadTooLargeError: Above ``max_bytes``.
        OSError: Write failure.
    """
    written = 0
    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise UploadTooLargeError(
                        f"Fichier trop volumineux : limite de {max_bytes // (1024 * 1024)} Mo (UPLOAD_MAX_MB)."
                    )
                handle.write(chunk)
    except BaseException:
        safe_remove_file(destination)
        raise
    return written


def save_upload_atomic(source: BinaryIO, destination: str, max_bytes: Optional[int] = None) -> int:
    """
    Replace ``destination`` with an uploaded stream, atomically.

    The stream is written to a temporary file next to ``destination`` and
    moved over it (``os.replace``) only once complete: a refused (too large)
    or failed upload leaves the previous file untouched.

    Args:
        source: Readable binary stream (``UploadFile.file``).
        destination: Target path (already checked with ``confined_path``).
        max_bytes: Size limit in bytes, None for no limit.

    Returns:
        Number of bytes written.

    Raises:
        UploadTooLargeError: Above ``max_bytes`` (``destination`` unchanged).
        OSError: Write failure (``destination`` unchanged).
    """
    directory = os.path.dirname(destination) or "."
    temporary = os.path.join(directory, f".{os.path.basename(destination)}.{uuid.uuid4().hex[:8]}.upload")
    written = save_upload(source, temporary, max_bytes)
    try:
        os.replace(temporary, destination)
    except BaseException:
        safe_remove_file(temporary)
        raise
    return written


def safe_remove_tree(path: str, root: str) -> bool:
    """
    Remove the directory ``path`` only when it lies strictly under ``root``.

    A symbolic link is never followed: it is refused (``rmtree`` would
    otherwise delete what it points to on some platforms).

    Args:
        path: Directory to remove.
        root: Directory that must contain ``path``.

    Returns:
        True when something was removed, False otherwise (absent, outside
        ``root`` or a symbolic link).
    """
    try:
        confined_path(root, os.path.relpath(path, root))
    except (UnsafePathError, ValueError):
        logger.warning(f"Refused to remove a directory outside {root!r}: {path!r}")
        return False
    if os.path.islink(path) or not os.path.isdir(path):
        return False
    shutil.rmtree(path)
    return True


def safe_remove_file(path: str, root: Optional[str] = None) -> bool:
    """
    Remove the file ``path`` (never a directory), optionally only under ``root``.

    Args:
        path: File to remove.
        root: Directory that must contain ``path`` (None: no confinement check).

    Returns:
        True when the file was removed.
    """
    if root is not None:
        try:
            confined_path(root, os.path.relpath(path, root))
        except (UnsafePathError, ValueError):
            logger.warning(f"Refused to remove a file outside {root!r}: {path!r}")
            return False
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
            return True
    except OSError as exc:
        logger.warning(f"Could not remove {path!r}: {exc}")
    return False


ARCHIVE_EXTENSIONS = (".zip", ".ZIP", ".tar.gz", ".tgz")


def remove_session_files(session_folder: str, upload_dir: str) -> dict:
    """
    Remove a session folder, its empty parent and its uploaded archive, confined to ``upload_dir``.

    Shared by the session deletion routes and the expired-session cleanup:
    a stored folder that does not resolve strictly under ``upload_dir`` (a
    legacy row with an unsafe name) removes nothing.

    Args:
        session_folder: Folder of the session, relative to ``upload_dir``
            (``uuid_name`` or ``uuid_name/Root`` for a ZIP with a single root).
        upload_dir: The uploads directory.

    Returns:
        ``{"deleted": [labels], "file_count": int, "total_size": int}``;
        labels are ``folder:<folder>``, ``parent:<name>``, ``archive:<name>``.
    """
    result = {"deleted": [], "file_count": 0, "total_size": 0}
    session_folder = str(session_folder or "")
    head = session_folder.split("/")[0] if "/" in session_folder else session_folder
    try:
        session_path = confined_path(upload_dir, session_folder)
    except UnsafePathError:
        logger.warning(f"Refused to remove a session outside {upload_dir!r}: {session_folder!r}")
        return result

    if os.path.isdir(session_path) and not os.path.islink(session_path):
        for dirpath, _dirnames, filenames in os.walk(session_path):
            for filename in filenames:
                try:
                    result["total_size"] += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    pass
                result["file_count"] += 1
        if safe_remove_tree(session_path, upload_dir):
            result["deleted"].append(f"folder:{session_folder}")

    if "/" in session_folder and head:
        try:
            parent_path = confined_path(upload_dir, head)
            if os.path.isdir(parent_path) and not os.path.islink(parent_path) and not os.listdir(parent_path):
                os.rmdir(parent_path)
                result["deleted"].append(f"parent:{head}")
        except (OSError, UnsafePathError):
            pass

    if head:
        for ext in ARCHIVE_EXTENSIONS:
            try:
                archive_path = confined_path(upload_dir, f"{head}{ext}")
            except UnsafePathError:
                continue
            if os.path.isfile(archive_path):
                try:
                    size = os.path.getsize(archive_path)
                except OSError:
                    size = 0
                if safe_remove_file(archive_path, upload_dir):
                    result["total_size"] += size
                    result["file_count"] += 1
                    result["deleted"].append(f"archive:{head}{ext}")
    return result


# =============================================================================
# ZIP extraction
# =============================================================================
# macOS Zotero exports use NFD (decomposed) Unicode for filenames. When
# extracted on some systems, the encoding gets corrupted. This map fixes
# common corruption patterns.
FILENAME_CORRUPTION_MAP = {
    # Combining accent sequences (macOS NFD decomposition artifacts)
    'ë̀': 'è', 'ë': 'é', 'é': 'é', 'è': 'è',
    'e╠ü': 'é', 'e╠Ç': 'è',
    'a╠Ç': 'à', 'à': 'à',
    'î': 'î', 'ô': 'ô',
    'ù': 'ù', 'û': 'û',
    'c╠º': 'ç', 'ç': 'ç',
    # Windows CP1252/UTF-8 misinterpretation artifacts
    'ΓÇÖ': "'", 'ΓÇô': '–', 'ΓÇ£': '"', 'ΓÇ¥': '"',
    'ΓÇª': '…', 'ΓÇö': '—',
    '┬½': '«', '┬╗': '»',
    'Γé¼': '€',
    # Double-encoded accents
    'é╠ü': 'é', 'è╠Ç': 'è',
    # Explicit NFD sequences
    'é': 'é', 'è': 'è',
    'à': 'à', 'ç': 'ç',
}


def fix_zip_filename_encoding(filename: str) -> str:
    """
    Fix corrupted filename encoding from ZIP extraction.

    This handles common issues when extracting ZIP files created on macOS
    with French/accented characters:
    - NFD (decomposed) Unicode normalization
    - CP437/UTF-8 encoding mismatches
    - Windows codepage artifacts

    Args:
        filename: The potentially corrupted filename

    Returns:
        The corrected filename with proper Unicode (NFC normalized)
    """
    fixed = filename

    # Apply corruption fixes (longer patterns first for correct replacement)
    for corrupt, correct in sorted(FILENAME_CORRUPTION_MAP.items(), key=lambda x: -len(x[0])):
        fixed = fixed.replace(corrupt, correct)

    # Normalize to NFC (composed form) - this is the standard for most systems
    fixed = unicodedata.normalize('NFC', fixed)

    return fixed


def zip_member_target(dst_dir: str, member_name: str) -> str:
    """
    Target path of a ZIP member, refused when it does not stay under ``dst_dir``.

    ``..`` components and absolute names (which ``os.path.join`` would keep
    as is) are resolved with symbolic links followed, then compared
    component by component with the resolved destination.

    Args:
        dst_dir: Extraction directory.
        member_name: Member name after the encoding fix.

    Returns:
        ``os.path.join(dst_dir, member_name)``.

    Raises:
        UnsafeZipMemberError: When the resolved target is not ``dst_dir`` or
            a path under it (or cannot be resolved).
    """
    target_path = os.path.join(dst_dir, member_name)
    root = os.path.realpath(dst_dir)
    try:
        resolved = os.path.realpath(target_path)
        inside = os.path.commonpath([root, resolved]) == root
    except ValueError:  # embedded NUL byte, or different drives (Windows)
        inside = False
    if not inside:
        logger.warning(f"Refused ZIP member outside the extraction directory: {member_name!r}")
        raise UnsafeZipMemberError(f"ZIP member outside the extraction directory: {member_name!r}")
    return target_path


def _check_zip_limits(infos) -> None:
    """Refuse an archive whose declared member count or uncompressed size exceeds the limits."""
    max_entries = max_zip_entries()
    if max_entries is not None and len(infos) > max_entries:
        raise ZipLimitError(
            f"Archive refusée : {len(infos)} fichiers, limite de {max_entries} (UPLOAD_MAX_ZIP_ENTRIES)."
        )
    max_total = max_unzipped_bytes()
    declared = sum(max(0, info.file_size) for info in infos)
    if max_total is not None and declared > max_total:
        raise ZipLimitError(
            f"Archive refusée : {declared // (1024 * 1024)} Mo une fois décompressée, "
            f"limite de {max_total // (1024 * 1024)} Mo (UPLOAD_MAX_UNZIPPED_MB)."
        )


def extract_zip_with_encoding_fix(zip_path: str, dst_dir: str) -> int:
    """
    Extract ZIP file with automatic filename encoding correction.

    This function extracts files from a ZIP archive while fixing common
    encoding issues with French/accented filenames. Every member name is
    checked before anything is written: when one would land outside
    ``dst_dir`` (zip slip), the whole archive is refused. The member count
    and the declared uncompressed size are checked first, and the bytes
    actually written are counted while extracting (headers may lie).

    Args:
        zip_path: Path to the ZIP file
        dst_dir: Destination directory for extraction

    Returns:
        Number of files that had their names corrected

    Raises:
        zipfile.BadZipFile: If the ZIP file is invalid, including
            ``UnsafeZipMemberError`` for a member outside ``dst_dir``
        ZipLimitError: Too many members, or too large once uncompressed
    """
    corrected_count = 0
    max_total = max_unzipped_bytes()
    written_total = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
        infos = z.infolist()
        _check_zip_limits(infos)
        members = [(info, fix_zip_filename_encoding(info.filename)) for info in infos]
        targets = [zip_member_target(dst_dir, fixed_name) for _info, fixed_name in members]

        for (info, fixed_name), target_path in zip(members, targets):
            member = info.filename
            # Track if we made corrections
            if fixed_name != member:
                corrected_count += 1
                logger.debug(f"Fixed filename: {member} -> {fixed_name}")

            # Handle directories
            if member.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
                continue

            # Ensure parent directory exists
            parent_dir = os.path.dirname(target_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Extract the file content and write with corrected name
            with z.open(info) as src, open(target_path, 'wb') as dst:
                while True:
                    chunk = src.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    written_total += len(chunk)
                    if max_total is not None and written_total > max_total:
                        raise ZipLimitError(
                            f"Archive refusée : plus de {max_total // (1024 * 1024)} Mo une fois "
                            f"décompressée (UPLOAD_MAX_UNZIPPED_MB)."
                        )
                    dst.write(chunk)

    if corrected_count > 0:
        logger.info(f"Fixed encoding for {corrected_count} filenames during ZIP extraction")

    return corrected_count


def split_processing_path(dst_dir: str) -> Tuple[str, list]:
    """
    Processing folder of an extracted archive and its file tree.

    When the archive holds a single root folder, the processing folder is
    that folder; otherwise the extraction directory itself.

    Args:
        dst_dir: Extraction directory.

    Returns:
        ``(processing_path, tree)``, the tree listing folders (with a
        trailing ``/``) and files relative to the processing folder.
    """
    extracted_items = os.listdir(dst_dir)
    processing_path = dst_dir
    if len(extracted_items) == 1:
        single_item_path = os.path.join(dst_dir, extracted_items[0])
        if os.path.isdir(single_item_path) and not os.path.islink(single_item_path):
            processing_path = single_item_path
    tree = []
    for root, dirs, files in os.walk(processing_path):
        for d in dirs:
            tree.append(os.path.relpath(os.path.join(root, d), processing_path) + '/')
        for fname in files:
            tree.append(os.path.relpath(os.path.join(root, fname), processing_path))
    return processing_path, tree
