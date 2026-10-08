"""OCR souverain Albert (DINUM) : LightOnOCR par le chat et Mistral OCR par ``/v1/ocr``.

Deux voies, appelées par le maillon Albert de ``rad_dataframe`` (placé avant
Mistral quand ``OCR_ENABLE_ALBERT=1``) :

* ``ocr_pdf_lightonocr`` (fournisseur ``albert_lightonocr``) : chaque page est
  rastérisée par PyMuPDF (``render_page_png_b64``, 200 DPI, plus grand côté
  1 540 px) puis envoyée **seule** (message image, sans consigne) à
  ``lightonocr-2-1b`` par ``/v1/chat/completions``. Un seul ``fitz.Document``,
  rendu séquentiel dans le fil appelant ; l'HTTP part dans un pool à fenêtre
  bornée, le sémaphore n'étant tenu que pendant l'envoi (couche de réessai du
  client). Chaque page reçoit son marqueur ``<!-- Page N -->`` (page vide
  comprise) ; une page en échec reçoit en plus
  ``<!-- OCR ÉCHOUÉ (albert_lightonocr) : raison -->``, qui ne correspond pas à
  ``PAGE_MARKER_RE`` de ``book_note_generator``. ``finish_reason=length`` rend
  le résultat partiel ; au-delà de ``ALBERT_OCR_MAX_FAILED_RATIO`` pages en
  échec, ``AlbertOcrFailed`` fait passer au maillon suivant.
* ``ocr_pdf_v1`` (fournisseur ``albert_mistral_ocr``) : le PDF part en URI
  ``data:`` avec des pages **indexées à partir de 0** (``list(range(0, n))``),
  découpé au besoin (``ALBERT_OCR_PART_MB``, ``ALBERT_OCR_PART_PAGES``) par les
  helpers de ``rad_dataframe`` injectés (compression, découpage, suppression,
  analyse ``pages[]``). Un 413 donne un seul redécoupage plus fin ; un second
  413, un refus d'accès (403 ou 404 « Model … not found », D3) ou une erreur de
  configuration remontent à l'appelant, qui bascule sur LightOnOCR. Une part
  en échec pour une autre raison (transitoire épuisé, réponse vide) est reprise
  **page par page** par LightOnOCR quand l'appelant fournit ``recover_fn``
  (``ocr_pages_lightonocr``) : seules les pages encore en échec après la
  reprise reçoivent un marqueur d'échec.

Erreurs : ``AlbertAuthError`` et ``AlbertQuotaExhausted`` (erreurs de compte)
interrompent la voie et remontent telles quelles (l'appelant les mémorise pour
le processus) ; une erreur permanente de configuration (``not_found``,
``validation``, ``no_access``) interrompt aussi la voie, sans essayer les pages
suivantes. Les autres échecs restent locaux à la page (ou à la part).

Ce module n'est importé que par le maillon Albert actif : il n'est jamais
chargé quand Albert est désactivé. Il n'importe pas httpx (le client est
fourni par l'appelant).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from .errors import AlbertAuthError, AlbertPermanentError, is_ocr_access_denied, redact

logger = logging.getLogger(__name__)

PROVIDER_LIGHTONOCR = "albert_lightonocr"
"""Valeur de ``texteocr_provider`` pour l'OCR par le chat (LightOnOCR)."""

PROVIDER_MISTRAL_OCR = "albert_mistral_ocr"
"""Valeur de ``texteocr_provider`` pour ``/v1/ocr`` (Mistral OCR hébergé par Albert)."""

PROVIDERS = (PROVIDER_LIGHTONOCR, PROVIDER_MISTRAL_OCR)
"""Fournisseurs OCR Albert."""

ABORT_PERMANENT_REASONS = ("not_found", "validation", "no_access")
"""Motifs d'``AlbertPermanentError`` qui interrompent la voie (toutes les pages échoueraient)."""

REASON_MAX_CHARS = 200
"""Longueur maximale de la raison recopiée dans un marqueur d'échec."""

RENDER_AHEAD_FACTOR = 2
"""Pages rendues d'avance par envoi en cours (fenêtre du pool LightOnOCR)."""

_MB = 1024 * 1024


class AlbertOcrOutcome(NamedTuple):
    """Résultat d'une voie OCR Albert pour un document.

    Attributes:
        text: Markdown avec un marqueur ``<!-- Page N -->`` par page traitée.
        provider: ``albert_lightonocr`` ou ``albert_mistral_ocr``.
        partial: vrai si des pages sont en échec ou tronquées.
        pages_done: pages OCRisées (tronquées comprises).
        pages_total: pages du document source.
        error: explication (pages en échec, pages tronquées) ou ``None``.
        pages_attempted: pages envoyées à l'OCR (après plafond éventuel).
        pages_failed: numéros (à partir de 1) des pages en échec.
        pages_truncated: numéros des pages tronquées (``finish_reason=length``).
        pages_recovered: numéros des pages d'une part ``/v1/ocr`` en échec
            reprises par LightOnOCR (texte LightOnOCR dans un résultat
            ``albert_mistral_ocr``).
    """

    text: str
    provider: str
    partial: bool = False
    pages_done: int = 0
    pages_total: int = 0
    error: Optional[str] = None
    pages_attempted: int = 0
    pages_failed: Tuple[int, ...] = ()
    pages_truncated: Tuple[int, ...] = ()
    pages_recovered: Tuple[int, ...] = ()


class AlbertOcrFailed(Exception):
    """La voie OCR Albert n'a pas produit de résultat exploitable (maillon suivant).

    Attributes:
        provider: voie concernée (``albert_lightonocr`` ou ``albert_mistral_ocr``).
        chat_attempted: vrai si LightOnOCR a déjà repris les pages en échec de
            ``/v1/ocr`` : refaire tout le document par LightOnOCR serait inutile.
    """

    def __init__(self, provider: str, message: str, *, chat_attempted: bool = False) -> None:
        """Construit l'erreur.

        Args:
            provider: voie concernée.
            message: explication française (sans secret).
            chat_attempted: LightOnOCR a déjà repris les pages en échec.
        """
        self.provider = provider
        self.chat_attempted = bool(chat_attempted)
        super().__init__(f"OCR Albert ({provider}) : {message}")


class PageResult(NamedTuple):
    """Issue de l'OCR d'une page : texte (``ok``), ou raison d'échec.

    Attributes:
        ok: vrai si la page a été transcrite (même vide ou tronquée).
        text: texte transcrit, sans blancs de bord.
        truncated: vrai si la sortie a été coupée (``finish_reason=length``).
        reason: raison de l'échec (``ok`` faux).
    """

    ok: bool
    text: str = ""
    truncated: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Marqueurs
# ---------------------------------------------------------------------------
def page_marker(number: int) -> str:
    """Marqueur de page ``<!-- Page N -->`` (``N`` à partir de 1)."""
    return f"<!-- Page {int(number)} -->"


def clean_reason(reason: Any) -> str:
    """Raison d'échec sûre dans un commentaire HTML : masquée, sur une ligne, sans ``--``.

    Les secrets sont masqués (``errors.redact``), les blancs réduits, la
    longueur limitée à ``REASON_MAX_CHARS`` ; toute suite de tirets est réduite
    à un seul, si bien que la raison ne peut ni fermer le commentaire ni en
    ouvrir un autre (donc jamais former un marqueur de page).
    """
    text = redact(reason, limit=REASON_MAX_CHARS, emails=False)
    text = re.sub(r"-{2,}", "-", text).replace("<!", "<").strip()
    return text or "erreur inconnue"


def failure_marker(provider: str, reason: Any) -> str:
    """Marqueur d'échec ``<!-- OCR ÉCHOUÉ (fournisseur) : raison -->``.

    Il ne correspond jamais à ``PAGE_MARKER_RE`` (``<!--\\s*Page\\s+N\\s*-->``) :
    le commentaire commence par ``OCR`` et la raison est nettoyée
    (``clean_reason``).
    """
    return f"<!-- OCR ÉCHOUÉ ({provider}) : {clean_reason(reason)} -->"


def _failed_page_block(number: int, provider: str, reason: Any) -> str:
    """Bloc d'une page en échec : son marqueur puis le marqueur d'échec."""
    return f"{page_marker(number)}\n{failure_marker(provider, reason)}"


def _page_block(number: int, text: str) -> str:
    """Bloc d'une page réussie (marqueur seul si la page est vide)."""
    return f"{page_marker(number)}\n{text}" if text else page_marker(number)


def format_page_numbers(numbers: Sequence[int]) -> str:
    """Liste compacte de numéros de pages (``3, 5-7, 10``)."""
    ordered = sorted(set(int(n) for n in numbers))
    ranges: List[str] = []
    start = prev = None
    for number in ordered:
        if start is None:
            start = prev = number
        elif number == prev + 1:
            prev = number
        else:
            ranges.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = number
    if start is not None:
        ranges.append(f"{start}-{prev}" if prev != start else str(start))
    return ", ".join(ranges)


# ---------------------------------------------------------------------------
# Rastérisation et outils PDF
# ---------------------------------------------------------------------------
def render_page_png_b64(page: Any, dpi: int, max_side: int) -> str:
    """Rastérise une page PyMuPDF en PNG encodé en base64 (sans préfixe ``data:``).

    Le facteur d'échelle vaut ``min(dpi / 72, max_side / max(largeur, hauteur))`` :
    rendu à ``dpi`` sauf si le plus grand côté dépasserait ``max_side`` pixels
    (ratio conservé). Une page A4 à 200 DPI et 1 540 px donne donc un plus grand
    côté de 1 540 px (1 541 au plus après arrondi).

    Args:
        page: page ``fitz.Page``.
        dpi: résolution cible (``ALBERT_OCR_DPI``).
        max_side: plus grand côté maximal en pixels (``ALBERT_OCR_MAX_SIDE`` ;
            0 ou négatif = pas de plafond).

    Returns:
        Le PNG en base64 (ASCII).
    """
    rect = page.rect
    longest = max(float(rect.width), float(rect.height))
    zoom = max(float(dpi), 1.0) / 72.0
    if longest > 0 and max_side and int(max_side) > 0:
        zoom = min(zoom, float(max_side) / longest)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def _page_count(pdf_path: str) -> int:
    """Nombre de pages d'un PDF (0 s'il est illisible)."""
    try:
        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)
    except Exception as exc:  # noqa: BLE001 — PDF illisible : 0 = inconnu
        logger.debug("Nombre de pages indisponible pour %s : %s", pdf_path, exc)
        return 0


def _size_mb(path: str) -> float:
    """Taille d'un fichier en mégaoctets."""
    return os.path.getsize(path) / _MB


def _page_limit(total: int, max_pages: Optional[int]) -> int:
    """Pages à traiter : les ``max_pages`` premières (toutes si ``None`` ou <= 0)."""
    if max_pages is not None and int(max_pages) > 0:
        return min(int(total), int(max_pages))
    return int(total)


def ensure_pinned_models(
    client: Any,
    names: Sequence[Optional[str]],
    *,
    cache: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Fait partir des ids épinglés, jamais des alias (décision 6, D5).

    ``response.model`` recopie l'alias demandé : seul l'id envoyé garantit la
    reproductibilité. Un nom qui est déjà l'id d'une fiche du catalogue part
    tel quel (aucune requête) ; un alias ou un nom inconnu est résolu par
    ``/v1/models`` (``client.resolve_model_id``), une fois par processus quand
    ``cache`` est fourni, puis mémorisé sur le client (``remember_models``).

    Args:
        client: ``AlbertClient``.
        names: modèles configurés (``ALBERT_OCR_CHAT_MODEL``…) ; ``None`` ignoré.
        cache: résolutions déjà connues (nom → id), complété sur place.

    Returns:
        Les résolutions appliquées (nom → id), vides si tout était épinglé.

    Raises:
        catalog.ModelNotFoundError: nom absent de ``/v1/models`` (``not_found``).
    """
    from . import catalog

    mapping: Dict[str, str] = {}
    for raw in names:
        name = str(raw or "").strip()
        if name[:7].lower() == "albert/":
            name = name[7:].strip()
        if not name:
            continue
        spec = catalog.model_spec(name)
        if spec is not None and spec.id == name:
            continue
        ident = cache.get(name) if cache is not None else None
        if not ident:
            ident = client.resolve_model_id(name)
            if cache is not None:
                cache[name] = ident
        mapping[name] = ident
    if mapping:
        client.remember_models(mapping)
    return mapping


def is_abort_error(error: BaseException) -> bool:
    """Vrai si ``error`` doit interrompre la voie entière (et non une seule page).

    Erreurs de compte (``AlbertAuthError``, ``AlbertQuotaExhausted``, clé
    absente) et erreurs permanentes de configuration (modèle introuvable, type
    de modèle refusé, accès refusé) : toutes les pages échoueraient pareil.
    """
    if isinstance(error, AlbertAuthError):
        return True
    return isinstance(error, AlbertPermanentError) and error.reason in ABORT_PERMANENT_REASONS


def _error_reason(error: BaseException) -> str:
    """Raison lisible d'un échec (classe et message, déjà masqué par le client)."""
    message = str(error).strip()
    name = type(error).__name__
    return f"{name} : {message}" if message else name


# ---------------------------------------------------------------------------
# Fin commune
# ---------------------------------------------------------------------------
def _finish(
    provider: str,
    blocks: List[str],
    *,
    pages_total: int,
    attempted: int,
    failed: List[int],
    truncated: List[int],
    first_reason: Optional[str],
    max_failed_ratio: float,
    recovered: Sequence[int] = (),
    chat_attempted: bool = False,
) -> AlbertOcrOutcome:
    """Contrôle les pages en échec et assemble le résultat.

    ``recovered`` liste les pages reprises par LightOnOCR (``/v1/ocr``) ;
    ``chat_attempted`` est recopié dans ``AlbertOcrFailed`` en cas d'échec.

    Raises:
        AlbertOcrFailed: aucune page réussie, ou part d'échec au-delà de
            ``max_failed_ratio``.
    """
    done = attempted - len(failed)
    if attempted <= 0 or done <= 0:
        detail = f" ; première erreur : {clean_reason(first_reason)}" if first_reason else ""
        raise AlbertOcrFailed(
            provider,
            f"aucune page OCRisée ({len(failed)}/{attempted} en échec){detail}.",
            chat_attempted=chat_attempted,
        )
    ratio = len(failed) / attempted
    if ratio > float(max_failed_ratio):
        detail = f" ; première erreur : {clean_reason(first_reason)}" if first_reason else ""
        raise AlbertOcrFailed(
            provider,
            f"{len(failed)}/{attempted} pages en échec, au-delà de "
            f"ALBERT_OCR_MAX_FAILED_RATIO={max_failed_ratio:g}{detail}.",
            chat_attempted=chat_attempted,
        )
    if recovered:
        logger.info(
            "OCR %s : %d page(s) reprise(s) par LightOnOCR (pages %s).",
            provider, len(recovered), format_page_numbers(recovered),
        )
    notes: List[str] = []
    if failed:
        notes.append(
            f"OCR {provider} partiel : {len(failed)}/{attempted} page(s) en échec "
            f"(pages {format_page_numbers(failed)})"
        )
    if truncated:
        notes.append(
            f"{len(truncated)} page(s) tronquée(s) par la limite de sortie "
            f"(finish_reason=length, pages {format_page_numbers(truncated)})"
        )
    if notes:
        logger.warning("%s.", " ; ".join(notes))
    return AlbertOcrOutcome(
        text="\n\n".join(blocks),
        provider=provider,
        partial=bool(notes),
        pages_done=done,
        pages_total=int(pages_total),
        error=" ; ".join(notes) + "." if notes else None,
        pages_attempted=attempted,
        pages_failed=tuple(failed),
        pages_truncated=tuple(truncated),
        pages_recovered=tuple(recovered),
    )


# ---------------------------------------------------------------------------
# Voie A : LightOnOCR par le chat
# ---------------------------------------------------------------------------
def _ocr_one_page(client: Any, image_b64: str, semaphore: Any, stop: threading.Event) -> Any:
    """Envoie une page à LightOnOCR (tâche du pool).

    Renvoie ``None`` sans appel si la voie a déjà été interrompue. Une erreur
    qui interrompt la voie lève le drapeau ``stop`` **avant** de remonter, pour
    qu'aucune page suivante ne parte, même avec un seul fil.
    """
    if stop.is_set():
        return None
    try:
        return client.ocr_image(image_b64, mime="image/png", semaphore=semaphore)
    except BaseException as exc:
        if is_abort_error(exc):
            stop.set()
        raise


def _collect(
    done: Sequence[Future],
    pending: Dict[Future, int],
    results: Dict[int, PageResult],
    on_page: Optional[Callable[[int, PageResult], Any]] = None,
) -> Optional[BaseException]:
    """Range les pages terminées ; renvoie la première erreur qui interrompt la voie.

    ``on_page(indice, résultat)`` est appelé pour chaque page réussie (point de
    reprise) ; une erreur de ce rappel n'interrompt jamais l'OCR.
    """
    abort: Optional[BaseException] = None
    for future in done:
        index = pending.pop(future)
        try:
            reply = future.result()
        except Exception as exc:  # noqa: BLE001 — classé ci-dessous
            if is_abort_error(exc):
                abort = abort or exc
                continue
            reason = _error_reason(exc)
            logger.warning("OCR Albert (LightOnOCR) : page %d en échec : %s", index + 1, clean_reason(reason))
            results[index] = PageResult(ok=False, reason=reason)
            continue
        if reply is None:
            results[index] = PageResult(ok=False, reason="voie interrompue avant l'envoi")
            continue
        content = getattr(reply, "content", None)
        text = content if isinstance(content, str) else str(reply)
        results[index] = PageResult(
            ok=True,
            text=text.strip(),
            truncated=getattr(reply, "finish_reason", None) == "length" or bool(getattr(reply, "truncated", False)),
        )
        if on_page is not None:
            try:
                on_page(index, results[index])
            except Exception as exc:  # noqa: BLE001 — le point de reprise ne bloque jamais l'OCR
                logger.warning("Point de reprise OCR : page %d non enregistrée (%s).", index + 1, type(exc).__name__)
    return abort


def _pool_workers(cfg: Any, concurrency: Optional[int]) -> int:
    """Taille du pool LightOnOCR : ``concurrency``, sinon ``cfg.ocr_concurrency`` (1 au moins)."""
    return max(1, int(concurrency if concurrency is not None else getattr(cfg, "ocr_concurrency", 1) or 1))


def _lightonocr_run(
    doc: Any,
    indices: Sequence[int],
    client: Any,
    cfg: Any,
    *,
    semaphore: Any,
    workers: int,
    render_fn: Callable[[Any, int, int], str],
    on_page: Optional[Callable[[int, PageResult], Any]] = None,
) -> Dict[int, PageResult]:
    """OCR LightOnOCR des pages ``indices`` (à partir de 0) d'un ``fitz.Document`` ouvert.

    Rendu séquentiel dans le fil appelant, dans l'ordre de ``indices`` ; envois
    dans un pool de ``workers`` fils avec au plus ``RENDER_AHEAD_FACTOR ×
    workers`` pages en vol ; le sémaphore n'est tenu que pendant l'envoi.

    Returns:
        ``{index: PageResult}`` ; une page absente n'a pas été traitée.

    Raises:
        AlbertAuthError: erreur de compte (aucune page suivante n'est envoyée).
        AlbertPermanentError: erreur de configuration (``is_abort_error``).
    """
    window = workers * RENDER_AHEAD_FACTOR
    results: Dict[int, PageResult] = {}
    abort: Optional[BaseException] = None
    stop = threading.Event()
    pending: Dict[Future, int] = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="albert-ocr")
    try:
        for index in indices:
            if abort is not None or stop.is_set():
                break
            try:
                image_b64 = render_fn(doc.load_page(index), cfg.ocr_dpi, cfg.ocr_max_side)
            except Exception as exc:  # noqa: BLE001 — échec de rendu : page seule
                reason = f"rastérisation impossible ({_error_reason(exc)})"
                logger.warning("OCR Albert (LightOnOCR) : page %d en échec : %s", index + 1, clean_reason(reason))
                results[index] = PageResult(ok=False, reason=reason)
                continue
            pending[pool.submit(_ocr_one_page, client, image_b64, semaphore, stop)] = index
            while len(pending) >= window and abort is None:
                done, _running = wait(list(pending), return_when=FIRST_COMPLETED)
                abort = _collect(list(done), pending, results, on_page) or abort
        while pending and abort is None:
            done, _running = wait(list(pending), return_when=FIRST_COMPLETED)
            abort = _collect(list(done), pending, results, on_page) or abort
    finally:
        if abort is not None:
            stop.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    if abort is not None:
        raise abort
    return results


def ocr_pdf_lightonocr(
    pdf_path: str,
    client: Any,
    *,
    max_pages: Optional[int] = None,
    cfg: Any = None,
    semaphore: Any = None,
    concurrency: Optional[int] = None,
    render_fn: Callable[[Any, int, int], str] = render_page_png_b64,
    checkpoint: Any = None,
) -> AlbertOcrOutcome:
    """OCR d'un PDF par LightOnOCR (``/v1/chat/completions``, une image par page).

    Rendu séquentiel sur un seul ``fitz.Document`` (fil appelant) ; envois dans
    un pool de ``concurrency`` fils avec au plus ``2 × concurrency`` pages en
    vol ; ``semaphore`` tenu pendant l'envoi seulement (sommeils de réessai
    hors sémaphore, couche unique du client).

    Args:
        pdf_path: chemin du PDF.
        client: ``AlbertClient`` (méthode ``ocr_image``).
        max_pages: nombre de premières pages à traiter (``None`` = toutes).
        cfg: configuration (défaut : ``client.cfg``) — ``ocr_dpi``,
            ``ocr_max_side``, ``ocr_concurrency``, ``ocr_max_failed_ratio``.
        semaphore: sémaphore du processus (``ALBERT_OCR_SEMAPHORE``).
        concurrency: taille du pool (défaut ``cfg.ocr_concurrency``).
        render_fn: rastérisation ``(page, dpi, max_side) -> base64``.
        checkpoint: point de reprise (``ocr_checkpoint.PageCheckpoint``) ou
            ``None`` : les pages déjà validées sont relues sans appel, chaque
            page réussie et non tronquée est enregistrée au fil de l'eau.

    Returns:
        ``AlbertOcrOutcome`` ; ``partial`` si des pages sont en échec ou tronquées.

    Raises:
        AlbertAuthError: erreur de compte (à mémoriser par l'appelant).
        AlbertPermanentError: erreur de configuration (modèle introuvable…).
        AlbertOcrFailed: PDF vide, aucune page réussie ou trop de pages en échec.
    """
    provider = PROVIDER_LIGHTONOCR
    cfg = cfg if cfg is not None else client.cfg
    workers = _pool_workers(cfg, concurrency)

    with fitz.open(pdf_path) as doc:
        total = int(doc.page_count)
        limit = _page_limit(total, max_pages)
        if limit <= 0:
            raise AlbertOcrFailed(provider, f"PDF sans page ({os.path.basename(pdf_path)}).")
        cached = checkpoint.load(range(limit)) if checkpoint is not None else {}
        todo = [index for index in range(limit) if index not in cached]
        if cached:
            logger.info(
                "OCR Albert (LightOnOCR) : %d page(s) de %s reprise(s) depuis le point de reprise, "
                "%d à transcrire.", len(cached), os.path.basename(pdf_path), len(todo),
            )
        logger.info(
            "OCR Albert (LightOnOCR) : %d/%d page(s) de %s, %d envoi(s) simultané(s) au plus.",
            len(todo), total, os.path.basename(pdf_path), workers,
        )

        def remember(index: int, page: PageResult) -> None:
            """Enregistre une page réussie et non tronquée dans le point de reprise."""
            if checkpoint is not None and page.ok and not page.truncated:
                checkpoint.save(index, page.text)

        results = _lightonocr_run(
            doc, todo, client, cfg, semaphore=semaphore, workers=workers, render_fn=render_fn,
            on_page=remember if checkpoint is not None else None,
        ) if todo else {}
        for index, text in cached.items():
            results[index] = PageResult(ok=True, text=text)

    blocks: List[str] = []
    failed: List[int] = []
    truncated: List[int] = []
    first_reason: Optional[str] = None
    for index in range(limit):
        number = index + 1
        page = results.get(index) or PageResult(ok=False, reason="page non traitée")
        if not page.ok:
            failed.append(number)
            first_reason = first_reason or page.reason
            blocks.append(_failed_page_block(number, provider, page.reason))
            continue
        if page.truncated:
            truncated.append(number)
        blocks.append(_page_block(number, page.text))
    return _finish(
        provider,
        blocks,
        pages_total=total,
        attempted=limit,
        failed=failed,
        truncated=truncated,
        first_reason=first_reason,
        max_failed_ratio=getattr(cfg, "ocr_max_failed_ratio", 0.5),
    )


def ocr_pages_lightonocr(
    pdf_path: str,
    client: Any,
    indices: Sequence[int],
    *,
    cfg: Any = None,
    semaphore: Any = None,
    concurrency: Optional[int] = None,
    render_fn: Callable[[Any, int, int], str] = render_page_png_b64,
) -> Dict[int, PageResult]:
    """OCR LightOnOCR de pages choisies d'un PDF (reprise d'une part ``/v1/ocr`` en échec).

    Mêmes rendu, pool, sémaphore et paramètres que ``ocr_pdf_lightonocr`` ;
    seules les pages ``indices`` (à partir de 0, hors limites ignorées) sont
    envoyées. Aucun seuil d'échec ici : l'appelant assemble et décide.

    Args:
        pdf_path: chemin du PDF d'origine (numérotation du document entier).
        client: ``AlbertClient`` (méthode ``ocr_image``).
        indices: indices des pages à transcrire, à partir de 0.
        cfg: configuration (défaut : ``client.cfg``).
        semaphore: sémaphore du processus (``ALBERT_OCR_SEMAPHORE``).
        concurrency: taille du pool (défaut ``cfg.ocr_concurrency``).
        render_fn: rastérisation ``(page, dpi, max_side) -> base64``.

    Returns:
        ``{index: PageResult}`` pour chaque page envoyée (ou dont le rendu a échoué).

    Raises:
        AlbertAuthError: erreur de compte.
        AlbertPermanentError: erreur de configuration (modèle introuvable, type
            refusé, accès refusé : ``is_abort_error``).
    """
    cfg = cfg if cfg is not None else client.cfg
    workers = _pool_workers(cfg, concurrency)
    with fitz.open(pdf_path) as doc:
        total = int(doc.page_count)
        wanted = [int(index) for index in dict.fromkeys(indices) if 0 <= int(index) < total]
        if not wanted:
            return {}
        logger.info(
            "OCR Albert (LightOnOCR) : reprise de %d page(s) de %s (pages %s).",
            len(wanted), os.path.basename(pdf_path), format_page_numbers([i + 1 for i in wanted]),
        )
        return _lightonocr_run(
            doc, wanted, client, cfg, semaphore=semaphore, workers=workers, render_fn=render_fn
        )


# ---------------------------------------------------------------------------
# Voie B : /v1/ocr (Mistral OCR hébergé par Albert)
# ---------------------------------------------------------------------------
def _v1_prepare_parts(
    pdf_path: str,
    total: int,
    cfg: Any,
    *,
    split_fn: Callable[..., List[str]],
    compress_fn: Optional[Callable[[str], str]],
    temp_paths: List[str],
) -> List[Tuple[str, int]]:
    """Parts à envoyer à ``/v1/ocr`` : ``[(chemin, pages)]`` dans l'ordre.

    Sous ``ALBERT_OCR_PART_MB`` et ``ALBERT_OCR_PART_PAGES`` : le fichier
    d'origine seul. Sinon, compression (si trop gros) puis découpage ; les
    fichiers temporaires sont ajoutés à ``temp_paths`` (supprimés par
    l'appelant).
    """
    part_mb = float(cfg.ocr_part_mb)
    part_pages = int(cfg.ocr_part_pages)
    size_mb = _size_mb(pdf_path)
    if size_mb <= part_mb and total <= part_pages:
        return [(pdf_path, total)]
    source = pdf_path
    if compress_fn is not None and size_mb > part_mb:
        try:
            compressed = compress_fn(pdf_path)
        except Exception as exc:  # noqa: BLE001 — compression facultative
            logger.warning("OCR Albert /v1/ocr : compression impossible pour %s : %s", pdf_path, exc)
        else:
            temp_paths.append(compressed)
            if _size_mb(compressed) <= part_mb and total <= part_pages:
                return [(compressed, total)]
            source = compressed
    paths = list(split_fn(source, max_size_mb=part_mb, max_pages=part_pages))
    temp_paths.extend(paths)
    logger.info(
        "OCR Albert /v1/ocr : %s découpé en %d part(s) (≤ %g Mo / ≤ %d pages).",
        os.path.basename(pdf_path), len(paths), part_mb, part_pages,
    )
    return [(path, _page_count(path)) for path in paths]


def _v1_ocr_part(
    client: Any,
    part_path: str,
    count: int,
    offset: int,
    *,
    parse_fn: Callable[..., str],
    split_fn: Callable[..., List[str]],
    unlink_fn: Callable[[Optional[str]], None],
    semaphore: Any,
    resplit: bool,
) -> str:
    """OCR des ``count`` premières pages d'une part (indices 0 à ``count - 1``).

    Les marqueurs valent ``index + 1 + offset`` (``parse_fn(payload,
    page_offset=offset)``). Un 413 donne un seul redécoupage plus fin si
    ``resplit`` est vrai et que la part a plus d'une page.

    Raises:
        AlbertPermanentError: 413 persistant, refus d'accès, autre erreur permanente.
        AlbertOcrFailed: réponse sans texte.
    """
    with open(part_path, "rb") as fh:
        data = fh.read()
    try:
        payload = client.ocr_document(
            data, pages=list(range(0, count)), mime="application/pdf", semaphore=semaphore
        )
    except AlbertPermanentError as exc:
        if exc.reason != "payload_too_large" or not resplit or count <= 1:
            raise
        logger.warning(
            "OCR Albert /v1/ocr : part trop volumineuse (413, pages %d-%d) : un seul redécoupage plus fin.",
            offset + 1, offset + count,
        )
        return _v1_resplit(
            client, part_path, count, offset,
            parse_fn=parse_fn, split_fn=split_fn, unlink_fn=unlink_fn, semaphore=semaphore,
        )
    text = parse_fn(payload, page_offset=offset)
    if not isinstance(text, str) or not text.strip():
        raise AlbertOcrFailed(PROVIDER_MISTRAL_OCR, f"réponse /v1/ocr vide (pages {offset + 1}-{offset + count}).")
    return text


def _v1_resplit(
    client: Any,
    part_path: str,
    count: int,
    offset: int,
    *,
    parse_fn: Callable[..., str],
    split_fn: Callable[..., List[str]],
    unlink_fn: Callable[[Optional[str]], None],
    semaphore: Any,
) -> str:
    """Redécoupage unique d'une part refusée en 413 : moitié de la taille et des pages.

    Les sous-parts ne sont jamais redécoupées : un nouveau 413 remonte.
    """
    half_pages = max(1, (int(count) + 1) // 2)
    sub_paths = list(split_fn(part_path, max_size_mb=max(_size_mb(part_path) / 2.0, 0.001), max_pages=half_pages))
    try:
        blocks: List[str] = []
        sub_offset = offset
        remaining = int(count)
        for sub_path in sub_paths:
            if remaining <= 0:
                break
            sub_total = _page_count(sub_path)
            sub_count = min(sub_total, remaining)
            if sub_count <= 0:
                continue
            blocks.append(_v1_ocr_part(
                client, sub_path, sub_count, sub_offset,
                parse_fn=parse_fn, split_fn=split_fn, unlink_fn=unlink_fn,
                semaphore=semaphore, resplit=False,
            ))
            sub_offset += sub_total
            remaining -= sub_total
        return "\n\n".join(blocks)
    finally:
        for path in sub_paths:
            unlink_fn(path)


class _V1Assembly:
    """Assemblage des parts de ``/v1/ocr`` : blocs, pages en échec, reprises LightOnOCR.

    Attributes:
        blocks: blocs de texte dans l'ordre des pages.
        failed: numéros (à partir de 1) des pages restées en échec.
        truncated: pages reprises dont la sortie LightOnOCR a été coupée.
        recovered: pages reprises par LightOnOCR.
        first_reason: raison du premier échec de part.
        recover_fn: reprise ``indices -> {index: PageResult}`` (``None`` : aucune).
        chat_attempted: vrai si LightOnOCR a traité au moins une page reprise.
    """

    def __init__(self, recover_fn: Optional[Callable[[List[int]], Mapping[int, PageResult]]]) -> None:
        """Prépare un assemblage vide ; ``recover_fn`` active la reprise page par page."""
        self.blocks: List[str] = []
        self.failed: List[int] = []
        self.truncated: List[int] = []
        self.recovered: List[int] = []
        self.first_reason: Optional[str] = None
        self.recover_fn = recover_fn
        self.chat_attempted = False

    def add_failed_part(self, offset: int, count: int, error: BaseException) -> None:
        """Part en échec (pages ``offset + 1`` à ``offset + count``) : reprise LightOnOCR, puis marqueurs.

        Les pages reprises reçoivent leur marqueur et le texte LightOnOCR ;
        les autres, leur marqueur suivi du marqueur d'échec ``albert_mistral_ocr``.

        Raises:
            AlbertAuthError: erreur de compte pendant la reprise (la voie s'arrête).
        """
        reason = _error_reason(error)
        self.first_reason = self.first_reason or reason
        indices = list(range(offset, offset + count))
        results = self._recover(indices, reason) if self.recover_fn is not None else None
        if results is None:
            logger.warning(
                "OCR Albert /v1/ocr : pages %d-%d en échec : %s", offset + 1, offset + count, clean_reason(reason)
            )
        for index in indices:
            number = index + 1
            page = results.get(index) if results else None
            if page is not None and page.ok:
                self.recovered.append(number)
                if page.truncated:
                    self.truncated.append(number)
                self.blocks.append(_page_block(number, page.text))
                continue
            self.failed.append(number)
            why = reason if page is None else f"{reason} ; reprise LightOnOCR : {page.reason}"
            self.blocks.append(_failed_page_block(number, PROVIDER_MISTRAL_OCR, why))

    def _recover(self, indices: List[int], reason: str) -> Mapping[int, PageResult]:
        """Reprise LightOnOCR des pages ``indices`` ; ``{}`` si elle échoue en bloc.

        Une erreur qui rend LightOnOCR inutilisable (``is_abort_error``) coupe
        les reprises suivantes de ce document ; une erreur de compte remonte.
        """
        first, last = indices[0] + 1, indices[-1] + 1
        logger.warning(
            "OCR Albert /v1/ocr : pages %d-%d en échec (%s) : reprise page par page par LightOnOCR.",
            first, last, clean_reason(reason),
        )
        try:
            results = self.recover_fn(indices) if self.recover_fn is not None else {}
        except AlbertAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 — pages laissées en échec, part suivante
            if is_abort_error(exc):
                self.recover_fn = None
            logger.warning(
                "OCR Albert /v1/ocr : reprise LightOnOCR des pages %d-%d impossible : %s",
                first, last, clean_reason(_error_reason(exc)),
            )
            return {}
        if results:
            self.chat_attempted = True
        return results or {}


def ocr_pdf_v1(
    pdf_path: str,
    client: Any,
    *,
    split_fn: Callable[..., List[str]],
    unlink_fn: Callable[[Optional[str]], None],
    parse_fn: Callable[..., str],
    compress_fn: Optional[Callable[[str], str]] = None,
    max_pages: Optional[int] = None,
    cfg: Any = None,
    semaphore: Any = None,
    recover_fn: Optional[Callable[[List[int]], Mapping[int, PageResult]]] = None,
) -> AlbertOcrOutcome:
    """OCR d'un PDF par ``/v1/ocr`` (``mistral-ocr-2512``, accès restreint).

    Le document part en URI ``data:application/pdf;base64,…`` avec
    ``pages = list(range(0, n))`` (indices **à partir de 0**, jamais ceux du
    chemin Mistral natif). Au-delà de ``ALBERT_OCR_PART_MB`` ou
    ``ALBERT_OCR_PART_PAGES``, le PDF est compressé (``compress_fn``) puis
    découpé (``split_fn``). Les marqueurs ``<!-- Page N -->`` sont continus
    d'une part à l'autre (``N = index + 1 + décalage``). Une part en échec
    (transitoire épuisé, réponse vide) est reprise page par page par
    ``recover_fn`` (LightOnOCR) quand il est fourni ; chaque page encore en
    échec reçoit un marqueur d'échec et rend le résultat partiel.

    Args:
        pdf_path: chemin du PDF.
        client: ``AlbertClient`` (méthode ``ocr_document``).
        split_fn: découpage ``(chemin, max_size_mb=, max_pages=) -> [parts]``
            (``rad_dataframe._split_pdf_for_ocr``).
        unlink_fn: suppression sûre d'un fichier temporaire (``_safe_unlink``).
        parse_fn: analyse ``(payload, page_offset=) -> markdown``
            (``rad_dataframe._ocr_pages_payload_to_markdown``).
        compress_fn: recompression ``(chemin) -> chemin temporaire``
            (``_compress_pdf_for_ocr``), facultative.
        max_pages: nombre de premières pages à traiter (``None`` = toutes).
        cfg: configuration (défaut : ``client.cfg``).
        semaphore: sémaphore du processus (``ALBERT_OCR_SEMAPHORE``).
        recover_fn: reprise des pages d'une part en échec, ``indices (à partir
            de 0, numérotation du document entier) -> {index: PageResult}``
            (``ocr_pages_lightonocr``) ; ``None`` : pas de reprise.

    Returns:
        ``AlbertOcrOutcome`` (fournisseur ``albert_mistral_ocr`` ;
        ``pages_recovered`` liste les pages servies par LightOnOCR).

    Raises:
        AlbertAuthError: erreur de compte.
        AlbertPermanentError: refus d'accès (``errors.is_ocr_access_denied``),
            413 persistant après redécoupage, erreur de configuration.
        AlbertOcrFailed: PDF vide, aucune page réussie ou trop de pages en
            échec (``chat_attempted`` vrai si LightOnOCR a déjà repris des pages).
    """
    provider = PROVIDER_MISTRAL_OCR
    cfg = cfg if cfg is not None else client.cfg
    total = _page_count(pdf_path)
    if total <= 0:
        raise AlbertOcrFailed(provider, f"PDF sans page ou illisible ({os.path.basename(pdf_path)}).")
    limit = _page_limit(total, max_pages)
    temp_paths: List[str] = []
    try:
        parts = _v1_prepare_parts(
            pdf_path, total, cfg, split_fn=split_fn, compress_fn=compress_fn, temp_paths=temp_paths
        )
        assembly = _V1Assembly(recover_fn)
        offset = 0
        remaining = limit
        for part_path, part_pages in parts:
            if remaining <= 0:
                break
            count = min(int(part_pages), remaining)
            if count <= 0:
                continue
            part_error: Optional[BaseException] = None
            try:
                assembly.blocks.append(_v1_ocr_part(
                    client, part_path, count, offset,
                    parse_fn=parse_fn, split_fn=split_fn, unlink_fn=unlink_fn,
                    semaphore=semaphore, resplit=True,
                ))
            except AlbertPermanentError as exc:
                if is_abort_error(exc) or is_ocr_access_denied(exc) or exc.reason == "payload_too_large":
                    raise
                part_error = exc
            except AlbertAuthError:
                raise
            except Exception as exc:  # noqa: BLE001 — échec local à la part : récupération
                part_error = exc
            if part_error is not None:
                assembly.add_failed_part(offset, count, part_error)
            offset += int(part_pages)
            remaining -= int(part_pages)
        return _finish(
            provider,
            assembly.blocks,
            pages_total=total,
            attempted=limit,
            failed=assembly.failed,
            truncated=assembly.truncated,
            first_reason=assembly.first_reason,
            max_failed_ratio=getattr(cfg, "ocr_max_failed_ratio", 0.5),
            recovered=assembly.recovered,
            chat_attempted=assembly.chat_attempted,
        )
    finally:
        for path in temp_paths:
            unlink_fn(path)


__all__ = [
    "ABORT_PERMANENT_REASONS",
    "PROVIDERS",
    "PROVIDER_LIGHTONOCR",
    "PROVIDER_MISTRAL_OCR",
    "AlbertOcrFailed",
    "AlbertOcrOutcome",
    "PageResult",
    "clean_reason",
    "ensure_pinned_models",
    "failure_marker",
    "format_page_numbers",
    "is_abort_error",
    "ocr_pages_lightonocr",
    "ocr_pdf_lightonocr",
    "ocr_pdf_v1",
    "page_marker",
    "render_page_png_b64",
]
