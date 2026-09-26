"""
Parallel Citation Processor
===========================

Module de traitement parallèle des citations Publish or Perish (PoP).
Architecture optimisée avec pré-filtrage LLM.

Architecture (optimisée):
    [Citation]
         ↓
    [Phase 1: Pré-filtrage LLM] ← Titre + Abstract uniquement (rapide)
         ↓
    Pertinent? ─── Non ──→ Skip immédiatement (pas de fetch web)
         │
        Oui
         ↓
    [Phase 2: Fetch web] ← Seulement si pré-filtrage positif
         ↓
    [Phase 3: Filtrage complet + enrichissement métadonnées]
         ↓
    [SSE event immédiat]

Avantages:
    - Articles non pertinents: 1 seul appel LLM (rapide), pas de fetch web
    - Articles pertinents: pré-filtrage + fetch + filtrage complet
    - Streaming: résultats dès qu'ils arrivent

Usage:
    async for event_type, event_data in process_citations_parallel(citations, config, ...):
        if event_type == "progress":
            yield sse_event(event_data)

Albert (modèle ``albert/<id>``, optionnel) :
    - la clé ``albert_api_key`` est transmise au pré-filtre et au filtrage complet ;
    - à la première erreur de compte Albert (``AlbertAuthError``,
      ``AlbertQuotaExhausted``), les tâches restantes sont annulées et
      l'exception remonte à la route (arrêt du job entier) ;
    - le délai de garde de 120 s n'arrête pas la boucle tant qu'une tâche
      Albert tourne (réessais longs), et un signal d'arrêt resté dans la file
      après la boucle est relancé : une erreur de compte n'est jamais perdue ;
    - aucune acquisition imbriquée du sémaphore LLM global sur la branche Albert.
"""

import asyncio
import contextlib
import logging
import os
from typing import List, Dict, Tuple, AsyncGenerator, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Configuration par défaut (peut être surchargée via .env)
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_CITATION_BATCH_SIZE", "10"))
MAX_CONCURRENT_FETCHES = int(os.getenv("MAX_CONCURRENT_WEB_FETCHES", "10"))
DEFAULT_FETCH_TIMEOUT = 30
DEFAULT_MAX_CHARS = 10000

# Default LLM model (uses OpenRouter format by default)
# Format "provider/model" → OpenRouter (e.g., google/gemini-2.5-flash)
# Format "model" → OpenAI direct (e.g., gpt-4o-mini)
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

# Délai de garde (secondes) sans nouveau résultat dans la file. Chemin
# historique : la boucle s'arrête. Branche Albert : un avertissement est
# journalisé et l'attente continue tant qu'une tâche tourne (les réessais
# Albert, 429 avec Retry-After ou 503, peuvent dépasser ce délai).
RESULT_WAIT_TIMEOUT = 120.0


@dataclass
class CitationProcessingResult:
    """
    Résultat du traitement d'une citation.

    Attributes:
        index: Index global de la citation dans la liste complète
        citation: Dictionnaire de la citation originale
        status: Statut du traitement ("relevant", "skipped", "error")
        filter_result: Résultat du filtrage LLM (si relevant)
        web_source: Source du contenu web ("html", "pdf", "none")
        error_message: Message d'erreur (si status="error")
    """
    index: int
    citation: Dict[str, Any]
    status: str  # "relevant", "skipped", "error"
    filter_result: Optional[Dict[str, Any]] = None
    web_source: str = "none"
    error_message: Optional[str] = None


@dataclass
class _JobAbort:
    """
    Signal déposé dans la file des résultats : erreur qui arrête le job entier.

    Attributes:
        index: Index de la citation qui a rencontré l'erreur
        error: Exception à relancer vers la route (erreur de compte Albert, ou
            modèle ``albert/…`` demandé alors qu'Albert est désactivé)
    """
    index: int
    error: BaseException


def _albert_selected(model: Optional[str]) -> bool:
    """
    Indique si le modèle sélectionne Albert, avec le résolveur du filtre de citations.

    Même résolution que ``pre_filter_citation`` et ``filter_citation_with_llm``
    (``resolve_llm_route``, sans modèle par défaut) : aucune lecture du
    ``.env``, et aucune configuration lue pour un modèle sans préfixe
    ``albert/``. Un ``albert/…`` refusé (Albert désactivé, identifiant vide)
    compte comme Albert : chaque appel de filtre relève alors l'erreur lui-même.

    Args:
        model: Modèle de la configuration du job

    Returns:
        True pour un modèle ``albert/…``, False sinon
    """
    from app.utils.llm_note_generator import PROVIDER_ALBERT, resolve_llm_route

    try:
        return resolve_llm_route(model).provider == PROVIDER_ALBERT
    except ValueError:  # AlbertDisabledError ou « albert/ » sans identifiant
        return True


async def _next_albert_item(
    results_queue: asyncio.Queue,
    tasks: List["asyncio.Task[Any]"],
    processed_count: int,
    total: int
) -> Optional[Any]:
    """
    Prochain élément de la file sur la branche Albert, sans abandon sur délai.

    Même garde que le chemin historique (``asyncio.wait_for`` de
    ``RESULT_WAIT_TIMEOUT`` secondes), mais son expiration n'arrête pas la
    boucle tant qu'une tâche tourne : un avertissement est journalisé et
    l'attente reprend. Une erreur de compte levée après de longs réessais (429
    avec Retry-After, puis ``AlbertQuotaExhausted``) atteint donc toujours la
    boucle principale, et aucun résultat tardif n'est perdu (une lecture
    annulée par la garde laisse l'élément dans la file).

    Args:
        results_queue: File des résultats (``CitationProcessingResult`` ou ``_JobAbort``)
        tasks: Tâches de traitement des citations
        processed_count: Résultats déjà lus (journal)
        total: Nombre total de citations (journal)

    Returns:
        L'élément suivant, ou None quand toutes les tâches sont terminées et la
        file vide (tâche interrompue sans résultat)
    """
    while True:
        try:
            return await asyncio.wait_for(results_queue.get(), timeout=RESULT_WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            running = sum(1 for task in tasks if not task.done())
            if not running and results_queue.empty():
                return None
            if running:
                logger.warning(
                    f"No citation result for {RESULT_WAIT_TIMEOUT:g} s (processed {processed_count}/{total}); "
                    f"{running} Albert task(s) still running (retries), still waiting"
                )


def _drain_job_abort(results_queue: asyncio.Queue) -> Optional[_JobAbort]:
    """
    Vide la file et renvoie le premier signal d'arrêt du job qu'elle contient.

    Appelée après la boucle principale, une fois toutes les tâches terminées :
    un ``_JobAbort`` déposé après la sortie de boucle (délai de garde du chemin
    historique) n'est jamais perdu. Les résultats tardifs restent ignorés,
    comme avant.

    Args:
        results_queue: File des résultats

    Returns:
        Le premier ``_JobAbort`` trouvé, ou None
    """
    abort = None
    while not results_queue.empty():
        item = results_queue.get_nowait()
        if abort is None and isinstance(item, _JobAbort):
            abort = item
    return abort


async def process_citations_parallel(
    citations: List[Any],
    config: Dict[str, str],
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent_fetches: int = MAX_CONCURRENT_FETCHES,
    albert_api_key: Optional[str] = None
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """
    Traite les citations en parallèle avec streaming SSE immédiat.

    Architecture optimisée:
    - Fetch web + LLM filtering lancés en parallèle via asyncio.Queue
    - Résultats streamés dès qu'ils sont prêts (pas d'attente batch)
    - Semaphores séparés pour fetch (10) et LLM (global)

    Args:
        citations: Liste de citations PopCitation ou dicts
        config: Configuration avec project_name, project_description, etc.
        openai_api_key: Clé API OpenAI (optionnelle)
        openrouter_api_key: Clé API OpenRouter (optionnelle)
        batch_size: Nombre de citations traitées en parallèle (défaut: 10)
        max_concurrent_fetches: Max fetches web simultanés (défaut: 10)
        albert_api_key: Clé API Albert (optionnelle, modèle ``albert/<id>`` seulement)

    Yields:
        Tuples (event_type, event_data):
        - ("init", {"total": N, "batch_size": M})
        - ("progress", {"current": i, "total": N, "status": str, "title": str, ...})
        - ("complete", {"relevant": X, "skipped": Y})

    Raises:
        AlbertAuthError: première erreur de compte Albert (clé invalide ou
            absente, compte expiré, budget ou quota épuisé) ; les tâches
            restantes sont annulées avant la relance.
        AlbertDisabledError: modèle ``albert/…`` alors qu'Albert est désactivé.

    Example:
        >>> async for event_type, event_data in process_citations_parallel(
        ...     citations, config, api_key, None, batch_size=10
        ... ):
        ...     if event_type == "progress":
        ...         print(f"{event_data['current']}/{event_data['total']}")
    """
    total = len(citations)

    if total == 0:
        yield ("init", {"total": 0, "batch_size": batch_size})
        yield ("complete", {"relevant": 0, "skipped": 0})
        return

    logger.info(
        f"Starting streaming parallel citation processing: {total} citations, "
        f"max_concurrent={batch_size}"
    )

    yield ("init", {"total": total, "batch_size": batch_size})

    # Erreurs qui arrêtent le job entier (jamais levées hors branche Albert)
    from app.utils.llm_note_generator import ALBERT_JOB_ABORT_ERRORS

    # Branche Albert : pas d'abandon sur délai de garde tant qu'une tâche tourne
    albert_job = _albert_selected(config.get("model", DEFAULT_LLM_MODEL))

    # Utiliser une Queue pour streamer les résultats dès qu'ils arrivent
    results_queue: asyncio.Queue = asyncio.Queue()

    async def process_single_citation(idx: int, citation: Any):
        """
        Traite une citation avec pré-filtrage optimisé.

        Flux:
        1. Pre-filter LLM (titre/abstract) → Si NA, skip immédiatement
        2. Si pertinent → Fetch web + filtrage complet
        """
        from app.utils.citation_fetcher import fetch_citation_content
        from app.utils.citation_filter import filter_citation_with_llm, pre_filter_citation

        # Convertir en dict si nécessaire
        if hasattr(citation, 'model_dump'):
            c = citation.model_dump(mode='json')
        else:
            c = dict(citation) if not isinstance(citation, dict) else citation

        error_msg = None
        filter_result = None
        source = "none"

        # Phase 1: PRÉ-FILTRAGE RAPIDE (titre + abstract uniquement)
        try:
            is_relevant = await pre_filter_citation(
                citation=c,
                project_name=config["project_name"],
                project_description=config["project_description"],
                collection_name=config["collection_name"],
                collection_description=config.get("collection_description", ""),
                model=config.get("model", DEFAULT_LLM_MODEL),
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
                albert_api_key=albert_api_key
            )

            if not is_relevant:
                # Skip immédiatement - pas de fetch web nécessaire
                logger.info(f"Citation {idx} skipped by pre-filter: {c.get('title', '')[:40]}...")
                status = "skipped"
            else:
                # Phase 2: FETCH WEB (seulement si pré-filtrage positif)
                try:
                    article_url = c.get("article_url")
                    fulltext_url = c.get("fulltext_url")
                    content, source = await fetch_citation_content(
                        article_url=str(article_url) if article_url else None,
                        fulltext_url=str(fulltext_url) if fulltext_url else None,
                        timeout=DEFAULT_FETCH_TIMEOUT,
                        max_chars=DEFAULT_MAX_CHARS
                    )
                except Exception as e:
                    logger.warning(f"Fetch failed for citation {idx}: {e}")
                    content, source = "", "none"

                # Phase 3: FILTRAGE COMPLET avec enrichissement métadonnées
                result = await filter_citation_with_llm(
                    citation=c,
                    web_content=content,
                    web_content_source=source,
                    project_name=config["project_name"],
                    project_description=config["project_description"],
                    collection_name=config["collection_name"],
                    collection_description=config.get("collection_description", ""),
                    model=config.get("model", DEFAULT_LLM_MODEL),
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key,
                    albert_api_key=albert_api_key
                )

                if isinstance(result, dict):
                    status = "relevant"
                    filter_result = result
                else:
                    status = "skipped"

        except ALBERT_JOB_ABORT_ERRORS as exc:
            # Erreur de compte Albert : arrêt du job entier (signal à la boucle principale)
            logger.error(f"Albert account error on citation {idx}, aborting the job: {exc}")
            await results_queue.put(_JobAbort(index=idx, error=exc))
            return
        except Exception as e:
            logger.error(f"Processing failed for citation {idx}: {e}")
            status = "error"
            error_msg = str(e)

        # Mettre le résultat dans la queue immédiatement
        await results_queue.put(CitationProcessingResult(
            index=idx,
            citation=c,
            status=status,
            filter_result=filter_result,
            web_source=source,
            error_message=error_msg
        ))

    # Lancer toutes les tâches avec un semaphore pour limiter la concurrence globale
    concurrency_semaphore = asyncio.Semaphore(batch_size)

    async def process_with_semaphore(idx: int, citation: Any):
        async with concurrency_semaphore:
            await process_single_citation(idx, citation)

    # Créer toutes les tâches
    tasks = [
        asyncio.create_task(process_with_semaphore(idx, citation))
        for idx, citation in enumerate(citations)
    ]

    # Collecter les résultats au fur et à mesure qu'ils arrivent
    global_relevant = 0
    global_skipped = 0
    global_errors = 0
    processed_count = 0

    while processed_count < total:
        # Attendre le prochain résultat (avec timeout pour éviter deadlock)
        try:
            if albert_job:
                # Branche Albert : attente jusqu'à la fin des tâches (réessais longs)
                result = await _next_albert_item(results_queue, tasks, processed_count, total)
                if result is None:
                    logger.error(
                        f"Citation tasks ended without a result (processed {processed_count}/{total})"
                    )
                    break
            else:
                result = await asyncio.wait_for(results_queue.get(), timeout=RESULT_WAIT_TIMEOUT)

            if isinstance(result, _JobAbort):
                # Arrêt du job : annuler les tâches restantes, puis relancer vers la route
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.error(
                    f"Citation job aborted after {processed_count}/{total} results "
                    f"(citation {result.index}): {type(result.error).__name__}"
                )
                raise result.error

            processed_count += 1

            if result.status == "relevant":
                global_relevant += 1
            elif result.status == "error":
                global_errors += 1
                global_skipped += 1
            else:
                global_skipped += 1

            # Yield immédiatement le résultat
            yield ("progress", {
                "current": processed_count,
                "total": total,
                "status": result.status,
                "title": result.citation.get("title", "")[:50],
                "filter_result": result.filter_result,
                "citation": result.citation,
                "web_source": result.web_source,
                "error_message": result.error_message
            })

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for citation result (processed {processed_count}/{total})")
            break

    # S'assurer que toutes les tâches sont terminées
    await asyncio.gather(*tasks, return_exceptions=True)

    # Erreur de compte arrivée après la sortie de boucle : jamais perdue
    late_abort = _drain_job_abort(results_queue)
    if late_abort is not None:
        logger.error(
            f"Citation job aborted after {processed_count}/{total} results "
            f"(citation {late_abort.index}): {type(late_abort.error).__name__}"
        )
        raise late_abort.error

    logger.info(
        f"Streaming parallel processing complete: {global_relevant} relevant, "
        f"{global_skipped} skipped ({global_errors} errors)"
    )

    yield ("complete", {
        "relevant": global_relevant,
        "skipped": global_skipped,
        "errors": global_errors
    })


async def _fetch_batch_parallel(
    citations: List[Any],
    max_concurrent: int = MAX_CONCURRENT_FETCHES,
    timeout: int = DEFAULT_FETCH_TIMEOUT,
    max_chars: int = DEFAULT_MAX_CHARS
) -> List[Tuple[str, str]]:
    """
    Fetch web content pour un batch de citations en parallèle.

    Utilise un semaphore local pour limiter les fetches concurrents.

    Args:
        citations: Liste de citations (PopCitation ou dicts)
        max_concurrent: Maximum de fetches simultanés
        timeout: Timeout par requête en secondes
        max_chars: Maximum de caractères par contenu

    Returns:
        Liste de tuples (content, source) dans le même ordre que citations
    """
    from app.utils.citation_fetcher import fetch_citation_content

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(citation: Any) -> Tuple[str, str]:
        """Fetch une citation avec semaphore."""
        async with semaphore:
            try:
                # Convertir en dict si c'est un modèle Pydantic
                if hasattr(citation, 'model_dump'):
                    c = citation.model_dump(mode='json')
                else:
                    c = citation

                article_url = c.get("article_url")
                fulltext_url = c.get("fulltext_url")

                return await fetch_citation_content(
                    article_url=str(article_url) if article_url else None,
                    fulltext_url=str(fulltext_url) if fulltext_url else None,
                    timeout=timeout,
                    max_chars=max_chars
                )
            except asyncio.TimeoutError:
                logger.warning(f"Fetch timeout for citation: {c.get('title', 'unknown')[:30]}")
                return ("", "timeout")
            except Exception as e:
                logger.warning(f"Fetch failed: {type(e).__name__}: {e}")
                return ("", "none")

    # Lancer tous les fetches en parallèle avec gather
    tasks = [fetch_one(c) for c in citations]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Traiter les exceptions éventuelles
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Unexpected fetch error for citation {i}: {result}")
            processed_results.append(("", "none"))
        else:
            processed_results.append(result)

    return processed_results


async def _filter_batch_parallel(
    citations: List[Any],
    web_results: List[Tuple[str, str]],
    config: Dict[str, str],
    start_idx: int,
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
    albert_api_key: Optional[str] = None
) -> List[CitationProcessingResult]:
    """
    Filter un batch de citations avec LLM en parallèle.

    Utilise le semaphore global LLM pour contrôler la concurrence
    et respecter les rate limits API. Sur la branche Albert, le semaphore
    global n'est pas tenu ici : ``filter_citation_with_llm`` l'acquiert
    lui-même pendant chaque envoi (``run_llm_slot``), sans acquisition imbriquée.

    Args:
        citations: Batch de citations
        web_results: Résultats du fetch web (même ordre)
        config: Configuration du projet
        start_idx: Index de départ dans la liste complète
        openai_api_key: Clé API OpenAI
        openrouter_api_key: Clé API OpenRouter
        albert_api_key: Clé API Albert (modèle ``albert/<id>`` seulement)

    Returns:
        Liste de CitationProcessingResult dans le même ordre

    Raises:
        AlbertAuthError: erreur de compte Albert (arrêt du job entier).
        AlbertDisabledError: modèle ``albert/…`` alors qu'Albert est désactivé.
    """
    from app.utils.citation_filter import filter_citation_with_llm
    from app.utils.llm_note_generator import ALBERT_JOB_ABORT_ERRORS, get_llm_semaphore

    semaphore = get_llm_semaphore()
    model = config.get("model", "gpt-4o-mini")
    # Branche Albert : pas de semaphore externe (sinon acquisition imbriquée).
    # Même résolution que filter_citation_with_llm, sans lecture du .env.
    llm_slot = contextlib.nullcontext() if _albert_selected(model) else semaphore

    async def filter_one(
        idx: int,
        citation: Any,
        content: str,
        source: str
    ) -> CitationProcessingResult:
        """Filter une citation avec semaphore LLM global."""
        # Convertir en dict si nécessaire
        if hasattr(citation, 'model_dump'):
            c = citation.model_dump(mode='json')
        else:
            c = dict(citation) if not isinstance(citation, dict) else citation

        global_idx = start_idx + idx

        async with llm_slot:
            logger.debug(f"Filtering citation {global_idx + 1}: {c.get('title', '')[:30]}...")

            try:
                result = await filter_citation_with_llm(
                    citation=c,
                    web_content=content,
                    web_content_source=source,
                    project_name=config["project_name"],
                    project_description=config["project_description"],
                    collection_name=config["collection_name"],
                    collection_description=config.get("collection_description", ""),
                    model=config.get("model", "gpt-4o-mini"),
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key,
                    albert_api_key=albert_api_key
                )

                # Déterminer le statut basé sur le résultat
                if isinstance(result, dict):
                    return CitationProcessingResult(
                        index=global_idx,
                        citation=c,
                        status="relevant",
                        filter_result=result,
                        web_source=source
                    )
                else:
                    return CitationProcessingResult(
                        index=global_idx,
                        citation=c,
                        status="skipped",
                        web_source=source
                    )

            except ALBERT_JOB_ABORT_ERRORS:
                raise
            except Exception as e:
                logger.error(
                    f"LLM filter failed for citation {global_idx}: "
                    f"{type(e).__name__}: {e}"
                )
                return CitationProcessingResult(
                    index=global_idx,
                    citation=c,
                    status="error",
                    error_message=str(e),
                    web_source=source
                )

    # Créer les tâches de filtrage
    tasks = [
        filter_one(i, c, web_results[i][0], web_results[i][1])
        for i, c in enumerate(citations)
    ]

    # Exécuter en parallèle (contrôlé par le semaphore global)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Erreur de compte Albert : arrêt du job entier (jamais sur les autres fournisseurs)
    for result in results:
        if isinstance(result, ALBERT_JOB_ABORT_ERRORS):
            raise result

    # Traiter les exceptions non gérées
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Convertir citation en dict pour le résultat d'erreur
            c = citations[i]
            if hasattr(c, 'model_dump'):
                c = c.model_dump(mode='json')
            elif not isinstance(c, dict):
                c = dict(c)

            logger.error(f"Unhandled exception for citation {start_idx + i}: {result}")
            processed_results.append(CitationProcessingResult(
                index=start_idx + i,
                citation=c,
                status="error",
                error_message=f"Unexpected error: {result}"
            ))
        else:
            processed_results.append(result)

    return processed_results
