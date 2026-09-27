"""
LLM Note Generator
==================

This module handles the generation of structured reading notes using Large Language Models
(LLMs). It integrates with OpenAI and OpenRouter APIs to analyze document content and
produce academic-style reading notes.

Key Features:
- Prompt Engineering: Dynamically builds prompts based on document metadata and language.
- Multi-Provider Support: Supports OpenAI and OpenRouter, plus Albert (DINUM) as an
  opt-in third provider selected by an explicit ``albert/<id>`` model name.
- Concurrency Control: Uses a global semaphore to limit concurrent API calls.
- Idempotence: Generates unique sentinels to track generated notes.
- Fallback Mechanism: Provides a template-based fallback if LLM generation fails.

Albert (sovereign provider, OFF by default):
- ``resolve_llm_route`` is the single provider resolver (``albert/`` prefix first,
  then the historical ``provider/model`` → OpenRouter heuristic).
- A request that selected Albert never falls back to OpenAI or OpenRouter; the
  ``gpt-4o-mini`` fallback is reserved to the OpenAI/OpenRouter branch.
- One retry layer only (``scripts/rad_albert/retry.py``): the historical
  ``max_attempts`` loop is bypassed on the Albert branch.
- ``run_llm_slot`` holds the global semaphore (then the Albert semaphore) during
  one attempt only; the limiter is acquired and the backoff slept outside them.
- Account errors (``AlbertAuthError``, ``AlbertQuotaExhausted``) always reach the
  caller; a transient error that exhausted its retries falls back to the template.
- Usage ledger of the web side: the public entry points accept a trailing
  keyword argument ``albert_usage_ledger`` (a ``UsageLedger`` created by the
  route for an Albert request only). On the Albert path it is handed to every
  ``AlbertClient`` built for the request (``ledger=``), so each Albert call is
  recorded into it (pinned model id sent, ``response.model``, tokens, cost,
  impacts); ``None`` or an OpenAI/OpenRouter path changes nothing.
"""

import os
import uuid
import math
import weakref
import logging
import asyncio
import functools
import importlib
import html as html_module
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values, find_dotenv

if TYPE_CHECKING:  # annotations only: the usage module is never imported here at runtime
    from scripts.rad_albert.usage import UsageLedger

try:
    from scripts.rad_albert.config import AlbertConfig
    from scripts.rad_albert.errors import (
        AlbertAuthError,
        AlbertDisabledError,
        AlbertModelBusy,
        AlbertQuotaExhausted,
        AlbertTruncatedError,
    )
    from scripts.rad_providers import (
        ALBERT_PREFIX,
        PROVIDER_ALBERT,
        PROVIDER_OPENROUTER,
        ProviderResolution,
        resolve_llm_provider,
    )
except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
    from rad_albert.config import AlbertConfig
    from rad_albert.errors import (
        AlbertAuthError,
        AlbertDisabledError,
        AlbertModelBusy,
        AlbertQuotaExhausted,
        AlbertTruncatedError,
    )
    from rad_providers import (
        ALBERT_PREFIX,
        PROVIDER_ALBERT,
        PROVIDER_OPENROUTER,
        ProviderResolution,
        resolve_llm_provider,
    )

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Sentinel prefix for idempotence
SENTINEL_PREFIX = "ragpy-note-id:"

# =============================================================================
# Note Mode Configuration
# =============================================================================

# Mapping of note modes to template files
# NOTE: "book" mode is multi-phase and dispatched to book_note_generator.py;
# its template file (book_prompt.md) is parsed there, not loaded by _load_prompt_template().
TEMPLATE_MAP = {
    "extended": "zotero_prompt.md",
    "short": "zotero_prompt_short.md",
    "pedagogique": "zotero_prompt_pedagogique.md",
    "evaluation": "zotero_prompt_evaluation.md",
    "book": "book_prompt.md"
}

# Mapping of note modes to display prefixes (for Zotero note identification)
NOTE_MODE_PREFIX = {
    "extended": "[FICHE]",
    "pedagogique": "[CLAIR]",
    "evaluation": "[EVAL]",
    "book": "[LIVRE]",
    "short": ""  # No prefix for short summaries
}

# Mapping of note modes to max_tokens (per LLM call ; book mode uses
# multiple smaller calls and overrides this in book_note_generator).
NOTE_MODE_MAX_TOKENS = {
    "extended": 16000,
    "short": 2000,
    "pedagogique": 10000,
    "evaluation": 10000,
    "book": 8000
}

# Display names for UI
NOTE_MODE_DISPLAY = {
    "extended": "Fiche de lecture [FICHE]",
    "pedagogique": "Fiche pédagogique [CLAIR]",
    "evaluation": "Grille d'évaluation [EVAL]",
    "book": "Fiche de lecture livre [LIVRE]",
    "short": "Résumé court"
}

# =============================================================================
# Global LLM Semaphore for Concurrency Control
# =============================================================================
_llm_semaphore: Optional[asyncio.Semaphore] = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """
    Get the global LLM semaphore (lazy initialization).

    This semaphore limits concurrent LLM API calls across ALL users
    to prevent API rate limiting and resource exhaustion.

    Returns:
        asyncio.Semaphore configured with MAX_CONCURRENT_LLM_CALLS
    """
    global _llm_semaphore
    if _llm_semaphore is None:
        max_concurrent = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', '5'))
        _llm_semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"LLM semaphore initialized: max {max_concurrent} concurrent calls")
    return _llm_semaphore


# Default web LLM model: environment variable name and last-resort value.
DEFAULT_LLM_MODEL_ENV = "OPENROUTER_DEFAULT_MODEL"
FALLBACK_LLM_MODEL = "gpt-4o-mini"


def resolve_default_llm_model(openrouter_model: Optional[str] = None) -> str:
    """
    Resolve the default web LLM model (single source for the web side).

    Resolution order, first non-empty value wins:

    1. ``openrouter_model`` (explicit argument);
    2. ``OPENROUTER_DEFAULT_MODEL`` read fresh, at each call, from the ``.env``
       file found by ``find_dotenv()``, with ``dotenv_values`` (``os.environ``
       is never modified, so no secret of the file is loaded into the process);
    3. ``OPENROUTER_DEFAULT_MODEL`` from the process environment;
    4. ``"gpt-4o-mini"``.

    The ``.env`` file therefore wins over the process environment whenever it
    holds a non-empty value, whatever keys the caller has. Compared with the
    former ``_get_llm_clients`` logic:

    * when a key was missing (``None``), the former overriding ``.env`` reload
      already made the ``.env`` value win: same model, but ``os.environ`` is no
      longer mutated;
    * when both keys were passed, the former code read the process environment
      only. The model now differs when the ``.env`` file and the process
      environment disagree (a shell export, or an admin edit through
      ``/save_credentials``, which writes ``.env`` but not ``os.environ``): the
      ``.env`` value wins, so such an edit applies without a restart;
    * an empty value is skipped (the former code could return ``""``).

    Without a ``.env`` file (Docker image: ``.env`` is excluded by
    ``.dockerignore`` and passed as ``env_file``), the process environment is
    used, as before.

    Args:
        openrouter_model: Explicit default model, if the caller has one.

    Returns:
        The default model identifier (never empty).
    """
    if openrouter_model:
        return openrouter_model
    try:
        file_value = dotenv_values(find_dotenv()).get(DEFAULT_LLM_MODEL_ENV)
    except Exception as exc:  # unreadable file: fall back to the process environment
        logger.debug(f"Could not read {DEFAULT_LLM_MODEL_ENV} from .env: {exc}")
        file_value = None
    if file_value:
        return file_value
    env_value = os.getenv(DEFAULT_LLM_MODEL_ENV)
    if env_value:
        return env_value
    return FALLBACK_LLM_MODEL


def _get_llm_clients(
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None
) -> Tuple[Optional[OpenAI], Optional[OpenAI], str]:
    """
    Initializes and returns LLM clients from explicit credentials only.

    API keys are never read from the environment or from ``.env`` here: callers
    pass the keys resolved for the current user (``get_credential_or_env``,
    which applies the role-based ``.env`` fallback for admins only). A missing
    key simply means that the matching client is not built.

    Args:
        openai_api_key: OpenAI API key. If None or empty, no OpenAI client.
        openrouter_api_key: OpenRouter API key. If None or empty, no OpenRouter client.
        openrouter_model: Default model. If None, resolved by
            ``resolve_default_llm_model`` (``.env`` value, then environment,
            then ``gpt-4o-mini``).

    Returns:
        A tuple containing:
        - openai_client (Optional[OpenAI]): An initialized OpenAI client if an
          OpenAI key was passed, otherwise None.
        - openrouter_client (Optional[OpenAI]): An initialized client for OpenRouter
          if an OpenRouter key was passed, otherwise None.
        - default_model (str): The default model identifier.

    Security Note:
        There is no environment fallback for the keys: a non-admin user without
        personal keys never gets the server keys through this function.
    """
    default_model = resolve_default_llm_model(openrouter_model)

    openai_client = None
    openrouter_client = None

    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.debug("OpenAI client initialized for note generation")

    if openrouter_api_key:
        openrouter_client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.debug("OpenRouter client initialized for note generation")

    return openai_client, openrouter_client, default_model


# =============================================================================
# Albert (DINUM) — sovereign chat provider, opt-in (ALBERT_ENABLED=1)
# =============================================================================

# Account errors: never retried, never swallowed, they stop the whole job and
# reach the route (``credential_required = albert_api_key``).
ALBERT_ACCOUNT_ERRORS = (AlbertAuthError, AlbertQuotaExhausted)

# Errors that abort a whole multi-call job (book note, citation batch): account
# errors, plus an ``albert/`` model requested while Albert is disabled.
ALBERT_JOB_ABORT_ERRORS = (AlbertAuthError, AlbertQuotaExhausted, AlbertDisabledError)

# System message of the Albert note calls (same wording as the OpenAI branch).
ALBERT_NOTE_SYSTEM_PROMPT = "Tu es un assistant spécialisé en rédaction de fiches de lecture académiques."

# Roles of the Albert model catalog (scripts/rad_albert/catalog.py).
ALBERT_NOTES_ROLE = "notes"
ALBERT_LONG_CONTEXT_ROLE = "long_context"

# Prompt size estimate (characters per token) and share of the model context
# beyond which the call switches to the ``long_context`` role.
ALBERT_CHARS_PER_TOKEN_ESTIMATE = 3.2
ALBERT_CONTEXT_MARGIN = 0.9

# Floor of the answer budget in "short" mode (before the reasoning headroom,
# which the Albert client adds itself for a reasoning model such as gpt-oss).
ALBERT_SHORT_MIN_TOKENS = 4096

# Empty answer with finish_reason=length: one retry with this budget factor and
# this reasoning effort, then an error (never an empty note).
ALBERT_TRUNCATION_BUDGET_FACTOR = 1.5
ALBERT_TRUNCATION_RETRY_EFFORT = "low"


# Package that provided AlbertConfig and the error classes above
# (``scripts.rad_albert`` or ``rad_albert``): the submodules imported lazily
# below come from the same package, so the error classes always match.
_ALBERT_PACKAGE = AlbertConfig.__module__.rsplit(".", 1)[0]


def _albert_module(name: str) -> Any:
    """
    Import lazily a submodule of the Albert package already in use.

    The httpx-based modules (``client``) and the other helpers (``catalog``,
    ``limiter``, ``retry``) are only loaded when Albert is selected.

    Args:
        name: Submodule name (``client``, ``catalog``, ``limiter``, ``retry``).

    Returns:
        The imported module.
    """
    return importlib.import_module(f"{_ALBERT_PACKAGE}.{name}")


def resolve_llm_route(model: Optional[str]) -> ProviderResolution:
    """
    Resolve the chat provider of a model name (single resolver of the web side).

    The ``albert/`` prefix is tested first (case-insensitive); the Albert switch
    (``ALBERT_ENABLED``) is only read for such a name, so any other model is
    routed exactly as before (``provider/model`` → OpenRouter, else OpenAI) and
    no configuration is read.

    Args:
        model: Model name as entered (``None`` is treated as ``""``).

    Returns:
        ``ProviderResolution(provider, wire_model, credential_key)``.

    Raises:
        AlbertDisabledError: ``albert/…`` requested while Albert is disabled.
        ValueError: ``albert/`` without a model identifier.
    """
    text = "" if model is None else str(model)
    albert_enabled = False
    if text[:len(ALBERT_PREFIX)].lower() == ALBERT_PREFIX:
        albert_enabled = AlbertConfig.from_env().enabled
    return resolve_llm_provider(text, albert_enabled=albert_enabled)


def albert_route(model: Optional[str]) -> Optional[ProviderResolution]:
    """
    Return the Albert resolution of the effective model, or ``None``.

    An empty model means the web default (``resolve_default_llm_model``), so an
    ``albert/…`` default model is honoured like an explicit one. While Albert is
    disabled, an empty model returns ``None`` without reading ``.env``: the
    historical path resolves the default itself (once, in the executor) and
    raises ``AlbertDisabledError`` there for an ``albert/…`` default.

    Args:
        model: Model name as entered, or ``None``/``""`` for the default.

    Returns:
        The resolution when the effective model selects Albert, otherwise
        ``None`` (OpenAI/OpenRouter: the historical code path applies).

    Raises:
        AlbertDisabledError: explicit ``albert/…`` requested while Albert is disabled.
        ValueError: ``albert/`` without a model identifier.
    """
    if not model and not AlbertConfig.from_env().enabled:
        return None
    effective = model if model else resolve_default_llm_model()
    resolution = resolve_llm_route(effective)
    return resolution if resolution.provider == PROVIDER_ALBERT else None


async def albert_route_async(model: Optional[str]) -> Tuple[Optional[ProviderResolution], Optional[str]]:
    """
    Event-loop version of ``albert_route``: ``.env`` is never read on the loop thread.

    * explicit model: ``albert_route(model)`` (no file read), model unchanged;
    * empty model, Albert disabled: ``(None, model)`` without any file read (the
      historical path resolves the default once, in the executor, as before);
    * empty model, Albert enabled: the web default is resolved once, in a worker
      thread, and returned so that the caller passes it on (the executor side
      then routes the very same model).

    Args:
        model: Model name as entered, or ``None``/``""`` for the default.

    Returns:
        ``(resolution, model)``: the Albert resolution or ``None``, and the model
        to pass on (the resolved default when it was read, otherwise ``model``).

    Raises:
        AlbertDisabledError: explicit ``albert/…`` requested while Albert is disabled.
        ValueError: ``albert/`` without a model identifier.
    """
    if model or not AlbertConfig.from_env().enabled:
        return albert_route(model), model
    effective = await asyncio.to_thread(resolve_default_llm_model)
    return albert_route(effective), effective


def albert_wire_id(wire_model: Optional[str]) -> Optional[str]:
    """
    Return the pinned model id sent to Albert for a wire model name.

    A known alias (``openweight-large``, ``openai/gpt-oss-120b``…) is mapped to
    its pinned id with the static catalog (no network call), because the API
    echoes the requested alias in ``response.model``; an unknown name is kept.

    Args:
        wire_model: Model name without the ``albert/`` prefix (or ``None``).

    Returns:
        The pinned id, the name unchanged, or ``None`` for an empty name.
    """
    if not wire_model or not str(wire_model).strip():
        return None
    return _albert_module("catalog").canonical_id(wire_model)


def _get_albert_client(
    albert_api_key: Optional[str],
    *,
    use_limiter: bool = True,
    ledger: Optional["UsageLedger"] = None,
):
    """
    Build an Albert client from the key passed by the caller (never from the environment).

    The key is the one resolved for the current user by the route
    (``get_credential_or_env``: personal key, ``.env`` fallback for admins
    only). The client module (httpx) is imported lazily, so it is never loaded
    while Albert is not selected.

    Args:
        albert_api_key: The caller's Albert key; ``None`` or empty raises.
        use_limiter: ``False`` when the proactive limiter is acquired by the
            caller (``run_llm_slot``), outside the semaphores.
        ledger: Usage ledger of the request (``UsageLedger``), shared by every
            client of the request so that each Albert call is recorded into
            it; ``None`` keeps the historical construction (the client then
            owns a private ledger, never written).

    Returns:
        An ``AlbertClient`` configured from ``AlbertConfig.from_env()``.

    Raises:
        AlbertMissingKeyError: No key (an ``AlbertAuthError``: account error,
            ``credential_required = albert_api_key``).
    """
    AlbertClient = _albert_module("client").AlbertClient
    options: Dict[str, Any] = {"use_limiter": use_limiter}
    if ledger is not None:
        options["ledger"] = ledger
    return AlbertClient(
        AlbertConfig.from_env(),
        albert_api_key if albert_api_key is not None else "",
        **options,
    )


_albert_llm_semaphore: Optional[asyncio.Semaphore] = None
_albert_llm_semaphore_loop: Optional["weakref.ReferenceType[asyncio.AbstractEventLoop]"] = None


def get_albert_llm_semaphore() -> asyncio.Semaphore:
    """
    Get the Albert web semaphore (lazy, ``ALBERT_NOTES_CONCURRENCY`` slots).

    It caps the Albert calls of the process (notes, book notes, citation
    filter) and is always acquired after the global LLM semaphore. It is
    created again when the running event loop changes (an asyncio semaphore is
    bound to one loop); in the server there is a single loop.

    Returns:
        The asyncio semaphore of the running loop.
    """
    global _albert_llm_semaphore, _albert_llm_semaphore_loop
    loop = asyncio.get_running_loop()
    bound = _albert_llm_semaphore_loop() if _albert_llm_semaphore_loop is not None else None
    if _albert_llm_semaphore is None or bound is not loop:
        size = AlbertConfig.from_env().notes_concurrency
        _albert_llm_semaphore = asyncio.Semaphore(size)
        _albert_llm_semaphore_loop = weakref.ref(loop)
        logger.debug(f"Albert LLM semaphore initialized: max {size} concurrent calls")
    return _albert_llm_semaphore


@dataclass(frozen=True)
class AlbertSlotPolicy:
    """
    Retry policy of an Albert LLM slot (``run_llm_slot``, decision 15).

    Attributes:
        retry: ``RetryPolicy`` of the single retry layer (not ``single_attempt``:
            each attempt of the thunk is a single send, the loop is here).
        limiter: Proactive limiter of the role, acquired outside the semaphores.
        tokens: Estimated input tokens of one attempt (TPM budget).
        sleep: Backoff sleep (default ``asyncio.sleep``), outside the semaphores.
        rng: Random generator of the backoff jitter.
        label: Label added to the retry logs.
    """

    retry: Any
    limiter: Any = None
    tokens: int = 0
    sleep: Optional[Callable[[float], Any]] = None
    rng: Any = None
    label: Optional[str] = None


def albert_slot_policy(role: str, tokens: int = 0, *, cfg: Any = None) -> AlbertSlotPolicy:
    """
    Build the slot policy of an Albert role from the server configuration.

    Args:
        role: Albert role (``notes``, ``citation``, ``book_structure``,
            ``long_context``…), which selects the limiter budget.
        tokens: Estimated input tokens of one attempt.
        cfg: ``AlbertConfig`` (default: ``AlbertConfig.from_env()``).

    Returns:
        The policy: full retry policy, process-wide limiter of the role.
    """
    cfg = cfg if cfg is not None else AlbertConfig.from_env()
    return AlbertSlotPolicy(
        retry=_albert_module("retry").RetryPolicy.from_config(cfg),
        limiter=_albert_module("limiter").get_limiter(role, cfg),
        tokens=int(tokens or 0),
        label=role,
    )


def albert_estimate_tokens(payload: Any, cfg: Any = None) -> int:
    """
    Estimate the input tokens of a prompt or a message list (limiter budget).

    Args:
        payload: Prompt text or chat messages.
        cfg: ``AlbertConfig`` (default: ``AlbertConfig.from_env()``).

    Returns:
        The estimate (``ALBERT_TOKEN_ESTIMATOR``), >= 0.
    """
    cfg = cfg if cfg is not None else AlbertConfig.from_env()
    return _albert_module("limiter").estimate_input_tokens(payload, estimator=cfg.token_estimator)


async def run_llm_slot(semaphore: Any, thunk: Callable[[], Any], *, albert_policy: Optional[AlbertSlotPolicy] = None) -> Any:
    """
    Run one blocking LLM call in a slot of the global LLM semaphore.

    Without ``albert_policy`` (OpenAI, OpenRouter) this is strictly the
    historical code: the semaphore is held while ``thunk`` runs in the default
    executor, and any exception propagates.

    With ``albert_policy`` (Albert), in this order, for each attempt:

    1. the limiter of the role is acquired outside any semaphore
       (``asyncio.to_thread``);
    2. one single attempt (``thunk``, run in a thread) under the global
       semaphore, then under the Albert semaphore (``ALBERT_NOTES_CONCURRENCY``),
       always acquired in this order and released right after the send;
    3. on a transient error, the backoff is slept outside both semaphores
       (``asyncio.sleep``), then a new attempt.

    This is the single retry layer of the Albert web path
    (``rad_albert.retry.acall_with_retry``); the event loop is never blocked.

    Args:
        semaphore: The global LLM semaphore (``get_llm_semaphore()``).
        thunk: Blocking callable without argument; on the Albert path it must
            make exactly one send (``single_attempt=True``).
        albert_policy: Albert slot policy, ``None`` for OpenAI/OpenRouter.

    Returns:
        The value returned by ``thunk``.

    Raises:
        Exception: Whatever ``thunk`` raises (after the retries on Albert:
            account, permanent and truncation errors are never retried).
    """
    if albert_policy is None:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, thunk)
    acall_with_retry = _albert_module("retry").acall_with_retry
    return await acall_with_retry(
        thunk,
        policy=albert_policy.retry,
        semaphore=[semaphore, get_albert_llm_semaphore()],
        limiter=albert_policy.limiter,
        tokens=albert_policy.tokens,
        sleep=albert_policy.sleep,
        rng=albert_policy.rng,
        label=albert_policy.label,
    )


@dataclass(frozen=True)
class AlbertChatRequest:
    """
    One Albert chat request of the web side (notes, book notes).

    Attributes:
        messages: Chat messages (system + user).
        model: Pinned model id placed first, or ``None`` for the role chain.
        role: Albert role (fallback chain on repeated 503, limiter, timeout).
        max_tokens: Answer budget before the reasoning headroom (added by the
            client for a reasoning model).
        temperature: Sampling temperature.
        response_format: Structured output format (``json_object`` or
            ``json_schema``, D10), or ``None``.
    """

    messages: List[Dict[str, str]]
    model: Optional[str]
    role: str
    max_tokens: int
    temperature: Optional[float]
    response_format: Optional[Dict[str, Any]] = None


def _albert_role_for_prompt(messages: List[Dict[str, str]], model: Optional[str], role: str) -> Tuple[str, Optional[str]]:
    """
    Switch to the ``long_context`` role when the prompt is too long for the model.

    The prompt size is estimated at ``len / 3.2`` tokens; beyond 0.9 × the
    context of the target model (the explicit model, else the primary model of
    the role), the call uses the ``long_context`` role chain (262 144 tokens).

    Args:
        messages: Chat messages.
        model: Explicit pinned model id, or ``None``.
        role: Requested Albert role.

    Returns:
        ``(role, model)`` unchanged, or ``("long_context", None)``.
    """
    if role == ALBERT_LONG_CONTEXT_ROLE:
        return role, model
    albert_catalog = _albert_module("catalog")
    target = model or albert_catalog.primary_model(role, today=date.today())
    context = albert_catalog.context_length(target) if target else None
    if not context:
        return role, model
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    estimate = chars / ALBERT_CHARS_PER_TOKEN_ESTIMATE
    if estimate > ALBERT_CONTEXT_MARGIN * context:
        logger.warning(
            f"Albert: prompt estimated at {int(estimate)} tokens (> {ALBERT_CONTEXT_MARGIN:.0%} of the "
            f"{context}-token context of {target}), using the {ALBERT_LONG_CONTEXT_ROLE} role"
        )
        return ALBERT_LONG_CONTEXT_ROLE, None
    return role, model


def albert_chat_request(
    prompt: str,
    wire_model: Optional[str],
    *,
    mode: str = "extended",
    temperature: Optional[float] = 0.2,
    role: str = ALBERT_NOTES_ROLE,
    response_format: Optional[Dict[str, Any]] = None,
) -> AlbertChatRequest:
    """
    Build the Albert chat request of a note prompt.

    Budget: ``NOTE_MODE_MAX_TOKENS[mode]`` (floor ``ALBERT_SHORT_MIN_TOKENS`` in
    short mode); the Albert client adds the reasoning headroom
    (``ALBERT_REASONING_HEADROOM``) and the reasoning effort for gpt-oss.

    Args:
        prompt: User prompt.
        wire_model: Model name without the ``albert/`` prefix (alias accepted),
            or ``None`` for the role chain.
        mode: Note mode (``extended``, ``short``, ``pedagogique``,
            ``evaluation``, ``book``).
        temperature: Sampling temperature.
        role: Albert role.
        response_format: Structured output format, or ``None``.

    Returns:
        The request.
    """
    messages = [
        {"role": "system", "content": ALBERT_NOTE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    role, model = _albert_role_for_prompt(messages, albert_wire_id(wire_model), role)
    max_tokens = NOTE_MODE_MAX_TOKENS.get(mode, 16000)
    if mode == "short":
        max_tokens = max(max_tokens, ALBERT_SHORT_MIN_TOKENS)
    return AlbertChatRequest(
        messages=messages,
        model=model,
        role=role,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )


def _albert_retry_budget(max_tokens: int) -> int:
    """Answer budget of the single retry after an empty truncated answer (× 1.5)."""
    return int(math.ceil(max_tokens * ALBERT_TRUNCATION_BUDGET_FACTOR))


def _albert_answer_text(result: Any) -> str:
    """
    Return the stripped text of an Albert chat result, never an empty note.

    Only ``message.content`` is ever used (the ``reasoning`` field of gpt-oss
    is never taken as content).

    Args:
        result: ``ChatResult`` returned by ``AlbertClient.chat``.

    Returns:
        The stripped content.

    Raises:
        ValueError: Empty content (finish reason other than ``length``).
    """
    finish = getattr(result, "finish_reason", None)
    served = getattr(result, "model", None)
    text = str(result if result is not None else "").strip()
    if not text:
        raise ValueError(f"Albert returned empty response (finish_reason={finish}). Model: {served}")
    if finish in ("length", "content_filter"):
        logger.warning(f"Albert answer truncated (finish_reason={finish}, model {served})")
    return text


def _generate_with_albert(
    prompt: str,
    wire_model: Optional[str],
    *,
    temperature: float,
    mode: str,
    albert_api_key: Optional[str],
    role: str = ALBERT_NOTES_ROLE,
    response_format: Optional[Dict[str, Any]] = None,
    albert_usage_ledger: Optional["UsageLedger"] = None,
) -> str:
    """
    Generate note content with Albert, synchronously (single retry layer: the client's).

    The client retries transient errors (backoff outside any semaphore, limiter
    of the role) and falls back along the role chain after repeated 503. An
    empty answer with ``finish_reason=length`` gets one retry with a budget
    × 1.5 and a ``low`` reasoning effort, then raises. There is no fallback to
    OpenAI or OpenRouter.

    Args:
        prompt: User prompt.
        wire_model: Model name without the ``albert/`` prefix.
        temperature: Sampling temperature.
        mode: Note mode (budget).
        albert_api_key: The caller's Albert key.
        role: Albert role.
        response_format: Structured output format, or ``None``.
        albert_usage_ledger: Usage ledger of the request, given to the client
            (every call, retries included, is recorded into it), or ``None``.

    Returns:
        The generated content.

    Raises:
        AlbertAuthError: Account error (invalid key, expired account, budget,
            quota), including a missing key.
        AlbertTruncatedError: Empty truncated answer twice.
        AlbertError: Other classified errors, after the retries.
        ValueError: Empty answer.
    """
    request = albert_chat_request(
        prompt, wire_model, mode=mode, temperature=temperature, role=role, response_format=response_format
    )
    client = _get_albert_client(albert_api_key, ledger=albert_usage_ledger)
    logger.info(f"Using Albert with model: {request.model or '(role ' + request.role + ')'}")
    try:
        def call(max_tokens: int, reasoning_effort: Optional[str]) -> Any:
            """One Albert chat call (client retries and role fallback included)."""
            return client.chat(
                request.messages,
                request.model,
                role=request.role,
                max_tokens=max_tokens,
                temperature=request.temperature,
                response_format=request.response_format,
                reasoning_effort=reasoning_effort,
            )

        try:
            result = call(request.max_tokens, None)
        except AlbertTruncatedError:
            budget = _albert_retry_budget(request.max_tokens)
            logger.warning(
                f"Albert returned an empty truncated answer (finish_reason=length); "
                f"single retry with max_tokens={budget} and reasoning effort '{ALBERT_TRUNCATION_RETRY_EFFORT}'"
            )
            result = call(budget, ALBERT_TRUNCATION_RETRY_EFFORT)
    finally:
        client.close()
    return _albert_answer_text(result)


async def _albert_slot_chat(
    client: Any,
    request: AlbertChatRequest,
    semaphore: Any,
    *,
    max_tokens: int,
    reasoning_effort: Optional[str],
) -> Any:
    """
    One Albert chat call through ``run_llm_slot``, with the role fallback on 503.

    Each model of the chain (explicit model first, then the role chain) is
    tried with single-attempt sends retried by ``run_llm_slot``; after
    ``ALBERT_BUSY_RETRIES`` answers « Model is too busy », the next model of
    the chain is used (``ALBERT_MODEL_FALLBACK=0`` keeps the first model only).

    Args:
        client: ``AlbertClient`` built without its own limiter.
        request: The chat request.
        semaphore: The global LLM semaphore.
        max_tokens: Answer budget of this call.
        reasoning_effort: Reasoning effort (``None`` = server configuration).

    Returns:
        The ``ChatResult``.

    Raises:
        AlbertModelBusy: Every model of the chain stayed busy.
        AlbertError: Other classified errors, after the retries.
    """
    chain = client.chat_chain(request.role, request.model)
    policy = albert_slot_policy(
        request.role, albert_estimate_tokens(request.messages, client.cfg), cfg=client.cfg
    )
    for index, wire in enumerate(chain):
        thunk = functools.partial(
            client.chat,
            request.messages,
            wire,
            role=request.role,
            max_tokens=max_tokens,
            temperature=request.temperature,
            response_format=request.response_format,
            reasoning_effort=reasoning_effort,
            single_attempt=True,
        )
        try:
            return await run_llm_slot(semaphore, thunk, albert_policy=policy)
        except AlbertModelBusy:
            if index + 1 < len(chain):
                logger.warning(
                    f"Albert: model {wire} busy (503), falling back to {chain[index + 1]} (role {request.role})"
                )
                continue
            raise
    raise AlbertModelBusy(detail=f"no model available for role {request.role}")


async def agenerate_with_albert(
    prompt: str,
    wire_model: Optional[str],
    *,
    semaphore: Any,
    temperature: float,
    mode: str,
    albert_api_key: Optional[str],
    role: str = ALBERT_NOTES_ROLE,
    response_format: Optional[Dict[str, Any]] = None,
    albert_usage_ledger: Optional["UsageLedger"] = None,
) -> str:
    """
    Generate note content with Albert from the event loop (``run_llm_slot``).

    Same contract as ``_generate_with_albert``, but each send holds the global
    semaphore then the Albert semaphore only during the send; the limiter is
    acquired and the backoff slept outside them.

    Args:
        prompt: User prompt.
        wire_model: Model name without the ``albert/`` prefix.
        semaphore: The global LLM semaphore (``get_llm_semaphore()``).
        temperature: Sampling temperature.
        mode: Note mode (budget).
        albert_api_key: The caller's Albert key.
        role: Albert role (``notes``, ``book_structure``…).
        response_format: Structured output format, or ``None``.
        albert_usage_ledger: Usage ledger of the request, given to the client
            (every send, retries and chain fallbacks included, is recorded
            into it), or ``None``.

    Returns:
        The generated content.

    Raises:
        AlbertAuthError: Account error, including a missing key.
        AlbertTruncatedError: Empty truncated answer twice.
        AlbertError: Other classified errors, after the retries.
        ValueError: Empty answer.
    """
    request = albert_chat_request(
        prompt, wire_model, mode=mode, temperature=temperature, role=role, response_format=response_format
    )
    client = _get_albert_client(albert_api_key, use_limiter=False, ledger=albert_usage_ledger)
    logger.info(f"Using Albert with model: {request.model or '(role ' + request.role + ')'}")
    try:
        try:
            result = await _albert_slot_chat(
                client, request, semaphore, max_tokens=request.max_tokens, reasoning_effort=None
            )
        except AlbertTruncatedError:
            budget = _albert_retry_budget(request.max_tokens)
            logger.warning(
                f"Albert returned an empty truncated answer (finish_reason=length); "
                f"single retry with max_tokens={budget} and reasoning effort '{ALBERT_TRUNCATION_RETRY_EFFORT}'"
            )
            result = await _albert_slot_chat(
                client, request, semaphore, max_tokens=budget, reasoning_effort=ALBERT_TRUNCATION_RETRY_EFFORT
            )
    finally:
        client.close()
    return _albert_answer_text(result)


def _detect_language(metadata: Dict) -> str:
    """
    Detect the target language for the note based on metadata.

    Args:
        metadata: Dictionary with item metadata (should contain 'language' field)

    Returns:
        Language code (e.g., "fr", "en", "es")
    """
    # Check if language is explicitly specified in metadata
    lang = metadata.get("language", "").lower()

    if lang:
        # Extract language code (e.g., "en-US" -> "en")
        lang_code = lang.split("-")[0].split("_")[0]
        if lang_code in ("fr", "en", "es", "de", "it", "pt"):
            return lang_code

    # Default to French
    return "fr"


def _load_prompt_template(mode: str = "extended") -> str:
    """
    Load the prompt template based on the specified mode.

    Args:
        mode: Note generation mode. One of:
              - "extended": Full analysis template (zotero_prompt.md)
              - "short": Quick summary template (zotero_prompt_short.md)
              - "pedagogique": Pedagogical template for L3 students (zotero_prompt_pedagogique.md)
              - "evaluation": Peer review evaluation grid (zotero_prompt_evaluation.md)

    Returns:
        Prompt template string with placeholders

    Raises:
        FileNotFoundError: If the prompt file is not found
        ValueError: If the mode is not recognized
    """
    # Validate mode
    if mode not in TEMPLATE_MAP:
        logger.warning(f"Unknown mode '{mode}', falling back to 'extended'")
        mode = "extended"

    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Choose template based on mode
    template_filename = TEMPLATE_MAP[mode]
    prompt_file = os.path.join(current_dir, template_filename)

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
        logger.info(f"Loaded prompt template from {prompt_file} (mode: {mode})")
        return template
    except FileNotFoundError:
        logger.error(f"Prompt template not found at {prompt_file}")
        raise


def _add_note_prefix(html_content: str, mode: str) -> str:
    """
    Add a mode prefix to the first h2 heading for Zotero note identification.

    This allows users to quickly identify the type of note in Zotero:
    - [FICHE] for extended analysis
    - [CLAIR] for pedagogical notes
    - [EVAL] for evaluation grids
    - (no prefix for short summaries)

    Args:
        html_content: The HTML content of the generated note
        mode: The note generation mode

    Returns:
        HTML content with the prefix injected in the first h2 heading

    Example:
        >>> html = "<h2>Smith (2024). Title...</h2><p>Content...</p>"
        >>> _add_note_prefix(html, "pedagogique")
        '<h2>[CLAIR] Smith (2024). Title...</h2><p>Content...</p>'
    """
    import re

    prefix = NOTE_MODE_PREFIX.get(mode, "")
    if not prefix:
        return html_content

    # Find the first <h2> and inject the prefix
    pattern = r'(<h2>)(.*?)(</h2>)'
    replacement = rf'\1{prefix} \2\3'
    return re.sub(pattern, replacement, html_content, count=1)


def _build_prompt(metadata: Dict, text_content: str, language: str, mode: str = "extended") -> str:
    """
    Build the LLM prompt by loading template and replacing placeholders.

    Args:
        metadata: Dictionary with item metadata
        text_content: Full text content (texteocr)
        language: Target language code
        mode: Note generation mode. One of:
              - "extended": Full analysis (no text limit)
              - "short": Quick summary (8000 char limit)
              - "pedagogique": Pedagogical note for L3 students (no text limit)
              - "evaluation": Peer review evaluation grid (no text limit)

    Returns:
        Formatted prompt string
    """
    # Extract key metadata and convert to strings (handle pandas NaN/float values)
    def safe_str(value, default="N/A"):
        """Convert value to string, handling NaN and None."""
        if value is None or value == "":
            return default
        # Check for pandas NaN (float type)
        if isinstance(value, float):
            import math
            if math.isnan(value):
                return default
        return str(value)

    title = safe_str(metadata.get("title"), "Sans titre")
    authors = safe_str(metadata.get("authors"), "N/A")
    date = safe_str(metadata.get("date"), "N/A")
    abstract = safe_str(metadata.get("abstract"), "")
    doi = safe_str(metadata.get("doi"), "")
    url = safe_str(metadata.get("url"), "")
    problematique = safe_str(metadata.get("problematique"), "Non spécifiée")

    # Language-specific names
    lang_instructions = {
        "fr": "français",
        "en": "English",
        "es": "español",
        "de": "Deutsch",
        "it": "italiano",
        "pt": "português"
    }
    target_lang = lang_instructions.get(language, "français")

    # Limit text based on mode (only 'short' mode has text limit)
    if mode == "short":
        # Limit to 8000 characters for quick summary
        text_limited = safe_str(text_content[:8000] if text_content else None, "Non disponible")
    else:
        # Use full text for all other modes (extended, pedagogique, evaluation)
        text_limited = safe_str(text_content if text_content else None, "Non disponible")

    abstract_text = abstract if abstract else "Non disponible"

    try:
        # Load template from file based on mode
        template = _load_prompt_template(mode=mode)

        # Replace placeholders
        prompt = template.replace("{TITLE}", title)
        prompt = prompt.replace("{AUTHORS}", authors)
        prompt = prompt.replace("{DATE}", date)
        prompt = prompt.replace("{DOI}", doi)
        prompt = prompt.replace("{URL}", url)
        prompt = prompt.replace("{PROBLEMATIQUE}", problematique)
        prompt = prompt.replace("{ABSTRACT}", abstract_text)
        prompt = prompt.replace("{TEXT}", text_limited)
        prompt = prompt.replace("{LANGUAGE}", target_lang)

        logger.debug(f"Built prompt from template for: {title} (mode: {mode})")
        return prompt

    except FileNotFoundError:
        # Fallback to hardcoded prompt if file not found
        logger.warning("Prompt template file not found, using fallback hardcoded prompt")
        prompt = f"""Tu es un assistant spécialisé dans la rédaction de fiches de lecture académiques.

CONTEXTE :
Titre : {title}
Auteurs : {authors}
Date : {date}
DOI : {doi}
URL : {url}
Problématique de recherche : {problematique}

Résumé (si disponible) :
{abstract_text}

TEXTE COMPLET :
{text_limited}

CONSIGNE :
Rédige une fiche de lecture structurée en {target_lang}, au format HTML simplifié (balises : <p>, <strong>, <em>, <ul>, <li>).

STRUCTURE REQUISE :
1. **Référence bibliographique** : Titre, auteurs, date, lien si disponible
2. **Problématique** : Question(s) de recherche ou objectif principal
3. **Méthodologie** : Approche, données, méthodes utilisées
4. **Résultats clés** : Principales conclusions ou découvertes
5. **Limites et perspectives** : Points faibles, questions ouvertes

CONTRAINTES :
- Ton : Neutre, informatif, académique
- Format : HTML propre (pas de <html>, <head>, <body>)
- Concentre-toi sur les points essentiels

Commence directement par le contenu HTML, sans préambule."""

        return prompt


def _generate_with_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    mode: str = "extended",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> str:
    """
    Generate note content using LLM.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (e.g., "gpt-4o-mini", "google/gemini-2.5-flash" or
               "albert/gpt-oss-120b"). If None, uses the default model
               (resolve_default_llm_model).
        temperature: Sampling temperature (0.0 to 1.0)
        mode: Note generation mode. Determines max_tokens:
              - "extended": 16000 tokens
              - "short": 2000 tokens
              - "pedagogique": 10000 tokens
              - "evaluation": 10000 tokens
        openai_api_key: Optional OpenAI API key. If None, no OpenAI client
                        (no environment fallback).
        openrouter_api_key: Optional OpenRouter API key. If None, no OpenRouter
                            client (no environment fallback).
        albert_api_key: Optional Albert key, used only for an ``albert/…``
                        model (never read from the environment).
        albert_usage_ledger: Optional usage ledger of the request
                        (``UsageLedger``): on the Albert path every Albert
                        call is recorded into it; ignored on the
                        OpenAI/OpenRouter path.

    Returns:
        Generated HTML content

    Raises:
        ValueError: If no LLM client is available
        AlbertDisabledError: ``albert/…`` model while Albert is disabled
        AlbertAuthError: Albert account error (no fallback to another provider)
        Exception: If the API call fails
    """
    # Single resolver, before the provider branch: an explicit albert/ model
    # goes to Albert, outside the historical retry loop below.
    resolution = resolve_llm_route(model) if model else None
    if resolution is not None and resolution.provider == PROVIDER_ALBERT:
        return _generate_with_albert(
            prompt, resolution.wire_model, temperature=temperature, mode=mode, albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger,
        )

    # Get clients from the credentials passed by the caller (no env fallback)
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")
        resolution = resolve_llm_route(model)
        if resolution.provider == PROVIDER_ALBERT:
            return _generate_with_albert(
                prompt, resolution.wire_model, temperature=temperature, mode=mode, albert_api_key=albert_api_key,
                albert_usage_ledger=albert_usage_ledger,
            )

    # Detect which client to use (OpenRouter models have format "provider/model")
    use_openrouter = resolution.provider == PROVIDER_OPENROUTER

    if use_openrouter:
        if not openrouter_client:
            logger.warning(f"OpenRouter model '{model}' requested but client not initialized. Falling back to OpenAI.")
            if not openai_client:
                raise ValueError("No LLM client available (neither OpenAI nor OpenRouter). Check your API keys in Settings.")
            active_client = openai_client
            model = "gpt-4o-mini"
        else:
            active_client = openrouter_client
            logger.info(f"Using OpenRouter with model: {model}")
    else:
        if not openai_client:
            raise ValueError("OpenAI client not initialized (OPENAI_API_KEY missing). Check your API keys in Settings.")
        active_client = openai_client
        logger.info(f"Using OpenAI with model: {model}")

    # Set max_tokens based on mode
    max_tokens = NOTE_MODE_MAX_TOKENS.get(mode, 16000)

    # Retry configuration: 1 retry with 2 second delay
    max_attempts = 2
    retry_delay = 2  # seconds
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"LLM API call attempt {attempt}/{max_attempts} for model: {model}")

            response = active_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant spécialisé en rédaction de fiches de lecture académiques."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate response structure
            if not response or not response.choices:
                logger.error(f"LLM API returned empty response or no choices. Response: {response}")
                raise ValueError(f"LLM API returned invalid response (no choices). Model: {model}")

            if not response.choices[0].message or response.choices[0].message.content is None:
                logger.error(f"LLM API returned empty message content. Response: {response}")
                raise ValueError(f"LLM API returned empty content. Model: {model}")

            content = response.choices[0].message.content.strip()

            if not content:
                logger.warning(f"LLM returned empty string for model {model}")
                raise ValueError(f"LLM returned empty response. Model: {model}")

            logger.debug(f"Generated note content (length: {len(content)} chars)")
            return content

        except Exception as e:
            last_error = e
            logger.error(f"LLM API error (attempt {attempt}/{max_attempts}): {e}")

            if attempt < max_attempts:
                logger.info(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_attempts} attempts failed for model {model}")

    # If we get here, all retries failed
    if last_error:
        raise last_error
    raise RuntimeError(f"LLM API failed after {max_attempts} attempts")


def _fallback_template(metadata: Dict, language: str) -> str:
    """
    Generate a simple HTML template if LLM is unavailable.

    Args:
        metadata: Dictionary with item metadata
        language: Target language code

    Returns:
        HTML template string
    """
    # Helper to safely convert values to strings
    def safe_str(value, default="N/A"):
        """Convert value to string, handling NaN and None."""
        if value is None or value == "":
            return default
        # Check for pandas NaN (float type)
        if isinstance(value, float):
            import math
            if math.isnan(value):
                return default
        return str(value)

    title = html_module.escape(safe_str(metadata.get("title"), "Sans titre"))
    authors = html_module.escape(safe_str(metadata.get("authors"), "N/A"))
    date = html_module.escape(safe_str(metadata.get("date"), "N/A")[:10])
    abstract_raw = safe_str(metadata.get("abstract"), "")
    abstract = html_module.escape(abstract_raw[:1200] if abstract_raw else "")
    url = html_module.escape(safe_str(metadata.get("url") or metadata.get("doi"), ""))

    # Language-specific labels
    labels = {
        "fr": {
            "title": "Fiche de lecture",
            "ref": "Référence",
            "problem": "Problématique",
            "method": "Méthodologie",
            "results": "Résultats clés",
            "limits": "Limites",
            "abstract": "Résumé",
            "tbd": "à compléter"
        },
        "en": {
            "title": "Reading Note",
            "ref": "Reference",
            "problem": "Research Question",
            "method": "Methodology",
            "results": "Key Results",
            "limits": "Limitations",
            "abstract": "Abstract",
            "tbd": "to be completed"
        }
    }

    lang_labels = labels.get(language, labels["fr"])

    return f"""<p><em>Fiche générée automatiquement (template).</em></p>
<h3>{lang_labels["title"]}</h3>
<p><strong>{lang_labels["ref"]} :</strong> {title} — {authors} — {date} — {url}</p>
<ul>
  <li><strong>{lang_labels["problem"]} :</strong> {lang_labels["tbd"]}</li>
  <li><strong>{lang_labels["method"]} :</strong> {lang_labels["tbd"]}</li>
  <li><strong>{lang_labels["results"]} :</strong> {lang_labels["tbd"]}</li>
  <li><strong>{lang_labels["limits"]} :</strong> {lang_labels["tbd"]}</li>
</ul>
<p><strong>{lang_labels["abstract"]} :</strong> {abstract}</p>"""


def build_note_html(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    use_llm: bool = True,
    mode: str = "extended",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> Tuple[str, str]:
    """
    Build a reading note in HTML format with a unique sentinel.

    This is the main entry point for generating reading notes.

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses the default model (resolve_default_llm_model).
               Examples: "gpt-4o-mini", "google/gemini-2.5-flash", "albert/gpt-oss-120b"
        use_llm: Whether to use LLM or fallback to template (default: True)
        mode: Note generation mode. One of:
              - "extended": Full analysis [FICHE] (2500-3000 words)
              - "short": Quick summary (400-600 words, no prefix)
              - "pedagogique": Pedagogical note [CLAIR] for L3 students (2400-2800 words)
              - "evaluation": Peer review evaluation grid [EVAL] (2200-2750 words)
        openai_api_key: Optional OpenAI API key for secure credential passing.
                        If None, no OpenAI client (no environment fallback).
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
                           If None, no OpenRouter client (no environment fallback).
        albert_api_key: Optional Albert key, used only for an ``albert/…`` model.
                        Albert account errors are raised (never replaced by the
                        template); other Albert failures fall back to the template.
        albert_usage_ledger: Optional usage ledger of the request
                        (``UsageLedger``), filled by the Albert calls only;
                        ``None`` or an OpenAI/OpenRouter model changes nothing.

    Returns:
        Tuple of (sentinel, note_html):
        - sentinel: Unique identifier (e.g., "ragpy-note-id:uuid")
        - note_html: Complete HTML with sentinel comment and mode prefix

    Example:
        >>> metadata = {
        ...     "title": "Machine Learning for NLP",
        ...     "authors": "Smith, J.",
        ...     "date": "2024",
        ...     "abstract": "This paper presents...",
        ...     "language": "en"
        ... }
        >>> sentinel, html = build_note_html(metadata, text_content="Full text...", mode="pedagogique")
        >>> print(sentinel)
        ragpy-note-id:abc123...
        >>> "[CLAIR]" in html
        True
    """
    # Validate mode
    if mode not in TEMPLATE_MAP:
        logger.warning(f"Unknown mode '{mode}', falling back to 'extended'")
        mode = "extended"

    # Get clients from the credentials passed by the caller (no env fallback)
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")

    # Detect target language
    language = _detect_language(metadata)
    logger.info(f"Generating note in language: {language} (mode: {mode})")

    # Albert is available as soon as the model selects it (a missing key is an
    # account error raised below, never a silent template)
    albert_selected = use_llm and resolve_llm_route(model).provider == PROVIDER_ALBERT

    # Generate the note body
    if use_llm and (openai_client or openrouter_client or albert_selected):
        try:
            # Use text_content if available, otherwise use abstract
            content = text_content or metadata.get("abstract", "")

            if not content:
                logger.warning("No text content or abstract available, using template fallback")
                body_html = _fallback_template(metadata, language)
            else:
                # Build prompt and generate with LLM
                prompt = _build_prompt(metadata, content, language, mode=mode)
                body_html = _generate_with_llm(
                    prompt,
                    model=model,
                    mode=mode,
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key,
                    albert_api_key=albert_api_key,
                    albert_usage_ledger=albert_usage_ledger
                )
                # Add mode prefix to first h2 heading
                body_html = _add_note_prefix(body_html, mode)
        except ALBERT_JOB_ABORT_ERRORS:
            raise
        except Exception as e:
            logger.error(f"LLM generation failed, using template fallback: {e}")
            body_html = _fallback_template(metadata, language)
    else:
        logger.info("LLM not available or disabled, using template")
        body_html = _fallback_template(metadata, language)

    # Generate unique sentinel
    sentinel = f"{SENTINEL_PREFIX}{uuid.uuid4()}"

    # Build complete HTML with sentinel comment
    note_html = f"<!-- {sentinel} -->\n{body_html}"

    logger.info(f"Generated note with sentinel: {sentinel}")
    return sentinel, note_html


def build_abstract_text(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> str:
    """
    Build an abstract/summary text to enrich Zotero's abstractNote field.

    This function generates a plain text summary (not HTML) that can be
    appended to the existing abstract in Zotero.

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses the default model (resolve_default_llm_model).
        openai_api_key: Optional OpenAI API key for secure credential passing.
                        If None, no OpenAI client (no environment fallback).
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
                           If None, no OpenRouter client (no environment fallback).
        albert_api_key: Optional Albert key, used only for an ``albert/…`` model.
        albert_usage_ledger: Optional usage ledger of the request
                        (``UsageLedger``), filled by the Albert calls only;
                        ``None`` or an OpenAI/OpenRouter model changes nothing.

    Returns:
        Plain text summary string (200-350 words)

    Example:
        >>> metadata = {
        ...     "title": "Machine Learning for NLP",
        ...     "authors": "Smith, J.",
        ...     "date": "2024",
        ...     "abstract": "This paper presents...",
        ...     "language": "en"
        ... }
        >>> summary = build_abstract_text(metadata, text_content="Full text...")
        >>> print(summary)
        This study investigates...
    """
    # Get clients from the credentials passed by the caller (no env fallback)
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")

    # Detect target language
    language = _detect_language(metadata)
    logger.info(f"Generating abstract summary in language: {language}")

    # Albert is available as soon as the model selects it
    albert_selected = resolve_llm_route(model).provider == PROVIDER_ALBERT

    # Check if LLM is available
    if not (openai_client or openrouter_client or albert_selected):
        logger.error("No LLM client available for abstract generation")
        raise ValueError("No LLM client available (neither OpenAI nor OpenRouter). Check your API keys in Settings.")

    # Use text_content if available, otherwise use abstract
    content = text_content or metadata.get("abstract", "")

    if not content:
        logger.warning("No text content or abstract available for summary generation")
        raise ValueError("No text content available to generate summary")

    try:
        # Build prompt using the short template (mode="short")
        prompt = _build_prompt(metadata, content, language, mode="short")

        # Generate with LLM (use smaller max_tokens for plain text summary)
        summary = _generate_with_llm(
            prompt,
            model=model,
            mode="short",
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger
        )

        # Clean up the response - remove any HTML tags that might have slipped through
        import re
        summary = re.sub(r'<[^>]+>', '', summary)
        summary = summary.strip()

        logger.info(f"Generated abstract summary (length: {len(summary)} chars)")
        return summary

    except Exception as e:
        logger.error(f"Abstract generation failed: {e}")
        raise


def sentinel_in_html(html_text: str) -> bool:
    """
    Check if a sentinel is present in HTML text.

    Args:
        html_text: HTML text to check

    Returns:
        True if a ragpy sentinel is found, False otherwise
    """
    return SENTINEL_PREFIX in (html_text or "")


def extract_sentinel_from_html(html_text: str) -> Optional[str]:
    """
    Extract the sentinel ID from HTML text.

    Args:
        html_text: HTML text containing a sentinel

    Returns:
        The sentinel string if found, None otherwise
    """
    if not html_text:
        return None

    # Look for the sentinel pattern in HTML comments
    import re
    pattern = rf"<!--\s*({SENTINEL_PREFIX}[a-f0-9\-]+)\s*-->"
    match = re.search(pattern, html_text)

    if match:
        return match.group(1)

    return None


# =============================================================================
# Async Wrapper Functions (with global concurrency control)
# =============================================================================

async def build_note_html_async(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    use_llm: bool = True,
    mode: str = "extended",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> Tuple[str, str]:
    """
    Async version of build_note_html with global concurrency control.

    Uses a semaphore to limit concurrent LLM calls across all users.
    See build_note_html for full documentation. With an ``albert/…`` model, the
    semaphore is held during each send only (``run_llm_slot``).

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses the default model (resolve_default_llm_model).
        use_llm: Whether to use LLM or fallback to template (default: True)
        mode: Note generation mode. One of:
              - "extended": Full analysis [FICHE]
              - "short": Quick summary (no prefix)
              - "pedagogique": Pedagogical note [CLAIR]
              - "evaluation": Peer review evaluation grid [EVAL]
        openai_api_key: Optional OpenAI API key for secure credential passing.
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
        albert_api_key: Optional Albert key, used only for an ``albert/…`` model.
        albert_usage_ledger: Optional usage ledger of the request
                        (``UsageLedger``), handed to the Albert branch only;
                        the executor path below is the OpenAI/OpenRouter one
                        (``albert_route_async`` already routed any Albert
                        model), so it never receives the ledger.
    """
    semaphore = get_llm_semaphore()

    albert = None
    if use_llm:
        # Default model resolved off the loop, and only while Albert is enabled
        albert, model = await albert_route_async(model)
    if albert is not None:
        return await _build_note_html_albert_async(
            metadata, text_content, albert, mode=mode, albert_api_key=albert_api_key, semaphore=semaphore,
            albert_usage_ledger=albert_usage_ledger,
        )

    async with semaphore:
        remaining = semaphore._value
        logger.debug(f"Acquired LLM slot ({remaining} slots remaining)")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: build_note_html(
                    metadata=metadata,
                    text_content=text_content,
                    model=model,
                    use_llm=use_llm,
                    mode=mode,
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key,
                    albert_api_key=albert_api_key
                )
            )
            return result
        finally:
            logger.debug("Released LLM slot")


async def build_abstract_text_async(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> str:
    """
    Async version of build_abstract_text with global concurrency control.

    Uses a semaphore to limit concurrent LLM calls across all users.
    See build_abstract_text for full documentation. With an ``albert/…``
    model, the semaphore is held during each send only (``run_llm_slot``).

    Args:
        openai_api_key: Optional OpenAI API key for secure credential passing.
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
        albert_api_key: Optional Albert key, used only for an ``albert/…`` model.
        albert_usage_ledger: Optional usage ledger of the request
                        (``UsageLedger``), handed to the Albert branch only
                        (the executor path is the OpenAI/OpenRouter one).
    """
    semaphore = get_llm_semaphore()

    # Default model resolved off the loop, and only while Albert is enabled
    albert, model = await albert_route_async(model)
    if albert is not None:
        return await _build_abstract_text_albert_async(
            metadata, text_content, albert, albert_api_key=albert_api_key, semaphore=semaphore,
            albert_usage_ledger=albert_usage_ledger,
        )

    async with semaphore:
        remaining = semaphore._value
        logger.debug(f"Acquired LLM slot ({remaining} slots remaining)")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: build_abstract_text(
                    metadata=metadata,
                    text_content=text_content,
                    model=model,
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key,
                    albert_api_key=albert_api_key
                )
            )
            return result
        finally:
            logger.debug("Released LLM slot")


async def _build_note_html_albert_async(
    metadata: Dict,
    text_content: Optional[str],
    albert: ProviderResolution,
    *,
    mode: str,
    albert_api_key: Optional[str],
    semaphore: Any,
    albert_usage_ledger: Optional["UsageLedger"] = None,
) -> Tuple[str, str]:
    """
    Albert branch of ``build_note_html_async`` (same contract as ``build_note_html``).

    The LLM call goes through ``agenerate_with_albert`` (``run_llm_slot``: the
    global and Albert semaphores are held during each send only). Account
    errors are raised; any other failure falls back to the template, as on the
    OpenAI/OpenRouter path.

    Args:
        metadata: Item metadata.
        text_content: Full text, or ``None`` (abstract used instead).
        albert: Albert resolution of the model.
        mode: Note generation mode.
        albert_api_key: The caller's Albert key.
        semaphore: The global LLM semaphore.
        albert_usage_ledger: Usage ledger of the request, or ``None``.

    Returns:
        ``(sentinel, note_html)``.

    Raises:
        AlbertAuthError: Account error (invalid or missing key, expired
            account, budget or quota exhausted).
    """
    if mode not in TEMPLATE_MAP:
        logger.warning(f"Unknown mode '{mode}', falling back to 'extended'")
        mode = "extended"

    language = _detect_language(metadata)
    logger.info(f"Generating note in language: {language} (mode: {mode})")

    try:
        content = text_content or metadata.get("abstract", "")
        if not content:
            logger.warning("No text content or abstract available, using template fallback")
            body_html = _fallback_template(metadata, language)
        else:
            prompt = _build_prompt(metadata, content, language, mode=mode)
            body_html = await agenerate_with_albert(
                prompt,
                albert.wire_model,
                semaphore=semaphore,
                temperature=0.2,
                mode=mode,
                albert_api_key=albert_api_key,
                albert_usage_ledger=albert_usage_ledger,
            )
            body_html = _add_note_prefix(body_html, mode)
    except ALBERT_JOB_ABORT_ERRORS:
        raise
    except Exception as e:
        logger.error(f"LLM generation failed, using template fallback: {e}")
        body_html = _fallback_template(metadata, language)

    sentinel = f"{SENTINEL_PREFIX}{uuid.uuid4()}"
    note_html = f"<!-- {sentinel} -->\n{body_html}"
    logger.info(f"Generated note with sentinel: {sentinel}")
    return sentinel, note_html


async def _build_abstract_text_albert_async(
    metadata: Dict,
    text_content: Optional[str],
    albert: ProviderResolution,
    *,
    albert_api_key: Optional[str],
    semaphore: Any,
    albert_usage_ledger: Optional["UsageLedger"] = None,
) -> str:
    """
    Albert branch of ``build_abstract_text_async`` (same contract as ``build_abstract_text``).

    Args:
        metadata: Item metadata.
        text_content: Full text, or ``None`` (abstract used instead).
        albert: Albert resolution of the model.
        albert_api_key: The caller's Albert key.
        semaphore: The global LLM semaphore.
        albert_usage_ledger: Usage ledger of the request, or ``None``.

    Returns:
        The plain text summary.

    Raises:
        ValueError: No text content available.
        AlbertAuthError: Account error.
        Exception: Any other generation failure (re-raised, as on the
            OpenAI/OpenRouter path).
    """
    import re

    language = _detect_language(metadata)
    logger.info(f"Generating abstract summary in language: {language}")

    content = text_content or metadata.get("abstract", "")
    if not content:
        logger.warning("No text content or abstract available for summary generation")
        raise ValueError("No text content available to generate summary")

    try:
        prompt = _build_prompt(metadata, content, language, mode="short")
        summary = await agenerate_with_albert(
            prompt,
            albert.wire_model,
            semaphore=semaphore,
            temperature=0.2,
            mode="short",
            albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger,
        )
        summary = re.sub(r'<[^>]+>', '', summary).strip()
        logger.info(f"Generated abstract summary (length: {len(summary)} chars)")
        return summary
    except Exception as e:
        logger.error(f"Abstract generation failed: {e}")
        raise
