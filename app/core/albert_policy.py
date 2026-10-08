"""
Albert Inference Policy (web layer)
===================================

Applies the server policy ``ALBERT_DATA_POLICY`` (``compatible`` by default,
or ``albert_only``) at the boundaries of the web application: chat models of
the chunking, notes and citation routes, dense embedding provider, OCR
launches and the environment of the pipeline subprocesses
(``scripts/rad_albert/policy.py`` holds the rules).

While the policy is ``compatible`` (default) every function returns its input
unchanged and nothing else is read or imported: the historical behaviour is
identical. With ``albert_only``:

* an empty chat model becomes the Albert default of the role, any model
  without the ``albert/`` prefix is refused (400 ``policy_violation``);
* ``embedding_provider=openai`` is refused, an empty value becomes ``albert``;
* the OCR routes only need the Albert key (or none: local OCR), never Mistral;
* the subprocess environment loses ``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``
  and ``MISTRAL_API_KEY``, which are added to ``RAGPY_DOTENV_DENY`` so that no
  script can reload them from ``.env``;
* with Albert disabled, every launch is refused (400
  ``policy_requires_albert``): no silent return to the historical providers.

The policy is a server setting: no request field can relax it.
"""
import os
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse

POLICY_ENV_VAR = "ALBERT_DATA_POLICY"
EXTERNAL_INFERENCE_KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY")
"""Credential variables of the inference providers blocked by ``albert_only``."""
DOTENV_DENY_ENV_VAR = "RAGPY_DOTENV_DENY"


def policy_config() -> Any:
    """
    Policy-relevant server configuration, read without the base URL.

    Only ``ALBERT_ENABLED`` and ``ALBERT_DATA_POLICY`` are read, so a refused
    ``ALBERT_BASE_URL`` never breaks a policy check.

    Returns:
        An ``AlbertConfig``.
    """
    try:
        from scripts.rad_albert.config import AlbertConfig
    except ImportError:  # scripts/ itself on sys.path
        from rad_albert.config import AlbertConfig
    names = ("ALBERT_ENABLED", POLICY_ENV_VAR)
    return AlbertConfig.from_env({name: os.environ[name] for name in names if name in os.environ})


def strict_policy() -> bool:
    """
    Tell whether the server policy is ``albert_only`` (cheap: one variable read).

    An unknown non-empty value counts as ``albert_only`` (fail closed, see
    ``rad_albert.config.normalise_data_policy``).

    Returns:
        True for ``ALBERT_DATA_POLICY=albert_only`` (or an unknown value).
    """
    raw = os.environ.get(POLICY_ENV_VAR)
    if raw is None or not raw.strip():
        return False
    try:
        from scripts.rad_albert.config import normalise_data_policy
    except ImportError:  # scripts/ itself on sys.path
        from rad_albert.config import normalise_data_policy
    return normalise_data_policy(raw) == "albert_only"


def _policy_module() -> Any:
    """``rad_albert.policy`` (imported lazily, both package spellings)."""
    try:
        from scripts.rad_albert import policy
    except ImportError:
        from rad_albert import policy
    return policy


def check_configuration() -> None:
    """
    Refuse every launch while ``albert_only`` is set and Albert is disabled.

    Raises:
        AlbertPolicyError: ``policy_requires_albert``.
    """
    if strict_policy():
        _policy_module().check_configuration(policy_config())


def apply_chat_policy(model: Optional[str], role: str = "recode") -> Optional[str]:
    """
    Chat model allowed by the policy (input unchanged while ``compatible``).

    Args:
        model: Model entered by the user (None or empty: default).
        role: Chat role (``recode``, ``citation``, ``notes``, ``book_structure``, ``answer``).

    Returns:
        The model to use (``albert/<id>`` default for an empty model with ``albert_only``).

    Raises:
        AlbertPolicyError: Model of another provider with ``albert_only``, or Albert disabled.
    """
    if not strict_policy():
        return model
    return _policy_module().check_chat_model(model, policy_config(), role=role)


def require_albert_model(model: Optional[str], role: str = "citation") -> None:
    """
    Refuse, with ``albert_only``, any model that is not ``albert/<id>`` (empty included).

    Used where the model entered is stored and reused as is (citation
    configurations), so a default cannot be substituted silently.

    Args:
        model: Model of the configuration.
        role: Chat role (suggested default in the message).

    Raises:
        AlbertPolicyError: ``albert_only`` and a non-Albert or empty model, or Albert disabled.
    """
    if not strict_policy():
        return
    policy = _policy_module()
    cfg = policy_config()
    policy.check_configuration(cfg)
    if not policy.is_albert_model(model):
        raise policy.AlbertPolicyError(
            f"Politique albert_only : choisir un modèle « albert/… » (par exemple "
            f"{policy.default_albert_model(role, cfg)}) ; « {model or 'vide'} » est refusé.",
            capability="chat",
        )


def apply_embedding_policy(provider: Optional[str]) -> Optional[str]:
    """
    Dense embedding provider allowed by the policy (input unchanged while ``compatible``).

    Args:
        provider: ``openai``, ``albert``, empty or None.

    Returns:
        The provider to use.

    Raises:
        AlbertPolicyError: ``openai`` with ``albert_only``, or Albert disabled.
    """
    if not strict_policy():
        return provider
    return _policy_module().check_embedding_provider(provider, policy_config())


def restrict_subprocess_env(env: Dict[str, str]) -> Dict[str, str]:
    """
    Remove the external inference keys from a subprocess environment (``albert_only`` only).

    The removed names are added to ``RAGPY_DOTENV_DENY`` so that
    ``load_dotenv_guarded`` never reloads them from ``.env`` (admins included).
    While the policy is ``compatible`` the environment is returned unchanged.

    Args:
        env: Environment built by ``build_subprocess_env``.

    Returns:
        The same dict, modified in place.
    """
    if not strict_policy():
        return env
    for name in EXTERNAL_INFERENCE_KEYS:
        env.pop(name, None)
    current = [n for n in (env.get(DOTENV_DENY_ENV_VAR) or "").split(",") if n.strip()]
    denied = sorted(set(current) | set(EXTERNAL_INFERENCE_KEYS))
    env[DOTENV_DENY_ENV_VAR] = ",".join(denied)
    return env


def policy_error_body(exc: BaseException) -> Dict[str, Any]:
    """
    JSON body of a policy refusal.

    Args:
        exc: The ``AlbertPolicyError``.

    Returns:
        ``{error, reason, policy}``.
    """
    return {"error": str(exc), "reason": getattr(exc, "reason", "policy_violation"), "policy": "albert_only"}


def policy_response(exc: BaseException) -> JSONResponse:
    """
    400 response of a policy refusal.

    Args:
        exc: The ``AlbertPolicyError``.

    Returns:
        The JSON response.
    """
    return JSONResponse(status_code=400, content=policy_error_body(exc))


def is_policy_error(exc: BaseException) -> bool:
    """Tell whether ``exc`` is an ``AlbertPolicyError`` (without importing the module while compatible)."""
    return type(exc).__name__ == "AlbertPolicyError"
