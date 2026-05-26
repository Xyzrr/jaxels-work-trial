"""Stack-specific defaults for the OpenHands/SWE-bench pod launcher.

The API key here is not a real credential for local vLLM. It is a compatibility
string that satisfies OpenAI-compatible client code while requests stay inside
the GPU pod. Keeping the selection in a small helper makes the stack-specific
SWE-Lego behavior visible without spreading conditionals through the launcher.
"""

from __future__ import annotations


def select_openhands_llm_api_key(
    eval_stack: str,
    llm_api_key_explicit: bool,
    current_key: str,
) -> str:
    """Return the API-key value to pass into the selected eval stack.

    Current upstream OpenHands presets use the repository default ``local-llm``
    placeholder for the pod-local vLLM endpoint. The vendored SWE-Lego stack
    expects a different dummy value when no user-provided key is present. That
    is an eval-stack reproduction detail, not a secret-management policy.
    """

    if eval_stack == "swe-lego" and not llm_api_key_explicit:
        # Preserve SWE-Lego's vendored local-serving contract. If the caller
        # explicitly sets LLM_API_KEY, respect it; otherwise use the dummy value
        # the SWE-Lego path expects for an OpenAI-compatible local server.
        return "dummy-key"

    # For all other stacks, and for explicit caller choices, keep the key the
    # launcher already resolved from the environment/defaults.
    return current_key
