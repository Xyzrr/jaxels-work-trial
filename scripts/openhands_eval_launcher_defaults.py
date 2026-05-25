from __future__ import annotations


def select_openhands_llm_api_key(
    eval_stack: str,
    llm_api_key_explicit: bool,
    current_key: str,
) -> str:
    if eval_stack == "swe-lego" and not llm_api_key_explicit:
        return "dummy-key"
    return current_key
