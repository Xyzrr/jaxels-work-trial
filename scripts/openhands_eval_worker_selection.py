"""Choose OpenHands eval worker concurrency for the selected eval stack.

This helper is intentionally small because worker count is an experiment
contract, not just a performance knob. In SWE-bench-style coding evals, each
OpenHands worker drives one task attempt against the model-serving endpoint.
Changing the worker count can change load on vLLM, timeout behavior, and the
shape of reproduced eval runs, so stack-specific contracts are kept explicit.
"""

from __future__ import annotations


def select_openhands_eval_num_workers(
    eval_stack: str,
    eval_limit: str | int | None,
    eval_ids: str | None,
    config_num_workers: str | int,
    total_agent_workers: str | int,
) -> int:
    """Return the worker count to pass to the eval stack.

    ``total_agent_workers`` is the pod capacity derived from model-serving
    topology, for example one current OpenHands worker pool per vLLM replica.
    ``config_num_workers`` is the preset's requested eval concurrency. The two
    differ because some smoke runs intentionally evaluate only a handful of
    instances while full pass@1 runs should use the preset's full concurrency.
    """

    config_num_workers_int = int(config_num_workers)
    total_agent_workers_int = int(total_agent_workers)
    eval_limit_raw = "" if eval_limit is None else str(eval_limit)
    eval_ids_raw = "" if eval_ids is None else eval_ids

    if eval_stack == "swe-lego" and config_num_workers_int > 0:
        # SWE-Lego's vendored OpenHands/SWE-bench stack is a reproduction path.
        # Its preset explicitly sets NUM_WORKERS=24, and the workflow docs treat
        # that as part of the reproduction contract. Even when a smoke provides
        # a short --eval-ids list, keep the configured worker count so the
        # runner environment matches the preset being reproduced.
        return config_num_workers_int

    if eval_limit_raw and 0 < int(eval_limit_raw) < total_agent_workers_int:
        # Upstream OpenHands smoke runs can be smaller than the pod's full vLLM
        # serving capacity. Clamping avoids launching many idle task workers for
        # a one-instance or few-instance infrastructure check while preserving
        # full capacity when eval_limit is empty or larger than the worker pool.
        return int(eval_limit_raw)

    if eval_ids_raw:
        eval_id_count = len([item for item in eval_ids_raw.split(",") if item.strip()])
        if 0 < eval_id_count < total_agent_workers_int:
            # A comma-separated eval id list is another bounded smoke path. For
            # upstream OpenHands, matching workers to selected instances keeps
            # the smoke deterministic and reduces unnecessary model-serving
            # pressure without changing which SWE-bench tasks are evaluated.
            return eval_id_count

    if config_num_workers_int > 0:
        # For full or preset-sized runs, honor the preset. That keeps eval
        # concurrency with the model-serving topology chosen in configs/eval/.
        return config_num_workers_int

    # Last resort for older presets: use the capacity inferred by the pod
    # launcher from vLLM replica count and per-replica task budget.
    return total_agent_workers_int
