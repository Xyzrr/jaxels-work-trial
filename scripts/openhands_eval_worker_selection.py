from __future__ import annotations


def select_openhands_eval_num_workers(
    eval_stack: str,
    eval_limit: str | int | None,
    eval_ids: str | None,
    config_num_workers: str | int,
    total_agent_workers: str | int,
) -> int:
    config_num_workers_int = int(config_num_workers)
    total_agent_workers_int = int(total_agent_workers)
    eval_limit_raw = "" if eval_limit is None else str(eval_limit)
    eval_ids_raw = "" if eval_ids is None else eval_ids

    if eval_stack == "swe-lego" and config_num_workers_int > 0:
        return config_num_workers_int

    if eval_limit_raw and 0 < int(eval_limit_raw) < total_agent_workers_int:
        return int(eval_limit_raw)

    if eval_ids_raw:
        eval_id_count = len([item for item in eval_ids_raw.split(",") if item.strip()])
        if 0 < eval_id_count < total_agent_workers_int:
            return eval_id_count

    if config_num_workers_int > 0:
        return config_num_workers_int
    return total_agent_workers_int
