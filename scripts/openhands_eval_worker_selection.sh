#!/usr/bin/env bash

select_openhands_eval_num_workers() {
  local eval_stack="$1"
  local eval_limit="$2"
  local eval_ids="$3"
  local config_num_workers="$4"
  local total_agent_workers="$5"

  if [[ "$eval_stack" == "swe-lego" && "$config_num_workers" -gt 0 ]]; then
    printf "%s\n" "$config_num_workers"
    return 0
  fi

  if [[ -n "$eval_limit" && "$eval_limit" -gt 0 && "$eval_limit" -lt "$total_agent_workers" ]]; then
    printf "%s\n" "$eval_limit"
    return 0
  fi

  if [[ -n "$eval_ids" ]]; then
    local eval_id_count
    eval_id_count="$(tr ',' '\n' <<<"$eval_ids" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
    if [[ "$eval_id_count" -gt 0 && "$eval_id_count" -lt "$total_agent_workers" ]]; then
      printf "%s\n" "$eval_id_count"
      return 0
    fi
  fi

  if [[ "$config_num_workers" -gt 0 ]]; then
    printf "%s\n" "$config_num_workers"
  else
    printf "%s\n" "$total_agent_workers"
  fi
}
