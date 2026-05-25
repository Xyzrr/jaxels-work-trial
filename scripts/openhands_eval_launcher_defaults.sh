#!/usr/bin/env bash

select_openhands_llm_api_key() {
  local eval_stack="$1"
  local llm_api_key_explicit="$2"
  local current_key="$3"

  if [[ "$eval_stack" == "swe-lego" && "$llm_api_key_explicit" != "1" ]]; then
    printf "%s\n" "dummy-key"
  else
    printf "%s\n" "$current_key"
  fi
}
