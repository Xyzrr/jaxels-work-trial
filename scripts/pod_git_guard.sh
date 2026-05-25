#!/usr/bin/env bash
# Shared guard for pod launchers that must execute repository code exactly as
# pushed to origin.

swehero_pod_git_error() {
  echo "error: $*" >&2
}

swehero_pod_git_die() {
  swehero_pod_git_error "$*"
  return 1
}

swehero_pod_git_status() {
  local repo_dir="$1"
  git -C "$repo_dir" status --porcelain=v1
}

swehero_require_clean_pod_git_status() {
  local repo_dir="$1"
  local label="$2"
  local status
  status="$(swehero_pod_git_status "$repo_dir")"
  if [[ -n "$status" ]]; then
    swehero_pod_git_error "$label has local git changes; clean it before launching:"
    printf "%s\n" "$status" | sed "s/^/  /" >&2
    return 1
  fi
}

swehero_require_pod_git_checkout() {
  local repo_dir="$1"
  local expected_branch="$2"
  local label="${3:-pod execution directory}"

  if [[ -z "$expected_branch" ]]; then
    swehero_pod_git_die \
      "SWEHERO_POD_GIT_BRANCH is required so the pod can match the current local worktree branch"
    return 1
  fi
  if ! git -C "$repo_dir" check-ref-format --branch "$expected_branch" >/dev/null 2>&1; then
    swehero_pod_git_die "invalid SWEHERO_POD_GIT_BRANCH: $expected_branch"
    return 1
  fi
  if ! git -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    swehero_pod_git_die "$label is not a git worktree: $repo_dir"
    return 1
  fi

  local top_level
  top_level="$(git -C "$repo_dir" rev-parse --show-toplevel)"
  local physical_top_level
  physical_top_level="$(cd "$top_level" && pwd -P)"
  local physical_repo_dir
  physical_repo_dir="$(cd "$repo_dir" && pwd -P)"
  if [[ "$physical_top_level" != "$physical_repo_dir" ]]; then
    swehero_pod_git_die "$label must be the git top-level: $repo_dir"
    return 1
  fi

  swehero_require_clean_pod_git_status "$repo_dir" "$label" || return 1
  git -C "$repo_dir" remote get-url origin >/dev/null 2>&1 || {
    swehero_pod_git_die "$label does not have an origin remote"
    return 1
  }

  local remote_ref="refs/remotes/origin/$expected_branch"
  if ! git -C "$repo_dir" fetch --prune origin \
    "+refs/heads/${expected_branch}:${remote_ref}"; then
    swehero_pod_git_die "could not fetch origin/$expected_branch for $label"
    return 1
  fi
  if ! git -C "$repo_dir" show-ref --verify --quiet "$remote_ref"; then
    swehero_pod_git_die "origin/$expected_branch does not exist for $label"
    return 1
  fi

  if git -C "$repo_dir" show-ref --verify --quiet "refs/heads/$expected_branch"; then
    if ! git -C "$repo_dir" merge-base --is-ancestor "$expected_branch" "$remote_ref"; then
      swehero_pod_git_die \
        "$label has commits on $expected_branch that are not on origin/$expected_branch; push or reset them before launching"
      return 1
    fi
    git -C "$repo_dir" checkout --quiet "$expected_branch"
  else
    git -C "$repo_dir" checkout --quiet -b "$expected_branch" --track "$remote_ref"
  fi

  local current_branch
  current_branch="$(git -C "$repo_dir" branch --show-current)"
  if [[ "$current_branch" != "$expected_branch" ]]; then
    swehero_pod_git_die \
      "$label is on branch '$current_branch', expected '$expected_branch'"
    return 1
  fi

  if ! git -C "$repo_dir" merge-base --is-ancestor HEAD "$remote_ref"; then
    swehero_pod_git_die \
      "$label has commits that are not on origin/$expected_branch; push or reset them before launching"
    return 1
  fi
  git -C "$repo_dir" merge --ff-only --quiet "$remote_ref"

  local head
  head="$(git -C "$repo_dir" rev-parse HEAD)"
  local remote_head
  remote_head="$(git -C "$repo_dir" rev-parse "$remote_ref^{commit}")"
  if [[ "$head" != "$remote_head" ]]; then
    swehero_pod_git_die \
      "$label is not at origin/$expected_branch after fast-forward: $head != $remote_head"
    return 1
  fi
  swehero_require_clean_pod_git_status "$repo_dir" "$label" || return 1
}
