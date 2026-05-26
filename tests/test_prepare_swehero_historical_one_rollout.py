"""Tests for building the public one-rollout SWE-HERO training artifact.

The dataset-prep script chooses one OpenHands rollout per SWE-bench task before
those rollouts become supervised training data. These tests document the
ML-facing curation rules: keep rows that look like useful CodeAct imitation
targets, reject rows that would train bad behavior, and choose one deterministic
rollout per task so no task is over-weighted during training.
"""

from __future__ import annotations

from typing import Any

from scripts import prepare_swehero_historical_one_rollout as prep


def assistant(
    tool_name: str = "think",
    tool_call_id: str = "call-1",
    content: str = "",
) -> dict[str, Any]:
    """Build one assistant action in the OpenHands CodeAct shape."""

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    }


def tool(tool_call_id: str = "call-1", content: str = "ok") -> dict[str, Any]:
    """Build the environment observation paired with an assistant tool call."""

    return {"role": "tool", "id": tool_call_id, "content": content, "tool_calls": []}


class TestPrepareSweHeroHistoricalOneRollout:
    """Verify the public-data approximation of the SWE-HERO row filters."""

    def test_evaluate_row_accepts_paper_available_valid_row(self) -> None:
        row = {
            "model_patch": "diff --git a/file.py b/file.py\n",
            "trajectory": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "issue"},
                assistant("str_replace_editor", "edit-1"),
                tool("edit-1", "Edited /workspace/file.py"),
                assistant("finish", "finish-1"),
                tool("finish-1", "done"),
            ],
        }

        evaluation = prep.evaluate_row(row)

        # A retained rollout should contain a concrete patch and assistant turns
        # that each choose exactly one environment action. That gives supervised
        # fine-tuning a clear target: observe context, emit one tool/action call,
        # observe the result, then continue.
        assert evaluation.accepted
        assert evaluation.assistant_turns == 2
        assert evaluation.str_replace_editor_errors == 0
        assert evaluation.reject_reasons == ()

    def test_evaluate_row_rejects_non_meaningful_public_filter_failures(self) -> None:
        blank_patch = prep.evaluate_row({"model_patch": "", "trajectory": []})
        # Without a model_patch, the rollout may contain exploration but does not
        # teach the model the complete "inspect, edit, produce patch" workflow.
        assert "null_model_patch" in blank_patch.reject_reasons

        missing_tool_call = prep.evaluate_row(
            {
                "model_patch": "diff --git a/file.py b/file.py\n",
                "trajectory": [
                    {"role": "assistant", "content": "done", "tool_calls": []}
                ],
            }
        )
        # The paper keeps trajectories whose assistant turns map cleanly to one
        # CodeAct tool call. Zero or multiple tool calls make the supervised target
        # ambiguous and can teach schema-recovery behavior instead of task solving.
        assert (
            "assistant_turn_without_exactly_one_tool_call"
            in missing_tool_call.reject_reasons
        )

        too_long = prep.evaluate_row(
            {
                "model_patch": "diff --git a/file.py b/file.py\n",
                "trajectory": [
                    assistant(tool_call_id=f"call-{index}") for index in range(3)
                ],
            },
            max_assistant_turns=2,
        )
        # Assistant turns are a public proxy for trajectory length and struggle.
        # Very long traces over-weight looping or recovery behavior and consume a
        # large fraction of the future model's context window.
        assert "exceeds_max_assistant_turns" in too_long.reject_reasons

    def test_str_replace_editor_errors_are_counted_by_tool_call_id(self) -> None:
        row = {
            "model_patch": "diff --git a/file.py b/file.py\n",
            "trajectory": [
                assistant("str_replace_editor", "edit-1"),
                tool("edit-1", "Error: old_str did not appear exactly once"),
                assistant("str_replace_editor", "edit-2"),
                tool("edit-2", "Here's the result of running `cat -n`; no problem"),
                assistant("finish", "finish-1"),
            ],
        }

        evaluation = prep.evaluate_row(row, max_str_replace_editor_errors=0)

        # str_replace_editor is the OpenHands file-edit action. Counting failures
        # by tool_call_id connects the assistant edit request to its tool
        # observation, so unrelated tool output does not affect the data-quality
        # filter.
        assert evaluation.str_replace_editor_errors == 1
        assert "too_many_str_replace_editor_errors" in evaluation.reject_reasons

    def test_str_replace_editor_source_view_text_is_not_an_error(self) -> None:
        row = {
            "model_patch": "diff --git a/file.py b/file.py\n",
            "trajectory": [
                assistant("str_replace_editor", "view-1"),
                tool(
                    "view-1",
                    "Here's the result of running `cat -n`:\n"
                    "    raise FileNotFoundError('not found')\n"
                    "    old_str = 'literal text in a test fixture'\n",
                ),
                assistant("finish", "finish-1"),
            ],
        }

        evaluation = prep.evaluate_row(row)

        # Source-code views can contain words like "FileNotFoundError" or
        # "old_str" as literal code. The failure detector should only count tool
        # observations that actually report a failed edit, or useful trajectories
        # could be dropped from the training corpus.
        assert evaluation.accepted
        assert evaluation.str_replace_editor_errors == 0

    def test_better_selection_prefers_fewer_errors_then_shorter_then_source_order(
        self,
    ) -> None:
        current_eval = prep.RowEvaluation(
            accepted=True,
            reject_reasons=(),
            assistant_turns=10,
            trajectory_messages=20,
            assistant_tool_call_violations=0,
            str_replace_editor_errors=1,
            has_model_patch=True,
        )
        better_eval = prep.RowEvaluation(
            accepted=True,
            reject_reasons=(),
            assistant_turns=20,
            trajectory_messages=40,
            assistant_tool_call_violations=0,
            str_replace_editor_errors=0,
            has_model_patch=True,
        )
        current = prep.SelectedRow(
            instance_id="instance",
            trajectory_id="current",
            source_file="a.parquet",
            source_row_index=1,
            selection_rank=prep.selection_rank(current_eval, 1),
            evaluation=current_eval,
        )
        candidate = prep.SelectedRow(
            instance_id="instance",
            trajectory_id="candidate",
            source_file="a.parquet",
            source_row_index=2,
            selection_rank=prep.selection_rank(better_eval, 2),
            evaluation=better_eval,
        )

        _row, selected = prep.better_selection(
            ({"id": "current"}, current), {"id": "candidate"}, candidate
        )

        # Multiple public rollouts can exist for one SWE-bench task. Keeping only
        # one prevents that task from having extra training weight. The rank first
        # prefers fewer edit failures, then shorter accepted trajectories, then
        # source order for deterministic reproducibility.
        assert selected.trajectory_id == "candidate"
