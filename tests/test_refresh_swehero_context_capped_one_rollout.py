import unittest

from scripts import prepare_swehero_historical_one_rollout as prep
from scripts import refresh_swehero_context_capped_one_rollout as refresh


class CharacterTokenizer:
    bos_id = None
    eos_id = None

    def encode(self, text, **_kwargs):
        return list(range(len(text)))


def selected_row(trajectory_id, rank):
    evaluation = prep.RowEvaluation(
        accepted=True,
        reject_reasons=(),
        assistant_turns=rank[1],
        trajectory_messages=rank[1] * 2 + 1,
        assistant_tool_call_violations=0,
        str_replace_editor_errors=rank[0],
        has_model_patch=True,
    )
    return refresh.ContextSelectedRow(
        row={"trajectory_id": trajectory_id},
        selected=prep.SelectedRow(
            instance_id="task",
            trajectory_id=trajectory_id,
            source_file="data/train.parquet",
            source_row_index=rank[2],
            selection_rank=rank,
            evaluation=evaluation,
        ),
        context=refresh.ContextEvaluation(
            token_count=100,
            shifted_input_length=99,
            max_shifted_context=128,
            fits_context=True,
        ),
    )


class RefreshSweHeroContextCappedOneRolloutTests(unittest.TestCase):
    def test_qwen_context_evaluation_uses_shifted_input_length(self):
        row = {
            "trajectory": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "issue"},
                {
                    "role": "assistant",
                    "content": "final answer",
                    "tool_calls": [],
                },
            ]
        }

        uncapped = refresh.evaluate_qwen_context(
            CharacterTokenizer(), row, max_shifted_context=10_000
        )
        exact_cap = refresh.evaluate_qwen_context(
            CharacterTokenizer(),
            row,
            max_shifted_context=uncapped.shifted_input_length,
        )
        under_cap = refresh.evaluate_qwen_context(
            CharacterTokenizer(),
            row,
            max_shifted_context=uncapped.shifted_input_length - 1,
        )

        self.assertEqual(uncapped.shifted_input_length, uncapped.token_count - 1)
        self.assertTrue(exact_cap.fits_context)
        self.assertFalse(under_cap.fits_context)

    def test_replacement_selection_preserves_original_rank_order(self):
        current = selected_row("current", (1, 10, 5))
        fewer_errors = selected_row("fewer-errors", (0, 50, 6))
        shorter = selected_row("shorter", (1, 5, 7))
        later_tie = selected_row("later-tie", (1, 10, 8))

        self.assertEqual(
            refresh.select_better_context_row(current, fewer_errors).selected.trajectory_id,
            "fewer-errors",
        )
        self.assertEqual(
            refresh.select_better_context_row(current, shorter).selected.trajectory_id,
            "shorter",
        )
        self.assertEqual(
            refresh.select_better_context_row(current, later_tie).selected.trajectory_id,
            "current",
        )

    def test_manifest_parser_restores_reject_reason_tuple(self):
        payload = {
            "instance_id": "task",
            "trajectory_id": "trajectory",
            "source_file": "data/train.parquet",
            "source_row_index": 42,
            "selection_rank": [0, 2, 42],
            "evaluation": {
                "accepted": True,
                "reject_reasons": [],
                "assistant_turns": 2,
                "trajectory_messages": 5,
                "assistant_tool_call_violations": 0,
                "str_replace_editor_errors": 0,
                "has_model_patch": True,
            },
        }

        selected = refresh.selected_row_from_json(payload)

        self.assertEqual(selected.selection_rank, (0, 2, 42))
        self.assertEqual(selected.evaluation.reject_reasons, ())


if __name__ == "__main__":
    unittest.main()
