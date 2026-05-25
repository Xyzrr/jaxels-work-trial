import argparse
import json
import tempfile
from pathlib import Path
from unittest import mock

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


def context_selected_row(instance_id, trajectory_id, rank, shifted_length, fits):
    base = selected_row(trajectory_id, rank).selected
    selected = prep.SelectedRow(
        instance_id=instance_id,
        trajectory_id=base.trajectory_id,
        source_file=base.source_file,
        source_row_index=base.source_row_index,
        selection_rank=base.selection_rank,
        evaluation=base.evaluation,
    )
    return refresh.ContextSelectedRow(
        row={"instance_id": instance_id, "trajectory_id": trajectory_id},
        selected=selected,
        context=refresh.ContextEvaluation(
            token_count=shifted_length + 1,
            shifted_input_length=shifted_length,
            max_shifted_context=100,
            fits_context=fits,
        ),
    )


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_tokenizer(tmp_path, *, tokenizer_json=b"json", tokenizer_config=b"config"):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_bytes(tokenizer_json)
    (tokenizer_path / "tokenizer_config.json").write_bytes(tokenizer_config)
    return tokenizer_path


def make_args(tmp_path, tokenizer_path, **overrides):
    values = {
        "batch_size": 64,
        "dataset_id": "dataset",
        "dataset_path": tmp_path,
        "include_model_patch": False,
        "max_assistant_turns": 100,
        "max_shifted_context": 100,
        "max_str_replace_editor_errors": 2,
        "output_dir": tmp_path,
        "overwrite": True,
        "revision": "revision",
        "rows_per_shard": 2048,
        "tokenizer_path": tokenizer_path,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_artifact(tmp_path, args, *, max_final=99, counts=None):
    counts = counts or {
        "current_over_context_rows": 1,
        "current_selected_rows": 2,
        "excluded_tasks_without_fit": 0,
        "final_selected_rows": 2,
        "replacement_rows": 1,
        "source_rows_scanned_for_replacements": 10,
    }
    context_filter = refresh.context_filter_contract(args)
    write_json(
        tmp_path / "metadata.json",
        {
            "context_filter": {
                **context_filter,
                "tokenizer_path": str(args.tokenizer_path),
            },
            "counts": counts,
            "dataset_id": args.dataset_id,
            "paper_reproducible_filters": {
                "exactly_one_tool_call_per_assistant_turn": True,
                "max_assistant_turns": args.max_assistant_turns,
                "max_str_replace_editor_errors": args.max_str_replace_editor_errors,
                "non_null_model_patch": True,
            },
            "source_revision": args.revision,
            "source_files": ["data/source.parquet"],
        },
    )
    write_json(
        tmp_path / "context_filter_report.json",
        {
            "excluded": [],
            "max_final_shifted_input_length": max_final,
            "replaced": [{"instance_id": "x"}] * counts["replacement_rows"],
            "summary": counts,
        },
    )
    (tmp_path / "selection_manifest.jsonl").write_text(
        "".join("{}\n" for _ in range(counts["final_selected_rows"]))
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train-00000-of-00001.parquet").write_bytes(b"placeholder")
    return context_filter


class TestRefreshSweHeroContextCappedOneRollout:
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

        assert uncapped.shifted_input_length == uncapped.token_count - 1
        assert exact_cap.fits_context
        assert not under_cap.fits_context

    def test_replacement_selection_preserves_original_rank_order(self):
        current = selected_row("current", (1, 10, 5))
        fewer_errors = selected_row("fewer-errors", (0, 50, 6))
        shorter = selected_row("shorter", (1, 5, 7))
        later_tie = selected_row("later-tie", (1, 10, 8))

        assert (
            refresh.select_better_context_row(
                current, fewer_errors
            ).selected.trajectory_id
            == "fewer-errors"
        )
        assert (
            refresh.select_better_context_row(current, shorter).selected.trajectory_id
            == "shorter"
        )
        assert (
            refresh.select_better_context_row(current, later_tie).selected.trajectory_id
            == "current"
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

        assert selected.selection_rank == (0, 2, 42)
        assert selected.evaluation.reject_reasons == ()

    def test_current_artifact_metadata_short_circuits_exact_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = make_tokenizer(tmp_path)
            args = make_args(tmp_path, tokenizer_path)
            make_artifact(tmp_path, args)

            with mock.patch.object(
                refresh,
                "_refresh_dataset_exact",
                side_effect=AssertionError("exact refresh should not run"),
            ):
                summary = refresh.refresh_dataset(args)

        assert summary.final_selected_rows == 2
        assert summary.replacement_rows == 1

    def test_current_artifact_rejects_stale_metadata_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = make_tokenizer(tmp_path)
            args = make_args(tmp_path, tokenizer_path)
            context_filter = make_artifact(tmp_path, args)

            status = refresh.current_artifact_status(
                tmp_path, args, context_filter=context_filter
            )
            assert status.is_current

            changed_settings_args = make_args(
                tmp_path, tokenizer_path, include_model_patch=True
            )
            changed_settings_status = refresh.current_artifact_status(
                tmp_path,
                changed_settings_args,
                context_filter=refresh.context_filter_contract(changed_settings_args),
            )
            assert not changed_settings_status.is_current
            assert (
                "context_filter.include_model_patch changed"
                in changed_settings_status.reasons
            )

            (tokenizer_path / "tokenizer.json").write_bytes(b"different tokenizer")
            changed_tokenizer_status = refresh.current_artifact_status(
                tmp_path,
                args,
                context_filter=refresh.context_filter_contract(args),
            )
            assert not changed_tokenizer_status.is_current
            assert (
                "context_filter.tokenizer_json_sha256 changed"
                in changed_tokenizer_status.reasons
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = make_tokenizer(tmp_path)
            args = make_args(tmp_path, tokenizer_path)
            stale_context_filter = make_artifact(tmp_path, args, max_final=101)
            stale_status = refresh.current_artifact_status(
                tmp_path, args, context_filter=stale_context_filter
            )
            assert not stale_status.is_current
            assert "final context maximum exceeds requested cap" in stale_status.reasons

    def test_stale_metadata_falls_back_to_over_context_exact_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = make_tokenizer(tmp_path)
            args = make_args(tmp_path, tokenizer_path)
            make_artifact(tmp_path, args, max_final=101)
            current = context_selected_row(
                "issue-1", "old-over-context", (0, 4, 10), 120, False
            )
            replacement = context_selected_row(
                "issue-1", "new-fitting", (0, 2, 11), 80, True
            )

            with mock.patch.object(
                refresh, "QwenTokenizerAdapter", return_value=object()
            ):
                with mock.patch.object(
                    refresh,
                    "load_current_selected_rows",
                    return_value=("schema", {"issue-1": current}),
                ):
                    with mock.patch.object(
                        refresh,
                        "find_fit_replacements",
                        return_value=(
                            25,
                            {"issue-1": replacement},
                            {"issue-1": 2},
                            {"issue-1": 1},
                        ),
                    ):
                        with mock.patch.object(
                            refresh.prep,
                            "api_dataset_files",
                            return_value=["data/source.parquet"],
                        ):
                            with mock.patch.object(refresh, "write_dataset") as write:
                                summary = refresh.refresh_dataset(args)

        assert summary.current_over_context_rows == 1
        assert summary.replacement_rows == 1
        assert summary.final_selected_rows == 1
        write_kwargs = write.call_args.kwargs
        assert [
            row.selected.trajectory_id for row in write_kwargs["selected_rows"]
        ] == ["new-fitting"]
        assert (
            write_kwargs["report"]["replaced"][0]["new_trajectory_id"] == "new-fitting"
        )

    def test_optimized_fallback_matches_exact_refresh_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = make_tokenizer(tmp_path)
            args = make_args(tmp_path, tokenizer_path)
            make_artifact(tmp_path, args, max_final=101)
            current = context_selected_row("issue-1", "old", (0, 4, 10), 120, False)
            replacement = context_selected_row("issue-1", "new", (0, 2, 11), 80, True)

            def run_once(func):
                with (
                    mock.patch.object(
                        refresh, "QwenTokenizerAdapter", return_value=object()
                    ),
                    mock.patch.object(
                        refresh,
                        "load_current_selected_rows",
                        return_value=("schema", {"issue-1": current}),
                    ),
                    mock.patch.object(
                        refresh,
                        "find_fit_replacements",
                        return_value=(
                            25,
                            {"issue-1": replacement},
                            {"issue-1": 2},
                            {"issue-1": 1},
                        ),
                    ),
                    mock.patch.object(
                        refresh.prep,
                        "api_dataset_files",
                        return_value=["data/source.parquet"],
                    ),
                    mock.patch.object(refresh, "write_dataset") as write,
                ):
                    summary = func()
                    return summary, write.call_args.kwargs

            exact_summary, exact_write = run_once(
                lambda: refresh._refresh_dataset_exact(
                    args, context_filter=refresh.context_filter_contract(args)
                )
            )
            optimized_summary, optimized_write = run_once(
                lambda: refresh.refresh_dataset(args)
            )

        assert exact_summary == optimized_summary
        assert exact_write["report"] == optimized_write["report"]
        assert [row.selected.trajectory_id for row in exact_write["selected_rows"]] == [
            row.selected.trajectory_id for row in optimized_write["selected_rows"]
        ]
