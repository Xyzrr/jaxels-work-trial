"""Materialize a one-rollout SWE-Hero dataset from the closest public history.

The SWE-ZERO to SWE-HERO paper reports a 13.2k SWE-HERO training set with a
single execution-backed rollout per task instance. The closest public Hugging
Face revision has 12,633 unique instances and mostly three rollouts per
instance. This script pins that historical revision, applies the paper filters
that are reproducible from public columns, and writes one selected trajectory per
instance as a local Parquet dataset.

The paper also filters trajectories whose model patch touches files from the
test patch. The public SWE-Hero trajectory rows do not include ``test_patch`` or
source task metadata, so that filter cannot be reproduced from this artifact.
The generated metadata records this limitation instead of silently pretending the
selection is the exact internal paper manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

DATASET_ID = "nvidia/SWE-Hero-openhands-trajectories"
HISTORICAL_REVISION = "5b2ed21270ad773a50163e2999c510f0cbb92cfa"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "swe-hero-openhands-trajectories-5b2ed21-one-rollout"
)
MAX_ASSISTANT_TURNS = 100
MAX_STR_REPLACE_EDITOR_ERRORS = 2
DEFAULT_ROWS_PER_SHARD = 2048

TEST_PATCH_FILTER_CAVEAT = (
    "The paper filters trajectories whose model_patch modifies any file present "
    "in the test patch, but the public SWE-Hero trajectory rows do not include "
    "test_patch or equivalent source-task metadata. This generated dataset "
    "therefore applies only the paper filters observable from public columns."
)


@dataclass(frozen=True)
class RowEvaluation:
    accepted: bool
    reject_reasons: tuple[str, ...]
    assistant_turns: int
    trajectory_messages: int
    assistant_tool_call_violations: int
    str_replace_editor_errors: int
    has_model_patch: bool


@dataclass(frozen=True)
class SelectedRow:
    instance_id: str
    trajectory_id: str
    source_file: str
    source_row_index: int
    selection_rank: tuple[int, int, int]
    evaluation: RowEvaluation


@dataclass(frozen=True)
class CountSummary:
    source_rows: int
    unique_instances_seen: int
    accepted_rows: int
    selected_rows: int
    rejected_rows: int
    duplicate_accepted_rows: int


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _trajectory(row: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = row.get("trajectory")
    return trajectory if isinstance(trajectory, list) else []


def _tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = turn.get("tool_calls") if isinstance(turn, dict) else None
    return tool_calls if isinstance(tool_calls, list) else []


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if isinstance(function, dict):
        return _string(function.get("name"))
    return ""


def _tool_call_id(tool_call: dict[str, Any]) -> str:
    return _string(tool_call.get("id")) if isinstance(tool_call, dict) else ""


def _is_str_replace_editor_error(content: str) -> bool:
    normalized = content.strip().lower()
    if not normalized:
        return False

    prefixes = (
        "error",
        "failed",
        "exception",
        "traceback",
    )
    if normalized.startswith(prefixes):
        return True

    editor_error_markers = ("no replacement was performed",)
    return any(marker in normalized for marker in editor_error_markers)


def count_str_replace_editor_errors(trajectory: list[dict[str, Any]]) -> int:
    tool_observations_by_id = {
        _string(turn.get("id")): _string(turn.get("content"))
        for turn in trajectory
        if isinstance(turn, dict) and turn.get("role") == "tool"
    }

    errors = 0
    for index, turn in enumerate(trajectory):
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        for tool_call in _tool_calls(turn):
            if _tool_call_name(tool_call) != "str_replace_editor":
                continue

            tool_call_id = _tool_call_id(tool_call)
            observation = tool_observations_by_id.get(tool_call_id)
            if observation is None:
                observation = _next_tool_observation(trajectory, index)
            if _is_str_replace_editor_error(observation or ""):
                errors += 1

    return errors


def _next_tool_observation(
    trajectory: list[dict[str, Any]], assistant_index: int
) -> str | None:
    for turn in trajectory[assistant_index + 1 :]:
        if isinstance(turn, dict) and turn.get("role") == "tool":
            return _string(turn.get("content"))
    return None


def evaluate_row(
    row: dict[str, Any],
    *,
    max_assistant_turns: int = MAX_ASSISTANT_TURNS,
    max_str_replace_editor_errors: int = MAX_STR_REPLACE_EDITOR_ERRORS,
) -> RowEvaluation:
    trajectory = _trajectory(row)
    assistant_turns = 0
    assistant_tool_call_violations = 0

    for turn in trajectory:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        assistant_turns += 1
        if len(_tool_calls(turn)) != 1:
            assistant_tool_call_violations += 1

    model_patch = _string(row.get("model_patch")).strip()
    has_model_patch = bool(model_patch and model_patch.lower() not in {"none", "null"})
    str_replace_editor_errors = count_str_replace_editor_errors(trajectory)

    reject_reasons: list[str] = []
    if not has_model_patch:
        reject_reasons.append("null_model_patch")
    if assistant_turns > max_assistant_turns:
        reject_reasons.append("exceeds_max_assistant_turns")
    if assistant_tool_call_violations:
        reject_reasons.append("assistant_turn_without_exactly_one_tool_call")
    if str_replace_editor_errors > max_str_replace_editor_errors:
        reject_reasons.append("too_many_str_replace_editor_errors")

    return RowEvaluation(
        accepted=not reject_reasons,
        reject_reasons=tuple(reject_reasons),
        assistant_turns=assistant_turns,
        trajectory_messages=len(trajectory),
        assistant_tool_call_violations=assistant_tool_call_violations,
        str_replace_editor_errors=str_replace_editor_errors,
        has_model_patch=has_model_patch,
    )


def selection_rank(
    evaluation: RowEvaluation, source_row_index: int
) -> tuple[int, int, int]:
    """Rank retained rollouts for deterministic one-per-instance selection."""

    return (
        evaluation.str_replace_editor_errors,
        evaluation.assistant_turns,
        source_row_index,
    )


def better_selection(
    current: tuple[dict[str, Any], SelectedRow] | None,
    candidate_row: dict[str, Any],
    candidate: SelectedRow,
) -> tuple[dict[str, Any], SelectedRow]:
    if current is None:
        return candidate_row, candidate

    _, current_meta = current
    if candidate.selection_rank < current_meta.selection_rank:
        return candidate_row, candidate
    return current


def api_dataset_files(dataset_id: str, revision: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{dataset_id}/revision/{revision}"
    with urlopen(url) as response:
        payload = json.load(response)

    files = [
        sibling["rfilename"]
        for sibling in payload.get("siblings", [])
        if sibling.get("rfilename", "").startswith("data/")
        and sibling.get("rfilename", "").endswith(".parquet")
    ]
    return sorted(files)


def resolve_url(dataset_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{filename}"


def read_rows_from_parquet_url(url: str, *, batch_size: int):
    import fsspec
    import pyarrow.parquet as pq

    with fsspec.open(url, "rb") as file_obj:
        parquet_file = pq.ParquetFile(file_obj)
        schema = parquet_file.schema_arrow
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield schema, row


def build_dataset(
    *,
    dataset_id: str,
    revision: str,
    output_dir: Path,
    max_assistant_turns: int,
    max_str_replace_editor_errors: int,
    rows_per_shard: int,
    batch_size: int,
    overwrite: bool,
) -> CountSummary:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty; pass --overwrite to replace it"
        )

    if output_dir.exists() and overwrite:
        _remove_output_dir(output_dir)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[str, tuple[dict[str, Any], SelectedRow]] = {}
    unique_instances_seen: set[str] = set()
    reject_counts: dict[str, int] = {}
    accepted_rows = 0
    source_rows = 0
    schema: pa.Schema | None = None

    files = api_dataset_files(dataset_id, revision)
    if not files:
        raise RuntimeError(f"No parquet data files found for {dataset_id}@{revision}")

    started_at = time.time()
    for filename in files:
        url = resolve_url(dataset_id, revision, filename)
        print(f"reading {filename}", file=sys.stderr)
        for row_schema, row in read_rows_from_parquet_url(url, batch_size=batch_size):
            schema = schema or row_schema
            source_rows += 1
            instance_id = _string(row.get("instance_id"))
            trajectory_id = _string(row.get("trajectory_id"))
            unique_instances_seen.add(instance_id)

            evaluation = evaluate_row(
                row,
                max_assistant_turns=max_assistant_turns,
                max_str_replace_editor_errors=max_str_replace_editor_errors,
            )
            if not evaluation.accepted:
                for reason in evaluation.reject_reasons:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue

            accepted_rows += 1
            candidate = SelectedRow(
                instance_id=instance_id,
                trajectory_id=trajectory_id,
                source_file=filename,
                source_row_index=source_rows - 1,
                selection_rank=selection_rank(evaluation, source_rows - 1),
                evaluation=evaluation,
            )
            selected[instance_id] = better_selection(
                selected.get(instance_id), row, candidate
            )

    if schema is None:
        raise RuntimeError("No rows read from source dataset")

    selected_items = sorted(
        selected.values(), key=lambda item: item[1].source_row_index
    )
    selected_rows = [row for row, _meta in selected_items]
    selected_meta = [_meta for _row, _meta in selected_items]

    for shard_index, start in enumerate(range(0, len(selected_rows), rows_per_shard)):
        shard_rows = selected_rows[start : start + rows_per_shard]
        shard_count = math.ceil(len(selected_rows) / rows_per_shard)
        shard_path = data_dir / (
            f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
        )
        table = pa.Table.from_pylist(shard_rows, schema=schema)
        pq.write_table(table, shard_path, compression="zstd")

    _write_jsonl(
        output_dir / "selection_manifest.jsonl",
        (_selected_row_to_json(meta) for meta in selected_meta),
    )

    summary = CountSummary(
        source_rows=source_rows,
        unique_instances_seen=len(unique_instances_seen),
        accepted_rows=accepted_rows,
        selected_rows=len(selected_rows),
        rejected_rows=source_rows - accepted_rows,
        duplicate_accepted_rows=accepted_rows - len(selected_rows),
    )
    metadata = {
        "dataset_id": dataset_id,
        "source_revision": revision,
        "source_files": files,
        "created_at_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "selection": {
            "goal": "one selected public rollout per instance_id",
            "rank": [
                "lowest str_replace_editor error count",
                "lowest assistant turn count",
                "earliest source row index",
            ],
        },
        "paper_reproducible_filters": {
            "non_null_model_patch": True,
            "max_assistant_turns": max_assistant_turns,
            "exactly_one_tool_call_per_assistant_turn": True,
            "max_str_replace_editor_errors": max_str_replace_editor_errors,
        },
        "paper_filter_not_reproducible_from_public_columns": {
            "model_patch_must_not_touch_test_patch_files": TEST_PATCH_FILTER_CAVEAT,
        },
        "counts": asdict(summary),
        "reject_counts": reject_counts,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_readme(output_dir / "README.md", metadata)
    return summary


def _selected_row_to_json(selected_row: SelectedRow) -> dict[str, Any]:
    return {
        "instance_id": selected_row.instance_id,
        "trajectory_id": selected_row.trajectory_id,
        "source_file": selected_row.source_file,
        "source_row_index": selected_row.source_row_index,
        "selection_rank": list(selected_row.selection_rank),
        "evaluation": asdict(selected_row.evaluation),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_readme(path: Path, metadata: dict[str, Any]) -> None:
    counts = metadata["counts"]
    path.write_text(
        "\n".join(
            [
                "# SWE-Hero Historical One-Rollout Public Approximation",
                "",
                "This local dataset was generated from the closest public historical",
                "revision of `nvidia/SWE-Hero-openhands-trajectories`.",
                "",
                "It is suitable for local training scripts that expect the public",
                "SWE-Hero OpenHands trajectory schema, but it is not the exact internal",
                "13.2k paper manifest.",
                "",
                "## Source",
                "",
                f"- Dataset: `{metadata['dataset_id']}`",
                f"- Revision: `{metadata['source_revision']}`",
                f"- Source rows scanned: {counts['source_rows']}",
                f"- Unique instances seen: {counts['unique_instances_seen']}",
                f"- Accepted rows after public-column filters: {counts['accepted_rows']}",
                f"- Selected rows: {counts['selected_rows']}",
                "",
                "## Selection",
                "",
                "Rows were filtered using the paper-described criteria observable",
                "from the public trajectory columns, then reduced to one rollout per",
                "`instance_id` by selecting the candidate with:",
                "",
                "1. the fewest `str_replace_editor` errors,",
                "2. the fewest assistant turns,",
                "3. the earliest source row index.",
                "",
                "## Limitation",
                "",
                TEST_PATCH_FILTER_CAVEAT,
                "",
                "See `metadata.json` and `selection_manifest.jsonl` for exact counts",
                "and selected `instance_id`/`trajectory_id` pairs.",
                "",
            ]
        )
    )


def _remove_output_dir(path: Path) -> None:
    import shutil

    if path.resolve() == Path("/").resolve():
        raise ValueError("Refusing to remove /")
    shutil.rmtree(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a one-rollout-per-instance local SWE-Hero dataset from the "
            "12,633-unique-instance historical public revision."
        )
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=HISTORICAL_REVISION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-assistant-turns", type=int, default=MAX_ASSISTANT_TURNS)
    parser.add_argument(
        "--max-str-replace-editor-errors",
        type=int,
        default=MAX_STR_REPLACE_EDITOR_ERRORS,
    )
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_dataset(
        dataset_id=args.dataset_id,
        revision=args.revision,
        output_dir=args.output_dir,
        max_assistant_turns=args.max_assistant_turns,
        max_str_replace_editor_errors=args.max_str_replace_editor_errors,
        rows_per_shard=args.rows_per_shard,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
