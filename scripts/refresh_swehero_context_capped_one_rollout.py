"""Refresh the local SWE-Hero one-rollout artifact under a Qwen context cap.

The existing public approximation selects one rollout per ``instance_id`` using
the public-column filters and deterministic rank in
``prepare_swehero_historical_one_rollout.py``. This script preserves those rows
unless their Qwen/OpenHands serialized shifted input length exceeds the training
context. For an over-context selected row, it scans the same pinned source
revision and selects the best same-task rollout that:

* passes the same public-column filters; and
* has shifted input length <= the configured context cap.

If no same-task rollout fits the context cap, the task is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import prepare_swehero_historical_one_rollout as prep
from scripts import qwen_swehero_train as train

DEFAULT_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tmp"
    / "hf"
    / "Qwen2.5-Coder-7B-Instruct-tokenizer-only"
)
TRACE_SERIALIZER = (
    "Qwen2.5 ChatML over OpenHands messages; same segments as "
    "scripts.qwen_swehero_train.encode_swehero_example"
)


@dataclass(frozen=True)
class ContextEvaluation:
    token_count: int
    shifted_input_length: int
    max_shifted_context: int
    fits_context: bool


@dataclass(frozen=True)
class ContextSelectedRow:
    row: dict[str, Any]
    selected: prep.SelectedRow
    context: ContextEvaluation


@dataclass(frozen=True)
class RefreshSummary:
    current_selected_rows: int
    current_over_context_rows: int
    replacement_rows: int
    excluded_tasks_without_fit: int
    final_selected_rows: int
    source_rows_scanned_for_replacements: int


@dataclass(frozen=True)
class CurrentArtifactStatus:
    is_current: bool
    reasons: tuple[str, ...]
    summary: RefreshSummary | None = None


class QwenTokenizerAdapter:
    """Small adapter matching TorchTitan's HuggingFaceTokenizer surface."""

    def __init__(self, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer

        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
        config = json.loads((tokenizer_path / "tokenizer_config.json").read_text())

        def token_content(key: str) -> str | None:
            value = config.get(key)
            if isinstance(value, dict):
                value = value.get("content")
            return value if isinstance(value, str) else None

        bos_token = token_content("bos_token")
        eos_token = token_content("eos_token")
        pad_token = token_content("pad_token")
        self.bos_id = self.tokenizer.token_to_id(bos_token) if bos_token else None
        self.eos_id = self.tokenizer.token_to_id(eos_token) if eos_token else None
        self.pad_id = self.tokenizer.token_to_id(pad_token) if pad_token else None

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return list(self.tokenizer.encode(text).ids)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)


def evaluate_qwen_context(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_shifted_context: int,
    include_model_patch: bool = False,
) -> ContextEvaluation:
    """Return the exact shifted input length used by SWE-Hero training."""

    token_count = 0
    bos_id = getattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None))
    if bos_id is not None:
        token_count += 1

    segments = train.qwen_openhands_segments(
        row, include_model_patch=include_model_patch
    )
    tokenized_segments = train._tokenize_texts(
        tokenizer, (text for text, _is_trainable in segments)
    )
    if len(tokenized_segments) != len(segments):
        raise RuntimeError(
            "Tokenizer returned a different number of segment encodings than "
            f"requested: {len(tokenized_segments)} != {len(segments)}"
        )
    token_count += sum(len(ids) for ids in tokenized_segments)

    eos_id = getattr(tokenizer, "eos_id", getattr(tokenizer, "eos_token_id", None))
    if eos_id is not None:
        token_count += 1

    shifted_input_length = max(0, token_count - 1)
    return ContextEvaluation(
        token_count=token_count,
        shifted_input_length=shifted_input_length,
        max_shifted_context=max_shifted_context,
        fits_context=shifted_input_length <= max_shifted_context,
    )


def select_better_context_row(
    current: ContextSelectedRow | None, candidate: ContextSelectedRow
) -> ContextSelectedRow:
    if current is None:
        return candidate
    if candidate.selected.selection_rank < current.selected.selection_rank:
        return candidate
    return current


def read_local_parquet_rows(dataset_path: Path, *, batch_size: int):
    import pyarrow.parquet as pq

    data_dir = dataset_path / "data"
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards found under {data_dir}")

    schema = None
    for path in files:
        parquet_file = pq.ParquetFile(path)
        schema = schema or parquet_file.schema_arrow
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield schema, row


def selected_row_from_json(payload: dict[str, Any]) -> prep.SelectedRow:
    evaluation_payload = dict(payload["evaluation"])
    evaluation_payload["reject_reasons"] = tuple(
        evaluation_payload.get("reject_reasons") or ()
    )
    return prep.SelectedRow(
        instance_id=str(payload["instance_id"]),
        trajectory_id=str(payload["trajectory_id"]),
        source_file=str(payload["source_file"]),
        source_row_index=int(payload["source_row_index"]),
        selection_rank=tuple(int(value) for value in payload["selection_rank"]),
        evaluation=prep.RowEvaluation(**evaluation_payload),
    )


def load_current_selected_rows(
    dataset_path: Path,
    tokenizer: Any,
    *,
    max_shifted_context: int,
    batch_size: int,
    include_model_patch: bool,
) -> tuple[Any, dict[str, ContextSelectedRow]]:
    manifest_path = dataset_path / "selection_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing selection manifest: {manifest_path}")

    manifest_by_instance: dict[str, prep.SelectedRow] = {}
    with manifest_path.open() as handle:
        for line in handle:
            selected = selected_row_from_json(json.loads(line))
            manifest_by_instance[selected.instance_id] = selected

    rows_by_instance: dict[str, ContextSelectedRow] = {}
    schema = None
    for row_schema, row in read_local_parquet_rows(dataset_path, batch_size=batch_size):
        schema = schema or row_schema
        instance_id = prep._string(row.get("instance_id"))
        selected = manifest_by_instance.get(instance_id)
        if selected is None:
            raise RuntimeError(
                f"Parquet row for {instance_id} is missing from selection manifest"
            )
        context = evaluate_qwen_context(
            tokenizer,
            row,
            max_shifted_context=max_shifted_context,
            include_model_patch=include_model_patch,
        )
        rows_by_instance[instance_id] = ContextSelectedRow(
            row=row, selected=selected, context=context
        )

    if schema is None:
        raise RuntimeError(f"No rows read from {dataset_path}")
    if set(rows_by_instance) != set(manifest_by_instance):
        missing_rows = sorted(set(manifest_by_instance) - set(rows_by_instance))
        extra_rows = sorted(set(rows_by_instance) - set(manifest_by_instance))
        raise RuntimeError(
            "Dataset Parquet rows and selection manifest disagree: "
            f"missing_rows={missing_rows[:10]}, extra_rows={extra_rows[:10]}"
        )
    return schema, rows_by_instance


def find_fit_replacements(
    *,
    instance_ids: set[str],
    tokenizer: Any,
    dataset_id: str,
    revision: str,
    max_assistant_turns: int,
    max_str_replace_editor_errors: int,
    max_shifted_context: int,
    batch_size: int,
    include_model_patch: bool,
) -> tuple[int, dict[str, ContextSelectedRow], dict[str, int], dict[str, int]]:
    replacements: dict[str, ContextSelectedRow] = {}
    accepted_candidates_by_instance = dict.fromkeys(instance_ids, 0)
    fit_candidates_by_instance = dict.fromkeys(instance_ids, 0)
    source_rows = 0

    files = prep.api_dataset_files(dataset_id, revision)
    if not files:
        raise RuntimeError(f"No parquet data files found for {dataset_id}@{revision}")

    for filename in files:
        print(f"reading {filename}", file=sys.stderr)
        url = prep.resolve_url(dataset_id, revision, filename)
        for _row_schema, row in prep.read_rows_from_parquet_url(
            url, batch_size=batch_size
        ):
            source_row_index = source_rows
            source_rows += 1
            instance_id = prep._string(row.get("instance_id"))
            if instance_id not in instance_ids:
                continue

            evaluation = prep.evaluate_row(
                row,
                max_assistant_turns=max_assistant_turns,
                max_str_replace_editor_errors=max_str_replace_editor_errors,
            )
            if not evaluation.accepted:
                continue
            accepted_candidates_by_instance[instance_id] += 1

            context = evaluate_qwen_context(
                tokenizer,
                row,
                max_shifted_context=max_shifted_context,
                include_model_patch=include_model_patch,
            )
            if not context.fits_context:
                continue
            fit_candidates_by_instance[instance_id] += 1

            selected = prep.SelectedRow(
                instance_id=instance_id,
                trajectory_id=prep._string(row.get("trajectory_id")),
                source_file=filename,
                source_row_index=source_row_index,
                selection_rank=prep.selection_rank(evaluation, source_row_index),
                evaluation=evaluation,
            )
            candidate = ContextSelectedRow(row=row, selected=selected, context=context)
            replacements[instance_id] = select_better_context_row(
                replacements.get(instance_id), candidate
            )

    return (
        source_rows,
        replacements,
        accepted_candidates_by_instance,
        fit_candidates_by_instance,
    )


def selected_context_to_json(selected: ContextSelectedRow) -> dict[str, Any]:
    payload = prep._selected_row_to_json(selected.selected)
    payload["context"] = asdict(selected.context)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def context_filter_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_shifted_context": args.max_shifted_context,
        "max_token_count": args.max_shifted_context + 1,
        "include_model_patch": args.include_model_patch,
        "tokenizer_json_sha256": hash_file(args.tokenizer_path / "tokenizer.json"),
        "tokenizer_config_sha256": hash_file(
            args.tokenizer_path / "tokenizer_config.json"
        ),
        "trace_serializer": TRACE_SERIALIZER,
    }


def _summary_from_counts(counts: Mapping[str, Any]) -> RefreshSummary | None:
    values: dict[str, int] = {}
    for field in RefreshSummary.__dataclass_fields__:
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < 0:
            return None
        values[field] = value
    return RefreshSummary(**values)


def _is_same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def current_artifact_status(
    dataset_path: Path,
    args: argparse.Namespace,
    *,
    context_filter: Mapping[str, Any],
) -> CurrentArtifactStatus:
    reasons: list[str] = []
    metadata_path = dataset_path / "metadata.json"
    report_path = dataset_path / "context_filter_report.json"
    manifest_path = dataset_path / "selection_manifest.jsonl"
    data_dir = dataset_path / "data"

    if not metadata_path.is_file():
        reasons.append("missing metadata.json")
    if not report_path.is_file():
        reasons.append("missing context_filter_report.json")
    if not manifest_path.is_file():
        reasons.append("missing selection_manifest.jsonl")
    if not data_dir.is_dir() or not any(data_dir.glob("*.parquet")):
        reasons.append("missing parquet shards")
    if reasons:
        return CurrentArtifactStatus(False, tuple(reasons))

    try:
        metadata = read_json(metadata_path)
        report = read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CurrentArtifactStatus(False, (f"unreadable refresh metadata: {exc}",))

    if metadata.get("dataset_id") != args.dataset_id:
        reasons.append("dataset_id changed")
    if metadata.get("source_revision") != args.revision:
        reasons.append("source revision changed")
    source_files = metadata.get("source_files")
    if (
        not isinstance(source_files, list)
        or not source_files
        or not all(isinstance(item, str) and item for item in source_files)
    ):
        reasons.append("missing source file manifest")

    expected_filters = {
        "exactly_one_tool_call_per_assistant_turn": True,
        "max_assistant_turns": args.max_assistant_turns,
        "max_str_replace_editor_errors": args.max_str_replace_editor_errors,
        "non_null_model_patch": True,
    }
    if metadata.get("paper_reproducible_filters") != expected_filters:
        reasons.append("public filter settings changed")

    stored_context = metadata.get("context_filter")
    if not isinstance(stored_context, Mapping):
        reasons.append("missing context_filter metadata")
    else:
        for key, expected_value in context_filter.items():
            if stored_context.get(key) != expected_value:
                reasons.append(f"context_filter.{key} changed")

    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        reasons.append("missing count summary")
        summary = None
    else:
        summary = _summary_from_counts(counts)
        if summary is None:
            reasons.append("invalid count summary")

    report_summary = report.get("summary")
    if isinstance(counts, Mapping) and report_summary != dict(counts):
        reasons.append("context report summary disagrees with metadata counts")

    if summary is not None:
        if summary.final_selected_rows <= 0:
            reasons.append("empty final dataset")
        manifest_rows = 0
        with manifest_path.open() as handle:
            for manifest_rows, _line in enumerate(handle, start=1):
                pass
        if manifest_rows != summary.final_selected_rows:
            reasons.append("selection manifest row count disagrees with metadata")

    max_final = report.get("max_final_shifted_input_length")
    if isinstance(max_final, bool) or not isinstance(max_final, int):
        reasons.append("missing final context maximum")
    elif max_final > args.max_shifted_context:
        reasons.append("final context maximum exceeds requested cap")

    replaced = report.get("replaced")
    excluded = report.get("excluded")
    if summary is not None:
        if not isinstance(replaced, list) or len(replaced) != summary.replacement_rows:
            reasons.append("replacement report count disagrees with metadata")
        if (
            not isinstance(excluded, list)
            or len(excluded) != summary.excluded_tasks_without_fit
        ):
            reasons.append("exclusion report count disagrees with metadata")

    return CurrentArtifactStatus(
        not reasons,
        tuple(reasons),
        summary if not reasons else None,
    )


def write_readme(path: Path, metadata: dict[str, Any]) -> None:
    context_filter = metadata["context_filter"]
    counts = metadata["counts"]
    path.write_text(
        "\n".join(
            [
                "# SWE-Hero Historical One-Rollout 128k Public Approximation",
                "",
                "This local dataset was refreshed from the closest public historical",
                "revision of `nvidia/SWE-Hero-openhands-trajectories`.",
                "",
                "It starts from the one-rollout public approximation and replaces",
                "selected rows that exceed the Qwen/OpenHands 128k shifted context",
                "with the best same-task rollout that fits the context cap.",
                "",
                "## Source",
                "",
                f"- Dataset: `{metadata['dataset_id']}`",
                f"- Revision: `{metadata['source_revision']}`",
                f"- Current selected rows analyzed: {counts['current_selected_rows']}",
                f"- Current rows over context: {counts['current_over_context_rows']}",
                f"- Replacement rows: {counts['replacement_rows']}",
                (
                    "- Tasks excluded with no fitting rollout: "
                    f"{counts['excluded_tasks_without_fit']}"
                ),
                f"- Final selected rows: {counts['final_selected_rows']}",
                "",
                "## Context Filter",
                "",
                (
                    "- Maximum shifted input length: "
                    f"{context_filter['max_shifted_context']}"
                ),
                "- Tokenization: Qwen2.5-Coder ChatML over OpenHands messages",
                "- Model patch appended: "
                f"{'yes' if context_filter['include_model_patch'] else 'no'}",
                "",
                "## Selection",
                "",
                "Rows are filtered using the paper-described criteria observable",
                "from the public trajectory columns, constrained to the context cap,",
                "then reduced to one rollout per `instance_id` by selecting the",
                "candidate with:",
                "",
                "1. the fewest `str_replace_editor` errors,",
                "2. the fewest assistant turns,",
                "3. the earliest source row index.",
                "",
                "## Limitation",
                "",
                prep.TEST_PATCH_FILTER_CAVEAT,
                "",
                "See `metadata.json`, `selection_manifest.jsonl`, and",
                "`context_filter_report.json` for exact replacement and exclusion",
                "details.",
                "",
            ]
        )
    )


def write_dataset(
    *,
    output_dir: Path,
    schema: Any,
    selected_rows: list[ContextSelectedRow],
    rows_per_shard: int,
    metadata: dict[str, Any],
    report: dict[str, Any],
    overwrite: bool,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty; pass --overwrite"
        )

    staging_dir = (
        output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    data_dir = staging_dir / "data"
    data_dir.mkdir(parents=True)

    try:
        shard_count = math.ceil(len(selected_rows) / rows_per_shard)
        for shard_index, start in enumerate(
            range(0, len(selected_rows), rows_per_shard)
        ):
            shard_rows = [
                selected.row
                for selected in selected_rows[start : start + rows_per_shard]
            ]
            shard_path = data_dir / (
                f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
            )
            table = pa.Table.from_pylist(shard_rows, schema=schema)
            pq.write_table(table, shard_path, compression="zstd")

        write_jsonl(
            staging_dir / "selection_manifest.jsonl",
            (selected_context_to_json(selected) for selected in selected_rows),
        )
        write_json(staging_dir / "metadata.json", metadata)
        write_json(staging_dir / "context_filter_report.json", report)
        write_readme(staging_dir / "README.md", metadata)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.rename(output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _refresh_dataset_exact(
    args: argparse.Namespace,
    *,
    context_filter: Mapping[str, Any],
) -> RefreshSummary:
    started_at = time.time()
    tokenizer = QwenTokenizerAdapter(args.tokenizer_path)

    print("analyzing current selected dataset", file=sys.stderr)
    schema, current = load_current_selected_rows(
        args.dataset_path,
        tokenizer,
        max_shifted_context=args.max_shifted_context,
        batch_size=args.batch_size,
        include_model_patch=args.include_model_patch,
    )
    over_context = {
        instance_id: selected
        for instance_id, selected in current.items()
        if not selected.context.fits_context
    }
    print(
        f"current rows over context: {len(over_context)} / {len(current)}",
        file=sys.stderr,
    )

    source_rows_scanned = 0
    replacements: dict[str, ContextSelectedRow] = {}
    accepted_candidates_by_instance: dict[str, int] = {}
    fit_candidates_by_instance: dict[str, int] = {}
    if over_context:
        (
            source_rows_scanned,
            replacements,
            accepted_candidates_by_instance,
            fit_candidates_by_instance,
        ) = find_fit_replacements(
            instance_ids=set(over_context),
            tokenizer=tokenizer,
            dataset_id=args.dataset_id,
            revision=args.revision,
            max_assistant_turns=args.max_assistant_turns,
            max_str_replace_editor_errors=args.max_str_replace_editor_errors,
            max_shifted_context=args.max_shifted_context,
            batch_size=args.batch_size,
            include_model_patch=args.include_model_patch,
        )

    final_by_instance: dict[str, ContextSelectedRow] = {}
    replaced: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for instance_id, selected in current.items():
        if instance_id not in over_context:
            final_by_instance[instance_id] = selected
            continue

        replacement = replacements.get(instance_id)
        if replacement is None:
            excluded.append(
                {
                    "instance_id": instance_id,
                    "current_trajectory_id": selected.selected.trajectory_id,
                    "current_source_file": selected.selected.source_file,
                    "current_source_row_index": selected.selected.source_row_index,
                    "current_context": asdict(selected.context),
                    "accepted_candidates": accepted_candidates_by_instance.get(
                        instance_id, 0
                    ),
                    "fit_candidates": fit_candidates_by_instance.get(instance_id, 0),
                }
            )
            continue

        final_by_instance[instance_id] = replacement
        replaced.append(
            {
                "instance_id": instance_id,
                "old_trajectory_id": selected.selected.trajectory_id,
                "old_source_file": selected.selected.source_file,
                "old_source_row_index": selected.selected.source_row_index,
                "old_selection_rank": list(selected.selected.selection_rank),
                "old_context": asdict(selected.context),
                "new_trajectory_id": replacement.selected.trajectory_id,
                "new_source_file": replacement.selected.source_file,
                "new_source_row_index": replacement.selected.source_row_index,
                "new_selection_rank": list(replacement.selected.selection_rank),
                "new_context": asdict(replacement.context),
                "accepted_candidates": accepted_candidates_by_instance.get(
                    instance_id, 0
                ),
                "fit_candidates": fit_candidates_by_instance.get(instance_id, 0),
            }
        )

    selected_rows = sorted(
        final_by_instance.values(), key=lambda item: item.selected.source_row_index
    )
    summary = RefreshSummary(
        current_selected_rows=len(current),
        current_over_context_rows=len(over_context),
        replacement_rows=len(replaced),
        excluded_tasks_without_fit=len(excluded),
        final_selected_rows=len(selected_rows),
        source_rows_scanned_for_replacements=source_rows_scanned,
    )
    source_files = prep.api_dataset_files(args.dataset_id, args.revision)
    metadata = {
        "dataset_id": args.dataset_id,
        "source_revision": args.revision,
        "source_files": source_files,
        "created_at_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "selection": {
            "goal": (
                "one selected public rollout per instance_id with shifted input "
                f"length <= {args.max_shifted_context}"
            ),
            "rank": [
                "lowest str_replace_editor error count",
                "lowest assistant turn count",
                "earliest source row index",
            ],
        },
        "paper_reproducible_filters": {
            "non_null_model_patch": True,
            "max_assistant_turns": args.max_assistant_turns,
            "exactly_one_tool_call_per_assistant_turn": True,
            "max_str_replace_editor_errors": args.max_str_replace_editor_errors,
        },
        "paper_filter_not_reproducible_from_public_columns": {
            "model_patch_must_not_touch_test_patch_files": prep.TEST_PATCH_FILTER_CAVEAT,
        },
        "context_filter": {
            **dict(context_filter),
            "tokenizer_path": str(args.tokenizer_path),
        },
        "base_dataset_path": str(args.dataset_path),
        "counts": asdict(summary),
    }
    report = {
        "summary": asdict(summary),
        "replaced": sorted(replaced, key=lambda item: item["instance_id"]),
        "excluded": sorted(excluded, key=lambda item: item["instance_id"]),
        "max_current_shifted_input_length": max(
            selected.context.shifted_input_length for selected in current.values()
        ),
        "max_final_shifted_input_length": max(
            selected.context.shifted_input_length for selected in selected_rows
        ),
    }

    write_dataset(
        output_dir=args.output_dir,
        schema=schema,
        selected_rows=selected_rows,
        rows_per_shard=args.rows_per_shard,
        metadata=metadata,
        report=report,
        overwrite=args.overwrite,
    )
    return summary


def refresh_dataset(args: argparse.Namespace) -> RefreshSummary:
    context_filter = context_filter_contract(args)
    if _is_same_path(args.dataset_path, args.output_dir):
        status = current_artifact_status(
            args.dataset_path,
            args,
            context_filter=context_filter,
        )
        if status.is_current and status.summary is not None:
            print(
                f"{args.dataset_path} already matches requested context refresh; "
                "leaving artifact unchanged",
                file=sys.stderr,
            )
            return status.summary
        if status.reasons:
            print(
                "refresh artifact metadata is not current; running exact refresh: "
                + "; ".join(status.reasons),
                file=sys.stderr,
            )

    return _refresh_dataset_exact(args, context_filter=context_filter)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace over-128k rows in the local one-rollout SWE-Hero artifact "
            "with same-task rollouts that fit the Qwen training context."
        )
    )
    parser.add_argument("--dataset-id", default=prep.DATASET_ID)
    parser.add_argument("--revision", default=prep.HISTORICAL_REVISION)
    parser.add_argument("--dataset-path", type=Path, default=prep.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=prep.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--max-shifted-context", type=int, default=train.PAPER_CONTEXT_LENGTH
    )
    parser.add_argument(
        "--max-assistant-turns", type=int, default=prep.MAX_ASSISTANT_TURNS
    )
    parser.add_argument(
        "--max-str-replace-editor-errors",
        type=int,
        default=prep.MAX_STR_REPLACE_EDITOR_ERRORS,
    )
    parser.add_argument(
        "--rows-per-shard", type=int, default=prep.DEFAULT_ROWS_PER_SHARD
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-model-patch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = refresh_dataset(args)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(f"artifact ready at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
