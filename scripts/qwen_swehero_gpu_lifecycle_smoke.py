#!/usr/bin/env python3
"""Run a small GPU lifecycle smoke for the SWE-HERO TorchTitan launcher.

This is a system smoke test, not a model-quality evaluation. Its job is to prove
that the GPU pod can run the same distributed training path used by the real
Qwen/SWE-Hero experiment, write a resumable TorchTitan checkpoint, export model
weights in Hugging Face format, and resume from the immutable run contract.

The default mode uses synthetic already-tokenized records. That deliberately
removes real dataset/tokenization variability so failures point at the lifecycle
itself: distributed launch, bucket routing, context parallel wiring,
checkpointing, final export, and resume validation. The
``--production-acceptance-smoke`` mode switches to a tiny real SWE-Hero subset
and keeps production provenance/W&B gates enabled for final pre-production
confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_OUT_DIR = Path("/workspace/qwen25-coder7b-swehero-lifecycle-smoke")
DEFAULT_HF_ASSETS_PATH = Path("/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct")

# The default lifecycle smoke should be fast and cheap. A 1024-token bucket is
# far below Qwen2.5-Coder's real long-context training setup, but it exercises
# the same TorchTitan stage planner and checkpoint/export code paths without
# allocating enough activations to make the test expensive.
DEFAULT_BUCKET = 1024

# The production acceptance smoke uses real SWE-Hero traces. 32768 tokens is
# Qwen2.5-Coder's native context window, so this still checks real ChatML
# serialization and model execution while avoiding the much more expensive 128k
# long-context YaRN path used by the full direct-to-hero run.
DEFAULT_ACCEPTANCE_BUCKET = 32_768

# Context parallelism splits a long sequence across multiple GPU ranks. Keeping
# the default at 1 avoids adding that distributed dimension to the fast smoke;
# callers can raise it when they specifically need to validate CP wiring.
DEFAULT_CP_DEGREE = 1
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60


class SmokeValidationError(RuntimeError):
    """Raised when the lifecycle smoke output is missing an expected artifact."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing Qwen SWE-HERO TorchTitan launcher on a tiny "
            "GPU workload, then resume the completed run and verify "
            "checkpoint/export validation artifacts. By default this uses "
            "synthetic tokenized buckets; --production-acceptance-smoke uses "
            "a bounded real SWE-HERO subset under the production gate."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--hf-assets-path",
        type=Path,
        default=DEFAULT_HF_ASSETS_PATH,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Optional real SWE-HERO dataset artifact for "
            "--production-acceptance-smoke. Defaults to the launcher dataset."
        ),
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=_repo_root() / "scripts" / "run_qwen_swehero_torchtitan_pod.py",
        help="Launcher wrapper to execute. Defaults to the canonical pod wrapper.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=_env_int("NPROC_PER_NODE", 8),
        help="GPU processes for torchrun. Defaults to NPROC_PER_NODE or 8.",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=None,
        help=(
            f"Sequence bucket. Defaults to {DEFAULT_BUCKET} for synthetic "
            f"lifecycle smoke and {DEFAULT_ACCEPTANCE_BUCKET} for production "
            "acceptance smoke."
        ),
    )
    parser.add_argument("--cp-degree", type=int, default=DEFAULT_CP_DEGREE)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Total optimizer steps for the smoke. One step is enough to cover checkpoint/export.",
    )
    parser.add_argument(
        "--production-acceptance-smoke",
        action="store_true",
        help=(
            "Run a final acceptance smoke with --production-mode enabled and "
            "a tiny real dataset subset instead of synthetic records."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=_env_int("NUM_EXAMPLES", 1),
        help="Accepted real examples for --production-acceptance-smoke.",
    )
    parser.add_argument(
        "--max-streamed-examples",
        type=int,
        default=_env_int("MAX_STREAMED_EXAMPLES", 1),
        help="Raw streamed real examples to inspect for --production-acceptance-smoke.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=_env_int("SHUFFLE_BUFFER", 0),
        help=(
            "Streaming shuffle buffer for --production-acceptance-smoke. "
            "Defaults to 0 so the final smoke does not fill a large buffer "
            "before accepting a tiny subset."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "shared"),
        default=os.environ.get("WANDB_MODE", "online"),
        help="Durable W&B mode for --production-acceptance-smoke.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get(
            "WANDB_RUN_NAME",
            "qwen25-coder7b-swehero-final-acceptance-smoke",
        ),
        help="W&B run name for --production-acceptance-smoke.",
    )
    parser.add_argument(
        "--wandb-run-tags",
        default=os.environ.get(
            "WANDB_RUN_TAGS",
            "direct-to-hero,final-acceptance-smoke",
        ),
        help="Comma-separated W&B tags for --production-acceptance-smoke.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout applied separately to the fresh launch and resume launch.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=20.0,
        help="Smoke-specific disk preflight threshold.",
    )
    parser.add_argument(
        "--min-free-gpu-memory-gb",
        type=float,
        default=20.0,
        help="Smoke-specific per-GPU free-memory preflight threshold.",
    )
    parser.add_argument(
        "--min-free-cpu-memory-gb",
        type=float,
        default=8.0,
        help="Smoke-specific CPU-memory preflight threshold.",
    )
    parser.add_argument(
        "--min-write-throughput-mb-s",
        type=float,
        default=10.0,
        help="Smoke-specific output filesystem write-throughput threshold.",
    )
    parser.add_argument(
        "--write-throughput-probe-mb",
        type=int,
        default=16,
        help="Smaller write probe for the lifecycle smoke.",
    )
    args = parser.parse_args(argv)
    if args.bucket is None:
        args.bucket = (
            DEFAULT_ACCEPTANCE_BUCKET
            if args.production_acceptance_smoke
            else DEFAULT_BUCKET
        )
    return args


def _validate_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.nproc_per_node <= 0:
        errors.append("--nproc-per-node must be positive")
    if args.cp_degree <= 0:
        errors.append("--cp-degree must be positive")
    if args.bucket <= 0:
        errors.append("--bucket must be positive")
    if args.max_steps <= 0:
        errors.append("--max-steps must be positive")
    if args.local_batch_size <= 0:
        errors.append("--local-batch-size must be positive")
    if args.timeout_seconds <= 0:
        errors.append("--timeout-seconds must be positive")
    if args.shuffle_buffer < 0:
        errors.append("--shuffle-buffer must be non-negative")
    if args.production_acceptance_smoke:
        if args.num_examples <= 0:
            errors.append("--num-examples must be positive for production acceptance")
        if args.max_streamed_examples <= 0:
            errors.append(
                "--max-streamed-examples must be positive for production acceptance"
            )
        elif args.max_streamed_examples < args.num_examples:
            errors.append(
                "--max-streamed-examples must be >= --num-examples for production acceptance"
            )
    if args.nproc_per_node > 0 and args.cp_degree > 0:
        # Each context-parallel group must contain an integer number of GPU
        # ranks. The remaining groups are data-parallel replicas that process
        # different training examples, so a non-divisible split would make the
        # global batch calculation and TorchTitan mesh invalid.
        if args.nproc_per_node % args.cp_degree != 0:
            errors.append("--cp-degree must divide --nproc-per-node")
    if errors:
        raise ValueError("Invalid lifecycle smoke inputs:\n" + "\n".join(errors))


def _data_parallel_degree(args: argparse.Namespace) -> int:
    """Return how many independent example replicas remain after CP splitting."""

    return args.nproc_per_node // args.cp_degree


def _global_batch_size(args: argparse.Namespace) -> int:
    """Return examples per optimizer step across all data-parallel replicas."""

    return _data_parallel_degree(args) * args.local_batch_size


def _synthetic_examples_per_bucket(args: argparse.Namespace) -> int:
    """Produce exactly enough synthetic records for the requested optimizer steps."""

    return _global_batch_size(args) * args.max_steps


def _common_launcher_args(args: argparse.Namespace) -> list[str]:
    """Build the shared trainer arguments for both fresh and resume launches."""

    bucket = str(args.bucket)
    command = [
        "--out-dir",
        str(args.out_dir),
        "--hf-assets-path",
        str(args.hf_assets_path),
    ]
    if args.dataset_path is not None:
        command.extend(["--dataset-path", str(args.dataset_path)])
    if args.production_acceptance_smoke:
        # Production acceptance keeps the real-data and provenance gates enabled
        # while bounding the record count. That proves the production launcher
        # can materialize SWE-Hero traces, record durable W&B identity, and still
        # finish quickly enough to be used as a final smoke.
        command.extend(
            [
                "--production-mode",
                "--production-acceptance-smoke",
                "--num-examples",
                str(args.num_examples),
                "--max-streamed-examples",
                str(args.max_streamed_examples),
                "--long-example-policy",
                "skip",
                "--shuffle-buffer",
                str(args.shuffle_buffer),
                "--enable-wandb",
                "--wandb-mode",
                args.wandb_mode,
                "--wandb-run-name",
                args.wandb_run_name,
                "--wandb-run-tags",
                args.wandb_run_tags,
            ]
        )
    else:
        # Synthetic buckets bypass real data and tokenizer work. They are still
        # useful for ML infrastructure because they generate shaped token arrays
        # that flow through the model, loss, optimizer, checkpoint, export, and
        # resume paths exactly like real examples would.
        command.extend(
            [
                "--smoke-synthetic-buckets",
                "--smoke-synthetic-examples-per-bucket",
                str(_synthetic_examples_per_bucket(args)),
            ]
        )
    command.extend(
        [
            # max-length is the model input length cap, while buckets and
            # bucket-cp tell the launcher which fixed sequence bucket and
            # context-parallel degree to test. Using a single bucket makes this
            # lifecycle smoke deterministic: one stage, one checkpoint step, one
            # final export step.
            "--max-length",
            bucket,
            "--buckets",
            bucket,
            "--bucket-cp",
            f"{bucket}:{args.cp_degree}",
            "--bucket-curriculum",
            "single-bucket",
            # TorchTitan's optimizer step sees the global batch across all
            # data-parallel ranks. Keeping local-batch-size explicit makes the
            # relationship between GPUs, CP degree, and examples per step
            # visible in the command the smoke validates.
            "--nproc-per-node",
            str(args.nproc_per_node),
            "--global-batch-size",
            str(_global_batch_size(args)),
            "--local-batch-size",
            str(args.local_batch_size),
            # One epoch plus an explicit max-steps cap makes the smoke's
            # training schedule independent of dataset size. The goal is to
            # execute lifecycle edges, not to improve the model.
            "--num-train-epochs",
            "1",
            "--max-steps",
            str(args.max_steps),
            # A checkpoint every step is intentionally aggressive for the smoke:
            # it proves the first-step checkpoint and final checkpoint paths
            # immediately. Async checkpointing is disabled so validation sees
            # fully written files before the launcher exits.
            "--checkpoint-interval",
            "1",
            "--checkpoint-async-mode",
            "disabled",
            "--validate-first-step-checkpoint",
            "--metrics-log-freq",
            "1",
            "--log-rank",
            "0",
            "--torchrun-log-rank-filter",
            "0",
            # torch.compile and FP8 can be valuable in real training, but they
            # add extra compiler/hardware-specific moving parts. The lifecycle
            # smoke disables both so a failure is more likely to indicate a
            # launcher/checkpoint/resume problem rather than a performance-path
            # issue.
            "--no-compile",
            "--no-enable-fp8",
            "--min-free-disk-gb",
            repr(args.min_free_disk_gb),
            "--min-free-gpu-memory-gb",
            repr(args.min_free_gpu_memory_gb),
            "--min-free-cpu-memory-gb",
            repr(args.min_free_cpu_memory_gb),
            "--min-write-throughput-mb-s",
            repr(args.min_write_throughput_mb_s),
            "--write-throughput-probe-mb",
            str(args.write_throughput_probe_mb),
        ]
    )
    return command


def fresh_launch_command(args: argparse.Namespace) -> list[str]:
    return [str(args.launcher), *_common_launcher_args(args), "--overwrite-output"]


def resume_launch_command(args: argparse.Namespace) -> list[str]:
    return [str(args.launcher), *_common_launcher_args(args), "--resume"]


def _run(command: Sequence[str], *, timeout_seconds: int) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=_repo_root(), timeout=timeout_seconds, check=True)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SmokeValidationError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SmokeValidationError(f"Malformed {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SmokeValidationError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeValidationError(message)


def _require_nonempty_file(path: Path, label: str) -> None:
    _require(path.is_file(), f"Missing {label}: {path}")
    _require(path.stat().st_size > 0, f"{label} is empty: {path}")


def _validate_dcp_checkpoint(out_dir: Path, step: int) -> dict[str, Any]:
    """Validate a Torch Distributed Checkpoint directory for one step.

    DCP checkpoints are resumable training snapshots: they include model weights
    plus distributed training state such as optimizer state and metadata needed
    by TorchTitan. That makes them different from the final Hugging Face export,
    which is meant for inference/loading rather than continuing optimization.
    """

    checkpoint = out_dir / "torchtitan" / "checkpoint" / f"step-{step}"
    _require(checkpoint.is_dir(), f"Missing DCP checkpoint directory: {checkpoint}")
    _require_nonempty_file(checkpoint / ".metadata", "DCP checkpoint metadata")
    payloads = sorted(checkpoint.glob("*.distcp"))
    _require(bool(payloads), f"DCP checkpoint has no .distcp payloads: {checkpoint}")
    for payload in payloads:
        _require_nonempty_file(payload, "DCP checkpoint payload")
    return {
        "path": str(checkpoint),
        "payload_count": len(payloads),
        "payload_total_bytes": sum(path.stat().st_size for path in payloads),
    }


def _validate_final_export(out_dir: Path, step: int) -> dict[str, Any]:
    """Validate the Hugging Face-style model-weight export for one step."""

    export = out_dir / "torchtitan" / "final_export" / f"step-{step}"
    _require(export.is_dir(), f"Missing final export directory: {export}")
    index = _read_json_object(
        export / "model.safetensors.index.json", "final export index"
    )
    weight_map = index.get("weight_map")
    _require(
        isinstance(weight_map, dict) and bool(weight_map),
        f"Final export index has no weight_map entries: {export}",
    )
    shards = sorted({str(value) for value in weight_map.values()})
    for shard in shards:
        _require_nonempty_file(export / shard, "final export shard")
    return {
        "path": str(export),
        "shard_count": len(shards),
    }


def _validate_stage_status(out_dir: Path, max_steps: int) -> dict[str, Any]:
    """Validate the launcher's staged training status document.

    A "stage" is one fixed bucket/parallelism segment of the overall training
    plan. This smoke uses one stage, but the validator accepts
    ``completed_before_resume`` so the second launch can prove resume behavior
    without rerunning already completed optimizer steps.
    """

    status_path = out_dir / "stage_status.json"
    status = _read_json_object(status_path, "stage status")
    stages = status.get("stages")
    _require(isinstance(stages, list) and bool(stages), "Stage status has no stages")

    stage_statuses = []
    attempt_count = 0
    for stage in stages:
        _require(isinstance(stage, Mapping), "Stage status entry is not an object")
        stage_statuses.append(str(stage.get("status")))
        _require(
            stage.get("status") in {"succeeded", "completed_before_resume"},
            f"Unexpected stage status for {stage.get('id')}: {stage.get('status')}",
        )
        attempts = stage.get("attempts")
        _require(isinstance(attempts, list), f"Stage has no attempts list: {stage}")
        attempt_count += len(attempts)
        for attempt in attempts:
            _require(isinstance(attempt, Mapping), "Stage attempt is not an object")
            _require(
                attempt.get("status") == "succeeded",
                f"Unexpected attempt status: {attempt.get('status')}",
            )
            logs = attempt.get("logs")
            _require(isinstance(logs, Mapping), "Stage attempt has no logs record")
            stdout = Path(str(logs.get("stdout")))
            stderr = Path(str(logs.get("stderr")))
            _require_nonempty_file(stdout, "torchrun stdout log")
            _require(stderr.is_file(), f"Missing torchrun stderr log: {stderr}")

    first_step = status.get("first_step_checkpoint_validation")
    _require(
        isinstance(first_step, Mapping) and first_step.get("status") == "succeeded",
        "First-step checkpoint validation did not succeed",
    )
    final_validation = status.get("final_artifact_validation")
    _require(
        isinstance(final_validation, Mapping)
        and final_validation.get("status") == "succeeded",
        "Final artifact validation did not succeed",
    )
    summary = status.get("summary")
    _require(isinstance(summary, Mapping), "Stage status summary is missing")
    _require(
        summary.get("final_artifact_validation_status") == "succeeded",
        "Stage status summary did not record final validation success",
    )
    _require(
        attempt_count > 0,
        "Lifecycle smoke must include at least one completed torchrun attempt",
    )
    return {
        "stage_statuses": stage_statuses,
        "attempt_count": attempt_count,
        "final_validation_status": final_validation.get("status"),
        "total_steps": max_steps,
    }


def _validate_production_acceptance_metadata(out_dir: Path) -> dict[str, Any]:
    """Confirm the acceptance smoke used real production gates, not synthetic data."""

    run_spec = _read_json_object(out_dir / "run_spec.json", "run spec")
    args = run_spec.get("args")
    _require(isinstance(args, Mapping), "Run spec has no args object")
    _require(
        args.get("production_mode") is True, "Run spec did not record production mode"
    )
    _require(
        args.get("production_acceptance_smoke") is True,
        "Run spec did not record production acceptance smoke mode",
    )
    _require(
        args.get("smoke_synthetic_buckets") is False,
        "Production acceptance smoke must not use synthetic buckets",
    )
    _require(
        int(args.get("num_examples") or 0) > 0,
        "Production acceptance smoke must record a bounded real subset",
    )

    manifest = _read_json_object(out_dir / "data" / "manifest.json", "data manifest")
    provenance = manifest.get("data_provenance")
    _require(isinstance(provenance, Mapping), "Data manifest has no provenance")
    materialization = provenance.get("materialization")
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("smoke_synthetic_buckets") is False,
        "Data manifest must record real dataset materialization",
    )
    dataset = provenance.get("dataset")
    dataset_artifact = (
        dataset.get("dataset_artifact") if isinstance(dataset, Mapping) else None
    )
    _require(
        isinstance(dataset_artifact, Mapping)
        and dataset_artifact.get("synthetic_smoke") is not True,
        "Data manifest dataset artifact must be real, not synthetic",
    )
    included = provenance.get("included")
    included_count = (
        int(included.get("count") or 0) if isinstance(included, Mapping) else 0
    )
    _require(included_count > 0, "Data manifest did not include real records")

    # Durable W&B identity is part of the production gate because production
    # experiments need auditable run identity across fresh launch and resume.
    # Offline/disabled modes are fine for local tests, but they are not enough
    # proof for this final acceptance path.
    wandb_identity = _read_json_object(
        out_dir / "wandb_identity.json",
        "W&B identity",
    )
    _require(wandb_identity.get("enabled") is True, "W&B identity is not enabled")
    _require(
        wandb_identity.get("mode") not in {"offline", "disabled"},
        "W&B identity is not durable",
    )
    _require(bool(wandb_identity.get("run_id")), "W&B identity has no run id")

    structured_logs = sorted(
        (out_dir / "torchtitan" / "structured_logs").glob("*.jsonl")
    )
    _require(bool(structured_logs), "Missing TorchTitan structured JSONL logs")
    for log in structured_logs:
        _require_nonempty_file(log, "TorchTitan structured JSONL log")

    _require_nonempty_file(out_dir / "run_spec.sha256", "run spec checksum")
    _require_nonempty_file(out_dir / "runtime_metadata.json", "runtime metadata")
    _require_nonempty_file(out_dir / "launcher_plan.json", "launcher plan")
    _require_nonempty_file(out_dir / "resume_contract.json", "resume contract")

    return {
        "run_spec": {
            "production_mode": args.get("production_mode"),
            "production_acceptance_smoke": args.get("production_acceptance_smoke"),
            "num_examples": args.get("num_examples"),
            "max_streamed_examples": args.get("max_streamed_examples"),
        },
        "data_manifest": {
            "included_count": included_count,
            "synthetic_smoke": dataset_artifact.get("synthetic_smoke", False),
        },
        "wandb": {
            "project": wandb_identity.get("project"),
            "entity": wandb_identity.get("entity"),
            "run_name": wandb_identity.get("run_name"),
            "run_id": wandb_identity.get("run_id"),
            "mode": wandb_identity.get("mode"),
        },
        "structured_logs": {
            "count": len(structured_logs),
            "total_bytes": sum(path.stat().st_size for path in structured_logs),
        },
    }


def validate_smoke_outputs(
    out_dir: Path,
    *,
    max_steps: int,
    require_production_acceptance: bool = False,
) -> dict[str, Any]:
    """Validate the artifacts that prove the training lifecycle completed."""

    first_step_report = _read_json_object(
        out_dir / "first_step_checkpoint_validation.json",
        "first-step checkpoint validation report",
    )
    _require(
        first_step_report.get("step") == 1,
        "First-step checkpoint validation report must describe step 1",
    )
    final_report = _read_json_object(
        out_dir / "final_artifact_validation.json",
        "final artifact validation report",
    )
    final_export_report = final_report.get("final_export")
    _require(
        final_report.get("plan_total_steps") == max_steps,
        f"Final artifact validation must describe plan total step {max_steps}",
    )
    _require(
        isinstance(final_export_report, Mapping)
        and final_export_report.get("step") == max_steps,
        f"Final export validation must describe step {max_steps}",
    )
    resumable = final_report.get("resumable_checkpoints")
    _require(
        isinstance(resumable, Mapping)
        and max_steps in list(resumable.get("steps") or []),
        f"Final artifact validation does not include checkpoint step {max_steps}",
    )

    # Validate both artifact families. The DCP checkpoint proves the run can
    # resume training; the final export proves the trained weights can be loaded
    # by downstream Hugging Face-compatible tooling.
    summary = {
        "first_step_validation": {
            "path": str(out_dir / "first_step_checkpoint_validation.json"),
            "step": first_step_report.get("step"),
        },
        "dcp_checkpoint": _validate_dcp_checkpoint(out_dir, max_steps),
        "final_export": _validate_final_export(out_dir, max_steps),
        "stage_status": _validate_stage_status(out_dir, max_steps),
    }
    if require_production_acceptance:
        summary["production_acceptance"] = _validate_production_acceptance_metadata(
            out_dir
        )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)

    # First run: start from scratch, force output overwrite, and require the
    # launcher to produce all lifecycle artifacts.
    _run(fresh_launch_command(args), timeout_seconds=args.timeout_seconds)
    fresh_summary = validate_smoke_outputs(
        args.out_dir,
        max_steps=args.max_steps,
        require_production_acceptance=args.production_acceptance_smoke,
    )
    print(json.dumps({"fresh_smoke_validation": fresh_summary}, indent=2), flush=True)

    # Second run: use the same immutable run spec with --resume. A successful
    # validation here proves the launcher can recognize completed stages and
    # still verify final artifacts instead of accidentally starting a new run.
    _run(resume_launch_command(args), timeout_seconds=args.timeout_seconds)
    resume_summary = validate_smoke_outputs(
        args.out_dir,
        max_steps=args.max_steps,
        require_production_acceptance=args.production_acceptance_smoke,
    )
    print(json.dumps({"resume_smoke_validation": resume_summary}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (
        SmokeValidationError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"Lifecycle smoke failed: {exc}", file=sys.stderr)
        raise
