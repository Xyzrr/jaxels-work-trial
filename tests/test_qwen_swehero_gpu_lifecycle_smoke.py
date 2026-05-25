import json
import tempfile
from pathlib import Path

import pytest

from scripts import qwen_swehero_gpu_lifecycle_smoke as smoke


class TestQwenSweHeroGpuLifecycleSmoke:
    def _flag_value(self, command: list[str], flag: str) -> str:
        index = command.index(flag)
        return command[index + 1]

    def test_fresh_and_resume_commands_cover_lifecycle_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = smoke.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--launcher",
                    str(Path(tmp) / "launcher.sh"),
                    "--nproc-per-node",
                    "4",
                    "--cp-degree",
                    "2",
                    "--bucket",
                    "2048",
                    "--max-steps",
                    "3",
                ]
            )

        smoke._validate_args(args)
        fresh = smoke.fresh_launch_command(args)
        resume = smoke.resume_launch_command(args)

        assert "--overwrite-output" in fresh
        assert "--resume" not in fresh
        assert "--resume" in resume
        assert "--overwrite-output" not in resume
        for command in (fresh, resume):
            assert "--smoke-synthetic-buckets" in command
            assert "--validate-first-step-checkpoint" in command
            assert "--no-compile" in command
            assert "--no-enable-fp8" in command
            assert self._flag_value(command, "--bucket-cp") == "2048:2"
            assert self._flag_value(command, "--bucket-curriculum") == "single-bucket"
            assert self._flag_value(command, "--checkpoint-interval") == "1"
            assert self._flag_value(command, "--checkpoint-async-mode") == "disabled"
            assert self._flag_value(command, "--global-batch-size") == "2"
            assert (
                self._flag_value(command, "--smoke-synthetic-examples-per-bucket")
                == "6"
            )

    def test_production_acceptance_commands_use_real_data_and_production_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = smoke.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--launcher",
                    str(Path(tmp) / "launcher.sh"),
                    "--nproc-per-node",
                    "8",
                    "--production-acceptance-smoke",
                    "--num-examples",
                    "8",
                    "--max-streamed-examples",
                    "64",
                    "--wandb-mode",
                    "online",
                ]
            )

        smoke._validate_args(args)
        fresh = smoke.fresh_launch_command(args)
        resume = smoke.resume_launch_command(args)

        for command in (fresh, resume):
            assert "--production-mode" in command
            assert "--production-acceptance-smoke" in command
            assert "--enable-wandb" in command
            assert "--smoke-synthetic-buckets" not in command
            assert self._flag_value(command, "--dataset-path") == str(
                Path(tmp) / "dataset"
            )
            assert self._flag_value(command, "--num-examples") == "8"
            assert self._flag_value(command, "--max-streamed-examples") == "64"
            assert self._flag_value(command, "--long-example-policy") == "skip"
            assert self._flag_value(command, "--shuffle-buffer") == "0"
            assert self._flag_value(command, "--max-length") == "32768"
            assert self._flag_value(command, "--wandb-mode") == "online"

    def _write_minimal_smoke_outputs(self, out_dir: Path, *, step: int) -> None:
        checkpoint = out_dir / "torchtitan" / "checkpoint" / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".metadata").write_bytes(b"metadata")
        (checkpoint / "__0_0.distcp").write_bytes(b"payload")

        export = out_dir / "torchtitan" / "final_export" / f"step-{step}"
        export.mkdir(parents=True)
        (export / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        (export / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 7},
                    "weight_map": {
                        "lm_head.weight": "model-00001-of-00001.safetensors"
                    },
                }
            )
        )

        (out_dir / "first_step_checkpoint_validation.json").write_text(
            json.dumps({"step": 1})
        )
        (out_dir / "final_artifact_validation.json").write_text(
            json.dumps(
                {
                    "plan_total_steps": step,
                    "resumable_checkpoints": {"steps": [step]},
                    "final_export": {"step": step},
                }
            )
        )

        log_dir = out_dir / "torchrun_logs"
        log_dir.mkdir()
        stdout = log_dir / "stage-01-bucket-1024-step-1-attempt-01.stdout.log"
        stderr = log_dir / "stage-01-bucket-1024-step-1-attempt-01.stderr.log"
        stdout.write_text("torchrun output\n")
        stderr.write_text("")
        (out_dir / "stage_status.json").write_text(
            json.dumps(
                {
                    "stages": [
                        {
                            "id": "stage-01-bucket-1024-step-1",
                            "status": "succeeded",
                            "attempts": [
                                {
                                    "status": "succeeded",
                                    "logs": {
                                        "stdout": str(stdout),
                                        "stderr": str(stderr),
                                    },
                                }
                            ],
                        }
                    ],
                    "first_step_checkpoint_validation": {"status": "succeeded"},
                    "final_artifact_validation": {"status": "succeeded"},
                    "summary": {"final_artifact_validation_status": "succeeded"},
                }
            )
        )

    def test_validate_smoke_outputs_accepts_lifecycle_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self._write_minimal_smoke_outputs(out_dir, step=1)

            summary = smoke.validate_smoke_outputs(out_dir, max_steps=1)

        assert summary["first_step_validation"]["step"] == 1
        assert summary["dcp_checkpoint"]["payload_count"] == 1
        assert summary["final_export"]["shard_count"] == 1
        assert summary["stage_status"]["attempt_count"] == 1

    def test_validate_smoke_outputs_accepts_production_acceptance_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self._write_minimal_smoke_outputs(out_dir, step=1)
            (out_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "args": {
                            "production_mode": True,
                            "production_acceptance_smoke": True,
                            "smoke_synthetic_buckets": False,
                            "num_examples": 8,
                            "max_streamed_examples": 64,
                        }
                    }
                )
            )
            (out_dir / "run_spec.sha256").write_text("abc123\n")
            (out_dir / "runtime_metadata.json").write_text("{}")
            (out_dir / "launcher_plan.json").write_text("{}")
            (out_dir / "resume_contract.json").write_text("{}")
            data_dir = out_dir / "data"
            data_dir.mkdir()
            (data_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "data_provenance": {
                            "materialization": {
                                "smoke_synthetic_buckets": False,
                            },
                            "dataset": {
                                "dataset_artifact": {
                                    "synthetic_smoke": False,
                                }
                            },
                            "included": {"count": 8},
                        }
                    }
                )
            )
            (out_dir / "wandb_identity.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "mode": "online",
                        "project": "proj",
                        "entity": "team",
                        "run_name": "run",
                        "run_id": "abc123",
                    }
                )
            )
            structured = out_dir / "torchtitan" / "structured_logs"
            structured.mkdir(parents=True)
            (structured / "training.global_rank_0.jsonl").write_text("{}\n")

            summary = smoke.validate_smoke_outputs(
                out_dir,
                max_steps=1,
                require_production_acceptance=True,
            )

        assert summary["production_acceptance"]["data_manifest"]["included_count"] == 8
        assert summary["production_acceptance"]["structured_logs"]["count"] == 1

    def test_validate_smoke_outputs_rejects_missing_first_step_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self._write_minimal_smoke_outputs(out_dir, step=1)
            (out_dir / "first_step_checkpoint_validation.json").unlink()

            with pytest.raises(
                smoke.SmokeValidationError, match="first-step checkpoint validation"
            ):
                smoke.validate_smoke_outputs(out_dir, max_steps=1)
