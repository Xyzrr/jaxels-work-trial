import ast
import contextlib
import io
import json
import os
import signal
import shlex
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import qwen_swehero_train as train


class FakeTokenizer:
    bos_id = None
    eos_id = None
    pad_id = 0

    def encode(self, text, **kwargs):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class QwenSweHeroTorchTitanLauncherTests(unittest.TestCase):
    def _resume_test_setup(self, tmp: str):
        out_dir = Path(tmp) / "run"
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True)
        args = train.parse_args(
            [
                "--out-dir",
                str(out_dir),
                "--dataset-path",
                str(Path(tmp) / "dataset"),
                "--hf-assets-path",
                str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                "--buckets",
                "8192,32768",
                "--bucket-cp",
                "8192:1,32768:2",
                "--max-length",
                "32768",
                "--num-examples",
                "34",
                "--max-streamed-examples",
                "100",
            ]
        )
        args.buckets = ",".join(str(b) for b in train.parse_bucket_list(args.buckets))
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
        args.bucket_cp = train._format_bucket_cp_map(bucket_cp)
        bucket_files = {
            8192: data_dir / "bucket_8192.jsonl",
            32768: data_dir / "bucket_32768.jsonl",
        }
        for path in bucket_files.values():
            path.write_text("")
        manifest = {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_id": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "dataset_artifact": {
                "path": str(args.dataset_path),
                "metadata_json_sha256": "metadata-sha",
                "selection_manifest_sha256": "selection-sha",
                "data_files": [],
                "total_data_bytes": 0,
            },
            "source_dataset_id": args.source_dataset_id,
            "source_dataset_revision": {
                "requested_revision": args.source_dataset_revision,
                "resolved_sha": "source-sha",
            },
            "model_assets": {
                "schema_version": train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "hf_assets_path": str(args.hf_assets_path),
                "hf_assets_realpath": str(args.hf_assets_path),
                "file_count": 1,
                "total_bytes": 10,
                "files": [
                    {
                        "path": "config.json",
                        "kind": "model_config",
                        "bytes": 10,
                        "sha256": "config-sha",
                    }
                ],
                "config": {
                    "path": "config.json",
                    "sha256": "config-sha",
                    "summary": {"model_type": "qwen2"},
                    "json_error": None,
                },
                "generation_config": {
                    "path": None,
                    "sha256": None,
                    "summary": {},
                    "json_error": None,
                },
                "safetensors": {
                    "index_path": None,
                    "index_sha256": None,
                    "metadata": {},
                    "weight_map_entries": 0,
                    "shard_files": [],
                    "unindexed_safetensors_files": [],
                    "index_error": None,
                },
                "tokenizer": {
                    "hf_assets_path": str(args.hf_assets_path),
                    "tokenizer_json_sha256": "tokenizer-sha",
                    "tokenizer_config_sha256": "tokenizer-config-sha",
                },
            },
            "tokenizer": {
                "hf_assets_path": str(args.hf_assets_path),
                "tokenizer_json_sha256": "tokenizer-sha",
                "tokenizer_config_sha256": "tokenizer-config-sha",
                "chat_template_sha256": "chat-template-sha",
                "bos_id": None,
                "eos_id": 151645,
                "pad_id": 151643,
                "trace_serializer": "test serializer",
            },
            "pad_token_id": 151643,
            "max_length": args.max_length,
            "buckets": [8192, 32768],
            "bucket_files": {
                str(bucket): str(path) for bucket, path in bucket_files.items()
            },
            "bucket_counts": {"8192": 33, "32768": 1},
            "num_usable_examples": 34,
            "streamed_examples_scanned": 34,
            "skipped": {},
            "include_model_patch": args.include_model_patch,
        }
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        plan = train.build_bucket_plan(
            bucket_counts={8192: 33, 32768: 1},
            bucket_files=bucket_files,
            bucket_cp=bucket_cp,
            epochs=args.num_train_epochs,
            global_batch_size=args.global_batch_size,
            warmup_ratio=args.warmup_ratio,
        )
        return args, manifest, plan

    def _materialize_with_fake_runtime(self, args, examples=(), *, synthetic=False):
        fake_tokenizer_module = types.ModuleType("torchtitan.components.tokenizer")

        class FakeHuggingFaceTokenizer(FakeTokenizer):
            def __init__(self, tokenizer_path):
                self.tokenizer_path = tokenizer_path

        fake_tokenizer_module.HuggingFaceTokenizer = FakeHuggingFaceTokenizer

        with (
            patch.dict(
                sys.modules,
                {
                    "torchtitan": types.ModuleType("torchtitan"),
                    "torchtitan.components": types.ModuleType(
                        "torchtitan.components"
                    ),
                    "torchtitan.components.tokenizer": fake_tokenizer_module,
                },
            ),
            patch.object(train, "load_training_dataset", return_value=iter(examples)),
            patch.object(
                train,
                "_dataset_artifact_metadata",
                return_value={"path": str(args.dataset_path), "data_files": []},
            ),
            patch.object(
                train,
                "_dataset_revision_info",
                return_value={"requested_revision": args.source_dataset_revision},
            ),
            patch.object(
                train,
                "_tokenizer_metadata",
                return_value={
                    "hf_assets_path": str(args.hf_assets_path),
                    "pad_id": 0,
                },
            ),
            patch.object(
                train,
                "_model_asset_provenance",
                return_value={
                    "schema_version": train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "hf_assets_path": str(args.hf_assets_path),
                    "hf_assets_realpath": str(args.hf_assets_path),
                    "file_count": 1,
                    "total_bytes": 10,
                    "files": [
                        {
                            "path": "config.json",
                            "kind": "model_config",
                            "bytes": 10,
                            "sha256": "config-sha",
                        }
                    ],
                    "config": {
                        "path": "config.json",
                        "sha256": "config-sha",
                        "summary": {"model_type": "qwen2"},
                        "json_error": None,
                    },
                    "generation_config": {
                        "path": None,
                        "sha256": None,
                        "summary": {},
                        "json_error": None,
                    },
                    "safetensors": {
                        "index_path": None,
                        "index_sha256": None,
                        "metadata": {},
                        "weight_map_entries": 0,
                        "shard_files": [],
                        "unindexed_safetensors_files": [],
                        "index_error": None,
                    },
                    "tokenizer": {
                        "hf_assets_path": str(args.hf_assets_path),
                        "pad_id": 0,
                    },
                },
            ),
            patch.object(train, "_package_versions", return_value={}),
            patch.object(train, "_run_git", return_value=None),
        ):
            if synthetic:
                return train.materialize_synthetic_smoke_buckets(args)
            return train.materialize_training_buckets(args)

    def _write_preflight_hf_assets(self, hf_assets: Path) -> None:
        hf_assets.mkdir(parents=True, exist_ok=True)
        (hf_assets / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        )
        (hf_assets / "tokenizer.json").write_text('{"version":"1.0"}')
        (hf_assets / "tokenizer_config.json").write_text("{}")
        (hf_assets / "model-00001-of-00001.safetensors").write_bytes(b"shard")
        (hf_assets / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 5},
                    "weight_map": {
                        "model.embed_tokens.weight": (
                            "model-00001-of-00001.safetensors"
                        )
                    },
                }
            )
        )

    def _model_assets_manifest(self, args) -> dict:
        tokenizer_metadata = {
            "hf_assets_path": str(args.hf_assets_path),
            "tokenizer_json_sha256": train._hash_file(
                args.hf_assets_path / "tokenizer.json"
            ),
            "tokenizer_config_sha256": train._hash_file(
                args.hf_assets_path / "tokenizer_config.json"
            ),
        }
        return {
            "model_assets": train._model_asset_provenance(
                model_id=args.model_id,
                model_revision=args.model_revision,
                hf_assets_path=args.hf_assets_path,
                tokenizer_metadata=tokenizer_metadata,
            )
        }

    def _write_dcp_checkpoint(self, out_dir: Path, step: int) -> Path:
        step_dir = train._checkpoint_dir(out_dir) / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / ".metadata").write_bytes(b"metadata")
        (step_dir / "__0_0.distcp").write_bytes(b"dcp-payload")
        return step_dir

    def _write_first_step_checkpoint_validation_report(self, out_dir: Path) -> dict:
        checkpoint = train._validate_dcp_checkpoint_step(
            self._write_dcp_checkpoint(out_dir, step=1)
        )
        report = {
            "schema_version": train.FIRST_STEP_CHECKPOINT_VALIDATION_SCHEMA_VERSION,
            "created_at_unix": 1.0,
            "step": 1,
            "checkpoint": checkpoint,
        }
        train._first_step_checkpoint_validation_path(out_dir).write_text(
            json.dumps(report, indent=2)
        )
        return report

    def _write_final_export(
        self,
        out_dir: Path,
        step: int,
        *,
        legacy: bool = False,
    ) -> Path:
        root = (
            train._checkpoint_dir(out_dir)
            if legacy
            else train._final_model_export_dir(out_dir)
        )
        step_dir = root / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
        (step_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
        (step_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": len(b"shard-1") + len(b"shard-2")},
                    "weight_map": {
                        "lm_head.weight": "model-00002-of-00002.safetensors",
                        "model.embed_tokens.weight": (
                            "model-00001-of-00002.safetensors"
                        ),
                    },
                }
            )
        )
        return step_dir

    def _validate_launch_args(self, extra_args: list[str]):
        args = train.parse_args(
            [
                "--buckets",
                "256",
                "--bucket-cp",
                "256:1",
                "--max-length",
                "256",
                *extra_args,
            ]
        )
        args.buckets = ",".join(str(b) for b in train.parse_bucket_list(args.buckets))
        buckets = train.parse_bucket_list(args.buckets)
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
        args.bucket_cp = train._format_bucket_cp_map(bucket_cp)
        train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
        train.validate_bucket_config(
            buckets=buckets,
            bucket_cp=bucket_cp,
            nproc_per_node=args.nproc_per_node,
            attention_backend=args.attention_backend,
        )
        return args

    def _validate_default_production_launch_args(
        self,
        extra_args: list[str] | None = None,
    ):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            train,
            "_detected_workspace_root",
            return_value=train.CANONICAL_WORKSPACE_ROOT,
        ):
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-mode",
                    *(extra_args or []),
                ]
            )
            args.buckets = ",".join(
                str(b) for b in train.parse_bucket_list(args.buckets)
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
            args.bucket_cp = train._format_bucket_cp_map(bucket_cp)
            train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
            train.validate_bucket_config(
                buckets=buckets,
                bucket_cp=bucket_cp,
                nproc_per_node=args.nproc_per_node,
                attention_backend=args.attention_backend,
            )
            return args

    def test_defaults_track_paper_hyperparameters_and_target_pod(self):
        args = train.parse_args([])
        buckets = train.parse_bucket_list(args.buckets)
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
        train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
        train.validate_bucket_config(
            buckets=buckets,
            bucket_cp=bucket_cp,
            nproc_per_node=args.nproc_per_node,
            attention_backend=args.attention_backend,
        )

        self.assertEqual(args.model_id, train.MODEL_ID)
        self.assertEqual(args.model_revision, train.MODEL_REVISION)
        self.assertEqual(args.dataset_id, train.DATASET_ID)
        self.assertEqual(args.dataset_path, train.DEFAULT_DATASET_PATH)
        self.assertEqual(args.source_dataset_id, train.SOURCE_DATASET_ID)
        self.assertEqual(args.source_dataset_revision, train.SOURCE_DATASET_REVISION)
        self.assertEqual(args.num_examples, 0)
        self.assertEqual(args.max_streamed_examples, 0)
        self.assertTrue(args.build_dataset_if_missing)
        self.assertEqual(args.max_length, train.PAPER_CONTEXT_LENGTH)
        self.assertEqual(args.num_train_epochs, 3.0)
        self.assertEqual(args.global_batch_size, 32)
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.min_learning_rate, 1e-8)
        self.assertEqual(args.warmup_ratio, 0.1)
        self.assertEqual(args.nproc_per_node, 8)
        self.assertEqual(args.nnodes, 1)
        self.assertEqual(args.node_rank, 0)
        self.assertEqual(args.rdzv_backend, "c10d")
        self.assertEqual(args.rdzv_endpoint, "localhost:0")
        self.assertEqual(args.rdzv_id, "")
        self.assertTrue(args.enable_fp8)
        self.assertEqual(args.attention_backend, "sdpa")
        self.assertEqual(args.optimizer_impl, "foreach")
        self.assertEqual(args.training_dtype, "float32")
        self.assertEqual(args.mixed_precision_param_dtype, "bfloat16")
        self.assertEqual(args.mixed_precision_reduce_dtype, "bfloat16")
        self.assertEqual(args.fsdp_reshard_after_forward, "never")
        self.assertFalse(args.production_mode)
        self.assertFalse(args.detect_anomaly)
        self.assertEqual(args.cuda_device_max_connections, "1")
        self.assertEqual(args.torch_nccl_async_error_handling, "1")
        self.assertEqual(args.bucket_curriculum, train.DEFAULT_BUCKET_CURRICULUM)
        self.assertFalse(args.enable_profiler)
        self.assertEqual(args.profiler_freq, 10)
        self.assertEqual(args.profiler_active, 1)
        self.assertEqual(args.profiler_warmup, 3)
        self.assertFalse(args.enable_memory_snapshot)
        self.assertEqual(train.parse_bucket_list(args.buckets), train.DEFAULT_BUCKETS)
        self.assertTrue(args.validate_first_step_checkpoint)
        self.assertEqual(args.workspace_root, train._detected_workspace_root())

    def test_production_mode_accepts_full_default_training_recipe(self):
        args = self._validate_default_production_launch_args()

        self.assertTrue(args.production_mode)
        self.assertEqual(args.num_examples, 0)
        self.assertEqual(args.max_streamed_examples, 0)
        self.assertEqual(args.max_steps, 0)
        self.assertEqual(args.max_length, train.PAPER_CONTEXT_LENGTH)
        self.assertTrue(args.validate_first_step_checkpoint)
        self.assertEqual(args.workspace_root, train.CANONICAL_WORKSPACE_ROOT)

    def test_production_mode_requires_canonical_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            train,
            "_detected_workspace_root",
            return_value=Path(tmp) / "repo",
        ):
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-mode",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
            with self.assertRaisesRegex(ValueError, "canonical workspace root"):
                train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)

    def test_detected_workspace_root_prefers_canonical_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            physical_root = Path(tmp) / "home" / "jaxels-work-trial"
            script_path = physical_root / "scripts" / "qwen_swehero_train.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text("# launcher\n")
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            canonical_root = workspace_dir / "jaxels-work-trial"
            canonical_root.symlink_to(physical_root, target_is_directory=True)

            with (
                patch.object(train, "__file__", str(script_path)),
                patch.object(train, "CANONICAL_WORKSPACE_ROOT", canonical_root),
            ):
                detected = train._detected_workspace_root()

        self.assertEqual(detected, canonical_root)

    def test_production_mode_rejects_smoke_and_subset_controls(self):
        cases = [
            (["--dry-run"], "--dry-run"),
            (["--prepare-data-only"], "--prepare-data-only"),
            (["--skip-data-prep"], "--skip-data-prep"),
            (["--smoke-synthetic-buckets"], "--smoke-synthetic-buckets"),
            (["--num-examples", "1"], "--num-examples=0"),
            (["--max-streamed-examples", "10"], "--max-streamed-examples=0"),
            (["--max-steps", "1"], "--max-steps=0"),
            (
                ["--no-validate-first-step-checkpoint"],
                "--validate-first-step-checkpoint=True",
            ),
        ]
        for extra_args, message in cases:
            with self.subTest(extra_args=extra_args):
                with self.assertRaisesRegex(ValueError, message):
                    self._validate_default_production_launch_args(extra_args)

    def test_production_mode_rejects_non_production_training_recipe(self):
        cases = [
            (["--max-length", "32768"], "--max-length=131072"),
            (["--buckets", "131072", "--bucket-cp", "131072:8"], "--buckets"),
            (
                [
                    "--bucket-cp",
                    "8192:1,16384:1,32768:1,65536:4,131072:8",
                ],
                "--bucket-cp",
            ),
            (["--bucket-curriculum", "long-to-short"], "--bucket-curriculum"),
            (["--num-train-epochs", "1"], "--num-train-epochs=3.0"),
            (["--global-batch-size", "8"], "--global-batch-size=32"),
            (["--local-batch-size", "2"], "--local-batch-size=1"),
            (["--learning-rate", "2e-5"], "--learning-rate=1e-05"),
            (["--min-learning-rate", "0"], "--min-learning-rate=1e-08"),
            (["--warmup-ratio", "0"], "--warmup-ratio=0.1"),
            (["--weight-decay", "0.1"], "--weight-decay=0.0"),
            (["--long-example-policy", "skip"], "--long-example-policy='error'"),
            (["--include-model-patch"], "--include-model-patch=False"),
            (["--min-trainable-tokens", "2"], "--min-trainable-tokens=1"),
            (["--source-dataset-revision", "main"], "--source-dataset-revision"),
        ]
        for extra_args, message in cases:
            with self.subTest(extra_args=extra_args):
                with self.assertRaisesRegex(ValueError, message):
                    self._validate_default_production_launch_args(extra_args)

    def test_launch_env_file_sets_defaults_before_full_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "NUM_EXAMPLES=7",
                        "MAX_STREAMED_EXAMPLES=11",
                        "export SWEHERO_BUCKETS=1024",
                        "SWEHERO_BUCKET_CP=1024:1",
                        "ENABLE_FP8=0 # disable for test",
                        "WANDB_RUN_NAME='dotenv-run'",
                    ]
                )
            )
            argv = ["--env-file", str(env_file)]

            with patch.dict(os.environ, {}, clear=True):
                loaded_env_file = train.load_launch_env_file(argv)
                args = train.parse_args(argv, env_file_default=loaded_env_file)

        self.assertEqual(args.env_file, str(env_file))
        self.assertEqual(args.num_examples, 7)
        self.assertEqual(args.max_streamed_examples, 11)
        self.assertEqual(args.buckets, "1024")
        self.assertEqual(args.bucket_cp, "1024:1")
        self.assertFalse(args.enable_fp8)
        self.assertEqual(args.wandb_run_name, "dotenv-run")

    def test_cli_flags_override_launch_env_file_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("NUM_EXAMPLES=7\nENABLE_FP8=0\n")
            argv = [
                "--env-file",
                str(env_file),
                "--num-examples",
                "3",
                "--enable-fp8",
            ]

            with patch.dict(os.environ, {}, clear=True):
                loaded_env_file = train.load_launch_env_file(argv)
                args = train.parse_args(argv, env_file_default=loaded_env_file)

        self.assertEqual(args.num_examples, 3)
        self.assertTrue(args.enable_fp8)

    def test_launch_argfile_supports_comments_quoting_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "configured run"
            argfile = Path(tmp) / "launch.args"
            argfile.write_text(
                "\n".join(
                    [
                        "# reviewed production launch flags",
                        f"--out-dir {shlex.quote(str(out_dir))}",
                        "--buckets 1024",
                        "--bucket-cp 1024:1",
                        "--max-length 1024",
                        "--num-examples 4",
                        "--no-enable-fp8",
                    ]
                )
            )

            args = train.parse_args([f"@{argfile}", "--num-examples", "7"])

        self.assertEqual(args.out_dir, out_dir)
        self.assertEqual(args.buckets, "1024")
        self.assertEqual(args.bucket_cp, "1024:1")
        self.assertEqual(args.max_length, 1024)
        self.assertEqual(args.num_examples, 7)
        self.assertFalse(args.enable_fp8)

    def test_process_env_overrides_launch_env_file_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("NUM_EXAMPLES=7\n")
            argv = ["--env-file", str(env_file)]

            with patch.dict(os.environ, {"NUM_EXAMPLES": "13"}, clear=True):
                loaded_env_file = train.load_launch_env_file(argv)
                args = train.parse_args(argv, env_file_default=loaded_env_file)

        self.assertEqual(args.num_examples, 13)

    def test_env_numeric_and_boolean_values_are_strict(self):
        cases = [
            ({"ENABLE_FP8": "maybe"}, "ENABLE_FP8 must be a boolean"),
            ({"PRODUCTION_MODE": "maybe"}, "PRODUCTION_MODE must be a boolean"),
            (
                {"VALIDATE_FIRST_STEP_CHECKPOINT": "maybe"},
                "VALIDATE_FIRST_STEP_CHECKPOINT must be a boolean",
            ),
            ({"NUM_EXAMPLES": "abc"}, "NUM_EXAMPLES must be an integer"),
            ({"PROFILER_REPEAT": "abc"}, "PROFILER_REPEAT must be an integer"),
            ({"LEARNING_RATE": "nan"}, "LEARNING_RATE must be finite"),
        ]
        for env, message in cases:
            with self.subTest(env=env):
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(ValueError, message):
                        train.parse_args([])

    def test_launch_input_validation_rejects_bad_numeric_values(self):
        cases = [
            (
                ["--model-id", "Qwen/Qwen2.5-Coder-14B-Instruct"],
                "--model-id must be",
            ),
            (["--model-revision", "main"], "--model-revision must be an exact"),
            (["--model-revision", "0" * 40], "--model-revision must be the pinned"),
            (["--source-dataset-rows-per-shard", "0"], "--source-dataset-rows-per-shard"),
            (["--source-dataset-build-batch-size", "0"], "--source-dataset-build-batch-size"),
            (["--num-examples", "-1"], "--num-examples"),
            (["--max-streamed-examples", "-1"], "--max-streamed-examples"),
            (["--shuffle-buffer", "-1"], "--shuffle-buffer"),
            (["--seed", "-1"], "--seed"),
            (["--max-length", "0"], "--max-length"),
            (["--min-trainable-tokens", "0"], "--min-trainable-tokens"),
            (["--num-train-epochs", "0"], "--num-train-epochs"),
            (["--max-steps", "-1"], "--max-steps"),
            (["--global-batch-size", "0"], "--global-batch-size"),
            (["--local-batch-size", "0"], "--local-batch-size"),
            (["--learning-rate", "0"], "--learning-rate"),
            (["--min-learning-rate", "-1"], "--min-learning-rate"),
            (["--warmup-ratio", "-0.1"], "--warmup-ratio"),
            (["--warmup-ratio", "1.1"], "--warmup-ratio"),
            (["--weight-decay", "-0.1"], "--weight-decay"),
            (["--max-grad-norm", "0"], "--max-grad-norm"),
            (["--cuda-device-max-connections", "0"], "--cuda-device-max-connections"),
            (
                ["--torch-nccl-async-error-handling", ""],
                "--torch-nccl-async-error-handling",
            ),
            (["--chunked-ce-chunks", "0"], "--chunked-ce-chunks"),
            (["--checkpoint-interval", "0"], "--checkpoint-interval"),
            (["--metrics-log-freq", "0"], "--metrics-log-freq"),
            (["--profiler-freq", "0"], "--profiler-freq"),
            (["--profiler-active", "0"], "--profiler-active"),
            (["--profiler-warmup", "-1"], "--profiler-warmup"),
            (["--profiler-repeat", "0"], "--profiler-repeat"),
            (["--profiler-skip-first", "-1"], "--profiler-skip-first"),
            (
                ["--profiler-skip-first-wait", "-1"],
                "--profiler-skip-first-wait",
            ),
            (
                [
                    "--profiler-freq",
                    "3",
                    "--profiler-active",
                    "1",
                    "--profiler-warmup",
                    "3",
                ],
                "--profiler-freq must be greater",
            ),
            (["--nproc-per-node", "0"], "--nproc-per-node"),
            (["--nnodes", "0"], "--nnodes"),
            (["--node-rank", "-1"], "--node-rank"),
            (["--node-rank", "1"], "--node-rank must be 0"),
            (
                [
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "2",
                    "--rdzv-endpoint",
                    "train-master:29400",
                    "--rdzv-id",
                    "run",
                ],
                "--node-rank must be less than --nnodes",
            ),
            (["--nnodes", "2"], "--rdzv-id is required"),
            (
                ["--nnodes", "2", "--rdzv-id", "run"],
                "--rdzv-endpoint must be a stable host:port",
            ),
            (
                ["--smoke-synthetic-examples-per-bucket", "0"],
                "--smoke-synthetic-examples-per-bucket",
            ),
            (
                ["--smoke-synthetic-examples-per-bucket", "2"],
                "--smoke-synthetic-examples-per-bucket only applies",
            ),
            (
                ["--smoke-synthetic-buckets", "--num-examples", "1"],
                "--smoke-synthetic-buckets cannot be combined with --num-examples",
            ),
            (
                ["--smoke-synthetic-buckets", "--max-streamed-examples", "1"],
                "--smoke-synthetic-buckets cannot be combined with --max-streamed-examples",
            ),
            (["--learning-rate", "nan"], "--learning-rate must be finite"),
            (["--min-learning-rate", "inf"], "--min-learning-rate must be finite"),
            (
                ["--learning-rate", "1e-5", "--min-learning-rate", "1e-4"],
                "--min-learning-rate cannot exceed --learning-rate",
            ),
            (["--global-batch-size", "10"], "--global-batch-size must be divisible"),
            (["--log-rank", "rank0"], "--log-rank contains a non-integer rank"),
            (
                ["--torchrun-log-rank-filter", "-1"],
                "--torchrun-log-rank-filter ranks must be non-negative",
            ),
        ]
        for cli_args, message in cases:
            with self.subTest(cli_args=cli_args):
                with self.assertRaisesRegex(ValueError, message):
                    self._validate_launch_args(cli_args)

    def test_launch_input_validation_rejects_context_and_bucket_mismatches(self):
        with self.assertRaisesRegex(ValueError, "paper context"):
            self._validate_launch_args(
                [
                    "--buckets",
                    str(train.PAPER_CONTEXT_LENGTH * 2),
                    "--bucket-cp",
                    f"{train.PAPER_CONTEXT_LENGTH * 2}:1",
                    "--max-length",
                    str(train.PAPER_CONTEXT_LENGTH + 1),
                ]
            )
        with self.assertRaisesRegex(ValueError, "largest bucket"):
            self._validate_launch_args(["--max-length", "512"])
        with self.assertRaisesRegex(ValueError, "not present in --buckets"):
            self._validate_launch_args(["--bucket-cp", "256:1,512:1"])
        with self.assertRaisesRegex(ValueError, "single-bucket"):
            self._validate_launch_args(
                [
                    "--buckets",
                    "128,256",
                    "--bucket-cp",
                    "128:1,256:1",
                    "--bucket-curriculum",
                    "single-bucket",
                ]
            )

    def test_bucket_parsers_reject_malformed_or_ambiguous_values(self):
        with self.assertRaisesRegex(ValueError, "invalid bucket size"):
            train.parse_bucket_list("256,abc")
        with self.assertRaisesRegex(ValueError, "duplicate bucket"):
            train.parse_bucket_list("256,256")
        with self.assertRaisesRegex(ValueError, "invalid bucket CP map entry"):
            train.parse_bucket_cp_map("256:not-a-cp")
        with self.assertRaisesRegex(ValueError, "duplicate bucket"):
            train.parse_bucket_cp_map("256:1,256:2")

    def test_wandb_identity_generates_run_id_and_env_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-project",
                    "proj",
                    "--wandb-entity",
                    "team",
                    "--wandb-run-name",
                    "run-name",
                    "--wandb-run-group",
                    "group-a",
                    "--wandb-run-job-type",
                    "train",
                    "--wandb-run-tags",
                    "direct-to-hero,smoke",
                    "--wandb-run-notes",
                    "notes",
                    "--wandb-mode",
                    "offline",
                ]
            )
            out_dir.mkdir(parents=True)
            identity = train.resolve_wandb_identity(args, resume_state=None)
            stage = train.BucketStage(
                bucket=256,
                cp_degree=1,
                example_count=1,
                steps=1,
                cumulative_steps=1,
                bucket_file=out_dir / "data" / "bucket_256.jsonl",
            )
            env = train.build_stage_env(
                args,
                stage=stage,
                total_steps=1,
                warmup_steps=0,
                pad_token_id=0,
            )
            persisted = json.loads(
                (out_dir / train.WANDB_IDENTITY_FILENAME).read_text()
            )

        self.assertIsNotNone(identity)
        self.assertTrue(identity["run_id"].startswith("swehero-"))
        self.assertLessEqual(len(identity["run_id"]), 64)
        self.assertEqual(identity, persisted)
        self.assertEqual(identity["resume"], "allow")
        self.assertEqual(env["SWEHERO_ENABLE_WANDB"], "1")
        self.assertEqual(env["WANDB_PROJECT"], "proj")
        self.assertEqual(env["WANDB_TEAM"], "team")
        self.assertEqual(env["WANDB_ENTITY"], "team")
        self.assertEqual(env["WANDB_RUN_NAME"], "run-name")
        self.assertEqual(env["WANDB_NAME"], "run-name")
        self.assertEqual(env["WANDB_RUN_ID"], identity["run_id"])
        self.assertEqual(env["WANDB_RESUME"], "allow")
        self.assertEqual(env["WANDB_RUN_GROUP"], "group-a")
        self.assertEqual(env["WANDB_RUN_JOB_TYPE"], "train")
        self.assertEqual(env["WANDB_JOB_TYPE"], "train")
        self.assertEqual(env["WANDB_RUN_TAGS"], "direct-to-hero,smoke")
        self.assertEqual(env["WANDB_TAGS"], "direct-to-hero,smoke")
        self.assertEqual(env["WANDB_RUN_NOTES"], "notes")
        self.assertEqual(env["WANDB_NOTES"], "notes")
        self.assertEqual(env["WANDB_MODE"], "offline")
        self.assertNotIn("WANDB_API_KEY", identity["env"])

    def test_wandb_identity_reuses_run_id_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(
                ["--out-dir", str(out_dir), "--enable-wandb"]
            )
            out_dir.mkdir(parents=True)
            train.resolve_wandb_identity(args, resume_state=None)
            run_id = args.wandb_run_id

            resume_args = train.parse_args(
                ["--out-dir", str(out_dir), "--enable-wandb", "--resume"]
            )
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=train._checkpoint_dir(out_dir),
                final_export_dir=train._final_model_export_dir(out_dir),
                latest_resumable_step=1,
                latest_model_export_step=None,
                latest_any_step=1,
            )
            identity = train.resolve_wandb_identity(
                resume_args,
                resume_state=resume_state,
            )

        self.assertEqual(resume_args.wandb_run_id, run_id)
        self.assertEqual(identity["run_id"], run_id)
        self.assertEqual(identity["resume"], "allow")

    def test_wandb_identity_rejects_changed_existing_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-run-id",
                    "original-run",
                ]
            )
            out_dir.mkdir(parents=True)
            train.resolve_wandb_identity(args, resume_state=None)
            changed = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-run-id",
                    "different-run",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "W&B identity"):
                train.resolve_wandb_identity(changed, resume_state=None)

    def test_wandb_resume_controls_reject_conflicts_and_bad_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            out_dir.mkdir(parents=True)
            conflicting = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-resume",
                    "allow",
                    "--wandb-resume-from",
                    "abc123?_step=10",
                ]
            )
            bad_run_id = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-run-id",
                    "bad/id",
                ]
            )

            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                train.resolve_wandb_identity(conflicting, resume_state=None)
            with self.assertRaisesRegex(ValueError, "forbids"):
                train.resolve_wandb_identity(bad_run_id, resume_state=None)

    def test_explicit_launch_env_file_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.env"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(FileNotFoundError, "Requested env file"):
                    train.load_launch_env_file(["--env-file", str(missing)])

    def test_default_launch_env_file_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.env"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(train.smoke, "ENV_FILE", str(missing)),
            ):
                loaded_env_file = train.load_launch_env_file([])

        self.assertEqual(loaded_env_file, str(missing))

    def test_expected_qwen_yarn_rope_config_tracks_128k_extension(self):
        rope = train.expected_qwen_yarn_rope_config()

        self.assertEqual(rope["rope_type"], "yarn")
        self.assertEqual(rope["max_position_embeddings"], train.PAPER_CONTEXT_LENGTH)
        self.assertEqual(
            rope["original_max_position_embeddings"],
            train.QWEN_NATIVE_CONTEXT_LENGTH,
        )
        self.assertEqual(rope["factor"], 4.0)
        self.assertEqual(rope["rope_theta"], 1_000_000.0)
        self.assertEqual(rope["beta_fast"], 32.0)
        self.assertEqual(rope["beta_slow"], 1.0)

    def test_torchtitan_qwen_registry_uses_standard_yarn_beta_names(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "torchtitan/torchtitan/models/qwen2_5/__init__.py").read_text()

        self.assertIn("max_seq_len=QWEN25_CODER_7B_CONTEXT", source)
        self.assertIn("theta=1_000_000.0", source)
        self.assertIn('scaling="yarn"', source)
        self.assertIn("rope_factor=QWEN25_CODER_7B_CONTEXT / QWEN25_NATIVE_CONTEXT", source)
        self.assertIn("beta_fast=32.0", source)
        self.assertIn("beta_slow=1.0", source)
        self.assertIn("original_seq_len=QWEN25_NATIVE_CONTEXT", source)

    def test_model_asset_provenance_records_complete_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_assets = Path(tmp) / "hf"
            hf_assets.mkdir()
            config = {
                "_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 3584,
                "max_position_embeddings": 32768,
                "model_type": "qwen2",
                "num_hidden_layers": 28,
                "rope_scaling": None,
                "torch_dtype": "bfloat16",
                "vocab_size": 152064,
            }
            (hf_assets / "config.json").write_text(json.dumps(config))
            (hf_assets / "generation_config.json").write_text(
                json.dumps({"eos_token_id": 151645, "pad_token_id": 151643})
            )
            (hf_assets / "tokenizer.json").write_text('{"tokenizer": true}')
            (hf_assets / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "template"})
            )
            (hf_assets / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
            (hf_assets / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
            (hf_assets / "orphan.safetensors").write_bytes(b"orphan")
            (hf_assets / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 13},
                        "weight_map": {
                            "lm_head.weight": "model-00002-of-00002.safetensors",
                            "model.embed_tokens.weight": (
                                "model-00001-of-00002.safetensors"
                            ),
                        },
                    }
                )
            )
            tokenizer_metadata = {
                "hf_assets_path": str(hf_assets),
                "tokenizer_json_sha256": train._hash_file(hf_assets / "tokenizer.json"),
                "tokenizer_config_sha256": train._hash_file(
                    hf_assets / "tokenizer_config.json"
                ),
            }

            provenance = train._model_asset_provenance(
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                model_revision=train.MODEL_REVISION,
                hf_assets_path=hf_assets,
                tokenizer_metadata=tokenizer_metadata,
            )

        files = {record["path"]: record for record in provenance["files"]}
        self.assertEqual(
            provenance["schema_version"],
            train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
        )
        self.assertEqual(provenance["model_revision"], train.MODEL_REVISION)
        self.assertEqual(provenance["file_count"], len(files))
        self.assertEqual(
            provenance["total_bytes"],
            sum(record["bytes"] for record in files.values()),
        )
        self.assertEqual(files["config.json"]["kind"], "model_config")
        self.assertEqual(
            files["model-00001-of-00002.safetensors"]["sha256"],
            train._sha256_text("shard-1"),
        )
        self.assertEqual(provenance["config"]["summary"]["model_type"], "qwen2")
        self.assertEqual(provenance["config"]["summary"]["hidden_size"], 3584)
        self.assertEqual(
            provenance["generation_config"]["summary"]["pad_token_id"],
            151643,
        )
        self.assertEqual(provenance["safetensors"]["weight_map_entries"], 2)
        self.assertEqual(
            [record["path"] for record in provenance["safetensors"]["shard_files"]],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )
        self.assertEqual(
            provenance["safetensors"]["unindexed_safetensors_files"],
            ["orphan.safetensors"],
        )
        self.assertEqual(provenance["tokenizer"], tokenizer_metadata)

    def test_hf_asset_preflight_requires_indexed_weight_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            shard = hf_assets / "model-00001-of-00001.safetensors"
            shard.unlink()
            args = train.parse_args(["--hf-assets-path", str(hf_assets)])

            with self.assertRaisesRegex(RuntimeError, "safetensors shard"):
                train.validate_hf_asset_preflight(args)

            shard.write_bytes(b"shard")
            summary = train.validate_hf_asset_preflight(args)

        self.assertEqual(summary["config_model_type"], "qwen2")
        self.assertEqual(summary["safetensors"]["shard_count"], 1)
        self.assertEqual(summary["safetensors"]["weight_map_entries"], 1)

    def test_hf_asset_preflight_rejects_manifest_asset_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            args = train.parse_args(["--hf-assets-path", str(hf_assets)])
            manifest = self._model_assets_manifest(args)

            summary = train.validate_hf_asset_preflight(args, manifest)
            mismatched_manifest = json.loads(json.dumps(manifest))
            mismatched_manifest["model_assets"]["model_id"] = "other/model"
            with self.assertRaisesRegex(RuntimeError, "model_assets.model_id"):
                train.validate_hf_asset_preflight(args, mismatched_manifest)

            mismatched_manifest = json.loads(json.dumps(manifest))
            mismatched_manifest["model_assets"]["model_revision"] = "0" * 40
            with self.assertRaisesRegex(RuntimeError, "model_assets.model_revision"):
                train.validate_hf_asset_preflight(args, mismatched_manifest)

            (hf_assets / "model-00001-of-00001.safetensors").write_bytes(b"drift")
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                train.validate_hf_asset_preflight(args, manifest)

            (hf_assets / "model-00001-of-00001.safetensors").write_bytes(
                b"changed-length"
            )
            with self.assertRaisesRegex(RuntimeError, "byte size"):
                train.validate_hf_asset_preflight(args, manifest)

        self.assertEqual(summary["manifest_model_assets"]["file_count"], 5)
        self.assertEqual(summary["manifest_model_assets"]["sha256_verified_files"], 5)

    def test_cuda_launch_summary_requires_visible_device_per_rank(self):
        class FakeCuda:
            def __init__(self, count):
                self.count = count

            def is_available(self):
                return self.count > 0

            def device_count(self):
                return self.count

            def get_device_name(self, index):
                return f"Fake GPU {index}"

            def get_device_capability(self, index):
                return (9, 0)

        fake_torch = types.SimpleNamespace(cuda=FakeCuda(1))

        with self.assertRaisesRegex(RuntimeError, "visible CUDA device"):
            train._cuda_launch_summary(fake_torch, nproc_per_node=2)

        summary = train._cuda_launch_summary(fake_torch, nproc_per_node=1)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["device_count"], 1)
        self.assertEqual(summary["devices"][0]["capability"], [9, 0])

    def test_nvidia_smi_metadata_parses_driver_and_cuda_version(self):
        def fake_command(command, *, timeout_seconds=5.0):
            if "--query-gpu=index,name,uuid,driver_version,memory.total" in command:
                return {
                    "command": command,
                    "available": True,
                    "returncode": 0,
                    "stdout": (
                        "0, NVIDIA H100 80GB HBM3, GPU-test, "
                        "570.195.03, 81559\n"
                    ),
                    "stderr": "",
                }
            return {
                "command": command,
                "available": True,
                "returncode": 0,
                "stdout": (
                    "| NVIDIA-SMI 570.195.03    Driver Version: 570.195.03"
                    "    CUDA Version: 12.8 |\n"
                ),
                "stderr": "",
            }

        with patch.object(train, "_run_metadata_command", side_effect=fake_command):
            metadata = train._nvidia_smi_metadata()

        self.assertEqual(metadata["cuda_version_from_banner"], "12.8")
        self.assertEqual(metadata["gpus"][0]["index"], "0")
        self.assertEqual(metadata["gpus"][0]["name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(metadata["gpus"][0]["uuid"], "GPU-test")
        self.assertEqual(metadata["gpus"][0]["driver_version"], "570.195.03")
        self.assertEqual(metadata["gpus"][0]["memory_total_mib"], "81559")

    def test_write_runtime_metadata_records_environment_and_lockfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run")])
            args.out_dir.mkdir(parents=True)
            lockfile = Path(tmp) / "lock.txt"
            lockfile.write_text("locked\n")
            runtime = {
                "python": sys.executable,
                "torch": "2.x",
                "cuda": {"device_count": 8},
            }

            with (
                patch.object(train.time, "time", return_value=123.0),
                patch.object(
                    train,
                    "_runtime_lockfile_metadata",
                    return_value=[
                        {
                            "kind": "test_lock",
                            **train._file_metadata(lockfile),
                        }
                    ],
                ),
                patch.object(
                    train,
                    "_nvidia_smi_metadata",
                    return_value={
                        "gpus": [
                            {
                                "index": "0",
                                "driver_version": "570.195.03",
                            }
                        ],
                        "cuda_version_from_banner": "12.8",
                    },
                ),
                patch.dict(
                    os.environ,
                    {
                        "NCCL_DEBUG": "INFO",
                        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                        "UNRELATED": "ignored",
                    },
                    clear=True,
                ),
            ):
                metadata = train.write_runtime_metadata(args, runtime)

            persisted = json.loads(
                (args.out_dir / train.RUNTIME_METADATA_FILENAME).read_text()
            )

        self.assertEqual(metadata, persisted)
        self.assertEqual(
            metadata["schema_version"],
            train.RUNTIME_METADATA_SCHEMA_VERSION,
        )
        self.assertEqual(metadata["created_at_unix"], 123.0)
        self.assertEqual(metadata["runtime"], runtime)
        self.assertEqual(
            metadata["lockfiles"][0]["sha256"],
            train._sha256_text("locked\n"),
        )
        self.assertEqual(
            metadata["hardware"]["nvidia_smi"]["cuda_version_from_banner"],
            "12.8",
        )
        self.assertEqual(
            metadata["environment"],
            {
                "NCCL_DEBUG": "INFO",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            },
        )
        self.assertEqual(
            metadata["workspace"]["configured_root"],
            str(train._configured_workspace_root(args)),
        )
        self.assertEqual(
            metadata["workspace"]["canonical_root"],
            str(train.CANONICAL_WORKSPACE_ROOT),
        )
        self.assertIn("cwd", metadata["workspace"])

    def test_runtime_lockfile_metadata_uses_invoked_venv_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            venv_root = Path(tmp) / "venv"
            (repo_root / "requirements").mkdir(parents=True)
            (repo_root / "torchtitan" / ".ci" / "docker").mkdir(parents=True)
            (venv_root / "bin").mkdir(parents=True)
            (repo_root / "requirements" / "torchtitan-pod-cu128.lock").write_text(
                "lock\n"
            )
            (repo_root / "requirements" / "torchtitan-pod-cu128.txt").write_text(
                "requirements\n"
            )
            (
                repo_root / "torchtitan" / ".ci" / "docker" / "requirements.txt"
            ).write_text("torchtitan\n")
            runtime_json = venv_root / "torchtitan-swehero-runtime.json"
            runtime_json.write_text('{"runtime": true}\n')

            metadata = train._runtime_lockfile_metadata(
                repo_root=repo_root,
                python_executable=str(venv_root / "bin" / "python"),
            )

        by_kind = {record["kind"]: record for record in metadata}
        self.assertEqual(
            by_kind["venv_runtime_metadata"]["path"],
            str(runtime_json),
        )
        self.assertTrue(by_kind["venv_runtime_metadata"]["exists"])
        self.assertEqual(
            by_kind["venv_runtime_metadata"]["sha256"],
            train._sha256_text('{"runtime": true}\n'),
        )

    def test_dataset_artifact_metadata_records_selection_and_shard_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            data_dir = dataset / "data"
            data_dir.mkdir(parents=True)
            metadata = {"rows": 2, "source_revision": "abc123"}
            (dataset / "metadata.json").write_text(json.dumps(metadata))
            (dataset / "selection_manifest.jsonl").write_text(
                '{"instance_id":"one"}\n{"instance_id":"two"}\n'
            )
            shard = data_dir / "train-00000-of-00001.parquet"
            shard.write_bytes(b"parquet bytes")

            artifact = train._dataset_artifact_metadata(dataset)

        self.assertEqual(artifact["path"], str(dataset))
        self.assertEqual(artifact["metadata"], metadata)
        self.assertEqual(
            artifact["metadata_json"]["sha256"],
            artifact["metadata_json_sha256"],
        )
        self.assertEqual(
            artifact["selection_manifest"]["sha256"],
            artifact["selection_manifest_sha256"],
        )
        self.assertEqual(artifact["data_file_count"], 1)
        self.assertEqual(
            artifact["data_files"][0]["relative_path"],
            shard.relative_to(dataset).as_posix(),
        )
        self.assertEqual(artifact["data_files"][0]["bytes"], len(b"parquet bytes"))
        self.assertEqual(
            artifact["data_files"][0]["sha256"],
            train._sha256_text("parquet bytes"),
        )
        self.assertEqual(artifact["total_data_bytes"], len(b"parquet bytes"))

    def test_cos_sin_yarn_uses_huggingface_correction_range_order(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "torchtitan/torchtitan/models/common/rope.py").read_text()

        fast_index = source.index("cfg.beta_fast * 2 * math.pi")
        slow_index = source.index("cfg.beta_slow * 2 * math.pi")
        self.assertLess(fast_index, slow_index)

    def test_choose_bucket_ceilings(self):
        buckets = (8, 16, 32)

        self.assertEqual(train.choose_bucket(1, buckets), 8)
        self.assertEqual(train.choose_bucket(8, buckets), 8)
        self.assertEqual(train.choose_bucket(9, buckets), 16)
        self.assertEqual(train.choose_bucket(17, buckets), 32)
        with self.assertRaises(ValueError):
            train.choose_bucket(33, buckets)

    def test_bucket_plan_uses_epochs_and_cumulative_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_files = {
                8192: Path(tmp) / "bucket_8192.jsonl",
                32768: Path(tmp) / "bucket_32768.jsonl",
            }
            plan = train.build_bucket_plan(
                bucket_counts={8192: 33, 32768: 1},
                bucket_files=bucket_files,
                bucket_cp={8192: 1, 32768: 2},
                epochs=3.0,
                global_batch_size=32,
                warmup_ratio=0.1,
            )

        self.assertEqual(plan.total_steps, 5)
        self.assertEqual(plan.warmup_steps, 1)
        self.assertEqual([stage.bucket for stage in plan.stages], [8192, 32768])
        self.assertEqual([stage.steps for stage in plan.stages], [4, 1])
        self.assertEqual([stage.cumulative_steps for stage in plan.stages], [4, 5])
        self.assertEqual([stage.cp_degree for stage in plan.stages], [1, 2])

    def test_bucket_plan_uses_explicit_curriculum_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_files = {
                8192: Path(tmp) / "bucket_8192.jsonl",
                32768: Path(tmp) / "bucket_32768.jsonl",
            }
            plan = train.build_bucket_plan(
                bucket_counts={8192: 33, 32768: 1},
                bucket_files=bucket_files,
                bucket_cp={8192: 1, 32768: 2},
                epochs=3.0,
                global_batch_size=32,
                warmup_ratio=0.1,
                bucket_curriculum="long-to-short",
            )

        self.assertEqual([stage.bucket for stage in plan.stages], [32768, 8192])
        self.assertEqual([stage.steps for stage in plan.stages], [1, 4])
        self.assertEqual([stage.cumulative_steps for stage in plan.stages], [1, 5])
        self.assertEqual([stage.cp_degree for stage in plan.stages], [2, 1])

    def test_single_bucket_curriculum_requires_one_non_empty_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_files = {
                8192: Path(tmp) / "bucket_8192.jsonl",
                32768: Path(tmp) / "bucket_32768.jsonl",
            }

            with self.assertRaisesRegex(ValueError, "single-bucket curriculum"):
                train.build_bucket_plan(
                    bucket_counts={8192: 33, 32768: 1},
                    bucket_files=bucket_files,
                    bucket_cp={8192: 1, 32768: 2},
                    epochs=3.0,
                    global_batch_size=32,
                    warmup_ratio=0.1,
                    bucket_curriculum="single-bucket",
                )

    def test_resume_requires_existing_artifact_then_full_dcp_for_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run"), "--resume"])
            with self.assertRaises(FileNotFoundError):
                train.validate_resume_request(args)

            args, _manifest, plan = self._resume_test_setup(tmp)
            export = train._final_model_export_dir(args.out_dir) / "step-3"
            export.mkdir(parents=True)
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)
            with self.assertRaisesRegex(RuntimeError, "full DCP checkpoint"):
                train.validate_resume_progress(plan, resume_state)

    def test_resume_rejects_destructive_refresh_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            (out_dir / "torchtitan" / "checkpoint" / "step-1").mkdir(parents=True)
            args = train.parse_args(
                ["--out-dir", str(out_dir), "--resume", "--overwrite-output"]
            )

            with self.assertRaisesRegex(ValueError, "overwrite-output"):
                train.validate_resume_request(args)

    def test_launch_input_validation_rejects_overlapping_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(shared),
                    "--dataset-path",
                    str(shared),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            with self.assertRaisesRegex(ValueError, "overlaps"):
                train.validate_launch_inputs(
                    args,
                    buckets=buckets,
                    bucket_cp=bucket_cp,
                )

    def test_launch_input_validation_rejects_dangerous_output_overwrite_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        args = train.parse_args(
            [
                "--out-dir",
                str(repo_root),
                "--dataset-path",
                str(repo_root / "datasets" / train.TRAINING_DATASET_NAME),
                "--hf-assets-path",
                str(repo_root / "assets" / "hf"),
                "--overwrite-output",
            ]
        )
        buckets = train.parse_bucket_list(args.buckets)
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

        with self.assertRaisesRegex(ValueError, "dangerous.*--out-dir"):
            train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)

    def test_launch_input_validation_rejects_dangerous_dataset_rebuild_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(repo_root),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--rebuild-source-dataset",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            with self.assertRaisesRegex(ValueError, "dangerous.*--dataset-path"):
                train.validate_launch_inputs(
                    args,
                    buckets=buckets,
                    bucket_cp=bucket_cp,
                )

    def test_resume_ignores_final_model_export_when_deciding_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            final_export_root = train._final_model_export_dir(args.out_dir)
            full = checkpoint_root / "step-5"
            export = final_export_root / "step-5"
            full.mkdir(parents=True)
            export.mkdir(parents=True)
            (full / ".metadata").write_text("{}")
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)
            train.validate_resume_progress(plan, resume_state)

        self.assertEqual(resume_state.latest_resumable_step, 5)
        self.assertEqual(resume_state.latest_model_export_step, 5)
        self.assertEqual(resume_state.latest_any_step, 5)
        self.assertEqual(train.stages_to_run_for_resume(plan, resume_state), ())

    def test_resume_rejects_completed_final_export_without_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            export = train._final_model_export_dir(args.out_dir) / "step-5"
            export.mkdir(parents=True)
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)

            with self.assertRaisesRegex(RuntimeError, "final.*full DCP"):
                train.validate_resume_progress(plan, resume_state)

    def test_resume_rejects_nonfinal_model_export_newer_than_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            final_export_root = train._final_model_export_dir(args.out_dir)
            full = checkpoint_root / "step-2"
            export = final_export_root / "step-3"
            full.mkdir(parents=True)
            export.mkdir(parents=True)
            (full / ".metadata").write_text("{}")
            (export / "model.safetensors.index.json").write_text("{}")
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                final_export_dir=final_export_root,
                latest_resumable_step=2,
                latest_model_export_step=3,
                latest_any_step=3,
            )

            with self.assertRaisesRegex(RuntimeError, "non-resumable export"):
                train.validate_resume_progress(plan, resume_state)

    def test_resume_rejects_incomplete_export_without_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            final_export_root = train._final_model_export_dir(args.out_dir)
            export = final_export_root / "step-3"
            export.mkdir(parents=True)
            (export / "model.safetensors.index.json").write_text("{}")
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=train._checkpoint_dir(args.out_dir),
                final_export_dir=final_export_root,
                latest_resumable_step=None,
                latest_model_export_step=3,
                latest_any_step=3,
            )

            with self.assertRaisesRegex(RuntimeError, "full DCP"):
                train.validate_resume_progress(plan, resume_state)

    def test_final_artifact_validation_writes_report_for_completed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)

            report = train.validate_final_artifacts(args, plan)
            persisted = json.loads(
                (args.out_dir / train.FINAL_ARTIFACT_VALIDATION_FILENAME).read_text()
            )

        self.assertEqual(
            report["schema_version"],
            train.FINAL_ARTIFACT_VALIDATION_SCHEMA_VERSION,
        )
        self.assertEqual(report, persisted)
        self.assertEqual(report["plan_total_steps"], 5)
        self.assertEqual(report["final_export"]["layout"], "final_export")
        self.assertEqual(report["final_export"]["shard_count"], 2)
        self.assertEqual(report["final_export"]["weight_map_entries"], 2)
        self.assertEqual(report["final_export"]["index_metadata_total_size"], 14)
        self.assertEqual(
            report["final_export"]["shards"][0]["sha256"],
            train._sha256_text("shard-1"),
        )
        self.assertEqual(report["resumable_checkpoints"]["steps"], [5])
        self.assertEqual(report["resumable_checkpoints"]["latest_step"], 5)
        self.assertEqual(
            report["resumable_checkpoints"]["checkpoints"][0]["payload_file_count"],
            1,
        )
        self.assertEqual(
            report["resumable_checkpoints"]["checkpoints"][0]["payload_files"][0][
                "rank"
            ],
            0,
        )

    def test_final_artifact_validation_requires_final_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=4)
            self._write_final_export(args.out_dir, step=5)

            with self.assertRaisesRegex(RuntimeError, "Final resumable DCP"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_missing_export_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            export = self._write_final_export(args.out_dir, step=5)
            (export / "model-00002-of-00002.safetensors").unlink()

            with self.assertRaisesRegex(RuntimeError, "final model export shard"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_unindexed_export_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            export = self._write_final_export(args.out_dir, step=5)
            (export / "orphan.safetensors").write_bytes(b"orphan")

            with self.assertRaisesRegex(RuntimeError, "unindexed"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_impossible_export_total_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            export = self._write_final_export(args.out_dir, step=5)
            index_path = export / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["metadata"]["total_size"] = 10_000
            index_path.write_text(json.dumps(index))

            with self.assertRaisesRegex(RuntimeError, "total_size"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_empty_dcp_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint = self._write_dcp_checkpoint(args.out_dir, step=5)
            (checkpoint / "__0_0.distcp").write_bytes(b"")
            self._write_final_export(args.out_dir, step=5)

            with self.assertRaisesRegex(RuntimeError, "DCP checkpoint payload"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_malformed_dcp_payload_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint = self._write_dcp_checkpoint(args.out_dir, step=5)
            (checkpoint / "__0_0.distcp").replace(checkpoint / "rank0.distcp")
            self._write_final_export(args.out_dir, step=5)

            with self.assertRaisesRegex(RuntimeError, "payload file name"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_first_step_checkpoint_validation_accepts_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, _plan = self._resume_test_setup(tmp)
            written = self._write_first_step_checkpoint_validation_report(args.out_dir)

            report = train.validate_first_step_checkpoint_report(args)

        self.assertEqual(report, written)
        self.assertEqual(report["step"], 1)
        self.assertEqual(report["checkpoint"]["payload_file_count"], 1)
        self.assertEqual(report["checkpoint"]["payload_rank_count"], 1)

    def test_first_step_checkpoint_validation_rejects_missing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, _plan = self._resume_test_setup(tmp)

            with self.assertRaisesRegex(RuntimeError, "First-step checkpoint"):
                train.validate_first_step_checkpoint_report(args)

    def test_first_step_checkpoint_validation_rejects_wrong_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, _plan = self._resume_test_setup(tmp)
            checkpoint = train._validate_dcp_checkpoint_step(
                self._write_dcp_checkpoint(args.out_dir, step=2)
            )
            report = {
                "schema_version": train.FIRST_STEP_CHECKPOINT_VALIDATION_SCHEMA_VERSION,
                "created_at_unix": 1.0,
                "step": 2,
                "checkpoint": checkpoint,
            }
            train._first_step_checkpoint_validation_path(args.out_dir).write_text(
                json.dumps(report, indent=2)
            )

            with self.assertRaisesRegex(RuntimeError, "step 1"):
                train.validate_first_step_checkpoint_report(args)

    def test_final_artifact_validation_rejects_duplicate_legacy_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_final_export(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5, legacy=True)

            with self.assertRaisesRegex(RuntimeError, "both"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_legacy_export_without_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_final_export(args.out_dir, step=5, legacy=True)

            with self.assertRaisesRegex(RuntimeError, "Final model export is missing"):
                train.validate_final_artifacts(args, plan, write_report=False)
            with self.assertRaisesRegex(RuntimeError, "Final resumable DCP"):
                train.validate_final_artifacts(
                    args,
                    plan,
                    allow_legacy_export=True,
                    write_report=False,
                )

    def test_post_training_eval_hook_records_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)
            final_validation = train.validate_final_artifacts(args, plan)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=(),
                dataloader_resume_flags={},
            )
            code = (
                "import os, pathlib; "
                "pathlib.Path(os.environ['SWEHERO_OUT_DIR'], "
                "'eval-step.txt').write_text("
                "os.environ['SWEHERO_FINAL_EXPORT_STEP'])"
            )
            args.post_training_eval_command = (
                f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
            )

            eval_status = train.run_post_training_eval(args, plan, final_validation)
            persisted = json.loads(
                (args.out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME).read_text()
            )
            stage_status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )
            eval_step_text = (args.out_dir / "eval-step.txt").read_text()

        self.assertIsNotNone(eval_status)
        self.assertEqual(eval_status, persisted)
        self.assertEqual(eval_status["status"], "succeeded")
        self.assertEqual(eval_status["returncode"], 0)
        self.assertEqual(
            eval_status["env_overrides"]["SWEHERO_FINAL_EXPORT_STEP"],
            "5",
        )
        self.assertEqual(eval_step_text, "5")
        self.assertEqual(stage_status["post_training_eval"]["status"], "succeeded")
        self.assertEqual(
            stage_status["summary"]["post_training_eval_status"], "succeeded"
        )

    def test_post_training_eval_hook_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)
            final_validation = train.validate_final_artifacts(args, plan)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=(),
                dataloader_resume_flags={},
            )
            code = "import sys; print('eval failed'); sys.exit(3)"
            args.post_training_eval_command = (
                f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
            )

            with self.assertRaisesRegex(RuntimeError, "return code 3"):
                train.run_post_training_eval(args, plan, final_validation)

            persisted = json.loads(
                (args.out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME).read_text()
            )
            stage_status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["returncode"], 3)
        self.assertIn("eval failed", persisted["stdout_tail"])
        self.assertEqual(stage_status["post_training_eval"]["status"], "failed")
        self.assertEqual(stage_status["summary"]["failure_count"], 1)

    def test_stage_status_records_successful_stage_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.validate_first_step_checkpoint = False
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            with (
                patch.object(train, "_run_command_with_signal_forwarding") as run_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        run_mock.assert_called_once()
        stage = status["stages"][0]
        self.assertEqual(stage["status"], "succeeded")
        self.assertEqual(stage["attempts"][0]["status"], "succeeded")
        self.assertIn("torchrun_command", stage["attempts"][0])
        self.assertEqual(status["failures"], [])
        self.assertEqual(status["summary"]["stage_status_counts"]["succeeded"], 1)

    def test_stage_status_records_failed_stage_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            error = train.subprocess.CalledProcessError(
                42,
                ["torchrun", "-m", "torchtitan.train"],
            )

            with (
                patch.object(
                    train,
                    "_run_command_with_signal_forwarding",
                    side_effect=error,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(train.subprocess.CalledProcessError):
                    train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        stage = status["stages"][0]
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["attempts"][0]["status"], "failed")
        self.assertEqual(stage["failure"]["returncode"], 42)
        self.assertEqual(status["failures"][0]["phase"], "stage")
        self.assertEqual(status["failures"][0]["stage_id"], stage["id"])
        self.assertEqual(status["summary"]["failure_count"], 1)

    def test_stage_status_records_signal_terminated_stage_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            error = train.SignalTerminationError(
                signum=int(signal.SIGTERM),
                command=["torchrun", "-m", "torchtitan.train"],
                returncode=-int(signal.SIGTERM),
            )

            with (
                patch.object(
                    train,
                    "_run_command_with_signal_forwarding",
                    side_effect=error,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(train.SignalTerminationError):
                    train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        stage = status["stages"][0]
        failure = stage["failure"]
        self.assertEqual(stage["status"], "failed")
        self.assertTrue(failure["terminated_by_signal"])
        self.assertEqual(failure["signum"], int(signal.SIGTERM))
        self.assertEqual(failure["signal_name"], "SIGTERM")
        self.assertEqual(failure["returncode"], -int(signal.SIGTERM))
        self.assertEqual(status["failures"][0], failure)

    def test_stage_command_forwards_sigterm_to_process_group(self):
        command = ["torchrun", "-m", "torchtitan.train"]

        class FakeProcess:
            pid = 12345

            def __init__(self) -> None:
                self.returncode = None

            def poll(self):
                return self.returncode

        fake_process = FakeProcess()
        installed_handlers = {}

        def fake_signal(signum, handler):
            signum = int(signum)
            previous = installed_handlers.get(signum, signal.SIG_DFL)
            installed_handlers[signum] = handler
            return previous

        def fake_sleep(_seconds):
            handler = installed_handlers[int(signal.SIGTERM)]
            handler(int(signal.SIGTERM), None)
            fake_process.returncode = -int(signal.SIGTERM)

        with (
            patch.object(train.subprocess, "Popen", return_value=fake_process) as popen,
            patch.object(train.signal, "signal", side_effect=fake_signal),
            patch.object(train.time, "sleep", side_effect=fake_sleep),
            patch.object(train, "_send_signal_to_process_group") as send_signal,
        ):
            with self.assertRaises(train.SignalTerminationError) as raised:
                train._run_command_with_signal_forwarding(
                    command,
                    env={"A": "B"},
                    cwd=Path("/tmp"),
                )

        popen.assert_called_once()
        self.assertEqual(popen.call_args.kwargs["env"], {"A": "B"})
        self.assertEqual(popen.call_args.kwargs["cwd"], Path("/tmp"))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        send_signal.assert_called_with(fake_process.pid, int(signal.SIGTERM))
        self.assertEqual(raised.exception.signum, int(signal.SIGTERM))
        self.assertEqual(raised.exception.returncode, -int(signal.SIGTERM))

    def test_stage_status_marks_resume_completed_and_pending_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=4)
            args.resume = True
            resume_state = train.validate_resume_request(args)
            stages_to_run = train.stages_to_run_for_resume(plan, resume_state)
            dataloader_resume_flags = train.dataloader_resume_flags_by_stage(
                plan,
                resume_state,
            )

            status = train.initialize_stage_status(
                args,
                plan,
                resume_state=resume_state,
                stages_to_run=stages_to_run,
                dataloader_resume_flags=dataloader_resume_flags,
            )

        self.assertEqual([stage["status"] for stage in status["stages"]], [
            "completed_before_resume",
            "pending",
        ])
        self.assertEqual(status["launch"]["stages_to_run"], [
            status["stages"][1]["id"],
        ])

    def test_final_validation_status_records_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)

            train.validate_final_artifacts_with_status(args, plan)
            status_path = args.out_dir / train.STAGE_STATUS_FILENAME
            status = json.loads(status_path.read_text())
            report_sha256 = train._hash_file(
                args.out_dir / train.FINAL_ARTIFACT_VALIDATION_FILENAME
            )

        final_status = status["final_artifact_validation"]
        self.assertEqual(final_status["status"], "succeeded")
        self.assertEqual(final_status["report_sha256"], report_sha256)
        self.assertEqual(
            final_status["summary"]["final_export"]["shard_count"],
            2,
        )
        self.assertEqual(
            status["summary"]["final_artifact_validation_status"],
            "succeeded",
        )

    def test_final_validation_status_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            with self.assertRaisesRegex(RuntimeError, "Final model export is missing"):
                train.validate_final_artifacts_with_status(args, plan)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        final_status = status["final_artifact_validation"]
        self.assertEqual(final_status["status"], "failed")
        self.assertEqual(final_status["failure"]["phase"], "final_artifact_validation")
        self.assertEqual(status["failures"][0]["phase"], "final_artifact_validation")
        self.assertEqual(status["summary"]["failure_count"], 1)

    def test_resume_contract_accepts_same_config_and_skips_completed_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train._write_resume_contract(args, plan, manifest)
            args.resume = True
            checkpoint_root = args.out_dir / "torchtitan" / "checkpoint"
            latest = checkpoint_root / "step-4"
            latest.mkdir(parents=True)
            (latest / ".metadata").write_text("{}")
            resume_state = train.validate_resume_request(args)

            train.validate_resume_contract(args, plan, manifest)
            stages = train.stages_to_run_for_resume(plan, resume_state)

        self.assertEqual([stage.bucket for stage in stages], [32768])

    def test_run_spec_is_written_once_with_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            with patch.dict(os.environ, {"SWEHERO_SECRET": "do-not-record"}):
                written = train.write_or_validate_run_spec(args, plan, manifest)
                written_again = train.write_or_validate_run_spec(args, plan, manifest)

            spec_path = args.out_dir / train.RUN_SPEC_FILENAME
            sha_path = args.out_dir / train.RUN_SPEC_SHA256_FILENAME
            spec_text = spec_path.read_text()
            spec_sha = sha_path.read_text().strip()
            spec = json.loads(spec_text)

        self.assertTrue(written)
        self.assertFalse(written_again)
        self.assertEqual(spec_sha, train._sha256_text(spec_text))
        self.assertEqual(spec["schema_version"], train.RUN_SPEC_SCHEMA_VERSION)
        self.assertFalse(spec["args"]["production_mode"])
        self.assertEqual(spec["args"]["max_length"], 32768)
        self.assertEqual(spec["manifest"], train._resume_manifest_contract(manifest))
        self.assertEqual(
            spec["paths"]["resumable_checkpoints"],
            str(train._checkpoint_dir(args.out_dir)),
        )
        self.assertEqual(
            spec["paths"]["final_model_exports"],
            str(train._final_model_export_dir(args.out_dir)),
        )
        self.assertEqual(
            spec["paths"]["first_step_checkpoint_validation"],
            str(train._first_step_checkpoint_validation_path(args.out_dir)),
        )
        self.assertEqual(
            spec["paths"]["workspace_root"],
            str(train._configured_workspace_root(args)),
        )
        self.assertEqual(
            spec["workspace"]["configured_root"],
            str(train._configured_workspace_root(args)),
        )
        self.assertEqual(
            spec["workspace"]["script_root"],
            str(train._detected_workspace_root()),
        )
        self.assertEqual(spec["plan"]["total_steps"], plan.total_steps)
        first_env = spec["plan"]["stages"][0]["env_overrides"]
        self.assertEqual(
            first_env["SWEHERO_WORKSPACE_ROOT"],
            str(train._configured_workspace_root(args)),
        )
        self.assertEqual(first_env["SWEHERO_BUCKET_SEQ_LEN"], "8192")
        self.assertEqual(
            first_env["SWEHERO_FINAL_EXPORT_FOLDER"],
            train.FINAL_MODEL_EXPORT_FOLDER,
        )
        self.assertEqual(first_env["SWEHERO_SAVE_FINAL_FULL_CHECKPOINT"], "1")
        self.assertEqual(first_env["SWEHERO_ENABLE_FIRST_STEP_CHECKPOINT"], "1")
        self.assertEqual(
            first_env["SWEHERO_FIRST_STEP_CHECKPOINT_VALIDATION_REPORT"],
            str(train._first_step_checkpoint_validation_path(args.out_dir)),
        )
        self.assertEqual(first_env["SWEHERO_ENABLE_PROFILER"], "0")
        self.assertEqual(first_env["SWEHERO_PROFILER_FREQ"], "10")
        self.assertEqual(first_env["SWEHERO_ENABLE_MEMORY_SNAPSHOT"], "0")
        self.assertNotIn("SWEHERO_SECRET", first_env)

    def test_production_mode_is_recorded_in_run_spec_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.production_mode = True
            spec = train.build_run_spec(args, plan, manifest)
            contract = train.build_resume_contract(args, plan, manifest)

        self.assertTrue(spec["args"]["production_mode"])
        self.assertTrue(contract["args"]["production_mode"])
        self.assertTrue(spec["paper_alignment"]["run_safety"]["production_mode"])
        self.assertEqual(spec["workspace"]["configured_root"], str(args.workspace_root))
        self.assertEqual(contract["workspace"]["configured_root"], str(args.workspace_root))

    def test_hidden_torchtitan_env_inputs_are_recorded_as_launch_args(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "SWEHERO_OPTIMIZER_IMPL": "fused",
                "SWEHERO_TRAINING_DTYPE": "bfloat16",
                "SWEHERO_MP_PARAM_DTYPE": "float32",
                "SWEHERO_MP_REDUCE_DTYPE": "float32",
                "SWEHERO_FSDP_RESHARD_AFTER_FORWARD": "always",
                "SWEHERO_DETECT_ANOMALY": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "2",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3",
            },
            clear=True,
        ):
            args, manifest, plan = self._resume_test_setup(tmp)
            spec = train.build_run_spec(args, plan, manifest)
            first_env = spec["plan"]["stages"][0]["env_overrides"]

        self.assertEqual(spec["args"]["optimizer_impl"], "fused")
        self.assertEqual(spec["args"]["training_dtype"], "bfloat16")
        self.assertEqual(spec["args"]["mixed_precision_param_dtype"], "float32")
        self.assertEqual(spec["args"]["mixed_precision_reduce_dtype"], "float32")
        self.assertEqual(spec["args"]["fsdp_reshard_after_forward"], "always")
        self.assertTrue(spec["args"]["detect_anomaly"])
        self.assertEqual(spec["args"]["cuda_device_max_connections"], "2")
        self.assertEqual(spec["args"]["torch_nccl_async_error_handling"], "3")
        self.assertEqual(first_env["SWEHERO_OPTIMIZER_IMPL"], "fused")
        self.assertEqual(first_env["SWEHERO_TRAINING_DTYPE"], "bfloat16")
        self.assertEqual(first_env["SWEHERO_MP_PARAM_DTYPE"], "float32")
        self.assertEqual(first_env["SWEHERO_MP_REDUCE_DTYPE"], "float32")
        self.assertEqual(first_env["SWEHERO_FSDP_RESHARD_AFTER_FORWARD"], "always")
        self.assertEqual(first_env["SWEHERO_DETECT_ANOMALY"], "1")
        self.assertEqual(first_env["CUDA_DEVICE_MAX_CONNECTIONS"], "2")
        self.assertEqual(first_env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "3")

    def test_first_step_checkpoint_validation_marks_status_after_first_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            stage = plan.stages[0]
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            def fake_run_stage(*_args, **_kwargs):
                self._write_first_step_checkpoint_validation_report(args.out_dir)

            with patch.object(train, "run_stage", side_effect=fake_run_stage):
                train.run_stage_with_status(args, stage, plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        self.assertEqual(status["stages"][0]["status"], "succeeded")
        first_validation = status["first_step_checkpoint_validation"]
        self.assertEqual(first_validation["status"], "succeeded")
        self.assertEqual(first_validation["summary"]["step"], 1)
        self.assertEqual(
            status["summary"]["first_step_checkpoint_validation_status"],
            "succeeded",
        )

    def test_first_step_checkpoint_validation_failure_marks_stage_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            stage = plan.stages[0]
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            with (
                patch.object(train, "run_stage", return_value=None),
                self.assertRaisesRegex(RuntimeError, "First-step checkpoint"),
            ):
                train.run_stage_with_status(args, stage, plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        self.assertEqual(status["stages"][0]["status"], "failed")
        first_validation = status["first_step_checkpoint_validation"]
        self.assertEqual(first_validation["status"], "failed")
        self.assertEqual(
            first_validation["failure"]["phase"],
            "first_step_checkpoint_validation",
        )
        self.assertEqual(
            status["failures"][0]["phase"],
            "first_step_checkpoint_validation",
        )

    def test_run_spec_rejects_hidden_torchtitan_env_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            dataset_path = Path(tmp) / "dataset"
            hf_assets_path = Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"
            base_argv = [
                "--out-dir",
                str(out_dir),
                "--dataset-path",
                str(dataset_path),
                "--hf-assets-path",
                str(hf_assets_path),
                "--buckets",
                "8192,32768",
                "--bucket-cp",
                "8192:1,32768:2",
                "--max-length",
                "32768",
                "--num-examples",
                "34",
                "--max-streamed-examples",
                "100",
            ]
            with patch.dict(
                os.environ,
                {"SWEHERO_TRAINING_DTYPE": "bfloat16"},
                clear=True,
            ):
                args, manifest, plan = self._resume_test_setup(tmp)
                train.write_or_validate_run_spec(args, plan, manifest)

            with patch.dict(os.environ, {}, clear=True):
                changed = train.parse_args(base_argv)
                changed.buckets = ",".join(
                    str(b) for b in train.parse_bucket_list(changed.buckets)
                )
                bucket_cp = train.parse_bucket_cp_map(changed.bucket_cp)
                changed.bucket_cp = train._format_bucket_cp_map(bucket_cp)

            with self.assertRaisesRegex(RuntimeError, "training_dtype"):
                train.write_or_validate_run_spec(changed, plan, manifest)

    def test_stage_env_exposes_profiler_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            args.enable_profiler = True
            args.profiler_trace_folder = "profiles/traces"
            args.profiler_freq = 4
            args.profiler_active = 1
            args.profiler_warmup = 1
            args.profiler_repeat = 2
            args.profiler_skip_first = 1
            args.profiler_skip_first_wait = 1
            args.enable_memory_snapshot = True
            args.memory_snapshot_folder = "profiles/memory"

            env = train.build_stage_env(
                args,
                stage=plan.stages[0],
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=0,
            )

        self.assertEqual(env["SWEHERO_ENABLE_PROFILER"], "1")
        self.assertEqual(env["SWEHERO_PROFILER_TRACE_FOLDER"], "profiles/traces")
        self.assertEqual(env["SWEHERO_PROFILER_FREQ"], "4")
        self.assertEqual(env["SWEHERO_PROFILER_ACTIVE"], "1")
        self.assertEqual(env["SWEHERO_PROFILER_WARMUP"], "1")
        self.assertEqual(env["SWEHERO_PROFILER_REPEAT"], "2")
        self.assertEqual(env["SWEHERO_PROFILER_SKIP_FIRST"], "1")
        self.assertEqual(env["SWEHERO_PROFILER_SKIP_FIRST_WAIT"], "1")
        self.assertEqual(env["SWEHERO_ENABLE_MEMORY_SNAPSHOT"], "1")
        self.assertEqual(env["SWEHERO_MEMORY_SNAPSHOT_FOLDER"], "profiles/memory")

    def test_run_spec_rejects_changed_launch_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.write_or_validate_run_spec(args, plan, manifest)
            changed = train.parse_args(
                [
                    "--out-dir",
                    str(args.out_dir),
                    "--dataset-path",
                    str(args.dataset_path),
                    "--hf-assets-path",
                    str(args.hf_assets_path),
                    "--buckets",
                    args.buckets,
                    "--bucket-cp",
                    args.bucket_cp,
                    "--max-length",
                    str(args.max_length),
                    "--num-examples",
                    str(args.num_examples),
                    "--max-streamed-examples",
                    str(args.max_streamed_examples),
                    "--learning-rate",
                    "2e-5",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "learning_rate"):
                train.write_or_validate_run_spec(changed, plan, manifest)

    def test_run_spec_rejects_tampered_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.write_or_validate_run_spec(args, plan, manifest)
            spec_path = args.out_dir / train.RUN_SPEC_FILENAME
            spec = json.loads(spec_path.read_text())
            spec["args"]["max_length"] = 123
            spec_path.write_text(json.dumps(spec, indent=2))

            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                train.write_or_validate_run_spec(args, plan, manifest)

    def test_resume_requires_existing_run_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)

            with self.assertRaisesRegex(RuntimeError, "requires an immutable run spec"):
                train.write_or_validate_run_spec(
                    args,
                    plan,
                    manifest,
                    require_existing=True,
                )

    def test_mid_stage_resume_loads_dataloader_state_for_current_stage_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                final_export_dir=train._final_model_export_dir(args.out_dir),
                latest_resumable_step=2,
                latest_model_export_step=None,
                latest_any_step=2,
            )

            flags = train.dataloader_resume_flags_by_stage(plan, resume_state)
            stages = train.stages_to_run_for_resume(plan, resume_state)

            first_env = train.build_stage_env(
                args,
                stage=plan.stages[0],
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=151643,
                load_dataloader_state=flags[plan.stages[0].cumulative_steps],
            )
            second_env = train.build_stage_env(
                args,
                stage=plan.stages[1],
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=151643,
                load_dataloader_state=flags[plan.stages[1].cumulative_steps],
            )

        self.assertEqual([stage.bucket for stage in stages], [8192, 32768])
        self.assertTrue(flags[plan.stages[0].cumulative_steps])
        self.assertFalse(flags[plan.stages[1].cumulative_steps])
        self.assertEqual(first_env["SWEHERO_LOAD_DATALOADER_STATE"], "1")
        self.assertEqual(second_env["SWEHERO_LOAD_DATALOADER_STATE"], "0")

    def test_stage_boundary_resume_does_not_load_previous_bucket_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                final_export_dir=train._final_model_export_dir(args.out_dir),
                latest_resumable_step=4,
                latest_model_export_step=None,
                latest_any_step=4,
            )

            flags = train.dataloader_resume_flags_by_stage(plan, resume_state)
            stages = train.stages_to_run_for_resume(plan, resume_state)

        self.assertEqual([stage.bucket for stage in stages], [32768])
        self.assertFalse(flags[plan.stages[0].cumulative_steps])
        self.assertFalse(flags[plan.stages[1].cumulative_steps])

    def test_swehero_config_can_load_dataloader_state_for_mid_stage_resume(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/experiments/swehero/config_registry.py"
        ).read_text()

        self.assertIn("SWEHERO_LOAD_DATALOADER_STATE", source)
        self.assertIn("_checkpoint_exclude_from_loading()", source)
        self.assertNotIn("exclude_from_loading=[\"dataloader\"]", source)

    def test_swehero_config_wires_profiler_env_controls(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/experiments/swehero/config_registry.py"
        ).read_text()

        self.assertIn("Profiler.Config", source)
        self.assertIn("SWEHERO_ENABLE_PROFILER", source)
        self.assertIn("SWEHERO_PROFILER_FREQ", source)
        self.assertIn("SWEHERO_ENABLE_MEMORY_SNAPSHOT", source)
        self.assertIn("SWEHERO_MEMORY_SNAPSHOT_FOLDER", source)

    def test_swehero_config_routes_final_export_outside_checkpoint_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/experiments/swehero/config_registry.py"
        ).read_text()

        self.assertIn("final_model_export_folder=", source)
        self.assertIn("SWEHERO_FINAL_EXPORT_FOLDER", source)
        self.assertIn("save_last_step_full_checkpoint=", source)
        self.assertIn("SWEHERO_SAVE_FINAL_FULL_CHECKPOINT", source)
        self.assertIn("enable_first_step_checkpoint=", source)
        self.assertIn("SWEHERO_ENABLE_FIRST_STEP_CHECKPOINT", source)
        self.assertIn("first_step_checkpoint_validation_report=", source)
        self.assertIn("SWEHERO_FIRST_STEP_CHECKPOINT_VALIDATION_REPORT", source)
        self.assertIn('"final_export"', source)

    def test_torchtitan_checkpoint_manager_supports_separate_final_exports(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/components/checkpoint.py"
        ).read_text()

        self.assertIn("final_model_export_folder", source)
        self.assertIn("self.final_model_export_folder", source)
        self.assertIn("save_last_step_full_checkpoint", source)
        self.assertIn("Saving a full resumable checkpoint at last step", source)
        self.assertIn("checkpoint.final_model_export_folder must differ", source)
        self.assertIn("first_step_checkpoint_validation_report", source)
        self.assertIn("_validate_first_step_checkpoint", source)

    def test_resume_contract_rejects_changed_training_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train._write_resume_contract(args, plan, manifest)
            changed = train.parse_args(
                [
                    "--out-dir",
                    str(args.out_dir),
                    "--dataset-path",
                    str(args.dataset_path),
                    "--hf-assets-path",
                    str(args.hf_assets_path),
                    "--buckets",
                    args.buckets,
                    "--bucket-cp",
                    args.bucket_cp,
                    "--max-length",
                    str(args.max_length),
                    "--num-examples",
                    str(args.num_examples),
                    "--max-streamed-examples",
                    str(args.max_streamed_examples),
                    "--learning-rate",
                    "2e-5",
                ]
            )
            changed.buckets = ",".join(
                str(b) for b in train.parse_bucket_list(changed.buckets)
            )
            changed.bucket_cp = train._format_bucket_cp_map(
                train.parse_bucket_cp_map(changed.bucket_cp)
            )

            with self.assertRaisesRegex(RuntimeError, "learning_rate"):
                train.validate_resume_contract(changed, plan, manifest)

    def test_varlen_attention_is_rejected_when_any_bucket_uses_cp(self):
        with self.assertRaisesRegex(ValueError, "VarlenAttention"):
            train.validate_bucket_config(
                buckets=(8192, 32768),
                bucket_cp={8192: 1, 32768: 2},
                nproc_per_node=8,
                attention_backend="varlen",
            )

    def test_source_dataset_command_builds_pod_local_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset"
            args = train.parse_args(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--source-dataset-revision",
                    "source-sha",
                    "--source-dataset-rows-per-shard",
                    "123",
                    "--source-dataset-build-batch-size",
                    "17",
                ]
            )
            command = train.build_source_dataset_command(args)

        self.assertIn("prepare_swehero_historical_one_rollout.py", " ".join(command))
        self.assertIn("--dataset-id", command)
        self.assertEqual(command[command.index("--dataset-id") + 1], train.SOURCE_DATASET_ID)
        self.assertIn("--revision", command)
        self.assertEqual(command[command.index("--revision") + 1], "source-sha")
        self.assertIn("--output-dir", command)
        self.assertEqual(command[command.index("--output-dir") + 1], str(dataset_path))
        self.assertIn("--rows-per-shard", command)
        self.assertEqual(command[command.index("--rows-per-shard") + 1], "123")
        self.assertIn("--batch-size", command)
        self.assertEqual(command[command.index("--batch-size") + 1], "17")
        self.assertNotIn("--overwrite", command)

    def test_ensure_training_dataset_does_not_implicitly_overwrite_nonempty_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset"
            dataset_path.mkdir()
            (dataset_path / "notes.txt").write_text("not a parquet dataset")
            args = train.parse_args(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--build-dataset-if-missing",
                ]
            )

            with (
                patch.object(train.subprocess, "run") as run,
                self.assertRaisesRegex(FileExistsError, "rebuild-source-dataset"),
            ):
                train.ensure_training_dataset(args)

            run.assert_not_called()
            self.assertEqual(
                (dataset_path / "notes.txt").read_text(),
                "not a parquet dataset",
            )

    def test_hf_asset_download_pins_model_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--download-hf-assets",
                ]
            )

            with patch.object(train.subprocess, "run") as run:
                train.download_hf_assets_if_requested(args)

        command = run.call_args.args[0]
        self.assertIn("--repo_id", command)
        self.assertEqual(command[command.index("--repo_id") + 1], train.MODEL_ID)
        self.assertIn("--revision", command)
        self.assertEqual(
            command[command.index("--revision") + 1],
            train.MODEL_REVISION,
        )

    def test_dataset_revision_alias_pins_source_revision(self):
        args = train.parse_args(["--dataset-revision", "legacy-sha"])

        self.assertEqual(args.source_dataset_revision, "legacy-sha")

    def test_training_dataset_files_expect_hf_style_parquet_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            data_dir = dataset_dir / "data"
            data_dir.mkdir(parents=True)
            later = data_dir / "train-00001-of-00002.parquet"
            earlier = data_dir / "train-00000-of-00002.parquet"
            later.write_bytes(b"")
            earlier.write_bytes(b"")

            files = train._training_dataset_files(dataset_dir)

        self.assertEqual(files, [earlier, later])

    def test_encode_masks_user_system_and_tool_observations(self):
        example = {
            "trajectory": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "please fix it"},
                {
                    "role": "assistant",
                    "content": "RUN_TESTS",
                    "tool_calls": [
                        {"name": "execute_bash", "arguments": {"cmd": "pytest"}}
                    ],
                },
                {"role": "tool", "content": "secret failing output"},
                {"role": "assistant", "content": "DONE"},
            ]
        }

        encoded = train.encode_swehero_example(
            FakeTokenizer(),
            example,
            max_length=4096,
            min_trainable_tokens=1,
        )

        self.assertIsNotNone(encoded)
        trainable_text = FakeTokenizer().decode(
            label
            for label in encoded["labels"]
            if label != train.IGNORE_INDEX
        )
        self.assertIn("RUN_TESTS", trainable_text)
        self.assertIn("execute_bash", trainable_text)
        self.assertIn("DONE", trainable_text)
        self.assertIn("<tool_call>", trainable_text)
        self.assertIn("<|im_end|>", trainable_text)
        self.assertNotIn("please fix it", trainable_text)
        self.assertNotIn("system prompt", trainable_text)
        self.assertNotIn("secret failing output", trainable_text)
        self.assertNotIn("<|assistant|>", trainable_text)
        self.assertNotIn("<|im_start|>assistant", trainable_text)

    def test_encode_rejects_long_examples_instead_of_truncating(self):
        example = {
            "trajectory": [
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": "x" * 100},
            ],
        }

        with self.assertRaisesRegex(train.LongExampleError, "exceeds --max-length"):
            train.encode_swehero_example(
                FakeTokenizer(),
                example,
                max_length=32,
                min_trainable_tokens=1,
            )

    def test_materialization_errors_on_long_examples_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "64",
                    "--max-length",
                    "64",
                    "--max-streamed-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "too-long",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "x" * 1000},
                ],
            }

            with self.assertRaisesRegex(RuntimeError, "would have been truncated"):
                self._materialize_with_fake_runtime(args, [example])

    def test_materialization_can_explicitly_skip_long_examples_with_manifest_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                    "--long-example-policy",
                    "skip",
                ]
            )
            examples = [
                {
                    "instance_id": "too-long",
                    "trajectory": [
                        {"role": "user", "content": "issue"},
                        {"role": "assistant", "content": "x" * 1000},
                    ],
                },
                {
                    "instance_id": "short",
                    "trajectory": [
                        {"role": "user", "content": "issue"},
                        {"role": "assistant", "content": "OK"},
                    ],
                },
            ]

            manifest = self._materialize_with_fake_runtime(args, examples)

        self.assertEqual(manifest["long_example_policy"], "skip")
        self.assertEqual(manifest["skipped"]["too_long_for_max_length"], 1)
        self.assertEqual(manifest["long_examples_sample"][0]["source_id"], "too-long")
        self.assertEqual(manifest["num_usable_examples"], 1)
        data_provenance = manifest["data_provenance"]
        self.assertEqual(
            data_provenance["schema_version"],
            train.DATA_PROVENANCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            data_provenance["materialization"]["long_example_policy"],
            "skip",
        )
        self.assertEqual(
            data_provenance["streamed"]["source_ids"],
            ["too-long", "short"],
        )
        self.assertEqual(data_provenance["included"]["source_ids"], ["short"])
        self.assertEqual(
            data_provenance["skipped"]["by_reason"]["too_long_for_max_length"][
                "source_ids"
            ],
            ["too-long"],
        )
        self.assertEqual(
            data_provenance["buckets"]["256"]["source_ids"]["source_ids"],
            ["short"],
        )
        self.assertEqual(data_provenance["buckets"]["256"]["record_count"], 1)

    def test_materialization_writes_self_verifying_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }

            manifest = self._materialize_with_fake_runtime(args, [example])
            loaded_manifest = train._load_manifest(args.out_dir)

            self.assertEqual(
                manifest["materialized_data_schema_version"],
                train.MATERIALIZED_DATA_SCHEMA_VERSION,
            )
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(manifest["bucket_counts"], {"256": 1})
            self.assertEqual(manifest["model_revision"], args.model_revision)
            self.assertEqual(
                manifest["model_assets"]["model_revision"],
                args.model_revision,
            )
            self.assertEqual(
                manifest["model_assets"]["schema_version"],
                train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
            )
            self.assertEqual(manifest["model_assets"]["file_count"], 1)
            self.assertEqual(
                manifest["data_provenance"]["schema_version"],
                train.DATA_PROVENANCE_SCHEMA_VERSION,
            )
            self.assertEqual(
                manifest["data_provenance"]["included"]["source_ids"],
                ["short"],
            )
            self.assertEqual(
                manifest["bucket_curriculum"], train.DEFAULT_BUCKET_CURRICULUM
            )
            self.assertEqual(
                manifest["data_provenance"]["materialization"][
                    "bucket_curriculum"
                ],
                train.DEFAULT_BUCKET_CURRICULUM,
            )
            self.assertEqual(
                manifest["data_provenance"]["buckets"]["256"]["integrity"],
                manifest["bucket_file_integrity"]["256"],
            )
            integrity = manifest["bucket_file_integrity"]["256"]
            bucket_path = Path(manifest["bucket_files"]["256"])
            self.assertEqual(integrity["records"], 1)
            self.assertEqual(integrity, train._bucket_file_stats(bucket_path))

    def test_synthetic_smoke_materialization_covers_configured_bucket_cp_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "128,256,512",
                    "--bucket-cp",
                    "128:1,256:2,512:4",
                    "--max-length",
                    "512",
                    "--nproc-per-node",
                    "4",
                    "--global-batch-size",
                    "4",
                    "--num-train-epochs",
                    "4",
                    "--smoke-synthetic-buckets",
                    "--smoke-synthetic-examples-per-bucket",
                    "2",
                ]
            )

            manifest = self._materialize_with_fake_runtime(args, synthetic=True)
            bucket_counts = train._bucket_counts_from_manifest(manifest)
            bucket_files = train._bucket_files_from_manifest(manifest)
            plan = train.build_bucket_plan(
                bucket_counts=bucket_counts,
                bucket_files=bucket_files,
                bucket_cp=train.parse_bucket_cp_map(args.bucket_cp),
                epochs=args.num_train_epochs,
                global_batch_size=args.global_batch_size,
                warmup_ratio=args.warmup_ratio,
            )

            self.assertTrue(manifest["smoke_synthetic_buckets"])
            self.assertEqual(manifest["smoke_synthetic_examples_per_bucket"], 2)
            self.assertEqual(
                manifest["bucket_curriculum"], train.DEFAULT_BUCKET_CURRICULUM
            )
            self.assertTrue(manifest["dataset_artifact"]["synthetic_smoke"])
            self.assertEqual(
                manifest["bucket_counts"], {"128": 2, "256": 2, "512": 2}
            )
            self.assertEqual(manifest["num_usable_examples"], 6)
            self.assertEqual(manifest["streamed_examples_scanned"], 6)
            self.assertTrue(
                manifest["data_provenance"]["materialization"][
                    "smoke_synthetic_buckets"
                ]
            )
            self.assertEqual(
                [stage.bucket for stage in plan.stages], [128, 256, 512]
            )
            self.assertEqual([stage.cp_degree for stage in plan.stages], [1, 2, 4])
            for bucket, path in bucket_files.items():
                rows = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["bucket"] == bucket for row in rows))
                self.assertTrue(all(row["trainable_tokens"] == 1 for row in rows))

    def test_load_manifest_rejects_missing_model_asset_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            manifest = self._materialize_with_fake_runtime(args, [example])
            manifest.pop("model_assets")
            (args.out_dir / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            with self.assertRaisesRegex(RuntimeError, "model_assets provenance"):
                train._load_manifest(args.out_dir)

    def test_load_manifest_rejects_missing_model_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            manifest = self._materialize_with_fake_runtime(args, [example])
            manifest.pop("model_revision")
            (args.out_dir / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            with self.assertRaisesRegex(RuntimeError, "model_revision"):
                train._load_manifest(args.out_dir)

    def test_load_manifest_rejects_inconsistent_data_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            manifest = self._materialize_with_fake_runtime(args, [example])
            manifest["data_provenance"]["included"]["source_ids"] = ["other"]
            (args.out_dir / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            with self.assertRaisesRegex(RuntimeError, "included.sha256"):
                train._load_manifest(args.out_dir)

    def test_main_writes_run_spec_for_dry_run_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            dataset_path = Path(tmp) / "dataset"
            hf_assets_path = Path(tmp) / "hf"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--dataset-path",
                    str(dataset_path),
                    "--hf-assets-path",
                    str(hf_assets_path),
                    "--buckets",
                    "256",
                    "--bucket-cp",
                    "256:1",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            self._materialize_with_fake_runtime(args, [example])

            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    train.main(
                        [
                            "--out-dir",
                            str(out_dir),
                            "--dataset-path",
                            str(dataset_path),
                            "--hf-assets-path",
                            str(hf_assets_path),
                            "--buckets",
                            "256",
                            "--bucket-cp",
                            "256:1",
                            "--max-length",
                            "256",
                            "--num-examples",
                            "1",
                            "--skip-data-prep",
                            "--dry-run",
                        ]
                    )

            run_spec = json.loads((out_dir / train.RUN_SPEC_FILENAME).read_text())
            launcher_plan = json.loads((out_dir / "launcher_plan.json").read_text())

        self.assertEqual(run_spec["args"]["max_length"], 256)
        self.assertFalse(run_spec["args"]["production_mode"])
        self.assertEqual(run_spec["args"]["model_revision"], train.MODEL_REVISION)
        self.assertEqual(
            run_spec["paper_alignment"]["kept"]["base_model_revision"],
            train.MODEL_REVISION,
        )
        self.assertEqual(
            run_spec["manifest"]["model_revision"],
            train.MODEL_REVISION,
        )
        self.assertEqual(
            run_spec["args"]["bucket_curriculum"], train.DEFAULT_BUCKET_CURRICULUM
        )
        self.assertEqual(
            run_spec["plan"]["bucket_curriculum"], train.DEFAULT_BUCKET_CURRICULUM
        )
        self.assertEqual(run_spec["plan"]["distributed"]["nnodes"], 1)
        self.assertEqual(run_spec["plan"]["distributed"]["world_size"], 8)
        self.assertEqual(
            launcher_plan["bucket_curriculum"], train.DEFAULT_BUCKET_CURRICULUM
        )
        self.assertEqual(launcher_plan["distributed"]["nnodes"], 1)
        self.assertEqual(launcher_plan["distributed"]["world_size"], 8)
        self.assertEqual(run_spec["plan"]["total_steps"], 1)
        self.assertEqual(
            run_spec["manifest"]["data_provenance"]["included"]["source_ids"],
            ["short"],
        )
        self.assertEqual(
            launcher_plan["run_spec"],
            str(out_dir / train.RUN_SPEC_FILENAME),
        )
        self.assertEqual(
            launcher_plan["wandb_identity"],
            str(out_dir / train.WANDB_IDENTITY_FILENAME),
        )
        self.assertEqual(
            run_spec["paths"]["runtime_metadata"],
            str(out_dir / train.RUNTIME_METADATA_FILENAME),
        )
        self.assertEqual(
            run_spec["paths"]["workspace_root"],
            str(train._configured_workspace_root(train.parse_args([]))),
        )
        self.assertEqual(
            launcher_plan["runtime_metadata"],
            str(out_dir / train.RUNTIME_METADATA_FILENAME),
        )
        self.assertEqual(run_spec["args"]["post_training_eval_command"], "")
        self.assertEqual(
            run_spec["paths"]["post_training_eval_status"],
            str(out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME),
        )
        self.assertEqual(
            launcher_plan["post_training_eval_status"],
            str(out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME),
        )

    def test_main_writes_wandb_identity_for_dry_run_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            dataset_path = Path(tmp) / "dataset"
            hf_assets_path = Path(tmp) / "hf"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--dataset-path",
                    str(dataset_path),
                    "--hf-assets-path",
                    str(hf_assets_path),
                    "--buckets",
                    "256",
                    "--bucket-cp",
                    "256:1",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            self._materialize_with_fake_runtime(args, [example])

            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    train.main(
                        [
                            "--out-dir",
                            str(out_dir),
                            "--dataset-path",
                            str(dataset_path),
                            "--hf-assets-path",
                            str(hf_assets_path),
                            "--buckets",
                            "256",
                            "--bucket-cp",
                            "256:1",
                            "--max-length",
                            "256",
                            "--num-examples",
                            "1",
                            "--skip-data-prep",
                            "--dry-run",
                            "--enable-wandb",
                            "--wandb-mode",
                            "offline",
                            "--wandb-run-name",
                            "dry-run-wandb",
                        ]
                    )

            identity = json.loads(
                (args.out_dir / train.WANDB_IDENTITY_FILENAME).read_text()
            )
            run_spec = json.loads(
                (args.out_dir / train.RUN_SPEC_FILENAME).read_text()
            )
            launcher_plan = json.loads((args.out_dir / "launcher_plan.json").read_text())

        self.assertEqual(identity["run_name"], "dry-run-wandb")
        self.assertTrue(identity["generated_run_id"])
        self.assertEqual(identity["resume"], "allow")
        self.assertEqual(run_spec["args"]["wandb_run_id"], identity["run_id"])
        self.assertEqual(run_spec["args"]["wandb_resume"], "allow")
        self.assertEqual(
            run_spec["paths"]["wandb_identity"],
            str(args.out_dir / train.WANDB_IDENTITY_FILENAME),
        )
        self.assertEqual(
            launcher_plan["wandb_identity"],
            str(args.out_dir / train.WANDB_IDENTITY_FILENAME),
        )

    def test_load_manifest_rejects_corrupt_bucket_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }

            manifest = self._materialize_with_fake_runtime(args, [example])
            bucket_path = Path(manifest["bucket_files"]["256"])
            with bucket_path.open("a") as handle:
                handle.write('{"unexpected": true}\n')

            with self.assertRaisesRegex(RuntimeError, "record|sha256"):
                train._load_manifest(args.out_dir)

    def test_failed_materialization_does_not_publish_partial_data(self):
        def broken_examples():
            yield {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--max-streamed-examples",
                    "2",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "boom"):
                self._materialize_with_fake_runtime(args, broken_examples())

            self.assertFalse((args.out_dir / "data").exists())
            self.assertEqual(
                list(args.out_dir.glob(".data.tmp-*")),
                [],
            )

    def test_failed_rematerialization_preserves_existing_data(self):
        def broken_examples():
            yield {
                "instance_id": "replacement",
                "trajectory": [
                    {"role": "user", "content": "different issue"},
                    {"role": "assistant", "content": "different answer"},
                ],
            }
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--max-streamed-examples",
                    "2",
                ]
            )
            original_example = {
                "instance_id": "original",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            original_manifest = self._materialize_with_fake_runtime(
                args,
                [original_example],
            )
            original_bucket_path = Path(original_manifest["bucket_files"]["256"])
            original_bucket_bytes = original_bucket_path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "boom"):
                self._materialize_with_fake_runtime(args, broken_examples())

            self.assertEqual(train._load_manifest(args.out_dir), original_manifest)
            self.assertEqual(original_bucket_path.read_bytes(), original_bucket_bytes)
            self.assertEqual(
                list(args.out_dir.glob(".data.tmp-*")),
                [],
            )

    def test_openhands_messages_render_as_qwen_chatml(self):
        example = {
            "trajectory": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "reported issue"},
                {
                    "role": "assistant",
                    "content": "assistant analysis",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "think",
                                "arguments": '{"thought": "consider options"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "environment output"},
            ],
        }

        rendered = "".join(
            text for text, _ in train.qwen_openhands_segments(example)
        )

        self.assertIn(
            "<|im_start|>system\nsystem prompt<|im_end|>\n", rendered
        )
        self.assertIn(
            "<|im_start|>user\nreported issue<|im_end|>\n", rendered
        )
        self.assertIn(
            '<|im_start|>assistant\nassistant analysis\n<tool_call>\n{"name": "think", "arguments": "{\\"thought\\": \\"consider options\\"}"}\n</tool_call><|im_end|>\n',
            rendered,
        )
        self.assertIn(
            "<|im_start|>user\n<tool_response>\nenvironment output\n</tool_response><|im_end|>\n",
            rendered,
        )
        self.assertNotIn("<|system|>", rendered)
        self.assertNotIn("<|assistant|>", rendered)
        self.assertNotIn("<|tool_calls|>", rendered)

    def test_tool_call_serialization_keeps_valid_json_payloads(self):
        text = train._qwen_tool_call_text(
            {
                "function": {
                    "name": 'quoted"tool',
                    "arguments": {"cmd": 'printf "hello"'},
                }
            }
        )
        payload = text.removeprefix("\n<tool_call>\n").removesuffix("\n</tool_call>")

        decoded = json.loads(payload)

        self.assertEqual(decoded["name"], 'quoted"tool')
        self.assertEqual(decoded["arguments"], {"cmd": 'printf "hello"'})

    def test_tool_call_serialization_preserves_argument_type(self):
        string_arguments = train._qwen_tool_call_text(
            {
                "function": {
                    "name": "think",
                    "arguments": '{"thought": "keep as string"}',
                }
            }
        )
        mapping_arguments = train._qwen_tool_call_text(
            {"name": "execute_bash", "arguments": {"cmd": "pytest -q"}}
        )

        self.assertEqual(
            json.loads(
                string_arguments.removeprefix("\n<tool_call>\n").removesuffix(
                    "\n</tool_call>"
                )
            ),
            {"name": "think", "arguments": '{"thought": "keep as string"}'},
        )
        self.assertEqual(
            json.loads(
                mapping_arguments.removeprefix("\n<tool_call>\n").removesuffix(
                    "\n</tool_call>"
                )
            ),
            {"name": "execute_bash", "arguments": {"cmd": "pytest -q"}},
        )

    def test_openhands_messages_match_hf_qwen_chat_template_when_available(self):
        hf_assets = Path(
            os.environ.get(
                "SWEHERO_TEST_HF_ASSETS_PATH",
                "/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct",
            )
        )
        if not hf_assets.exists():
            self.skipTest("Qwen HF tokenizer assets are not available")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            self.skipTest(f"transformers is not available: {exc}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(hf_assets),
            local_files_only=True,
        )
        cases = [
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "reported issue"},
                {
                    "role": "assistant",
                    "content": "assistant analysis",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "think",
                                "arguments": {"thought": "consider options"},
                            },
                        },
                        {
                            "name": "execute_bash",
                            "arguments": {"cmd": "pytest -q"},
                        },
                    ],
                },
                {"role": "tool", "content": "first observation"},
                {"role": "tool", "content": "second observation"},
                {"role": "assistant", "content": "done"},
            ],
            [
                {"role": "user", "content": "reported issue"},
                {"role": "assistant", "content": "done"},
            ],
        ]
        for messages in cases:
            with self.subTest(first_role=messages[0]["role"]):
                expected = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                rendered = "".join(
                    text
                    for text, _is_trainable in train.qwen_openhands_segments(
                        {"trajectory": messages}
                    )
                )

                self.assertEqual(rendered, expected)

    def test_stage_environment_and_torchrun_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(["--out-dir", str(out_dir)])
            stage = train.BucketStage(
                bucket=32768,
                cp_degree=2,
                example_count=4,
                steps=1,
                cumulative_steps=3,
                bucket_file=out_dir / "data" / "bucket_32768.jsonl",
            )
            env = train.build_stage_env(
                args,
                stage=stage,
                total_steps=5,
                warmup_steps=1,
                pad_token_id=151643,
            )
            command = train.build_torchrun_command(args)

        self.assertEqual(env["SWEHERO_BUCKET_CP"], "2")
        self.assertEqual(env["SWEHERO_BUCKET_SEQ_LEN"], "32768")
        self.assertEqual(env["SWEHERO_MODEL_REVISION"], train.MODEL_REVISION)
        self.assertEqual(env["SWEHERO_ENABLE_FP8"], "1")
        self.assertEqual(env["SWEHERO_CUMULATIVE_STEPS"], "3")
        self.assertEqual(env["SWEHERO_OPTIMIZER_IMPL"], "foreach")
        self.assertEqual(env["SWEHERO_TRAINING_DTYPE"], "float32")
        self.assertEqual(env["SWEHERO_MP_PARAM_DTYPE"], "bfloat16")
        self.assertEqual(env["SWEHERO_MP_REDUCE_DTYPE"], "bfloat16")
        self.assertEqual(env["SWEHERO_FSDP_RESHARD_AFTER_FORWARD"], "never")
        self.assertEqual(env["SWEHERO_DETECT_ANOMALY"], "0")
        self.assertEqual(env["CUDA_DEVICE_MAX_CONNECTIONS"], "1")
        self.assertEqual(env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
        self.assertEqual(
            env["SWEHERO_FINAL_EXPORT_FOLDER"],
            train.FINAL_MODEL_EXPORT_FOLDER,
        )
        self.assertEqual(env["SWEHERO_SAVE_FINAL_FULL_CHECKPOINT"], "1")
        self.assertIn("-m", command)
        self.assertIn("torchtitan.train", command)
        self.assertIn("--module", command)
        self.assertIn("swehero", command)
        self.assertIn("--config", command)
        self.assertIn("qwen25_coder7b_direct_to_hero", command)
        self.assertNotIn("--nnodes", command)
        self.assertNotIn("--node_rank", command)
        self.assertIn("localhost:0", command)

    def test_multinode_torchrun_command_requires_explicit_rendezvous(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "1",
                    "--rdzv-endpoint",
                    "train-master:29400",
                    "--rdzv-id",
                    "swehero-run",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            train.validate_launch_inputs(
                args,
                buckets=buckets,
                bucket_cp=bucket_cp,
            )
            command = train.build_torchrun_command(args)
            distributed = train._distributed_launch_summary(args)

        self.assertIn("--nnodes", command)
        self.assertIn("2", command)
        self.assertIn("--node_rank", command)
        self.assertIn("1", command)
        self.assertIn("--rdzv_endpoint", command)
        self.assertIn("train-master:29400", command)
        self.assertIn("--rdzv_id", command)
        self.assertIn("swehero-run", command)
        self.assertEqual(distributed["world_size"], 16)
        self.assertEqual(distributed["rdzv_id"], "swehero-run")

    def test_launch_preflight_checks_executable_assets_and_bucket_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            bucket_file = out_dir / "data" / "bucket_256.jsonl"
            bucket_file.parent.mkdir(parents=True)
            bucket_file.write_text('{"input_ids":[1],"labels":[1]}\n')
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--hf-assets-path",
                    str(hf_assets),
                    "--torchrun-bin",
                    sys.executable,
                ]
            )
            manifest = self._model_assets_manifest(args)
            stage = train.BucketStage(
                bucket=256,
                cp_degree=1,
                example_count=1,
                steps=1,
                cumulative_steps=1,
                bucket_file=bucket_file,
            )
            plan = train.BucketPlan((stage,), total_steps=1, warmup_steps=0)

            summary = train.validate_launch_preflight(args, plan, manifest)
            bucket_file.unlink()
            with self.assertRaisesRegex(RuntimeError, "bucket file"):
                train.validate_launch_preflight(args, plan, manifest)

        self.assertEqual(summary["torchrun_bin"]["resolved"], sys.executable)
        self.assertEqual(summary["bucket_files"][0]["bucket"], 256)

    def test_hf_logits_parity_command_uses_paper_yarn_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            hf_assets = Path(tmp) / "Qwen2.5-Coder-7B-Instruct"
            args = train.parse_args(
                ["--out-dir", str(out_dir), "--hf-assets-path", str(hf_assets)]
            )
            command = train.build_hf_logits_parity_command(args)

        self.assertIn("qwen_swehero_logits_parity.py", " ".join(command))
        self.assertIn("--reference-context", command)
        self.assertEqual(
            command[command.index("--reference-context") + 1],
            "paper-yarn-128k",
        )
        self.assertIn("--reference-model-path", command)
        self.assertEqual(
            command[command.index("--reference-model-path") + 1],
            str(hf_assets),
        )
        self.assertIn("--hf-model-revision", command)
        self.assertEqual(
            command[command.index("--hf-model-revision") + 1],
            train.MODEL_REVISION,
        )
        self.assertIn("--json-out", command)
        self.assertEqual(
            command[command.index("--json-out") + 1],
            str(out_dir / "hf_logits_parity.json"),
        )

    def test_dataparallel_mesh_dims_must_come_from_torch(self):
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "torchtitan/torchtitan/distributed/full_dtensor.py",
            "torchtitan/torchtitan/models/llama3/parallelize.py",
        ):
            source = (repo_root / relative_path).read_text()
            self.assertIn("DataParallelMeshDims", source)
            self.assertNotIn("class DataParallelMeshDims", source)
            self.assertNotIn("Compatibility shim", source)
            self.assertNotIn("except ImportError", source)

    def test_torchtitan_rmsnorm_uses_upstream_forward(self):
        repo_root = Path(__file__).resolve().parents[1]
        source_path = repo_root / "torchtitan/torchtitan/models/common/rmsnorm.py"
        source = source_path.read_text()
        tree = ast.parse(source)
        rmsnorm_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RMSNorm"
        )
        method_names = {
            node.name for node in rmsnorm_class.body if isinstance(node, ast.FunctionDef)
        }

        self.assertNotIn("forward", method_names)
        self.assertNotIn("weight.clone", source)

    def test_pod_setup_uses_pinned_uv(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "scripts/setup_torchtitan_pod_venv.sh").read_text()

        self.assertIn('TORCHTITAN_POD_UV_VERSION="0.11.16"', source)
        self.assertIn("UV_X86_64_UNKNOWN_LINUX_GNU_SHA256=", source)
        self.assertIn("UV_VERSION override is not supported", source)
        self.assertIn("require_uv_version", source)
        self.assertNotIn('UV_VERSION="${UV_VERSION:-', source)


if __name__ == "__main__":
    unittest.main()
