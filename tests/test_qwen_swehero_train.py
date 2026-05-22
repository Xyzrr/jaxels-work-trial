import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts import qwen_swehero_train as train


class FakeTokenizer:
    bos_id = None
    eos_id = None

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

    def test_defaults_track_paper_hyperparameters_and_target_pod(self):
        args = train.parse_args([])

        self.assertEqual(args.model_id, train.MODEL_ID)
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
        self.assertTrue(args.enable_fp8)
        self.assertEqual(args.attention_backend, "sdpa")
        self.assertEqual(train.parse_bucket_list(args.buckets), train.DEFAULT_BUCKETS)

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

    def test_resume_requires_existing_full_dcp_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run"), "--resume"])
            with self.assertRaises(FileNotFoundError):
                train.validate_resume_request(args)

            checkpoint_dir = Path(tmp) / "run" / "torchtitan" / "checkpoint" / "step-5"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.safetensors.index.json").write_text("{}")

            with self.assertRaisesRegex(RuntimeError, "full DCP checkpoint"):
                train.validate_resume_request(args)

    def test_resume_rejects_destructive_refresh_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            (out_dir / "torchtitan" / "checkpoint" / "step-1").mkdir(parents=True)
            args = train.parse_args(
                ["--out-dir", str(out_dir), "--resume", "--overwrite-output"]
            )

            with self.assertRaisesRegex(ValueError, "overwrite-output"):
                train.validate_resume_request(args)

    def test_resume_ignores_final_model_export_when_deciding_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = args.out_dir / "torchtitan" / "checkpoint"
            full = checkpoint_root / "step-4"
            export = checkpoint_root / "step-5"
            full.mkdir(parents=True)
            export.mkdir()
            (full / ".metadata").write_text("{}")
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)
            train.validate_resume_progress(plan, resume_state)

        self.assertEqual(resume_state.latest_resumable_step, 4)
        self.assertEqual(resume_state.latest_any_step, 5)
        self.assertEqual(train.stages_to_run_for_resume(plan, resume_state), ())

    def test_resume_rejects_nonfinal_model_export_newer_than_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = args.out_dir / "torchtitan" / "checkpoint"
            full = checkpoint_root / "step-2"
            export = checkpoint_root / "step-3"
            full.mkdir(parents=True)
            export.mkdir()
            (full / ".metadata").write_text("{}")
            (export / "model.safetensors.index.json").write_text("{}")
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                latest_resumable_step=2,
                latest_any_step=3,
            )

            with self.assertRaisesRegex(RuntimeError, "non-resumable export"):
                train.validate_resume_progress(plan, resume_state)

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

    def test_mid_stage_resume_loads_dataloader_state_for_current_stage_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = args.out_dir / "torchtitan" / "checkpoint"
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                latest_resumable_step=2,
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
            checkpoint_root = args.out_dir / "torchtitan" / "checkpoint"
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                latest_resumable_step=4,
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
        self.assertEqual(env["SWEHERO_ENABLE_FP8"], "1")
        self.assertEqual(env["SWEHERO_CUMULATIVE_STEPS"], "3")
        self.assertIn("-m", command)
        self.assertIn("torchtitan.train", command)
        self.assertIn("--module", command)
        self.assertIn("swehero", command)
        self.assertIn("--config", command)
        self.assertIn("qwen25_coder7b_direct_to_hero", command)

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
