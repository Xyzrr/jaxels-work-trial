import ast
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
    def test_defaults_track_paper_hyperparameters_and_target_pod(self):
        args = train.parse_args([])

        self.assertEqual(args.model_id, train.MODEL_ID)
        self.assertEqual(args.dataset_id, train.DATASET_ID)
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

    def test_varlen_attention_is_rejected_when_any_bucket_uses_cp(self):
        with self.assertRaisesRegex(ValueError, "VarlenAttention"):
            train.validate_bucket_config(
                buckets=(8192, 32768),
                bucket_cp={8192: 1, 32768: 2},
                nproc_per_node=8,
                attention_backend="varlen",
            )

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
