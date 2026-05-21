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
        self.assertNotIn("please fix it", trainable_text)
        self.assertNotIn("system prompt", trainable_text)
        self.assertNotIn("secret failing output", trainable_text)
        self.assertNotIn("<|assistant|>", trainable_text)

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


if __name__ == "__main__":
    unittest.main()
