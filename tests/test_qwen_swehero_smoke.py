import unittest

from scripts import qwen_swehero_smoke as smoke


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class QwenSweHeroSmokeTests(unittest.TestCase):
    def test_encode_example_masks_non_assistant_turns(self):
        example = {
            "trajectory": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "reported issue"},
                {
                    "role": "assistant",
                    "content": "assistant analysis",
                    "tool_calls": [{"function": {"name": "think"}}],
                },
                {"role": "tool", "content": "environment output"},
            ],
        }

        encoded = smoke.encode_example(
            FakeTokenizer(), example, max_length=10_000, include_model_patch=False
        )

        self.assertIsNotNone(encoded)
        trained_text = "".join(
            chr(token)
            for token, label in zip(encoded["input_ids"], encoded["labels"])
            if label != -100
        )
        self.assertIn("assistant analysis", trained_text)
        self.assertIn('"name": "think"', trained_text)
        self.assertNotIn("system prompt", trained_text)
        self.assertNotIn("reported issue", trained_text)
        self.assertNotIn("environment output", trained_text)

    def test_effective_batch_defaults_to_paper_global_batch(self):
        config = smoke.effective_batch_config()

        self.assertEqual(config["global_batch_size"], 32)
        self.assertEqual(config["effective_global_batch_size"], 32)
        self.assertEqual(config["gradient_accumulation_steps"], 32)

    def test_default_context_length_matches_paper(self):
        self.assertEqual(smoke.MAX_LENGTH, smoke.PAPER_CONTEXT_LENGTH)
        self.assertEqual(smoke.MAX_LENGTH, 131_072)

    def test_yarn_config_sets_current_and_legacy_rope_shapes(self):
        class Config:
            rope_theta = 1_000_000.0
            max_position_embeddings = smoke.QWEN_NATIVE_CONTEXT_LENGTH

        config = Config()
        smoke.maybe_enable_yarn(config)

        self.assertEqual(config.max_position_embeddings, smoke.PAPER_CONTEXT_LENGTH)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_scaling["type"], "yarn")
        self.assertEqual(config.rope_parameters["factor"], 4.0)

    def test_yarn_config_accepts_explicit_context_length(self):
        class Config:
            rope_theta = 1_000_000.0
            max_position_embeddings = smoke.QWEN_NATIVE_CONTEXT_LENGTH

        config = Config()
        smoke.maybe_enable_yarn(config, max_length=65_536)

        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rope_scaling["factor"], 2.0)

    def test_cosine_scheduler_accepts_explicit_lr_floor(self):
        lr_lambda = smoke.build_cosine_with_min_lr_lambda(
            4,
            learning_rate=2.0,
            min_learning_rate=1.0,
            warmup_ratio=0.0,
        )

        self.assertEqual(lr_lambda(0), 1.0)
        self.assertAlmostEqual(lr_lambda(4), 0.5)


if __name__ == "__main__":
    unittest.main()
