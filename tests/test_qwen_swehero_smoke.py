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


if __name__ == "__main__":
    unittest.main()
