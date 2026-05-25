from scripts import qwen_swehero_smoke as smoke


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class TestQwenSweHeroSmoke:
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

        assert encoded is not None
        trained_text = "".join(
            chr(token)
            for token, label in zip(encoded["input_ids"], encoded["labels"])
            if label != -100
        )
        assert "assistant analysis" in trained_text
        assert '"name": "think"' in trained_text
        assert "system prompt" not in trained_text
        assert "reported issue" not in trained_text
        assert "environment output" not in trained_text

    def test_effective_batch_defaults_to_paper_global_batch(self):
        config = smoke.effective_batch_config()

        assert config["global_batch_size"] == 32
        assert config["effective_global_batch_size"] == 32
        assert config["gradient_accumulation_steps"] == 32

    def test_default_context_length_matches_paper(self):
        assert smoke.MAX_LENGTH == smoke.PAPER_CONTEXT_LENGTH
        assert smoke.MAX_LENGTH == 131_072

    def test_yarn_config_sets_current_and_legacy_rope_shapes(self):
        class Config:
            rope_theta = 1_000_000.0
            max_position_embeddings = smoke.QWEN_NATIVE_CONTEXT_LENGTH

        config = Config()
        smoke.maybe_enable_yarn(config)

        assert config.max_position_embeddings == smoke.PAPER_CONTEXT_LENGTH
        assert config.rope_parameters["rope_type"] == "yarn"
        assert config.rope_scaling["type"] == "yarn"
        assert config.rope_parameters["factor"] == 4.0

    def test_yarn_config_accepts_explicit_context_length(self):
        class Config:
            rope_theta = 1_000_000.0
            max_position_embeddings = smoke.QWEN_NATIVE_CONTEXT_LENGTH

        config = Config()
        smoke.maybe_enable_yarn(config, max_length=65_536)

        assert config.rope_parameters["factor"] == 2.0
        assert config.rope_scaling["factor"] == 2.0

    def test_cosine_scheduler_accepts_explicit_lr_floor(self):
        lr_lambda = smoke.build_cosine_with_min_lr_lambda(
            4,
            learning_rate=2.0,
            min_learning_rate=1.0,
            warmup_ratio=0.0,
        )

        assert lr_lambda(0) == 1.0
        assert round(abs(lr_lambda(4) - 0.5), 7) == 0
