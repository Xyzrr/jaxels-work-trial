"""Tests for the lightweight Qwen SWE-HERO smoke-training script.

The smoke script is an older, environment-driven path for quickly proving that a
small Qwen/OpenHands training loop can run. These tests do not measure model
quality; they lock down the ML-facing assumptions that make the smoke run
representative enough to catch configuration drift before a larger TorchTitan
training run.
"""

from __future__ import annotations

from scripts import qwen_swehero_smoke as smoke


class FakeTokenizer:
    """Deterministic character tokenizer for loss-mask assertions.

    The real smoke run uses Qwen's tokenizer, where tokens are subword pieces and
    special chat markers. For these tests, one character per token makes it easy
    to reconstruct exactly which text was marked trainable.
    """

    bos_token_id = None
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


class TestQwenSweHeroSmoke:
    def test_encode_example_masks_non_assistant_turns(self) -> None:
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
        # Supervised fine-tuning trains on labels, not merely input text. The
        # system prompt, user issue, and tool observation stay visible in
        # input_ids as context, but label -100 tells PyTorch cross entropy not to
        # train the model to imitate those non-assistant tokens.
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

    def test_effective_batch_defaults_to_paper_global_batch(self) -> None:
        config = smoke.effective_batch_config()

        # The paper-facing recipe uses global batch size 32. On a single-process
        # smoke run with per-device batch size 1, gradient accumulation performs
        # 32 small forward/backward passes before one optimizer update so the
        # effective batch still matches the recipe.
        assert config["global_batch_size"] == 32
        assert config["effective_global_batch_size"] == 32
        assert config["gradient_accumulation_steps"] == 32

    def test_default_context_length_matches_paper(self) -> None:
        # 128k tokens lets a long OpenHands trajectory keep earlier context while
        # later assistant actions are supervised. This is intentionally larger
        # than Qwen2.5's native 32k context and therefore relies on YaRN below.
        assert smoke.MAX_LENGTH == smoke.PAPER_CONTEXT_LENGTH
        assert smoke.MAX_LENGTH == 131_072

    def test_yarn_config_sets_current_and_legacy_rope_shapes(self) -> None:
        class Config:
            rope_theta = 1_000_000.0
            max_position_embeddings = smoke.QWEN_NATIVE_CONTEXT_LENGTH

        config = Config()
        smoke.maybe_enable_yarn(config)

        # RoPE is Qwen's positional encoding. YaRN rescales it so positions above
        # the native 32k window are represented intentionally. The script writes
        # both current and legacy config shapes because different Transformers
        # versions read different field names.
        assert config.max_position_embeddings == smoke.PAPER_CONTEXT_LENGTH
        assert config.rope_parameters["rope_type"] == "yarn"
        assert config.rope_scaling["type"] == "yarn"
        assert config.rope_parameters["factor"] == 4.0

    def test_yarn_config_accepts_explicit_context_length(self) -> None:
        class Config:
            rope_theta = 1_000_000.0
            max_position_embeddings = smoke.QWEN_NATIVE_CONTEXT_LENGTH

        config = Config()
        smoke.maybe_enable_yarn(config, max_length=65_536)

        # The scaling factor is target_context / native_context. A 65,536-token
        # smoke context is 2x Qwen2.5's native 32,768 positions.
        assert config.rope_parameters["factor"] == 2.0
        assert config.rope_scaling["factor"] == 2.0

    def test_cosine_scheduler_accepts_explicit_lr_floor(self) -> None:
        lr_lambda = smoke.build_cosine_with_min_lr_lambda(
            4,
            learning_rate=2.0,
            min_learning_rate=1.0,
            warmup_ratio=0.0,
        )

        # The scheduler returns a multiplier for the optimizer's base learning
        # rate. With a floor of 1.0 under a base LR of 2.0, the final multiplier
        # is 0.5 instead of decaying all the way to zero.
        assert lr_lambda(0) == 1.0
        assert round(abs(lr_lambda(4) - 0.5), 7) == 0
