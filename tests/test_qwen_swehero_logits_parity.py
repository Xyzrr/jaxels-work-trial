"""Unit coverage for the Qwen HF-vs-TorchTitan logits parity preflight.

The full parity script is expensive because it loads model weights twice: once
through Hugging Face and once through TorchTitan's state-dict adapter. These
tests cover the cheap but high-risk configuration pieces before that heavy path
runs.

For non-ML readers: logits are the raw next-token scores produced by a language
model before softmax. The parity script compares logits from both loaders; if
they match for the same token IDs and token positions, the TorchTitan loader is
interpreting the Hugging Face checkpoint the same way the reference model does.
"""

import argparse

import pytest

from scripts import qwen_swehero_logits_parity as parity


class TestQwenSweHeroLogitsParity:
    def test_paper_yarn_reference_patches_standard_hf_config(self):
        # Qwen2.5-Coder's released Hugging Face config is native 32k. The
        # SWE-HERO eval/training recipe uses the paper's 128k context, so the HF
        # reference must be patched with the same YaRN positional scaling that
        # TorchTitan receives.
        standard_config = {
            "model_type": "qwen2",
            "max_position_embeddings": parity.QWEN_NATIVE_CONTEXT_LENGTH,
            "sliding_window": parity.PAPER_CONTEXT_LENGTH,
            "eos_token_id": 151645,
            "vocab_size": 152064,
        }

        patched = parity.patch_hf_config_dict(
            standard_config,
            "paper-yarn-128k",
        )

        assert patched["max_position_embeddings"] == parity.PAPER_CONTEXT_LENGTH
        assert patched["sliding_window"] == parity.PAPER_CONTEXT_LENGTH
        assert patched["rope_scaling"] == {
            "factor": 4.0,
            "original_max_position_embeddings": parity.QWEN_NATIVE_CONTEXT_LENGTH,
            "type": "yarn",
        }
        assert patched["eos_token_id"] == 151645
        assert "rope_scaling" not in standard_config

    def test_standard_reference_leaves_hf_config_unchanged(self):
        # The unpatched reference is a diagnostic mode: it proves ordinary
        # state-dict loading independently from the long-context YaRN override.
        standard_config = {
            "model_type": "qwen2",
            "max_position_embeddings": parity.QWEN_NATIVE_CONTEXT_LENGTH,
            "sliding_window": parity.PAPER_CONTEXT_LENGTH,
        }

        patched = parity.patch_hf_config_dict(standard_config, "standard-hf")

        assert patched == standard_config
        assert patched is not standard_config

    def test_default_offsets_fit_reference_contexts(self):
        # Explicit position offsets let the parity check place the same prompt
        # near risky context boundaries. That catches RoPE/YaRN mismatches that a
        # short prompt at position 0 would never exercise.
        paper_offsets = parity.default_position_offsets("paper-yarn-128k", 16)
        standard_offsets = parity.default_position_offsets("standard-hf", 16)

        assert 0 in paper_offsets
        assert parity.QWEN_NATIVE_CONTEXT_LENGTH - 16 in paper_offsets
        assert parity.PAPER_CONTEXT_LENGTH // 2 in paper_offsets
        assert parity.PAPER_CONTEXT_LENGTH - 16 in paper_offsets
        assert all(
            offset + 16 <= parity.PAPER_CONTEXT_LENGTH for offset in paper_offsets
        )
        assert all(
            offset + 16 <= parity.QWEN_NATIVE_CONTEXT_LENGTH
            for offset in standard_offsets
        )

    def test_parse_int_csv_rejects_empty_and_negative_values(self):
        # Position IDs are zero-based token locations in the model context. Empty
        # and negative samples are nonsensical, so reject them before model load.
        assert parity.parse_int_csv("0, 16,32") == [0, 16, 32]
        with pytest.raises(argparse.ArgumentTypeError):
            parity.parse_int_csv("")
        with pytest.raises(argparse.ArgumentTypeError):
            parity.parse_int_csv("0,-1")

    def test_cli_defaults_to_paper_yarn_reference(self):
        # Default to the expensive-but-relevant check: paper-aligned 128k YaRN,
        # float32 arithmetic, eager HF attention, and math SDPA for TorchTitan.
        # Those defaults minimize numerical noise while testing the production
        # long-context configuration.
        args = parity.parse_args([])

        assert args.hf_model_revision == parity.MODEL_REVISION
        assert args.reference_context == "paper-yarn-128k"
        assert args.dtype == "float32"
        assert args.hf_attn_implementation == "eager"
        assert args.force_math_attention

    def test_remote_revision_kwargs_only_apply_to_hub_ids(self):
        # Local asset directories already pin a concrete model snapshot on disk.
        # Revision kwargs are only needed when the parity script must resolve a
        # Hugging Face Hub model ID.
        assert parity._remote_revision_kwargs(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            parity.MODEL_REVISION,
        ) == {"revision": parity.MODEL_REVISION}
        assert parity._remote_revision_kwargs(".", parity.MODEL_REVISION) == {}
