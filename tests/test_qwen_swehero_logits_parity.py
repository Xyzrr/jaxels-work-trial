import argparse
import unittest

from scripts import qwen_swehero_logits_parity as parity


class QwenSweHeroLogitsParityTests(unittest.TestCase):
    def test_paper_yarn_reference_patches_standard_hf_config(self):
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

        self.assertEqual(
            patched["max_position_embeddings"],
            parity.PAPER_CONTEXT_LENGTH,
        )
        self.assertEqual(patched["sliding_window"], parity.PAPER_CONTEXT_LENGTH)
        self.assertEqual(
            patched["rope_scaling"],
            {
                "factor": 4.0,
                "original_max_position_embeddings": parity.QWEN_NATIVE_CONTEXT_LENGTH,
                "type": "yarn",
            },
        )
        self.assertEqual(patched["eos_token_id"], 151645)
        self.assertNotIn("rope_scaling", standard_config)

    def test_standard_reference_leaves_hf_config_unchanged(self):
        standard_config = {
            "model_type": "qwen2",
            "max_position_embeddings": parity.QWEN_NATIVE_CONTEXT_LENGTH,
            "sliding_window": parity.PAPER_CONTEXT_LENGTH,
        }

        patched = parity.patch_hf_config_dict(standard_config, "standard-hf")

        self.assertEqual(patched, standard_config)
        self.assertIsNot(patched, standard_config)

    def test_default_offsets_fit_reference_contexts(self):
        paper_offsets = parity.default_position_offsets("paper-yarn-128k", 16)
        standard_offsets = parity.default_position_offsets("standard-hf", 16)

        self.assertIn(0, paper_offsets)
        self.assertIn(parity.QWEN_NATIVE_CONTEXT_LENGTH - 16, paper_offsets)
        self.assertIn(parity.PAPER_CONTEXT_LENGTH // 2, paper_offsets)
        self.assertIn(parity.PAPER_CONTEXT_LENGTH - 16, paper_offsets)
        self.assertTrue(all(offset + 16 <= parity.PAPER_CONTEXT_LENGTH for offset in paper_offsets))
        self.assertTrue(
            all(
                offset + 16 <= parity.QWEN_NATIVE_CONTEXT_LENGTH
                for offset in standard_offsets
            )
        )

    def test_parse_int_csv_rejects_empty_and_negative_values(self):
        self.assertEqual(parity.parse_int_csv("0, 16,32"), [0, 16, 32])
        with self.assertRaises(argparse.ArgumentTypeError):
            parity.parse_int_csv("")
        with self.assertRaises(argparse.ArgumentTypeError):
            parity.parse_int_csv("0,-1")

    def test_cli_defaults_to_paper_yarn_reference(self):
        args = parity.parse_args([])

        self.assertEqual(args.hf_model_revision, parity.MODEL_REVISION)
        self.assertEqual(args.reference_context, "paper-yarn-128k")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.hf_attn_implementation, "eager")
        self.assertTrue(args.force_math_attention)

    def test_remote_revision_kwargs_only_apply_to_hub_ids(self):
        self.assertEqual(
            parity._remote_revision_kwargs(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                parity.MODEL_REVISION,
            ),
            {"revision": parity.MODEL_REVISION},
        )
        self.assertEqual(parity._remote_revision_kwargs(".", parity.MODEL_REVISION), {})


if __name__ == "__main__":
    unittest.main()
