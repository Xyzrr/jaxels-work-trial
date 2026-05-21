import json
import tempfile
import unittest
from pathlib import Path

from scripts import openhands_swebench_eval as eval_script


class OpenHandsSweBenchEvalTests(unittest.TestCase):
    def _args(self, *extra: str):
        temp_root = Path(self.tempdir.name)
        return eval_script.parse_args(
            [
                "--dry-run",
                "--output-dir",
                str(temp_root / "run"),
                "--openhands-dir",
                str(temp_root / "OpenHands"),
                *extra,
            ]
        )

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_defaults_match_paper_pass1_eval_settings(self):
        args = self._args()

        self.assertEqual(args.dataset, "princeton-nlp/SWE-bench_Verified")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.agent, "CodeActAgent")
        self.assertEqual(args.max_iterations, 100)
        self.assertEqual(args.max_input_tokens, 131_072)
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.top_k, 20)
        self.assertIsNone(args.eval_limit)
        self.assertTrue(args.native_tool_calling)

    def test_expected_output_path_matches_openhands_layout(self):
        args = self._args("--eval-note", "paper-pass1", "--eval-limit", "1")
        output = eval_script.expected_output_jsonl(args)

        self.assertTrue(str(output).startswith(str(args.output_dir)))
        self.assertIn("princeton-nlp__SWE-bench_Verified-test", output.parts)
        self.assertIn("CodeActAgent", output.parts)
        self.assertEqual(output.name, "output.jsonl")
        self.assertIn(
            "Qwen2.5-Coder-7B-Instruct_maxiter_100_N_paper-pass1",
            output.parts,
        )

    def test_generated_config_disables_non_paper_tools_and_sets_sampling(self):
        args = self._args()
        config = eval_script.build_openhands_config(args)

        self.assertIn("[llm.swehero_qwen25_coder7b]", config)
        self.assertIn('model = "openai/Qwen/Qwen2.5-Coder-7B-Instruct"', config)
        self.assertIn("temperature = 0.7", config)
        self.assertIn("top_p = 0.8", config)
        self.assertIn("top_k = 20", config)
        self.assertIn("max_input_tokens = 131072", config)
        self.assertIn("native_tool_calling = true", config)
        self.assertIn("[agent.swehero_openhands]", config)
        self.assertIn("enable_jupyter = false", config)
        self.assertIn("enable_browsing = false", config)
        self.assertIn("enable_llm_editor = false", config)
        self.assertNotIn("custom_llm_provider", config)

    def test_generated_config_can_omit_client_top_k_for_strict_local_servers(self):
        args = self._args("--no-send-top-k-param")
        config = eval_script.build_openhands_config(args)

        self.assertIn("temperature = 0.7", config)
        self.assertIn("top_p = 0.8", config)
        self.assertNotIn("top_k =", config)

    def test_write_scaffold_records_commands_without_leaking_real_api_key(self):
        args = self._args("--api-key", "sk-real-secret")
        paths, commands = eval_script.write_scaffold(args)

        self.assertTrue(paths.config_path.exists())
        self.assertTrue(paths.commands_path.exists())
        self.assertIn("sk-real-secret", paths.config_path.read_text())
        metadata = json.loads(paths.metadata_path.read_text())
        self.assertEqual(metadata["commands"]["serve_vllm"][8], "<redacted>")
        self.assertTrue(metadata["model"]["send_top_k_param"])
        self.assertNotIn("sk-real-secret", paths.metadata_path.read_text())
        self.assertIn("--dataset", commands.run_infer)
        self.assertIn("princeton-nlp/SWE-bench_Verified", commands.run_infer)
        self.assertIn(
            "evaluation/benchmarks/swe_bench/scripts/eval_infer.sh",
            commands.run_eval,
        )

    def test_summarize_report_computes_pass_at_1_from_resolved_ids(self):
        report_path = Path(self.tempdir.name) / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "resolved_ids": ["django__django-1", "sympy__sympy-2"],
                    "unresolved_ids": ["pytest__pytest-3"],
                    "error_ids": [],
                }
            )
        )

        summary = eval_script.summarize_report(report_path)

        self.assertEqual(summary["resolved"], 2)
        self.assertEqual(summary["total"], 3)
        self.assertAlmostEqual(summary["pass_at_1"], 2 / 3)

    def test_summarize_report_uses_submitted_instances_for_limited_runs(self):
        report_path = Path(self.tempdir.name) / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "total_instances": 500,
                    "submitted_instances": 1,
                    "resolved_instances": 0,
                    "empty_patch_instances": 1,
                }
            )
        )

        summary = eval_script.summarize_report(report_path)

        self.assertEqual(summary["resolved"], 0)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["pass_at_1"], 0)


if __name__ == "__main__":
    unittest.main()
