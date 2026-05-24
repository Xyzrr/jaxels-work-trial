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
        self.assertEqual(args.tool_choice, "required")
        self.assertTrue(args.tool_call_preflight)

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
        self.assertIn('completion_kwargs = { tool_choice = "required" }', config)
        self.assertIn("[agent.swehero_openhands]", config)
        self.assertIn("enable_jupyter = false", config)
        self.assertIn("enable_browsing = false", config)
        self.assertIn("enable_llm_editor = false", config)
        self.assertNotIn("custom_llm_provider", config)

    def test_write_scaffold_records_commands_without_leaking_real_api_key(self):
        args = self._args("--api-key", "sk-real-secret")
        paths, commands = eval_script.write_scaffold(args)

        self.assertTrue(paths.config_path.exists())
        self.assertTrue(paths.commands_path.exists())
        self.assertIn("sk-real-secret", paths.config_path.read_text())
        metadata = json.loads(paths.metadata_path.read_text())
        self.assertEqual(metadata["commands"]["serve_vllm"][8], "<redacted>")
        self.assertEqual(metadata["model"]["tool_choice"], "required")
        self.assertTrue(metadata["model"]["tool_call_preflight"])
        self.assertNotIn("sk-real-secret", paths.metadata_path.read_text())
        self.assertIn("--dataset", commands.run_infer)
        self.assertIn("princeton-nlp/SWE-bench_Verified", commands.run_infer)
        self.assertIn(
            "evaluation/benchmarks/swe_bench/scripts/eval_infer.sh",
            commands.run_eval,
        )

    def test_vllm_command_enables_qwen_native_tool_calling(self):
        args = self._args()
        _paths, commands = eval_script.write_scaffold(args)

        self.assertIn("--enable-auto-tool-choice", commands.serve_vllm)
        self.assertIn("--tool-call-parser", commands.serve_vllm)
        self.assertIn("hermes", commands.serve_vllm)

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

    def test_tool_call_preflight_payload_targets_served_model(self):
        args = self._args("--served-model-name", "qwen-7b")

        payload = eval_script.tool_call_preflight_payload(args)

        self.assertEqual(payload["model"], "qwen-7b")
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(
            payload["tools"][0]["function"]["name"],
            eval_script.TOOL_CALL_PREFLIGHT_NAME,
        )

    def test_tool_choice_override_updates_config_and_preflight(self):
        args = self._args("--tool-choice", "auto")

        config = eval_script.build_openhands_config(args)
        payload = eval_script.tool_call_preflight_payload(args)

        self.assertIn('completion_kwargs = { tool_choice = "auto" }', config)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_tool_call_preflight_accepts_structured_tool_calls(self):
        check = eval_script.validate_tool_call_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": eval_script.TOOL_CALL_PREFLIGHT_NAME,
                                        "arguments": "{\"status\":\"ok\"}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )

        self.assertTrue(check.ok)

    def test_tool_call_preflight_rejects_plain_message_text(self):
        check = eval_script.validate_tool_call_response(
            {"choices": [{"message": {"content": "I will call the tool now."}}]}
        )

        self.assertFalse(check.ok)
        self.assertIn("message.tool_calls missing", check.detail)

    def test_summarize_agent_tool_use_counts_real_tools_and_loops(self):
        output_path = Path(self.tempdir.name) / "output.jsonl"
        output_path.write_text(
            json.dumps(
                {
                    "history": [
                        {"source": "agent", "action": "system"},
                        {"source": "agent", "action": "message"},
                        {"source": "agent", "action": "think"},
                        {"source": "agent", "action": "run"},
                    ],
                    "error": None,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "history": [{"source": "agent", "action": "message"}],
                    "error": "AgentStuckInLoopError: Agent got stuck in a loop",
                }
            )
            + "\n"
        )

        summary = eval_script.summarize_agent_tool_use(output_path)

        self.assertTrue(summary["used_real_tools"])
        self.assertEqual(summary["agent_tool_actions"], 2)
        self.assertEqual(summary["agent_message_actions"], 2)
        self.assertEqual(summary["tool_action_counts"], {"think": 1, "run": 1})
        self.assertEqual(summary["loop_errors"], 1)


if __name__ == "__main__":
    unittest.main()
