import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import openhands_swebench_eval as eval_script

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(args.context_mode, "paper-yarn-128k")
        self.assertEqual(args.max_input_tokens, 131_072)
        self.assertEqual(args.vllm_max_model_len, 131_072)
        self.assertEqual(
            json.loads(args.vllm_rope_scaling),
            {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32_768,
            },
        )
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.max_output_tokens, 4096)
        self.assertEqual(args.eval_note, "swehero-qwen25-coder7b-pass1")
        self.assertIsNone(args.eval_limit)
        self.assertTrue(args.native_tool_calling)
        self.assertEqual(args.tool_choice, "required")
        self.assertTrue(args.tool_call_preflight)
        self.assertEqual(args.docker_smoke_image, "hello-world:latest")
        self.assertFalse(args.skip_docker_run_check)
        self.assertFalse(args.skip_docker_buildx_check)
        self.assertEqual(args.num_workers, 192)
        self.assertEqual(args.vllm_tensor_parallel_size, 1)
        self.assertEqual(args.vllm_pipeline_parallel_size, 1)
        self.assertEqual(args.vllm_server_count, 8)
        self.assertEqual(args.vllm_agent_tasks_per_server, 24)
        self.assertEqual(args.vllm_router_port, 8090)
        self.assertEqual(args.vllm_gpu_memory_utilization, 0.90)
        self.assertEqual(args.vllm_dtype, "bfloat16")
        self.assertEqual(args.vllm_distributed_executor_backend, "mp")
        self.assertTrue(args.vllm_enforce_eager)

    def test_base_url_is_required_for_non_dry_run(self):
        with self.assertRaisesRegex(ValueError, "--base-url is required"):
            eval_script.parse_args(
                [
                    "--output-dir",
                    str(Path(self.tempdir.name) / "run"),
                    "--openhands-dir",
                    str(Path(self.tempdir.name) / "OpenHands"),
                ]
            )

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
        self.assertIn("max_output_tokens = 4096", config)
        self.assertIn("native_tool_calling = true", config)
        self.assertIn('completion_kwargs = { tool_choice = "required" }', config)
        self.assertIn("[agent.swehero_openhands]", config)
        self.assertIn("enable_jupyter = false", config)
        self.assertIn("enable_browsing = false", config)
        self.assertIn("enable_llm_editor = false", config)
        self.assertNotIn("custom_llm_provider", config)

    def test_max_output_tokens_can_be_omitted_for_ablation(self):
        args = self._args("--max-output-tokens", "none")
        config = eval_script.build_openhands_config(args)

        self.assertIsNone(args.max_output_tokens)
        self.assertNotIn("max_output_tokens", config)

    def test_api_key_is_llm_api_key_env_only(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-ignored"}, clear=True):
            args = self._args()

        self.assertEqual(args.api_key, "local-llm")
        self.assertEqual(args.api_key_source, "default")

    def test_write_scaffold_records_commands_without_leaking_real_api_key(self):
        with mock.patch.dict("os.environ", {"LLM_API_KEY": "sk-real-secret"}):
            args = self._args()
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
        self.assertEqual(metadata["context"]["mode"], "paper-yarn-128k")
        self.assertEqual(metadata["context"]["max_input_tokens"], 131_072)
        self.assertEqual(metadata["context"]["vllm_max_model_len"], 131_072)
        self.assertEqual(metadata["context"]["vllm_rope_scaling"]["rope_type"], "yarn")

    def test_vllm_command_enables_qwen_native_tool_calling(self):
        args = self._args()
        _paths, commands = eval_script.write_scaffold(args)

        self.assertIn("--max-model-len", commands.serve_vllm)
        self.assertIn("131072", commands.serve_vllm)
        self.assertIn("--rope-scaling", commands.serve_vllm)
        self.assertIn(
            '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}',
            commands.serve_vllm,
        )
        self.assertIn("--enable-auto-tool-choice", commands.serve_vllm)
        self.assertIn("--tool-call-parser", commands.serve_vllm)
        self.assertIn("hermes", commands.serve_vllm)
        self.assertIn("--tensor-parallel-size", commands.serve_vllm)
        self.assertIn("1", commands.serve_vllm)
        self.assertIn("--pipeline-parallel-size", commands.serve_vllm)
        self.assertIn("1", commands.serve_vllm)
        self.assertIn("--gpu-memory-utilization", commands.serve_vllm)
        self.assertIn("0.9", commands.serve_vllm)
        self.assertIn("--dtype", commands.serve_vllm)
        self.assertIn("bfloat16", commands.serve_vllm)
        self.assertIn("--distributed-executor-backend", commands.serve_vllm)
        self.assertIn("mp", commands.serve_vllm)
        self.assertIn("--enforce-eager", commands.serve_vllm)

    def test_base_native_32k_context_mode_uses_native_context_without_yarn(self):
        args = self._args("--context-mode", "base-native-32k")
        paths, commands = eval_script.write_scaffold(args)

        self.assertEqual(args.max_input_tokens, 32_768)
        self.assertEqual(args.vllm_max_model_len, 32_768)
        self.assertIsNone(args.vllm_rope_scaling)
        self.assertEqual(args.eval_note, "base-native-32k-pass1")
        self.assertNotIn("--rope-scaling", commands.serve_vllm)
        self.assertIn("32768", commands.serve_vllm)
        config = paths.config_path.read_text()
        self.assertIn("max_input_tokens = 32768", config)
        metadata = json.loads(paths.metadata_path.read_text())
        self.assertEqual(metadata["context"]["mode"], "base-native-32k")
        self.assertIsNone(metadata["context"]["vllm_rope_scaling"])

    def test_base_paper_yarn_128k_context_mode_uses_context_matched_yarn(self):
        args = self._args("--context-mode", "base-paper-yarn-128k")
        paths, commands = eval_script.write_scaffold(args)

        self.assertEqual(args.max_input_tokens, 131_072)
        self.assertEqual(args.vllm_max_model_len, 131_072)
        self.assertEqual(args.eval_note, "base-paper-yarn-128k-pass1")
        self.assertIn("--rope-scaling", commands.serve_vllm)
        self.assertIn(
            '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}',
            commands.serve_vllm,
        )
        metadata = json.loads(paths.metadata_path.read_text())
        self.assertEqual(metadata["context"]["mode"], "base-paper-yarn-128k")
        self.assertEqual(metadata["context"]["vllm_rope_scaling"]["rope_type"], "yarn")

    def test_eval_presets_swap_context_contracts(self):
        cases = [
            (
                "openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args",
                "paper-yarn-128k",
                131_072,
                "swehero-qwen25-coder7b-pass1",
            ),
            (
                "openhands-swebench-verified-qwen25-coder-7b-base-native-32k.args",
                "base-native-32k",
                32_768,
                "base-native-32k-pass1",
            ),
            (
                "openhands-swebench-verified-qwen25-coder-7b-base-paper-yarn-128k.args",
                "base-paper-yarn-128k",
                131_072,
                "base-paper-yarn-128k-pass1",
            ),
        ]

        for filename, mode, max_tokens, eval_note in cases:
            preset = REPO_ROOT / "configs" / "eval" / filename
            args = self._args(f"@{preset}")

            self.assertEqual(args.context_mode, mode)
            self.assertEqual(args.max_input_tokens, max_tokens)
            self.assertEqual(args.vllm_max_model_len, max_tokens)
            self.assertEqual(args.eval_note, eval_note)

    def test_base_native_32k_rejects_forced_128k_context(self):
        with self.assertRaisesRegex(ValueError, "base-native-32k requires"):
            self._args(
                "--context-mode",
                "base-native-32k",
                "--max-input-tokens",
                "131072",
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

    def test_docker_preflight_runs_container_and_checks_buildx(self):
        args = self._args()
        completed = [
            subprocess.CompletedProcess(["docker", "info"], 0, stdout="daemon ok\n"),
            subprocess.CompletedProcess(
                ["docker", "run", "--rm", "hello-world:latest"],
                0,
                stdout="Hello from Docker!\n",
            ),
            subprocess.CompletedProcess(
                ["docker", "buildx", "version"],
                0,
                stdout="github.com/docker/buildx 0.30.1\n",
            ),
        ]

        with mock.patch.object(
            eval_script.shutil, "which", return_value="/usr/bin/docker"
        ):
            with mock.patch.object(
                eval_script.subprocess, "run", side_effect=completed
            ) as run:
                checks = eval_script.check_docker_runtime(args)

        self.assertEqual(
            [check.name for check in checks],
            ["docker_daemon", "docker_run", "docker_buildx"],
        )
        self.assertTrue(all(check.ok for check in checks))
        run.assert_any_call(
            ["docker", "run", "--rm", "hello-world:latest"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def test_docker_preflight_reports_unprivileged_run_failure(self):
        args = self._args()
        completed = [
            subprocess.CompletedProcess(["docker", "info"], 0, stdout="daemon ok\n"),
            subprocess.CompletedProcess(
                ["docker", "run", "--rm", "hello-world:latest"],
                1,
                stdout="Error response from daemon: unshare: operation not permitted\n",
            ),
            subprocess.CompletedProcess(
                ["docker", "buildx", "version"],
                0,
                stdout="github.com/docker/buildx 0.30.1\n",
            ),
        ]

        with mock.patch.object(
            eval_script.shutil, "which", return_value="/usr/bin/docker"
        ):
            with mock.patch.object(eval_script.subprocess, "run", side_effect=completed):
                checks = eval_script.check_docker_runtime(args)

        docker_run = next(check for check in checks if check.name == "docker_run")
        self.assertFalse(docker_run.ok)
        self.assertIn("unshare: operation not permitted", docker_run.detail)


if __name__ == "__main__":
    unittest.main()
