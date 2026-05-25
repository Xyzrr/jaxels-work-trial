import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from scripts import openhands_swebench_eval as eval_script

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestOpenHandsSweBenchEval:
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

    def setup_method(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.tempdir.cleanup()

    def test_defaults_match_paper_pass1_eval_settings(self):
        args = self._args()

        assert args.dataset == "princeton-nlp/SWE-bench_Verified"
        assert args.split == "test"
        assert args.agent == "CodeActAgent"
        assert args.max_iterations == 100
        assert args.context_mode == "paper-yarn-128k"
        assert args.max_input_tokens == 131_072
        assert args.vllm_max_model_len == 131_072
        assert json.loads(args.vllm_rope_scaling) == {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32_768,
        }
        assert args.temperature == 0.7
        assert args.top_p == 0.8
        assert args.top_k == 20
        assert args.max_output_tokens == 4096
        assert args.eval_note == "swehero-qwen25-coder7b-pass1"
        assert args.eval_limit is None
        assert args.native_tool_calling
        assert args.tool_choice == "required"
        assert args.tool_call_preflight
        assert args.docker_smoke_image == "hello-world:latest"
        assert not args.skip_docker_run_check
        assert not args.skip_docker_buildx_check
        assert args.num_workers == 192
        assert args.vllm_tensor_parallel_size == 1
        assert args.vllm_pipeline_parallel_size == 1
        assert args.vllm_server_count == 8
        assert args.vllm_agent_tasks_per_server == 24
        assert args.vllm_router_port == 8090
        assert args.vllm_gpu_memory_utilization == 0.90
        assert args.vllm_dtype == "bfloat16"
        assert args.vllm_distributed_executor_backend == "mp"
        assert args.vllm_enforce_eager

    def test_base_url_is_required_for_non_dry_run(self):
        with pytest.raises(ValueError, match="--base-url is required"):
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

        assert str(output).startswith(str(args.output_dir))
        assert "princeton-nlp__SWE-bench_Verified-test" in output.parts
        assert "CodeActAgent" in output.parts
        assert output.name == "output.jsonl"
        assert "Qwen2.5-Coder-7B-Instruct_maxiter_100_N_paper-pass1" in output.parts

    def test_generated_config_disables_non_paper_tools_and_sets_sampling(self):
        args = self._args()
        config = eval_script.build_openhands_config(args)

        assert "[llm.swehero_qwen25_coder7b]" in config
        assert 'model = "openai/Qwen/Qwen2.5-Coder-7B-Instruct"' in config
        assert "temperature = 0.7" in config
        assert "top_p = 0.8" in config
        assert "top_k = 20" in config
        assert "max_input_tokens = 131072" in config
        assert "max_output_tokens = 4096" in config
        assert "native_tool_calling = true" in config
        assert 'completion_kwargs = { tool_choice = "required" }' in config
        assert "[agent.swehero_openhands]" in config
        assert "enable_jupyter = false" in config
        assert "enable_browsing = false" in config
        assert "enable_llm_editor = false" in config
        assert "custom_llm_provider" not in config

    def test_max_output_tokens_can_be_omitted_for_ablation(self):
        args = self._args("--max-output-tokens", "none")
        config = eval_script.build_openhands_config(args)

        assert args.max_output_tokens is None
        assert "max_output_tokens" not in config

    def test_api_key_is_llm_api_key_env_only(self):
        with mock.patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-ignored"}, clear=True
        ):
            args = self._args()

        assert args.api_key == "local-llm"
        assert args.api_key_source == "default"

    def test_write_scaffold_records_commands_without_leaking_real_api_key(self):
        with mock.patch.dict("os.environ", {"LLM_API_KEY": "sk-real-secret"}):
            args = self._args()
            paths, commands = eval_script.write_scaffold(args)

        assert paths.config_path.exists()
        assert paths.commands_path.exists()
        assert "sk-real-secret" in paths.config_path.read_text()
        metadata = json.loads(paths.metadata_path.read_text())
        assert metadata["commands"]["serve_vllm"][8] == "<redacted>"
        assert metadata["model"]["tool_choice"] == "required"
        assert metadata["model"]["tool_call_preflight"]
        assert "sk-real-secret" not in paths.metadata_path.read_text()
        assert "--dataset" in commands.run_infer
        assert "princeton-nlp/SWE-bench_Verified" in commands.run_infer
        assert (
            "evaluation/benchmarks/swe_bench/scripts/eval_infer.sh" in commands.run_eval
        )
        assert metadata["context"]["mode"] == "paper-yarn-128k"
        assert metadata["context"]["max_input_tokens"] == 131_072
        assert metadata["context"]["vllm_max_model_len"] == 131_072
        assert metadata["context"]["vllm_rope_scaling"]["rope_type"] == "yarn"

    def test_vllm_command_enables_qwen_native_tool_calling(self):
        args = self._args()
        _paths, commands = eval_script.write_scaffold(args)

        assert "--max-model-len" in commands.serve_vllm
        assert "131072" in commands.serve_vllm
        assert "--rope-scaling" in commands.serve_vllm
        assert (
            '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
            in commands.serve_vllm
        )
        assert "--enable-auto-tool-choice" in commands.serve_vllm
        assert "--tool-call-parser" in commands.serve_vllm
        assert "hermes" in commands.serve_vllm
        assert "--tensor-parallel-size" in commands.serve_vllm
        assert "1" in commands.serve_vllm
        assert "--pipeline-parallel-size" in commands.serve_vllm
        assert "1" in commands.serve_vllm
        assert "--gpu-memory-utilization" in commands.serve_vllm
        assert "0.9" in commands.serve_vllm
        assert "--dtype" in commands.serve_vllm
        assert "bfloat16" in commands.serve_vllm
        assert "--distributed-executor-backend" in commands.serve_vllm
        assert "mp" in commands.serve_vllm
        assert "--enforce-eager" in commands.serve_vllm

    def test_base_native_32k_context_mode_uses_native_context_without_yarn(self):
        args = self._args("--context-mode", "base-native-32k")
        paths, commands = eval_script.write_scaffold(args)

        assert args.max_input_tokens == 32_768
        assert args.vllm_max_model_len == 32_768
        assert args.vllm_rope_scaling is None
        assert args.eval_note == "base-native-32k-pass1"
        assert "--rope-scaling" not in commands.serve_vllm
        assert "32768" in commands.serve_vllm
        config = paths.config_path.read_text()
        assert "max_input_tokens = 32768" in config
        metadata = json.loads(paths.metadata_path.read_text())
        assert metadata["context"]["mode"] == "base-native-32k"
        assert metadata["context"]["vllm_rope_scaling"] is None

    def test_base_paper_yarn_128k_context_mode_uses_context_matched_yarn(self):
        args = self._args("--context-mode", "base-paper-yarn-128k")
        paths, commands = eval_script.write_scaffold(args)

        assert args.max_input_tokens == 131_072
        assert args.vllm_max_model_len == 131_072
        assert args.eval_note == "base-paper-yarn-128k-pass1"
        assert "--rope-scaling" in commands.serve_vllm
        assert (
            '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
            in commands.serve_vllm
        )
        metadata = json.loads(paths.metadata_path.read_text())
        assert metadata["context"]["mode"] == "base-paper-yarn-128k"
        assert metadata["context"]["vllm_rope_scaling"]["rope_type"] == "yarn"

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
            (
                "openhands-swebench-verified-swe-lego-qwen3-8b.args",
                "swe-lego-qwen3-160k",
                147_456,
                "swe-lego-qwen3-8b-pass1",
            ),
        ]

        for filename, mode, max_tokens, eval_note in cases:
            preset = REPO_ROOT / "configs" / "eval" / filename
            args = self._args(f"@{preset}")

            assert args.context_mode == mode
            assert args.max_input_tokens == max_tokens
            if mode == "swe-lego-qwen3-160k":
                assert args.vllm_max_model_len == 163_840
            else:
                assert args.vllm_max_model_len == max_tokens
            assert args.eval_note == eval_note

    def test_swe_lego_qwen3_preset_matches_upstream_serving_and_llm_contract(self):
        preset = (
            REPO_ROOT
            / "configs"
            / "eval"
            / "openhands-swebench-verified-swe-lego-qwen3-8b.args"
        )
        args = self._args(f"@{preset}")
        paths, commands = eval_script.write_scaffold(args)

        assert args.eval_stack == "swe-lego"
        assert args.model_id == "SWE-Lego/SWE-Lego-Qwen3-8B"
        assert args.served_model_name == "Qwen/Qwen3-8B"
        assert args.temperature == 0.0
        assert args.top_p is None
        assert args.top_k is None
        assert args.max_input_tokens == 147_456
        assert args.max_output_tokens == 16_384
        assert args.vllm_max_model_len == 163_840
        assert args.vllm_rope_scaling is None
        assert args.vllm_tensor_parallel_size == 8
        assert args.vllm_server_count == 1
        assert not args.vllm_use_router
        assert args.vllm_max_num_seqs == 24
        assert not (
            args.native_tool_calling and not args.omit_native_tool_calling_config
        )
        assert not args.tool_call_preflight
        assert args.num_workers == 24
        assert args.swebench_cache_level == "instance"
        assert args.swebench_timeout == 500
        assert args.swebench_max_workers == 10

        config = paths.config_path.read_text()
        assert "[llm.eval_qwen3_8b]" in config
        assert 'model = "openai/Qwen/Qwen3-8B"' in config
        assert "temperature = 0.0" in config
        assert "max_input_tokens = 147456" in config
        assert "max_output_tokens = 16384" in config
        assert "top_p" not in config
        assert "top_k" not in config
        assert "native_tool_calling" not in config
        assert "tool_choice" not in config

        assert "--tensor-parallel-size" in commands.serve_vllm
        assert "8" in commands.serve_vllm
        assert "--max-model-len" in commands.serve_vllm
        assert "163840" in commands.serve_vllm
        assert "--max-num-seqs" in commands.serve_vllm
        assert "24" in commands.serve_vllm
        assert "--rope-scaling" not in commands.serve_vllm
        assert "--enable-auto-tool-choice" not in commands.serve_vllm
        assert commands.convert_output is not None
        assert any(
            "convert_oh_output_to_swe_json.py" in part
            for part in commands.convert_output
        )
        assert "swebench.harness.run_evaluation" in commands.run_eval
        assert "--cache_level" in commands.run_eval
        assert "instance" in commands.run_eval
        assert "--timeout" in commands.run_eval
        assert "500" in commands.run_eval

    def test_base_native_32k_rejects_forced_128k_context(self):
        with pytest.raises(ValueError, match="base-native-32k requires"):
            self._args(
                "--context-mode",
                "base-native-32k",
                "--max-input-tokens",
                "131072",
            )

    def test_eval_ids_are_passed_to_openhands_and_conflict_with_eval_limit(self):
        args = self._args("--eval-ids", "django__django-13670,sympy__sympy-15599")
        _paths, commands = eval_script.write_scaffold(args)

        assert "--eval-ids" in commands.run_infer
        assert "django__django-13670,sympy__sympy-15599" in commands.run_infer

        with pytest.raises(ValueError, match="mutually exclusive"):
            self._args(
                "--eval-limit",
                "1",
                "--eval-ids",
                "django__django-13670",
            )

    def test_selected_id_filter_config_is_restored_after_use(self):
        openhands_dir = Path(self.tempdir.name) / "OpenHands-filter"
        config_dir = openhands_dir / "evaluation" / "benchmarks" / "swe_bench"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.toml"
        config_path.write_text('selected_repos = ["django/django"]\n')

        state = eval_script.write_swebench_filter_config(
            openhands_dir, ["django__django-13670", "sympy__sympy-15599"]
        )
        assert '"django__django-13670"' in config_path.read_text()
        assert '"sympy__sympy-15599"' in config_path.read_text()

        eval_script.restore_swebench_filter_config(*state)

        assert config_path.read_text() == 'selected_repos = ["django/django"]\n'

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

        assert summary["resolved"] == 2
        assert summary["total"] == 3
        assert round(abs(summary["pass_at_1"] - 2 / 3), 7) == 0

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

        assert summary["resolved"] == 0
        assert summary["total"] == 1
        assert summary["pass_at_1"] == 0

    def test_tool_call_preflight_payload_targets_served_model(self):
        args = self._args("--served-model-name", "qwen-7b")

        payload = eval_script.tool_call_preflight_payload(args)

        assert payload["model"] == "qwen-7b"
        assert payload["tool_choice"] == "required"
        assert (
            payload["tools"][0]["function"]["name"]
            == eval_script.TOOL_CALL_PREFLIGHT_NAME
        )

    def test_tool_choice_override_updates_config_and_preflight(self):
        args = self._args("--tool-choice", "auto")

        config = eval_script.build_openhands_config(args)
        payload = eval_script.tool_call_preflight_payload(args)

        assert 'completion_kwargs = { tool_choice = "auto" }' in config
        assert payload["tool_choice"] == "auto"

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
                                        "arguments": '{"status":"ok"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )

        assert check.ok

    def test_tool_call_preflight_rejects_plain_message_text(self):
        check = eval_script.validate_tool_call_response(
            {"choices": [{"message": {"content": "I will call the tool now."}}]}
        )

        assert not check.ok
        assert "message.tool_calls missing" in check.detail

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

        assert summary["used_real_tools"]
        assert summary["agent_tool_actions"] == 2
        assert summary["agent_message_actions"] == 2
        assert summary["tool_action_counts"] == {"think": 1, "run": 1}
        assert summary["loop_errors"] == 1

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

        assert [check.name for check in checks] == [
            "docker_daemon",
            "docker_run",
            "docker_buildx",
        ]
        assert all(check.ok for check in checks)
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
            with mock.patch.object(
                eval_script.subprocess, "run", side_effect=completed
            ):
                checks = eval_script.check_docker_runtime(args)

        docker_run = next(check for check in checks if check.name == "docker_run")
        assert not docker_run.ok
        assert "unshare: operation not permitted" in docker_run.detail
