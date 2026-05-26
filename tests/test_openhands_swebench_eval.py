"""Tests for the OpenHands SWE-bench eval scaffold.

These tests verify experiment contracts rather than model quality. A successful
run means the wrapper will launch OpenHands, vLLM, and the SWE-bench grader with
the intended model-serving settings; it does not mean the model will solve any
repository issue. The comments in this file explain why each ML-facing default
matters so engineers who are comfortable with Python/Kubernetes but new to LLM
evals can tell which assertions are paper fidelity, which are control baselines,
and which are infrastructure safety checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from scripts import openhands_swebench_eval as eval_script

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestOpenHandsSweBenchEval:
    """Exercise the eval wrapper without cloning OpenHands or running GPUs."""

    def _args(self, *extra: str) -> argparse.Namespace:
        temp_root = Path(self.tempdir.name)
        return eval_script.parse_args(
            [
                # Dry-run lets the tests inspect generated configs and commands.
                # Real OpenHands inference would require a running vLLM model
                # server and Dockerized SWE-bench evaluation containers.
                "--dry-run",
                "--output-dir",
                str(temp_root / "run"),
                "--openhands-dir",
                str(temp_root / "OpenHands"),
                *extra,
            ]
        )

    def setup_method(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def teardown_method(self) -> None:
        self.tempdir.cleanup()

    def test_defaults_match_paper_pass1_eval_settings(self) -> None:
        args = self._args()

        # SWE-bench Verified is the curated benchmark split used for pass@1
        # software-engineering evals: each task asks the agent to produce one
        # repository patch, then the SWE-bench harness grades that patch.
        assert args.dataset == "princeton-nlp/SWE-bench_Verified"
        assert args.split == "test"
        # CodeActAgent is the OpenHands agent that turns model tool calls into
        # concrete shell/editor actions. Keeping this explicit avoids silently
        # comparing a different agent policy against the SWE-HERO baseline.
        assert args.agent == "CodeActAgent"
        assert args.max_iterations == 100

        # The paper-aligned setup lets OpenHands send prompts up to 128k tokens.
        # Qwen2.5-Coder-7B natively supports 32k positions, so vLLM needs the
        # matching 128k model length plus YaRN/RoPE scaling metadata to interpret
        # token positions beyond the native window.
        assert args.context_mode == "paper-yarn-128k"
        assert args.max_input_tokens == 131_072
        assert args.vllm_max_model_len == 131_072
        assert json.loads(args.vllm_rope_scaling) == {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32_768,
        }

        # These sampling settings control how deterministic patch generation is.
        # Temperature adds randomness, top-p keeps the smallest probability mass
        # above 0.8, and top-k restricts sampling to the 20 most likely tokens.
        assert args.temperature == 0.7
        assert args.top_p == 0.8
        assert args.top_k == 20
        # The output budget caps how much patch/action text one model response
        # may emit. It must be recorded because changing it changes the agent's
        # chance to produce large diffs or long tool arguments.
        assert args.max_output_tokens == 4096
        assert args.eval_note == "swehero-qwen25-coder7b-pass1"
        assert args.eval_limit is None

        # Native tool calling means the model must emit structured function-call
        # records that OpenHands can execute. Requiring a tool choice avoids
        # free-form assistant prose where the agent expected a concrete action.
        assert args.native_tool_calling
        assert args.tool_choice == "required"
        assert args.tool_call_preflight

        # The Docker checks are infrastructure guards for SWE-bench grading, not
        # ML behavior. Grading each patch requires Docker containers that run the
        # repository tests for the benchmark instance.
        assert args.docker_smoke_image == "hello-world:latest"
        assert not args.skip_docker_run_check
        assert not args.skip_docker_buildx_check

        # The default pod topology runs eight one-GPU vLLM replicas. 192 workers
        # equals 24 concurrent OpenHands tasks per replica, while tensor/pipeline
        # parallel sizes of 1 mean each GPU hosts an independent full model copy.
        assert args.num_workers == 192
        assert args.vllm_tensor_parallel_size == 1
        assert args.vllm_pipeline_parallel_size == 1
        assert args.vllm_server_count == 8
        assert args.vllm_agent_tasks_per_server == 24
        assert args.vllm_router_port == 8090
        assert args.vllm_gpu_memory_utilization == 0.90
        assert args.vllm_dtype == "bfloat16"
        assert args.vllm_distributed_executor_backend == "mp"
        # Eager execution trades some vLLM graph optimization for more predictable
        # compatibility during prototype evals, which is useful when comparing
        # model/checkpoint behavior rather than serving compiler behavior.
        assert args.vllm_enforce_eager

    def test_base_url_is_required_for_non_dry_run(self) -> None:
        with pytest.raises(ValueError, match="--base-url is required"):
            eval_script.parse_args(
                [
                    "--output-dir",
                    str(Path(self.tempdir.name) / "run"),
                    "--openhands-dir",
                    str(Path(self.tempdir.name) / "OpenHands"),
                ]
            )

    def test_expected_output_path_matches_openhands_layout(self) -> None:
        args = self._args("--eval-note", "paper-pass1", "--eval-limit", "1")
        output = eval_script.expected_output_jsonl(args)

        assert str(output).startswith(str(args.output_dir))
        assert "princeton-nlp__SWE-bench_Verified-test" in output.parts
        assert "CodeActAgent" in output.parts
        assert output.name == "output.jsonl"
        assert "Qwen2.5-Coder-7B-Instruct_maxiter_100_N_paper-pass1" in output.parts

    def test_generated_config_disables_non_paper_tools_and_sets_sampling(self) -> None:
        args = self._args()
        config = eval_script.build_openhands_config(args)

        # OpenHands uses LiteLLM-style model names for OpenAI-compatible servers.
        # The generated config must point at the served Qwen checkpoint and carry
        # the same sampling/context/tool-call contract that the launcher records.
        assert "[llm.swehero_qwen25_coder7b]" in config
        assert 'model = "openai/Qwen/Qwen2.5-Coder-7B-Instruct"' in config
        assert "temperature = 0.7" in config
        assert "top_p = 0.8" in config
        assert "top_k = 20" in config
        assert "max_input_tokens = 131072" in config
        assert "max_output_tokens = 4096" in config
        assert "native_tool_calling = true" in config
        assert 'completion_kwargs = { tool_choice = "required" }' in config

        # Jupyter, browsing, and the LLM editor are powerful OpenHands tools, but
        # they are not part of this paper-aligned SWE-HERO reproduction. Leaving
        # them enabled would change the action space the model can use.
        assert "[agent.swehero_openhands]" in config
        assert "enable_jupyter = false" in config
        assert "enable_browsing = false" in config
        assert "enable_llm_editor = false" in config
        assert "custom_llm_provider" not in config

    def test_max_output_tokens_can_be_omitted_for_ablation(self) -> None:
        args = self._args("--max-output-tokens", "none")
        config = eval_script.build_openhands_config(args)

        # Some ablations intentionally remove the wrapper-level output cap so the
        # underlying OpenHands/vLLM defaults decide response length. The absence
        # must be explicit so reviewers can tell it is an experiment choice.
        assert args.max_output_tokens is None
        assert "max_output_tokens" not in config

    def test_api_key_is_llm_api_key_env_only(self) -> None:
        with mock.patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-ignored"}, clear=True
        ):
            args = self._args()

        assert args.api_key == "local-llm"
        assert args.api_key_source == "default"

    def test_write_scaffold_records_commands_without_leaking_real_api_key(self) -> None:
        with mock.patch.dict("os.environ", {"LLM_API_KEY": "sk-real-secret"}):
            args = self._args()
            paths, commands = eval_script.write_scaffold(args)

        # The generated OpenHands config needs the real key because the runtime
        # client sends it to the OpenAI-compatible vLLM endpoint. Metadata keeps a
        # redacted command record so run artifacts remain shareable.
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

        # Context metadata is part of reproducibility. A pass@1 score without the
        # prompt budget and long-context scaling is not comparable to another run.
        assert metadata["context"]["mode"] == "paper-yarn-128k"
        assert metadata["context"]["max_input_tokens"] == 131_072
        assert metadata["context"]["vllm_max_model_len"] == 131_072
        assert metadata["context"]["vllm_rope_scaling"]["rope_type"] == "yarn"

    def test_vllm_command_enables_qwen_native_tool_calling(self) -> None:
        args = self._args()
        _paths, commands = eval_script.write_scaffold(args)

        # vLLM must accept the same long prompt length that OpenHands is allowed
        # to send. YaRN/RoPE scaling is the model-position math that makes 128k
        # serving intentional for a Qwen2.5 checkpoint whose native window is 32k.
        assert "--max-model-len" in commands.serve_vllm
        assert "131072" in commands.serve_vllm
        assert "--rope-scaling" in commands.serve_vllm
        assert (
            '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
            in commands.serve_vllm
        )

        # Qwen emits OpenAI-style tool-call records through vLLM's parser. Hermes
        # is the parser mode vLLM uses for this family of tool-call templates.
        assert "--enable-auto-tool-choice" in commands.serve_vllm
        assert "--tool-call-parser" in commands.serve_vllm
        assert "hermes" in commands.serve_vllm

        # Tensor and pipeline parallelism split one model across multiple GPUs.
        # The SWE-HERO 7B eval instead runs independent one-GPU replicas, so both
        # sizes stay at 1 and concurrency is handled by the router/worker count.
        assert "--tensor-parallel-size" in commands.serve_vllm
        assert "1" in commands.serve_vllm
        assert "--pipeline-parallel-size" in commands.serve_vllm
        assert "1" in commands.serve_vllm

        # bfloat16 lowers memory pressure while preserving a broad numeric range;
        # the utilization cap leaves memory for KV cache and runtime allocations.
        assert "--gpu-memory-utilization" in commands.serve_vllm
        assert "0.9" in commands.serve_vllm
        assert "--dtype" in commands.serve_vllm
        assert "bfloat16" in commands.serve_vllm
        assert "--distributed-executor-backend" in commands.serve_vllm
        assert "mp" in commands.serve_vllm
        assert "--enforce-eager" in commands.serve_vllm

    def test_base_native_32k_context_mode_uses_native_context_without_yarn(
        self,
    ) -> None:
        args = self._args("--context-mode", "base-native-32k")
        paths, commands = eval_script.write_scaffold(args)

        # This is the clean base-model control: keep both OpenHands and vLLM
        # inside Qwen2.5-Coder's original 32k context and do not apply YaRN.
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

    def test_base_paper_yarn_128k_context_mode_uses_context_matched_yarn(self) -> None:
        args = self._args("--context-mode", "base-paper-yarn-128k")
        paths, commands = eval_script.write_scaffold(args)

        # This control keeps the released base checkpoint but gives it the same
        # 128k YaRN serving context as the trained SWE-HERO-style checkpoint. That
        # separates "better model weights" from "larger context window".
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

    def test_eval_presets_swap_context_contracts(self) -> None:
        # Presets are the public experiment contract. Each one must select a
        # named context mode so readers can tell whether the run is paper-aligned,
        # native-context, context-matched base, or SWE-Lego/Qwen3.
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
                # SWE-Lego reserves a larger vLLM model context than the
                # OpenHands input budget because its serving contract uses a
                # 160k-class Qwen3 context while OpenHands caps prompt tokens at
                # 147456.
                assert args.vllm_max_model_len == 163_840
            else:
                assert args.vllm_max_model_len == max_tokens
            assert args.eval_note == eval_note

    def test_swe_lego_qwen3_preset_matches_upstream_serving_and_llm_contract(
        self,
    ) -> None:
        preset = (
            REPO_ROOT
            / "configs"
            / "eval"
            / "openhands-swebench-verified-swe-lego-qwen3-8b.args"
        )
        args = self._args(f"@{preset}")
        paths, commands = eval_script.write_scaffold(args)

        # SWE-Lego is not a minor preset tweak: it uses a different Qwen3
        # checkpoint, vendored OpenHands/SWE-bench versions, and its own serving
        # contract. These assertions prevent SWE-HERO/Qwen2.5 assumptions from
        # leaking into that stack.
        assert args.eval_stack == "swe-lego"
        assert args.model_id == "SWE-Lego/SWE-Lego-Qwen3-8B"
        assert args.served_model_name == "Qwen/Qwen3-8B"
        # Temperature 0 makes decoding deterministic for this stack. Omitting
        # top-p/top-k leaves no extra sampling filters beyond greedy selection.
        assert args.temperature == 0.0
        assert args.top_p is None
        assert args.top_k is None
        # Qwen3's long-context settings are shipped in the checkpoint config, so
        # the wrapper must not pass a separate vLLM RoPE override.
        assert args.max_input_tokens == 147_456
        assert args.max_output_tokens == 16_384
        assert args.vllm_max_model_len == 163_840
        assert args.vllm_rope_scaling is None
        # One 8-GPU tensor-parallel server shards the model across all GPUs. That
        # is the opposite of the Qwen2.5 recipe, which runs eight independent
        # one-GPU replicas behind a router.
        assert args.vllm_tensor_parallel_size == 8
        assert args.vllm_server_count == 1
        assert not args.vllm_use_router
        assert args.vllm_max_num_seqs == 24
        # SWE-Lego's OpenHands/Qwen3 contract does not use this wrapper's Qwen2.5
        # native-tool-calling config or preflight probe.
        assert not (
            args.native_tool_calling and not args.omit_native_tool_calling_config
        )
        assert not args.tool_call_preflight
        assert args.num_workers == 24
        assert args.swebench_cache_level == "instance"
        assert args.swebench_timeout == 500
        assert args.swebench_max_workers == 10

        config = paths.config_path.read_text()
        # The config must expose the model under the served Qwen3 name and omit
        # sampling/tool-call options that are intentionally disabled above.
        assert "[llm.eval_qwen3_8b]" in config
        assert 'model = "openai/Qwen/Qwen3-8B"' in config
        assert "temperature = 0.0" in config
        assert "max_input_tokens = 147456" in config
        assert "max_output_tokens = 16384" in config
        assert "top_p" not in config
        assert "top_k" not in config
        assert "native_tool_calling" not in config
        assert "tool_choice" not in config

        # The vLLM and grader commands should mirror SWE-Lego's upstream serving
        # and evaluation path, including conversion from OpenHands output JSONL
        # into the standalone SWE-bench harness input format.
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

    def test_base_native_32k_rejects_forced_128k_context(self) -> None:
        # A native-context control stops being native if a caller forces a 128k
        # prompt budget. The parser rejects that instead of silently producing an
        # invalid baseline.
        with pytest.raises(ValueError, match="base-native-32k requires"):
            self._args(
                "--context-mode",
                "base-native-32k",
                "--max-input-tokens",
                "131072",
            )

    def test_eval_ids_are_passed_to_openhands_and_conflict_with_eval_limit(
        self,
    ) -> None:
        args = self._args("--eval-ids", "django__django-13670,sympy__sympy-15599")
        _paths, commands = eval_script.write_scaffold(args)

        # Explicit instance IDs support targeted reruns of specific benchmark
        # tasks. They must remain separate from eval_limit because "first N tasks"
        # and "these named tasks" are different sampling contracts.
        assert "--eval-ids" in commands.run_infer
        assert "django__django-13670,sympy__sympy-15599" in commands.run_infer

        with pytest.raises(ValueError, match="mutually exclusive"):
            self._args(
                "--eval-limit",
                "1",
                "--eval-ids",
                "django__django-13670",
            )

    def test_selected_id_filter_config_is_restored_after_use(self) -> None:
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

    def test_summarize_report_computes_pass_at_1_from_resolved_ids(self) -> None:
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

        # pass@1 means "one generated patch attempt per task." The numerator is
        # resolved task IDs and the denominator is every submitted task outcome.
        assert summary["resolved"] == 2
        assert summary["total"] == 3
        assert round(abs(summary["pass_at_1"] - 2 / 3), 7) == 0

    def test_summarize_report_uses_submitted_instances_for_limited_runs(self) -> None:
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

        # SWE-bench reports may describe the full benchmark size even for a smoke
        # run. For eval_limit runs, the denominator must be submitted instances so
        # one-task dry runs do not look like 0/500 full evaluations.
        assert summary["resolved"] == 0
        assert summary["total"] == 1
        assert summary["pass_at_1"] == 0

    def test_tool_call_preflight_payload_targets_served_model(self) -> None:
        args = self._args("--served-model-name", "qwen-7b")

        payload = eval_script.tool_call_preflight_payload(args)

        # The preflight asks the exact served model name for a structured tool
        # call before OpenHands starts expensive SWE-bench tasks. This catches
        # vLLM parser/template mismatches early.
        assert payload["model"] == "qwen-7b"
        assert payload["tool_choice"] == "required"
        assert (
            payload["tools"][0]["function"]["name"]
            == eval_script.TOOL_CALL_PREFLIGHT_NAME
        )

    def test_tool_choice_override_updates_config_and_preflight(self) -> None:
        args = self._args("--tool-choice", "auto")

        config = eval_script.build_openhands_config(args)
        payload = eval_script.tool_call_preflight_payload(args)

        # Tool-choice policy must stay aligned between OpenHands config and the
        # preflight request. Otherwise the smoke check could pass a different
        # model behavior than the real eval will use.
        assert 'completion_kwargs = { tool_choice = "auto" }' in config
        assert payload["tool_choice"] == "auto"

    def test_tool_call_preflight_accepts_structured_tool_calls(self) -> None:
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

        # A real tool-call response contains machine-readable tool_calls entries,
        # not just natural language saying it intends to use a tool.
        assert check.ok

    def test_tool_call_preflight_rejects_plain_message_text(self) -> None:
        check = eval_script.validate_tool_call_response(
            {"choices": [{"message": {"content": "I will call the tool now."}}]}
        )

        assert not check.ok
        assert "message.tool_calls missing" in check.detail

    def test_summarize_agent_tool_use_counts_real_tools_and_loops(self) -> None:
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

        # OpenHands histories mix bookkeeping actions ("system"), assistant text,
        # and real repository actions ("run", "think", editor tools, etc.). The
        # summary distinguishes whether the agent actually used tools and whether
        # OpenHands hit a loop failure during a task.
        assert summary["used_real_tools"]
        assert summary["agent_tool_actions"] == 2
        assert summary["agent_message_actions"] == 2
        assert summary["tool_action_counts"] == {"think": 1, "run": 1}
        assert summary["loop_errors"] == 1

    def test_docker_preflight_runs_container_and_checks_buildx(self) -> None:
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

    def test_docker_preflight_reports_unprivileged_run_failure(self) -> None:
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
