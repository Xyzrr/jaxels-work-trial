"""Tests for the pod-side OpenHands SWE-bench eval launcher.

This launcher coordinates several ML-adjacent systems inside the GPU pod:
vLLM serves the coding model, OpenHands drives the agent loop, Docker/SWE-bench
grades the generated patch, and the pod git guard pins the code revision. These
tests are intentionally white-box because the most expensive failures here are
miswired launch contracts that only appear after GPUs, model weights, and Docker
workers have already started.
"""

from __future__ import annotations

from pathlib import Path

from scripts.openhands_eval_launcher_defaults import select_openhands_llm_api_key
from scripts.openhands_eval_worker_selection import select_openhands_eval_num_workers

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_openhands_swebench_eval_pod.py"
COMMON = REPO_ROOT / "scripts" / "pod_startup_common.py"


class TestOpenHandsEvalPodLauncher:
    """Document the launch contracts that keep eval runs reproducible."""

    def test_launcher_is_pod_only_and_uses_gpu_pod_endpoint(self) -> None:
        script = SCRIPT.read_text()
        common = COMMON.read_text()

        # Eval must run inside the GPU pod because the local workstation does not
        # have the model weights, GPUs, Docker daemon, or pod networking assumed
        # by OpenHands and vLLM.
        assert "pod_startup_common.require_pod_runtime" in script
        assert "this launcher is pod-only" in common
        assert 'Path("/workspace").is_dir()' in common
        assert "require_pod_runtime(" in script
        assert "self.workspace_root" in script
        # These tools are runtime prerequisites, not optional conveniences:
        # nvidia-smi proves GPUs are visible, Docker is needed by SWE-bench
        # grading, curl probes local APIs, and git verifies the pod checkout.
        assert '"nvidia-smi"' in script
        assert '"docker"' in script
        assert '"curl"' in script
        assert '"git"' in script
        # OpenHands talks to vLLM through the pod IP so multi-process Docker
        # workers can reach the same model server. A localhost base URL would
        # only work from the launcher process and would break grading workers.
        assert 'pod_ip = os.environ.get("POD_IP", self.pod_ip())' in script
        assert 'llm_base_url = f"http://{pod_ip}:{self.vllm_router_port}/v1"' in script
        assert '"--base-url",' in script
        assert "127.0.0.1" not in script
        assert "openhands" + "-eval-driver" not in script

    def test_launcher_enforces_pushed_clean_pod_git_checkout(self) -> None:
        script = SCRIPT.read_text()
        common = COMMON.read_text()

        # Eval scores are only meaningful if reviewers can recover exactly which
        # code launched the model server, OpenHands wrapper, and grader. The pod
        # checkout guard fast-forwards the named branch from origin before work
        # starts, avoiding hidden workstation-only changes.
        assert "pod_startup_common.prepare_pod_checkout" in script
        assert "pod_git_guard.require_pod_git_checkout" in common
        assert "SWEHERO_POD_GIT_BRANCH" in script
        assert "OpenHands eval pod execution directory" in script
        assert "supervised_env_args" in script
        # vLLM restart and NCCL allocator settings must survive tmux supervision.
        # Otherwise a resumed eval could silently reuse an incompatible model
        # server or lose the multi-GPU shared-memory fix.
        assert "VLLM_FORCE_RESTART={self.vllm_force_restart}" in script
        assert "VLLM_NCCL_CUMEM_ENABLE={self.vllm_nccl_cumem_enable}" in script
        assert "{shell_join(self.supervised_env_args())}" in script

    def test_launcher_verifies_docker_and_buildx(self) -> None:
        script = SCRIPT.read_text()

        # SWE-bench validates patches by building/running task repositories in
        # Docker. Checking Docker and buildx up front fails fast before vLLM has
        # spent minutes loading a large model into GPU memory.
        assert '"docker", "run", "--rm", self.docker_smoke_image' in script
        assert '"docker", "buildx", "version"' in script
        assert "dockerd --host=unix:///var/run/docker.sock" in script

    def test_launcher_bootstraps_and_repairs_pinned_python_envs(self) -> None:
        script = SCRIPT.read_text()

        # vLLM, OpenHands, SWE-bench, PyTorch, and tokenizer packages are version
        # sensitive. The launcher repairs pinned virtualenvs instead of relying
        # on whatever happens to be installed in the pod image.
        assert 'OPENHANDS_EVAL_UV_VERSION = "0.11.16"' in script
        assert "UV_X86_64_UNKNOWN_LINUX_GNU_SHA256" in script
        assert "uv_bin = launcher.ensure_uv()" in script
        assert "self.openhands_eval_poetry_version = os.environ.get(" in script
        assert "OPENHANDS_EVAL_POETRY_VERSION" in script
        assert '"2.1.3"' in script
        assert "poetry=={self.openhands_eval_poetry_version}" in script
        assert "VLLM_REQUIREMENTS_PATH" in script
        assert "requirements" in script
        assert "openhands-vllm.txt" in script
        assert '"pip",\n                "compile",' in script
        assert '"--output-file",' in script
        assert '"sync"' in script
        assert "resolved_requirements" in script
        assert 'self.vllm_venv / "bin" / "python"' in script
        assert "ensure_vllm_python" in script

    def test_launcher_starts_vllm_and_scaffold_in_same_pod_flow(self) -> None:
        script = SCRIPT.read_text()

        # The launcher starts vLLM before OpenHands so the agent loop sees an
        # OpenAI-compatible model endpoint. Tool-call parsing is required because
        # OpenHands represents repository edits as structured tool calls, not as
        # plain chat text.
        assert 'str(self.vllm_venv / "bin" / "vllm")' in script
        assert '"serve",' in script
        assert "CUDA_VISIBLE_DEVICES={gpu}" in script
        assert "CUDA_VISIBLE_DEVICES={self.vllm_visible_devices}" in script
        # Context length, RoPE/YaRN scaling, and model identity come from the
        # argparse preset. They are ML behavior, so this pod launcher should
        # resolve them from the preset rather than inventing shell defaults.
        assert "CONFIG_PRESET" in script
        assert "resolve_eval_config" in script
        assert 'f"@{self.config_preset_path}"' in script
        assert '--context-mode "$CONTEXT_MODE"' not in script
        assert 'CONTEXT_MODE="${CONTEXT_MODE:-' not in script
        assert "--rope-scaling" in script
        assert "--max-model-len" in script
        assert "vllm_context_signature" in script
        assert "scripts/openai_vllm_router.py" in script
        assert "--enable-auto-tool-choice" in script
        assert "--tool-call-parser" in script
        assert 'str(self.eval_venv / "bin" / "python")' in script
        assert '"scripts/openhands_swebench_eval.py"' in script
        assert "--preflight-only" in script

    def test_launcher_supports_swe_lego_single_multi_gpu_vllm_without_router(
        self,
    ) -> None:
        script = SCRIPT.read_text()

        # SWE-Lego's vendored eval stack has a different serving contract from
        # the current OpenHands path: one multi-GPU vLLM server can be addressed
        # directly, while multi-server OpenHands evals need the router.
        assert "self.vllm_parallel_gpu_count = int(" in script
        assert "self.vllm_tensor_parallel_size" in script
        assert "self.vllm_pipeline_parallel_size" in script
        assert (
            "direct vLLM base URL without router requires VLLM_SERVER_COUNT=1" in script
        )
        assert '"--max-num-seqs", str(self.vllm_max_num_seqs)' in script
        assert "if env_bool(str(self.vllm_use_router)):" in script
        assert 'llm_base_url = f"http://{pod_ip}:{self.vllm_port}/v1"' in script
        assert "--eval-ids" in script
        assert "SWE_LEGO_SWEBENCH_DIR" in script
        assert "Path(self.swe_lego_swebench_dir).is_dir()" in script
        # GPU servers and multiprocessing workers can leave stale processes and
        # shared-memory files after interruption. Cleaning those up prevents the
        # next eval from inheriting old model state or NCCL rendezvous files.
        assert "for stale_gpu in range(self.visible_gpu_count):" in script
        assert "self.vllm_session_name(stale_gpu)" in script
        assert "cleanup_vllm_runtime" in script
        assert (
            'self.kill_process_pattern(str(self.vllm_venv / "bin" / "vllm"))' in script
        )
        assert "from multiprocessing" in script
        assert "bin" in script
        assert "python" in script
        assert "effective_vllm_nccl_cumem_enable" in script
        assert 'env_parts.append(f"NCCL_CUMEM_ENABLE={nccl_cumem_enable}")' in script
        assert "/dev/shm/psm_*" in script
        assert "/dev/shm/sem.mp-*" in script
        assert "/dev/shm/nccl-*" in script

    def test_worker_selection_preserves_swe_lego_num_workers_contract(self) -> None:
        # SWE-Lego expects the preset's worker count to be honored even for an
        # eval-id subset; its vendored runner controls task scheduling itself.
        assert self._select_workers("swe-lego", "", "a,b,c", 24, 24) == 24

    def test_worker_selection_still_clamps_current_eval_id_smokes(self) -> None:
        # The current OpenHands eval path clamps small eval-id smoke runs so a
        # three-task sanity check does not start 192 parallel agent workers.
        assert self._select_workers("openhands", "", "a,b,c", 192, 192) == 3

    def test_swe_lego_uses_vendored_dummy_api_key_by_default(self) -> None:
        # SWE-Lego's vendored OpenHands/vLLM scripts use a literal dummy key for
        # local model serving, so the default preserves that stack's contract.
        assert (
            self._select_llm_api_key(
                "swe-lego", explicit=False, current_key="local-llm"
            )
            == "dummy-key"
        )

    def test_explicit_llm_api_key_overrides_swe_lego_default(self) -> None:
        # Explicit user configuration still wins for custom deployments where
        # the local OpenAI-compatible endpoint expects a non-default bearer token.
        assert (
            self._select_llm_api_key(
                "swe-lego", explicit=True, current_key="custom-key"
            )
            == "custom-key"
        )

    def test_current_eval_keeps_local_llm_default(self) -> None:
        # The non-vendored OpenHands path already standardizes on local-llm for
        # pod-local vLLM, so it should not inherit SWE-Lego's dummy-key default.
        assert (
            self._select_llm_api_key(
                "openhands", explicit=False, current_key="local-llm"
            )
            == "local-llm"
        )

    def test_vllm_requirement_is_pinned(self) -> None:
        requirements = REPO_ROOT / "requirements" / "openhands-vllm.txt"

        text = requirements.read_text()
        # These pins protect model-serving behavior. vLLM and transformers
        # jointly determine tokenizer handling, long-context serving, tool-call
        # support, and CUDA/PyTorch compatibility for the eval model server.
        assert "vllm==0.9.2" in text
        assert "transformers==4.53.3" in text

    def _select_workers(
        self,
        eval_stack: str,
        eval_limit: str,
        eval_ids: str,
        config_num_workers: int,
        total_agent_workers: int,
    ) -> int:
        """Route worker-selection examples through the shared helper."""

        return select_openhands_eval_num_workers(
            eval_stack,
            eval_limit,
            eval_ids,
            config_num_workers,
            total_agent_workers,
        )

    def _select_llm_api_key(
        self,
        eval_stack: str,
        explicit: bool,
        current_key: str,
    ) -> str:
        """Route API-key default examples through the shared helper."""

        return select_openhands_llm_api_key(eval_stack, explicit, current_key)
