from pathlib import Path

from scripts.openhands_eval_launcher_defaults import select_openhands_llm_api_key
from scripts.openhands_eval_worker_selection import select_openhands_eval_num_workers

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_openhands_swebench_eval_pod.py"
COMMON = REPO_ROOT / "scripts" / "pod_startup_common.py"


class TestOpenHandsEvalPodLauncher:
    def test_launcher_is_pod_only_and_uses_gpu_pod_endpoint(self):
        script = SCRIPT.read_text()
        common = COMMON.read_text()

        assert "pod_startup_common.require_pod_runtime" in script
        assert "this launcher is pod-only" in common
        assert 'Path("/workspace").is_dir()' in common
        assert "require_pod_runtime(" in script
        assert "self.workspace_root" in script
        assert '"nvidia-smi"' in script
        assert '"docker"' in script
        assert '"curl"' in script
        assert '"git"' in script
        assert 'pod_ip = os.environ.get("POD_IP", self.pod_ip())' in script
        assert 'llm_base_url = f"http://{pod_ip}:{self.vllm_router_port}/v1"' in script
        assert '"--base-url",' in script
        assert "127.0.0.1" not in script
        assert "openhands" + "-eval-driver" not in script

    def test_launcher_enforces_pushed_clean_pod_git_checkout(self):
        script = SCRIPT.read_text()
        common = COMMON.read_text()

        assert "pod_startup_common.prepare_pod_checkout" in script
        assert "pod_git_guard.require_pod_git_checkout" in common
        assert "SWEHERO_POD_GIT_BRANCH" in script
        assert "OpenHands eval pod execution directory" in script
        assert "supervised_env_args" in script
        assert "VLLM_FORCE_RESTART={self.vllm_force_restart}" in script
        assert "VLLM_NCCL_CUMEM_ENABLE={self.vllm_nccl_cumem_enable}" in script
        assert "{shell_join(self.supervised_env_args())}" in script

    def test_launcher_verifies_docker_and_buildx(self):
        script = SCRIPT.read_text()

        assert '"docker", "run", "--rm", self.docker_smoke_image' in script
        assert '"docker", "buildx", "version"' in script
        assert "dockerd --host=unix:///var/run/docker.sock" in script

    def test_launcher_bootstraps_and_repairs_pinned_python_envs(self):
        script = SCRIPT.read_text()

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

    def test_launcher_starts_vllm_and_scaffold_in_same_pod_flow(self):
        script = SCRIPT.read_text()

        assert 'str(self.vllm_venv / "bin" / "vllm")' in script
        assert '"serve",' in script
        assert "CUDA_VISIBLE_DEVICES={gpu}" in script
        assert "CUDA_VISIBLE_DEVICES={self.vllm_visible_devices}" in script
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

    def test_launcher_supports_swe_lego_single_multi_gpu_vllm_without_router(self):
        script = SCRIPT.read_text()

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

    def test_worker_selection_preserves_swe_lego_num_workers_contract(self):
        assert self._select_workers("swe-lego", "", "a,b,c", 24, 24) == 24

    def test_worker_selection_still_clamps_current_eval_id_smokes(self):
        assert self._select_workers("openhands", "", "a,b,c", 192, 192) == 3

    def test_swe_lego_uses_vendored_dummy_api_key_by_default(self):
        assert (
            self._select_llm_api_key(
                "swe-lego", explicit=False, current_key="local-llm"
            )
            == "dummy-key"
        )

    def test_explicit_llm_api_key_overrides_swe_lego_default(self):
        assert (
            self._select_llm_api_key(
                "swe-lego", explicit=True, current_key="custom-key"
            )
            == "custom-key"
        )

    def test_current_eval_keeps_local_llm_default(self):
        assert (
            self._select_llm_api_key(
                "openhands", explicit=False, current_key="local-llm"
            )
            == "local-llm"
        )

    def test_vllm_requirement_is_pinned(self):
        requirements = REPO_ROOT / "requirements" / "openhands-vllm.txt"

        text = requirements.read_text()
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
        return select_openhands_llm_api_key(eval_stack, explicit, current_key)
