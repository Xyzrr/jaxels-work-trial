import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_openhands_swebench_eval_pod.sh"


class OpenHandsEvalPodLauncherTests(unittest.TestCase):
    def test_launcher_is_pod_only_and_uses_gpu_pod_endpoint(self):
        script = SCRIPT.read_text()

        self.assertIn('die "this launcher is pod-only', script)
        self.assertIn('[[ -d /workspace ]]', script)
        self.assertIn("command -v nvidia-smi", script)
        self.assertIn('POD_IP="${POD_IP:-$(pod_ip)}"', script)
        self.assertIn('--base-url "http://${POD_IP}:${VLLM_ROUTER_PORT}/v1"', script)
        self.assertNotIn("127.0.0.1", script)
        self.assertNotIn("openhands" + "-eval-driver", script)

    def test_launcher_verifies_docker_and_buildx(self):
        script = SCRIPT.read_text()

        self.assertIn('docker run --rm "$DOCKER_SMOKE_IMAGE"', script)
        self.assertIn("docker buildx version", script)
        self.assertIn("dockerd --host=unix:///var/run/docker.sock", script)

    def test_launcher_bootstraps_and_repairs_pinned_python_envs(self):
        script = SCRIPT.read_text()

        self.assertIn('readonly OPENHANDS_EVAL_UV_VERSION="0.11.16"', script)
        self.assertIn("UV_X86_64_UNKNOWN_LINUX_GNU_SHA256", script)
        self.assertIn('PINNED_UV_BIN="$(ensure_uv)"', script)
        self.assertIn('OPENHANDS_EVAL_POETRY_VERSION="${OPENHANDS_EVAL_POETRY_VERSION:-2.1.3}"', script)
        self.assertIn('"poetry==${OPENHANDS_EVAL_POETRY_VERSION}"', script)
        self.assertIn('VLLM_REQUIREMENTS_PATH="${VLLM_REQUIREMENTS_PATH:-$ROOT_DIR/requirements/openhands-vllm.txt}"', script)
        self.assertIn('"$PINNED_UV_BIN" pip compile', script)
        self.assertIn('--output-file "$resolved_requirements"', script)
        self.assertIn('"$PINNED_UV_BIN" pip sync --python "$VLLM_VENV/bin/python" "$resolved_requirements"', script)
        self.assertIn("ensure_vllm_python", script)

    def test_launcher_starts_vllm_and_scaffold_in_same_pod_flow(self):
        script = SCRIPT.read_text()

        self.assertIn('"$VLLM_VENV/bin/vllm") serve', script)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"$gpu\"", script)
        self.assertIn('CONTEXT_MODE="${CONTEXT_MODE:-paper-yarn-128k}"', script)
        self.assertIn('PAPER_YARN_ROPE_SCALING=', script)
        self.assertIn("--context-mode \"$CONTEXT_MODE\"", script)
        self.assertIn("--max-input-tokens \"$MAX_INPUT_TOKENS\"", script)
        self.assertIn("--vllm-max-model-len \"$VLLM_MAX_MODEL_LEN\"", script)
        self.assertIn("--rope-scaling", script)
        self.assertIn("vllm_context_signature", script)
        self.assertIn("scripts/openai_vllm_router.py", script)
        self.assertIn('VLLM_AGENT_TASKS_PER_SERVER="${VLLM_AGENT_TASKS_PER_SERVER:-24}"', script)
        self.assertIn("--enable-auto-tool-choice", script)
        self.assertIn("--tool-call-parser hermes", script)
        self.assertIn('"$EVAL_VENV/bin/python" scripts/openhands_swebench_eval.py', script)
        self.assertIn("--preflight-only", script)

    def test_vllm_requirement_is_pinned(self):
        requirements = REPO_ROOT / "requirements" / "openhands-vllm.txt"

        text = requirements.read_text()
        self.assertIn("vllm==0.9.2", text)
        self.assertIn("transformers==4.53.3", text)


if __name__ == "__main__":
    unittest.main()
