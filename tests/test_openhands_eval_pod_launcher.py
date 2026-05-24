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
        self.assertIn('--base-url "http://${POD_IP}:${VLLM_PORT}/v1"', script)
        self.assertNotIn("127.0.0.1", script)
        self.assertNotIn("openhands" + "-eval-driver", script)

    def test_launcher_verifies_docker_and_buildx(self):
        script = SCRIPT.read_text()

        self.assertIn('docker run --rm "$DOCKER_SMOKE_IMAGE"', script)
        self.assertIn("docker buildx version", script)
        self.assertIn("dockerd --host=unix:///var/run/docker.sock", script)

    def test_launcher_starts_vllm_and_scaffold_in_same_pod_flow(self):
        script = SCRIPT.read_text()

        self.assertIn('"$VLLM_VENV/bin/vllm") serve', script)
        self.assertIn("--enable-auto-tool-choice", script)
        self.assertIn("--tool-call-parser hermes", script)
        self.assertIn('"$EVAL_VENV/bin/python" scripts/openhands_swebench_eval.py', script)
        self.assertIn("--preflight-only", script)


if __name__ == "__main__":
    unittest.main()
