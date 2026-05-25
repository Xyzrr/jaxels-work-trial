import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prebuild_openhands_swebench_images_pod.sh"
DOC = REPO_ROOT / "docs" / "openhands_swebench_gpu_pod_eval.md"


class OpenHandsImagePrebuildPodLauncherTests(unittest.TestCase):
    def test_launcher_shell_syntax_is_valid(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_launcher_rejects_nonpositive_parallel_builds_before_pod_checks(self):
        result = subprocess.run(
            [str(SCRIPT), "--foreground", "--parallel-builds", "0"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--parallel-builds must be positive", result.stderr)

    def test_launcher_rejects_replace_session_in_foreground_mode(self):
        result = subprocess.run(
            [str(SCRIPT), "--foreground", "--replace-session"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--replace-session only applies", result.stderr)

    def test_launcher_uses_tmux_and_attaches_to_existing_session_by_default(self):
        script = SCRIPT.read_text()

        self.assertIn('TMUX_SESSION="openhands-swebench-image-prebuild"', script)
        self.assertIn('tmux has-session -t "$TMUX_SESSION"', script)
        self.assertIn('tmux session already exists: $TMUX_SESSION', script)
        self.assertIn('exec tmux attach-session -t "$TMUX_SESSION"', script)
        self.assertIn('tmux new-session -d -s "$TMUX_SESSION"', script)

    def test_launcher_can_replace_existing_tmux_session(self):
        script = SCRIPT.read_text()

        self.assertIn("--replace-session)", script)
        self.assertIn('REPLACE_SESSION=0', script)
        self.assertIn('replacing tmux session: $TMUX_SESSION', script)
        self.assertIn('tmux kill-session -t "$TMUX_SESSION"', script)
        self.assertIn("--replace-session only applies to tmux-supervised launches", script)

    def test_launcher_is_pod_only_and_enforces_pod_git_checkout(self):
        script = SCRIPT.read_text()

        self.assertIn('die "this launcher is pod-only', script)
        self.assertIn('[[ -d /workspace ]]', script)
        self.assertIn('source "$ROOT_DIR/scripts/pod_git_guard.sh"', script)
        self.assertIn("swehero_require_pod_git_checkout", script)
        self.assertIn("SWEHERO_POD_GIT_BRANCH", script)

    def test_launcher_derives_images_from_eval_preset_and_openhands_code(self):
        script = SCRIPT.read_text()

        self.assertIn("from scripts import openhands_swebench_eval as eval_script", script)
        self.assertIn('"DATASET": args.dataset', script)
        self.assertIn('"SPLIT": args.split', script)
        self.assertIn('"OPENHANDS_REF": args.openhands_ref', script)
        self.assertIn("set_dataset_type(args.dataset)", script)
        self.assertIn("get_instance_docker_image(", script)
        self.assertIn("swebench_official_image=True", script)
        self.assertIn("platform=\"linux/amd64\"", script)
        self.assertIn("enable_browser=False", script)

    def test_launcher_parallelizes_missing_runtime_image_builds(self):
        script = SCRIPT.read_text()

        self.assertIn("PARALLEL_BUILDS=4", script)
        self.assertIn("--parallel-builds)", script)
        self.assertIn('--parallel-builds "$PARALLEL_BUILDS"', script)
        self.assertIn("from concurrent.futures import ThreadPoolExecutor, as_completed", script)
        self.assertIn("ThreadPoolExecutor(max_workers=active_workers)", script)
        self.assertIn("executor.submit(build_runtime_job, job)", script)

    def test_launcher_skips_exact_local_runtime_image_before_building(self):
        script = SCRIPT.read_text()

        self.assertIn("target_image = runtime_target_image(base_image, source_hash)", script)
        self.assertIn("if local_image_exists(probe_client, target_image):", script)
        self.assertIn('if local_image_exists(client, job["target_image"]):', script)
        self.assertIn("skip {target_image}", script)
        self.assertIn("continue", script)
        self.assertIn("build_runtime_image(", script)
        self.assertIn('if image_name != job["target_image"]:', script)

    def test_launcher_keeps_cli_surface_deduplicated(self):
        script = SCRIPT.read_text()

        self.assertIn("--config)", script)
        self.assertIn("--eval-limit)", script)
        self.assertIn("--parallel-builds)", script)
        self.assertIn("--tmux-session)", script)
        self.assertIn("--replace-session)", script)
        self.assertNotIn("-h|--help", script)

    def test_docs_explain_concise_usage_and_idempotence(self):
        doc = DOC.read_text()

        self.assertIn("scripts/prebuild_openhands_swebench_images_pod.sh", doc)
        self.assertIn("openhands-swebench-image-prebuild", doc)
        self.assertIn("rerun attaches", doc)
        self.assertIn("--replace-session", doc)
        self.assertIn("--parallel-builds N", doc)
        self.assertIn("already-built images are skipped", doc)


if __name__ == "__main__":
    unittest.main()
