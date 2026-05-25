import py_compile
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prebuild_openhands_swebench_images_pod.py"
DOC = REPO_ROOT / "docs" / "openhands_swebench_gpu_pod_eval.md"
COMMON = REPO_ROOT / "scripts" / "pod_startup_common.py"


class TestOpenHandsImagePrebuildPodLauncher:
    def test_launcher_python_syntax_is_valid(self):
        py_compile.compile(str(SCRIPT), doraise=True)
        py_compile.compile(str(COMMON), doraise=True)

    def test_launcher_rejects_nonpositive_parallel_builds_before_pod_checks(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--foreground", "--parallel-builds", "0"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "--parallel-builds must be positive" in result.stderr

    def test_launcher_rejects_replace_session_in_foreground_mode(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--foreground", "--replace-session"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "--replace-session only applies" in result.stderr

    def test_launcher_uses_tmux_and_attaches_to_existing_session_by_default(self):
        script = SCRIPT.read_text()

        assert 'self.tmux_session = "openhands-swebench-image-prebuild"' in script
        assert '"tmux", "has-session", "-t", self.tmux_session' in script
        assert "tmux session already exists: {self.tmux_session}" in script
        assert (
            'exec_process(["tmux", "attach-session", "-t", self.tmux_session])'
            in script
        )
        assert '"new-session"' in script
        assert '"-d"' in script
        assert '"-s"' in script
        assert "self.tmux_session" in script
        assert "exec > >(tee -a {shell_quote(self.tmux_log_path)}) 2>&1" in script
        assert "$command 2>&1 | tee" not in script

    def test_launcher_can_replace_existing_tmux_session(self):
        script = SCRIPT.read_text()

        assert '"--replace-session"' in script
        assert "self.replace_session = False" in script
        assert "replacing tmux session: {self.tmux_session}" in script
        assert '"tmux", "kill-session", "-t", self.tmux_session' in script
        assert "def kill_foreground_prebuilds" in script
        assert "terminating foreground prebuild process groups" in script
        assert "--replace-session only applies to tmux-supervised launches" in script

    def test_launcher_stores_and_compares_tmux_launch_context(self):
        script = SCRIPT.read_text()

        assert 'f"{self.tmux_session}.context.json"' in script
        assert "def write_tmux_context" in script
        assert "def compare_tmux_context" in script
        assert "self.compare_tmux_context(script_path, command)" in script
        assert "self.write_tmux_context(script_path, command)" in script
        assert '"@swehero_launch_context"' in script
        assert 'print(f"context: {self.tmux_context_path}")' in script

        main_block = script[script.index("def main(") :]
        assert main_block.index("resolve_eval_config(config_path)") < main_block.index(
            "launcher.launch_tmux_if_needed()"
        )

        launch_block = script[
            script.index("def launch_tmux_if_needed") : script.index(
                "def terminate_foreground_worker"
            )
        ]
        assert launch_block.index(
            "pod_startup_common.prepare_pod_checkout"
        ) < launch_block.index("self.write_tmux_context(script_path, command)")

    def test_launcher_context_compares_normalized_fields(self):
        script = SCRIPT.read_text()
        context_helper = script[
            script.index("def build_context") : script.index("def write_tmux_context")
        ]

        assert '"kind": "openhands_swebench_image_prebuild"' in context_helper
        assert '"requested": {' in context_helper
        assert '"parallel_builds": self.parallel_builds' in context_helper
        assert '"resolved_eval_config": {' in context_helper
        assert "def comparable_context(context: dict[str, Any])" in script
        assert '"created_at_utc": now_utc()' in context_helper
        assert '"foreground_command": foreground_command' in context_helper
        assert "LLM_API_KEY" not in context_helper

    def test_existing_session_without_matching_context_fails_clearly(self):
        script = SCRIPT.read_text()

        assert "launch context is missing" in script
        assert "different launch context" in script
        assert (
            "Use --replace-session to restart it with the requested context" in script
        )
        assert "or --tmux-session NAME to launch a separate prebuild" in script
        assert "foreground prebuild process already exists" in script
        assert "ensure_no_foreground_prebuild" in script

    def test_foreground_worker_cleans_up_child_process_group(self):
        script = SCRIPT.read_text()

        assert "--foreground-worker" in script
        assert "def run_supervised_foreground_worker" in script
        assert "subprocess.Popen(" in script
        assert '"setsid"' in script
        assert "*worker_command" in script
        assert "signal.signal(signum, self.terminate_foreground_worker)" in script
        assert "os.killpg(process.pid, signal.SIGTERM)" in script
        assert "os.killpg(process.pid, signal.SIGKILL)" in script
        assert "foreground prebuild received termination" in script
        assert '"--foreground-worker" not in tokens' in script

    def test_launcher_is_pod_only_and_enforces_pod_git_checkout(self):
        script = SCRIPT.read_text()
        common = COMMON.read_text()

        assert "pod_startup_common.require_pod_runtime" in script
        assert "this launcher is pod-only" in common
        assert 'Path("/workspace").is_dir()' in common
        assert "pod_git_guard.require_pod_git_checkout" in common
        assert "pod_startup_common.prepare_pod_checkout" in script
        assert "SWEHERO_POD_GIT_BRANCH" in script

    def test_launcher_derives_images_from_eval_preset_and_openhands_code(self):
        script = SCRIPT.read_text()

        assert "from scripts import openhands_swebench_eval as eval_script" in script
        assert '"EVAL_STACK": args.eval_stack' in script
        assert '"DATASET": args.dataset' in script
        assert '"SPLIT": args.split' in script
        assert '"OPENHANDS_REF": args.openhands_ref' in script
        assert '"OPENHANDS_DIR": eval_script.effective_openhands_dir(args)' in script
        assert "set_dataset_type(args.dataset)" in script
        assert "get_instance_docker_image(" in script
        assert "swebench_official_image=True" in script
        assert 'platform="linux/amd64"' in script
        assert "enable_browser=False" in script
        assert "from openhands import __version__ as openhands_version" in script
        assert "from openhands.version import get_version" not in script

    def test_launcher_supports_swe_lego_vendored_openhands_checkout(self):
        script = SCRIPT.read_text()

        assert '"SWE_LEGO_REPO": args.swe_lego_repo' in script
        assert '"SWE_LEGO_REF": args.swe_lego_ref' in script
        assert (
            '"SWE_LEGO_SWEBENCH_DIR": eval_script.effective_swebench_dir(args) or ""'
            in script
        )
        assert 'if self.eval_stack == "swe-lego":' in script
        assert '"git", "clone", self.swe_lego_repo, str(swe_lego_dir)' in script
        assert '"checkout"' in script
        assert '"--detach"' in script
        assert "self.swe_lego_ref" in script
        assert "SWE-Lego OpenHands directory missing: {self.openhands_dir}" in script
        assert (
            "SWE-Lego SWE-bench directory missing: {self.swe_lego_swebench_dir}"
            in script
        )

    def test_launcher_parallelizes_missing_runtime_image_builds(self):
        script = SCRIPT.read_text()

        assert "self.parallel_builds = 4" in script
        assert '"--parallel-builds"' in script
        assert '"--parallel-builds"' in script
        assert "self.parallel_builds" in script
        assert (
            "from concurrent.futures import ThreadPoolExecutor, as_completed" in script
        )
        assert "ThreadPoolExecutor(max_workers=active_workers)" in script
        assert "executor.submit(build_runtime_job, job)" in script

    def test_launcher_skips_exact_local_runtime_image_before_building(self):
        script = SCRIPT.read_text()

        assert "target_image = runtime_target_image(base_image, source_hash)" in script
        assert "if local_image_exists(probe_client, target_image):" in script
        assert 'if local_image_exists(client, job["target_image"]):' in script
        assert "skip {target_image}" in script
        assert "continue" in script
        assert "build_runtime_image(" in script
        assert 'if image_name != job["target_image"]:' in script

    def test_launcher_keeps_cli_surface_deduplicated(self):
        script = SCRIPT.read_text()

        assert '"--config"' in script
        assert '"--eval-limit"' in script
        assert '"--parallel-builds"' in script
        assert '"--tmux-session"' in script
        assert '"--replace-session"' in script
        assert "-h|--help" not in script

    def test_docs_explain_concise_usage_and_idempotence(self):
        doc = DOC.read_text()

        assert "scripts/run_midtraining_pod.py prebuild" in doc
        assert "openhands-swebench-image-prebuild" in doc
        assert "rerun attaches" in doc
        assert "--replace-session" in doc
        assert "--parallel-builds N" in doc
        assert ".context.json" in doc
        assert "already-built images are skipped" in doc
