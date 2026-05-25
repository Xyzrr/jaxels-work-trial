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

    def test_launcher_stores_and_compares_tmux_launch_context(self):
        script = SCRIPT.read_text()

        self.assertIn('TMUX_CONTEXT_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.context.json"', script)
        self.assertIn("write_tmux_context()", script)
        self.assertIn("compare_tmux_context()", script)
        self.assertIn('compare_tmux_context "$TMUX_CONTEXT_PATH" "$script_path" "$command"', script)
        self.assertIn('write_tmux_context "$TMUX_CONTEXT_PATH" "$script_path" "$command"', script)
        self.assertIn('@swehero_launch_context "$TMUX_CONTEXT_PATH"', script)
        self.assertIn('echo "context: $TMUX_CONTEXT_PATH"', script)
        launch_block = script[
            script.index('if [[ "$LAUNCH_SESSION" == "1" ]]') :
            script.index('if ! tmux new-session')
        ]
        self.assertLess(
            launch_block.index("ensure_pod_git_checkout"),
            launch_block.index('eval "$(resolve_eval_config "$CONFIG_PRESET_PATH")"'),
        )
        self.assertLess(
            launch_block.index('eval "$(resolve_eval_config "$CONFIG_PRESET_PATH")"'),
            launch_block.index('write_tmux_context "$TMUX_CONTEXT_PATH" "$script_path" "$command"'),
        )

    def test_launcher_context_compares_normalized_fields(self):
        script = SCRIPT.read_text()
        context_helper = script[
            script.index("tmux_launch_context()") : script.index("ensure_docker()")
        ]

        self.assertIn('"kind": "openhands_swebench_image_prebuild"', context_helper)
        self.assertIn('"requested": {', context_helper)
        self.assertIn('"parallel_builds": int(parallel_builds)', context_helper)
        self.assertIn('"resolved_eval_config": {', context_helper)
        self.assertIn('def comparable_context(context: dict[str, Any])', context_helper)
        self.assertIn('"created_at_utc": now_utc()', context_helper)
        self.assertIn('"foreground_command": foreground_command', context_helper)
        self.assertNotIn("LLM_API_KEY", context_helper)

    def test_existing_session_without_matching_context_fails_clearly(self):
        script = SCRIPT.read_text()

        self.assertIn("launch context is missing", script)
        self.assertIn("different launch context", script)
        self.assertIn("Use --replace-session to restart it with the requested context", script)
        self.assertIn("or --tmux-session NAME to launch a separate prebuild", script)

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
        self.assertIn('"EVAL_STACK": args.eval_stack', script)
        self.assertIn('"DATASET": args.dataset', script)
        self.assertIn('"SPLIT": args.split', script)
        self.assertIn('"OPENHANDS_REF": args.openhands_ref', script)
        self.assertIn('"OPENHANDS_DIR": eval_script.effective_openhands_dir(args)', script)
        self.assertIn("set_dataset_type(args.dataset)", script)
        self.assertIn("get_instance_docker_image(", script)
        self.assertIn("swebench_official_image=True", script)
        self.assertIn("platform=\"linux/amd64\"", script)
        self.assertIn("enable_browser=False", script)

    def test_launcher_supports_swe_lego_vendored_openhands_checkout(self):
        script = SCRIPT.read_text()

        self.assertIn('"SWE_LEGO_REPO": args.swe_lego_repo', script)
        self.assertIn('"SWE_LEGO_REF": args.swe_lego_ref', script)
        self.assertIn(
            '"SWE_LEGO_SWEBENCH_DIR": eval_script.effective_swebench_dir(args) or ""',
            script,
        )
        self.assertIn('if [[ "$EVAL_STACK" == "swe-lego" ]]; then', script)
        self.assertIn('git clone "$SWE_LEGO_REPO" "$SWE_LEGO_DIR"', script)
        self.assertIn('git -C "$SWE_LEGO_DIR" checkout --detach "$SWE_LEGO_REF"', script)
        self.assertIn('SWE-Lego OpenHands directory missing: $OPENHANDS_DIR', script)
        self.assertIn('SWE-Lego SWE-bench directory missing: $SWE_LEGO_SWEBENCH_DIR', script)

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
        self.assertIn(".context.json", doc)
        self.assertIn("already-built images are skipped", doc)


if __name__ == "__main__":
    unittest.main()
