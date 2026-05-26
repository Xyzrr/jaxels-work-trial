"""Tests for the canonical GPU pod manifest.

The manifest is not ML code, but it defines the runtime substrate for every
model-training and SWE-bench eval job in this repo. These assertions document
the pod guarantees that make those jobs reproducible: persistent workspace and
Docker state, enough GPU access, Git/SSH setup for pinned code revisions, and
tooling needed by TorchTitan, vLLM, OpenHands, and SWE-bench.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "manifests" / "midtraining-hostpath.yaml"


class TestMidtrainingHostpathManifest:
    """Guard the pod setup that training/eval launchers assume exists."""

    def test_canonical_pod_manifest_installs_launch_prerequisites(self) -> None:
        manifest = MANIFEST.read_text()

        # The pod image is intentionally thin, so the startup script repairs only
        # the tools required by the current launch stack. tmux keeps long GPU
        # jobs reconnectable; git/ssh make the pod checkout reproducible; cc and
        # Python support pinned virtualenv repairs; lspci/nvidia-smi diagnostics
        # help debug GPU visibility; Docker/buildx/curl/jq support SWE-bench and
        # local vLLM/OpenHands API checks.
        assert "missing_packages=()" in manifest
        assert "command -v tmux >/dev/null 2>&1 || missing_packages+=(tmux)" in manifest
        assert "command -v git >/dev/null 2>&1 || missing_packages+=(git)" in manifest
        assert (
            "command -v ssh >/dev/null 2>&1 || missing_packages+=(openssh-client)"
            in manifest
        )
        assert (
            "command -v cc >/dev/null 2>&1 || missing_packages+=(build-essential)"
            in manifest
        )
        assert (
            "command -v lspci >/dev/null 2>&1 || missing_packages+=(pciutils)"
            in manifest
        )
        assert "command -v curl >/dev/null 2>&1 || missing_packages+=(curl)" in manifest
        assert (
            "command -v docker >/dev/null 2>&1 || missing_packages+=(docker.io)"
            in manifest
        )
        assert "command -v jq >/dev/null 2>&1 || missing_packages+=(jq)" in manifest
        assert (
            "command -v python3 >/dev/null 2>&1 || missing_packages+=(python3 python3-venv python3-pip)"
            in manifest
        )
        assert (
            "docker buildx version >/dev/null 2>&1 || missing_packages+=(docker-buildx)"
            in manifest
        )
        assert (
            'apt-get install -y --no-install-recommends "${missing_packages[@]}"'
            in manifest
        )

        # Git SSH credentials are mounted from a Kubernetes secret rather than
        # baked into the image. StrictHostKeyChecking plus a provided
        # known_hosts file avoids the tempting but unsafe pattern of trusting a
        # live ssh-keyscan during pod startup.
        assert (
            "install -m 600 /var/run/midtraining-git-ssh/id_ed25519 /root/.ssh/id_ed25519"
            in manifest
        )
        assert "Host github.com" in manifest
        assert "IdentityFile /root/.ssh/id_ed25519" in manifest
        assert "StrictHostKeyChecking yes" in manifest
        assert (
            "git config --global include.path /var/run/midtraining-git-ssh/gitconfig"
            in manifest
        )
        assert (
            "git config --global --add safe.directory /workspace/jaxels-work-trial"
            in manifest
        )
        assert "ssh-keyscan github.com" not in manifest

        # OpenHands/SWE-bench grading builds and runs Docker images inside this
        # pod. Privileged mode and persistent Docker graph storage are deliberate:
        # they allow Docker-in-pod and keep expensive runtime images warm across
        # repeated eval/prebuild runs.
        assert "securityContext:\n        privileged: true" in manifest
        assert "mountPath: /var/lib/docker" in manifest
        assert "path: /workspace/pod-docker-data/midtraining-dev" in manifest

        # The optional secret keeps local development flexible, but when present
        # it is mounted read-only with restrictive permissions so Git operations
        # can authenticate without broadening file access inside the pod.
        assert "mountPath: /var/run/midtraining-git-ssh" in manifest
        assert "secretName: midtraining-git-ssh" in manifest
        assert "optional: true" in manifest
        assert "defaultMode: 0400" in manifest
