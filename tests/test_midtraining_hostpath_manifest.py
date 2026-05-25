from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMidtrainingHostpathManifest:
    def test_canonical_pod_manifest_installs_launch_prerequisites(self):
        manifest = (REPO_ROOT / "manifests" / "midtraining-hostpath.yaml").read_text()

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
        assert (
            "command -v docker >/dev/null 2>&1 || missing_packages+=(docker.io)"
            in manifest
        )
        assert (
            "docker buildx version >/dev/null 2>&1 || missing_packages+=(docker-buildx)"
            in manifest
        )
        assert "command -v curl >/dev/null 2>&1 || missing_packages+=(curl)" in manifest
        assert (
            'apt-get install -y --no-install-recommends "${missing_packages[@]}"'
            in manifest
        )
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
        assert "securityContext:\n        privileged: true" in manifest
        assert "mountPath: /var/lib/docker" in manifest
        assert "path: /workspace/pod-docker-data/midtraining-dev" in manifest
        assert "mountPath: /var/run/midtraining-git-ssh" in manifest
        assert "secretName: midtraining-git-ssh" in manifest
        assert "optional: true" in manifest
        assert "defaultMode: 0400" in manifest
