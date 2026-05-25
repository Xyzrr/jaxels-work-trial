import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MidtrainingHostpathManifestTests(unittest.TestCase):
    def test_canonical_pod_manifest_installs_launch_prerequisites(self):
        manifest = (REPO_ROOT / "manifests" / "midtraining-hostpath.yaml").read_text()

        self.assertIn("missing_packages=()", manifest)
        self.assertIn(
            "command -v tmux >/dev/null 2>&1 || missing_packages+=(tmux)",
            manifest,
        )
        self.assertIn(
            "command -v git >/dev/null 2>&1 || missing_packages+=(git)",
            manifest,
        )
        self.assertIn(
            "command -v ssh >/dev/null 2>&1 || missing_packages+=(openssh-client)",
            manifest,
        )
        self.assertIn(
            "command -v cc >/dev/null 2>&1 || missing_packages+=(build-essential)",
            manifest,
        )
        self.assertIn(
            "command -v lspci >/dev/null 2>&1 || missing_packages+=(pciutils)",
            manifest,
        )
        self.assertIn(
            "command -v docker >/dev/null 2>&1 || missing_packages+=(docker.io)",
            manifest,
        )
        self.assertIn(
            "docker buildx version >/dev/null 2>&1 || missing_packages+=(docker-buildx)",
            manifest,
        )
        self.assertIn(
            "command -v curl >/dev/null 2>&1 || missing_packages+=(curl)",
            manifest,
        )
        self.assertIn(
            'apt-get install -y --no-install-recommends "${missing_packages[@]}"',
            manifest,
        )
        self.assertIn(
            "install -m 600 /var/run/midtraining-git-ssh/id_ed25519 /root/.ssh/id_ed25519",
            manifest,
        )
        self.assertIn("Host github.com", manifest)
        self.assertIn("IdentityFile /root/.ssh/id_ed25519", manifest)
        self.assertIn("StrictHostKeyChecking yes", manifest)
        self.assertIn(
            "git config --global include.path /var/run/midtraining-git-ssh/gitconfig",
            manifest,
        )
        self.assertIn(
            "git config --global --add safe.directory /workspace/jaxels-work-trial",
            manifest,
        )
        self.assertNotIn("ssh-keyscan github.com", manifest)
        self.assertIn("securityContext:\n        privileged: true", manifest)
        self.assertIn("mountPath: /var/lib/docker", manifest)
        self.assertIn("path: /workspace/pod-docker-data/midtraining-dev", manifest)
        self.assertIn("mountPath: /var/run/midtraining-git-ssh", manifest)
        self.assertIn("secretName: midtraining-git-ssh", manifest)
        self.assertIn("optional: true", manifest)
        self.assertIn("defaultMode: 0400", manifest)


if __name__ == "__main__":
    unittest.main()
