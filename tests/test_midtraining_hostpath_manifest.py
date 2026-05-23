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
            'apt-get install -y --no-install-recommends "${missing_packages[@]}"',
            manifest,
        )


if __name__ == "__main__":
    unittest.main()
