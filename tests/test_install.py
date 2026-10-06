"""GHCR install and publish contract. Does not pull or build images."""

import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MANAGER = "ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest"
INSTANCE = "ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130"
INSTANCE_CU124 = "ghcr.io/recognizeyourprivilege/comfyfleet-images:cu124"
OLD_INSTANCE = re.compile(
    r"ghcr\.io/recognizeyourprivilege/comfyfleet(?!-images|-manager)\b"
)


def _fake_docker(tmp: str) -> tuple[dict[str, str], Path]:
    bin_dir = Path(tmp) / "bin"
    bin_dir.mkdir()
    log = Path(tmp) / "docker.log"
    fake = bin_dir / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin", "DOCKER_LOG": str(log)}
    return env, log


def _pull_lines(log: Path) -> list[str]:
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("pull ")]


class InstallScriptTests(unittest.TestCase):
    def test_install_script_points_only_at_the_manager_package(self):
        script = ROOT / "install.sh"
        text = script.read_text(encoding="utf-8")
        self.assertTrue(script.stat().st_mode & 0o111, "install.sh must be executable")
        self.assertIn('MANAGER_REPO="ghcr.io/recognizeyourprivilege/comfyfleet-manager"', text)
        self.assertIn('INSTANCE_REPO="ghcr.io/recognizeyourprivilege/comfyfleet-images"', text)
        self.assertNotIn("comfyfleet-manager-" + "legacy", text)
        self.assertIsNone(OLD_INSTANCE.search(text), text)
        self.assertNotRegex(text, r"comfyfleet-manager-[A-Za-z0-9]")
        self.assertIn("--with-images", text)
        self.assertIn("--cuda-tag", text)
        self.assertIn("--pull-only", text)
        self.assertIn("--compose", text)
        self.assertIn("COMFYFLEET_INSTANCE_IMAGE", text)
        self.assertIn("COMFYFLEET_PASSWORD", text)
        self.assertIn("COMFYFLEET_PUBLIC_HOST", text)
        self.assertIn("/var/run/docker.sock:/var/run/docker.sock", text)
        self.assertIn("/home/ComfyFleet:/home/ComfyFleet", text)
        self.assertNotIn("/home:/home", text)
        self.assertIn("--gpus all", text)
        self.assertIn("9100:9100", text)
        self.assertIn("<ip-address>", text)
        self.assertNotIn("192.168" + ".", text)
        self.assertNotIn("rm -rf", text)
        self.assertNotIn("volume rm", text)
        self.assertNotIn("system prune", text)
        self.assertNotIn('echo "${COMFYFLEET_PASSWORD', text)
        self.assertNotIn("echo ${COMFYFLEET_PASSWORD", text)
        checked = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        help_run = subprocess.run(
            ["bash", str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn(MANAGER, help_run.stdout)
        self.assertIn(INSTANCE, help_run.stdout)
        self.assertIn(INSTANCE_CU124, help_run.stdout)
        self.assertIn("--with-images", help_run.stdout)
        self.assertIn("<ip-address>", help_run.stdout)
        self.assertNotIn("192.168" + ".", help_run.stdout)

        missing = subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("COMFYFLEET_PASSWORD", missing.stderr)
        self.assertNotIn("docker pull", missing.stderr)

        bad_host = subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "COMFYFLEET_PASSWORD": "not-printed",
                "COMFYFLEET_PUBLIC_HOST": "not a host",
            },
        )
        self.assertNotEqual(bad_host.returncode, 0)
        self.assertIn("COMFYFLEET_PUBLIC_HOST", bad_host.stderr)
        self.assertNotIn("not-printed", bad_host.stderr)
        self.assertNotIn("docker pull", bad_host.stderr)

        rejected = subprocess.run(
            ["bash", str(script), "--cuda-tag", "cu128", "--pull-only"],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("cu130 or cu124", rejected.stderr)
        self.assertNotIn("docker pull", rejected.stderr)

        both = subprocess.run(
            ["bash", str(script), "--cuda-tag", "both", "--pull-only"],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertNotEqual(both.returncode, 0)
        self.assertIn("--with-images", both.stderr)
        self.assertNotIn("docker pull", both.stderr)

        images = subprocess.run(
            ["bash", str(script), "--with-images", "cu128", "--pull-only"],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertNotEqual(images.returncode, 0)
        self.assertIn("--with-images", images.stderr)
        self.assertNotIn("docker pull", images.stderr)

    def test_default_pull_is_manager_only_and_with_images_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, log = _fake_docker(tmp)
            script = str(ROOT / "install.sh")

            silent = subprocess.run(
                ["bash", script, "--pull-only"],
                check=False,
                capture_output=True,
                text=True,
                env=base,
            )
            self.assertEqual(silent.returncode, 0, silent.stderr)
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}"])
            self.assertIn("no instance image was pulled", silent.stdout)
            self.assertNotIn("run -d", log.read_text(encoding="utf-8"))

            log.write_text("", encoding="utf-8")
            cu124 = subprocess.run(
                ["bash", script, "--pull-only", "--cuda-tag", "cu124"],
                check=False,
                capture_output=True,
                text=True,
                env=base,
            )
            self.assertEqual(cu124.returncode, 0, cu124.stderr)
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}"])
            self.assertIn(INSTANCE_CU124, cu124.stdout)
            self.assertIn("no instance image was pulled", cu124.stdout)

            log.write_text("", encoding="utf-8")
            one = subprocess.run(
                ["bash", script, "--pull-only", "--with-images", "cu130"],
                check=False,
                capture_output=True,
                text=True,
                env=base,
            )
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}", f"pull {INSTANCE}"])

            log.write_text("", encoding="utf-8")
            other = subprocess.run(
                ["bash", script, "--pull-only", "--with-images", "cu124"],
                check=False,
                capture_output=True,
                text=True,
                env=base,
            )
            self.assertEqual(other.returncode, 0, other.stderr)
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}", f"pull {INSTANCE_CU124}"])

            log.write_text("", encoding="utf-8")
            both = subprocess.run(
                ["bash", script, "--pull-only", "--with-images", "both"],
                check=False,
                capture_output=True,
                text=True,
                env=base,
            )
            self.assertEqual(both.returncode, 0, both.stderr)
            self.assertEqual(
                _pull_lines(log),
                [f"pull {MANAGER}", f"pull {INSTANCE}", f"pull {INSTANCE_CU124}"],
            )

            log.write_text("", encoding="utf-8")
            mixed = subprocess.run(
                ["bash", script, "--pull-only", "--cuda-tag", "cu124", "--with-images", "cu130"],
                check=False,
                capture_output=True,
                text=True,
                env=base,
            )
            self.assertEqual(mixed.returncode, 0, mixed.stderr)
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}", f"pull {INSTANCE}"])
            self.assertIn(INSTANCE_CU124, mixed.stdout)

    def test_start_sets_the_instance_image_without_pulling_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, log = _fake_docker(tmp)
            started = subprocess.run(
                ["bash", str(ROOT / "install.sh"), "--cuda-tag", "cu124"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **base,
                    "COMFYFLEET_PASSWORD": "not-printed",
                    "COMFYFLEET_PUBLIC_HOST": "0.0.0.0",
                },
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            self.assertNotIn("not-printed", started.stdout + started.stderr)
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}"])
            recorded = log.read_text(encoding="utf-8").splitlines()
            run_lines = [line for line in recorded if line.startswith("run -d --name ")]
            self.assertEqual(len(run_lines), 1, recorded)
            run = run_lines[0]
            self.assertIn("run -d --name comfyfleet-manager ", run)
            self.assertIn("--restart unless-stopped", run)
            self.assertIn("--gpus all", run)
            self.assertIn("-p 9100:9100", run)
            self.assertIn("-v /var/run/docker.sock:/var/run/docker.sock", run)
            self.assertIn("-v /home/ComfyFleet:/home/ComfyFleet", run)
            self.assertIn(f"-e COMFYFLEET_INSTANCE_IMAGE={INSTANCE_CU124}", run)
            self.assertIn("-e COMFYFLEET_CUDA_TAG=cu124", run)
            self.assertIn("-e COMFYFLEET_PUBLIC_HOST=0.0.0.0", run)
            self.assertTrue(run.endswith(MANAGER))

    def test_rerun_replaces_the_manager_container_and_keeps_data(self):
        """A running manager is replaced in place.

        The previous image does not matter: the container name, port 9100,
        and /home/ComfyFleet mount stay. docker rm -f does not delete that
        directory or any volume.
        """

        with tempfile.TemporaryDirectory() as tmp:
            base, log = _fake_docker(tmp)
            started = subprocess.run(
                ["bash", str(ROOT / "install.sh")],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **base,
                    "COMFYFLEET_PASSWORD": "not-printed",
                    "COMFYFLEET_PUBLIC_HOST": "lan-host.example",
                },
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            self.assertIn("replacing container comfyfleet-manager", started.stdout)
            self.assertIn("Files under /home/ComfyFleet are kept", started.stdout)
            self.assertNotIn("not-printed", started.stdout + started.stderr)
            recorded = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(_pull_lines(log), [f"pull {MANAGER}"])
            self.assertIn("container inspect comfyfleet-manager", recorded)
            self.assertIn("rm -f comfyfleet-manager", recorded)
            run_lines = [line for line in recorded if line.startswith("run -d --name ")]
            self.assertEqual(len(run_lines), 1, recorded)
            run = run_lines[0]
            self.assertIn("-p 9100:9100", run)
            self.assertIn("-v /home/ComfyFleet:/home/ComfyFleet", run)
            self.assertTrue(run.endswith(MANAGER))
            self.assertLess(recorded.index("rm -f comfyfleet-manager"), recorded.index(run))
            self.assertFalse(any("volume" in line for line in recorded))
            self.assertFalse(any(line.startswith("system prune") or "system prune" in line for line in recorded))
            self.assertNotIn("rm -rf", "\n".join(recorded))

    def test_manager_digest_stays_on_the_manager_package(self):
        digest = "sha256:" + ("ab" * 32)
        with tempfile.TemporaryDirectory() as tmp:
            base, log = _fake_docker(tmp)
            pinned = subprocess.run(
                ["bash", str(ROOT / "install.sh"), "--pull-only"],
                check=False,
                capture_output=True,
                text=True,
                env={**base, "COMFYFLEET_MANAGER_DIGEST": digest},
            )
            self.assertEqual(pinned.returncode, 0, pinned.stderr)
            pin = f"ghcr.io/recognizeyourprivilege/comfyfleet-manager@{digest}"
            self.assertEqual(_pull_lines(log), [f"pull {pin}"])

    def test_workflow_publishes_only_the_manager_image(self):
        text = (ROOT / ".github" / "workflows" / "publish-manager.yml").read_text(encoding="utf-8")
        self.assertIn("branches:\n      - main\n", text)
        self.assertIn("packages: write", text)
        self.assertIn("secrets.GITHUB_TOKEN", text)
        self.assertIn("file: Dockerfile.manager\n", text)
        self.assertIn("linux/amd64", text)
        self.assertIn("ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest", text)
        self.assertIn("ghcr.io/recognizeyourprivilege/comfyfleet-manager:${{ github.sha }}", text)
        self.assertIn(
            "org.opencontainers.image.source=https://github.com/RecognizeYourPrivilege/ComfyFleet-Manager",
            text,
        )
        self.assertIn("provenance: false", text)
        self.assertNotIn("Dockerfile.cu124", text)
        self.assertNotIn("comfyfleet-images", text)
        self.assertNotIn("comfyfleet-manager-" + "legacy", text)
        self.assertIsNone(OLD_INSTANCE.search(text))

    def test_compose_uses_the_manager_image_and_instance_default(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        self.assertIn(MANAGER, compose)
        self.assertIn(INSTANCE, compose)
        self.assertIn("container_name: comfyfleet-manager", compose)
        self.assertIn("--shm-size 8g", compose)
        self.assertIn("shm_size: '8g'", compose)
        self.assertNotIn("dockerfile:", compose.lower())
        self.assertNotIn("comfyfleet-manager-" + "legacy", compose)
        self.assertIsNone(OLD_INSTANCE.search(compose))
        overlay = (ROOT / "compose.build.yaml").read_text(encoding="utf-8")
        self.assertIn("Dockerfile.manager", overlay)
        self.assertIn(INSTANCE, overlay)
        self.assertIn(MANAGER, overlay)
        self.assertNotIn("Dockerfile.cu124", overlay)

    def test_readme_legacy_line_is_the_only_one_in_the_tree(self):
        needle = "comfyfleet-manager-" + "legacy"
        hits = []
        skipped_suffixes = {".jpg", ".png", ".gif", ".webp", ".pyc"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in skipped_suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("192.168" + ".", text, path)
            self.assertIsNone(OLD_INSTANCE.search(text), path)
            if needle in text:
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(hits, ["README.md"])
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count(needle), 1)
        self.assertIn(
            "docker image rm ghcr.io/recognizeyourprivilege/comfyfleet-manager-" + "legacy:latest",
            readme,
        )
        self.assertLess(readme.index("## Install with Docker"), readme.index("## Update"))
        self.assertLess(readme.index("## Update"), readme.index("## Free up space"))
        self.assertLess(readme.index("## Free up space"), readme.index("## Features"))
        self.assertLess(readme.index("## Features"), readme.index("## Build your own"))
        self.assertIn("--with-images cu130", readme)
        self.assertIn("--cuda-tag cu124", readme)
        self.assertIn(INSTANCE, readme)
        self.assertIn(INSTANCE_CU124, readme)
        self.assertIn(MANAGER, readme)


if __name__ == "__main__":
    unittest.main()
