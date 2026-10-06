"""Manager image contract and in-process fleet lifecycle.

Live ``docker build`` of the manager is ``scripts/manager-smoke.sh``.
This module does not need a GPU or a Docker daemon.
"""

import json
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from comfyfleet.auth import LoginGuard, SessionStore
from comfyfleet.control import list_url_host, resolve_instance_image
from comfyfleet.docker import (
    DockerCLI,
    engine_problem,
    explain_docker_failure,
    parse_published_ports,
)
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu, detect_gpus
from comfyfleet.http_api import ApiContext, dispatch, make_server
from comfyfleet.paths import DEFAULT_IMAGE, FleetLayout
from comfyfleet.public_host import configured_public_host, open_host

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "test-password"


class FakeDocker:
    def __init__(self):
        self.containers = {}
        self.calls = []

    def create(self, args):
        self.calls.append(("create", list(args)))
        name = args[args.index("--name") + 1]
        self.containers[name] = {"status": "created", "args": list(args)}

    def start(self, name):
        self.calls.append(("start", name))
        self.containers[name]["status"] = "running"

    def stop(self, name):
        self.calls.append(("stop", name))
        self.containers[name]["status"] = "exited"

    def remove(self, name):
        self.calls.append(("rm", name))
        self.containers.pop(name, None)

    def update_restart(self, name, policy):
        self.calls.append(("update-restart", name, policy))

    def status(self, name):
        item = self.containers.get(name)
        if item is None:
            return None
        return item["status"]

    def running_names(self):
        return [name for name, item in self.containers.items() if item["status"] == "running"]


def _workflow() -> bytes:
    payload = {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "version": 0.4,
    }
    return json.dumps(payload).encode("utf-8")


def _multipart(filename: str, content: bytes, fields: list[tuple[str, str]]):
    boundary = "----comfyfleetmanager"
    marker = boundary.encode("ascii")
    chunks = []
    for key, value in fields:
        chunks.append(
            b"--"
            + marker
            + b"\r\n"
            + f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            + value.encode()
            + b"\r\n"
        )
    chunks.append(
        b"--"
        + marker
        + b"\r\n"
        + f'Content-Disposition: form-data; name="workflow"; filename="{filename}"\r\n'.encode()
        + b"Content-Type: application/json\r\n\r\n"
        + content
        + b"\r\n"
    )
    chunks.append(b"--" + marker + b"--\r\n")
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class ManagerImageContractTests(unittest.TestCase):
    def test_manager_dockerfile_serves_ui_without_cuda(self):
        text = (ROOT / "Dockerfile.manager").read_text(encoding="utf-8")
        self.assertIn("FROM debian:bookworm-slim", text)
        self.assertIn("EXPOSE 9100", text)
        self.assertIn("COMFYFLEET_BIND_HOST=0.0.0.0", text)
        self.assertIn("COMFYFLEET_BIND_PORT=9100", text)
        self.assertIn("COMFYFLEET_MANAGER=1", text)
        self.assertIn("manager-entrypoint.sh", text)
        self.assertIn("COPY comfyfleet", text)
        self.assertIn("\n        git \\\n", text)
        self.assertIn("COPY ui", text)
        self.assertIn("/usr/local/bin/docker", text)
        self.assertNotIn("cuda-libraries", text)
        self.assertNotIn("torch==", text)
        self.assertNotIn("dockerd", text)
        self.assertNotIn("Dockerfile.cu124", text)

    def test_entrypoint_boots_the_ui_and_warns(self):
        script = ROOT / "docker" / "manager-entrypoint.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("COMFYFLEET_MANAGER=1", text)
        self.assertIn("comfyfleet ui --host", text)
        self.assertIn("0.0.0.0", text)
        self.assertIn("/var/run/docker.sock", text)
        self.assertIn("permission denied", text)
        self.assertIn("nvidia-smi", text)
        self.assertIn("--gpus all", text)
        self.assertIn("/home/ComfyFleet", text)
        self.assertIn("-v /home/ComfyFleet:/home/ComfyFleet", text)
        self.assertNotIn("-v /home:/home", text)
        self.assertIn("COMFYFLEET_PUBLIC_HOST", text)
        self.assertIn("COMFYFLEET_PASSWORD", text)
        self.assertIn("refusing to start", text)
        self.assertIn("no open-LAN fallback", text)
        self.assertIn("does not run a Docker daemon", text)
        refused = subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "COMFYFLEET_PASSWORD": "   "},
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("COMFYFLEET_PASSWORD", refused.stderr)
        self.assertIn("refusing to start", refused.stderr)
        self.assertNotIn("super-secret-value", refused.stderr)
        checked = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_compose_and_readme_lead_with_the_manager(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        self.assertIn("ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest", compose)
        self.assertIn("/var/run/docker.sock:/var/run/docker.sock", compose)
        self.assertIn('"9100:9100"', compose)
        self.assertIn("/home/ComfyFleet:/home/ComfyFleet", compose)
        self.assertNotIn("/home:/home", compose)
        self.assertIn("gpus: all", compose)
        self.assertIn("COMFYFLEET_PUBLIC_HOST", compose)
        self.assertIn("COMFYFLEET_PASSWORD", compose)
        self.assertIn("ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130", compose)
        self.assertIn("COMFYFLEET_CUDA_TAG", compose)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertLess(readme.index("## Install with Docker"), readme.index("## Update"))
        self.assertLess(readme.index("## Update"), readme.index("## Free up space"))
        self.assertLess(readme.index("## Free up space"), readme.index("## Features"))
        self.assertLess(readme.index("## Features"), readme.index("## Build your own"))
        self.assertIn("ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest", readme)
        self.assertIn("-p 9100:9100", readme)
        self.assertIn("/var/run/docker.sock:/var/run/docker.sock", readme)
        self.assertIn("-v /home/ComfyFleet:/home/ComfyFleet", readme)
        self.assertNotIn("-v /home:/home", readme)
        self.assertIn("COMFYFLEET_PUBLIC_HOST", readme)
        self.assertIn("-e COMFYFLEET_PASSWORD='your-password'", readme)
        self.assertIn("--gpus all", readme)
        self.assertIn("nvidia-smi", readme)
        self.assertNotIn("Phase ", readme)
        self.assertEqual(readme.count("comfyfleet-manager-" + "legacy"), 1)
        ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("examples", ignore)
        self.assertIn("/comfyfleet-logo-ships.jpg", ignore)
        self.assertNotIn("comfyfleet", ignore)
        self.assertNotIn("pyproject.toml", ignore)

    def test_smoke_script_documents_the_live_check(self):
        script = ROOT / "scripts" / "manager-smoke.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("Dockerfile.manager", text)
        self.assertIn("/api/health", text)
        self.assertIn("ComfyFleet", text)
        self.assertIn("COMFYFLEET_PASSWORD", text)
        self.assertIn("smoke-not-a-real-secret", text)
        checked = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)


class PublicHostTests(unittest.TestCase):
    def test_env_overrides_request_host(self):
        self.assertEqual(
            open_host("127.0.0.1:9100", "127.0.0.1", public_host="lan-host.example"),
            "lan-host.example",
        )
        self.assertEqual(
            open_host("phone.lan:9100", "127.0.0.1", public_host="gpu.lan:9100"),
            "gpu.lan",
        )

    def test_request_host_used_when_env_unset(self):
        self.assertEqual(open_host("phone.lan:9100", "127.0.0.1", None), "phone.lan")
        self.assertEqual(open_host("[::1]:9100", "127.0.0.1"), "[::1]")

    def test_unsafe_request_host_falls_back(self):
        self.assertEqual(open_host("0.0.0.0:9100", "127.0.0.1"), "127.0.0.1")
        self.assertEqual(open_host("bad host", "127.0.0.1"), "127.0.0.1")
        self.assertEqual(open_host(None, "127.0.0.1"), "127.0.0.1")

    def test_invalid_public_host_is_an_error(self):
        with self.assertRaises(FleetError) as ctx:
            open_host("phone.lan", "127.0.0.1", public_host="http://evil")
        self.assertIn("COMFYFLEET_PUBLIC_HOST", str(ctx.exception))
        with mock.patch.dict(os.environ, {"COMFYFLEET_PUBLIC_HOST": "not a host"}):
            with self.assertRaises(FleetError):
                configured_public_host()
        with mock.patch.dict(os.environ, {"COMFYFLEET_PUBLIC_HOST": "  "}):
            self.assertIsNone(configured_public_host())
        with mock.patch.dict(os.environ, {"COMFYFLEET_PUBLIC_HOST": "0.0.0.0"}):
            self.assertEqual(configured_public_host(), "0.0.0.0")
        self.assertEqual(open_host("phone.lan", "127.0.0.1", public_host="0.0.0.0"), "0.0.0.0")


class EngineErrorTests(unittest.TestCase):
    def test_missing_socket(self):
        message = engine_problem(environ={}, exists=lambda _path: False, access=lambda *_args: False)
        self.assertIn("missing", message)
        self.assertIn("/var/run/docker.sock", message)
        self.assertIn("sibling", message)
        self.assertIn("does not start a Docker daemon", message)

    def test_permission_denied_on_socket(self):
        message = engine_problem(
            environ={},
            exists=lambda _path: True,
            access=lambda *_args: False,
        )
        self.assertIn("permission denied", message)
        self.assertIn("root-equivalent", message)

    def test_socket_ok_and_non_unix_host(self):
        self.assertIsNone(
            engine_problem(
                environ={},
                exists=lambda _path: True,
                access=lambda *_args: True,
            )
        )
        self.assertIsNone(
            engine_problem(
                environ={"DOCKER_HOST": "tcp://127.0.0.1:2375"},
                exists=lambda _path: False,
                access=lambda *_args: False,
            )
        )

    def test_cli_permission_and_nvidia_errors(self):
        permission = explain_docker_failure(
            "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock"
        )
        self.assertIn("permission denied", permission)
        self.assertIn("root-equivalent", permission)
        missing = explain_docker_failure(
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
        )
        self.assertIn("cannot reach the Docker engine", missing)
        nvidia = explain_docker_failure(
            "could not select device driver \"\" with capabilities: [[gpu]] nvidia"
        )
        self.assertIn("NVIDIA Container Toolkit", nvidia)

    def test_injected_runner_skips_the_live_socket(self):
        def run(argv):
            return subprocess.CompletedProcess(argv, 0, "created\n", "")

        cli = DockerCLI(run=run)
        self.assertEqual(cli.status("portrait"), "created")

    def test_published_ports_parse(self):
        text = "0.0.0.0:8188->8188/tcp, :::8188->8188/tcp\n127.0.0.1:9100->9100/tcp\n"
        self.assertEqual(parse_published_ports(text), {8188, 9100})

    def test_published_ports_use_the_host_side_not_the_container_port(self):
        text = "0.0.0.0:8189->8188/tcp, [::]:8189->8188/tcp\n"
        self.assertEqual(parse_published_ports(text), {8189})

    def test_host_listener_probe_reads_proc_on_the_host_network(self):
        seen: list[list[str]] = []

        def run(argv):
            seen.append(list(argv))
            if len(argv) > 1 and argv[1] == "run":
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    "   0: 00000000:1FFC 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 1 1 0 100 0 0 10 0\n",
                    "",
                )
            return subprocess.CompletedProcess(argv, 1, "", "unexpected")

        cli = DockerCLI(run=run)
        with mock.patch.dict(os.environ, {"COMFYFLEET_MANAGER_IMAGE": "comfyfleet-manager:latest"}):
            text = cli.host_tcp_tables()
        self.assertEqual(len(seen), 1)
        argv = seen[0]
        self.assertEqual(argv[0], "docker")
        self.assertIn("--network", argv)
        self.assertEqual(argv[argv.index("--network") + 1], "host")
        self.assertEqual(argv[argv.index("--entrypoint") + 1], "/usr/bin/python3")
        self.assertIn("comfyfleet-manager:latest", argv)
        self.assertIn("1FFC", text)

    def test_missing_nvidia_smi_mentions_manager_gpus(self):
        with self.assertRaises(FleetError) as ctx:
            detect_gpus(run=lambda _argv: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
        message = str(ctx.exception)
        self.assertIn("nvidia-smi", message)
        self.assertIn("--gpus all", message)
        self.assertIn("NVIDIA Container Toolkit", message)


class EnsureImageTests(unittest.TestCase):
    def test_pulls_only_when_the_image_is_missing(self):
        calls: list[list[str]] = []

        def run(argv):
            calls.append(list(argv))
            if argv[1:3] == ["image", "inspect"]:
                return subprocess.CompletedProcess(argv, 1, "", "No such image")
            return subprocess.CompletedProcess(argv, 0, "", "")

        ref = "ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130"
        DockerCLI(run=run).ensure_image(ref)
        self.assertEqual(calls[0][:3], ["docker", "image", "inspect"])
        self.assertEqual(calls[0][-1], ref)
        self.assertEqual(calls[1], ["docker", "pull", ref])

    def test_skips_pull_when_the_image_is_present(self):
        calls: list[list[str]] = []

        def run(argv):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, "[]", "")

        ref = "ghcr.io/recognizeyourprivilege/comfyfleet-images:cu124"
        DockerCLI(run=run).ensure_image(ref)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:3], ["docker", "image", "inspect"])

    def test_list_urls_keep_a_zero_public_host(self):
        with mock.patch.dict(os.environ, {"COMFYFLEET_PUBLIC_HOST": "0.0.0.0"}):
            self.assertEqual(list_url_host(), "0.0.0.0")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(list_url_host(), "0.0.0.0")


class InstanceImageOverrideTests(unittest.TestCase):
    def test_env_overrides_only_the_default_tag(self):
        with mock.patch.dict(os.environ, {"COMFYFLEET_INSTANCE_IMAGE": "comfyfleet:custom"}):
            self.assertEqual(resolve_instance_image(DEFAULT_IMAGE), "comfyfleet:custom")
            self.assertEqual(resolve_instance_image("other:tag"), "other:tag")
        self.assertEqual(resolve_instance_image(DEFAULT_IMAGE), DEFAULT_IMAGE)


class ManagerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.layout = FleetLayout(root / "home")
        self.docker = FakeDocker()
        self.ctx = ApiContext(
            layout=self.layout,
            docker=self.docker,
            detect_gpus=lambda: [Gpu(0, "GPU0", "8192 MiB")],
            port_in_use=lambda _port: False,
            ui_dir=ROOT / "ui",
            public_host="lan-host.example",
            password=PASSWORD,
            sessions=SessionStore(),
            login_guard=LoginGuard(fail_delay_s=0),
        )
        self.httpd = make_server("127.0.0.1", 0, self.ctx)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _open(self, method, path, data=None, headers=None, auth=True):
        merged = dict(headers or {})
        if auth and "Authorization" not in merged:
            merged["Authorization"] = f"Bearer {PASSWORD}"
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers=merged,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_health_ui_create_start_stop(self):
        status, raw = self._open("GET", "/api/health", auth=False)
        self.assertEqual(status, 200)
        health = json.loads(raw.decode("utf-8"))
        self.assertTrue(health["ok"])
        self.assertEqual(health["auth"], "required")
        self.assertNotIn(PASSWORD, raw.decode("utf-8"))

        status, page = self._open("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"ComfyFleet", page)
        self.assertIn(b"New instance", page)
        self.assertIn(b"/comfyfleet-logo-ships.jpg", page)

        body, content_type = _multipart("Portrait.json", _workflow(), [("gpu", "0")])
        status, raw = self._open(
            "POST",
            "/api/instances",
            data=body,
            headers={"Content-Type": content_type, "Host": "127.0.0.1:9100"},
        )
        self.assertEqual(status, 200, raw)
        created = json.loads(raw.decode("utf-8"))
        self.assertTrue(created["ok"])
        self.assertFalse(created["started"])
        self.assertEqual(created["instance"]["name"], "portrait")
        self.assertEqual(created["instance"]["status"], "created")
        self.assertEqual(created["instance"]["port"], 8188)
        self.assertNotIn("url", created["instance"])
        self.assertTrue((self.layout.files / "portrait" / "default_workflow.json").is_file())

        args = self.docker.containers["portrait"]["args"]
        self.assertIn(DEFAULT_IMAGE, args)
        self.assertEqual(args[args.index("--gpus") + 1], "device=0")
        self.assertEqual(args[args.index("-p") + 1], "8188:8188")
        self.assertIn(f"{self.layout.models}:/opt/ComfyUI/models", args)
        self.assertNotIn("/var/run/docker.sock", " ".join(args))
        self.assertNotIn("build", [call[0] for call in self.docker.calls])

        status, raw = self._open("POST", "/api/instances/portrait/start")
        self.assertEqual(status, 200, raw)
        started = json.loads(raw.decode("utf-8"))
        self.assertEqual(started["instance"]["status"], "running")
        self.assertEqual(started["instance"]["port"], 8188)
        self.assertNotIn("url", started["instance"])
        self.assertNotIn("lan-host.example", raw.decode("utf-8"))
        self.assertIn(("start", "portrait"), self.docker.calls)
        self.assertNotIn("build", [call[0] for call in self.docker.calls])

        status, raw = self._open("POST", "/api/instances/portrait/stop")
        self.assertEqual(status, 200, raw)
        stopped = json.loads(raw.decode("utf-8"))
        self.assertEqual(stopped["instance"]["status"], "exited")
        self.assertNotIn("url", stopped["instance"])
        self.assertIn("portrait", self.docker.containers)

    def test_dispatch_ignores_public_host_for_open(self):
        body, content_type = _multipart("Still.json", _workflow(), [("gpus", "0"), ("start", "true")])
        response = dispatch(
            self.ctx,
            "POST",
            "/api/instances",
            "127.0.0.1:9100",
            body,
            content_type,
            {"Authorization": f"Bearer {PASSWORD}"},
        )
        text = response.body.decode("utf-8")
        payload = json.loads(text)
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["instance"]["port"], 8188)
        self.assertNotIn("url", payload["instance"])
        self.assertNotIn("lan-host.example", text)


if __name__ == "__main__":
    unittest.main()
