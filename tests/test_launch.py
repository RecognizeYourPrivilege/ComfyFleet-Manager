"""Per-instance ComfyUI launch flags: catalog, argv order, and docker create."""

import json
import tempfile
import unittest
from pathlib import Path

from comfyfleet.control import Instance, create_instance, start_instance, update_instance_launch
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu
from comfyfleet.launch import (
    ATTENTION_FLAGS,
    BOOL_FLAGS,
    VRAM_FLAGS,
    LaunchConfig,
    main_argv,
    parse_launch,
    strip_locked_args,
)
from comfyfleet.paths import CONTAINER_PORT, DEFAULT_IMAGE, FleetLayout


ROOT = Path(__file__).resolve().parents[1]


def _mounts(args: list[str]) -> list[str]:
    found = []
    index = 0
    while index < len(args):
        if args[index] == "-v" and index + 1 < len(args):
            found.append(args[index + 1])
        index += 1
    return found


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


def _workflow(directory: Path) -> Path:
    path = directory / "Portrait.json"
    path.write_text(
        json.dumps(
            {
                "last_node_id": 0,
                "last_link_id": 0,
                "nodes": [],
                "links": [],
                "version": 0.4,
            }
        ),
        encoding="utf-8",
    )
    return path


class LaunchParseTests(unittest.TestCase):
    def test_default_adds_nothing(self):
        launch = parse_launch()
        self.assertEqual(launch.argv(), [])

    def test_order_is_vram_attention_flags_values_then_extra(self):
        launch = parse_launch(
            vram="lowvram",
            attention="use-pytorch-cross-attention",
            flags=["disable-smart-memory", "--force-fp16"],
            reserve_vram="1.5",
            preview_method="auto",
            extra_args="--listen 127.0.0.1 --port 9 --quick-test-for-ci",
        )
        self.assertEqual(
            launch.argv(),
            [
                "--lowvram",
                "--use-pytorch-cross-attention",
                "--force-fp16",
                "--disable-smart-memory",
                "--reserve-vram",
                "1.5",
                "--preview-method",
                "auto",
                "--quick-test-for-ci",
            ],
        )
        self.assertNotIn("--listen", launch.argv())
        self.assertNotIn("--port", launch.argv())
        self.assertNotIn("127.0.0.1", launch.extra_args)

    def test_strip_listen_and_port_without_eating_the_next_flag(self):
        self.assertEqual(
            strip_locked_args(["--listen", "--cache-none", "--port=8188", "--mmap-torch-files"]),
            ["--cache-none", "--mmap-torch-files"],
        )
        self.assertEqual(
            strip_locked_args(["--port", "9000", "--listen=10.0.0.2", "plain"]),
            ["plain"],
        )
        launch = parse_launch(extra_args="--listen --quick-test-for-ci")
        self.assertEqual(launch.argv(), ["--quick-test-for-ci"])

    def test_conflicts_and_unknown_flags_are_refused(self):
        with self.assertRaises(FleetError):
            parse_launch(vram="--lowvram", flags=["--cpu"])
        with self.assertRaises(FleetError):
            parse_launch(flags=["--force-fp16", "--force-fp32"])
        with self.assertRaises(FleetError):
            parse_launch(flags=["--cuda-malloc", "disable-cuda-malloc"])
        with self.assertRaises(FleetError):
            parse_launch(vram="--gpu-only")
        with self.assertRaises(FleetError):
            parse_launch(flags=["--enable-cors-header"])
        with self.assertRaises(FleetError):
            parse_launch(extra_args="--lowvram")
        with self.assertRaises(FleetError):
            parse_launch(extra_args="--preview-method auto")

    def test_unlisted_real_flag_stays_in_extra(self):
        launch = parse_launch(extra_args="--quick-test-for-ci --mmap-torch-files")
        self.assertEqual(launch.argv(), ["--quick-test-for-ci", "--mmap-torch-files"])

    def test_cache_flags_are_exclusive(self):
        with self.assertRaises(FleetError):
            parse_launch(flags=["--cache-none", "--cache-classic"])
        launch = parse_launch(vram="--lowvram", flags=["--cache-none", "--disable-dynamic-vram"])
        self.assertEqual(
            launch.argv(),
            ["--lowvram", "--disable-dynamic-vram", "--cache-none"],
        )

    def test_metadata_without_launch_stays_stock(self):
        instance = Instance.from_json(
            {
                "name": "portrait",
                "port": 8188,
                "gpus": [0],
                "image": DEFAULT_IMAGE,
                "workflow_host_path": "/home/ComfyFleet/files/portrait/default_workflow.json",
            },
            Path("comfyfleet.json"),
        )
        self.assertEqual(instance.launch, LaunchConfig())
        self.assertEqual(instance.launch.argv(), [])

    def test_ui_lists_every_catalog_flag(self):
        html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        for flag in VRAM_FLAGS:
            self.assertIn(f'name="vram" value="{flag}"', html)
        for flag in ATTENTION_FLAGS:
            self.assertIn(f'name="attention" value="{flag}"', html)
        for item in BOOL_FLAGS:
            if item.exclusive:
                needle = f'value="{item.flag}" data-exclusive="{item.exclusive}"'
            else:
                needle = f'value="{item.flag}"'
            self.assertIn(needle, html)


class LaunchCreateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "home"
        self.layout = FleetLayout(self.root)
        self.docker = FakeDocker()
        self.sources = Path(self.tmp.name) / "src"
        self.sources.mkdir()
        self.gpus = [Gpu(0, "RTX A2000", "12288 MiB")]

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_appends_flags_after_the_image_and_persists_them(self):
        launch = parse_launch(
            vram="--lowvram",
            flags=["--disable-dynamic-vram", "--disable-smart-memory", "--cache-classic"],
            reserve_vram="1",
            extra_args="--listen 127.0.0.1 --mmap-torch-files",
        )
        result = create_instance(
            _workflow(self.sources),
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            gpu="0",
            port_in_use=lambda _port: False,
            launch=launch,
        )
        self.assertEqual(
            result.instance.launch.argv(),
            [
                "--lowvram",
                "--disable-smart-memory",
                "--disable-dynamic-vram",
                "--cache-classic",
                "--reserve-vram",
                "1",
                "--mmap-torch-files",
            ],
        )
        args = self.docker.containers["portrait"]["args"]
        image_at = args.index(DEFAULT_IMAGE)
        self.assertEqual(args[image_at + 1 :], result.instance.launch.argv())
        self.assertNotIn("--listen", args)
        self.assertNotIn("127.0.0.1", args)
        stored = json.loads((self.root / "files" / "portrait" / "comfyfleet.json").read_text())
        self.assertEqual(stored["launch"]["vram"], "--lowvram")
        self.assertNotIn("argv", stored["launch"])

    def test_stock_create_does_not_append_args(self):
        create_instance(
            _workflow(self.sources),
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            gpu="0",
            port_in_use=lambda _port: False,
        )
        args = self.docker.containers["portrait"]["args"]
        self.assertEqual(args[-1], DEFAULT_IMAGE)
        self.assertEqual(args[args.index("--shm-size") + 1], "8g")
        self.assertLess(args.index("--shm-size"), args.index(DEFAULT_IMAGE))

    def test_update_recreates_same_name_port_and_mounts(self):
        created = create_instance(
            _workflow(self.sources),
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            gpu="0",
            port_in_use=lambda _port: False,
            launch=parse_launch(vram="--lowvram"),
        )
        workflow = (self.layout.files / "portrait" / "default_workflow.json").read_bytes()
        first = self.docker.containers["portrait"]["args"]
        self.docker.start("portrait")
        updated = update_instance_launch(
            "portrait",
            parse_launch(vram="--novram", flags=["--disable-dynamic-vram", "--cache-none"]),
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            port_in_use=lambda _port: False,
        )
        self.assertTrue(updated.started)
        self.assertEqual(updated.instance.name, created.instance.name)
        self.assertEqual(updated.instance.port, created.instance.port)
        self.assertEqual(list(self.docker.containers), ["portrait"])
        second = self.docker.containers["portrait"]["args"]
        self.assertEqual(_mounts(first), _mounts(second))
        self.assertEqual(first[first.index("--name") + 1], second[second.index("--name") + 1])
        self.assertEqual(first[first.index("-p") + 1], second[second.index("-p") + 1])
        self.assertEqual(first[first.index("--gpus") + 1], second[second.index("--gpus") + 1])
        self.assertEqual(second[second.index("--shm-size") + 1], "8g")
        self.assertEqual(
            second[second.index(DEFAULT_IMAGE) + 1 :],
            ["--novram", "--disable-dynamic-vram", "--cache-none"],
        )
        self.assertNotIn("--listen", second)
        self.assertNotIn("--port", second[second.index(DEFAULT_IMAGE) + 1 :])
        self.assertEqual((self.layout.files / "portrait" / "default_workflow.json").read_bytes(), workflow)
        self.assertEqual(self.docker.status("portrait"), "running")

    def test_start_recreate_keeps_saved_flags(self):
        create_instance(
            _workflow(self.sources),
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            gpu="0",
            port_in_use=lambda _port: False,
            launch=parse_launch(vram="novram", attention="use-sage-attention"),
        )
        moved = start_instance(
            "portrait",
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            port_in_use=lambda port: port == 8188,
        )
        self.assertEqual(moved.instance.port, 8189)
        args = self.docker.containers["portrait"]["args"]
        self.assertEqual(
            args[args.index(DEFAULT_IMAGE) + 1 :],
            ["--novram", "--use-sage-attention"],
        )
        self.assertEqual(args[args.index("-p") + 1], "8189:8188")
        # Host port is the Docker publish. main.py still gets the container port.
        self.assertEqual(
            main_argv(moved.instance.launch)[:4],
            ["--listen", "0.0.0.0", "--port", str(CONTAINER_PORT)],
        )
        self.assertNotIn("8189", main_argv(moved.instance.launch))


class FleetLockTests(unittest.TestCase):
    """VRAM presets and extra flags cannot move the locked listen/port pair."""

    def test_vram_presets_are_only_the_three_flags(self):
        self.assertEqual(VRAM_FLAGS, ("--lowvram", "--novram", "--highvram"))
        html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('name="vram" value="" checked', html)
        for flag in VRAM_FLAGS:
            self.assertIn(f'name="vram" value="{flag}"', html)
        self.assertNotIn("--normalvram", html)
        self.assertNotIn("normalvram", html)

    def test_default_preset_adds_no_vram_flag(self):
        self.assertEqual(parse_launch().vram, "")
        self.assertEqual(parse_launch(vram="").argv(), [])
        self.assertEqual(parse_launch(vram="default").argv(), [])
        self.assertEqual(parse_launch(vram="none").argv(), [])
        command = main_argv(parse_launch())
        self.assertEqual(command, ["--listen", "0.0.0.0", "--port", "8188"])
        for flag in (*VRAM_FLAGS, "--normalvram", "--cpu", "--gpu-only"):
            self.assertNotIn(flag, command)

    def test_one_preset_is_appended_after_listen_and_port(self):
        for flag in VRAM_FLAGS:
            command = main_argv(parse_launch(vram=flag, extra_args="--mmap-torch-files"))
            self.assertEqual(
                command,
                ["--listen", "0.0.0.0", "--port", "8188", flag, "--mmap-torch-files"],
            )

    def test_presets_are_mutually_exclusive(self):
        with self.assertRaises(FleetError):
            parse_launch(vram="--lowvram", flags=["--novram"])
        with self.assertRaises(FleetError):
            parse_launch(vram="--highvram", flags=["--lowvram"])
        with self.assertRaises(FleetError):
            parse_launch(vram="--normalvram")

    def test_extra_flags_append_and_cannot_replace_listen_or_port(self):
        launch = parse_launch(
            vram="--lowvram",
            extra_args="--listen 10.1.1.1 --port 9 --listen=127.0.0.1 --port=1 --mmap-torch-files",
        )
        self.assertEqual(launch.argv(), ["--lowvram", "--mmap-torch-files"])
        command = main_argv(launch)
        self.assertEqual(command[:4], ["--listen", "0.0.0.0", "--port", "8188"])
        self.assertEqual(command[-1], "--mmap-torch-files")
        self.assertNotIn("10.1.1.1", command)
        self.assertNotIn("9", command)
        self.assertEqual(command.count("--listen"), 1)
        self.assertEqual(command.count("--port"), 1)

    def test_published_host_port_stays_on_docker_publish(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "home"
        layout = FleetLayout(root)
        sources = Path(tmp.name) / "src"
        sources.mkdir()
        workflow = sources / "Portrait.json"
        workflow.write_text(
            json.dumps({"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "version": 0.4}),
            encoding="utf-8",
        )
        docker = FakeDocker()
        create_instance(
            workflow,
            layout=layout,
            docker=docker,
            gpus=[Gpu(0, "RTX A2000", "12288 MiB")],
            gpu="0",
            port_in_use=lambda port: port == 8188,
            launch=parse_launch(vram="--novram", extra_args="--port 8188 --listen 0.0.0.0 --mmap-torch-files"),
        )
        args = docker.containers["portrait"]["args"]
        self.assertEqual(args[args.index("-p") + 1], "8189:8188")
        after_image = args[args.index(DEFAULT_IMAGE) + 1 :]
        self.assertEqual(after_image, ["--novram", "--mmap-torch-files"])
        self.assertNotIn("--listen", after_image)
        self.assertNotIn("--port", after_image)


if __name__ == "__main__":
    unittest.main()
