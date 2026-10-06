import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfyfleet.control import (
    create_instance,
    delete_instance,
    force_stop_instance,
    format_list,
    list_instances,
    start_instance,
    stop_instance,
    terminal_argv,
    update_instance_launch,
)
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu
from comfyfleet.launch import parse_launch
from comfyfleet.paths import DEFAULT_IMAGE, FleetLayout, image_for_cuda_tag


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

    def kill(self, name):
        self.calls.append(("kill", name))
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


def _workflow(directory: Path, filename: str, marker: str) -> Path:
    path = directory / filename
    path.write_text(
        json.dumps(
            {
                "last_node_id": 0,
                "last_link_id": 0,
                "nodes": [],
                "links": [],
                "extra": {"marker": marker},
                "version": 0.4,
            }
        ),
        encoding="utf-8",
    )
    return path


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "home"
        self.layout = FleetLayout(self.root)
        self.docker = FakeDocker()
        self.sources = Path(self.tmp.name) / "src"
        self.sources.mkdir()
        self.one = [Gpu(0, "GPU0", "8192 MiB")]
        self.two = [Gpu(0, "GPU0", "8192 MiB"), Gpu(1, "GPU1", "8192 MiB")]

    def tearDown(self):
        self.tmp.cleanup()

    def _create(self, filename, *, gpus=None, **kwargs):
        return create_instance(
            _workflow(self.sources, filename, filename),
            layout=self.layout,
            docker=self.docker,
            gpus=gpus or self.one,
            gpu="0" if gpus is None and "gpus_spec" not in kwargs and "interactive" not in kwargs else kwargs.pop("gpu", None),
            port_in_use=kwargs.pop("port_in_use", lambda _port: False),
            **kwargs,
        )

    def test_typed_container_name_wins_over_the_workflow_filename(self):
        result = self._create("Portrait.json", name="My Studio")
        self.assertEqual(result.instance.name, "my_studio")
        self.assertEqual(self.docker.status("my_studio"), "created")
        self.assertIsNone(self.docker.status("portrait"))
        self.assertTrue((self.root / "files" / "my_studio" / "default_workflow.json").is_file())
        self.assertFalse((self.root / "files" / "portrait").exists())

    def test_blank_container_name_uses_the_workflow_filename(self):
        result = self._create("My Flow.json", name="   ")
        self.assertEqual(result.instance.name, "my_flow")
        self.assertEqual(self.docker.status("my_flow"), "created")

    def test_create_without_start_reserves_mounts_and_port(self):
        result = self._create("Portrait.json")
        self.assertFalse(result.started)
        self.assertEqual(result.instance.name, "portrait")
        self.assertEqual(result.instance.port, 8188)
        self.assertEqual(self.docker.status("portrait"), "created")
        self.assertNotIn(("start", "portrait"), self.docker.calls)
        args = self.docker.containers["portrait"]["args"]
        self.assertIn(f"{self.root}/models:/opt/ComfyUI/models", args)
        self.assertIn(f"{self.root}/custom_nodes_portrait:/opt/ComfyUI/custom_nodes", args)
        self.assertIn(f"{self.root}/files/portrait/input:/opt/ComfyUI/input", args)
        self.assertIn(f"{self.root}/files/portrait/output:/opt/ComfyUI/output", args)
        self.assertIn(f"{self.root}/files/portrait/temp:/opt/ComfyUI/temp", args)
        self.assertIn(f"{self.root}/files/portrait:/opt/comfyfleet/instance", args)
        self.assertIn("--restart", args)
        self.assertEqual(args[args.index("--restart") + 1], "no")
        self.assertIn("--gpus", args)
        self.assertEqual(args[args.index("--gpus") + 1], "device=0")
        self.assertEqual(args[args.index("--shm-size") + 1], "8g")
        self.assertLess(args.index("--shm-size"), args.index(result.instance.image))
        self.assertIn("-p", args)
        self.assertEqual(args[args.index("-p") + 1], "8188:8188")
        self.assertTrue((self.root / "files" / "portrait" / "default_workflow.json").is_file())
        self.assertTrue((self.root / "files" / "portrait" / "comfyfleet.json").is_file())
        self.assertTrue((self.root / "models" / "checkpoints").is_dir())
        self.assertEqual(result.instance.cuda_tag, "cu130")
        self.assertEqual(result.instance.image, DEFAULT_IMAGE)
        self.assertIn("comfyfleet.cuda_tag=cu130", args)

    def test_cuda_tag_picker_persists_and_keeps_shm(self):
        created = self._create("Portrait.json", cuda_tag="cu124")
        self.assertEqual(created.instance.cuda_tag, "cu124")
        cu124 = image_for_cuda_tag("cu124")
        self.assertEqual(created.instance.image, cu124)
        args = self.docker.containers["portrait"]["args"]
        self.assertEqual(args[args.index("--shm-size") + 1], "8g")
        self.assertLess(args.index("--shm-size"), args.index(cu124))
        self.assertIn("comfyfleet.cuda_tag=cu124", args)
        stored = json.loads((self.root / "files" / "portrait" / "comfyfleet.json").read_text())
        self.assertEqual(stored["cuda_tag"], "cu124")
        self.assertEqual(stored["image"], cu124)
        text = format_list(list_instances(self.layout, self.docker))
        self.assertIn("cu124", text)

        with self.assertRaises(FleetError) as bad:
            self._create("Other.json", cuda_tag="cu128")
        self.assertIn("cu130 or cu124", str(bad.exception))

        switched = self._create(
            "Portrait.json",
            force=True,
            cuda_tag="cu130",
            port_in_use=lambda _port: False,
        )
        self.assertEqual(switched.instance.cuda_tag, "cu130")
        self.assertEqual(switched.instance.image, DEFAULT_IMAGE)
        replaced = self.docker.containers["portrait"]["args"]
        self.assertEqual(replaced[replaced.index("--shm-size") + 1], "8g")

    def test_force_keeps_a_different_line_and_apply_does_not_swap(self):
        self._create("Portrait.json", cuda_tag="cu124")
        with mock.patch.dict(os.environ, {"COMFYFLEET_INSTANCE_IMAGE": "comfyfleet:cu130", "COMFYFLEET_CUDA_TAG": "cu130"}):
            kept = self._create("Portrait.json", force=True, port_in_use=lambda _port: False)
        self.assertEqual(kept.instance.cuda_tag, "cu124")
        self.assertEqual(kept.instance.image, image_for_cuda_tag("cu124"))

        digest = "ghcr.io/recognizeyourprivilege/comfyfleet-images:cu124@sha256:bbbb"
        with mock.patch.dict(
            os.environ,
            {"COMFYFLEET_INSTANCE_IMAGE": digest, "COMFYFLEET_CUDA_TAG": "cu124"},
        ):
            refreshed = self._create("Portrait.json", force=True, port_in_use=lambda _port: False)
        self.assertEqual(refreshed.instance.image, digest)
        self.assertEqual(refreshed.instance.cuda_tag, "cu124")

        self.docker.start("portrait")
        with mock.patch.dict(
            os.environ,
            {"COMFYFLEET_INSTANCE_IMAGE": "comfyfleet:cu130", "COMFYFLEET_CUDA_TAG": "cu130"},
        ):
            updated = update_instance_launch(
                "portrait",
                parse_launch(vram="lowvram"),
                layout=self.layout,
                docker=self.docker,
                gpus=self.one,
                port_in_use=lambda _port: False,
            )
        self.assertEqual(updated.instance.image, digest)
        self.assertEqual(updated.instance.cuda_tag, "cu124")
        args = self.docker.containers["portrait"]["args"]
        self.assertEqual(args[args.index("--shm-size") + 1], "8g")
        self.assertLess(args.index("--shm-size"), args.index(digest))

    def test_create_ensures_the_instance_image_before_create(self):
        class Pulling(FakeDocker):
            def ensure_image(self, image):
                self.calls.append(("ensure_image", image))

        self.docker = Pulling()
        result = self._create("Portrait.json")
        self.assertEqual(self.docker.calls[0], ("ensure_image", DEFAULT_IMAGE))
        self.assertEqual(self.docker.calls[1][0], "create")
        self.assertEqual(result.instance.image, DEFAULT_IMAGE)

        class Missing(FakeDocker):
            def ensure_image(self, image):
                self.calls.append(("ensure_image", image))
                raise FleetError(f"docker pull failed: {image}")

        self.docker = Missing()
        with self.assertRaises(FleetError) as ctx:
            self._create("Other.json")
        self.assertIn("docker pull failed", str(ctx.exception))
        self.assertEqual([call[0] for call in self.docker.calls], ["ensure_image"])
        self.assertFalse((self.root / "files" / "other" / "comfyfleet.json").is_file())

    def test_second_instance_gets_next_port_and_create_does_not_start(self):
        self._create("Alpha.json")
        second = self._create("Beta.json")
        self.assertEqual(second.instance.port, 8189)
        self.assertEqual(self.docker.running_names(), [])

    def test_collision_is_refused(self):
        self._create("Same.json")
        with self.assertRaises(FleetError):
            self._create("same.json")

    def test_force_replaces_a_stopped_instance(self):
        self._create("Same.json")
        self.docker.containers["same"]["status"] = "exited"
        result = self._create("same.json", force=True)
        self.assertEqual(result.instance.name, "same")
        self.assertIn(("rm", "same"), self.docker.calls)
        self.assertEqual(self.docker.status("same"), "created")

    def test_force_refuses_running_container(self):
        self._create("Same.json")
        start_instance(
            "same",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        with self.assertRaises(FleetError):
            self._create("same.json", force=True)

    def test_create_many_run_few_without_rebuild(self):
        self._create("One.json")
        self._create("Two.json")
        start_instance(
            "one",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        stop_instance("one", layout=self.layout, docker=self.docker)
        start_instance(
            "two",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        kinds = [call[0] for call in self.docker.calls]
        self.assertNotIn("build", kinds)
        self.assertEqual(self.docker.status("one"), "exited")
        self.assertEqual(self.docker.status("two"), "running")
        self.assertIn(("update-restart", "two", "unless-stopped"), self.docker.calls)

    def test_start_keeps_recorded_port_and_moves_when_busy(self):
        self._create("Gamma.json")
        busy = start_instance(
            "gamma",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda port: port == 8188,
        )
        self.assertEqual(busy.instance.port, 8189)
        self.assertIn("-p", self.docker.containers["gamma"]["args"])
        self.assertEqual(
            self.docker.containers["gamma"]["args"][
                self.docker.containers["gamma"]["args"].index("-p") + 1
            ],
            "8189:8188",
        )

    def test_start_skips_occupied_neighbors(self):
        self._create("Neighbor.json")
        moved = start_instance(
            "neighbor",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda port: port in {8188, 8189},
        )
        self.assertEqual(moved.instance.port, 8190)
        args = self.docker.containers["neighbor"]["args"]
        self.assertEqual(args[args.index("-p") + 1], "8190:8188")
        self.assertIn(("rm", "neighbor"), self.docker.calls)
        meta = json.loads((self.root / "files" / "neighbor" / "comfyfleet.json").read_text())
        self.assertEqual(meta["port"], 8190)

    def test_create_probes_host_when_no_callback_is_passed(self):
        with mock.patch(
            "comfyfleet.control.make_port_in_use",
            return_value=lambda port: port in {8188, 8189},
        ) as probe:
            result = create_instance(
                _workflow(self.sources, "Outside.json", "outside"),
                layout=self.layout,
                docker=self.docker,
                gpus=self.one,
                gpu="0",
            )
        probe.assert_called_once_with(self.docker)
        self.assertEqual(result.instance.port, 8190)
        args = self.docker.containers["outside"]["args"]
        self.assertEqual(args[args.index("-p") + 1], "8190:8188")

    def test_start_probes_host_when_no_callback_is_passed(self):
        self._create("Gamma.json")
        with mock.patch(
            "comfyfleet.control.make_port_in_use",
            return_value=lambda port: port in {8188, 8189},
        ) as probe:
            result = start_instance(
                "gamma",
                layout=self.layout,
                docker=self.docker,
                gpus=self.one,
            )
        probe.assert_called_once_with(self.docker)
        self.assertEqual(result.instance.port, 8190)
        args = self.docker.containers["gamma"]["args"]
        self.assertEqual(args[args.index("-p") + 1], "8190:8188")

    def test_list_shows_port(self):
        self._create("Listed.json")
        text = format_list(list_instances(self.layout, self.docker))
        self.assertIn("listed", text)
        self.assertIn("8188", text)
        self.assertIn("http://0.0.0.0:8188", text)

    def test_multi_gpu_flag(self):
        result = create_instance(
            _workflow(self.sources, "Dual.json", "dual"),
            layout=self.layout,
            docker=self.docker,
            gpus=self.two,
            gpus_spec="0,1",
            port_in_use=lambda _port: False,
        )
        self.assertEqual(result.instance.gpus, [0, 1])
        args = self.docker.containers["dual"]["args"]
        self.assertEqual(args[args.index("--gpus") + 1], '"device=0,1"')
        self.assertIn("NVIDIA_VISIBLE_DEVICES=0,1", args)

    def test_multi_gpu_without_flag_does_not_create(self):
        with self.assertRaises(FleetError):
            create_instance(
                _workflow(self.sources, "Dual.json", "dual"),
                layout=self.layout,
                docker=self.docker,
                gpus=self.two,
                interactive=False,
                port_in_use=lambda _port: False,
            )
        self.assertEqual(self.docker.calls, [])

    def test_warns_when_running_more_than_gpu_count(self):
        self._create("One.json")
        self._create("Two.json")
        start_instance(
            "one",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        result = start_instance(
            "two",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        self.assertIsNotNone(result.warning)
        self.assertIn("2", result.warning)

    def test_configured_max_concurrent_blocks(self):
        self._create("One.json")
        self._create("Two.json")
        start_instance(
            "one",
            layout=self.layout,
            docker=self.docker,
            gpus=self.two,
            port_in_use=lambda _port: False,
            max_concurrent=1,
        )
        with self.assertRaises(FleetError):
            start_instance(
                "two",
                layout=self.layout,
                docker=self.docker,
                gpus=self.two,
                port_in_use=lambda _port: False,
                max_concurrent=1,
            )
        self.assertEqual(self.docker.status("two"), "created")

    def test_force_stop_kills_only_a_running_instance(self):
        self._create("Portrait.json")
        start_instance(
            "portrait",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        result = force_stop_instance("portrait", layout=self.layout, docker=self.docker)
        self.assertEqual(result.name, "portrait")
        self.assertIn(("kill", "portrait"), self.docker.calls)
        self.assertNotIn(("stop", "portrait"), self.docker.calls)
        self.assertEqual(self.docker.status("portrait"), "exited")
        self.assertTrue((self.root / "files" / "portrait" / "comfyfleet.json").is_file())

    def test_delete_removes_the_named_container_and_record_only(self):
        self._create("Portrait.json")
        self._create("Other.json")
        start_instance(
            "portrait",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        workflow = self.root / "files" / "portrait" / "default_workflow.json"
        deleted = delete_instance("portrait", layout=self.layout, docker=self.docker)
        self.assertEqual(deleted.name, "portrait")
        self.assertIn(("kill", "portrait"), self.docker.calls)
        self.assertIsNone(self.docker.status("portrait"))
        self.assertFalse((self.root / "files" / "portrait" / "comfyfleet.json").is_file())
        self.assertTrue(workflow.is_file())
        self.assertTrue((self.root / "custom_nodes_portrait").is_dir())
        self.assertEqual(self.docker.status("other"), "created")
        names = [item.name for item, _status in list_instances(self.layout, self.docker)]
        self.assertEqual(names, ["other"])
        with self.assertRaises(FleetError):
            terminal_argv("other", layout=self.layout, docker=self.docker)
        start_instance(
            "other",
            layout=self.layout,
            docker=self.docker,
            gpus=self.one,
            port_in_use=lambda _port: False,
        )
        argv = terminal_argv("other", layout=self.layout, docker=self.docker)
        self.assertEqual(argv, ["docker", "exec", "-it", "other", "/bin/bash"])
        self.assertNotIn("docker.sock", " ".join(argv))

    def test_missing_workflow_argument_fails_in_the_parser(self):
        from comfyfleet.cli import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["create"])

    def test_default_layout_is_under_comfyfleet_home(self):
        layout = FleetLayout()
        root = Path("/home/ComfyFleet")
        self.assertEqual(layout.root, root)
        self.assertEqual(layout.models, root / "models")
        self.assertEqual(layout.wildcards, root / "wildcards")
        self.assertEqual(layout.files, root / "files")
        self.assertEqual(layout.custom_nodes("portrait"), root / "custom_nodes_portrait")
        self.assertEqual(layout.instance_dir("portrait"), root / "files" / "portrait")
        self.assertEqual(layout.input_dir("portrait"), root / "files" / "portrait" / "input")
        self.assertEqual(layout.output_dir("portrait"), root / "files" / "portrait" / "output")
        self.assertEqual(layout.temp_dir("portrait"), root / "files" / "portrait" / "temp")
        owned = (
            layout.models,
            layout.wildcards,
            layout.files,
            layout.custom_nodes("portrait"),
            layout.instance_dir("portrait"),
        )
        for path in owned:
            self.assertIn(
                path.relative_to(root).parts[0],
                {"models", "wildcards", "files", "custom_nodes_portrait"},
            )
            self.assertNotEqual(path.parent, Path("/home"))


if __name__ == "__main__":
    unittest.main()
