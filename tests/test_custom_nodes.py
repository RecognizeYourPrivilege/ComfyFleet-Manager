"""Create-time custom nodes: blank no-op, zip traversal, missing-workflow install."""

import io
import json
import subprocess
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from comfyfleet.cli import build_parser, main
from comfyfleet.control import ActionResult, Instance, create_instance
from comfyfleet.custom_nodes import (
    clone_git_urls,
    extract_custom_nodes_zip,
    extract_custom_nodes_zips,
    is_allowed_git_url,
    parse_node_map,
    plan_missing_installs,
    trusted_manager_install,
)
from comfyfleet.gpu import Gpu
from comfyfleet.launch import LaunchConfig
from comfyfleet.paths import DEFAULT_IMAGE, FleetLayout


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


def _workflow(directory: Path, filename: str, payload: dict | None = None) -> Path:
    path = directory / filename
    body = payload or {
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "version": 0.4,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _pyproject(name: str, display: str = "Pretty Label") -> bytes:
    return (
        "[project]\n"
        f'name = "{name}"\n'
        "\n"
        "[tool.comfy]\n"
        f'DisplayName = "{display}"\n'
    ).encode()


class CustomNodeUnitTests(unittest.TestCase):
    def test_allowed_schemes(self):
        self.assertTrue(is_allowed_git_url("https://github.com/a/b"))
        self.assertTrue(is_allowed_git_url("ssh://git@github.com/a/b.git"))
        self.assertTrue(is_allowed_git_url("git@github.com:a/b.git"))
        self.assertFalse(is_allowed_git_url("http://github.com/a/b"))
        self.assertFalse(is_allowed_git_url("file:///tmp/repo"))
        self.assertFalse(is_allowed_git_url("git://github.com/a/b"))
        self.assertFalse(is_allowed_git_url(""))

    def test_blank_urls_do_not_invoke_git(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            calls = []

            def run(*_args, **_kwargs):
                calls.append(True)
                raise AssertionError("git should not run")

            warnings, cloned = clone_git_urls(["", "  "], dest, run=run)
            self.assertEqual(warnings, [])
            self.assertEqual(cloned, [])
            self.assertEqual(calls, [])

    def test_clone_uses_argv_and_rejects_bad_scheme(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            seen = []

            def run(argv, **_kwargs):
                seen.append(list(argv))
                target = Path(argv[-1])
                target.mkdir(parents=True)
                (target / "pack.py").write_text("ok\n", encoding="utf-8")
                return subprocess.CompletedProcess(argv, 0, "", "")

            warnings, cloned = clone_git_urls(
                [
                    "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
                    "file:///tmp/evil",
                    "ssh://git@github.com/example/Other-Node.git",
                ],
                dest,
                run=run,
                timeout=5,
            )
            self.assertEqual(cloned, ["ComfyUI-Impact-Pack", "Other-Node"])
            self.assertTrue(any("file:///tmp/evil" in item for item in warnings))
            self.assertTrue((dest / "ComfyUI-Impact-Pack" / "pack.py").is_file())
            self.assertEqual(seen[0][:4], ["git", "clone", "--depth", "1"])
            self.assertEqual(seen[0][4], "--")
            self.assertTrue(seen[0][5].startswith("https://"))

    def test_path_traversal_zip_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "custom_nodes_portrait"
            outside = root / "outside.txt"
            warnings, extracted = extract_custom_nodes_zip(
                _zip(
                    {
                        "../outside.txt": b"pwned",
                        "/tmp/abs.txt": b"pwned",
                        "ok/node.py": b"print(1)\n",
                    }
                ),
                dest,
            )
            self.assertFalse(extracted)
            self.assertTrue(warnings)
            self.assertIn("rejected", warnings[0])
            self.assertFalse(outside.exists())
            self.assertFalse((dest / "ok" / "node.py").exists())
            self.assertFalse((dest / "outside.txt").exists())

    def test_backslash_traversal_is_rejected(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(
                _zip({"..\\..\\outside.txt": b"pwned"}),
                dest,
            )
            self.assertFalse(extracted)
            self.assertIn("path traversal", warnings[0])
            self.assertEqual(list(dest.rglob("*")), [])

    def test_safe_zip_extracts_into_custom_nodes(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(
                _zip({"MyNode/__init__.py": b"NODE_CLASS_MAPPINGS = {'MyNode': None}\n"}),
                dest,
            )
            self.assertEqual(warnings, [])
            self.assertTrue(extracted)
            self.assertIn("MyNode", (dest / "MyNode" / "__init__.py").read_text(encoding="utf-8"))

    def test_blank_name_uses_pyproject_not_display_name_or_json(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(
                _zip(
                    {
                        "ComfyUI-Impact-Pack-main/__init__.py": b"zipped\n",
                        "ComfyUI-Impact-Pack-main/pyproject.toml": _pyproject(
                            "comfyui-impact-pack"
                        ),
                        "ComfyUI-Impact-Pack-main/node_list.json": b'{"name": "not-the-pack"}\n',
                    }
                ),
                dest,
                directory="  ",
            )
            self.assertEqual(warnings, [])
            self.assertTrue(extracted)
            packed = dest / "comfyui-impact-pack" / "__init__.py"
            self.assertEqual(packed.read_text(encoding="utf-8"), "zipped\n")
            self.assertFalse((dest / "ComfyUI-Impact-Pack-main").exists())
            self.assertFalse((dest / "Pretty Label").exists())
            self.assertFalse((dest / "not-the-pack").exists())

    def test_typed_name_wins_over_pyproject(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(
                _zip(
                    {
                        "Repo-main/__init__.py": b"typed\n",
                        "Repo-main/pyproject.toml": _pyproject("from-toml"),
                    }
                ),
                dest,
                directory="ComfyUI-Impact-Pack",
            )
            self.assertEqual(warnings, [])
            self.assertTrue(extracted)
            self.assertEqual(
                (dest / "ComfyUI-Impact-Pack" / "__init__.py").read_text(encoding="utf-8"),
                "typed\n",
            )
            self.assertFalse((dest / "from-toml").exists())

    def test_flat_registry_zip_uses_project_name(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(
                _zip(
                    {
                        "__init__.py": b"flat\n",
                        "pyproject.toml": _pyproject("flat-pack"),
                    }
                ),
                dest,
            )
            self.assertEqual(warnings, [])
            self.assertTrue(extracted)
            self.assertEqual(
                (dest / "flat-pack" / "__init__.py").read_text(encoding="utf-8"),
                "flat\n",
            )
            self.assertFalse((dest / "__init__.py").exists())

    def test_invalid_name_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(
                _zip({"Ok/a.py": b"x=1\n", "Ok/pyproject.toml": _pyproject("ok-pack")}),
                dest,
                directory="../etc",
            )
            self.assertFalse(extracted)
            self.assertIn("name is invalid", warnings[0])
            self.assertFalse(dest.exists())

    def test_several_zips_each_keep_their_own_name(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "nodes"
            warnings, extracted = extract_custom_nodes_zips(
                [
                    _zip(
                        {
                            "A-main/__init__.py": b"aaa\n",
                            "A-main/pyproject.toml": _pyproject("pack-a"),
                        }
                    ),
                    _zip({"../outside.txt": b"pwned", "Ok/a.py": b"x"}),
                    _zip(
                        {
                            "C-main/__init__.py": b"ccc\n",
                            "C-main/pyproject.toml": _pyproject("pack-c"),
                        }
                    ),
                ],
                dest,
                names=["", " ", "TypedPack"],
                labels=["a.zip", "bad.zip", "c.zip"],
            )
            self.assertTrue(extracted)
            self.assertTrue(any(item.startswith("bad.zip:") and "rejected" in item for item in warnings))
            self.assertEqual((dest / "pack-a" / "__init__.py").read_text(encoding="utf-8"), "aaa\n")
            self.assertEqual((dest / "TypedPack" / "__init__.py").read_text(encoding="utf-8"), "ccc\n")
            self.assertFalse((dest / "pack-c").exists())
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse((dest / "Ok").exists())

    def test_malformed_and_absent_zip(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            warnings, extracted = extract_custom_nodes_zip(b"not a zip", dest)
            self.assertFalse(extracted)
            self.assertIn("not a valid zip", warnings[0])
            warnings, extracted = extract_custom_nodes_zip(None, dest)
            self.assertEqual((warnings, extracted), ([], False))
            warnings, extracted = extract_custom_nodes_zip(b"", dest)
            self.assertFalse(extracted)
            self.assertTrue(warnings)

    def test_missing_install_is_only_referenced_nodes(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            dest.mkdir()
            workflow = {
                "nodes": [
                    {"type": "KSampler"},
                    {"type": "ExampleNode", "properties": {"aux_id": "example/ExamplePack"}},
                    {"type": "Already", "properties": {"aux_id": "someone/AlreadyThere"}},
                ]
            }
            (dest / "AlreadyThere").mkdir()
            urls, warnings = plan_missing_installs(workflow, dest, node_map=None)
            self.assertEqual(warnings, [])
            self.assertEqual(urls, ["https://github.com/example/ExamplePack"])
            registry = {
                "https://github.com/example/ExamplePack": [["ExampleNode"], {"title_aux": "Example"}],
                "https://github.com/example/unrelated": [["UnrelatedNode"], {}],
            }
            urls, warnings = plan_missing_installs(workflow, dest, parse_node_map(registry))
            self.assertEqual(warnings, [])
            self.assertEqual(urls, ["https://github.com/example/ExamplePack"])

    def test_present_class_and_baked_pack_are_not_installed(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nodes"
            pack = dest / "Local"
            pack.mkdir(parents=True)
            (pack / "nodes.py").write_text(
                "NODE_CLASS_MAPPINGS = {'ImpactSwitch': None}\n",
                encoding="utf-8",
            )
            workflow = {
                "nodes": [
                    {
                        "type": "ImpactSwitch",
                        "properties": {"aux_id": "ltdrdata/ComfyUI-Impact-Pack"},
                    },
                    {
                        "type": "ResNode",
                        "properties": {"aux_id": "ClownsharkBatwing/RES4LYF"},
                    },
                ]
            }
            urls, warnings = plan_missing_installs(workflow, dest, None)
            self.assertEqual(urls, [])
            self.assertEqual(warnings, [])

    def test_trusted_install_posts_only_given_urls(self):
        posted = []

        class Exec:
            def exec(self, name, command, *, timeout):
                posted.append((name, list(command), timeout))
                return subprocess.CompletedProcess(command, 0, "", "")

        warnings, installed = trusted_manager_install(
            [
                "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
                "https://github.com/example/Other",
            ],
            docker=Exec(),
            name="portrait",
            ready_timeout=1,
            install_timeout=5,
            sleep=lambda _seconds: None,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(
            installed,
            [
                "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
                "https://github.com/example/Other",
            ],
        )
        install_cmds = [command for _name, command, _timeout in posted if len(command) > 3 and command[3] == "install"]
        self.assertEqual(len(install_cmds), 2)
        blob = "\n".join(" ".join(command) for _name, command, _timeout in posted)
        self.assertIn("/customnode/install/git_url", blob)
        self.assertNotIn("customnode/getlist", blob)
        self.assertNotIn("manager/queue", blob)

    def test_api_prompt_aux_id_resolves_to_one_github_url(self):
        with TemporaryDirectory() as tmp:
            urls, warnings = plan_missing_installs(
                {
                    "3": {
                        "class_type": "ImpactSwitch",
                        "inputs": {},
                        "properties": {"aux_id": "example/ExamplePack"},
                    },
                    "4": {"class_type": "KSampler", "inputs": {}},
                },
                Path(tmp),
                None,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(urls, ["https://github.com/example/ExamplePack"])

    def test_trusted_install_without_exec_warns_immediately(self):
        warnings, installed = trusted_manager_install(
            ["https://github.com/a/b"],
            docker=object(),
            name="portrait",
            ready_timeout=30,
        )
        self.assertEqual(installed, [])
        self.assertIn("cannot exec", warnings[0])


class CreateCustomNodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "home"
        self.layout = FleetLayout(self.root)
        self.docker = FakeDocker()
        self.sources = Path(self.tmp.name) / "src"
        self.sources.mkdir()
        self.gpus = [Gpu(0, "GPU0", "8192 MiB")]

    def tearDown(self):
        self.tmp.cleanup()

    def _create(self, filename, payload=None, **kwargs):
        return create_instance(
            _workflow(self.sources, filename, payload),
            layout=self.layout,
            docker=self.docker,
            gpus=self.gpus,
            gpu="0",
            port_in_use=lambda _port: False,
            **kwargs,
        )

    def test_blank_fields_are_a_noop_and_default_install_does_not_call_manager(self):
        def git_run(*_args, **_kwargs):
            raise AssertionError("git should not run")

        def installer(*_args, **_kwargs):
            raise AssertionError("installer should not run")

        with mock.patch("comfyfleet.control.trusted_manager_install", side_effect=AssertionError("default installer")):
            result = self._create(
                "Portrait.json",
                custom_node_git_urls=["", " "],
                custom_nodes_zip=None,
                git_run=git_run,
                node_installer=installer,
            )
        self.assertEqual(result.warnings, [])
        self.assertFalse(result.started)
        self.assertEqual(self.docker.status("portrait"), "created")
        self.assertEqual(result.instance.launch.argv(), [])
        import inspect

        self.assertIs(
            inspect.signature(create_instance).parameters["install_missing_from_workflow"].default,
            True,
        )

    def test_bad_clone_warns_and_create_still_succeeds(self):
        def git_run(argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 1, "", "boom")

        result = self._create(
            "Portrait.json",
            custom_node_git_urls=["https://github.com/a/Missing", "http://insecure/a/b"],
            git_run=git_run,
            install_missing_from_workflow=False,
        )
        self.assertFalse(result.started)
        self.assertEqual(self.docker.status("portrait"), "created")
        self.assertTrue(any("boom" in item for item in result.warnings))
        self.assertTrue(any("http://insecure/a/b" in item for item in result.warnings))
        self.assertFalse((self.layout.custom_nodes("portrait") / "Missing").exists())

    def test_zip_traversal_does_not_fail_create(self):
        result = self._create(
            "Portrait.json",
            custom_nodes_zip=_zip({"../../outside.txt": b"pwned", "Ok/a.py": b"x=1\n"}),
            install_missing_from_workflow=False,
        )
        self.assertEqual(self.docker.status("portrait"), "created")
        self.assertTrue(any("rejected" in item for item in result.warnings))
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertFalse((self.layout.custom_nodes("portrait") / "Ok" / "a.py").exists())

    def test_safe_zip_and_git_land_in_the_volume(self):
        def git_run(argv, **_kwargs):
            target = Path(argv[-1])
            target.mkdir(parents=True)
            (target / "__init__.py").write_text("cloned\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        result = self._create(
            "Portrait.json",
            custom_node_git_urls=["git@github.com:example/FromGit.git"],
            custom_nodes_zip=_zip({"FromZip/__init__.py": b"zipped\n"}),
            git_run=git_run,
            install_missing_from_workflow=False,
        )
        self.assertEqual(result.warnings, [])
        nodes = self.layout.custom_nodes("portrait")
        self.assertEqual((nodes / "FromGit" / "__init__.py").read_text(encoding="utf-8"), "cloned\n")
        self.assertEqual((nodes / "FromZip" / "__init__.py").read_text(encoding="utf-8"), "zipped\n")
        self.assertNotIn(("start", "portrait"), self.docker.calls)

    def test_create_names_each_zip(self):
        result = self._create(
            "Portrait.json",
            custom_nodes_zip=[
                _zip(
                    {
                        "A-main/__init__.py": b"aaa\n",
                        "A-main/pyproject.toml": _pyproject("pack-a"),
                    }
                ),
                _zip({"B-main/__init__.py": b"bbb\n", "B-main/pyproject.toml": _pyproject("pack-b")}),
            ],
            custom_nodes_zip_names=["", "TypedB"],
            install_missing_from_workflow=False,
        )
        self.assertEqual(result.warnings, [])
        nodes = self.layout.custom_nodes("portrait")
        self.assertEqual((nodes / "pack-a" / "__init__.py").read_text(encoding="utf-8"), "aaa\n")
        self.assertEqual((nodes / "TypedB" / "__init__.py").read_text(encoding="utf-8"), "bbb\n")
        self.assertTrue(str(nodes).startswith(str(self.layout.root)))
        self.assertIn("/custom_nodes_portrait", str(nodes).replace("\\", "/"))

    def test_default_install_missing_posts_only_that_workflow_node(self):
        seen = []

        def installer(urls, *, docker, name):
            seen.append((name, list(urls)))
            return [], list(urls)

        workflow = {
            "nodes": [
                {"type": "KSampler"},
                {"type": "ExampleNode", "properties": {"aux_id": "example/ExamplePack"}},
            ]
        }
        node_map = {
            "UnrelatedNode": "https://github.com/example/unrelated",
        }
        result = self._create(
            "Portrait.json",
            workflow,
            node_installer=installer,
            node_map=node_map,
        )
        self.assertEqual(seen, [("portrait", ["https://github.com/example/ExamplePack"])])
        self.assertFalse(result.started)
        self.assertEqual(self.docker.status("portrait"), "exited")
        self.assertIn(("start", "portrait"), self.docker.calls)
        self.assertIn(("stop", "portrait"), self.docker.calls)

    def test_start_true_stays_running_after_a_missing_node_install(self):
        def installer(urls, *, docker, name):
            return [], list(urls)

        result = self._create(
            "Portrait.json",
            {
                "nodes": [
                    {
                        "type": "ImpactSwitch",
                        "properties": {"aux_id": "ltdrdata/ComfyUI-Impact-Pack"},
                    }
                ]
            },
            start=True,
            node_installer=installer,
        )
        self.assertTrue(result.started)
        self.assertEqual(self.docker.status("portrait"), "running")
        self.assertEqual(result.warnings, [])

    def test_install_missing_false_skips_even_when_the_workflow_names_a_pack(self):
        def installer(*_args, **_kwargs):
            raise AssertionError("installer should not run")

        workflow = {
            "nodes": [
                {"type": "ImpactSwitch", "properties": {"aux_id": "ltdrdata/ComfyUI-Impact-Pack"}},
            ]
        }
        result = self._create(
            "Portrait.json",
            workflow,
            install_missing_from_workflow=False,
            node_installer=installer,
        )
        self.assertEqual(result.warnings, [])
        self.assertNotIn(("start", "portrait"), self.docker.calls)
        self.assertEqual(self.docker.status("portrait"), "created")

    def test_comfy_extra_args_reach_the_container(self):
        from comfyfleet.launch import parse_launch

        result = self._create(
            "Portrait.json",
            launch=parse_launch(extra_args="--mmap-torch-files --fast"),
            install_missing_from_workflow=False,
        )
        args = self.docker.containers["portrait"]["args"]
        image_at = args.index(DEFAULT_IMAGE)
        self.assertEqual(args[image_at + 1 :], ["--mmap-torch-files", "--fast"])


class CliForwardTests(unittest.TestCase):
    def test_parser_defaults_install_missing_true(self):
        args = build_parser().parse_args(["create", "--workflow", "flow.json"])
        self.assertTrue(args.install_missing_from_workflow)
        self.assertIsNone(args.custom_node_git_urls)
        self.assertIsNone(args.custom_nodes_zip)
        self.assertIsNone(args.name)
        skipped = build_parser().parse_args(
            [
                "create",
                "--workflow",
                "flow.json",
                "--no-install-missing-from-workflow",
                "--custom-node-git-url",
                "https://github.com/a/b",
                "--comfy-extra-args=--mmap-torch-files",
            ]
        )
        self.assertFalse(skipped.install_missing_from_workflow)
        self.assertEqual(skipped.custom_node_git_urls, ["https://github.com/a/b"])

    def test_cli_forwards_fields(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "Portrait.json"
            workflow.write_text("{}\n", encoding="utf-8")
            archive = root / "nodes.zip"
            archive.write_bytes(_zip({"A/a.py": b"x"}))
            instance = Instance(
                name="portrait",
                port=8188,
                gpus=[0],
                image="img",
                workflow_host_path=str(workflow),
                workflow_source=str(workflow),
                created_at="now",
                launch=LaunchConfig(),
            )
            with (
                mock.patch("comfyfleet.cli.detect_gpus", return_value=[Gpu(0, "GPU0", "1")]),
                mock.patch("comfyfleet.cli.DockerCLI"),
                mock.patch(
                    "comfyfleet.cli.create_instance",
                    return_value=ActionResult(instance=instance, started=False),
                ) as create,
            ):
                code = main(
                    [
                        "create",
                        "--workflow",
                        str(workflow),
                        "--gpu",
                        "0",
                        "--custom-node-git-url",
                        "https://github.com/a/b",
                        "--custom-nodes-zip",
                        str(archive),
                        "--name",
                        "My Studio",
                        "--comfy-extra-args=--mmap-torch-files",
                    ]
                )
            self.assertEqual(code, 0)
            kwargs = create.call_args.kwargs
            self.assertEqual(kwargs["custom_node_git_urls"], ["https://github.com/a/b"])
            self.assertEqual(kwargs["custom_nodes_zip"], archive.read_bytes())
            self.assertEqual(kwargs["name"], "My Studio")
            self.assertIsNone(kwargs["custom_nodes_zip_names"])
            self.assertEqual(kwargs["custom_nodes_zip_labels"], ["nodes.zip"])
            self.assertTrue(kwargs["install_missing_from_workflow"])
            self.assertIn("--mmap-torch-files", kwargs["launch"].argv())

    def test_cli_forwards_several_zips_and_names(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = root / "Portrait.json"
            workflow.write_text("{}\n", encoding="utf-8")
            first = root / "a.zip"
            second = root / "b.zip"
            first.write_bytes(_zip({"A/a.py": b"x"}))
            second.write_bytes(_zip({"B/b.py": b"y"}))
            instance = Instance(
                name="portrait",
                port=8188,
                gpus=[0],
                image="img",
                workflow_host_path=str(workflow),
                workflow_source=str(workflow),
                created_at="now",
                launch=LaunchConfig(),
            )
            with (
                mock.patch("comfyfleet.cli.detect_gpus", return_value=[Gpu(0, "GPU0", "1")]),
                mock.patch("comfyfleet.cli.DockerCLI"),
                mock.patch(
                    "comfyfleet.cli.create_instance",
                    return_value=ActionResult(instance=instance, started=False),
                ) as create,
            ):
                code = main(
                    [
                        "create",
                        "--workflow",
                        str(workflow),
                        "--gpu",
                        "0",
                        "--custom-nodes-zip",
                        str(first),
                        "--custom-nodes-zip-name",
                        "",
                        "--custom-nodes-zip",
                        str(second),
                        "--custom-nodes-zip-name",
                        "TypedB",
                    ]
                )
            self.assertEqual(code, 0)
            kwargs = create.call_args.kwargs
            self.assertEqual(kwargs["custom_nodes_zip"], [first.read_bytes(), second.read_bytes()])
            self.assertEqual(kwargs["custom_nodes_zip_names"], ["", "TypedB"])
            self.assertEqual(kwargs["custom_nodes_zip_labels"], ["a.zip", "b.zip"])


if __name__ == "__main__":
    unittest.main()
