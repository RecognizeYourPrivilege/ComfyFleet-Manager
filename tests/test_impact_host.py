"""Wildcards bind, fix-owner allowlist, and prune exclusion."""

import inspect
import json
import os
import stat
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from comfyfleet.auth import LoginGuard, SessionStore
from comfyfleet.cli import build_parser
from comfyfleet.control import create_instance
from comfyfleet.docker import DockerCLI
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu
from comfyfleet.http_api import ApiContext, make_server
from comfyfleet.ownership import (
    BAKED_CUSTOM_NODES,
    DEFAULT_GROUP_NAME,
    DEFAULT_OWNER_NAME,
    assert_allowlisted,
    chown_inode,
    chown_new_directory,
    create_new_host_dirs,
    ensure_wildcards_dir,
    fix_owner,
    resolve_owner_ids,
)
from comfyfleet.paths import MODEL_SUBDIRS, WILDCARDS_CONTAINER, FleetLayout
from comfyfleet.prune import (
    ContainerRecord,
    is_managed,
    parse_ps,
    prune_dangling_containers,
    select_prune_targets,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = "429d0159ad429e64d2b3916e6e7be9c22d025c3c"
SUB = "50c7b71a6a224734cc9b21963c6d1926816a97f1"
PASSWORD = "test-password"


def _workflow(directory: Path, filename: str) -> Path:
    path = directory / filename
    path.write_text(
        json.dumps(
            {
                "last_node_id": 0,
                "last_link_id": 0,
                "nodes": [],
                "links": [],
                "extra": {},
                "version": 0.4,
            }
        ),
        encoding="utf-8",
    )
    return path


class _Docker:
    def __init__(self):
        self.containers = {}
        self.calls = []

    def create(self, args):
        self.calls.append(("create", list(args)))
        name = args[args.index("--name") + 1]
        self.containers[name] = {"status": "created", "args": list(args)}

    def start(self, name):
        self.containers[name]["status"] = "running"

    def stop(self, name):
        self.containers[name]["status"] = "exited"

    def remove(self, name):
        self.containers.pop(name, None)

    def update_restart(self, name, policy):
        return None

    def status(self, name):
        item = self.containers.get(name)
        return None if item is None else item["status"]

    def running_names(self):
        return [name for name, item in self.containers.items() if item["status"] == "running"]


class WildcardsBindTests(unittest.TestCase):
    def test_create_binds_wildcards_and_chowns_only_when_it_creates_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "home"
            layout = FleetLayout(root)
            sources = Path(tmp) / "src"
            sources.mkdir()
            docker = _Docker()
            created = []

            def spy(paths):
                created.extend(Path(path) for path in paths)

            with mock.patch("comfyfleet.ownership.chown_new_directories", spy):
                create_instance(
                    _workflow(sources, "Portrait.json"),
                    layout=layout,
                    docker=docker,
                    gpus=[Gpu(0, "GPU0", "8192 MiB")],
                    gpu="0",
                    port_in_use=lambda _port: False,
                )
            args = docker.containers["portrait"]["args"]
            self.assertIn(f"{root}/wildcards:{WILDCARDS_CONTAINER}", args)
            self.assertEqual(WILDCARDS_CONTAINER, "/home/wildcards")
            self.assertTrue((root / "wildcards").is_dir())
            self.assertTrue((root / "models" / "sams").is_dir())
            self.assertEqual(
                created,
                [root / "wildcards", root / "models"]
                + [root / "models" / sub for sub in MODEL_SUBDIRS]
                + [
                    root / "custom_nodes_portrait",
                    root / "files",
                    root / "files" / "portrait",
                    root / "files" / "portrait" / "input",
                    root / "files" / "portrait" / "output",
                    root / "files" / "portrait" / "temp",
                ],
            )
            self.assertNotIn(root, created)

            marker = root / "wildcards" / "kept.txt"
            marker.write_text("leave-me", encoding="utf-8")
            again = []
            with mock.patch(
                "comfyfleet.ownership.chown_new_directories",
                lambda paths: again.extend(Path(path) for path in paths),
            ):
                create_instance(
                    _workflow(sources, "Other.json"),
                    layout=layout,
                    docker=docker,
                    gpus=[Gpu(0, "GPU0", "8192 MiB")],
                    gpu="0",
                    port_in_use=lambda _port: False,
                )
            self.assertEqual(
                again,
                [
                    root / "custom_nodes_other",
                    root / "files" / "other",
                    root / "files" / "other" / "input",
                    root / "files" / "other" / "output",
                    root / "files" / "other" / "temp",
                ],
            )
            self.assertNotIn(root / "wildcards", again)
            self.assertNotIn(root / "models", again)
            self.assertNotIn(root / "custom_nodes_portrait", again)
            self.assertEqual(marker.read_text(encoding="utf-8"), "leave-me")
            other = docker.containers["other"]["args"]
            self.assertIn(f"{root}/wildcards:{WILDCARDS_CONTAINER}", other)

    def test_new_custom_nodes_directory_is_owned_by_comfyuser(self):
        self.assertEqual(DEFAULT_OWNER_NAME, "comfyuser")
        self.assertEqual(DEFAULT_GROUP_NAME, "comfyuser")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "home"
            layout = FleetLayout(root)
            sources = Path(tmp) / "src"
            sources.mkdir()
            docker = _Docker()
            chowned = []

            def spy(path, uid, gid, follow_symlinks=True):
                chowned.append((Path(path), uid, gid, follow_symlinks))

            with mock.patch.dict(os.environ, {"COMFYFLEET_MANAGER": "1"}):
                with mock.patch(
                    "comfyfleet.ownership.resolve_owner_ids",
                    return_value=(4242, 4243),
                ) as resolve:
                    with mock.patch("comfyfleet.ownership.os.chown", spy):
                        create_instance(
                            _workflow(sources, "Portrait.json"),
                            layout=layout,
                            docker=docker,
                            gpus=[Gpu(0, "GPU0", "8192 MiB")],
                            gpu="0",
                            port_in_use=lambda _port: False,
                        )
            nodes = root / "custom_nodes_portrait"
            self.assertTrue(nodes.is_dir())
            self.assertIn((nodes, 4242, 4243, False), chowned)
            self.assertNotIn(root, [path for path, _uid, _gid, _follow in chowned])
            self.assertTrue(nodes.stat().st_mode & stat.S_IWUSR)
            for call in resolve.call_args_list:
                self.assertEqual(call.args, (DEFAULT_OWNER_NAME, DEFAULT_GROUP_NAME))

            kept = nodes / "kept.txt"
            kept.write_text("leave-me", encoding="utf-8")
            chowned.clear()
            with mock.patch.dict(os.environ, {"COMFYFLEET_MANAGER": "1"}):
                with mock.patch(
                    "comfyfleet.ownership.resolve_owner_ids",
                    return_value=(4242, 4243),
                ):
                    with mock.patch("comfyfleet.ownership.os.chown", spy):
                        create_instance(
                            _workflow(sources, "Other.json"),
                            layout=layout,
                            docker=docker,
                            gpus=[Gpu(0, "GPU0", "8192 MiB")],
                            gpu="0",
                            port_in_use=lambda _port: False,
                        )
            self.assertNotIn(nodes, [path for path, _uid, _gid, _follow in chowned])
            self.assertEqual(kept.read_text(encoding="utf-8"), "leave-me")
            other = root / "custom_nodes_other"
            self.assertIn((other, 4242, 4243, False), chowned)
            self.assertTrue(other.is_dir())
            self.assertTrue(other.stat().st_mode & stat.S_IWUSR)

    def test_create_refuses_a_host_directory_outside_the_storage_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "home"
            root.mkdir()
            outside = Path(tmp) / "custom_nodes_portrait"
            with self.assertRaises(FleetError) as ctx:
                create_new_host_dirs(outside, root)
            self.assertIn(str(root), str(ctx.exception))
            self.assertFalse(outside.exists())

    def test_manager_process_refuses_a_missing_comfyuser_on_the_one_shot_chown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wildcards"
            path.mkdir()
            with mock.patch.dict(os.environ, {"COMFYFLEET_MANAGER": "1"}):
                with mock.patch(
                    "comfyfleet.ownership.resolve_owner_ids",
                    side_effect=FleetError("cannot resolve host user comfyuser by name"),
                ):
                    with self.assertRaises(FleetError) as ctx:
                        chown_new_directory(path)
            self.assertIn("comfyuser", str(ctx.exception))
            with mock.patch.dict(os.environ, {"COMFYFLEET_MANAGER": ""}, clear=False):
                os.environ.pop("COMFYFLEET_MANAGER", None)
                with mock.patch(
                    "comfyfleet.ownership.resolve_owner_ids",
                    side_effect=FleetError("cannot resolve host user comfyuser by name"),
                ):
                    chown_new_directory(path)

    def test_existing_directory_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = FleetLayout(Path(tmp) / "home")
            layout.wildcards.mkdir(parents=True)
            (layout.wildcards / "a.txt").write_text("x", encoding="utf-8")
            self.assertFalse(ensure_wildcards_dir(layout))
            self.assertEqual((layout.wildcards / "a.txt").read_text(encoding="utf-8"), "x")


class FixOwnerTests(unittest.TestCase):
    def test_cli_has_no_path_argument(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["fix-owner", "/etc"])
        args = parser.parse_args(["fix-owner"])
        self.assertEqual(args.command, "fix-owner")
        self.assertIsNone(args.user)
        self.assertIsNone(args.group)
        chosen = parser.parse_args(["fix-owner", "--user", "alice", "--group", "render"])
        self.assertEqual(chosen.user, "alice")
        self.assertEqual(chosen.group, "render")
        parameters = inspect.signature(fix_owner).parameters
        self.assertNotIn("path", parameters)
        self.assertIn("user", parameters)
        self.assertIn("group", parameters)

    def test_allowlist_refuses_escapes_and_chowns_only_listed_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "home"
            layout = FleetLayout(root)
            layout.wildcards.mkdir(parents=True)
            (layout.wildcards / "nested.txt").write_text("w", encoding="utf-8")
            layout.models.mkdir()
            (layout.root / "custom_nodes_portrait").mkdir()
            (layout.root / "custom_nodes_portrait" / "node.py").write_text("n", encoding="utf-8")
            layout.files.mkdir()
            (layout.root / "secret").mkdir()
            (layout.root / "custom_nodes").mkdir()
            outside = Path(tmp) / "etc"
            outside.mkdir()
            escape = layout.wildcards / "escape"
            escape.symlink_to(outside)

            with self.assertRaises(FleetError):
                assert_allowlisted(Path("/etc"), layout)
            with self.assertRaises(FleetError):
                assert_allowlisted(layout.root / ".." / "etc", layout)
            with self.assertRaises(FleetError):
                assert_allowlisted(layout.wildcards / ".." / ".." / "etc", layout)
            with self.assertRaises(FleetError):
                assert_allowlisted(layout.root / "secret", layout)
            with self.assertRaises(FleetError):
                assert_allowlisted(layout.root / "custom_nodes", layout)
            with self.assertRaises(FleetError):
                assert_allowlisted(escape, layout)
            assert_allowlisted(layout.wildcards / "nested.txt", layout)
            assert_allowlisted(layout.root / "custom_nodes_portrait", layout)

            chowned = []

            def spy(path, uid, gid):
                chowned.append(Path(path))
                self.assertEqual((uid, gid), (7, 8))

            with self.assertRaises(FleetError):
                fix_owner(layout, resolve=lambda: (7, 8), chown=spy)
            self.assertNotIn(outside, chowned)
            self.assertTrue(all(item != outside and outside not in item.parents for item in chowned))

            escape.unlink()
            chowned.clear()
            result = fix_owner(layout, resolve=lambda: (7, 8), chown=spy)
            names = {path.name for path in chowned}
            self.assertIn("wildcards", names)
            self.assertIn("nested.txt", names)
            self.assertIn("models", names)
            self.assertIn("custom_nodes_portrait", names)
            self.assertIn("node.py", names)
            self.assertIn("files", names)
            self.assertNotIn("secret", names)
            self.assertNotIn("custom_nodes", names)
            self.assertEqual(result.user, "comfyuser")
            self.assertEqual(result.group, "comfyuser")
            self.assertEqual(result.uid, 7)
            self.assertTrue(any(path.endswith("/wildcards") or path.endswith("\\wildcards") for path in result.paths))

    def test_baked_manager_symlink_does_not_abort_and_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ComfyFleet"
            layout = FleetLayout(root)
            layout.wildcards.mkdir(parents=True)
            (layout.wildcards / "nested.txt").write_text("w", encoding="utf-8")
            layout.models.mkdir()
            (layout.models / "checkpoints").mkdir()
            (layout.models / "checkpoints" / "model.safetensors").write_text("m", encoding="utf-8")
            layout.files.mkdir()
            (layout.files / "k2fb").mkdir()
            nodes = layout.root / "custom_nodes_k2fb"
            nodes.mkdir()
            local = nodes / "LocalNode"
            local.mkdir()
            (local / "node.py").write_text("n", encoding="utf-8")
            manager = nodes / "ComfyUI-Manager"
            manager.symlink_to(BAKED_CUSTOM_NODES / "ComfyUI-Manager")
            impact = nodes / "ComfyUI-Impact-Pack"
            impact.symlink_to(BAKED_CUSTOM_NODES / "ComfyUI-Impact-Pack")
            nested = local / "also-baked"
            nested.symlink_to(BAKED_CUSTOM_NODES / "RES4LYF")
            assert_allowlisted(manager, layout)

            chowned = []

            def spy(path, uid, gid):
                chowned.append(Path(path))
                self.assertEqual((uid, gid), (7, 8))

            result = fix_owner(layout, resolve=lambda: (7, 8), chown=spy)
            names = {path.name for path in chowned}
            self.assertIn("wildcards", names)
            self.assertIn("nested.txt", names)
            self.assertIn("models", names)
            self.assertIn("checkpoints", names)
            self.assertIn("model.safetensors", names)
            self.assertIn("files", names)
            self.assertIn("k2fb", names)
            self.assertIn("custom_nodes_k2fb", names)
            self.assertIn("LocalNode", names)
            self.assertIn("node.py", names)
            self.assertIn(manager, chowned)
            self.assertIn(impact, chowned)
            self.assertIn(nested, chowned)
            for path in chowned:
                self.assertFalse(path == BAKED_CUSTOM_NODES or BAKED_CUSTOM_NODES in path.parents)
            self.assertEqual(result.user, "comfyuser")
            self.assertEqual(result.group, "comfyuser")
            self.assertTrue(any(path.endswith("custom_nodes_k2fb") for path in result.paths))

    def test_existing_baked_target_is_not_chowned_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ComfyFleet"
            layout = FleetLayout(root)
            nodes = layout.root / "custom_nodes_k2fb"
            nodes.mkdir(parents=True)
            (nodes / "node.py").write_text("n", encoding="utf-8")
            baked = Path(tmp) / "image_baked_custom_nodes"
            image_node = baked / "ComfyUI-Manager"
            image_node.mkdir(parents=True)
            secret = image_node / "secret.py"
            secret.write_text("image", encoding="utf-8")
            (image_node / "nested").mkdir()
            (image_node / "nested" / "weight.bin").write_text("w", encoding="utf-8")
            link = nodes / "ComfyUI-Manager"
            link.symlink_to(image_node)
            chowned = []

            def spy(path, uid, gid):
                chowned.append(Path(path))
                self.assertEqual((uid, gid), (7, 8))

            with mock.patch("comfyfleet.ownership.BAKED_CUSTOM_NODES", baked):
                result = fix_owner(layout, resolve=lambda: (7, 8), chown=spy)
            self.assertEqual(result.user, "comfyuser")
            self.assertEqual(result.group, "comfyuser")
            self.assertIn(nodes / "node.py", chowned)
            self.assertIn(link, chowned)
            self.assertNotIn(secret, chowned)
            self.assertNotIn(image_node, chowned)
            self.assertNotIn(image_node / "nested" / "weight.bin", chowned)
            self.assertTrue(all(path != baked and baked not in path.parents for path in chowned))

    def test_symlink_escape_is_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ComfyFleet"
            layout = FleetLayout(root)
            nodes = layout.root / "custom_nodes_k2fb"
            nodes.mkdir(parents=True)
            (nodes / "node.py").write_text("n", encoding="utf-8")
            manager = nodes / "ComfyUI-Manager"
            manager.symlink_to(BAKED_CUSTOM_NODES / "ComfyUI-Manager")
            outside = Path(tmp) / "outside"
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("nope", encoding="utf-8")
            escape = nodes / "escape"
            escape.symlink_to(outside)
            lookalike = nodes / "lookalike"
            lookalike.symlink_to(Path("/opt/comfyfleet/baked_custom_nodes_evil/ComfyUI-Manager"))
            dotted = nodes / "dotted"
            dotted.symlink_to(BAKED_CUSTOM_NODES / "ComfyUI-Manager" / ".." / ".." / ".." / "etc")
            named = nodes / "ComfyUI-Impact-Pack"
            named.symlink_to(secret)

            for path in (escape, lookalike, dotted, named):
                with self.assertRaises(FleetError) as ctx:
                    assert_allowlisted(path, layout)
                message = str(ctx.exception)
                self.assertIn("refusing symlink that leaves the fix-owner allowlist", message)
                self.assertNotIn(str(BAKED_CUSTOM_NODES / "ComfyUI-Manager"), message)

            chowned = []

            def spy(path, uid, gid):
                chowned.append(Path(path))
                self.assertEqual((uid, gid), (7, 8))

            with self.assertRaises(FleetError) as ctx:
                fix_owner(layout, resolve=lambda: (7, 8), chown=spy)
            message = str(ctx.exception)
            self.assertIn("refusing symlink that leaves the fix-owner allowlist", message)
            self.assertNotIn(str(BAKED_CUSTOM_NODES / "ComfyUI-Manager"), message)
            self.assertNotIn(outside, chowned)
            self.assertNotIn(secret, chowned)
            self.assertTrue(all(path != outside and outside not in path.parents for path in chowned))
            self.assertTrue(
                all(path != BAKED_CUSTOM_NODES and BAKED_CUSTOM_NODES not in path.parents for path in chowned)
            )

    def test_chown_inode_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("x", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with mock.patch("comfyfleet.ownership.os.chown") as chown:
                chown_inode(link, 7, 8)
            chown.assert_called_once_with(link, 7, 8, follow_symlinks=False)

    def test_missing_comfyuser_account_is_created_without_a_home_sibling(self):
        users: dict[str, tuple[int, int]] = {}
        groups: dict[str, int] = {}
        calls: list[list[str]] = []

        def getpwnam(name):
            if name not in users:
                raise KeyError(name)
            uid, gid = users[name]
            return mock.Mock(pw_uid=uid, pw_gid=gid)

        def getgrnam(name):
            if name not in groups:
                raise KeyError(name)
            return mock.Mock(gr_gid=groups[name])

        def run(argv):
            calls.append(list(argv))
            if argv[0] == "groupadd":
                groups[argv[-1]] = 1100
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[0] == "useradd":
                users[argv[-1]] = (1101, 1100)
                return subprocess.CompletedProcess(argv, 0, "", "")
            raise AssertionError(argv)

        with mock.patch("comfyfleet.ownership.pwd.getpwnam", getpwnam), mock.patch(
            "comfyfleet.ownership.grp.getgrnam", getgrnam
        ), mock.patch("comfyfleet.ownership._account_binary", lambda tool, _kind, _account: tool):
            uid, gid = resolve_owner_ids(run=run)
            blank_uid, blank_gid = resolve_owner_ids("  ", "", run=run)

        self.assertEqual((uid, gid), (1101, 1100))
        self.assertEqual((blank_uid, blank_gid), (1101, 1100))
        self.assertEqual(calls[0], ["groupadd", "comfyuser"])
        useradd = calls[1]
        self.assertEqual(useradd[0], "useradd")
        self.assertIn("--no-create-home", useradd)
        self.assertNotIn("--create-home", useradd)
        self.assertNotIn("-m", useradd)
        self.assertEqual(useradd[useradd.index("--gid") + 1], "comfyuser")
        self.assertEqual(useradd[useradd.index("--home-dir") + 1], "/home/ComfyFleet")
        self.assertNotIn("/home/comfyuser", useradd)
        self.assertNotIn("/home/comfyui", useradd)
        self.assertTrue(all("adduser" not in part for part in useradd))
        self.assertEqual(useradd[-1], "comfyuser")
        self.assertEqual(calls[2:], [])

        calls.clear()
        with mock.patch("comfyfleet.ownership.pwd.getpwnam", getpwnam), mock.patch(
            "comfyfleet.ownership.grp.getgrnam", getgrnam
        ):
            again = resolve_owner_ids(run=run)
        self.assertEqual(again, (1101, 1100))
        self.assertEqual(calls, [])

    def test_requested_account_is_created_without_a_home_sibling(self):
        users: dict[str, tuple[int, int]] = {}
        groups: dict[str, int] = {}
        calls: list[list[str]] = []

        def getpwnam(name):
            if name not in users:
                raise KeyError(name)
            uid, gid = users[name]
            return mock.Mock(pw_uid=uid, pw_gid=gid)

        def getgrnam(name):
            if name not in groups:
                raise KeyError(name)
            return mock.Mock(gr_gid=groups[name])

        def run(argv):
            calls.append(list(argv))
            if argv[0] == "groupadd":
                groups[argv[-1]] = 20
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[0] == "useradd":
                users[argv[-1]] = (21, 20)
                return subprocess.CompletedProcess(argv, 0, "", "")
            raise AssertionError(argv)

        with mock.patch("comfyfleet.ownership.pwd.getpwnam", getpwnam), mock.patch(
            "comfyfleet.ownership.grp.getgrnam", getgrnam
        ), mock.patch("comfyfleet.ownership._account_binary", lambda tool, _kind, _account: tool):
            uid, gid = resolve_owner_ids("alice", "render", run=run)

        self.assertEqual((uid, gid), (21, 20))
        self.assertEqual(calls[0], ["groupadd", "render"])
        useradd = calls[1]
        self.assertEqual(useradd[-1], "alice")
        self.assertEqual(useradd[useradd.index("--gid") + 1], "render")
        self.assertEqual(useradd[useradd.index("--home-dir") + 1], "/home/ComfyFleet")
        self.assertIn("--no-create-home", useradd)
        self.assertNotIn("/home/alice", useradd)
        self.assertNotIn("/home/comfyuser", useradd)
        self.assertNotIn("/home/comfyui", useradd)

    def test_fix_owner_chowns_after_creating_a_missing_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = FleetLayout(Path(tmp) / "ComfyFleet")
            layout.models.mkdir(parents=True)
            state = {"user": False}
            created = []
            chowned = []

            def getpwnam(name):
                if name != "comfyuser" or not state["user"]:
                    raise KeyError(name)
                return mock.Mock(pw_uid=42, pw_gid=43)

            def getgrnam(name):
                if name != "comfyuser":
                    raise KeyError(name)
                return mock.Mock(gr_gid=43)

            def run(argv):
                created.append(list(argv))
                if argv[0] == "useradd":
                    state["user"] = True
                return subprocess.CompletedProcess(argv, 0, "", "")

            def spy(path, uid, gid):
                chowned.append((Path(path), uid, gid))

            with mock.patch("comfyfleet.ownership.pwd.getpwnam", getpwnam), mock.patch(
                "comfyfleet.ownership.grp.getgrnam", getgrnam
            ), mock.patch("comfyfleet.ownership._account_binary", lambda tool, _kind, _account: tool):
                result = fix_owner(layout, resolve=lambda: resolve_owner_ids(run=run), chown=spy)

            self.assertEqual((result.uid, result.gid), (42, 43))
            self.assertEqual(result.user, "comfyuser")
            self.assertEqual(result.group, "comfyuser")
            self.assertEqual(chowned, [(layout.models, 42, 43)])
            self.assertEqual(created[0][0], "useradd")
            self.assertNotIn("groupadd", [item[0] for item in created])
            self.assertIn("--no-create-home", created[0])
            self.assertEqual(created[0][-1], "comfyuser")

    def test_invalid_account_name_does_not_chown(self):
        chowned = []
        with tempfile.TemporaryDirectory() as tmp:
            layout = FleetLayout(Path(tmp) / "ComfyFleet")
            layout.models.mkdir(parents=True)
            with self.assertRaises(FleetError) as ctx:
                fix_owner(
                    layout,
                    user="alice bob",
                    group="comfyuser",
                    chown=lambda *args: chowned.append(args),
                )
        self.assertIn("host user name is invalid", str(ctx.exception))
        self.assertEqual(chowned, [])

    def test_account_create_failure_names_the_host_user(self):
        def getpwnam(_name):
            raise KeyError("comfyuser")

        def getgrnam(_name):
            return mock.Mock(gr_gid=7)

        def run(argv):
            return subprocess.CompletedProcess(argv, 1, "", "useradd: Permission denied")

        with mock.patch("comfyfleet.ownership.pwd.getpwnam", getpwnam), mock.patch(
            "comfyfleet.ownership.grp.getgrnam", getgrnam
        ), mock.patch("comfyfleet.ownership._account_binary", lambda tool, _kind, _account: tool):
            with self.assertRaises(FleetError) as ctx:
                resolve_owner_ids("alice", run=run)
        message = str(ctx.exception)
        self.assertIn("cannot resolve host user alice by name", message)
        self.assertIn("Permission denied", message)
        self.assertNotIn("comfyui", message)
        self.assertNotIn("hose", message)
        self.assertNotIn("Create the comfyuser user before running fix-owner", message)

        chowned = []
        with tempfile.TemporaryDirectory() as tmp:
            layout = FleetLayout(Path(tmp))
            layout.files.mkdir()
            with mock.patch(
                "comfyfleet.ownership.resolve_owner_ids",
                side_effect=FleetError(message),
            ):
                with self.assertRaises(FleetError) as blocked:
                    fix_owner(layout, user="alice", chown=lambda *args: chowned.append(args))
            self.assertEqual(str(blocked.exception), message)
            self.assertEqual(chowned, [])


class PruneTests(unittest.TestCase):
    def test_stopped_fleet_instances_are_excluded(self):
        managed = "comfyfleet.managed=true,comfyfleet.name=portrait"
        text = "\n".join(
            [
                "aaaaaaaaaaaa\tjunk\tExited (0) 3 hours ago\t",
                f"bbbbbbbbbbbb\tportrait\tExited (1) 2 days ago\t{managed}",
                "cccccccccccc\trunning\tUp 4 seconds\t",
                "dddddddddddd\tfleet-up\tUp 4 seconds\tcomfyfleet.managed=true",
                "eeeeeeeeeeee\tcreated-junk\tCreated\t",
                "ffffffffffff\tpaused\tPaused\t",
            ]
        )
        records = parse_ps(text)
        self.assertTrue(is_managed(records[1]))
        self.assertTrue(is_managed(records[3]))
        remove, kept = select_prune_targets(records)
        self.assertEqual([item.id for item in remove], ["aaaaaaaaaaaa", "eeeeeeeeeeee"])
        self.assertEqual([item.id for item in kept], ["bbbbbbbbbbbb", "dddddddddddd"])

        class Engine:
            def __init__(self):
                self.records = records
                self.removed = []

            def list_container_records(self):
                return list(self.records)

            def remove_stopped(self, container_id):
                self.removed.append(container_id)

        engine = Engine()
        result = prune_dangling_containers(engine)
        self.assertEqual(result.removed, ("aaaaaaaaaaaa", "eeeeeeeeeeee"))
        self.assertEqual(engine.removed, ["aaaaaaaaaaaa", "eeeeeeeeeeee"])
        self.assertIn("bbbbbbbbbbbb", result.kept_managed)
        self.assertNotIn("bbbbbbbbbbbb", result.removed)

    def test_docker_rm_is_not_force_and_rejects_a_managed_id_shape(self):
        calls = []

        def run(argv, timeout=None):
            calls.append(list(argv))
            if argv[1:3] == ["ps", "-a"]:
                stdout = "aaaaaaaaaaaa\tjunk\tExited (0) 1s\t\nbbbbbbbbbbbb\tfleet\tExited (0) 1s\tcomfyfleet.managed=true\n"
            else:
                stdout = ""
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

        result = prune_dangling_containers(DockerCLI(run=run))
        self.assertEqual(result.removed, ("aaaaaaaaaaaa",))
        rm = [argv for argv in calls if argv[1] == "rm"]
        self.assertEqual(rm, [["docker", "rm", "aaaaaaaaaaaa"]])
        self.assertNotIn("-f", rm[0])
        client = DockerCLI(run=run)
        with self.assertRaises(FleetError):
            client.remove_stopped("comfyfleet.managed=true")


class HostApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "home"
        self.layout = FleetLayout(root)
        self.layout.wildcards.mkdir(parents=True)
        self.docker = _PruneDocker()
        self.ctx = ApiContext(
            layout=self.layout,
            docker=self.docker,
            detect_gpus=lambda: [],
            port_in_use=lambda _port: False,
            ui_dir=root,
            password=PASSWORD,
            sessions=SessionStore(),
            login_guard=LoginGuard(fail_delay_s=0),
        )
        self.httpd = make_server("127.0.0.1", 0, self.ctx)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _open(self, method, path, auth=True, data=None, headers=None):
        request_headers = {}
        if auth:
            request_headers["Authorization"] = f"Bearer {PASSWORD}"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_fix_owner_requires_auth_and_uses_the_helper(self):
        chowned = []
        status, raw = self._open("POST", "/api/host/fix-owner", auth=False)
        self.assertEqual(status, 401)
        self.assertEqual(chowned, [])
        with mock.patch("comfyfleet.ownership.resolve_owner_ids", return_value=(4, 5)) as resolve, mock.patch(
            "comfyfleet.ownership.chown_inode",
            lambda path, uid, gid: chowned.append((str(path), uid, gid)),
        ):
            status, raw = self._open("POST", "/api/host/fix-owner", auth=True)
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["uid"], 4)
        self.assertEqual(payload["user"], "comfyuser")
        self.assertEqual(payload["group"], "comfyuser")
        resolve.assert_called_once_with("comfyuser", "comfyuser")
        self.assertTrue(any(path.endswith("wildcards") for path in payload["paths"]))
        self.assertTrue(chowned)
        status, raw = self._open("GET", "/api/host/fix-owner", auth=True)
        self.assertEqual(status, 405)

    def test_fix_owner_route_uses_a_requested_account(self):
        body = json.dumps({"user": "alice", "group": ""}).encode("utf-8")
        with mock.patch("comfyfleet.ownership.resolve_owner_ids", return_value=(9, 9)) as resolve, mock.patch(
            "comfyfleet.ownership.chown_inode",
            lambda *_args: None,
        ):
            status, raw = self._open(
                "POST",
                "/api/host/fix-owner",
                data=body,
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["user"], "alice")
        self.assertEqual(payload["group"], "comfyuser")
        resolve.assert_called_once_with("alice", "comfyuser")

        status, raw = self._open(
            "POST",
            "/api/host/fix-owner",
            data=b'{"user":"bad name"}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertIn("host user name is invalid", payload["error"])
        self.assertNotIn("comfyui", payload["error"])

    def test_fix_owner_route_accepts_a_baked_manager_symlink(self):
        nodes = self.layout.root / "custom_nodes_k2fb"
        nodes.mkdir()
        (nodes / "node.py").write_text("n", encoding="utf-8")
        link = nodes / "ComfyUI-Manager"
        link.symlink_to(BAKED_CUSTOM_NODES / "ComfyUI-Manager")
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("nope", encoding="utf-8")
        chowned = []
        with mock.patch("comfyfleet.ownership.resolve_owner_ids", return_value=(4, 5)), mock.patch(
            "comfyfleet.ownership.chown_inode",
            lambda path, uid, gid: chowned.append((str(path), uid, gid)),
        ):
            status, raw = self._open("POST", "/api/host/fix-owner", auth=True)
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["user"], "comfyuser")
        self.assertEqual(payload["group"], "comfyuser")
        self.assertTrue(any(path.endswith("custom_nodes_k2fb") for path in payload["paths"]))
        joined = "\n".join(path for path, _uid, _gid in chowned)
        self.assertIn("node.py", joined)
        self.assertIn(str(link), joined)
        self.assertNotIn(str(BAKED_CUSTOM_NODES / "ComfyUI-Manager"), joined)
        self.assertNotIn(str(outside), joined)

        escape = nodes / "escape"
        escape.symlink_to(outside)
        chowned.clear()
        with mock.patch("comfyfleet.ownership.resolve_owner_ids", return_value=(4, 5)), mock.patch(
            "comfyfleet.ownership.chown_inode",
            lambda path, uid, gid: chowned.append((str(path), uid, gid)),
        ):
            status, raw = self._open("POST", "/api/host/fix-owner", auth=True)
        self.assertEqual(status, 400, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertIn("refusing symlink that leaves the fix-owner allowlist", payload["error"])
        self.assertIn(str(outside), payload["error"])
        self.assertNotIn(str(BAKED_CUSTOM_NODES / "ComfyUI-Manager"), payload["error"])
        self.assertTrue(all(outside.as_posix() not in path for path, _uid, _gid in chowned))

    def test_prune_route_keeps_managed_containers(self):
        status, _raw = self._open("POST", "/api/host/prune-dangling", auth=False)
        self.assertEqual(status, 401)
        self.assertEqual(self.docker.removed, [])
        status, raw = self._open("POST", "/api/host/prune-dangling", auth=True)
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["removed"], ["aaaaaaaaaaaa"])
        self.assertIn("bbbbbbbbbbbb", payload["kept_managed"])
        self.assertNotIn("bbbbbbbbbbbb", self.docker.removed)


class _PruneDocker:
    def __init__(self):
        self.removed = []

    def list_container_records(self):
        return parse_ps(
            "aaaaaaaaaaaa\tjunk\tExited (0) 1s\t\n"
            "bbbbbbbbbbbb\tportrait\tExited (0) 1s\tcomfyfleet.managed=true\n"
        )

    def remove_stopped(self, container_id):
        self.removed.append(container_id)

    def status(self, _name):
        return None

    def running_names(self):
        return []


if __name__ == "__main__":
    unittest.main()
