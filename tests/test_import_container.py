"""Import dedupe, copy/move, and job resume."""

import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from comfyfleet.auth import LoginGuard, SessionStore
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu
from comfyfleet.http_api import ApiContext, make_server
from comfyfleet.import_container import (
    DuplicatesStore,
    Gate,
    HashCache,
    ImportCancelled,
    ImportService,
    build_index,
    classify_model,
    copy_verified,
    duplicate_entry,
    hash_file,
    hash_tar_member,
    parse_container_inspect,
    perform_action,
    plan_tree,
    stream_tar_member,
    verify_then_delete,
)
from comfyfleet.paths import FleetLayout
from comfyfleet.prune import ContainerRecord


PASSWORD = "test-password"
WORKFLOW = {
    "last_node_id": 0,
    "last_link_id": 0,
    "nodes": [],
    "links": [],
    "version": 0.4,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class FakeDocker:
    def __init__(self):
        self.containers = {}
        self.calls = []

    def add(self, name, inspect, status="exited"):
        self.containers[name] = {"status": status, "inspect": inspect}

    def inspect_container(self, name):
        item = self.containers.get(name)
        if item is None:
            raise FleetError(f"no such container {name}")
        return item["inspect"]

    def list_container_records(self):
        rows = []
        for name, item in self.containers.items():
            labels = ((item["inspect"].get("Config") or {}).get("Labels") or {})
            rows.append(
                ContainerRecord(
                    id=str(item["inspect"].get("Id") or ""),
                    name=name,
                    status=item["status"],
                    labels=labels,
                    raw_labels="",
                )
            )
        return rows

    def status(self, name):
        item = self.containers.get(name)
        if item is None:
            return None
        return item["status"]

    def create(self, args):
        self.calls.append(("create", list(args)))
        name = args[args.index("--name") + 1]
        self.containers[name] = {
            "status": "created",
            "inspect": {
                "Id": "b" * 12,
                "Name": "/" + name,
                "State": {"Status": "created"},
                "Config": {
                    "Image": "comfyfleet:cu130",
                    "Env": [],
                    "Labels": {"comfyfleet.managed": "true", "comfyfleet.name": name},
                },
                "HostConfig": {},
                "Mounts": [],
            },
        }

    def start(self, name):
        self.calls.append(("start", name))

    def stop(self, name):
        self.calls.append(("stop", name))

    def remove(self, name):
        self.calls.append(("rm", name))
        self.containers.pop(name, None)

    def remove_stopped(self, container_id):
        self.calls.append(("rm-stopped", container_id))
        for name, item in list(self.containers.items()):
            if item["inspect"].get("Id") == container_id:
                self.containers.pop(name, None)

    def running_names(self):
        return [name for name, item in self.containers.items() if item["status"] == "running"]

    def update_restart(self, name, policy):
        self.calls.append(("update-restart", name, policy))


def _inspect(name, mounts, *, status="exited", gpus=None, env=None, managed=False):
    return {
        "Id": "a" * 64,
        "Name": "/" + name,
        "State": {"Status": status},
        "Config": {
            "Image": "comfyui:local",
            "Env": env or ["FOO=bar", "API_TOKEN=sekret"],
            "Labels": {"comfyfleet.managed": "true"} if managed else {},
        },
        "HostConfig": {
            "DeviceRequests": [{"DeviceIDs": [str(index) for index in (gpus or [0])]}],
            "PortBindings": {"8188/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8288"}]},
        },
        "Mounts": mounts,
    }


class ClassifyTests(unittest.TestCase):
    def test_four_dedupe_cases(self):
        root = "/home/ComfyFleet/models"
        same = _sha(b"same-bytes")
        other = _sha(b"other-bytes")
        fresh = _sha(b"fresh-bytes")
        index = {
            "root": root,
            "by_rel": {
                "checkpoints/demo.safetensors": {
                    "path": root + "/checkpoints/demo.safetensors",
                    "rel": "checkpoints/demo.safetensors",
                    "sha256": same,
                    "size": 10,
                },
                "checkpoints/kept.safetensors": {
                    "path": root + "/checkpoints/kept.safetensors",
                    "rel": "checkpoints/kept.safetensors",
                    "sha256": other,
                    "size": 11,
                },
                "loras/existing.safetensors": {
                    "path": root + "/loras/existing.safetensors",
                    "rel": "loras/existing.safetensors",
                    "sha256": _sha(b"dupe-bytes"),
                    "size": 12,
                },
            },
            "by_hash": {},
        }
        for row in index["by_rel"].values():
            index["by_hash"][row["sha256"]] = row

        skip = classify_model("checkpoints/demo.safetensors", same, 10, index)
        self.assertEqual(skip["action"], "skip_same")

        conflict = classify_model("checkpoints/kept.safetensors", fresh, 9, index)
        self.assertEqual(conflict["action"], "conflict")
        self.assertTrue(conflict["dest"].endswith("kept (imported).safetensors"))
        self.assertNotEqual(conflict["dest"], index["by_rel"]["checkpoints/kept.safetensors"]["path"])

        duplicate = classify_model("checkpoints/renamed.safetensors", _sha(b"dupe-bytes"), 12, index)
        self.assertEqual(duplicate["action"], "skip_duplicate")
        self.assertEqual(duplicate["existing"], root + "/loras/existing.safetensors")

        transfer = classify_model("vae/new.safetensors", fresh, 9, index)
        self.assertEqual(transfer["action"], "transfer")
        self.assertTrue(transfer["dest"].endswith("vae/new.safetensors"))

    def test_renamed_copy_that_already_matches_is_a_duplicate(self):
        root = "/home/ComfyFleet/models"
        digest = _sha(b"imported-copy")
        imported = "checkpoints/demo (imported).safetensors"
        index = {
            "root": root,
            "by_rel": {
                "checkpoints/demo.safetensors": {
                    "path": root + "/checkpoints/demo.safetensors",
                    "rel": "checkpoints/demo.safetensors",
                    "sha256": _sha(b"original"),
                    "size": 8,
                },
                imported: {
                    "path": root + "/" + imported,
                    "rel": imported,
                    "sha256": digest,
                    "size": 13,
                },
            },
            "by_hash": {},
        }
        for row in index["by_rel"].values():
            index["by_hash"].setdefault(row["sha256"], row)
        decision = classify_model("checkpoints/demo.safetensors", digest, 13, index)
        self.assertEqual(decision["action"], "skip_duplicate")
        self.assertEqual(decision["existing_rel"], imported)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fleet = self.root / "fleet"
        self.layout = FleetLayout(self.fleet)
        self.source = self.root / "mount" / "models"
        self.dest = self.fleet / "models"
        self.cache = HashCache(self.fleet / ".import" / "hashes.json")
        self.dupes = DuplicatesStore(self.fleet / ".import" / "duplicates.json")
        self.same = b"same-model-bytes"
        self.clash = b"destination-bytes"
        self.incoming = b"incoming-different"
        self.dupe = b"duplicate-model"
        self.fresh = b"brand-new-model"
        _write(self.dest / "checkpoints" / "demo.safetensors", self.same)
        _write(self.dest / "checkpoints" / "kept.safetensors", self.clash)
        _write(self.dest / "loras" / "existing.safetensors", self.dupe)
        _write(self.source / "checkpoints" / "demo.safetensors", self.same)
        _write(self.source / "checkpoints" / "kept.safetensors", self.incoming)
        _write(self.source / "checkpoints" / "renamed.safetensors", self.dupe)
        _write(self.source / "vae" / "new.safetensors", self.fresh)

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_hashes_without_writing_models(self):
        before = (self.dest / "checkpoints" / "kept.safetensors").read_bytes()
        actions = plan_tree(self.source, self.dest, cache=self.cache)
        self.cache.save()
        self.assertEqual((self.dest / "checkpoints" / "kept.safetensors").read_bytes(), before)
        kinds = {Path(item["rel"]).name: item["action"] for item in actions}
        self.assertEqual(
            kinds,
            {
                "demo.safetensors": "skip_same",
                "kept.safetensors": "conflict",
                "renamed.safetensors": "skip_duplicate",
                "new.safetensors": "transfer",
            },
        )
        again = plan_tree(self.source, self.dest, cache=HashCache(self.cache.path))
        self.assertEqual([item["sha256"] for item in again], [item["sha256"] for item in actions])

    def test_hash_cache_skips_a_second_read(self):
        path = self.source / "vae" / "new.safetensors"
        digest = hash_file(path, self.cache)
        self.cache.save()
        st = path.stat()
        path.write_bytes(b"y" * st.st_size)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        cached = hash_file(path, HashCache(self.cache.path))
        self.assertEqual(cached, digest)
        fresh = hash_file(path, HashCache(self.cache.path), fresh=True)
        self.assertNotEqual(fresh, digest)

    def test_copy_leaves_the_source_and_never_overwrites(self):
        actions = plan_tree(self.source, self.dest, cache=self.cache)
        self._apply(actions, "copy")
        self.assertEqual((self.source / "vae" / "new.safetensors").read_bytes(), self.fresh)
        self.assertEqual((self.dest / "vae" / "new.safetensors").read_bytes(), self.fresh)
        self.assertEqual(_sha((self.dest / "vae" / "new.safetensors").read_bytes()), _sha(self.fresh))
        self.assertEqual((self.dest / "checkpoints" / "kept.safetensors").read_bytes(), self.clash)
        imported = self.dest / "checkpoints" / "kept (imported).safetensors"
        self.assertEqual(imported.read_bytes(), self.incoming)
        self.assertTrue((self.source / "checkpoints" / "kept.safetensors").is_file())
        self.assertFalse((self.dest / "checkpoints" / "renamed.safetensors").exists())

    def test_move_deletes_source_only_after_destination_hash_matches(self):
        actions = plan_tree(self.source, self.dest, cache=self.cache)
        self._apply(actions, "move")
        self.assertFalse((self.source / "vae" / "new.safetensors").exists())
        self.assertEqual((self.dest / "vae" / "new.safetensors").read_bytes(), self.fresh)
        self.assertFalse((self.source / "checkpoints" / "demo.safetensors").exists())
        self.assertEqual((self.dest / "checkpoints" / "demo.safetensors").read_bytes(), self.same)
        self.assertFalse((self.source / "checkpoints" / "renamed.safetensors").exists())
        self.assertEqual((self.dest / "loras" / "existing.safetensors").read_bytes(), self.dupe)
        self.assertFalse((self.source / "checkpoints" / "kept.safetensors").exists())
        self.assertEqual((self.dest / "checkpoints" / "kept.safetensors").read_bytes(), self.clash)

    def test_verify_failure_keeps_the_source(self):
        source = self.source / "vae" / "new.safetensors"
        dest = self.dest / "vae" / "new.safetensors"
        _write(dest, self.fresh)
        with self.assertRaises(FleetError):
            verify_then_delete(source, dest, "0" * 64, self.cache)
        self.assertTrue(source.is_file())
        self.assertEqual(dest.read_bytes(), self.fresh)

    def test_duplicates_list_is_append_only_and_read_only(self):
        actions = plan_tree(self.source, self.dest, cache=self.cache)
        duplicate = next(item for item in actions if item["action"] == "skip_duplicate")
        self._apply([duplicate], "copy")
        rows = self.dupes.read()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            set(rows[0]),
            {"incoming_name", "existing_name", "existing_path", "size", "hash", "import_id", "date"},
        )
        self.assertEqual(rows[0]["incoming_name"], "renamed.safetensors")
        self.assertEqual(rows[0]["existing_name"], "existing.safetensors")
        self.assertEqual(rows[0]["hash"], _sha(self.dupe))
        self.assertEqual(rows[0]["import_id"], "imp-test")
        mode = stat.S_IMODE(self.dupes.path.stat().st_mode)
        self.assertEqual(mode & 0o222, 0)
        extra = duplicate_entry(duplicate, "imp-second", "2026-10-06T00:00:00Z")
        self.dupes.append(extra)
        self.assertEqual(len(self.dupes.read()), 2)
        self.assertEqual(self.dupes.read()[0]["import_id"], "imp-test")
        self.assertFalse(hasattr(self.dupes, "delete"))
        self.assertFalse(hasattr(self.dupes, "replace"))

    def test_cancel_leaves_no_partial_file(self):
        source = self.root / "mount" / "big.bin"
        source.write_bytes(b"z" * (1024 * 1024 + 32))
        dest = self.fleet / "models" / "checkpoints" / "big.safetensors"
        gate = Gate()

        def cancel_mid(_done, _total):
            gate.cancel()

        with self.assertRaises(ImportCancelled):
            copy_verified(
                source,
                dest,
                _sha(source.read_bytes()),
                self.cache,
                gate=gate,
                on_chunk=cancel_mid,
                fleet_root=self.fleet,
                source_roots=[self.root / "mount"],
            )
        self.assertFalse(dest.exists())
        self.assertEqual(list(dest.parent.glob("*.comfyfleet-partial")), [])
        self.assertTrue(source.is_file())

    def test_refuses_paths_outside_the_fleet_and_source_mounts(self):
        outside = self.root / "secret.bin"
        outside.write_bytes(b"nope")
        action = {
            "action": "transfer",
            "source": str(outside),
            "dest": str(self.dest / "checkpoints" / "secret.safetensors"),
            "rel": "checkpoints/secret.safetensors",
            "sha256": _sha(b"nope"),
            "size": 4,
            "existing": "",
        }
        with self.assertRaises(FleetError):
            perform_action(
                action,
                mode="copy",
                cache=self.cache,
                duplicates=None,
                import_id="imp-test",
                fleet_root=self.fleet,
                source_roots=[self.source],
            )
        self.assertFalse((self.dest / "checkpoints" / "secret.safetensors").exists())

    def _apply(self, actions, mode):
        for action in actions:
            action["models"] = True
            perform_action(
                action,
                mode=mode,
                cache=self.cache,
                duplicates=self.dupes if action["action"] == "skip_duplicate" else None,
                import_id="imp-test",
                fleet_root=self.fleet,
                source_roots=[self.source],
                now="2026-10-06T00:00:00Z",
            )


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fleet = self.root / "fleet"
        self.layout = FleetLayout(self.fleet)
        self.mount = self.root / "old"
        self.docker = FakeDocker()
        self.gpus = [Gpu(0, "GPU0", "8192 MiB")]
        _write(self.mount / "models" / "checkpoints" / "new.safetensors", b"new-weights")
        _write(
            self.mount / "user" / "default" / "workflows" / "portrait.json",
            json.dumps(WORKFLOW).encode(),
        )
        _write(self.mount / "input" / "ref.png", b"png")
        mounts = [
            {"Type": "bind", "Source": str(self.mount / "models"), "Destination": "/opt/ComfyUI/models"},
            {"Type": "bind", "Source": str(self.mount / "input"), "Destination": "/opt/ComfyUI/input"},
            {"Type": "bind", "Source": str(self.mount / "user"), "Destination": "/opt/ComfyUI/user"},
        ]
        self.docker.add("old-ui", _inspect("old-ui", mounts))
        self.service = ImportService(self.layout, port_in_use=lambda _port: False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_copy_creates_an_instance_and_leaves_the_old_container(self):
        job = self.service.create_job(
            self.docker,
            self.gpus,
            {"container": "old-ui", "gpus": [0], "mode": "copy"},
            inline=True,
        )
        self.assertEqual(job["status"], "awaiting_confirm")
        self.assertGreaterEqual(job["summary"]["transfer"], 1)
        self.assertIn("space saved", "\n".join(job["log_tail"]))
        done = self.service.start_transfer(job["id"], inline=True)
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertTrue((self.mount / "models" / "checkpoints" / "new.safetensors").is_file())
        self.assertEqual(
            (self.fleet / "models" / "checkpoints" / "new.safetensors").read_bytes(),
            b"new-weights",
        )
        self.assertTrue((self.fleet / "files" / "old-ui-import" / "default_workflow.json").is_file())
        self.assertEqual(self.docker.status("old-ui"), "exited")
        self.assertNotIn("rm-stopped", [name for name, *_rest in self.docker.calls])
        self.assertTrue(any(name == "create" for name, *_rest in self.docker.calls))

    def test_move_removes_the_source_file_after_verify(self):
        job = self.service.create_job(
            self.docker,
            self.gpus,
            {"container": "old-ui", "gpus": [0], "mode": "move", "name": "moved"},
            inline=True,
        )
        done = self.service.start_transfer(job["id"], inline=True)
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertFalse((self.mount / "models" / "checkpoints" / "new.safetensors").exists())
        self.assertEqual(
            (self.fleet / "models" / "checkpoints" / "new.safetensors").read_bytes(),
            b"new-weights",
        )
        self.assertEqual(self.docker.status("old-ui"), "exited")

    def test_remove_old_is_separate_and_refuses_a_running_container(self):
        job = self.service.create_job(
            self.docker,
            self.gpus,
            {"container": "old-ui", "gpus": [0], "mode": "copy", "name": "kept"},
            inline=True,
        )
        self.service.start_transfer(job["id"], inline=True)
        self.docker.containers["old-ui"]["status"] = "running"
        self.docker.containers["old-ui"]["inspect"]["State"]["Status"] = "running"
        with self.assertRaises(FleetError):
            self.service.remove_old(job["id"], self.docker)
        self.assertEqual(self.docker.status("old-ui"), "running")
        self.docker.containers["old-ui"]["status"] = "exited"
        self.docker.containers["old-ui"]["inspect"]["State"]["Status"] = "exited"
        self.service.remove_old(job["id"], self.docker)
        self.assertIsNone(self.docker.status("old-ui"))
        self.assertTrue((self.fleet / "models" / "checkpoints" / "new.safetensors").is_file())

    def test_job_persists_and_a_new_service_can_reopen_it(self):
        job = self.service.create_job(
            self.docker,
            self.gpus,
            {"container": "old-ui", "gpus": [0]},
            inline=True,
        )
        reloaded = ImportService(self.layout, port_in_use=lambda _port: False)
        active = reloaded.active_job()
        self.assertIsNotNone(active)
        self.assertEqual(active["id"], job["id"])
        self.assertEqual(active["status"], "awaiting_confirm")
        self.assertNotIn("actions", active)
        stored = json.loads((self.fleet / ".import" / "jobs" / f"{job['id']}.json").read_text())
        stored["status"] = "running"
        stored["phase"] = "copying"
        stored["current_file"] = "checkpoints/new.safetensors"
        stored["current_bytes"] = 4
        stored["current_size"] = 11
        stored["files_done"] = 1
        stored["files_total"] = 3
        (self.fleet / ".import" / "jobs" / f"{job['id']}.json").write_text(json.dumps(stored))
        resumed = ImportService(self.layout, port_in_use=lambda _port: False)
        active = resumed.active_job()
        self.assertEqual(active["status"], "paused")
        self.assertEqual(active["current_file"], "checkpoints/new.safetensors")
        self.assertEqual(active["current_bytes"], 4)
        self.assertTrue(any("manager restarted" in line for line in active["log_tail"]))

    def test_running_container_cannot_be_moved(self):
        self.docker.containers["old-ui"]["status"] = "running"
        self.docker.containers["old-ui"]["inspect"]["State"]["Status"] = "running"
        with self.assertRaises(FleetError):
            self.service.create_job(
                self.docker,
                self.gpus,
                {"container": "old-ui", "gpus": [0], "mode": "move"},
                inline=True,
            )
        self.assertEqual(self.docker.status("old-ui"), "running")


class OverlayHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = FleetLayout(Path(self.tmp.name) / "fleet")
        self.docker = FakeDocker()
        self.service = ImportService(self.layout, port_in_use=lambda _port: False)
        job = {
            "id": "imp-abc123def456",
            "status": "running",
            "phase": "copying",
            "dismissed": False,
            "container": "old-ui",
            "container_id": "a" * 64,
            "name": "portrait",
            "port": 8188,
            "gpus": [0],
            "mode": "copy",
            "custom_nodes": "as-is",
            "cuda_tag": "cu130",
            "workflow": "",
            "mounts": [],
            "created_at": "2026-10-06T00:00:00Z",
            "updated_at": "2026-10-06T00:00:00Z",
            "elapsed_s": 3,
            "current_file": "checkpoints/new.safetensors",
            "current_bytes": 4,
            "current_size": 11,
            "files_done": 1,
            "files_total": 4,
            "bytes_done": 4,
            "bytes_total": 20,
            "speed_current": 0,
            "speed_average": 0,
            "eta_s": None,
            "counts": {"moved": 0, "skipped": 0, "renamed_dupes": 1, "conflicts": 0, "errors": 0},
            "free_bytes": 0,
            "low_space": False,
            "summary": {},
            "actions": [],
            "log_tail": ["duplicate: renamed.safetensors matches existing.safetensors"],
            "error": "",
            "partial": "",
        }
        self.service._jobs[job["id"]] = job
        self.service._put(job)
        self.ctx = ApiContext(
            layout=self.layout,
            docker=self.docker,
            detect_gpus=lambda: [Gpu(0, "GPU0", "8192 MiB")],
            port_in_use=lambda _port: False,
            password=PASSWORD,
            sessions=SessionStore(),
            login_guard=LoginGuard(fail_delay_s=0),
            importer=self.service,
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

    def test_active_job_survives_reload_and_duplicates_stay_read_only(self):
        status, raw, _headers = self._open("GET", "/api/import/active", auth=False)
        self.assertEqual(status, 401)
        status, raw, _headers = self._open("GET", "/api/import/active")
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertEqual(payload["job"]["id"], "imp-abc123def456")
        self.assertEqual(payload["job"]["current_file"], "checkpoints/new.safetensors")
        reloaded = ImportService(self.layout, port_in_use=lambda _port: False)
        self.ctx.importer = reloaded
        status, raw, _headers = self._open("GET", "/api/import/active")
        payload = json.loads(raw)
        self.assertEqual(payload["job"]["status"], "paused")
        self.assertEqual(payload["job"]["current_file"], "checkpoints/new.safetensors")
        self.assertEqual(payload["job"]["files_done"], 1)
        dupes = DuplicatesStore(self.layout.root / ".import" / "duplicates.json")
        dupes.append(
            {
                "incoming_name": "renamed.safetensors",
                "existing_name": "existing.safetensors",
                "existing_path": "/home/ComfyFleet/models/loras/existing.safetensors",
                "size": 4,
                "hash": "ab" * 32,
                "import_id": "imp-abc123def456",
                "date": "2026-10-06T00:00:00Z",
            }
        )
        status, raw, _headers = self._open("GET", "/api/import/duplicates")
        self.assertEqual(json.loads(raw)["duplicates"][0]["incoming_name"], "renamed.safetensors")
        status, raw, _headers = self._open(
            "POST",
            "/api/import/duplicates",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 405)
        status, raw, _headers = self._open("DELETE", "/api/import/duplicates")
        self.assertEqual(status, 405)
        self.assertEqual(len(dupes.read()), 1)

    def _open(self, method, path, data=None, headers=None, auth=True):
        merged = dict(headers or {})
        if auth and "Authorization" not in merged:
            merged["Authorization"] = f"Bearer {PASSWORD}"
        request = urllib.request.Request(self.base + path, data=data, headers=merged, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers.items())


class _FakeProc:
    def __init__(self, data: bytes):
        self.stdout = io.BytesIO(data)
        self.stderr = io.BytesIO(b"")
        self.returncode = 0

    def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class TarDocker(FakeDocker):
    def __init__(self):
        super().__init__()
        self.archives = {}

    def open_container_tar(self, container, path):
        data = self.archives.get((container, path))
        if data is None:
            raise FleetError(f"no archive {container}:{path}")
        return _FakeProc(data)


def _tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 1_700_000_000
            archive.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class UnseenMountTests(unittest.TestCase):
    def test_copy_reads_docker_cp_and_move_is_refused(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fleet = Path(tmp.name) / "fleet"
        layout = FleetLayout(fleet)
        _write(fleet / "models" / "loras" / "existing.safetensors", b"dupe-weights")
        workflow = json.dumps(WORKFLOW).encode()
        docker = TarDocker()
        mounts = [
            {"Type": "bind", "Source": "/host/old/models", "Destination": "/opt/ComfyUI/models"},
            {"Type": "bind", "Source": "/host/old/user", "Destination": "/opt/ComfyUI/user"},
        ]
        docker.add("old-ui", _inspect("old-ui", mounts))
        docker.archives[("old-ui", "/opt/ComfyUI/models")] = _tar(
            {
                "models/checkpoints/new.safetensors": b"new-weights",
                "models/checkpoints/renamed.safetensors": b"dupe-weights",
            }
        )
        docker.archives[("old-ui", "/opt/ComfyUI/user")] = _tar(
            {"user/default/workflows/portrait.json": workflow}
        )
        docker.archives[("old-ui", "/opt/ComfyUI/models/checkpoints/new.safetensors")] = _tar(
            {"new.safetensors": b"new-weights"}
        )
        docker.archives[("old-ui", "/opt/ComfyUI/user/default/workflows/portrait.json")] = _tar(
            {"portrait.json": workflow}
        )
        service = ImportService(layout, port_in_use=lambda _port: False)
        gpus = [Gpu(0, "GPU0", "8192 MiB")]
        with self.assertRaises(FleetError):
            service.create_job(docker, gpus, {"container": "old-ui", "gpus": [0], "mode": "move", "name": "nope"})
        self.assertFalse((fleet / "models" / "checkpoints" / "new.safetensors").exists())
        job = service.create_job(
            docker,
            gpus,
            {"container": "old-ui", "gpus": [0], "mode": "copy", "name": "from-tar"},
            inline=True,
        )
        self.assertEqual(job["status"], "awaiting_confirm", job.get("error"))
        self.assertGreaterEqual(job["summary"]["duplicates"], 1)
        self.assertFalse((fleet / "models" / "checkpoints" / "new.safetensors").exists())
        done = service.start_transfer(job["id"], inline=True)
        self.assertEqual(done["status"], "completed", done.get("error"))
        self.assertEqual((fleet / "models" / "checkpoints" / "new.safetensors").read_bytes(), b"new-weights")
        self.assertFalse((fleet / "models" / "checkpoints" / "renamed.safetensors").exists())
        rows = service.duplicates.read()
        self.assertEqual(rows[0]["incoming_name"], "renamed.safetensors")
        self.assertEqual(rows[0]["existing_name"], "existing.safetensors")
        self.assertIn("duplicate:", service.log_text(job["id"]))
        self.assertTrue((fleet / "files" / "from-tar" / "default_workflow.json").is_file())
        self.assertEqual(docker.status("old-ui"), "exited")

    def test_plan_screen_marks_an_unseen_mount(self):
        docker = TarDocker()
        mounts = [
            {"Type": "bind", "Source": "/host/old/models", "Destination": "/opt/ComfyUI/models"},
        ]
        docker.add("old-ui", _inspect("old-ui", mounts))
        docker.archives[("old-ui", "/opt/ComfyUI/models")] = _tar({})
        service = ImportService(FleetLayout(Path(tempfile.mkdtemp()) / "fleet"))
        draft = service.inspect(docker, "old-ui")
        self.assertFalse(draft["container"]["mounts"][0]["visible"])


class TarStreamTests(unittest.TestCase):
    def test_tar_member_round_trip(self):
        payload = b"weights-from-docker"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as archive:
            info = tarfile.TarInfo("new.safetensors")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        digest, size = hash_tar_member(io.BytesIO(buf.getvalue()))
        self.assertEqual(size, len(payload))
        self.assertEqual(digest, _sha(payload))
        dest = Path(tempfile.mkdtemp()) / "new.safetensors.comfyfleet-partial"
        written = stream_tar_member(io.BytesIO(buf.getvalue()), dest)
        self.assertEqual(written, len(payload))
        self.assertEqual(dest.read_bytes(), payload)


class InspectParseTests(unittest.TestCase):
    def test_parse_mounts_ports_gpu_and_redacts_secrets(self):
        parsed = parse_container_inspect(
            _inspect(
                "studio",
                [
                    {
                        "Type": "bind",
                        "Source": "/data/models",
                        "Destination": "/opt/ComfyUI/models",
                    }
                ],
                env=["VISIBLE=1", "HF_TOKEN=sekret"],
            )
        )
        self.assertEqual(parsed["name"], "studio")
        self.assertEqual(parsed["mounts"][0]["role"], "models")
        self.assertEqual(parsed["ports"], [{"container": 8188, "host": 8288}])
        self.assertEqual(parsed["gpus"], [0])
        self.assertEqual(parsed["env"][1]["value"], "***")


if __name__ == "__main__":
    unittest.main()
