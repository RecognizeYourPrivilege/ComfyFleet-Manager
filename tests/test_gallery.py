"""Gallery discovery, ordering, path safety, and delete."""

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
from comfyfleet.gallery import (
    delete_gallery_file,
    gif_first_frame,
    list_gallery,
    open_gallery_file,
    open_gallery_thumb,
    poster_bytes,
    safe_output_dir,
)
from comfyfleet.errors import FleetError
from comfyfleet.http_api import ApiContext, make_server
from comfyfleet.paths import FleetLayout


PASSWORD = "test-password"
PNG = b"\x89PNG\r\n\x1a\n-gallery-image"
GIF_ONE = bytes.fromhex(
    "47494638396101000100800000ffffff00000021f90400000000002c00000000010001000002024401003b"
)


def _two_frame_gif() -> bytes:
    body = GIF_ONE[:-1]
    second = GIF_ONE[GIF_ONE.index(b"\x21") : -1]
    return body + second + b"\x3b"


def _webp(animated: bool) -> bytes:
    flags = 0x02 if animated else 0x00
    chunk = b"VP8X" + (10).to_bytes(4, "little") + bytes([flags]) + b"\x00" * 9
    payload = b"WEBP" + chunk
    return b"RIFF" + len(payload).to_bytes(4, "little") + payload


class Docker:
    def __init__(self, statuses=None):
        self.statuses = dict(statuses or {})

    def status(self, name):
        return self.statuses.get(name)

    def running_names(self):
        return [name for name, status in self.statuses.items() if status == "running"]


def _write(path: Path, data: bytes, mtime: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _metadata(layout: FleetLayout, name: str, port: int) -> None:
    payload = {
        "name": name,
        "port": port,
        "gpus": [0],
        "image": "comfyfleet:cu130",
        "workflow_host_path": str(layout.workflow_file(name)),
    }
    path = layout.metadata_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class GalleryDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = FleetLayout(Path(self.tmp.name) / "home" / "ComfyFleet")
        self.docker = Docker({"portrait": "running", "background": "exited"})
        portrait = self.layout.output_dir("portrait")
        background = self.layout.output_dir("background")
        _write(portrait / "older.png", PNG, 1_000)
        _write(portrait / "newer.png", PNG + b"-new", 3_000)
        _write(portrait / "clips" / "shot.mp4", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16, 2_000)
        _write(portrait / "anim.gif", _two_frame_gif(), 2_500)
        _write(portrait / "still.webp", _webp(False), 1_500)
        _write(portrait / "motion.webp", _webp(True), 1_600)
        _write(portrait / "notes.txt", b"nope", 9_000)
        _write(portrait / ".hidden.png", PNG, 9_000)
        _write(self.layout.input_dir("portrait") / "ref.png", b"input-secret", 9_000)
        _write(background / "plate.jpg", b"\xff\xd8\xff\xd9", 4_000)
        _write(self.layout.root / "models" / "checkpoints" / "x.png", PNG, 9_000)
        _write(self.layout.files / "Notes" / "output" / "skip.png", PNG, 9_000)
        _metadata(self.layout, "portrait", 8188)
        _metadata(self.layout, "background", 8189)
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        self.outside_file = outside / "secret.png"
        self.outside_file.write_bytes(b"SECRET-OUTSIDE")
        (portrait / "link.png").symlink_to(self.outside_file)
        outside_dir = outside / "linked-dir"
        outside_dir.mkdir()
        (outside_dir / "nested.png").write_bytes(b"NESTED-SECRET")
        (portrait / "linked").symlink_to(outside_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovers_output_dirs_and_orders_newest_first(self):
        page = list_gallery(self.layout, self.docker)
        paths = [(item["instance"], item["path"], item["kind"]) for item in page["items"]]
        self.assertEqual(
            paths,
            [
                ("background", "plate.jpg", "image"),
                ("portrait", "newer.png", "image"),
                ("portrait", "anim.gif", "animated"),
                ("portrait", "clips/shot.mp4", "video"),
                ("portrait", "motion.webp", "animated"),
                ("portrait", "still.webp", "image"),
                ("portrait", "older.png", "image"),
            ],
        )
        self.assertEqual(page["total"], 7)
        self.assertEqual([item["name"] for item in page["instances"]], ["background", "portrait"])
        portrait = next(item for item in page["items"] if item["path"] == "newer.png")
        self.assertEqual(portrait["port"], 8188)
        self.assertEqual(portrait["status"], "running")
        self.assertNotIn(str(self.layout.root), json.dumps(page))
        self.assertNotIn("SECRET-OUTSIDE", json.dumps(page))
        self.assertNotIn("notes.txt", json.dumps(page))
        self.assertNotIn("ref.png", json.dumps(page))

    def test_filter_and_pagination(self):
        filtered = list_gallery(self.layout, self.docker, instance="portrait", offset=1, limit=2)
        self.assertEqual(
            [item["path"] for item in filtered["items"]],
            ["anim.gif", "clips/shot.mp4"],
        )
        self.assertEqual(filtered["total"], 6)
        self.assertEqual(len(filtered["instances"]), 2)
        empty = list_gallery(self.layout, self.docker, instance="missing")
        self.assertEqual(empty["items"], [])
        self.assertEqual(empty["total"], 0)
        with self.assertRaises(FleetError):
            list_gallery(self.layout, self.docker, instance="../portrait")
        with self.assertRaises(FleetError):
            list_gallery(self.layout, self.docker, offset=-1)

    def test_output_symlink_is_not_discovered(self):
        rogue = self.layout.output_dir("rogue")
        rogue.parent.mkdir(parents=True, exist_ok=True)
        outside = Path(self.tmp.name) / "rogue-output"
        outside.mkdir()
        (outside / "leaked.png").write_bytes(PNG)
        rogue.symlink_to(outside)
        self.assertIsNone(safe_output_dir(self.layout, "rogue"))
        page = list_gallery(self.layout, self.docker)
        self.assertNotIn("rogue", [item["name"] for item in page["instances"]])
        self.assertNotIn("leaked.png", json.dumps(page))

    def test_symlink_and_traversal_are_refused(self):
        with self.assertRaises(FleetError) as listed:
            open_gallery_file(self.layout, "portrait", "link.png")
        self.assertIn("symlink", str(listed.exception))
        with self.assertRaises(FleetError):
            delete_gallery_file(self.layout, "portrait", "link.png")
        self.assertEqual(self.outside_file.read_bytes(), b"SECRET-OUTSIDE")
        self.assertTrue((self.layout.output_dir("portrait") / "link.png").is_symlink())
        for relative in (
            "../input/ref.png",
            "..",
            "/etc/passwd",
            "clips/../../input/ref.png",
            "linked/nested.png",
            "notes.txt",
        ):
            with self.assertRaises(FleetError):
                delete_gallery_file(self.layout, "portrait", relative)
        self.assertTrue((self.layout.input_dir("portrait") / "ref.png").is_file())
        self.assertEqual((Path(self.tmp.name) / "outside" / "linked-dir" / "nested.png").read_bytes(), b"NESTED-SECRET")

    def test_delete_removes_only_that_output_file(self):
        deleted = delete_gallery_file(self.layout, "portrait", "clips/shot.mp4")
        self.assertEqual(deleted["deleted"]["path"], "clips/shot.mp4")
        self.assertFalse((self.layout.output_dir("portrait") / "clips" / "shot.mp4").exists())
        self.assertTrue((self.layout.output_dir("background") / "plate.jpg").is_file())
        page = list_gallery(self.layout, self.docker, instance="portrait")
        self.assertNotIn("clips/shot.mp4", [item["path"] for item in page["items"]])
        with self.assertRaises(FleetError):
            delete_gallery_file(self.layout, "portrait", "clips/shot.mp4")

    def test_gif_poster_is_the_first_frame_and_video_poster_is_not_the_video(self):
        gif = self.layout.output_dir("portrait") / "anim.gif"
        frame = gif_first_frame(gif)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.count(b"\x2c"), 1)
        self.assertEqual(gif.read_bytes().count(b"\x2c"), 2)
        data, content_type = poster_bytes(gif, "animated")
        self.assertEqual(content_type, "image/gif")
        self.assertEqual(data.count(b"\x2c"), 1)
        video = self.layout.output_dir("portrait") / "clips" / "shot.mp4"
        with mock.patch("comfyfleet.gallery.shutil.which", return_value=None):
            poster, poster_type = poster_bytes(video, "video")
        self.assertEqual(poster_type, "image/png")
        self.assertTrue(poster.startswith(b"\x89PNG"))
        self.assertNotEqual(poster, video.read_bytes())
        self.assertFalse(poster.startswith(b"\x00\x00\x00\x18"))

    def test_ffmpeg_poster_uses_the_open_descriptor_not_a_shell(self):
        video = self.layout.output_dir("portrait") / "clips" / "shot.mp4"
        seen = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = list(argv)
            seen["kwargs"] = kwargs
            Path(argv[-1]).write_bytes(b"\xff\xd8\xff\xd9")
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("comfyfleet.gallery.shutil.which", return_value="/usr/bin/ffmpeg"), mock.patch(
            "comfyfleet.gallery.subprocess.run", side_effect=fake_run
        ):
            data, content_type = poster_bytes(video, "video")
        self.assertEqual(content_type, "image/jpeg")
        self.assertTrue(data.startswith(b"\xff\xd8"))
        argv = seen["argv"]
        self.assertIsInstance(argv, list)
        self.assertFalse(seen["kwargs"].get("shell", False))
        source = argv[argv.index("-i") + 1]
        self.assertTrue(source.startswith("/proc/self/fd/"))
        self.assertNotIn(str(video), argv)
        self.assertIn(seen["kwargs"]["pass_fds"][0], range(0, 1_000_000))


class GalleryHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = FleetLayout(Path(self.tmp.name) / "home" / "ComfyFleet")
        self.docker = Docker({"portrait": "running"})
        output = self.layout.output_dir("portrait")
        _write(output / "a.png", PNG, 2_000)
        _write(output / "b.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32, 3_000)
        _write(output / "c.gif", _two_frame_gif(), 1_000)
        _metadata(self.layout, "portrait", 8188)
        outside = Path(self.tmp.name) / "secret.png"
        outside.write_bytes(b"DO-NOT-LEAK")
        (output / "escape.png").symlink_to(outside)
        self.outside = outside
        self.ctx = ApiContext(
            layout=self.layout,
            docker=self.docker,
            detect_gpus=lambda: [],
            port_in_use=lambda _port: False,
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
        request = urllib.request.Request(self.base + path, data=data, headers=merged, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers.items())

    def test_auth_gates_list_media_and_delete(self):
        status, raw, _headers = self._open("GET", "/api/gallery", auth=False)
        self.assertEqual(status, 401)
        self.assertNotIn(b"a.png", raw)
        status, raw, _headers = self._open("GET", "/api/gallery/media?instance=portrait&path=a.png", auth=False)
        self.assertEqual(status, 401)
        self.assertNotEqual(raw, PNG)
        status, raw, _headers = self._open(
            "POST",
            "/api/gallery/delete",
            data=json.dumps({"instance": "portrait", "path": "a.png"}).encode(),
            headers={"Content-Type": "application/json"},
            auth=False,
        )
        self.assertEqual(status, 401)
        self.assertTrue((self.layout.output_dir("portrait") / "a.png").is_file())

    def test_list_media_range_thumb_and_delete(self):
        status, raw, _headers = self._open("GET", "/api/gallery?limit=1&offset=0")
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["items"][0]["path"], "b.mp4")
        self.assertEqual(payload["total"], 3)
        self.assertNotIn(str(self.layout.root), raw.decode("utf-8"))

        status, raw, headers = self._open("GET", "/api/gallery/media?instance=portrait&path=a.png")
        self.assertEqual(status, 200)
        self.assertEqual(raw, PNG)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertIn("inline", headers["Content-Disposition"])

        status, raw, headers = self._open(
            "GET",
            "/api/gallery/media?instance=portrait&path=b.mp4&download=1",
            headers={"Range": "bytes=0-7"},
        )
        self.assertEqual(status, 206)
        self.assertEqual(raw, b"\x00\x00\x00\x18ftyp")
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-7/"))
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn("b.mp4", headers["Content-Disposition"])

        status, raw, headers = self._open("GET", "/api/gallery/thumb?instance=portrait&path=b.mp4")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("image/"))
        self.assertFalse(headers["Content-Type"].startswith("video/"))
        self.assertNotEqual(raw, (self.layout.output_dir("portrait") / "b.mp4").read_bytes())
        self.assertFalse(raw.startswith(b"\x00\x00\x00\x18"))

        status, raw, headers = self._open("GET", "/api/gallery/thumb?instance=portrait&path=c.gif")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/gif")
        self.assertEqual(raw.count(b"\x2c"), 1)

        status, raw, _headers = self._open(
            "GET",
            "/api/gallery/media?instance=portrait&path=" + "..%2Finput%2Fnope.png",
        )
        self.assertEqual(status, 400)
        status, raw, _headers = self._open("GET", "/api/gallery/media?instance=portrait&path=escape.png")
        self.assertEqual(status, 400)
        self.assertIn(b"symlink", raw)
        self.assertEqual(self.outside.read_bytes(), b"DO-NOT-LEAK")

        body = json.dumps({"instance": "portrait", "path": "escape.png"}).encode()
        status, raw, _headers = self._open(
            "POST",
            "/api/gallery/delete",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(self.outside.read_bytes(), b"DO-NOT-LEAK")

        status, _raw, _headers = self._open("DELETE", "/api/gallery/delete")
        self.assertEqual(status, 405)
        self.assertTrue((self.layout.output_dir("portrait") / "a.png").is_file())

        status, raw, _headers = self._open(
            "POST",
            "/api/gallery/delete",
            data=json.dumps({"instance": "portrait", "path": "a.png"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw)
        self.assertFalse((self.layout.output_dir("portrait") / "a.png").exists())
        status, raw, _headers = self._open("GET", "/api/gallery")
        names = [item["path"] for item in json.loads(raw.decode("utf-8"))["items"]]
        self.assertNotIn("a.png", names)
        self.assertNotIn("escape.png", names)


class GalleryThumbFdTests(unittest.TestCase):
    def test_image_thumb_stream_is_closed_by_the_caller_contract(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        layout = FleetLayout(Path(tmp.name) / "ComfyFleet")
        _write(layout.output_dir("portrait") / "a.png", PNG, 10)
        opened = open_gallery_thumb(layout, "portrait", "a.png")
        self.assertFalse(isinstance(opened, tuple))
        try:
            os.lseek(opened.fd, 0, os.SEEK_SET)
            self.assertEqual(os.read(opened.fd, 100), PNG)
        finally:
            os.close(opened.fd)


if __name__ == "__main__":
    unittest.main()
