"""Manager-proxied shell. The browser talks to the control server only."""

import base64
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from comfyfleet.auth import LoginGuard, SessionStore
from comfyfleet.control import create_instance, start_instance
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu
from comfyfleet.http_api import ApiContext, make_server
from comfyfleet.paths import FleetLayout
from comfyfleet.terminal import accept_value, terminal_exec_argv

PASSWORD = "test-password"
ECHO = """
import os, sys, termios
fd = 0
attr = termios.tcgetattr(fd)
attr[3] = attr[3] & ~(termios.ECHO | termios.ICANON)
termios.tcsetattr(fd, termios.TCSANOW, attr)
os.write(1, b"ready\\n")
buf = b""
while b"\\n" not in buf:
    chunk = os.read(0, 1024)
    if not chunk:
        break
    buf += chunk
os.write(1, b"out:" + buf)
"""


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
        self.containers[name]["status"] = "exited"

    def kill(self, name):
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


def _workflow(directory: Path) -> Path:
    path = directory / "Portrait.json"
    path.write_text(
        json.dumps({"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "version": 0.4}),
        encoding="utf-8",
    )
    return path


class TerminalArgvTests(unittest.TestCase):
    def test_exec_is_fixed_and_rejects_injection(self):
        argv = terminal_exec_argv("portrait")
        self.assertEqual(argv, ["docker", "exec", "-it", "portrait", "/bin/bash"])
        self.assertNotIn("docker.sock", " ".join(argv))
        with self.assertRaises(FleetError):
            terminal_exec_argv("portrait;rm")

    def test_accept_value_matches_the_rfc_sample(self):
        self.assertEqual(accept_value("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")


class TerminalProxyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.layout = FleetLayout(root / "home")
        self.docker = FakeDocker()
        self.execs = []
        self.ctx = ApiContext(
            layout=self.layout,
            docker=self.docker,
            detect_gpus=lambda: [Gpu(0, "GPU0", "12288 MiB")],
            port_in_use=lambda _port: False,
            ui_dir=root / "ui",
            password=PASSWORD,
            sessions=SessionStore(),
            login_guard=LoginGuard(fail_delay_s=0),
            terminal_argv=self._argv,
        )
        sources = root / "src"
        sources.mkdir()
        create_instance(
            _workflow(sources),
            layout=self.layout,
            docker=self.docker,
            gpus=[Gpu(0, "GPU0", "12288 MiB")],
            gpu="0",
            port_in_use=lambda _port: False,
        )
        self.httpd = make_server("127.0.0.1", 0, self.ctx)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _argv(self, name):
        self.execs.append(name)
        return [sys.executable, "-c", ECHO]

    def test_stopped_instance_does_not_exec(self):
        status, _body = _upgrade(self.port, "/api/instances/portrait/terminal", cookie=_login(self.port))
        self.assertEqual(status, 400)
        self.assertEqual(self.execs, [])

    def test_anonymous_upgrade_is_rejected(self):
        start_instance(
            "portrait",
            layout=self.layout,
            docker=self.docker,
            gpus=[Gpu(0, "GPU0", "12288 MiB")],
            port_in_use=lambda _port: False,
        )
        status, body = _upgrade(self.port, "/api/instances/portrait/terminal")
        self.assertEqual(status, 401)
        self.assertIn(b"unauthorized", body)
        self.assertEqual(self.execs, [])

    def test_websocket_reaches_only_the_named_instance(self):
        start_instance(
            "portrait",
            layout=self.layout,
            docker=self.docker,
            gpus=[Gpu(0, "GPU0", "12288 MiB")],
            port_in_use=lambda _port: False,
        )
        sock, pending = _open_socket(self.port, "/api/instances/portrait/terminal", cookie=_login(self.port))
        try:
            sock.settimeout(5)
            seen = _collect(sock, pending, b"ready")
            _send_text(sock, "hi\n")
            seen += _collect(sock, b"", b"out:hi")
        finally:
            sock.close()
        self.assertEqual(self.execs, ["portrait"])
        self.assertIn(b"out:hi", seen)


def _login(port: int) -> str:
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps({"password": PASSWORD}).encode("utf-8")
    connection.request(
        "POST",
        "/api/login",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = connection.getresponse()
    response.read()
    cookie = (response.getheader("Set-Cookie") or "").split(";", 1)[0]
    connection.close()
    return cookie


def _upgrade(port: int, path: str, cookie: str | None = None):
    sock, _pending = _open_socket(port, path, cookie=cookie, expect=None)
    sock.close()
    return _upgrade.status, _upgrade.body


def _collect(sock: socket.socket, pending: bytes, marker: bytes) -> bytes:
    seen = b""
    buf = bytearray(pending)
    while marker not in seen:
        while True:
            frame = _take_frame(buf)
            if frame is None:
                break
            seen += frame
        if marker in seen:
            break
        piece = sock.recv(4096)
        if not piece:
            break
        buf.extend(piece)
    return seen


def _take_frame(buf: bytearray) -> bytes | None:
    if len(buf) < 2:
        return None
    length = buf[1] & 0x7F
    offset = 2
    if length == 126:
        if len(buf) < 4:
            return None
        length = struct.unpack("!H", bytes(buf[2:4]))[0]
        offset = 4
    elif length == 127:
        if len(buf) < 10:
            return None
        length = struct.unpack("!Q", bytes(buf[2:10]))[0]
        offset = 10
    if len(buf) < offset + length:
        return None
    payload = bytes(buf[offset : offset + length])
    del buf[: offset + length]
    if (buf and False):
        return payload
    return payload


def _open_socket(port: int, path: str, cookie: str | None = None, expect=101):
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    lines = [
        f"GET {path} HTTP/1.1",
        "Host: 127.0.0.1",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if cookie:
        lines.append(f"Cookie: {cookie}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    text = head.decode("iso-8859-1")
    status = int(text.split(" ", 2)[1])
    content_length = 0
    for line in text.split("\r\n")[1:]:
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
    while len(rest) < content_length:
        chunk = sock.recv(4096)
        if not chunk:
            break
        rest += chunk
    body = rest[:content_length]
    pending = rest[content_length:]
    _upgrade.status = status
    _upgrade.body = body
    if expect is not None and status != expect:
        sock.close()
        raise AssertionError(f"{status} {body[:300]!r}")
    sock.settimeout(5)
    return sock, pending


def _send_text(sock: socket.socket, text: str) -> None:
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    header = bytes((0x81, 0x80 | len(payload))) + mask
    sock.sendall(header + masked)


def _read_payload(sock: socket.socket) -> bytes:
    header = _exact(sock, 2)
    opcode = header[0] & 0x0F
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _exact(sock, 8))[0]
    payload = _exact(sock, length) if length else b""
    if opcode == 0x8:
        return b""
    return payload


def _exact(sock: socket.socket, count: int) -> bytes:
    buf = b""
    while len(buf) < count:
        piece = sock.recv(count - len(buf))
        if not piece:
            raise AssertionError("socket closed")
        buf += piece
    return buf


if __name__ == "__main__":
    unittest.main()
