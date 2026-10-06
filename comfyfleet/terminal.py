"""Web terminal proxy.

The browser opens a WebSocket to the manager. This process runs
``docker exec -it <instance> /bin/bash`` and copies bytes between the
socket and that PTY. The Docker socket stays on the manager. It is not
sent to the browser, and the browser cannot choose the command.
"""

from __future__ import annotations

import json
import os
import pty
import signal
import socket
import struct
import subprocess
import threading
from comfyfleet.errors import FleetError
from comfyfleet.naming import is_instance_name

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_FRAME = 1_000_000


def terminal_exec_argv(name: str) -> list[str]:
    """Fixed ``docker exec`` for one fleet instance. No shell, no extra args."""

    if not is_instance_name(name):
        raise FleetError(f"invalid instance name {name!r}")
    return ["docker", "exec", "-it", name, "/bin/bash"]


def _spawn_pty(argv: list[str]):
    """Start ``argv`` on a new PTY. No shell. The parent keeps the master fd."""

    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        os.close(master)
        os.close(slave)
        raise FleetError(f"terminal failed to start: {exc}") from exc
    os.close(slave)
    return proc, master


def accept_value(key: str) -> str:
    import base64
    import hashlib

    digest = hashlib.sha1((key.strip() + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def bridge_exec(sock: socket.socket, argv: list[str]) -> None:
    """Attach ``sock`` (already upgraded) to a PTY running ``argv``."""

    if not argv or not argv[0]:
        raise FleetError("terminal command is empty")
    proc, fd = _spawn_pty(argv)
    pid = proc.pid
    lock = threading.Lock()
    reader = threading.Thread(
        target=_pump_output,
        args=(sock, fd, lock),
        name="comfyfleet-terminal",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            frame = read_frame(sock)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x8:
                break
            if opcode == 0x9:
                with lock:
                    write_frame(sock, payload, opcode=0xA)
                continue
            if opcode == 0xA:
                continue
            if opcode == 0x1:
                text = payload.decode("utf-8", "replace")
                if _consume_resize(fd, pid, text):
                    continue
                _write_all(fd, payload)
            elif opcode == 0x2:
                _write_all(fd, payload)
    except (ConnectionError, OSError):
        return
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def read_frame(sock: socket.socket) -> tuple[int, bytes] | None:
    """Read one client frame. Client frames must be masked."""

    try:
        header = _recv_exact(sock, 2)
    except ConnectionError:
        return None
    opcode = header[0] & 0x0F
    masked = header[1] & 0x80
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if length > _MAX_FRAME:
        raise FleetError("terminal frame is too large")
    mask = _recv_exact(sock, 4) if masked else b""
    if not masked:
        raise FleetError("terminal client frames must be masked")
    payload = _recv_exact(sock, length) if length else b""
    payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def write_frame(sock: socket.socket, payload: bytes, *, opcode: int = 0x2) -> None:
    """Write one server frame. Server frames are not masked."""

    first = 0x80 | (opcode & 0x0F)
    size = len(payload)
    if size < 126:
        header = bytes((first, size))
    elif size < 65536:
        header = bytes((first, 126)) + struct.pack("!H", size)
    else:
        header = bytes((first, 127)) + struct.pack("!Q", size)
    sock.sendall(header + payload)


def _pump_output(sock: socket.socket, fd: int, lock: threading.Lock) -> None:
    try:
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            with lock:
                try:
                    write_frame(sock, data, opcode=0x2)
                except OSError:
                    break
    finally:
        with lock:
            try:
                write_frame(sock, b"", opcode=0x8)
            except OSError:
                pass


def _consume_resize(fd: int, pid: int, text: str) -> bool:
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("type") != "resize":
        return False
    cols = payload.get("cols")
    rows = payload.get("rows")
    if isinstance(cols, bool) or isinstance(rows, bool):
        return True
    if isinstance(cols, int) and isinstance(rows, int) and 1 <= cols <= 500 and 1 <= rows <= 500:
        import fcntl
        import termios

        packed = struct.pack("HHHH", rows, cols, 0, 0)
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
            os.kill(pid, signal.SIGWINCH)
        except OSError:
            return True
    return True


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        try:
            piece = sock.recv(remaining)
        except ConnectionError as exc:
            raise ConnectionError from exc
        if not piece:
            raise ConnectionError("terminal closed")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)
