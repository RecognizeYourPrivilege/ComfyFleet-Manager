"""Host port allocation starting at 8188.

Create and start ask :func:`make_port_in_use` for one snapshot of ports that
must not be published:

* TCP ports in ``LISTEN`` on the host (``/proc/net/tcp`` and ``tcp6``)
* host ports already published by Docker, including mappings whose container
  port is 8188 but whose host port is something else

Inside the manager container those listeners are not in this process's
network namespace. The snapshot reads them through the mounted Docker socket.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from pathlib import Path

from comfyfleet.errors import FleetError

PORT_START = 8188
PORT_SCAN = 1000
_LISTEN_STATE = "0A"


def tcp_port_in_use(port: int) -> bool:
    """True when this process cannot bind ``port`` on IPv4.

    This sees listeners in the current network namespace only. Docker port
    mappings that use iptables (userland proxy off) do not bind a socket, and
    the manager container does not share the host netns. Prefer
    :func:`make_port_in_use` for allocation.
    """

    for host in ("0.0.0.0", "127.0.0.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                return True
    return False


def choose_port(
    reserved: set[int],
    *,
    preferred: int | None = None,
    in_use=tcp_port_in_use,
    start: int = PORT_START,
    scan: int = PORT_SCAN,
) -> int:
    """Prefer ``preferred`` when it is free and not reserved by another instance.

    Stopped instances keep their recorded port in ``reserved`` so two created
    containers do not both claim 8188. A recorded port that is busy at start
    time is skipped, and so is every later occupied port, until the next free
    port from ``start``.
    """

    if preferred is not None and preferred not in reserved and not in_use(preferred):
        return preferred
    for port in range(start, start + scan):
        if port in reserved or in_use(port):
            continue
        return port
    raise FleetError(
        f"no free host port in {start}..{start + scan - 1}. "
        "Stop another listener or instance and retry."
    )


def manager_mode() -> bool:
    """True when this process is the manager container (set by its image)."""

    return os.environ.get("COMFYFLEET_MANAGER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_proc_net_tcp(text: str) -> set[int]:
    """TCP ports in LISTEN state from ``/proc/net/tcp`` or ``tcp6`` text.

    Established and time-wait rows are ignored so ephemeral client ports are
    not treated as allocations. Any local address counts, including
    ``127.0.0.1``, because publishing ``0.0.0.0`` on that port would fail.
    """

    found: set[int] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[3] != _LISTEN_STATE:
            continue
        _addr, sep, port_hex = fields[1].rpartition(":")
        if not sep:
            continue
        try:
            port = int(port_hex, 16)
        except ValueError:
            continue
        if 1 <= port <= 65535:
            found.add(port)
    return found


def read_proc_net(root: Path) -> str:
    """Concatenate ``tcp`` and ``tcp6`` under ``root``. Missing files are skipped."""

    chunks: list[str] = []
    for name in ("tcp", "tcp6"):
        path = root / name
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def load_occupied_ports(
    docker,
    *,
    manager: bool | None = None,
    proc_root: Path | None = None,
) -> set[int]:
    """Host ports that are published by Docker or already listening.

    ``docker`` is the same client create/start use (the host engine via the
    mounted socket when this process is the manager).
    """

    in_manager = manager_mode() if manager is None else manager
    published = set(docker.published_host_ports())
    if in_manager:
        try:
            tables = docker.host_tcp_tables()
        except FleetError as exc:
            raise FleetError(
                "cannot read which TCP ports are listening on the host. "
                "The manager checks Docker-published ports and, through the "
                "mounted Docker socket, the host network namespace "
                "(`docker run --network host`). "
                f"{exc}"
            ) from exc
    else:
        tables = read_proc_net(proc_root or Path("/proc/net"))
    return published | parse_proc_net_tcp(tables)


def make_port_in_use(
    docker,
    *,
    manager: bool | None = None,
    proc_root: Path | None = None,
    bind: Callable[[int], bool] | None = None,
) -> Callable[[int], bool]:
    """One snapshot for a create or start, then a per-port check.

    The snapshot is taken once so a scan from 8188 does not run ``docker ps``
    (or a host-network probe) on every candidate. On the host CLI, ``bind``
    is an extra check for a listener that appeared in this netns. In the
    manager, ``bind`` is not used: it would inspect the container netns.
    """

    in_manager = manager_mode() if manager is None else manager
    occupied = load_occupied_ports(docker, manager=in_manager, proc_root=proc_root)
    if bind is None:
        bind = (lambda _port: False) if in_manager else tcp_port_in_use

    def in_use(port: int) -> bool:
        return port in occupied or bind(port)

    return in_use
