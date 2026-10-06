import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfyfleet.docker import parse_published_ports
from comfyfleet.ports import (
    PORT_START,
    choose_port,
    make_port_in_use,
    parse_proc_net_tcp,
    tcp_port_in_use,
)


def _listen(port: int, addr: str = "00000000") -> str:
    return (
        f"   0: {addr}:{port:04X} 00000000:0000 0A "
        "00000000:00000000 00:00000000 00000000 0 0 1 1 0 100 0 0 10 0\n"
    )


def _established(port: int) -> str:
    return (
        f"   1: 0100007F:{port:04X} 0100007F:1F90 01 "
        "00000000:00000000 00:00000000 00000000 0 0 1 1 0 100 0 0 10 0\n"
    )


class _Engine:
    def __init__(self, published: set[int], tables: str = ""):
        self._published = set(published)
        self._tables = tables
        self.calls: list[str] = []

    def published_host_ports(self) -> set[int]:
        self.calls.append("published")
        return set(self._published)

    def host_tcp_tables(self) -> str:
        self.calls.append("tables")
        return self._tables


class PortTests(unittest.TestCase):
    def test_starts_at_8188(self):
        self.assertEqual(choose_port(set(), in_use=lambda _port: False), PORT_START)

    def test_skips_reserved_and_in_use(self):
        port = choose_port({8188}, in_use=lambda port: port == 8189)
        self.assertEqual(port, 8190)

    def test_prefers_recorded_port_when_free(self):
        port = choose_port(set(), preferred=8191, in_use=lambda port: port == 8188)
        self.assertEqual(port, 8191)

    def test_recorded_port_busy_moves_to_next_free(self):
        port = choose_port({8190}, preferred=8188, in_use=lambda port: port == 8188)
        self.assertEqual(port, 8189)

    def test_does_not_assign_listener_outside_metadata(self):
        port = choose_port(set(), in_use=lambda port: port == 8188)
        self.assertEqual(port, 8189)

    def test_8188_free_in_metadata_but_8189_listening_allocates_8190(self):
        """8188 is not reserved by fleet metadata.

        It is listening on the host (outside metadata) and 8189 is the host
        port of another ComfyUI (``8189->8188``). The next free port is 8190.
        The container-side 8188 in that mapping is not itself a host port.
        """

        published = parse_published_ports(
            "0.0.0.0:8189->8188/tcp, [::]:8189->8188/tcp\n"
        )
        self.assertEqual(published, {8189})
        listening = parse_proc_net_tcp(_listen(8188) + _established(40000))
        self.assertEqual(listening, {8188})
        occupied = listening | published
        self.assertEqual(choose_port(set(), in_use=lambda port: port in occupied), 8190)

    def test_only_8189_published_keeps_8188(self):
        published = parse_published_ports("0.0.0.0:8189->8188/tcp\n")
        self.assertEqual(choose_port(set(), in_use=lambda port: port in published), 8188)

    def test_proc_parser_keeps_localhost_listen_and_ipv6(self):
        text = _listen(8188, addr="0100007F") + (
            "   2: 00000000000000000000000000000000:1FFD "
            "00000000000000000000000000000000:0000 0A "
            "00000000:00000000 00:00000000 00000000 0 0 1 1 0 100 0 0 10 0\n"
        )
        self.assertEqual(parse_proc_net_tcp(text), {8188, 8189})

    def test_manager_probe_skips_host_listener_and_published_neighbor(self):
        docker = _Engine(published={8189}, tables=_listen(8188, addr="0100007F"))
        with mock.patch("comfyfleet.ports.tcp_port_in_use", side_effect=AssertionError("bind")):
            in_use = make_port_in_use(docker, manager=True)
            self.assertEqual(choose_port(set(), in_use=in_use), 8190)
        self.assertEqual(docker.calls, ["published", "tables"])

    def test_host_probe_reads_proc_and_docker_without_a_helper_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tcp").write_text(_listen(8188), encoding="utf-8")
            (root / "tcp6").write_text(
                "   0: 00000000000000000000000000000000:1F91 "
                "00000000000000000000000000000000:0000 0A "
                "00000000:00000000 00:00000000 00000000 0 0 1 1 0 100 0 0 10 0\n",
                encoding="utf-8",
            )
            docker = _Engine(published={8189}, tables="unused")
            in_use = make_port_in_use(
                docker,
                manager=False,
                proc_root=root,
                bind=lambda _port: False,
            )
        self.assertEqual(docker.calls, ["published"])
        self.assertTrue(in_use(8188))
        self.assertTrue(in_use(8189))
        self.assertTrue(in_use(8081))
        self.assertEqual(choose_port(set(), in_use=in_use), 8190)

    def test_host_bind_backstop_marks_a_port_the_tables_missed(self):
        docker = _Engine(published=set(), tables="")
        in_use = make_port_in_use(
            docker,
            manager=False,
            proc_root=Path("/no/such/proc"),
            bind=lambda port: port == 8188,
        )
        self.assertEqual(choose_port(set(), in_use=in_use), 8189)

    def test_tcp_port_in_use_sees_a_real_listener(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            self.assertTrue(tcp_port_in_use(port))
        self.assertFalse(tcp_port_in_use(port))


if __name__ == "__main__":
    unittest.main()
