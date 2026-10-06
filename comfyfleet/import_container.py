"""Import an existing Docker container into the ComfyFleet host layout.

The manager reads mounts, ports, GPU requests, and env from ``docker inspect``.
Models always land in the shared ``/home/ComfyFleet/models`` tree, in the same
subfolder they came from. Other mounts map into ``files/<name>/...`` and
``custom_nodes_<name>``. The old container is not stopped or removed here.

Nothing is written outside ``/home/ComfyFleet`` except deleting a source file
on Move, and only when that file is under one of the source container's mount
paths and the destination hash has been checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from comfyfleet.control import authorize, create_instance
from comfyfleet.errors import FleetError
from comfyfleet.naming import is_instance_name, sanitize_stem
from comfyfleet.ownership import create_new_host_dirs, resolve_owner_ids
from comfyfleet.paths import MODEL_SUBDIRS, FleetLayout

PARTIAL_SUFFIX = ".comfyfleet-partial"
_CHUNK = 1024 * 1024
_LOW_FREE = 1024 ** 3
_MARGIN = 256 * 1024 * 1024
_FILE_MARGIN = 32 * 1024 * 1024
_SECRET = ("PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")
_ROLES = {
    "models",
    "input",
    "output",
    "temp",
    "custom_nodes",
    "workflows",
    "wildcards",
    "user",
    "skip",
}
_LIVE = {"hashing", "awaiting_confirm", "running", "paused"}
_TERMINAL = {"completed", "cancelled", "failed"}
_MODEL_SUBDIR_NAMES = {name.lower() for name in MODEL_SUBDIRS}


class ImportCancelled(Exception):
    """The operator cancelled the job. Partial files are already removed."""


class Gate:
    """Pause blocks the worker. Cancel unblocks it and raises at the next check."""

    def __init__(self) -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._lock = threading.Lock()
        self.cancelled = False

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
        self._resume.set()

    def checkpoint(self) -> None:
        self._resume.wait()
        with self._lock:
            if self.cancelled:
                raise ImportCancelled()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()


class HashCache:
    """Size and mtime cache so a second scan does not read the file again."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        self._load()

    def get(self, path: Path, size: int, mtime_ns: int) -> str | None:
        row = self._entries.get(str(path))
        if not row:
            return None
        if int(row.get("size", -1)) != size or int(row.get("mtime_ns", -1)) != mtime_ns:
            return None
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return None
        return digest

    def put(self, path: Path, size: int, mtime_ns: int, digest: str) -> None:
        self._entries[str(path)] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "sha256": digest,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.path, {"files": self._entries})

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        files = payload.get("files") if isinstance(payload, dict) else None
        if isinstance(files, dict):
            self._entries = {str(key): value for key, value in files.items() if isinstance(value, dict)}


class DuplicatesStore:
    """Append-only list. There is no replace or delete."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> list[dict]:
        with self._lock:
            return list(self._load())

    def append(self, entry: dict) -> None:
        required = ("incoming_name", "existing_name", "existing_path", "size", "hash", "import_id", "date")
        missing = [key for key in required if key not in entry]
        if missing:
            raise FleetError(f"duplicate entry is missing {', '.join(missing)}")
        with self._lock:
            rows = self._load()
            rows.append({key: entry[key] for key in required})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.chmod(self.path.stat().st_mode | stat.S_IWUSR)
            _atomic_json(self.path, rows)
            self.path.chmod(0o444)

    def _load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FleetError(f"cannot read duplicates list: {exc}") from exc
        if not isinstance(payload, list):
            raise FleetError("duplicates list is invalid")
        return [row for row in payload if isinstance(row, dict)]


def classify_model(rel: str, digest: str, size: int, index: dict) -> dict:
    """One of the four model outcomes. ``index`` is not modified.

    ``skip_same`` is the same relative name and the same hash.
    ``skip_duplicate`` is the same hash under a different name, including a
    previous ``name (imported).ext`` whose bytes match.
    ``conflict`` keeps the existing name and writes ``name (imported).ext``.
    ``transfer`` is a new file.
    """

    rel = _rel_posix(rel)
    same_name = index["by_rel"].get(rel)
    if same_name is not None and same_name.get("sha256") == digest:
        return {
            "action": "skip_same",
            "rel": rel,
            "dest": same_name["path"],
            "existing": same_name["path"],
            "existing_rel": rel,
            "sha256": digest,
            "size": size,
        }
    hashed = index["by_hash"].get(digest)
    if hashed is not None and hashed.get("rel") != rel:
        return {
            "action": "skip_duplicate",
            "rel": rel,
            "dest": hashed["path"],
            "existing": hashed["path"],
            "existing_rel": hashed["rel"],
            "sha256": digest,
            "size": size,
        }
    natural = str(Path(index["root"]) / rel)
    current = index["by_rel"].get(rel)
    if current is not None:
        dest_rel = _free_imported_rel(rel, index)
        return {
            "action": "conflict",
            "rel": rel,
            "dest": str(Path(index["root"]) / dest_rel),
            "dest_rel": dest_rel,
            "existing": current["path"],
            "existing_rel": rel,
            "sha256": digest,
            "size": size,
        }
    return {
        "action": "transfer",
        "rel": rel,
        "dest": natural,
        "existing": "",
        "existing_rel": "",
        "sha256": digest,
        "size": size,
    }


def verify_then_delete(source: Path, dest: Path, expected: str, cache: HashCache) -> None:
    """Delete ``source`` only after ``dest`` hashes to ``expected``.

    A mismatch leaves both files in place.
    """

    actual = hash_file(dest, cache, fresh=True)
    if actual != expected:
        raise FleetError(
            f"refusing to delete {source}: destination hash does not match "
            f"(expected {expected}, got {actual})"
        )
    if _same_file(source, dest):
        return
    try:
        source.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FleetError(f"cannot delete source {source}: {exc}") from exc


def hash_file(
    path: Path,
    cache: HashCache | None,
    *,
    fresh: bool = False,
    gate: Gate | None = None,
    on_chunk=None,
) -> str:
    """SHA-256 of a regular file. ``fresh`` ignores the cache and still updates it."""

    if path.is_symlink():
        raise FleetError(f"refusing to hash symlink {path}")
    try:
        st = path.stat()
    except OSError as exc:
        raise FleetError(f"cannot stat {path}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise FleetError(f"refusing to hash non-file {path}")
    if cache is not None and not fresh:
        cached = cache.get(path, st.st_size, st.st_mtime_ns)
        if cached:
            if on_chunk is not None:
                on_chunk(st.st_size, st.st_size)
            return cached
    digest = hashlib.sha256()
    done = 0
    try:
        with path.open("rb") as handle:
            while True:
                if gate is not None:
                    gate.checkpoint()
                block = handle.read(_CHUNK)
                if not block:
                    break
                digest.update(block)
                done += len(block)
                if on_chunk is not None:
                    on_chunk(done, st.st_size)
    except ImportCancelled:
        raise
    except OSError as exc:
        raise FleetError(f"cannot read {path}: {exc}") from exc
    hexdigest = digest.hexdigest()
    if cache is not None:
        cache.put(path, st.st_size, st.st_mtime_ns, hexdigest)
    return hexdigest


def copy_verified(
    source: Path,
    dest: Path,
    expected: str,
    cache: HashCache | None,
    *,
    gate: Gate | None = None,
    on_chunk=None,
    fleet_root: Path,
    source_roots: list[Path],
) -> None:
    """Copy to a partial name, hash it, then publish. Cancel removes the partial.

    ``dest`` is never replaced. The source is not deleted here.
    """

    _assert_readable(source, fleet_root, source_roots)
    _assert_dest(dest, fleet_root)
    if dest.exists() or dest.is_symlink():
        raise FleetError(f"refusing to overwrite {dest}")
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    _assert_dest(partial, fleet_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() or partial.is_symlink():
        _unlink_partial(partial)
    try:
        with source.open("rb") as incoming, partial.open("xb") as outgoing:
            done = 0
            while True:
                if gate is not None:
                    gate.checkpoint()
                block = incoming.read(_CHUNK)
                if not block:
                    break
                outgoing.write(block)
                done += len(block)
                if on_chunk is not None:
                    on_chunk(done, expected and done)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except ImportCancelled:
        _unlink_partial(partial)
        raise
    except OSError as exc:
        _unlink_partial(partial)
        raise FleetError(f"cannot copy {source} to {dest}: {exc}") from exc
    _commit_partial(partial, dest, expected, cache, gate=gate, label=str(source))


def stream_tar_member(stream, dest: Path, *, gate: Gate | None = None, on_chunk=None) -> int:
    """Write the first regular file in a ``docker cp`` tar to ``dest`` (a partial path)."""

    while True:
        header = _read_exact(stream, 512)
        if header is None or header == b"\0" * 512:
            raise FleetError("container archive has no file")
        size = _tar_size(header)
        kind = header[156:157]
        if kind in {b"x", b"g", b"L", b"K"}:
            _discard(stream, size + _tar_pad(size))
            continue
        if kind not in {b"0", b"\0"}:
            _discard(stream, size + _tar_pad(size))
            continue
        done = 0
        with dest.open("xb") as outgoing:
            remaining = size
            while remaining:
                if gate is not None:
                    gate.checkpoint()
                block = stream.read(min(_CHUNK, remaining))
                if not block:
                    raise FleetError("container archive ended early")
                outgoing.write(block)
                remaining -= len(block)
                done += len(block)
                if on_chunk is not None:
                    on_chunk(done, size)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _discard(stream, _tar_pad(size))
        return size


def hash_tar_member(stream, *, gate: Gate | None = None, on_chunk=None) -> tuple[str, int]:
    """SHA-256 the first regular file in a tar stream without writing it."""

    while True:
        header = _read_exact(stream, 512)
        if header is None or header == b"\0" * 512:
            raise FleetError("container archive has no file")
        size = _tar_size(header)
        kind = header[156:157]
        if kind in {b"x", b"g", b"L", b"K"}:
            _discard(stream, size + _tar_pad(size))
            continue
        if kind not in {b"0", b"\0"}:
            _discard(stream, size + _tar_pad(size))
            continue
        digest = hashlib.sha256()
        done = 0
        remaining = size
        while remaining:
            if gate is not None:
                gate.checkpoint()
            block = stream.read(min(_CHUNK, remaining))
            if not block:
                raise FleetError("container archive ended early")
            digest.update(block)
            remaining -= len(block)
            done += len(block)
            if on_chunk is not None:
                on_chunk(done, size)
        _discard(stream, _tar_pad(size))
        return digest.hexdigest(), size


def parse_container_inspect(payload: dict) -> dict:
    """Mounts, published ports, GPU ids, and env from one ``docker inspect`` object."""

    if not isinstance(payload, dict):
        raise FleetError("docker inspect payload must be an object")
    name = str(payload.get("Name") or "").lstrip("/")
    if not name:
        raise FleetError("docker inspect payload has no container name")
    state = payload.get("State") if isinstance(payload.get("State"), dict) else {}
    config = payload.get("Config") if isinstance(payload.get("Config"), dict) else {}
    host = payload.get("HostConfig") if isinstance(payload.get("HostConfig"), dict) else {}
    mounts = []
    for item in payload.get("Mounts") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("Source") or "")
        destination = str(item.get("Destination") or "")
        if not source or not destination:
            continue
        mounts.append(
            {
                "source": source,
                "destination": destination,
                "type": str(item.get("Type") or "bind"),
                "role": guess_role(destination),
            }
        )
    return {
        "id": str(payload.get("Id") or ""),
        "name": name,
        "status": str(state.get("Status") or ""),
        "image": str(config.get("Image") or ""),
        "mounts": mounts,
        "ports": _published_ports(payload),
        "gpus": _gpu_ids(host, config.get("Env") or []),
        "env": _public_env(config.get("Env") or []),
        "labels": dict(config.get("Labels") or {}) if isinstance(config.get("Labels"), dict) else {},
    }


def guess_role(destination: str) -> str:
    """Map a container path onto the ComfyFleet layout. Unknown paths are skipped."""

    parts = [part.lower() for part in PurePosixPath(destination).parts if part not in {"", "/"}]
    if not parts:
        return "skip"
    if parts[-1] == "workflows" or (len(parts) >= 2 and parts[-2:] == ["default", "workflows"]):
        return "workflows"
    tail = parts[-1]
    if tail == "models" or tail in _MODEL_SUBDIR_NAMES:
        return "models"
    return {
        "output": "output",
        "outputs": "output",
        "input": "input",
        "inputs": "input",
        "temp": "temp",
        "custom_nodes": "custom_nodes",
        "wildcards": "wildcards",
        "user": "user",
    }.get(tail, "skip")


def model_rel(container_destination: str, relative: str) -> str:
    """Keep the ComfyUI subfolder. A mount of only ``checkpoints`` stays ``checkpoints/...``."""

    relative = _rel_posix(relative)
    tail = PurePosixPath(container_destination).name.lower()
    if tail in _MODEL_SUBDIR_NAMES and not relative.startswith(tail + "/"):
        return f"{tail}/{relative}"
    return relative


def suggested_port(ports: list[dict]) -> int:
    for item in ports:
        if int(item.get("container") or 0) == 8188 and int(item.get("host") or 0):
            return int(item["host"])
    for item in ports:
        if int(item.get("host") or 0):
            return int(item["host"])
    return 8188


def summarize_actions(actions: list[dict]) -> dict:
    summary = {
        "skip": 0,
        "duplicates": 0,
        "conflicts": 0,
        "transfer": 0,
        "bytes_transfer": 0,
        "bytes_saved": 0,
        "bytes_total": 0,
        "errors": 0,
    }
    for action in actions:
        size = int(action.get("size") or 0)
        summary["bytes_total"] += size
        kind = action.get("action")
        if kind == "skip_same":
            summary["skip"] += 1
            summary["bytes_saved"] += size
        elif kind == "skip_duplicate":
            summary["duplicates"] += 1
            summary["bytes_saved"] += size
        elif kind == "conflict":
            summary["conflicts"] += 1
            summary["bytes_transfer"] += size
        elif kind == "transfer":
            summary["transfer"] += 1
            summary["bytes_transfer"] += size
    return summary


def build_index(dest_root: Path, cache: HashCache, *, gate: Gate | None = None, on_file=None) -> dict:
    """Hash every regular file under ``dest_root``. The tree is not modified."""

    index = {"root": str(dest_root), "by_rel": {}, "by_hash": {}}
    if not dest_root.exists():
        return index
    if dest_root.is_symlink():
        raise FleetError(f"refusing to index symlink {dest_root}")
    for path in _walk_files(dest_root):
        if gate is not None:
            gate.checkpoint()
        rel = path.relative_to(dest_root).as_posix()
        if on_file is not None:
            on_file(rel, path)
        digest = hash_file(path, cache, gate=gate)
        row = {"path": str(path), "rel": rel, "sha256": digest, "size": path.stat().st_size}
        index["by_rel"][rel] = row
        index["by_hash"].setdefault(digest, row)
    return index


def plan_tree(
    source_root: Path,
    dest_root: Path,
    *,
    cache: HashCache,
    rel_for=None,
    gate: Gate | None = None,
    on_file=None,
    on_chunk=None,
    include=None,
) -> list[dict]:
    """Classify files. Reads and updates the hash cache. Does not copy or delete."""

    index = build_index(dest_root, cache, gate=gate, on_file=on_file)
    actions = []
    if not source_root.exists():
        raise FleetError(f"source mount is not visible: {source_root}")
    for path in _walk_files(source_root):
        if gate is not None:
            gate.checkpoint()
        rel = path.relative_to(source_root).as_posix()
        if include is not None and not include(rel, path):
            continue
        if rel_for is not None:
            rel = rel_for(rel)
        if on_file is not None:
            on_file(rel, path)
        digest = hash_file(path, cache, gate=gate, on_chunk=on_chunk)
        decision = classify_model(rel, digest, path.stat().st_size, index)
        decision["source"] = str(path)
        decision["container_path"] = ""
        actions.append(decision)
        if decision["action"] in {"transfer", "conflict"}:
            stored_rel = decision.get("dest_rel") or rel
            index["by_rel"][stored_rel] = {
                "path": decision["dest"],
                "rel": stored_rel,
                "sha256": digest,
                "size": decision["size"],
            }
            index["by_hash"].setdefault(
                digest,
                index["by_rel"][stored_rel],
            )
    return actions


def plan_tar_stream(
    stream,
    dest_root: Path,
    *,
    destination: str,
    container: str,
    cache: HashCache,
    rel_for=None,
    gate: Gate | None = None,
    on_file=None,
    on_chunk=None,
    include=None,
) -> list[dict]:
    """Classify a ``docker cp`` archive. Does not copy or delete."""

    index = build_index(dest_root, cache, gate=gate, on_file=on_file)
    actions: list[dict] = []

    def handle(name: str, size: int, mtime: int, reader: _LimitedReader) -> None:
        rel = _tar_rel(name, destination)
        if not rel or PurePosixPath(rel).name.endswith(PARTIAL_SUFFIX):
            _drain(reader)
            return
        stat_path = _MemberStat(rel, size)
        if include is not None and not include(rel, stat_path):
            _drain(reader)
            return
        stored_rel = rel_for(rel) if rel_for is not None else rel
        if on_file is not None:
            on_file(stored_rel, _MemberStat(stored_rel, size))
        container_path = str(PurePosixPath(destination) / rel)
        digest = _hash_reader(
            reader,
            size,
            mtime,
            cache,
            _docker_cache_key(container, container_path),
            gate=gate,
            on_chunk=on_chunk,
        )
        decision = classify_model(stored_rel, digest, size, index)
        decision["source"] = container_path
        decision["container_path"] = container_path
        decision["via"] = "container"
        actions.append(decision)
        if decision["action"] in {"transfer", "conflict"}:
            stored = decision.get("dest_rel") or stored_rel
            index["by_rel"][stored] = {
                "path": decision["dest"],
                "rel": stored,
                "sha256": digest,
                "size": decision["size"],
            }
            index["by_hash"].setdefault(digest, index["by_rel"][stored])

    _for_each_tar_file(stream, handle, gate=gate)
    return actions


def perform_action(
    action: dict,
    *,
    mode: str,
    cache: HashCache,
    duplicates: DuplicatesStore | None,
    import_id: str,
    fleet_root: Path,
    source_roots: list[Path],
    gate: Gate | None = None,
    on_chunk=None,
    owner: tuple[int, int] | None = None,
    now: str = "",
    docker=None,
    container: str = "",
) -> str:
    """Apply one classified file. Returns ``skipped``, ``duplicates``, ``conflicts``, or ``moved``."""

    if mode not in {"copy", "move"}:
        raise FleetError(f"mode must be copy or move, got {mode!r}")
    source = Path(action["source"])
    kind = action["action"]
    expected = action["sha256"]
    if action.get("via") == "container":
        return _perform_container_action(
            action,
            mode=mode,
            cache=cache,
            duplicates=duplicates,
            import_id=import_id,
            fleet_root=fleet_root,
            gate=gate,
            on_chunk=on_chunk,
            owner=owner,
            now=now,
            docker=docker,
            container=container,
        )
    if kind == "skip_same":
        dest = Path(action["dest"])
        _assert_readable(source, fleet_root, source_roots)
        if mode == "move":
            _assert_dest(dest, fleet_root)
            verify_then_delete(source, dest, expected, cache)
        return "skipped"
    if kind == "skip_duplicate":
        existing = Path(action["existing"])
        _assert_readable(source, fleet_root, source_roots)
        _assert_dest(existing, fleet_root)
        if mode == "move":
            verify_then_delete(source, existing, expected, cache)
        if duplicates is not None:
            duplicates.append(duplicate_entry(action, import_id, now or _now()))
        return "duplicates"
    if kind not in {"transfer", "conflict"}:
        raise FleetError(f"unknown import action {kind!r}")
    dest = Path(action["dest"])
    _assert_readable(source, fleet_root, source_roots)
    _assert_dest(dest, fleet_root)
    created = create_new_host_dirs(dest.parent, fleet_root)
    _chown_paths(created, owner)
    copy_verified(
        source,
        dest,
        expected,
        cache,
        gate=gate,
        on_chunk=on_chunk,
        fleet_root=fleet_root,
        source_roots=source_roots,
    )
    _chown_paths([dest], owner)
    if mode == "move":
        verify_then_delete(source, dest, expected, cache)
    return "conflicts" if kind == "conflict" else "moved"


def _perform_container_action(
    action: dict,
    *,
    mode: str,
    cache: HashCache,
    duplicates: DuplicatesStore | None,
    import_id: str,
    fleet_root: Path,
    gate: Gate | None,
    on_chunk,
    owner: tuple[int, int] | None,
    now: str,
    docker,
    container: str,
) -> str:
    """Copy one file out of the container. Move is refused: the host path is not visible."""

    if mode == "move":
        raise FleetError(
            "Move needs the source file on a path this manager can see, "
            "so it can be deleted after the destination hash matches. Use Copy."
        )
    kind = action["action"]
    if kind == "skip_same":
        return "skipped"
    if kind == "skip_duplicate":
        if duplicates is not None:
            duplicates.append(duplicate_entry(action, import_id, now or _now()))
        return "duplicates"
    if kind not in {"transfer", "conflict"}:
        raise FleetError(f"unknown import action {kind!r}")
    dest = Path(action["dest"])
    _assert_dest(dest, fleet_root)
    created = create_new_host_dirs(dest.parent, fleet_root)
    _chown_paths(created, owner)
    copy_container_file(
        docker,
        container or "",
        str(action.get("container_path") or action.get("source") or ""),
        dest,
        action["sha256"],
        cache,
        gate=gate,
        on_chunk=on_chunk,
        fleet_root=fleet_root,
    )
    _chown_paths([dest], owner)
    return "conflicts" if kind == "conflict" else "moved"


def copy_container_file(
    docker,
    container: str,
    container_path: str,
    dest: Path,
    expected: str,
    cache: HashCache | None,
    *,
    gate: Gate | None = None,
    on_chunk=None,
    fleet_root: Path,
) -> None:
    """Stream one container file into a partial, then publish it after the hash matches."""

    _assert_dest(dest, fleet_root)
    if dest.exists() or dest.is_symlink():
        raise FleetError(f"refusing to overwrite {dest}")
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    _assert_dest(partial, fleet_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() or partial.is_symlink():
        _unlink_partial(partial)
    proc = _open_tar(docker, container, container_path)
    try:
        stdout = getattr(proc, "stdout", None)
        if stdout is None:
            raise FleetError(f"docker cp produced no output for {container_path}")
        stream_tar_member(stdout, partial, gate=gate, on_chunk=on_chunk)
        _close_proc(proc)
    except Exception:
        _abort_proc(proc)
        _unlink_partial(partial)
        raise
    _commit_partial(partial, dest, expected, cache, gate=gate, label=container_path)


def _commit_partial(
    partial: Path,
    dest: Path,
    expected: str,
    cache: HashCache | None,
    *,
    gate: Gate | None,
    label: str,
) -> None:
    try:
        actual = hash_file(partial, None, fresh=True, gate=gate)
    except ImportCancelled:
        _unlink_partial(partial)
        raise
    except Exception:
        _unlink_partial(partial)
        raise
    if actual != expected:
        _unlink_partial(partial)
        raise FleetError(f"copied hash does not match for {label} (expected {expected}, got {actual})")
    if dest.exists() or dest.is_symlink():
        _unlink_partial(partial)
        raise FleetError(f"refusing to overwrite {dest}")
    try:
        os.replace(partial, dest)
    except OSError as exc:
        _unlink_partial(partial)
        raise FleetError(f"cannot publish {dest}: {exc}") from exc
    confirmed = hash_file(dest, cache, fresh=True)
    if confirmed != expected:
        try:
            dest.unlink()
        except OSError:
            pass
        raise FleetError(f"destination {dest} failed verification; source was kept")


def duplicate_entry(action: dict, import_id: str, date: str) -> dict:
    existing = Path(action["existing"])
    return {
        "incoming_name": Path(action["rel"]).name,
        "existing_name": existing.name,
        "existing_path": str(existing),
        "size": int(action["size"]),
        "hash": action["sha256"],
        "import_id": import_id,
        "date": date,
    }


class ImportService:
    """One fleet root. Jobs persist under ``<root>/.import`` and survive a refresh."""

    def __init__(self, layout: FleetLayout, *, port_in_use=None, create=None) -> None:
        self.layout = layout
        self.port_in_use = port_in_use
        self._create = create or create_instance
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._gates: dict[str, Gate] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._runtime: dict[str, dict] = {}
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    @property
    def store_dir(self) -> Path:
        return self.layout.root / ".import"

    @property
    def cache(self) -> HashCache:
        return HashCache(self.store_dir / "hashes.json")

    @property
    def duplicates(self) -> DuplicatesStore:
        return DuplicatesStore(self.store_dir / "duplicates.json")

    def list_containers(self, docker) -> list[dict]:
        authorize("import")
        records = docker.list_container_records()
        return [
            {
                "id": record.id,
                "name": record.name,
                "status": record.status,
                "managed": record.labels.get("comfyfleet.managed") == "true",
            }
            for record in records
        ]

    def inspect(self, docker, container: str) -> dict:
        authorize("import")
        parsed = parse_container_inspect(docker.inspect_container(container))
        for mount in parsed["mounts"]:
            source = Path(mount["source"])
            mount["visible"] = source.is_dir() and not source.is_symlink()
        workflows = _visible_workflows(parsed["mounts"], docker=docker, container=parsed["name"])
        name = _free_instance_name(docker, sanitize_stem(parsed["name"]))
        return {
            "ok": True,
            "container": parsed,
            "suggested_name": name,
            "suggested_port": suggested_port(parsed["ports"]),
            "workflows": workflows,
            "owner": "comfyuser",
            "group": "comfyuser",
            "custom_nodes": "as-is",
            "mode": "copy",
        }

    def create_job(self, docker, gpus: list, body: dict, *, inline: bool = False) -> dict:
        authorize("import")
        self._reject_if_busy()
        draft = self.inspect(docker, str(body.get("container") or ""))
        parsed = draft["container"]
        name = sanitize_stem(str(body.get("name") or draft["suggested_name"]))
        if not is_instance_name(name):
            raise FleetError(f"invalid instance name {name!r}")
        if (self.layout.metadata_file(name)).is_file():
            raise FleetError(f"instance {name!r} already exists")
        existing = docker.status(name)
        if existing is not None:
            raise FleetError(
                f"container {name!r} already exists ({existing}). "
                "Choose a different name. Import does not replace the old container."
            )
        port = int(body.get("port") if body.get("port") is not None else draft["suggested_port"])
        if port < 1 or port > 65535:
            raise FleetError(f"port must be 1..65535, got {port}")
        mode = str(body.get("mode") or "copy").lower()
        if mode not in {"copy", "move"}:
            raise FleetError("mode must be copy or move")
        if mode == "move" and parsed["status"] == "running":
            raise FleetError(
                "the old container is running. Stop it before Move. "
                "Import does not stop or remove it."
            )
        nodes = str(body.get("custom_nodes") or "as-is")
        if nodes not in {"as-is", "fresh"}:
            raise FleetError("custom_nodes must be as-is or fresh")
        selected = _gpu_spec(body.get("gpus"), gpus)
        mounts = _merge_mounts(parsed["mounts"], body.get("mounts"))
        if mode == "move":
            _assert_move_visible(mounts, nodes)
        workflow = str(body.get("workflow") or (draft["workflows"][0]["path"] if draft["workflows"] else ""))
        if not workflow:
            raise FleetError(
                "workflow is required. The source container has no workflow JSON to import."
            )
        cuda_tag = str(body.get("cuda_tag") or "cu130")
        job_id = "imp-" + uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "hashing",
            "phase": "hashing",
            "dismissed": False,
            "container": parsed["name"],
            "container_id": parsed["id"],
            "container_status": parsed["status"],
            "name": name,
            "port": port,
            "gpus": selected,
            "mode": mode,
            "custom_nodes": nodes,
            "cuda_tag": cuda_tag,
            "workflow": workflow,
            "mounts": mounts,
            "created_at": _now(),
            "updated_at": _now(),
            "started_at": time.time(),
            "elapsed_s": 0.0,
            "current_file": "",
            "current_bytes": 0,
            "current_size": 0,
            "files_done": 0,
            "files_total": 0,
            "bytes_done": 0,
            "bytes_total": 0,
            "speed_current": 0.0,
            "speed_average": 0.0,
            "eta_s": None,
            "counts": {"moved": 0, "skipped": 0, "renamed_dupes": 0, "conflicts": 0, "errors": 0},
            "free_bytes": _free_bytes(self.layout.root),
            "low_space": False,
            "summary": summarize_actions([]),
            "actions": [],
            "log_tail": [],
            "error": "",
            "partial": "",
        }
        self._put(job)
        self._log(job, f"hashing mounts for {parsed['name']} -> {name}")
        self._runtime[job_id] = {"docker": docker, "gpus": gpus, "samples": [], "slice_start": time.monotonic()}
        gate = Gate()
        self._gates[job_id] = gate
        if inline:
            self._hash_job(job_id)
            return self.get_job(job_id)
        self._spawn(job_id, self._hash_job)
        return self.get_job(job_id)

    def start_transfer(self, job_id: str, *, mode: str | None = None, inline: bool = False) -> dict:
        authorize("import")
        job = self._require(job_id)
        if job["status"] != "awaiting_confirm":
            raise FleetError("import is not waiting to start")
        if mode:
            picked = str(mode).lower()
            if picked not in {"copy", "move"}:
                raise FleetError("mode must be copy or move")
            if picked == "move" and job.get("container_status") == "running":
                raise FleetError(
                    "the old container is running. Stop it before Move. "
                    "Import does not stop or remove it."
                )
            job["mode"] = picked
        if job["mode"] == "move":
            _assert_move_visible(job["mounts"], job.get("custom_nodes") or "as-is")
        job["status"] = "running"
        job["phase"] = "copying"
        job["files_done"] = 0
        job["bytes_done"] = 0
        job["bytes_total"] = int(job["summary"].get("bytes_total") or 0)
        job["started_at"] = time.time()
        job["elapsed_s"] = 0.0
        self._put(job)
        self._log(job, f"starting {job['mode']}")
        runtime = self._runtime.get(job_id) or {}
        runtime["slice_start"] = time.monotonic()
        runtime["samples"] = []
        self._runtime[job_id] = runtime
        if inline:
            self._transfer_job(job_id)
            return self.get_job(job_id)
        self._spawn(job_id, self._transfer_job)
        return self.get_job(job_id)

    def pause(self, job_id: str) -> dict:
        authorize("import")
        job = self._require(job_id)
        if job["status"] not in {"hashing", "running"}:
            raise FleetError("import is not running")
        gate = self._gates.get(job_id)
        if gate is not None:
            gate.pause()
        job["status"] = "paused"
        self._flush_elapsed(job)
        self._put(job)
        self._log(job, "paused")
        return self.get_job(job_id)

    def resume(self, job_id: str, docker=None, gpus=None, *, inline: bool = False) -> dict:
        authorize("import")
        job = self._require(job_id)
        if job["status"] != "paused":
            raise FleetError("import is not paused")
        gate = self._gates.get(job_id)
        thread = self._threads.get(job_id)
        if gate is not None and thread is not None and thread.is_alive():
            gate.resume()
            job["status"] = "hashing" if job.get("phase") == "hashing" and not job.get("actions") else "running"
            self._runtime.setdefault(job_id, {})["slice_start"] = time.monotonic()
            self._put(job)
            self._log(job, "resumed")
            return self.get_job(job_id)
        if docker is not None:
            runtime = self._runtime.setdefault(job_id, {})
            runtime["docker"] = docker
            if gpus is not None:
                runtime["gpus"] = gpus
        gate = self._gates.get(job_id)
        if gate is None:
            gate = Gate()
            self._gates[job_id] = gate
        gate.resume()
        phase = job.get("phase") or "copying"
        job["status"] = "hashing" if phase == "hashing" and not job.get("actions") else "running"
        if job["status"] == "running":
            job["phase"] = "copying"
        self._runtime.setdefault(job_id, {})["slice_start"] = time.monotonic()
        self._put(job)
        self._log(job, "resumed")
        target = self._hash_job if job["status"] == "hashing" else self._transfer_job
        if inline:
            target(job_id)
            return self.get_job(job_id)
        self._spawn(job_id, target)
        return self.get_job(job_id)

    def cancel(self, job_id: str) -> dict:
        authorize("import")
        job = self._require(job_id)
        if job["status"] in _TERMINAL:
            raise FleetError("import is already finished")
        gate = self._gates.get(job_id)
        if gate is not None:
            gate.cancel()
        thread = self._threads.get(job_id)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=30)
        job = self._require(job_id)
        if job["status"] not in _TERMINAL:
            self._drop_partial(job)
            job["status"] = "cancelled"
            job["phase"] = "done"
            self._flush_elapsed(job)
            self._put(job)
            self._log(job, "cancelled")
        return self.get_job(job_id)

    def dismiss(self, job_id: str) -> dict:
        authorize("import")
        job = self._require(job_id)
        if job["status"] not in _TERMINAL:
            raise FleetError("dismiss the import after it finishes")
        job["dismissed"] = True
        self._put(job)
        return self.get_job(job_id)

    def remove_old(self, job_id: str, docker) -> dict:
        authorize("import")
        job = self._require(job_id)
        if job["status"] not in _TERMINAL:
            raise FleetError("finish or cancel the import before removing the old container")
        container = job["container"]
        if container == job["name"]:
            raise FleetError("refusing to remove the new instance")
        info = parse_container_inspect(docker.inspect_container(container))
        if info["labels"].get("comfyfleet.managed") == "true":
            raise FleetError(
                "that container is a fleet instance. Use Delete on its card. "
                "Remove old does not delete fleet instances."
            )
        if info["status"] == "running":
            raise FleetError(
                "the old container is running. Stop it before Remove old. "
                "Import does not stop it."
            )
        if not info["id"]:
            raise FleetError("old container has no id")
        docker.remove_stopped(info["id"])
        self._log(job, f"removed old container {container}")
        job["old_removed"] = True
        self._put(job)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict:
        authorize("import")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise FleetError(f"unknown import {job_id}")
            snapshot = json.loads(json.dumps(job))
        snapshot["free_bytes"] = _free_bytes(self.layout.root)
        snapshot["low_space"] = _low_space(snapshot["free_bytes"], snapshot)
        return snapshot

    def public_job(self, job: dict) -> dict:
        shown = dict(job)
        shown.pop("actions", None)
        return shown

    def active_job(self) -> dict | None:
        authorize("import")
        with self._lock:
            rows = [job for job in self._jobs.values() if not job.get("dismissed")]
        if not rows:
            return None
        rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return self.public_job(self.get_job(rows[0]["id"]))

    def list_jobs(self) -> list[dict]:
        authorize("import")
        with self._lock:
            rows = list(self._jobs.values())
        rows.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return [
            {
                "id": job["id"],
                "status": job["status"],
                "container": job["container"],
                "name": job["name"],
                "mode": job["mode"],
                "created_at": job["created_at"],
                "summary": job.get("summary") or {},
                "counts": job.get("counts") or {},
                "error": job.get("error") or "",
                "dismissed": bool(job.get("dismissed")),
            }
            for job in rows
        ]

    def log_text(self, job_id: str) -> str:
        authorize("import")
        self._require(job_id)
        path = self._log_path(job_id)
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FleetError(f"cannot read import log: {exc}") from exc

    def _hash_job(self, job_id: str) -> None:
        job = self._require(job_id)
        gate = self._gates[job_id]
        cache = self.cache
        docker = (self._runtime.get(job_id) or {}).get("docker")
        try:
            actions: list[dict] = []
            mounts = job["mounts"]
            files = _count_planned_files(mounts, job)
            job["files_total"] = files[0]
            job["bytes_total"] = files[1]
            self._put(job)
            for mount in mounts:
                role = mount.get("role") or "skip"
                if role == "skip":
                    continue
                if role == "custom_nodes" and job["custom_nodes"] == "fresh":
                    self._log(job, f"skipping custom nodes from {mount['destination']} (fresh install)")
                    continue
                source_root = Path(mount["source"])
                dest_root = self._dest_root(role, job["name"])
                if dest_root is None:
                    continue
                visible = source_root.is_dir() and not source_root.is_symlink()
                if visible:
                    self._log(job, f"hashing {source_root} -> {dest_root}")
                else:
                    self._log(
                        job,
                        f"reading {mount['destination']} with docker cp "
                        f"({source_root} is not visible) -> {dest_root}",
                    )

                progress = {"base": int(job.get("bytes_done") or 0)}

                def _on_file(rel, path, _job=job, _progress=progress):
                    gate.checkpoint()
                    _progress["base"] = int(_job.get("bytes_done") or 0)
                    _job["current_file"] = rel
                    _job["current_size"] = path.stat().st_size if path.is_file() else 0
                    _job["current_bytes"] = 0
                    _job["phase"] = "hashing"
                    self._note_progress(_job, 0)

                def _on_chunk(done, total, _job=job, _progress=progress):
                    _job["current_bytes"] = done
                    _job["current_size"] = total or _job.get("current_size") or 0
                    _job["bytes_done"] = _progress["base"] + done
                    self._note_progress(_job, done)

                def _rel(rel, _mount=mount, _role=role):
                    if _role == "models":
                        return model_rel(_mount["destination"], rel)
                    return rel

                def _include(rel, path, _role=role):
                    if _role == "user":
                        parts = PurePosixPath(rel).parts
                        return "workflows" in parts and path.suffix.lower() == ".json"
                    return True

                if visible:
                    planned = plan_tree(
                        source_root,
                        dest_root,
                        cache=cache,
                        rel_for=_rel if role == "models" else None,
                        gate=gate,
                        on_file=_on_file,
                        on_chunk=_on_chunk,
                        include=_include if role == "user" else None,
                    )
                else:
                    proc = _open_tar(docker, job["container"], mount["destination"])
                    try:
                        stdout = getattr(proc, "stdout", None)
                        if stdout is None:
                            raise FleetError(f"docker cp produced no output for {mount['destination']}")
                        planned = plan_tar_stream(
                            stdout,
                            dest_root,
                            destination=mount["destination"],
                            container=job["container"],
                            cache=cache,
                            rel_for=_rel if role == "models" else None,
                            gate=gate,
                            on_file=_on_file,
                            on_chunk=_on_chunk,
                            include=_include if role == "user" else None,
                        )
                        _close_proc(proc)
                    except Exception:
                        _abort_proc(proc)
                        raise
                for action in planned:
                    action["role"] = role
                    action["mount_source"] = mount["source"]
                    if not action.get("container_path"):
                        action["container_path"] = _container_path(
                            mount["destination"], action["source"], mount["source"]
                        )
                    if role == "models":
                        action["models"] = True
                actions.extend(planned)
                job["files_done"] = len(actions)
                self._put(job)
            cache.save()
            self._ensure_workflow(job, actions)
            job["actions"] = actions
            job["summary"] = summarize_actions(actions)
            job["status"] = "awaiting_confirm"
            job["phase"] = "hashing"
            job["current_file"] = ""
            job["current_bytes"] = 0
            summary = job["summary"]
            self._log(
                job,
                "plan: "
                f"skip {summary['skip']}, duplicates {summary['duplicates']}, "
                f"rename {summary['conflicts']}, transfer {summary['transfer']}, "
                f"space saved {summary['bytes_saved']} bytes",
            )
            self._flush_elapsed(job)
            self._put(job)
        except ImportCancelled:
            self._finish_cancelled(job_id)
        except Exception as exc:
            self._finish_failed(job_id, exc)

    def _transfer_job(self, job_id: str) -> None:
        job = self._require(job_id)
        gate = self._gates[job_id]
        cache = self.cache
        docker = (self._runtime.get(job_id) or {}).get("docker")
        try:
            if _port_taken_by(self.layout, int(job["port"]), job["name"], self.port_in_use):
                raise FleetError(f"port {job['port']} is already in use. Pick another port.")
            actions = list(job.get("actions") or [])
            job["files_total"] = len(actions)
            job["bytes_total"] = sum(int(item.get("size") or 0) for item in actions)
            job["bytes_done"] = 0
            job["files_done"] = 0
            self._put(job)
            source_roots = [Path(item["source"]) for item in job["mounts"]]
            owner = _owner_ids()
            seen_bytes = 0
            for index, action in enumerate(actions):
                gate.checkpoint()
                if action.get("done") or _already_done(action, job["mode"]):
                    action["done"] = True
                    seen_bytes += int(action.get("size") or 0)
                    job["files_done"] = index + 1
                    job["bytes_done"] = seen_bytes
                    job["current_file"] = action.get("rel") or ""
                    self._note_progress(job, 0)
                    continue
                kind = action["action"]
                if kind in {"transfer", "conflict"}:
                    free = _free_bytes(self.layout.root)
                    job["free_bytes"] = free
                    needed = int(action.get("size") or 0) + _FILE_MARGIN
                    if free < needed:
                        job["low_space"] = True
                        self._log(job, f"low disk space: {free} bytes free, need {needed} for {action['rel']}")
                        self._put(job)
                        gate.pause()
                        job["status"] = "paused"
                        self._put(job)
                        gate.checkpoint()
                        job["status"] = "running"
                job["phase"] = "verifying" if kind in {"skip_same", "skip_duplicate"} else "copying"
                job["current_file"] = action.get("rel") or ""
                job["current_size"] = int(action.get("size") or 0)
                job["current_bytes"] = 0
                job["partial"] = str(Path(action["dest"]).with_name(Path(action["dest"]).name + PARTIAL_SUFFIX)) if kind in {"transfer", "conflict"} else ""

                def _on_chunk(done, _total, _job=job, _base=seen_bytes):
                    _job["current_bytes"] = done
                    _job["bytes_done"] = _base + done
                    self._note_progress(_job, done)

                try:
                    label = perform_action(
                        action,
                        mode=job["mode"],
                        cache=cache,
                        duplicates=self.duplicates if action.get("models") and action["action"] == "skip_duplicate" else None,
                        import_id=job["id"],
                        fleet_root=self.layout.root,
                        source_roots=source_roots,
                        gate=gate,
                        on_chunk=_on_chunk if kind in {"transfer", "conflict"} else None,
                        owner=owner,
                        now=job["created_at"],
                        docker=docker,
                        container=job["container"],
                    )
                except FleetError as exc:
                    job["counts"]["errors"] = int(job["counts"].get("errors") or 0) + 1
                    self._log(job, f"error {action.get('rel')}: {exc}")
                    self._drop_partial(job)
                    seen_bytes += int(action.get("size") or 0)
                    job["files_done"] = index + 1
                    job["bytes_done"] = seen_bytes
                    continue
                if action.get("models") and action["action"] == "skip_duplicate":
                    entry = duplicate_entry(action, job["id"], job["created_at"])
                    self._log(
                        job,
                        "duplicate: "
                        f"{entry['incoming_name']} matches {entry['existing_name']} "
                        f"at {entry['existing_path']} size {entry['size']} sha256 {entry['hash']}",
                    )
                key = {"skipped": "skipped", "duplicates": "renamed_dupes", "conflicts": "conflicts", "moved": "moved"}[label]
                job["counts"][key] = int(job["counts"].get(key) or 0) + 1
                action["done"] = True
                self._log(job, f"{action['action']} {action.get('rel')} ({job['mode']})")
                seen_bytes += int(action.get("size") or 0)
                job["files_done"] = index + 1
                job["bytes_done"] = seen_bytes
                job["current_bytes"] = int(action.get("size") or 0)
                job["partial"] = ""
                self._note_progress(job, 0)
            cache.save()
            job["phase"] = "cleanup"
            self._put(job)
            self._publish_workflow(job, owner)
            self._create_instance(job)
            job["status"] = "completed"
            job["phase"] = "done"
            job["current_file"] = ""
            self._flush_elapsed(job)
            counts = job["counts"]
            self._log(
                job,
                "finished: "
                f"moved {counts['moved']}, skipped {counts['skipped']}, "
                f"duplicates {counts['renamed_dupes']}, conflicts {counts['conflicts']}, "
                f"errors {counts['errors']}",
            )
            self._put(job)
        except ImportCancelled:
            self._finish_cancelled(job_id)
        except Exception as exc:
            self._finish_failed(job_id, exc)

    def _create_instance(self, job: dict) -> None:
        runtime = self._runtime.get(job["id"]) or {}
        docker = runtime.get("docker")
        gpus = runtime.get("gpus") or []
        if docker is None:
            raise FleetError("import lost its Docker client. Resume the job to continue.")
        workflow = self.layout.workflow_file(job["name"])
        if not workflow.is_file():
            raise FleetError(f"workflow was not imported: {workflow}")
        spec = ",".join(str(index) for index in job["gpus"])
        self._log(job, f"creating instance {job['name']} on port {job['port']}")
        self._create(
            workflow,
            layout=self.layout,
            docker=docker,
            gpus=gpus,
            gpus_spec=spec,
            name=job["name"],
            requested_port=int(job["port"]),
            start=False,
            install_missing_from_workflow=False,
            port_in_use=self.port_in_use,
            cuda_tag=job.get("cuda_tag") or None,
        )
        self._log(job, f"instance {job['name']} created. The old container was left in place.")

    def _publish_workflow(self, job: dict, owner: tuple[int, int] | None) -> None:
        target = self.layout.workflow_file(job["name"])
        source = self._workflow_source(job)
        if source is None:
            raise FleetError("workflow file was not found after import")
        if _same_file(source, target):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        partial = target.with_suffix(".json.tmp")
        partial.write_bytes(data)
        os.replace(partial, target)
        _chown_paths([target], owner)

    def _workflow_source(self, job: dict) -> Path | None:
        wanted = str(job.get("workflow") or "")
        for action in job.get("actions") or []:
            if action.get("source") == wanted and Path(action["dest"]).is_file():
                return Path(action["dest"])
        for action in job.get("actions") or []:
            if action.get("role") in {"workflows", "user"} and Path(action.get("dest") or "").is_file():
                if Path(action["dest"]).suffix.lower() == ".json":
                    return Path(action["dest"])
        direct = Path(wanted)
        if direct.is_file():
            return direct
        return None

    def _ensure_workflow(self, job: dict, actions: list[dict]) -> None:
        wanted = str(job.get("workflow") or "")
        if any(action.get("source") == wanted for action in actions):
            return
        path = Path(wanted)
        if path.is_file() and path.suffix.lower() == ".json":
            return
        raise FleetError(f"workflow is not part of this import: {wanted}")

    def _dest_root(self, role: str, name: str) -> Path | None:
        if role == "models":
            return self.layout.models
        if role == "input":
            return self.layout.input_dir(name)
        if role == "output":
            return self.layout.output_dir(name)
        if role == "temp":
            return self.layout.temp_dir(name)
        if role == "custom_nodes":
            return self.layout.custom_nodes(name)
        if role in {"workflows", "user"}:
            return self.layout.instance_dir(name) / "workflows"
        if role == "wildcards":
            return self.layout.wildcards
        return None

    def _reject_if_busy(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                if job["status"] in _LIVE and not job.get("dismissed"):
                    raise FleetError(
                        f"import {job['id']} is still {job['status']}. "
                        "Finish, cancel, or dismiss it before starting another."
                    )

    def _spawn(self, job_id: str, target) -> None:
        thread = threading.Thread(target=target, args=(job_id,), name=f"comfyfleet-{job_id}", daemon=True)
        self._threads[job_id] = thread
        thread.start()

    def _require(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise FleetError(f"unknown import {job_id}")
        return job

    def _put(self, job: dict) -> None:
        job["updated_at"] = _now()
        with self._lock:
            self._jobs[job["id"]] = job
        path = self._job_path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = json.loads(json.dumps(job))
        _atomic_json(path, stored)

    def _load_jobs(self) -> None:
        folder = self.store_dir / "jobs"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not payload.get("id"):
                continue
            if payload.get("status") in {"hashing", "running"}:
                payload["status"] = "paused"
                payload.setdefault("log_tail", [])
                self._append_log_line(
                    payload,
                    "Import paused because the manager restarted. Resume continues from the last finished file.",
                )
                _atomic_json(path, payload)
            self._jobs[str(payload["id"])] = payload

    def _log(self, job: dict, line: str) -> None:
        self._append_log_line(job, line)
        self._put(job)

    def _append_log_line(self, job: dict, line: str) -> None:
        stamped = f"{_now()} {line}"
        tail = list(job.get("log_tail") or [])
        tail.append(stamped)
        job["log_tail"] = tail[-200:]
        path = self._log_path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    def _note_progress(self, job: dict, chunk: int) -> None:
        del chunk
        now = time.monotonic()
        self._flush_elapsed(job, now)
        elapsed = float(job.get("elapsed_s") or 0)
        done = int(job.get("bytes_done") or 0)
        total = int(job.get("bytes_total") or 0)
        samples = self._runtime.setdefault(job["id"], {}).setdefault("samples", [])
        samples.append((now, done))
        samples[:] = [(stamp, value) for stamp, value in samples if now - stamp <= 5]
        if len(samples) >= 2 and now > samples[0][0]:
            job["speed_current"] = (done - samples[0][1]) / (now - samples[0][0])
        else:
            job["speed_current"] = 0.0
        job["speed_average"] = (done / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, total - done)
        speed = job["speed_current"] or job["speed_average"]
        job["eta_s"] = (remaining / speed) if speed > 0 else None
        job["free_bytes"] = _free_bytes(self.layout.root)
        job["low_space"] = _low_space(job["free_bytes"], job)
        self._put(job)

    def _flush_elapsed(self, job: dict, now: float | None = None) -> None:
        runtime = self._runtime.setdefault(job["id"], {})
        start = runtime.get("slice_start")
        if start is None:
            runtime["slice_start"] = time.monotonic()
            return
        current = time.monotonic() if now is None else now
        job["elapsed_s"] = float(job.get("elapsed_s") or 0) + max(0.0, current - start)
        runtime["slice_start"] = current

    def _finish_cancelled(self, job_id: str) -> None:
        job = self._require(job_id)
        self._drop_partial(job)
        job["status"] = "cancelled"
        job["phase"] = "done"
        self._flush_elapsed(job)
        self._log(job, "cancelled")

    def _finish_failed(self, job_id: str, exc: Exception) -> None:
        job = self._require(job_id)
        self._drop_partial(job)
        job["status"] = "failed"
        job["phase"] = "done"
        job["error"] = str(exc)
        self._flush_elapsed(job)
        self._log(job, f"failed: {exc}")

    def _drop_partial(self, job: dict) -> None:
        raw = str(job.get("partial") or "")
        job["partial"] = ""
        if not raw:
            return
        path = Path(raw)
        if path.name.endswith(PARTIAL_SUFFIX) and _is_within(path, self.layout.root):
            _unlink_partial(path)

    def _job_path(self, job_id: str) -> Path:
        return self.store_dir / "jobs" / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self.store_dir / "jobs" / f"{job_id}.log"


def _port_taken_by(layout: FleetLayout, port: int, name: str, port_in_use) -> bool:
    files = layout.files
    if files.is_dir():
        for child in files.iterdir():
            meta = child / "comfyfleet.json"
            if not meta.is_file():
                continue
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("name") or child.name) == name:
                continue
            try:
                if int(payload.get("port") or 0) == port:
                    return True
            except (TypeError, ValueError):
                continue
    if port_in_use is not None and port_in_use(port):
        return True
    return False


def _already_done(action: dict, mode: str) -> bool:
    dest = Path(action.get("dest") or "")
    source = Path(action.get("source") or "")
    if action["action"] in {"transfer", "conflict"}:
        if not dest.is_file():
            return False
        try:
            if dest.stat().st_size != int(action.get("size") or -1):
                return False
        except OSError:
            return False
        if mode == "move" and source.exists():
            return False
        return True
    if mode == "move":
        return not source.exists()
    return False


def _count_planned_files(mounts: list[dict], job: dict) -> tuple[int, int]:
    count = 0
    total = 0
    for mount in mounts:
        role = mount.get("role") or "skip"
        if role == "skip" or (role == "custom_nodes" and job.get("custom_nodes") == "fresh"):
            continue
        root = Path(mount["source"])
        if not root.is_dir():
            continue
        for path in _walk_files(root):
            rel = path.relative_to(root).as_posix()
            if role == "user":
                parts = PurePosixPath(rel).parts
                if "workflows" not in parts or path.suffix.lower() != ".json":
                    continue
            count += 1
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return count, total


def _visible_workflows(mounts: list[dict], *, docker=None, container: str = "") -> list[dict]:
    found = []
    for mount in mounts:
        if mount.get("role") not in {"workflows", "user"}:
            continue
        root = Path(mount["source"])
        if root.is_dir() and not root.is_symlink():
            for path in _walk_files(root):
                if path.suffix.lower() != ".json":
                    continue
                rel = path.relative_to(root).as_posix()
                if mount.get("role") == "user" and "workflows" not in PurePosixPath(rel).parts:
                    continue
                found.append({"path": str(path), "name": path.name})
            continue
        if docker is None or not container:
            continue
        try:
            names = _tar_rels(docker, container, mount["destination"])
        except (FleetError, OSError):
            continue
        for rel in names:
            if not rel.lower().endswith(".json"):
                continue
            if mount.get("role") == "user" and "workflows" not in PurePosixPath(rel).parts:
                continue
            container_path = str(PurePosixPath(mount["destination"]) / rel)
            found.append({"path": container_path, "name": PurePosixPath(rel).name})
    return found


def _assert_move_visible(mounts: list[dict], custom_nodes: str) -> None:
    for mount in mounts:
        role = mount.get("role") or "skip"
        if role == "skip" or (role == "custom_nodes" and custom_nodes == "fresh"):
            continue
        source = Path(mount["source"])
        if source.is_dir() and not source.is_symlink():
            continue
        raise FleetError(
            f"{mount['source']} is not visible to the manager. "
            "Move deletes a source file only after the destination hash matches, "
            "and that delete needs the host path. Use Copy."
        )


def _tar_rels(docker, container: str, destination: str) -> list[str]:
    proc = _open_tar(docker, container, destination)
    found: list[str] = []

    def handle(name: str, _size: int, _mtime: int, reader: _LimitedReader) -> None:
        rel = _tar_rel(name, destination)
        _drain(reader)
        if rel:
            found.append(rel)

    try:
        stdout = getattr(proc, "stdout", None)
        if stdout is None:
            raise FleetError(f"docker cp produced no output for {destination}")
        _for_each_tar_file(stdout, handle)
        _close_proc(proc)
    except Exception:
        _abort_proc(proc)
        raise
    return found


def _merge_mounts(inspected: list[dict], override) -> list[dict]:
    by_source = {item["source"]: dict(item) for item in inspected}
    if override is None:
        return list(by_source.values())
    if not isinstance(override, list):
        raise FleetError("mounts must be a list")
    merged = []
    for item in override:
        if not isinstance(item, dict):
            raise FleetError("each mount must be an object")
        source = str(item.get("source") or "")
        if source not in by_source:
            raise FleetError(f"refusing mount that is not on the container: {source}")
        role = str(item.get("role") or by_source[source]["role"])
        if role not in _ROLES:
            raise FleetError(f"unknown mount role {role!r}")
        row = dict(by_source[source])
        row["role"] = role
        merged.append(row)
    return merged


def _gpu_spec(requested, gpus: list) -> list[int]:
    available = {int(gpu.index) for gpu in gpus}
    if not requested:
        raise FleetError("GPU selection is required")
    if not isinstance(requested, list) or not requested:
        raise FleetError("gpus must be a list of indexes")
    picked = []
    for item in requested:
        try:
            index = int(item)
        except (TypeError, ValueError) as exc:
            raise FleetError(f"invalid GPU index {item!r}") from exc
        if index not in available:
            raise FleetError(f"GPU {index} is not on this host")
        if index not in picked:
            picked.append(index)
    if not picked:
        raise FleetError("GPU selection is required")
    return picked


def _container_path(destination: str, source_file: str, source_root: str) -> str:
    rel = Path(source_file).relative_to(source_root).as_posix()
    return str(PurePosixPath(destination) / rel)


def _published_ports(payload: dict) -> list[dict]:
    host = payload.get("HostConfig") if isinstance(payload.get("HostConfig"), dict) else {}
    ports = []
    bindings = host.get("PortBindings") or {}
    if isinstance(bindings, dict):
        for key, value in bindings.items():
            container_port = _port_key(key)
            for item in value or []:
                if not isinstance(item, dict):
                    continue
                host_port = _port_key(str(item.get("HostPort") or ""))
                if container_port and host_port:
                    ports.append({"container": container_port, "host": host_port})
    if ports:
        return ports
    network = payload.get("NetworkSettings") if isinstance(payload.get("NetworkSettings"), dict) else {}
    published = network.get("Ports") or {}
    if isinstance(published, dict):
        for key, value in published.items():
            container_port = _port_key(key)
            for item in value or []:
                if not isinstance(item, dict):
                    continue
                host_port = _port_key(str(item.get("HostPort") or ""))
                if container_port and host_port:
                    ports.append({"container": container_port, "host": host_port})
    return ports


def _port_key(value: str) -> int:
    text = str(value or "").split("/", 1)[0]
    if not text.isdigit():
        return 0
    port = int(text)
    if port < 1 or port > 65535:
        return 0
    return port


def _gpu_ids(host: dict, env: list) -> list[int]:
    found: list[int] = []
    for request in host.get("DeviceRequests") or []:
        if not isinstance(request, dict):
            continue
        for device in request.get("DeviceIDs") or []:
            if str(device).isdigit():
                index = int(device)
                if index not in found:
                    found.append(index)
    if found:
        return found
    for item in env:
        if not isinstance(item, str) or not item.startswith("NVIDIA_VISIBLE_DEVICES="):
            continue
        raw = item.split("=", 1)[1]
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                index = int(part)
                if index not in found:
                    found.append(index)
    return found


def _public_env(env: list) -> list[dict]:
    rows = []
    for item in env:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if any(token in key.upper() for token in _SECRET):
            value = "***"
        rows.append({"key": key, "value": value})
    return rows


def _free_instance_name(docker, name: str) -> str:
    """Prefer the old container name, then ``name-import``, when that name is free."""

    if docker.status(name) is None:
        return name
    alt = sanitize_stem(f"{name}-import")
    if docker.status(alt) is None:
        return alt
    return name


def _free_imported_rel(rel: str, index: dict) -> str:
    path = PurePosixPath(rel)
    parent = path.parent.as_posix()
    if parent == ".":
        parent = ""
    stem = path.stem
    suffix = path.suffix
    for number in range(1, 1000):
        label = f"{stem} (imported){suffix}" if number == 1 else f"{stem} (imported {number}){suffix}"
        candidate = f"{parent}/{label}" if parent else label
        if candidate not in index["by_rel"]:
            return candidate
    raise FleetError(f"no free import name for {rel}")


def _rel_posix(rel: str) -> str:
    path = PurePosixPath(rel)
    if path.is_absolute() or not path.parts:
        raise FleetError(f"invalid relative path {rel!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise FleetError(f"invalid relative path {rel!r}")
    return path.as_posix()


def _walk_files(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir() or root.is_symlink():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        kept = []
        for name in dirnames:
            path = current / name
            if path.is_symlink():
                continue
            if not _is_within(path, root):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            if name.endswith(PARTIAL_SUFFIX):
                continue
            if not _is_within(path, root):
                continue
            found.append(path)
    found.sort()
    return found


def _assert_readable(path: Path, fleet_root: Path, source_roots: list[Path]) -> None:
    if path.is_symlink():
        raise FleetError(f"refusing symlink {path}")
    if not path.is_file():
        raise FleetError(f"source file is missing: {path}")
    if _is_within(path, fleet_root):
        return
    for root in source_roots:
        if _is_within(path, root):
            return
    raise FleetError(f"refusing to read {path}: outside the fleet root and the source mounts")


def _assert_dest(path: Path, fleet_root: Path) -> None:
    if any(part in {"", ".", ".."} for part in path.parts if part != path.anchor):
        raise FleetError(f"refusing to write {path}")
    parent = path.parent
    if parent.exists() and not _is_within(parent, fleet_root):
        raise FleetError(f"refusing to write {path}: outside {fleet_root}")
    lexical = _lexical(path)
    root = _real_dir(fleet_root)
    if not _is_under_lexical(lexical, root):
        raise FleetError(f"refusing to write {path}: outside {fleet_root}")


def _chown_paths(paths: list[Path], owner: tuple[int, int] | None) -> None:
    if not owner:
        return
    uid, gid = owner
    for path in paths:
        try:
            os.chown(path, uid, gid, follow_symlinks=False)
        except OSError as exc:
            if os.environ.get("COMFYFLEET_MANAGER") == "1":
                raise FleetError(f"cannot chown {path} to comfyuser: {exc}") from exc


def _owner_ids() -> tuple[int, int] | None:
    try:
        return resolve_owner_ids(None, None)
    except FleetError:
        if os.environ.get("COMFYFLEET_MANAGER") == "1":
            raise
        return None


def _unlink_partial(path: Path) -> None:
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FleetError(f"cannot remove partial file {path}: {exc}") from exc


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.realpath(left) == os.path.realpath(right)
    except OSError:
        return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(Path(os.path.realpath(root)))
    except (OSError, ValueError):
        return False
    return True


def _is_under_lexical(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _lexical(path: Path) -> Path:
    parent = path.parent
    if parent.exists():
        base = Path(os.path.realpath(parent))
    else:
        base = Path(os.path.abspath(parent))
    return base / path.name


def _real_dir(path: Path) -> Path:
    if path.exists():
        return Path(os.path.realpath(path))
    return Path(os.path.abspath(path))


def _free_bytes(path: Path) -> int:
    probe = path if path.exists() else path.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return 0


def _low_space(free: int, job: dict) -> bool:
    remaining = max(0, int(job.get("bytes_total") or 0) - int(job.get("bytes_done") or 0))
    if free <= 0:
        return True
    if free < _LOW_FREE:
        return True
    return free < remaining + _MARGIN


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class _LimitedReader:
    def __init__(self, stream, size: int) -> None:
        self.stream = stream
        self.remaining = size

    def read(self, size: int) -> bytes:
        if self.remaining <= 0:
            return b""
        block = self.stream.read(min(size, self.remaining))
        if not block:
            raise FleetError("container archive ended early")
        self.remaining -= len(block)
        return block


class _MemberStat:
    """Enough of a path for the hash progress callback and the user-role filter."""

    def __init__(self, rel: str, size: int) -> None:
        self.rel = rel
        self.suffix = PurePosixPath(rel).suffix
        self._size = size

    def is_file(self) -> bool:
        return True

    def stat(self):
        return self

    @property
    def st_size(self) -> int:
        return self._size


def _for_each_tar_file(stream, handle, *, gate: Gate | None = None) -> None:
    long_name: str | None = None
    while True:
        if gate is not None:
            gate.checkpoint()
        header = _read_exact(stream, 512)
        if header is None or header == b"\0" * 512:
            return
        size = _tar_size(header)
        kind = header[156:157]
        if kind == b"L":
            data = _read_exact(stream, size) or b""
            _discard(stream, _tar_pad(size))
            long_name = data.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
            continue
        name = long_name if long_name is not None else _ustar_name(header)
        long_name = None
        mtime = _tar_mtime(header)
        if kind in {b"x", b"g", b"K"} or kind not in {b"0", b"\0"} or name.endswith("/"):
            _discard(stream, size + _tar_pad(size))
            continue
        reader = _LimitedReader(stream, size)
        handle(name, size, mtime, reader)
        if reader.remaining:
            _discard(stream, reader.remaining)
        _discard(stream, _tar_pad(size))


def _hash_reader(
    reader: _LimitedReader,
    size: int,
    mtime: int,
    cache: HashCache | None,
    key: Path,
    *,
    gate: Gate | None,
    on_chunk,
) -> str:
    mtime_ns = int(mtime) * 1_000_000_000
    if cache is not None:
        cached = cache.get(key, size, mtime_ns)
        if cached:
            _drain(reader)
            if on_chunk is not None:
                on_chunk(size, size)
            return cached
    digest = hashlib.sha256()
    done = 0
    while True:
        if gate is not None:
            gate.checkpoint()
        block = reader.read(_CHUNK)
        if not block:
            break
        digest.update(block)
        done += len(block)
        if on_chunk is not None:
            on_chunk(done, size)
    if done != size:
        raise FleetError("container archive ended early")
    hexdigest = digest.hexdigest()
    if cache is not None:
        cache.put(key, size, mtime_ns, hexdigest)
    return hexdigest


def _drain(reader: _LimitedReader) -> None:
    while reader.read(_CHUNK):
        pass


def _tar_rel(name: str, destination: str) -> str:
    text = name.replace("\\", "/").lstrip("/")
    while text.startswith("./"):
        text = text[2:]
    base = PurePosixPath(destination).name
    if text == base or text == base + "/":
        return ""
    if base and (text.startswith(base + "/") or text.startswith("./" + base + "/")):
        text = text.split(base + "/", 1)[1]
    if not text or text.endswith("/"):
        return ""
    try:
        return _rel_posix(text)
    except FleetError:
        return ""


def _docker_cache_key(container: str, container_path: str) -> Path:
    return Path("/comfyfleet-import-cache") / container / container_path.lstrip("/")


def _ustar_name(header: bytes) -> str:
    name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
    prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8", "surrogateescape").strip("\0")
    if prefix:
        return prefix.rstrip("/") + "/" + name
    return name


def _tar_mtime(header: bytes) -> int:
    raw = header[136:148].split(b"\0", 1)[0].strip() or b"0"
    try:
        return int(raw, 8)
    except ValueError:
        return 0


def _open_tar(docker, container: str, container_path: str):
    opener = getattr(docker, "open_container_tar", None)
    if opener is None:
        raise FleetError(
            f"cannot read {container_path}: the source mount is not visible and docker cp is unavailable"
        )
    return opener(container, container_path)


def _close_proc(proc) -> None:
    stdout = getattr(proc, "stdout", None)
    stderr = getattr(proc, "stderr", None)
    err = b""
    if stdout is not None:
        try:
            stdout.close()
        except Exception:
            pass
    if stderr is not None:
        try:
            err = stderr.read() or b""
        except Exception:
            err = b""
        try:
            stderr.close()
        except Exception:
            pass
    wait = getattr(proc, "wait", None)
    code = wait() if wait else 0
    if code not in (0, None):
        detail = err.decode("utf-8", "replace").strip()
        raise FleetError(detail or f"docker cp failed ({code})")


def _abort_proc(proc) -> None:
    kill = getattr(proc, "kill", None)
    if kill is not None:
        try:
            kill()
        except Exception:
            pass
    wait = getattr(proc, "wait", None)
    if wait is not None:
        try:
            wait()
        except Exception:
            pass


def _read_exact(stream, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining:
        block = stream.read(remaining)
        if not block:
            if not chunks:
                return None
            raise FleetError("container archive ended early")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _discard(stream, size: int) -> None:
    remaining = size
    while remaining:
        block = stream.read(min(_CHUNK, remaining))
        if not block:
            raise FleetError("container archive ended early")
        remaining -= len(block)


def _tar_size(header: bytes) -> int:
    raw = header[124:136].split(b"\0", 1)[0].strip() or b"0"
    try:
        return int(raw, 8)
    except ValueError as exc:
        raise FleetError("container archive has a bad size") from exc


def _tar_pad(size: int) -> int:
    extra = size % 512
    if extra == 0:
        return 0
    return 512 - extra
