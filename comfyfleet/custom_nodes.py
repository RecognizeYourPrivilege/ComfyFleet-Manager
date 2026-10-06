"""Create-time custom nodes on the instance volume.

Git clones and zip extracts write ``/home/ComfyFleet/custom_nodes_<name>`` (the host
side of the instance ``custom_nodes`` mount). They do not bake a new image.

Missing nodes are taken only from the create-time workflow JSON. Each one
that is not already in that volume or in the baked image set is installed
with one ``POST /customnode/install/git_url`` inside the instance — the
``COMFYFLEET_TRUSTED_INSTALL`` gate from PR #12. The Manager registry is
not installed as a catalog.

Blank URL lists and a missing zip are no-ops. Several zip files in one
create are extracted into the same ``custom_nodes_<name>`` directory.
Each archive can name its own folder. A blank name uses
``[project].name`` from that zip's ``pyproject.toml``. A typed name wins.
Clone, extract, and install failures are warnings. They do not raise.

Timeouts (seconds), overridable by environment:

| Step | Default | Variable |
|---|---|---|
| each ``git clone`` | 120 | ``COMFYFLEET_GIT_CLONE_TIMEOUT`` |
| wait until ComfyUI answers | 180 | ``COMFYFLEET_NODE_READY_TIMEOUT`` |
| each trusted git-URL install | 180 | ``COMFYFLEET_NODE_INSTALL_TIMEOUT`` |
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from comfyfleet.paths import CONTAINER_PORT

# Directory names the instance entrypoint symlinks from the baked image.
# A workflow that resolves to one of these is already on the image.
BAKED_NODE_DIRS = (
    "ComfyUI-Manager",
    "ComfyUI-Pixaroma",
    "ComfyUI-ComfyDock",
    "RES4LYF",
    "ComfyUI-Impact-Pack",
    "ComfyUI-Impact-Subpack",
    "comfyfleet_default_workflow",
)

DEFAULT_GIT_CLONE_TIMEOUT_S = 120.0
DEFAULT_NODE_READY_TIMEOUT_S = 180.0
DEFAULT_NODE_INSTALL_TIMEOUT_S = 180.0
_READY_PROBE_S = 5.0

MAX_ZIP_MEMBERS = 2000
MAX_ZIP_UNCOMPRESSED = 512 * 1024 * 1024
MAX_ZIP_MEMBER = 128 * 1024 * 1024
_MAX_PY_BYTES = 1_000_000

_OWNER_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_DIR = re.compile(r"^[A-Za-z0-9._+-]+$")
_WIN_DRIVE = re.compile(r"^[A-Za-z]:")

_INSTALL_PY = r"""
import json, sys, urllib.error, urllib.request
mode, port, timeout_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
base = "http://127.0.0.1:%s" % port
if mode == "ready":
    target = base + "/object_info"
    payload = None
    method = "GET"
    headers = {}
else:
    target = base + "/customnode/install/git_url"
    payload = json.dumps({"url": sys.argv[4]}).encode("utf-8")
    method = "POST"
    headers = {"Content-Type": "application/json"}
request = urllib.request.Request(target, data=payload, headers=headers, method=method)
try:
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response.read(65536)
        if response.status >= 300:
            sys.stderr.write("HTTP %s" % response.status)
            sys.exit(1)
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read(500).decode("utf-8", "replace"))
    sys.exit(1)
except Exception as exc:
    sys.stderr.write(str(exc))
    sys.exit(1)
"""


def partition_git_urls(values: list[str] | None) -> tuple[list[str], list[str]]:
    """Drop blanks. Invalid entries become warnings and are not cloned."""

    warnings: list[str] = []
    valid: list[str] = []
    for raw in values or []:
        text = str(raw).strip()
        if not text:
            continue
        if is_allowed_git_url(text):
            valid.append(text)
            continue
        warnings.append(
            "skipping custom node git URL "
            f"{text!r}; allowed schemes are https://, ssh://, and git@host:path"
        )
    return valid, warnings


def is_allowed_git_url(text: str) -> bool:
    """HTTPS, SSH (``ssh://``), or scp-style ``git@host:path``."""

    if not text or any(char.isspace() or ord(char) < 32 for char in text):
        return False
    if len(text) > 2000:
        return False
    if text.startswith("-"):
        return False
    if text.startswith("https://") or text.startswith("ssh://"):
        parts = urlsplit(text)
        if parts.scheme not in {"https", "ssh"}:
            return False
        return bool(parts.hostname) and bool(parts.path.strip("/"))
    if text.startswith("git@") and ":" in text[4:]:
        host, _, path = text[4:].partition(":")
        if not host or "/" in host or not path.strip("/"):
            return False
        return True
    return False


def clone_git_urls(
    urls: list[str] | None,
    dest_dir: Path,
    *,
    run=None,
    timeout: float | None = None,
) -> tuple[list[str], list[str]]:
    """Clone each URL into ``dest_dir/<repo>``. Returns warnings and cloned names.

    An existing directory is left in place. A failed clone removes the
    partial directory it created. ``git`` is invoked without a shell.
    """

    warnings: list[str] = []
    cloned: list[str] = []
    cleaned, invalid = partition_git_urls(urls)
    warnings.extend(invalid)
    if not cleaned:
        return warnings, cloned
    limit = timeout if timeout is not None else _env_seconds(
        "COMFYFLEET_GIT_CLONE_TIMEOUT", DEFAULT_GIT_CLONE_TIMEOUT_S, warnings
    )
    runner = run or subprocess.run
    if run is None and shutil.which("git") is None:
        warnings.append(
            "git is not installed on the manager; custom node clones were skipped. "
            "The instance was still created."
        )
        return warnings, cloned
    dest_dir.mkdir(parents=True, exist_ok=True)
    for url in cleaned:
        try:
            name = repo_dirname(url)
        except ValueError as exc:
            warnings.append(f"skipping custom node git URL {url!r}: {exc}")
            continue
        target = dest_dir / name
        if target.exists():
            warnings.append(
                f"custom node directory {name} already exists; left it in place and did not clone {url}"
            )
            continue
        argv = ["git", "clone", "--depth", "1", "--", url, str(target)]
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
        try:
            completed = runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=limit,
                env=env,
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(target, ignore_errors=True)
            warnings.append(f"git clone timed out after {limit:g}s: {url}")
            continue
        except OSError as exc:
            shutil.rmtree(target, ignore_errors=True)
            warnings.append(f"git clone failed for {url}: {exc}")
            continue
        if getattr(completed, "returncode", 1) != 0:
            shutil.rmtree(target, ignore_errors=True)
            detail = (getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or "").strip()
            warnings.append(f"git clone failed for {url}: {detail or 'non-zero exit'}")
            continue
        cloned.append(name)
    return warnings, cloned


def extract_custom_nodes_zip(
    payload: bytes | None,
    dest_dir: Path,
    *,
    directory: str | None = None,
) -> tuple[list[str], bool]:
    """Extract a custom-node zip into ``dest_dir``.

    Any absolute path, ``..``, NUL, symlink, or encrypted member rejects the
    whole archive: nothing from that zip is written. A malformed zip is a
    warning. ``payload is None`` (field absent) is a no-op.

    ``directory`` is the folder under ``dest_dir``. Blank uses
    ``[project].name`` from that archive's ``pyproject.toml`` when the file
    identifies one pack. A typed directory wins over that name. With neither,
    members keep the paths stored in the zip.
    """

    if payload is None:
        return [], False
    if payload == b"":
        return ["custom_nodes_zip was empty; nothing was extracted"], False
    try:
        archive_cm = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        return ["custom_nodes_zip is not a valid zip; nothing was extracted"], False
    with archive_cm as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_MEMBERS:
            return [
                f"custom_nodes_zip has {len(infos)} members; refused (limit {MAX_ZIP_MEMBERS})"
            ], False
        total = 0
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            reason = _unsafe_zip_member(info)
            if reason:
                return [
                    f"custom_nodes_zip rejected ({reason}): {info.filename!r}"
                ], False
            size = info.file_size
            if size > MAX_ZIP_MEMBER:
                return [
                    f"custom_nodes_zip rejected (member larger than {MAX_ZIP_MEMBER} bytes): {info.filename!r}"
                ], False
            total += size
            if total > MAX_ZIP_UNCOMPRESSED:
                return [
                    "custom_nodes_zip rejected (uncompressed size exceeds "
                    f"{MAX_ZIP_UNCOMPRESSED} bytes)"
                ], False
            relative = _zip_relative(info.filename)
            if relative is None:
                continue
            root = dest_dir.resolve()
            target = (dest_dir / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                return [
                    f"custom_nodes_zip rejected (path escapes custom_nodes): {info.filename!r}"
                ], False
            if not info.is_dir():
                planned.append((info, relative))
        chosen, name_error = _resolve_pack_directory(directory, archive, planned)
        if name_error:
            return [name_error], False
        if chosen:
            stripped = _strip_single_root([relative for _info, relative in planned])
            planned = [
                (info, Path(chosen, *relative.parts))
                for (info, _old), relative in zip(planned, stripped, strict=True)
            ]
        writes: list[tuple[zipfile.ZipInfo, Path]] = []
        root = dest_dir.resolve()
        for info, relative in planned:
            target = (dest_dir / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                return [
                    f"custom_nodes_zip rejected (path escapes custom_nodes): {info.filename!r}"
                ], False
            writes.append((info, target))
        if not writes:
            return ["custom_nodes_zip contained no files; nothing was extracted"], False
        dest_dir.mkdir(parents=True, exist_ok=True)
        for info, target in writes:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return [], True


def extract_custom_nodes_zips(
    payloads: bytes | Sequence[bytes] | None,
    dest_dir: Path,
    *,
    names: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> tuple[list[str], bool]:
    """Extract one or more custom-node zips into ``dest_dir``.

    A single ``bytes`` value is the one-archive path. Each archive is
    checked on its own: a rejected archive writes nothing from that zip,
    and archives that pass are still extracted into the same directory.
    ``names`` lines up with the archives. A blank entry uses that zip's
    ``pyproject.toml`` project name. A typed entry wins. ``None`` and an
    empty sequence are no-ops.
    """

    if payloads is None:
        return [], False
    if isinstance(payloads, (bytes, bytearray)):
        items = [bytes(payloads)]
    else:
        items = [bytes(item) for item in payloads]
    if not items:
        return [], False
    requested = [names] if isinstance(names, str) else list(names or [])
    shown = [labels] if isinstance(labels, str) else list(labels or [])
    warnings: list[str] = []
    extracted_any = False
    for index, payload in enumerate(items):
        directory = requested[index] if index < len(requested) else None
        if directory is not None and not str(directory).strip():
            directory = None
        one_warnings, extracted = extract_custom_nodes_zip(
            payload,
            dest_dir,
            directory=directory,
        )
        label = shown[index] if index < len(shown) else ""
        label = str(label).strip() if label else ""
        if label:
            one_warnings = [f"{label}: {item}" for item in one_warnings]
        warnings.extend(one_warnings)
        extracted_any = extracted_any or extracted
    return warnings, extracted_any


def plan_missing_installs(
    workflow: dict,
    custom_nodes_dir: Path,
    node_map: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Git URLs for workflow nodes that are not already present.

    Presence is a class name in a ``NODE_CLASS_MAPPINGS`` on the volume, or
    a pack directory already on that volume or in the baked image. Only
    types referenced by ``workflow`` are considered. ``node_map`` maps those
    class names to git URLs (a direct map, or Manager's extension-node-map
    shape). Entries that the workflow does not reference are ignored.
    """

    warnings: list[str] = []
    present_types = class_types_in_tree(custom_nodes_dir)
    present_dirs = set(BAKED_NODE_DIRS)
    if custom_nodes_dir.is_dir():
        present_dirs.update(path.name for path in custom_nodes_dir.iterdir() if path.is_dir())
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        if not is_allowed_git_url(url):
            warnings.append(
                f"skipping unresolved custom node URL {url!r}; "
                "allowed schemes are https://, ssh://, and git@host:path"
            )
            return
        try:
            directory = repo_dirname(url)
        except ValueError as exc:
            warnings.append(f"skipping custom node URL {url}: {exc}")
            return
        if directory in present_dirs:
            return
        if url in seen:
            return
        seen.add(url)
        urls.append(url)

    referenced: set[str] = set()
    for node in iter_workflow_nodes(workflow):
        class_type = _class_type(node)
        if class_type:
            referenced.add(class_type)
        if class_type and class_type in present_types:
            continue
        for raw in _embedded_git_refs(node):
            coerced, warning = coerce_git_reference(raw)
            if warning:
                warnings.append(warning)
            if coerced:
                add(coerced)
    if node_map:
        for class_type in referenced:
            if class_type in present_types:
                continue
            mapped = node_map.get(class_type)
            if isinstance(mapped, str) and mapped.strip():
                add(mapped.strip())
    return urls, warnings


def class_types_in_tree(root: Path) -> set[str]:
    """String keys of ``NODE_CLASS_MAPPINGS`` under ``root``. Imports nothing."""

    found: set[str] = set()
    if not root.is_dir():
        return found
    for path in root.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            if path.stat().st_size > _MAX_PY_BYTES:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS":
                    found.update(_dict_string_keys(node.value))
    return found


def iter_workflow_nodes(workflow: dict, _depth: int = 0):
    """UI ``nodes`` entries, otherwise API-format prompt nodes, plus subgraphs."""

    if _depth > 8 or not isinstance(workflow, dict):
        return
    nodes = workflow.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                yield node
    else:
        for node in workflow.values():
            if isinstance(node, dict) and _class_type(node):
                yield node
    definitions = workflow.get("definitions")
    if isinstance(definitions, dict):
        subgraphs = definitions.get("subgraphs")
        if isinstance(subgraphs, list):
            for subgraph in subgraphs:
                if isinstance(subgraph, dict):
                    yield from iter_workflow_nodes(subgraph, _depth + 1)


def parse_node_map(payload: object) -> dict[str, str]:
    """Class name → git URL.

    Accepts either ``{"Class": "https://..."}`` or Manager's
    extension-node-map object ``{"https://...": [["Class"], {...}]}``.
    """

    if not isinstance(payload, dict):
        raise ValueError("extension node map must be a JSON object")
    if payload and all(isinstance(value, str) for value in payload.values()):
        return {str(key): value for key, value in payload.items() if isinstance(key, str)}
    mapped: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not is_allowed_git_url(key):
            continue
        names: list[str] = []
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, list):
                names = [item for item in first if isinstance(item, str)]
            else:
                names = [item for item in value if isinstance(item, str)]
        for name in names:
            mapped.setdefault(name, key)
    return mapped


def node_map_from_env() -> tuple[dict[str, str] | None, list[str]]:
    """Load ``COMFYFLEET_EXTENSION_NODE_MAP`` when it is set. Unset is no map."""

    raw = os.environ.get("COMFYFLEET_EXTENSION_NODE_MAP", "").strip()
    if not raw:
        return None, []
    path = Path(raw)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_node_map(payload), []
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, [
            f"COMFYFLEET_EXTENSION_NODE_MAP ({raw}) was not loaded: {exc}. "
            "Missing-node install will use git URLs embedded in the workflow only."
        ]


def trusted_manager_install(
    urls: list[str],
    *,
    docker,
    name: str,
    ready_timeout: float | None = None,
    install_timeout: float | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[list[str], list[str]]:
    """POST each URL to in-container ``/customnode/install/git_url``.

    The request runs inside the instance, where the entrypoint has exported
    ``COMFYFLEET_TRUSTED_INSTALL=1``. This does not call Manager's node list
    or install any URL that was not passed in.
    """

    warnings: list[str] = []
    installed: list[str] = []
    if not urls:
        return warnings, installed
    if getattr(docker, "exec", None) is None:
        warnings.append(
            "trusted install skipped: this docker client cannot exec into the instance. "
            "Missing workflow nodes were not installed."
        )
        return warnings, installed
    ready_limit = ready_timeout if ready_timeout is not None else _env_seconds(
        "COMFYFLEET_NODE_READY_TIMEOUT", DEFAULT_NODE_READY_TIMEOUT_S, warnings
    )
    install_limit = install_timeout if install_timeout is not None else _env_seconds(
        "COMFYFLEET_NODE_INSTALL_TIMEOUT", DEFAULT_NODE_INSTALL_TIMEOUT_S, warnings
    )
    if not _wait_until_ready(docker, name, ready_limit, sleep=sleep, clock=clock):
        warnings.append(
            "ComfyUI did not become ready for trusted custom-node install "
            f"within {ready_limit:g}s; missing workflow nodes were not installed"
        )
        return warnings, installed
    for url in urls:
        error = _post_git_url(docker, name, url, install_limit)
        if error:
            warnings.append(f"trusted install failed for {url}: {error}")
            continue
        installed.append(url)
    return warnings, installed


def coerce_git_reference(value: str) -> tuple[str | None, str | None]:
    """Turn ``owner/repo`` or a git URL into a clone URL. ``(None, None)`` ignores it."""

    text = value.strip()
    if not text:
        return None, None
    if is_allowed_git_url(text):
        return text, None
    if _OWNER_REPO.fullmatch(text) and ".." not in text.split("/"):
        return f"https://github.com/{text}", None
    if "://" in text:
        return None, (
            f"skipping unsupported custom node URL {text!r}; "
            "allowed schemes are https://, ssh://, and git@host:path"
        )
    return None, None


def repo_dirname(url: str) -> str:
    if url.startswith("git@") and ":" in url:
        path = url.split(":", 1)[1]
    else:
        path = urlsplit(url).path
    name = path.strip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[: -len(".git")]
    if not name or not _SAFE_DIR.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"cannot derive a safe directory name from {url}")
    return name


def _embedded_git_refs(node: dict) -> list[str]:
    bags = [node]
    properties = node.get("properties")
    if isinstance(properties, dict):
        bags.append(properties)
    found: list[str] = []
    for bag in bags:
        for key in ("aux_id", "git_url", "repository", "repo", "Git URL"):
            value = bag.get(key)
            if isinstance(value, str) and value.strip():
                found.append(value.strip())
    return found


def _class_type(node: dict) -> str | None:
    for key in ("class_type", "type"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dict_string_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _unsafe_zip_member(info: zipfile.ZipInfo) -> str | None:
    name = info.filename
    if "\x00" in name:
        return "NUL in member path"
    if info.flag_bits & 0x1:
        return "encrypted member"
    mode = (info.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        return "symlink member"
    raw = name.replace("\\", "/")
    if raw.startswith("/") or raw.startswith("//"):
        return "absolute path"
    if _WIN_DRIVE.match(raw):
        return "absolute path"
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return "path traversal"
    return None


def _resolve_pack_directory(
    requested: str | None,
    archive: zipfile.ZipFile,
    members: list[tuple[zipfile.ZipInfo, Path]],
) -> tuple[str | None, str | None]:
    """Return ``(directory, error)``. An error means this zip writes nothing.

    A published Comfy node zip does not store the pack id in JSON.
    ``node_list.json`` maps renamed node classes. The id ComfyUI-Manager
    uses as the ``custom_nodes`` subdirectory is ``[project].name`` in
    ``pyproject.toml``. ``[tool.comfy].DisplayName`` is a label, not a folder.
    """

    if requested is not None and requested.strip():
        chosen = _pack_directory_name(requested)
        if chosen is None:
            return None, f"custom_nodes_zip name is invalid: {requested.strip()!r}"
        return chosen, None
    discovered = _project_name_from_zip(archive, members)
    if not discovered:
        return None, None
    chosen = _pack_directory_name(discovered)
    if chosen is None:
        return None, f"custom_nodes_zip name from pyproject.toml is invalid: {discovered!r}"
    return chosen, None


def _project_name_from_zip(
    archive: zipfile.ZipFile,
    members: list[tuple[zipfile.ZipInfo, Path]],
) -> str | None:
    candidates = [
        (info, relative)
        for info, relative in members
        if relative.name == "pyproject.toml" and len(relative.parts) <= 2
    ]
    if len(candidates) != 1:
        return None
    info, relative = candidates[0]
    if len(relative.parts) == 2:
        root = relative.parts[0]
        if any(path.parts[0] != root for _info, path in members):
            return None
    try:
        raw = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return None
    if len(raw) > _MAX_PY_BYTES:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _project_name_from_toml(text)


def _project_name_from_toml(text: str) -> str | None:
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    project = payload.get("project")
    if not isinstance(project, dict):
        return None
    name = project.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name or None


def _pack_directory_name(text: str) -> str | None:
    name = text.strip()
    if (
        not name
        or name.startswith(".")
        or not _SAFE_DIR.fullmatch(name)
        or name in {".", ".."}
    ):
        return None
    return name


def _strip_single_root(paths: list[Path]) -> list[Path]:
    """Drop one shared top directory so a GitHub zip lands next to its files."""

    if not paths:
        return paths
    if any(len(path.parts) < 2 for path in paths):
        return paths
    roots = {path.parts[0] for path in paths}
    if len(roots) != 1:
        return paths
    return [Path(*path.parts[1:]) for path in paths]


def _zip_relative(name: str) -> Path | None:
    raw = name.replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if not parts or raw.endswith("/"):
        return None
    return Path(*parts)


def _env_seconds(name: str, default: float, warnings: list[str]) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        warnings.append(f"{name} must be a positive number of seconds; using {default:g}")
        return default
    if value <= 0:
        warnings.append(f"{name} must be a positive number of seconds; using {default:g}")
        return default
    return value


def _wait_until_ready(docker, name: str, timeout: float, *, sleep, clock) -> bool:
    deadline = clock() + timeout
    while True:
        if _exec_ok(docker, name, ["ready", str(CONTAINER_PORT), str(_READY_PROBE_S)], _READY_PROBE_S + 5):
            return True
        if clock() >= deadline:
            return False
        sleep(min(2.0, max(0.0, deadline - clock())))


def _post_git_url(docker, name: str, url: str, timeout: float) -> str | None:
    completed = _exec(
        docker,
        name,
        ["install", str(CONTAINER_PORT), str(timeout), url],
        timeout + 15,
    )
    if completed is None:
        return "docker exec is not available on this client"
    if getattr(completed, "returncode", 1) == 0:
        return None
    detail = (getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or "").strip()
    return detail or "non-zero exit"


def _exec_ok(docker, name: str, args: list[str], timeout: float) -> bool:
    completed = _exec(docker, name, args, timeout)
    return completed is not None and getattr(completed, "returncode", 1) == 0


def _exec(docker, name: str, args: list[str], timeout: float):
    from comfyfleet.errors import FleetError

    command = ["/opt/venv/bin/python", "-c", _INSTALL_PY, *args]
    exec_in = getattr(docker, "exec", None)
    if exec_in is None:
        return None
    try:
        return exec_in(name, command, timeout=timeout)
    except (FleetError, subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess(command, 1, "", str(exc))

