"""Allowlisted host ownership for ComfyFleet.

``comfyfleet fix-owner`` and ``POST /api/host/fix-owner`` both call
:func:`fix_owner`. The allowlist is fixed: there is no path argument.
Recursive ``chown`` runs only on those trees, and a path that escapes them
is refused before it is chowned. A symlink whose target is the image
directory ``/opt/comfyfleet/baked_custom_nodes`` (ComfyUI-Manager and the
other baked custom nodes) is not followed and does not abort the walk.

Instance create uses :func:`ensure_wildcards_dir` and
:func:`ensure_instance_host_dirs`. Each creates a missing directory under
the fleet root and chowns that new directory to ``comfyuser:comfyuser``.
An existing directory is left alone. The storage root is not chowned.
"""

from __future__ import annotations

import os
import pwd
import grp
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from comfyfleet.errors import FleetError
from comfyfleet.paths import HOST_ROOT, FleetLayout

DEFAULT_OWNER_NAME = "comfyuser"
DEFAULT_GROUP_NAME = "comfyuser"
_ACCOUNT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,31}\Z")
# Absolute directory inside the instance image. The entrypoint symlinks each
# baked custom node into the host ``custom_nodes_*`` volume with this prefix.
# It is not a host path under ``/home/ComfyFleet``.
BAKED_CUSTOM_NODES = Path("/opt/comfyfleet/baked_custom_nodes")
_CUSTOM_NODES_PREFIX = "custom_nodes_"
_NODE_SUFFIX = re.compile(r"[A-Za-z0-9._-]+\Z")


@dataclass(frozen=True)
class FixOwnerResult:
    uid: int
    gid: int
    user: str
    group: str
    paths: tuple[str, ...]


def normalize_owner_names(user: str | None, group: str | None) -> tuple[str, str]:
    """User and group for a chown. Blank values are ``comfyuser``."""

    return (
        _account_name(user, DEFAULT_OWNER_NAME, "user"),
        _account_name(group, DEFAULT_GROUP_NAME, "group"),
    )


def _account_name(value: str | None, default: str, kind: str) -> str:
    text = (value or "").strip()
    if not text:
        return default
    if _ACCOUNT_NAME.fullmatch(text) is None:
        raise FleetError(
            f"host {kind} name is invalid: {text!r}. "
            "Use a letter or underscore, then letters, digits, underscore, or hyphen."
        )
    return text


def resolve_owner_ids(
    user: str | None = None,
    group: str | None = None,
    *,
    run=None,
) -> tuple[int, int]:
    """Uid and gid for ``user:group``.

    Blank names are ``comfyuser``. A missing user or group is created with
    ``useradd`` and ``groupadd`` (the same tools on Arch and on the Debian
    manager image). ``useradd`` uses home ``/home/ComfyFleet`` and does not
    create ``/home/<name>``.
    """

    owner, group_name = normalize_owner_names(user, group)
    if _named_user(owner) is None or _named_group(group_name) is None:
        _ensure_account(owner, group_name, run or _run_account_tool)
    user_row = _named_user(owner)
    group_row = _named_group(group_name)
    if user_row is None:
        raise FleetError(_unresolved("user", owner))
    if group_row is None:
        raise FleetError(_unresolved("group", group_name))
    return user_row.pw_uid, group_row.gr_gid


def _named_user(name: str):
    try:
        return pwd.getpwnam(name)
    except KeyError:
        return None


def _named_group(name: str):
    try:
        return grp.getgrnam(name)
    except KeyError:
        return None


def _ensure_account(user: str, group: str, run) -> None:
    if _named_group(group) is None:
        binary = _account_binary("groupadd", "group", group)
        detail = _invoke_account_tool(run, [binary, group])
        if _named_group(group) is None:
            raise FleetError(
                _unresolved("group", group, detail or "groupadd did not create the group.")
            )
    if _named_user(user) is None:
        binary = _account_binary("useradd", "user", user)
        detail = _invoke_account_tool(run, _useradd_argv(binary, user, group))
        if _named_user(user) is None:
            raise FleetError(
                _unresolved("user", user, detail or "useradd did not create the user.")
            )


def _useradd_argv(binary: str, user: str, group: str) -> list[str]:
    """``useradd`` arguments that do not add a ``/home/<user>`` directory."""

    return [
        binary,
        "--no-create-home",
        "--no-user-group",
        "--gid",
        group,
        "--home-dir",
        str(HOST_ROOT),
        user,
    ]


def _account_binary(tool: str, kind: str, account: str) -> str:
    found = shutil.which(tool)
    if found:
        return found
    for candidate in (f"/usr/sbin/{tool}", f"/sbin/{tool}", f"/usr/bin/{tool}"):
        if os.access(candidate, os.X_OK):
            return candidate
    raise FleetError(_unresolved(kind, account, f"{tool} is not available."))


def _unresolved(kind: str, name: str, detail: str = "") -> str:
    message = (
        f"cannot resolve host {kind} {name} by name. "
        f"Creating the {name} {kind} failed"
    )
    if detail:
        return f"{message}: {detail}"
    return f"{message}."


def _invoke_account_tool(run, argv: list[str]) -> str:
    """Run ``argv``. Return ``""`` on success, or a failure detail."""

    try:
        completed = run(argv)
    except OSError as exc:
        return str(exc)
    if getattr(completed, "returncode", 1) == 0:
        return ""
    stderr = getattr(completed, "stderr", "") or ""
    stdout = getattr(completed, "stdout", "") or ""
    detail = stderr.strip() or stdout.strip()
    if detail:
        return detail
    return f"exit {completed.returncode}"


def _run_account_tool(argv: list[str]):
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def ensure_wildcards_dir(layout: FleetLayout) -> bool:
    """Create the shared wildcards directory only when it is missing.

    Returns True when this call created it and chowned that directory
    (not its children). An existing directory, including one that already
    has files, is left alone. A second create does not ``chown -R``.
    The storage root is created when it is missing and is not chowned.
    """

    path = layout.wildcards
    if path.is_symlink() or path.exists():
        return False
    created = create_new_host_dirs(path, layout.root)
    if not created:
        return False
    chown_new_directories(created)
    return True


def ensure_instance_host_dirs(
    layout: FleetLayout,
    name: str,
    model_subdirs: Sequence[str],
) -> list[Path]:
    """Create instance host directories and chown the ones this call made.

    ``models`` and each model subdirectory, ``custom_nodes_<name>``, and
    ``files/<name>/{input,output,temp}`` (including the ``files`` and
    instance parents when those are missing). An existing directory is
    left alone. The storage root is not chowned. Returns the directories
    this call created.
    """

    created: list[Path] = []
    for path in (
        layout.models,
        *[layout.models / sub for sub in model_subdirs],
        layout.custom_nodes(name),
        layout.input_dir(name),
        layout.output_dir(name),
        layout.temp_dir(name),
    ):
        created.extend(create_new_host_dirs(path, layout.root))
    chown_new_directories(created)
    return created


def create_new_host_dirs(path: Path, root: Path) -> list[Path]:
    """Create ``path`` and missing parents beneath ``root``.

    Returns the directories this call created, nearest to ``root`` first.
    ``root`` is created when it is missing so the child can be made, and
    it is not included in the result. An existing ``path``, including a
    symlink, is left alone and yields an empty list.
    """

    root_norm = _abs(root)
    target = _abs(path)
    if target == root_norm:
        return []
    if not _is_under(target, root_norm):
        raise FleetError(f"refusing to create {path} outside {root_norm}")
    if path.is_symlink() or path.exists():
        return []

    missing: list[Path] = []
    current = target
    while current != root_norm:
        if current.exists() or current.is_symlink():
            break
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    missing.reverse()

    if not root_norm.exists() and not root_norm.is_symlink():
        root_norm.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for directory in missing:
        if directory.exists() or directory.is_symlink():
            continue
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        created.append(directory)
    return created


def chown_new_directory(path: Path) -> None:
    """``chown comfyuser:comfyuser`` on one directory. Not recursive.

    The manager entrypoint exports ``COMFYFLEET_MANAGER=1`` and runs as root.
    In that process a failed account create or a failed chown is an error.
    A missing ``comfyuser`` user is created first. A unit test or a host
    checkout without that variable leaves the new directory as-is when
    create or chown fails. ``fix_owner`` does not use this skip.
    """

    chown_new_directories([path])


def chown_new_directories(paths: Sequence[Path]) -> None:
    """``chown comfyuser:comfyuser`` on each new directory. Not recursive.

    The account is resolved once. See :func:`chown_new_directory` for the
    manager-process error rule. An empty list does nothing.
    """

    if not paths:
        return
    strict = os.environ.get("COMFYFLEET_MANAGER") == "1"
    owner, group_name = normalize_owner_names(None, None)
    try:
        uid, gid = resolve_owner_ids(owner, group_name)
    except FleetError:
        if strict:
            raise
        return
    for path in paths:
        try:
            os.chown(path, uid, gid, follow_symlinks=False)
        except OSError as exc:
            if not strict:
                return
            raise FleetError(
                f"cannot chown {path} to {owner}:{group_name} ({uid}:{gid}): {exc}. "
                "The manager image runs as root so it can set the owner of a directory it just created."
            ) from exc


def fix_owner(
    layout: FleetLayout,
    *,
    user: str | None = None,
    group: str | None = None,
    resolve=None,
    chown=None,
) -> FixOwnerResult:
    """Recursively chown the fixed allowlist.

    Blank ``user`` and ``group`` are ``comfyuser``. No caller-supplied
    path. Roots outside ``layout.root`` are refused. Symlinks into
    ``/opt/comfyfleet/baked_custom_nodes`` do not fail the walk and are
    not followed.
    """

    from comfyfleet.control import authorize

    authorize("fix-owner")
    owner, group_name = normalize_owner_names(user, group)
    if resolve is None:
        uid, gid = resolve_owner_ids(owner, group_name)
    else:
        uid, gid = resolve()
    actor = chown or chown_inode
    changed: list[str] = []
    for path in allowlisted_roots(layout):
        assert_allowlisted(path, layout)
        chown_tree(path, uid, gid, layout, actor)
        changed.append(str(_abs(path)))
    return FixOwnerResult(
        uid=uid,
        gid=gid,
        user=owner,
        group=group_name,
        paths=tuple(changed),
    )


def allowlisted_roots(layout: FleetLayout) -> list[Path]:
    """Existing allowlisted directories under the fleet root.

    ``/home/ComfyFleet/wildcards``, ``/home/ComfyFleet/models``,
    ``/home/ComfyFleet/files``, and every ``/home/ComfyFleet/custom_nodes_*``
    directory. Missing entries are skipped.
    """

    found: list[Path] = []
    for path in (layout.wildcards, layout.models, layout.files):
        if path.is_symlink() or path.exists():
            found.append(path)
    root = _abs(layout.root)
    if root.is_dir():
        try:
            children = list(root.iterdir())
        except OSError as exc:
            raise FleetError(f"cannot list {root}: {exc}") from exc
        for child in sorted(children, key=lambda item: item.name):
            if _is_custom_nodes_name(child.name) and (child.is_symlink() or child.is_dir()):
                found.append(child)
    return found


def assert_allowlisted(path: Path, layout: FleetLayout) -> Path:
    """Raise ``FleetError`` unless ``path`` stays inside an allowlisted root.

    ``..``, absolute paths outside the fleet root, and symlinks whose
    target leaves the allowlist are refused. A symlink that resolves to
    ``/opt/comfyfleet/baked_custom_nodes`` or a path under it is the
    instance entrypoint's link to an image directory. That link is not an
    escape. The caller chowns the symlink inode and does not follow it.
    The returned path is normalized without following symlinks.
    """

    root = _abs(layout.root)
    normalized = _abs(path)
    if normalized != root and not _is_under(normalized, root):
        raise FleetError(f"refusing path outside the {root} allowlist: {path}")
    if not _is_allowlisted_lexical(normalized, layout):
        raise FleetError(f"refusing path outside the fix-owner allowlist: {path}")
    if _is_baked_custom_node_symlink(path):
        return normalized
    if path.is_symlink() or path.exists():
        real = Path(os.path.realpath(path))
        real_norm = _abs(real)
        if not _is_under(real_norm, root) or not _is_allowlisted_lexical(real_norm, layout):
            raise FleetError(
                f"refusing symlink that leaves the fix-owner allowlist: {path} -> {real}"
            )
    return normalized


def chown_tree(root: Path, uid: int, gid: int, layout: FleetLayout, chown) -> None:
    """``chown`` ``root`` and its descendants. Symlinks are not followed.

    A symlink into the image baked custom nodes is chowned as a symlink
    inode and is not entered. Any other symlink whose target leaves the
    allowlist is refused before it is chowned.
    """

    assert_allowlisted(root, layout)
    if not root.exists() and not root.is_symlink():
        return
    if root.is_symlink() or not root.is_dir():
        chown(root, uid, gid)
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        assert_allowlisted(current, layout)
        chown(current, uid, gid)
        for name in filenames:
            path = current / name
            assert_allowlisted(path, layout)
            chown(path, uid, gid)
        kept: list[str] = []
        for name in dirnames:
            path = current / name
            assert_allowlisted(path, layout)
            if path.is_symlink():
                chown(path, uid, gid)
                continue
            kept.append(name)
        dirnames[:] = kept


def chown_inode(path: Path, uid: int, gid: int) -> None:
    """Chown one inode. Does not follow a symlink."""

    try:
        os.chown(path, uid, gid, follow_symlinks=False)
    except OSError as exc:
        raise FleetError(f"cannot chown {path} to {uid}:{gid}: {exc}") from exc


def _is_baked_custom_node_symlink(path: Path) -> bool:
    """True when ``path`` is a symlink into the image baked custom nodes.

    The resolved target is compared with the literal baked directory.
    ``..`` and a lookalike prefix such as ``baked_custom_nodes_evil`` do
    not match. The baked directory itself is not realpath'd, so a host
    symlink planted at that path cannot widen the match onto another tree.
    """

    if not path.is_symlink():
        return False
    real = _abs(Path(os.path.realpath(path)))
    baked = _abs(BAKED_CUSTOM_NODES)
    return _is_under(real, baked)


def _is_custom_nodes_name(name: str) -> bool:
    if not name.startswith(_CUSTOM_NODES_PREFIX):
        return False
    suffix = name[len(_CUSTOM_NODES_PREFIX) :]
    if not suffix or suffix in {".", ".."} or ".." in suffix:
        return False
    return _NODE_SUFFIX.fullmatch(suffix) is not None


def _abs(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _named_roots(layout: FleetLayout) -> tuple[Path, ...]:
    root = _abs(layout.root)
    return (
        _abs(layout.wildcards),
        _abs(layout.models),
        _abs(layout.files),
    )


def _is_allowlisted_lexical(path: Path, layout: FleetLayout) -> bool:
    """True when ``path`` is an allowlisted root or a descendant of one.

    ``path`` must already be absolute and normalized without symlink resolution.
    """

    for candidate in _named_roots(layout):
        if path == candidate or _is_under(path, candidate):
            return True
    parent = path if path.parent == path else path
    # A descendant of custom_nodes_* matches on the first path component
    # under the fleet root. The root directory itself is not allowlisted.
    root = _abs(layout.root)
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    parts = relative.parts
    if not parts:
        return False
    if not _is_custom_nodes_name(parts[0]):
        return False
    return True
