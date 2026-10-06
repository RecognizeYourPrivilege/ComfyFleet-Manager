"""Gallery of files in each instance output directory.

The manager sees the host storage root at ``/home/ComfyFleet`` (bind-mounted
there; nothing the fleet owns is a sibling of that root). Create puts one
instance's ComfyUI output at ``<root>/files/<name>/output``. This module
finds those directories by scanning ``files/`` and the fleet records. It
does not take a configured path list.

Listing, reads, and deletes stay inside the real output directory. A
``..`` segment, a symlink, or a file whose opened descriptor resolves
outside that directory is refused. Video and animated files are never
returned as the thumbnail body.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
import zlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from pathlib import Path

from comfyfleet.auth import AuthError
from comfyfleet.control import authorize, list_instances
from comfyfleet.errors import FleetError
from comfyfleet.naming import is_instance_name
from comfyfleet.paths import FleetLayout

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}
ANIMATED_SUFFIXES = {".gif"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES | ANIMATED_SUFFIXES

DEFAULT_LIMIT = 48
MAX_LIMIT = 240
_MAX_RELATIVE = 1024
_MAX_COMPONENT = 255
_MAX_WALK_DEPTH = 32
_GIF_POSTER_LIMIT = 2_000_000
_POSTER_LOCKS: dict[str, threading.Lock] = {}
_POSTER_LOCKS_GUARD = threading.Lock()

_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


class GalleryMissing(FleetError):
    """The instance output directory or the requested file is not there."""


@dataclass(frozen=True)
class GalleryFile:
    """An open regular file that was checked against an output directory."""

    fd: int
    size: int
    content_type: str
    filename: str
    kind: str


@dataclass(frozen=True)
class _Output:
    name: str
    root: Path
    port: int | None
    status: str | None


def list_gallery(
    layout: FleetLayout,
    docker: object,
    *,
    instance: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """Media under discovered output directories, newest first.

    ``instance`` empty means every discovered output. A name that is not a
    fleet instance name is rejected. An unknown name yields an empty page
    and still lists the outputs that do exist.
    """

    authorize("gallery")
    if offset < 0:
        raise FleetError("gallery offset must be >= 0")
    if limit < 1:
        raise FleetError("gallery limit must be >= 1")
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT
    selected = _instance_filter(instance)
    outputs = _discover(layout, docker)
    by_name = {item.name: item for item in outputs}
    if selected is not None and selected not in by_name:
        rows: list[dict] = []
    else:
        chosen = [by_name[selected]] if selected is not None else outputs
        rows = _collect(chosen)
        rows.sort(key=lambda item: (-item["mtime_ns"], item["instance"], item["path"]))
    total = len(rows)
    page = rows[offset : offset + limit]
    return {
        "ok": True,
        "instances": [
            {
                "name": item.name,
                "port": item.port,
                "status": item.status,
            }
            for item in outputs
        ],
        "items": [_public_item(item) for item in page],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def open_gallery_file(layout: FleetLayout, instance: str, relative: str) -> GalleryFile:
    """Open one media file inside that instance's output directory."""

    authorize("gallery")
    output = _require_output(layout, instance)
    rel = _relative(relative)
    kind = _kind_for_suffix(rel, path=None)
    if kind is None:
        raise GalleryMissing("file not found")
    fd = _open_nofollow(output.root, rel)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise FleetError("refusing a path that is not a regular file")
        if not _fd_inside(fd, output.root):
            raise FleetError("refusing a path that leaves the instance output directory")
        if rel.suffix.lower() == ".webp":
            kind = "animated" if _webp_animated_fd(fd) else "image"
            os.lseek(fd, 0, os.SEEK_SET)
        return GalleryFile(
            fd=fd,
            size=info.st_size,
            content_type=_CONTENT_TYPES.get(rel.suffix.lower(), "application/octet-stream"),
            filename=rel.name,
            kind=kind,
        )
    except Exception:
        os.close(fd)
        raise


def open_gallery_thumb(layout: FleetLayout, instance: str, relative: str) -> GalleryFile | tuple[bytes, str]:
    """Thumbnail bytes or a stream of an image file.

    Images are the source file, so the browser can lazy-load them. Gif
    posters are the first frame. mp4, webm, mov, and animated webp posters
    are one JPEG frame when ``ffmpeg`` is on PATH, otherwise a small play
    poster. The original video bytes are not the thumbnail.
    """

    authorize("gallery")
    output = _require_output(layout, instance)
    rel = _relative(relative)
    kind = _kind_for_suffix(rel, path=None)
    if kind is None:
        raise GalleryMissing("file not found")
    fd = _open_nofollow(output.root, rel)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise FleetError("refusing a path that is not a regular file")
        if not _fd_inside(fd, output.root):
            raise FleetError("refusing a path that leaves the instance output directory")
        suffix = rel.suffix.lower()
        if suffix == ".webp":
            kind = "animated" if _webp_animated_fd(fd) else "image"
            os.lseek(fd, 0, os.SEEK_SET)
        if kind == "image":
            return GalleryFile(
                fd=fd,
                size=info.st_size,
                content_type=_CONTENT_TYPES[suffix],
                filename=rel.name,
                kind=kind,
            )
        try:
            return _poster_fd(fd, kind, suffix)
        finally:
            os.close(fd)
            fd = -1
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise


def delete_gallery_file(layout: FleetLayout, instance: str, relative: str) -> dict:
    """Unlink one media file inside that instance's output directory."""

    authorize("gallery")
    output = _require_output(layout, instance)
    rel = _relative(relative)
    if _kind_for_suffix(rel, path=None) is None:
        raise GalleryMissing("file not found")
    fd = _open_nofollow(output.root, rel)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise FleetError("refusing to delete a path that is not a regular file")
        if not _fd_inside(fd, output.root):
            raise FleetError("refusing a path that leaves the instance output directory")
    finally:
        os.close(fd)
    parent = rel.parent
    dir_fd = _open_dir_nofollow(output.root, parent)
    try:
        try:
            os.unlink(rel.name, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise GalleryMissing("file not found") from exc
            raise FleetError("cannot delete that file") from exc
    finally:
        os.close(dir_fd)
    return {"ok": True, "deleted": {"instance": output.name, "path": rel.as_posix()}}


def poster_bytes(path: Path, kind: str) -> tuple[bytes, str]:
    """A small image for a video or animated file. Never the source video.

    The file is opened with ``O_NOFOLLOW``. ``ffmpeg`` receives that
    descriptor through ``/proc/self/fd``, not a shell command.
    """

    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        return _PLAY_POSTER, "image/png"
    try:
        return _poster_fd(fd, kind, path.suffix.lower())
    finally:
        os.close(fd)


def _poster_fd(fd: int, kind: str, suffix: str) -> tuple[bytes, str]:
    if kind == "animated" and suffix == ".gif":
        frame = _gif_first_frame_fd(fd)
        if frame is not None:
            return frame, "image/gif"
    jpeg = _ffmpeg_jpeg_fd(fd)
    if jpeg is not None:
        return jpeg, "image/jpeg"
    return _PLAY_POSTER, "image/png"


def gif_first_frame(path: Path, limit: int = _GIF_POSTER_LIMIT) -> bytes | None:
    """Return a one-frame GIF, or None when the first frame is not usable."""

    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        return _gif_first_frame_fd(fd, limit)
    finally:
        os.close(fd)


def _gif_first_frame_fd(fd: int, limit: int = _GIF_POSTER_LIMIT) -> bytes | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, limit + 1)
    except OSError:
        return None
    if len(data) > limit:
        data = data[:limit]
    return _gif_first_frame_bytes(data)


def _instance_filter(instance: str | None) -> str | None:
    if instance is None:
        return None
    text = instance.strip()
    if not text or text == "all":
        return None
    if not is_instance_name(text):
        raise FleetError("invalid gallery instance")
    return text


def _discover(layout: FleetLayout, docker: object) -> list[_Output]:
    records = _records(layout, docker)
    names: set[str] = set(records)
    files_root = layout.files
    if _real_child_dir(layout.root, files_root):
        try:
            children = list(files_root.iterdir())
        except OSError:
            children = []
        for child in children:
            try:
                if child.is_symlink() or not child.is_dir():
                    continue
            except OSError:
                continue
            if is_instance_name(child.name):
                names.add(child.name)
    found: list[_Output] = []
    for name in sorted(names):
        root = safe_output_dir(layout, name)
        if root is None:
            continue
        port, status = records.get(name, (None, None))
        found.append(_Output(name=name, root=root, port=port, status=status))
    return found


def _records(layout: FleetLayout, docker: object) -> dict[str, tuple[int | None, str | None]]:
    try:
        rows = list_instances(layout, docker)
    except FleetError:
        return {}
    except AuthError:
        raise
    found: dict[str, tuple[int | None, str | None]] = {}
    for instance, status in rows:
        found[instance.name] = (instance.port, status)
    return found


def safe_output_dir(layout: FleetLayout, name: str) -> Path | None:
    """Real ``files/<name>/output`` directory, or None when it escapes.

    Every component from the storage root must be a real directory. A
    symlink at ``files``, the instance directory, or ``output`` is refused
    even when its target sits somewhere else under the storage root.
    """

    if not is_instance_name(name):
        return None
    root = layout.root
    try:
        if root.is_symlink() or not root.is_dir():
            return None
        root_real = root.resolve(strict=True)
    except OSError:
        return None
    current = root
    for part in ("files", name, "output"):
        current = current / part
        try:
            if current.is_symlink() or not current.is_dir():
                return None
        except OSError:
            return None
    try:
        real = current.resolve(strict=True)
    except OSError:
        return None
    try:
        real.relative_to(root_real)
    except ValueError:
        return None
    if real != root_real / "files" / name / "output":
        return None
    return real


def _require_output(layout: FleetLayout, instance: str) -> _Output:
    if not isinstance(instance, str) or not is_instance_name(instance):
        raise FleetError("invalid gallery instance")
    root = safe_output_dir(layout, instance)
    if root is None:
        raise GalleryMissing("file not found")
    return _Output(name=instance, root=root, port=None, status=None)


def _real_child_dir(root: Path, child: Path) -> bool:
    try:
        if root.is_symlink() or child.is_symlink():
            return False
        if not child.is_dir():
            return False
        child.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _collect(outputs: list[_Output]) -> list[dict]:
    rows: list[dict] = []
    for output in outputs:
        _walk(output, output.root, PurePosixPath(), 0, rows)
    return rows


def _walk(output: _Output, directory: Path, prefix: PurePosixPath, depth: int, rows: list[dict]) -> None:
    if depth > _MAX_WALK_DEPTH:
        return
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if not name or name.startswith(".") or name in {".", ".."}:
            continue
        if len(name) > _MAX_COMPONENT or "/" in name or "\\" in name or "\x00" in name:
            continue
        relative = prefix / name
        if len(relative.as_posix()) > _MAX_RELATIVE:
            continue
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                _walk(output, Path(entry.path), relative, depth + 1, rows)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            info = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        kind = _kind_for_suffix(relative, Path(entry.path))
        if kind is None:
            continue
        rows.append(
            {
                "instance": output.name,
                "path": relative.as_posix(),
                "name": name,
                "kind": kind,
                "mtime_ns": info.st_mtime_ns,
                "size": info.st_size,
                "port": output.port,
                "status": output.status,
            }
        )


def _public_item(item: dict) -> dict:
    return {
        "instance": item["instance"],
        "path": item["path"],
        "name": item["name"],
        "kind": item["kind"],
        "mtime_ms": item["mtime_ns"] // 1_000_000,
        "size": item["size"],
        "port": item["port"],
        "status": item["status"],
    }


def _kind_for_suffix(relative: PurePosixPath, path: Path | None) -> str | None:
    suffix = relative.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in ANIMATED_SUFFIXES:
        return "animated"
    if suffix == ".webp":
        if path is None:
            return "image"
        return "animated" if _webp_animated_path(path) else "image"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return None


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FleetError("invalid gallery path")
    if len(value) > _MAX_RELATIVE or "\x00" in value or "\\" in value or "\n" in value or "\r" in value:
        raise FleetError("invalid gallery path")
    if value.startswith("/") or value.startswith("//"):
        raise FleetError("invalid gallery path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts:
        raise FleetError("invalid gallery path")
    for part in relative.parts:
        if part in {"", ".", ".."} or len(part) > _MAX_COMPONENT:
            raise FleetError("invalid gallery path")
        if "/" in part or "\\" in part or "\x00" in part:
            raise FleetError("invalid gallery path")
    return relative


def _open_nofollow(root: Path, relative: PurePosixPath) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    dir_fd = _open_dir_nofollow(root, relative.parent)
    try:
        return os.open(relative.name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EPERM, errno.ENOTDIR}:
            raise FleetError("refusing symlink") from exc
        if exc.errno == errno.ENOENT:
            raise GalleryMissing("file not found") from exc
        raise FleetError("cannot open that file") from exc
    finally:
        os.close(dir_fd)


def _open_dir_nofollow(root: Path, relative: PurePosixPath) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        raise GalleryMissing("file not found") from exc
    try:
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise FleetError("invalid gallery path")
            try:
                nxt = os.open(part, flags, dir_fd=fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.EPERM, errno.ENOTDIR}:
                    raise FleetError("refusing symlink") from exc
                if exc.errno == errno.ENOENT:
                    raise GalleryMissing("file not found") from exc
                raise FleetError("cannot open that file") from exc
            os.close(fd)
            fd = nxt
        return fd
    except Exception:
        os.close(fd)
        raise


def _fd_inside(fd: int, root: Path) -> bool:
    link = f"/proc/self/fd/{fd}"
    try:
        real = Path(os.path.realpath(link))
        root_real = root.resolve(strict=True)
        real.relative_to(root_real)
    except (OSError, ValueError):
        return False
    if real == root_real:
        return False
    return True


def _webp_animated_path(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    try:
        return _webp_animated_fd(fd)
    finally:
        os.close(fd)


def _webp_animated_fd(fd: int) -> bool:
    try:
        header = os.read(fd, 12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
            return False
        scanned = 12
        while scanned < 256 * 1024:
            chunk = os.read(fd, 8)
            if len(chunk) < 8:
                return False
            tag = chunk[:4]
            size = int.from_bytes(chunk[4:8], "little")
            if tag == b"ANIM":
                return True
            if tag == b"VP8X" and size >= 1:
                flags = os.read(fd, 1)
                return bool(flags and flags[0] & 0x02)
            skip = size + (size & 1)
            if skip:
                os.lseek(fd, skip, os.SEEK_CUR)
            scanned += 8 + skip
    except OSError:
        return False
    return False


def _gif_first_frame_bytes(data: bytes) -> bytes | None:
    if len(data) < 13 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        return None
    packed = data[10]
    gct = 3 * (2 << (packed & 0x07)) if packed & 0x80 else 0
    pos = 13 + gct
    if pos > len(data):
        return None
    while pos < len(data):
        block = data[pos]
        if block == 0x3B:
            return None
        if block == 0x21:
            if pos + 2 >= len(data):
                return None
            nxt = _skip_subblocks(data, pos + 2)
            if nxt is None:
                return None
            pos = nxt
            continue
        if block == 0x2C:
            if pos + 10 > len(data):
                return None
            img_packed = data[pos + 9]
            pos += 10
            if img_packed & 0x80:
                pos += 3 * (2 << (img_packed & 0x07))
            if pos >= len(data):
                return None
            pos += 1
            end = _skip_subblocks(data, pos)
            if end is None:
                return None
            return data[:end] + b"\x3b"
        return None
    return None


def _skip_subblocks(data: bytes, pos: int) -> int | None:
    while pos < len(data):
        size = data[pos]
        pos += 1
        if size == 0:
            return pos
        pos += size
        if pos > len(data):
            return None
    return None


def _ffmpeg_jpeg_fd(fd: int) -> bytes | None:
    binary = shutil.which("ffmpeg")
    if not binary:
        return None
    try:
        info = os.fstat(fd)
        name = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return None
    # Path plus inode. A recycled inode must not reuse another file's poster.
    key = hashlib.sha256(
        f"{name}\0{info.st_dev}\0{info.st_ino}\0{info.st_mtime_ns}\0{info.st_size}".encode()
    ).hexdigest()
    lock = _poster_lock(key)
    with lock:
        cache = Path(tempfile.gettempdir()) / "comfyfleet-gallery-posters" / f"{key}.jpg"
        try:
            cached = cache.read_bytes()
        except OSError:
            cached = b""
        if cached.startswith(b"\xff\xd8"):
            return cached
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return _ffmpeg_to_temp(binary, fd)
        temporary = cache.with_suffix(".jpg.part")
        jpeg = _run_ffmpeg(binary, fd, temporary)
        if jpeg is None:
            return None
        try:
            temporary.replace(cache)
        except OSError:
            return jpeg
        return jpeg


def _ffmpeg_to_temp(binary: str, fd: int) -> bytes | None:
    handle = tempfile.NamedTemporaryFile(prefix="comfyfleet-poster-", suffix=".jpg", delete=False)
    handle.close()
    dest = Path(handle.name)
    try:
        return _run_ffmpeg(binary, fd, dest)
    finally:
        try:
            dest.unlink()
        except OSError:
            pass


def _run_ffmpeg(binary: str, fd: int, dest: Path) -> bytes | None:
    # The child inherits this descriptor. /proc/self/fd refers to that
    # inode, so a swapped symlink is not what ffmpeg reads.
    source = f"/proc/self/fd/{fd}"
    argv = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        "0",
        "-i",
        source,
        "-frames:v",
        "1",
        "-an",
        "-vf",
        "scale='min(480,iw)':-2",
        "-f",
        "image2",
        "-q:v",
        "5",
        str(dest),
    ]
    try:
        completed = subprocess.run(
            argv,
            check=False,
            shell=False,
            timeout=20,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(fd,),
        )
    except (OSError, subprocess.TimeoutExpired):
        try:
            dest.unlink()
        except OSError:
            pass
        return None
    if completed.returncode != 0:
        try:
            dest.unlink()
        except OSError:
            pass
        return None
    try:
        data = dest.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"\xff\xd8"):
        try:
            dest.unlink()
        except OSError:
            pass
        return None
    return data


def _poster_lock(key: str) -> threading.Lock:
    with _POSTER_LOCKS_GUARD:
        lock = _POSTER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _POSTER_LOCKS[key] = lock
        return lock


def _png(width: int, height: int, rgba: bytes) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + rgba * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


_PLAY_POSTER = _png(8, 8, bytes((28, 28, 30, 255)))
