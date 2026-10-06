"""Per-instance ComfyUI ``main.py`` flags.

The catalog matches ComfyUI v0.37.4 (``comfy/cli_args.py`` at
``8ff6dc384ba5c410266b40e137799e049459d4f2``). ``--listen`` and ``--port``
are not in the catalog: the entrypoint always supplies
``--listen 0.0.0.0`` and the container port, then these arguments.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass, field

from comfyfleet.errors import FleetError
from comfyfleet.paths import CONTAINER_PORT

# VRAM presets. Exactly these three, mutually exclusive. Empty means no
# VRAM flag at all (not --normalvram). They are appended after the locked
# ``--listen 0.0.0.0 --port <container port>`` pair.
VRAM_FLAGS = ("--lowvram", "--novram", "--highvram")
ATTENTION_FLAGS = (
    "--use-pytorch-cross-attention",
    "--use-sage-attention",
    "--use-flash-attention",
    "--use-split-cross-attention",
    "--use-quad-cross-attention",
    "--use-ck-attention",
)
PREVIEW_METHODS = ("auto", "latent2rgb", "taesd", "none")
_VALUE_FLAGS = ("--reserve-vram", "--vram-headroom", "--preview-method", "--preview-size")
_LOCKED = ("--listen", "--port")
_MAX_EXTRA_CHARS = 2000
_MAX_EXTRA_TOKENS = 64


@dataclass(frozen=True)
class BoolFlag:
    """One store_true flag from ComfyUI's parser."""

    flag: str
    group: str
    label: str
    # Flags that share an exclusive id cannot be combined. ``vram`` also
    # conflicts with the VRAM radio (same mutually exclusive group in ComfyUI).
    exclusive: str | None = None


BOOL_FLAGS: tuple[BoolFlag, ...] = (
    BoolFlag("--force-fp16", "dtype", "Force fp16 (also forces the UNet to fp16).", "fp"),
    BoolFlag("--force-fp32", "dtype", "Force fp32.", "fp"),
    BoolFlag("--fp16-unet", "dtype", "Run the diffusion model in fp16.", "unet"),
    BoolFlag("--bf16-unet", "dtype", "Run the diffusion model in bf16.", "unet"),
    BoolFlag("--fp32-unet", "dtype", "Run the diffusion model in fp32.", "unet"),
    BoolFlag("--fp8_e4m3fn-unet", "dtype", "Store UNet weights in fp8_e4m3fn.", "unet"),
    BoolFlag("--fp8_e5m2-unet", "dtype", "Store UNet weights in fp8_e5m2.", "unet"),
    BoolFlag("--fp8_e8m0fnu-unet", "dtype", "Store UNet weights in fp8_e8m0fnu.", "unet"),
    BoolFlag("--fp16-vae", "dtype", "Run the VAE in fp16.", "vae"),
    BoolFlag("--fp32-vae", "dtype", "Run the VAE in fp32.", "vae"),
    BoolFlag("--bf16-vae", "dtype", "Run the VAE in bf16.", "vae"),
    BoolFlag("--fp8_e4m3fn-text-enc", "dtype", "Store text encoder weights in fp8 e4m3fn.", "textenc"),
    BoolFlag("--fp8_e5m2-text-enc", "dtype", "Store text encoder weights in fp8 e5m2.", "textenc"),
    BoolFlag("--fp16-text-enc", "dtype", "Store text encoder weights in fp16.", "textenc"),
    BoolFlag("--fp32-text-enc", "dtype", "Store text encoder weights in fp32.", "textenc"),
    BoolFlag("--bf16-text-enc", "dtype", "Store text encoder weights in bf16.", "textenc"),
    BoolFlag("--cpu", "memory", "Use the CPU for everything (slow).", "vram"),
    BoolFlag("--gpu-only", "memory", "Store and run everything on the GPU.", "vram"),
    BoolFlag("--disable-smart-memory", "memory", "Offload to system RAM instead of keeping models in VRAM."),
    BoolFlag(
        "--disable-dynamic-vram",
        "memory",
        "Disable dynamic VRAM. --lowvram has no effect while dynamic VRAM is on.",
        "dynvram",
    ),
    BoolFlag("--enable-dynamic-vram", "memory", "Enable dynamic VRAM where it is not the default.", "dynvram"),
    BoolFlag("--cuda-malloc", "memory", "Enable cudaMallocAsync.", "cudamalloc"),
    BoolFlag("--disable-cuda-malloc", "memory", "Disable cudaMallocAsync.", "cudamalloc"),
    BoolFlag("--disable-xformers", "memory", "Disable xformers."),
    BoolFlag("--disable-pinned-memory", "memory", "Disable pinned memory."),
    BoolFlag("--fast-disk", "memory", "Prefer disk-backed dynamic loading (fast NVMe).", "fastdisk"),
    BoolFlag("--disable-fast-disk", "memory", "Disable disk-backed dynamic loading.", "fastdisk"),
    BoolFlag("--cpu-vae", "memory", "Run the VAE on the CPU."),
    BoolFlag("--deterministic", "memory", "Use slower deterministic PyTorch algorithms when possible."),
    BoolFlag("--force-channels-last", "memory", "Force channels-last when running the models."),
    BoolFlag("--supports-fp8-compute", "memory", "Act as if the device supports fp8 compute."),
    BoolFlag("--fp64-unet", "dtype", "Run the diffusion model in fp64.", "unet"),
    BoolFlag("--fp16-intermediates", "dtype", "Use fp16 for intermediate tensors between nodes."),
    BoolFlag("--cache-classic", "caching", "Use the old aggressive caching.", "cache"),
    BoolFlag("--cache-none", "caching", "Execute every node on each run to save RAM.", "cache"),
    BoolFlag("--high-ram", "caching", "Prefer RAM or pagefile over reloading models.", "cache"),
)

_BOOL_BY_FLAG = {item.flag: item for item in BOOL_FLAGS}
_PANEL_FLAGS = set(VRAM_FLAGS) | set(ATTENTION_FLAGS) | set(_BOOL_BY_FLAG)
_FLAG_ORDER = {item.flag: index for index, item in enumerate(BOOL_FLAGS)}


@dataclass
class LaunchConfig:
    """Operator choices stored on the instance. Empty means stock ComfyUI."""

    vram: str = ""
    attention: str = ""
    flags: list[str] = field(default_factory=list)
    reserve_vram: str | None = None
    vram_headroom: str | None = None
    preview_method: str = ""
    preview_size: str | None = None
    extra_args: str = ""

    def argv(self) -> list[str]:
        """Flags appended after ``--listen 0.0.0.0 --port <container port>``."""

        args: list[str] = []
        if self.vram:
            args.append(self.vram)
        if self.attention:
            args.append(self.attention)
        args.extend(self.flags)
        if self.reserve_vram is not None:
            args.extend(["--reserve-vram", self.reserve_vram])
        if self.vram_headroom is not None:
            args.extend(["--vram-headroom", self.vram_headroom])
        if self.preview_method:
            args.extend(["--preview-method", self.preview_method])
        if self.preview_size is not None:
            args.extend(["--preview-size", self.preview_size])
        if self.extra_args:
            args.extend(shlex.split(self.extra_args))
        return args

    def to_json(self) -> dict:
        return {
            "vram": self.vram,
            "attention": self.attention,
            "flags": list(self.flags),
            "reserve_vram": self.reserve_vram,
            "vram_headroom": self.vram_headroom,
            "preview_method": self.preview_method,
            "preview_size": self.preview_size,
            "extra_args": self.extra_args,
        }


def main_argv(launch: LaunchConfig | None = None) -> list[str]:
    """``main.py`` arguments. Locked listen and container port, then presets.

    The published host port is not an argument. Docker ``-p`` maps that host
    port onto ``CONTAINER_PORT``. Extra flags are already inside ``launch.argv``
    and cannot replace ``--listen`` or ``--port``.
    """

    body = [] if launch is None else launch.argv()
    return ["--listen", "0.0.0.0", "--port", str(CONTAINER_PORT), *body]


def parse_launch(
    *,
    vram: str | None = None,
    attention: str | None = None,
    flags: list[str] | str | None = None,
    reserve_vram: str | None = None,
    vram_headroom: str | None = None,
    preview_method: str | None = None,
    preview_size: str | None = None,
    extra_args: str | None = None,
) -> LaunchConfig:
    """Validate operator input and return the canonical config."""

    chosen_vram = _choice(vram, VRAM_FLAGS, "VRAM mode", ("default", "none", ""))
    chosen_attention = _choice(
        attention, ATTENTION_FLAGS, "attention backend", ("default", "none", "")
    )
    chosen_flags = _bool_flags(flags)
    _reject_exclusive(chosen_vram, chosen_flags)
    cleaned_extra = _extra_args(extra_args)
    config = LaunchConfig(
        vram=chosen_vram,
        attention=chosen_attention,
        flags=chosen_flags,
        reserve_vram=_optional_float(reserve_vram, "reserve_vram"),
        vram_headroom=_optional_float(vram_headroom, "vram_headroom"),
        preview_method=_preview_method(preview_method),
        preview_size=_optional_int(preview_size, "preview_size"),
        extra_args=cleaned_extra,
    )
    return config


def launch_from_json(payload: object, *, path: str) -> LaunchConfig:
    if payload is None:
        return LaunchConfig()
    if not isinstance(payload, dict):
        raise FleetError(f"instance launch flags are invalid: {path}")
    try:
        return parse_launch(
            vram=_json_str(payload.get("vram")),
            attention=_json_str(payload.get("attention")),
            flags=payload.get("flags"),
            reserve_vram=_json_str(payload.get("reserve_vram")),
            vram_headroom=_json_str(payload.get("vram_headroom")),
            preview_method=_json_str(payload.get("preview_method")),
            preview_size=_json_str(payload.get("preview_size")),
            extra_args=_json_str(payload.get("extra_args")),
        )
    except FleetError as exc:
        raise FleetError(f"instance launch flags are invalid ({path}): {exc}") from exc


def combine_extra_args(*parts: str | None) -> str | None:
    """Join ``extra_args`` and ``comfy_extra_args`` into one shell string.

    Blank parts are skipped. The result is still checked by ``parse_launch``.
    """

    chunks = [str(part).strip() for part in parts if part and str(part).strip()]
    if not chunks:
        return None
    return " ".join(chunks)


def split_flag_field(value: str | None) -> list[str]:
    """Comma- or whitespace-separated flag list from a form field."""

    if value is None or not str(value).strip():
        return []
    parts: list[str] = []
    for chunk in str(value).split(","):
        parts.extend(chunk.split())
    return parts


def strip_locked_args(tokens: list[str]) -> list[str]:
    """Drop ``--listen`` / ``--port`` and a following value that is not a flag."""

    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        base, sep, _value = token.partition("=")
        if base in _LOCKED:
            index += 1
            if sep == "" and index < len(tokens) and not tokens[index].startswith("-"):
                index += 1
            continue
        kept.append(token)
        index += 1
    return kept


def _choice(value: str | None, allowed: tuple[str, ...], label: str, empty: tuple[str, ...]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in empty or text == "":
        return ""
    flag = _canonical_flag(text)
    if flag not in allowed:
        choices = ", ".join(("default", *allowed))
        raise FleetError(f"{label} must be one of: {choices} (got {value!r})")
    return flag


def _bool_flags(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = split_flag_field(value)
    elif isinstance(value, list):
        raw_items = []
        for item in value:
            if not isinstance(item, str):
                raise FleetError("launch flags must be strings")
            raw_items.append(item)
    else:
        raise FleetError("launch flags must be a list of strings")
    chosen: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not str(raw).strip():
            continue
        flag = _canonical_flag(str(raw))
        if flag in _LOCKED or flag.partition("=")[0] in _LOCKED:
            raise FleetError(
                f"{flag} is locked. ComfyFleet always starts main.py with "
                "--listen 0.0.0.0 and the container port."
            )
        if flag in VRAM_FLAGS:
            raise FleetError(f"set {flag} with the VRAM control, not the flag list")
        if flag in ATTENTION_FLAGS:
            raise FleetError(f"set {flag} with the attention control, not the flag list")
        if flag not in _BOOL_BY_FLAG:
            raise FleetError(
                f"unknown launch flag {flag}. "
                "Listed flags belong in the panel; other main.py flags go in extra args."
            )
        if flag in seen:
            raise FleetError(f"duplicate launch flag {flag}")
        seen.add(flag)
        chosen.append(flag)
    chosen.sort(key=lambda item: _FLAG_ORDER[item])
    return chosen


def _reject_exclusive(vram: str, flags: list[str]) -> None:
    buckets: dict[str, list[str]] = {}
    if vram:
        buckets.setdefault("vram", []).append(vram)
    for flag in flags:
        exclusive = _BOOL_BY_FLAG[flag].exclusive
        if exclusive:
            buckets.setdefault(exclusive, []).append(flag)
    for chosen in buckets.values():
        if len(chosen) > 1:
            raise FleetError(
                f"{', '.join(chosen)} cannot be combined. "
                "ComfyUI accepts only one of that group."
            )


def _extra_args(value: str | None) -> str:
    if value is None or not str(value).strip():
        return ""
    text = str(value).strip()
    if len(text) > _MAX_EXTRA_CHARS:
        raise FleetError(f"extra Comfy args must be at most {_MAX_EXTRA_CHARS} characters")
    if "\x00" in text:
        raise FleetError("extra Comfy args cannot contain NUL")
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        raise FleetError(f"extra Comfy args are not valid shell words: {exc}") from exc
    tokens = strip_locked_args(tokens)
    if len(tokens) > _MAX_EXTRA_TOKENS:
        raise FleetError(f"extra Comfy args must be at most {_MAX_EXTRA_TOKENS} words")
    for token in tokens:
        base = token.partition("=")[0]
        if base in _PANEL_FLAGS or base in _VALUE_FLAGS:
            raise FleetError(
                f"extra args cannot include {base}. Set it in the launch flags panel."
            )
    return shlex.join(tokens)


def _preview_method(value: str | None) -> str:
    if value is None or not str(value).strip():
        return ""
    text = str(value).strip().lower()
    if text not in PREVIEW_METHODS:
        choices = ", ".join(PREVIEW_METHODS)
        raise FleetError(f"preview_method must be one of: {choices} (got {value!r})")
    return text


def _optional_float(value: str | None, name: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise FleetError(f"{name} must be a number of GB, got {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise FleetError(f"{name} must be a non-negative number of GB, got {value!r}")
    rendered = f"{number:.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _optional_int(value: str | None, name: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if not text.isdigit():
        raise FleetError(f"{name} must be a positive integer, got {value!r}")
    number = int(text)
    if number < 1 or number > 8192:
        raise FleetError(f"{name} must be from 1 to 8192, got {value!r}")
    return str(number)


def _canonical_flag(value: str) -> str:
    text = value.strip()
    if not text or any(char.isspace() for char in text):
        raise FleetError(f"invalid launch flag {value!r}")
    if text.startswith("--"):
        return text
    if text.startswith("-"):
        raise FleetError(f"invalid launch flag {value!r}")
    return f"--{text}"


def _json_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FleetError(f"expected a string, got {value!r}")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    raise FleetError(f"expected a string, got {value!r}")
