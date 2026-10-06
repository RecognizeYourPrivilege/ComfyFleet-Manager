"""GPU detection via nvidia-smi. Phase 1 does not treat CPU-only as success."""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass

from comfyfleet.errors import FleetError


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory: str


def parse_nvidia_smi(text: str) -> list[Gpu]:
    gpus: list[Gpu] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        if len(cells) < 2:
            raise FleetError(f"unexpected nvidia-smi line: {','.join(row)}")
        try:
            index = int(cells[0])
        except ValueError as exc:
            raise FleetError(f"unexpected nvidia-smi GPU index in: {','.join(row)}") from exc
        memory = cells[2] if len(cells) > 2 else ""
        gpus.append(Gpu(index=index, name=cells[1], memory=memory))
    return gpus


def detect_gpus(run=None) -> list[Gpu]:
    """Run nvidia-smi. Abort when it is missing, failing, or lists no GPU."""

    argv = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader",
    ]
    runner = run or _default_run
    try:
        completed = runner(argv)
    except FileNotFoundError as exc:
        raise FleetError(_missing_message()) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        hint = ""
        low = detail.lower()
        if any(
            token in low
            for token in (
                "nvml",
                "nvidia driver",
                "couldn't communicate",
                "could not communicate",
            )
        ):
            hint = (
                " If this is the manager container, pass --gpus all and install the "
                "NVIDIA Container Toolkit on the host. The probe is nvidia-smi inside "
                "the manager; the toolkit injects the host driver so the list matches "
                "the host GPUs. The manager image does not ship a CUDA stack."
            )
        raise FleetError(
            "nvidia-smi failed. A working NVIDIA GPU is required; "
            f"CPU-only is not a successful create or start. {detail}{hint}".rstrip()
        )
    gpus = parse_nvidia_smi(completed.stdout or "")
    if not gpus:
        raise FleetError(
            "nvidia-smi reported no GPUs. Phase 1 requires a working NVIDIA GPU."
        )
    return gpus


def select_gpus(
    gpus: list[Gpu],
    *,
    gpu: str | None = None,
    gpus_spec: str | None = None,
    interactive: bool = False,
    prompt=None,
) -> list[int]:
    """Choose GPU indices. Multi-GPU hosts must be asked unless a flag is set."""

    if not gpus:
        raise FleetError("no GPUs available to attach")
    if gpu is not None and gpus_spec is not None:
        raise FleetError("pass only one of --gpu and --gpus")
    available = {item.index for item in gpus}
    spec = gpus_spec if gpus_spec is not None else gpu
    if spec is not None:
        return _parse_spec(spec, available)
    if not interactive:
        shown = ", ".join(str(item.index) for item in gpus)
        raise FleetError(
            "GPU selection is required. Re-run in a terminal to be asked, "
            f"or pass --gpu / --gpus. Detected GPU index(es): {shown}."
        )
    if prompt is None:
        raise FleetError("interactive GPU selection has no prompt")
    _print_gpus(gpus)
    if len(gpus) == 1:
        only = gpus[0]
        answer = prompt(f"Attach GPU {only.index} ({only.name})? [y/N]: ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise FleetError(
                f"GPU attach declined. Re-run and confirm, or pass --gpu {only.index}."
            )
        return [only.index]
    answer = prompt(
        "Which GPU(s) should this instance use? "
        "Enter indices (example: 0 or 0,1 or all): "
    )
    if not answer.strip():
        raise FleetError(
            "no GPU selected. Multi-GPU create must be told which GPU(s) to use."
        )
    return _parse_spec(answer, available)


def _parse_spec(spec: str, available: set[int]) -> list[int]:
    text = spec.strip().lower()
    if not text:
        raise FleetError("no GPU selected")
    if text == "all":
        return sorted(available)
    chosen: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item.isdigit():
            raise FleetError(
                f"invalid GPU selection {spec!r}. Use indices such as 0 or 0,1, or all."
            )
        index = int(item)
        if index not in available:
            known = ", ".join(str(value) for value in sorted(available))
            raise FleetError(f"GPU {index} is not present. Detected: {known}.")
        if index not in chosen:
            chosen.append(index)
    if not chosen:
        raise FleetError("no GPU selected")
    return chosen


def _print_gpus(gpus: list[Gpu]) -> None:
    print(f"Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        memory = f" ({gpu.memory})" if gpu.memory else ""
        print(f"  [{gpu.index}] {gpu.name}{memory}")


def _missing_message() -> str:
    return (
        "nvidia-smi was not found. The GPU probe runs nvidia-smi in this process "
        "so it sees the host GPUs. Install the NVIDIA driver and confirm nvidia-smi "
        "works on the host. When this process is the manager container, start it "
        "with --gpus all so the NVIDIA Container Toolkit injects the host nvidia-smi "
        "and driver libraries. The manager image does not ship a CUDA stack. "
        "CPU-only is not a successful create or start."
    )


def _default_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)
