"""Validate the operator workflow. There is no stock substitute."""

from __future__ import annotations

import json
from pathlib import Path

from comfyfleet.errors import FleetError


def load_operator_workflow(path: Path) -> dict:
    if path.suffix.lower() != ".json":
        raise FleetError(
            f"workflow must be a .json file, got {path}. "
            "Create requires an operator-supplied workflow. There is no stock default."
        )
    if not path.is_file():
        raise FleetError(
            f"workflow not found or not a file: {path}. "
            "Pass --workflow /path/to/flow.json."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FleetError(f"cannot read workflow {path}: {exc}") from exc
    if not text.strip():
        raise FleetError(f"workflow file is empty: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FleetError(f"workflow is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise FleetError(
            f"workflow JSON must be an object, not {type(data).__name__} ({path})."
        )
    return data
