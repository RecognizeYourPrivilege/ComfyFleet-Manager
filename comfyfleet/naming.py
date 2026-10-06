"""Instance names derived from the operator workflow filename.

Algorithm (FR-N2):

1. Take the filename stem (``my_flow.json`` -> ``my_flow``).
2. Lowercase it.
3. Replace every character outside ``[a-z0-9_-]`` with ``_``.
4. Collapse repeated underscores and strip leading/trailing ``_`` and ``-``.
5. If the result is empty or does not start with ``[a-z0-9]``, prefix ``wf``.
6. If the result is at most 63 characters and matches ``[a-z0-9][a-z0-9_-]*``,
   use it unchanged.
7. Otherwise truncate and append ``-`` plus the first 8 hex digits of
   SHA-256(original stem UTF-8). The final name is at most 63 characters.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

MAX_NAME_LENGTH = 63
_UNSAFE = re.compile(r"[^a-z0-9_-]+")
_VALID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def instance_name_from_workflow(path: str | Path) -> str:
    return sanitize_stem(Path(path).stem)


def resolve_instance_name(workflow: str | Path, requested: str | None = None) -> str:
    """Container name for one create.

    A blank ``requested`` uses the workflow JSON filename stem. A typed
    value wins and goes through the same sanitizer, so ``Portrait`` and
    ``Portrait.json`` both become ``portrait``.
    """

    if requested is not None and requested.strip():
        return sanitize_stem(requested.strip())
    return instance_name_from_workflow(workflow)


def sanitize_stem(stem: str) -> str:
    original = stem
    cleaned = _UNSAFE.sub("_", stem.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"wf_{cleaned}".strip("_-")
    if not cleaned:
        cleaned = "workflow"
    if len(cleaned) <= MAX_NAME_LENGTH and _VALID.fullmatch(cleaned):
        return cleaned
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    keep = MAX_NAME_LENGTH - len(digest) - 1
    prefix = cleaned[:keep].rstrip("_-")
    if not prefix or not prefix[0].isalnum():
        prefix = "wf"
    name = f"{prefix}-{digest}"
    return name[:MAX_NAME_LENGTH]


def is_instance_name(name: str) -> bool:
    return bool(_VALID.fullmatch(name)) and len(name) <= MAX_NAME_LENGTH
