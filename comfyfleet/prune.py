"""Prune dangling containers without touching fleet instances.

The set is the one ``docker container prune`` removes: stopped containers
(``exited``, ``created``, or ``dead``). A container labeled
``comfyfleet.managed=true`` is excluded whether it is running or stopped.
Running containers that are not fleet instances are left alone; they are
not dangling.
"""

from __future__ import annotations

from dataclasses import dataclass

from comfyfleet.errors import FleetError

MANAGED_LABEL = "comfyfleet.managed"
MANAGED_VALUE = "true"
MANAGED_PAIR = f"{MANAGED_LABEL}={MANAGED_VALUE}"
_DANGLING_STATES = {"exited", "created", "dead"}


@dataclass(frozen=True)
class ContainerRecord:
    id: str
    name: str
    status: str
    labels: dict[str, str]
    raw_labels: str


@dataclass(frozen=True)
class PruneResult:
    removed: tuple[str, ...]
    kept_managed: tuple[str, ...]


def parse_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in (text or "").split(","):
        item = part.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not key:
            continue
        labels[key] = value if sep else ""
    return labels


def parse_ps(text: str) -> list[ContainerRecord]:
    """Parse ``docker ps -a --format '{{.ID}}\\t{{.Names}}\\t{{.Status}}\\t{{.Labels}}'``."""

    records: list[ContainerRecord] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        container_id, name, status, raw_labels = (line.split("\t", 3) + ["", "", "", ""])[:4]
        records.append(
            ContainerRecord(
                id=container_id.strip(),
                name=name.strip(),
                status=status.strip(),
                labels=parse_labels(raw_labels),
                raw_labels=raw_labels.strip(),
            )
        )
    return records


def is_managed(record: ContainerRecord) -> bool:
    if record.labels.get(MANAGED_LABEL) == MANAGED_VALUE:
        return True
    return MANAGED_PAIR in record.raw_labels


def is_dangling_status(status: str) -> bool:
    word = (status or "").strip().split(" ", 1)[0].lower()
    return word in _DANGLING_STATES


def select_prune_targets(
    records: list[ContainerRecord],
) -> tuple[list[ContainerRecord], list[ContainerRecord]]:
    """Return ``(remove, kept_managed)``. Managed containers are never removed."""

    remove: list[ContainerRecord] = []
    kept: list[ContainerRecord] = []
    for record in records:
        if is_managed(record):
            kept.append(record)
            continue
        if is_dangling_status(record.status):
            remove.append(record)
    for record in remove:
        if is_managed(record):
            raise FleetError(
                "refusing to prune a container labeled comfyfleet.managed=true"
            )
    return remove, kept


def prune_dangling_containers(docker) -> PruneResult:
    """Remove dangling non-fleet containers. Fleet instances stay.

    ``docker`` provides ``list_container_records`` and ``remove_stopped``.
    ``remove_stopped`` must not force-remove a running container.
    """

    from comfyfleet.control import authorize

    authorize("prune-dangling")
    records = list(docker.list_container_records())
    remove, kept = select_prune_targets(records)
    removed: list[str] = []
    for record in remove:
        if is_managed(record) or not is_dangling_status(record.status):
            raise FleetError(
                "refusing to prune a container labeled comfyfleet.managed=true "
                "or a container that is not dangling"
            )
        if not record.id:
            raise FleetError("refusing to prune a container with an empty id")
        docker.remove_stopped(record.id)
        removed.append(record.id)
    return PruneResult(
        removed=tuple(removed),
        kept_managed=tuple(record.id for record in kept if record.id),
    )
