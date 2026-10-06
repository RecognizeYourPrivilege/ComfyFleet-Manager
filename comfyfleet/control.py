"""Create, start, stop, and list workflow instances.

The control HTTP API calls these functions and does not reimplement mounts,
naming, the operator workflow copy, or port assignment. ``authorize`` is the
fail-closed check on that HTTP path. A local CLI call is not an HTTP request.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from comfyfleet.auth import AuthError, http_auth_state
from comfyfleet.custom_nodes import (
    clone_git_urls,
    extract_custom_nodes_zips,
    node_map_from_env,
    plan_missing_installs,
    trusted_manager_install,
)
from comfyfleet.docker import DockerCLI, build_create_args
from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu, select_gpus
from comfyfleet.launch import LaunchConfig, launch_from_json, parse_launch
from comfyfleet.naming import is_instance_name, resolve_instance_name
from comfyfleet.ownership import ensure_instance_host_dirs, ensure_wildcards_dir
from comfyfleet.paths import (
    CUDA_TAGS,
    CU130_PUBLISHED_DIGEST,
    DEFAULT_CUDA_TAG,
    DEFAULT_IMAGE,
    MODEL_SUBDIRS,
    FleetLayout,
    image_for_cuda_tag,
)
from comfyfleet.ports import choose_port, make_port_in_use
from comfyfleet.workflow import load_operator_workflow

METADATA_SCHEMA = 2


@dataclass
class Instance:
    name: str
    port: int
    gpus: list[int]
    image: str
    workflow_host_path: str
    workflow_source: str
    created_at: str
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    cuda_tag: str = ""

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["schema"] = METADATA_SCHEMA
        payload["launch"] = self.launch.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: dict, path: Path) -> "Instance":
        try:
            return cls(
                name=str(payload["name"]),
                port=int(payload["port"]),
                gpus=[int(item) for item in payload["gpus"]],
                image=str(payload["image"]),
                workflow_host_path=str(payload["workflow_host_path"]),
                workflow_source=str(payload.get("workflow_source", "")),
                created_at=str(payload.get("created_at", "")),
                launch=launch_from_json(payload.get("launch"), path=str(path)),
                cuda_tag=_stored_cuda_tag(payload, str(payload["image"])),
            )
        except FleetError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise FleetError(f"instance metadata is invalid: {path}") from exc


@dataclass
class ActionResult:
    instance: Instance
    started: bool
    warning: str | None = None
    warnings: list[str] = field(default_factory=list)


def authorize(action: str) -> None:
    """Allow a control action, or fail closed on an unauthenticated HTTP request.

    Outside an HTTP request (the host CLI, or ``docker exec`` inside the
    manager) this only checks that ``action`` is known. That process is
    local: it does not read the session cookie. The HTTP server grants
    this check only after a valid session cookie or Bearer token, and
    refuses to start when ``COMFYFLEET_PASSWORD`` is missing.
    """

    if action not in {
        "create",
        "start",
        "stop",
        "force-stop",
        "delete",
        "terminal",
        "restart",
        "list",
        "update",
        "fix-owner",
        "prune-dangling",
        "gallery",
        "import",
    }:
        raise FleetError(f"unknown control action {action!r}")
    if http_auth_state() is False:
        raise AuthError("unauthorized")


def create_instance(
    workflow: Path,
    *,
    layout: FleetLayout,
    docker: DockerCLI,
    gpus: list[Gpu],
    gpu: str | None = None,
    gpus_spec: str | None = None,
    interactive: bool = False,
    prompt=None,
    image: str = DEFAULT_IMAGE,
    cuda_tag: str | None = None,
    instance_image: str | None = None,
    start: bool = False,
    force: bool = False,
    port_in_use=None,
    max_concurrent: int | None = None,
    use_env_limit: bool = False,
    launch: LaunchConfig | None = None,
    custom_node_git_urls: list[str] | None = None,
    custom_nodes_zip: bytes | Sequence[bytes] | None = None,
    custom_nodes_zip_names: Sequence[str] | None = None,
    custom_nodes_zip_labels: Sequence[str] | None = None,
    install_missing_from_workflow: bool = True,
    node_installer: Callable | None = None,
    node_map: dict[str, str] | None = None,
    git_run: Callable | None = None,
    name: str | None = None,
    requested_port: int | None = None,
) -> ActionResult:
    authorize("create")
    launch = _canonicalize_launch(launch)
    source = Path(workflow)
    workflow_data = load_operator_workflow(source)
    name = resolve_instance_name(source, name)
    _require_name(name)
    previous = _load_if_present(layout, name)
    container_status = docker.status(name)
    if previous is not None or container_status is not None:
        if not force:
            raise FleetError(
                f"instance {name!r} already exists (status: {container_status or 'metadata only'}). "
                "Refusing to overwrite. Stop it and re-run with --force to replace a stopped instance. "
                "Changing the GPU set or the CUDA line requires recreate (--force), not an in-place edit. "
                "start, restart, and launch Apply keep the image this instance was created with."
            )
        if container_status == "running":
            raise FleetError(
                f"instance {name!r} is running. Stop it before --force replace. "
                "A running container is never overwritten."
            )
    image, cuda_tag = resolve_create_image(
        image,
        cuda_tag=cuda_tag,
        instance_image=instance_image,
        previous=previous,
        reuse_previous=bool(force and previous is not None),
    )
    selected = select_gpus(
        gpus,
        gpu=gpu,
        gpus_spec=gpus_spec,
        interactive=interactive,
        prompt=prompt,
    )
    if force and container_status is not None:
        docker.remove(name)
    reserved = _reserved_ports(layout, exclude=name)
    in_use = _port_in_use(port_in_use, docker)
    if requested_port is not None and (requested_port < 1 or requested_port > 65535):
        raise FleetError(f"port must be 1..65535, got {requested_port}")
    port = choose_port(reserved, preferred=requested_port, in_use=in_use)
    if requested_port is not None and port != requested_port:
        raise FleetError(
            f"port {requested_port} is already in use. Pick another port."
        )
    _prepare_dirs(layout, name)
    dest = layout.workflow_file(name)
    _copy_workflow(source, dest)
    instance = Instance(
        name=name,
        port=port,
        gpus=selected,
        image=image,
        workflow_host_path=str(dest),
        workflow_source=str(source.resolve()),
        created_at=_now(),
        launch=launch,
        cuda_tag=cuda_tag,
    )
    _write_metadata(layout, instance)
    try:
        # Fakes used by tests omit this. The real client pulls a missing ref.
        ensure_image = getattr(docker, "ensure_image", None)
        if callable(ensure_image):
            ensure_image(instance.image)
        docker.create(_create_args(layout, instance))
    except Exception:
        if previous is not None:
            _write_metadata(layout, previous)
        else:
            _remove_metadata(layout, name)
        raise
    node_warnings: list[str] = []
    nodes_dir = layout.custom_nodes(name)
    install_urls: list[str] = []
    try:
        clone_warnings, _cloned = clone_git_urls(
            custom_node_git_urls,
            nodes_dir,
            run=git_run,
        )
        node_warnings.extend(clone_warnings)
        zip_warnings, _extracted = extract_custom_nodes_zips(
            custom_nodes_zip,
            nodes_dir,
            names=custom_nodes_zip_names,
            labels=custom_nodes_zip_labels,
        )
        node_warnings.extend(zip_warnings)
        if install_missing_from_workflow:
            resolved_map = node_map
            if resolved_map is None:
                resolved_map, map_warnings = node_map_from_env()
                node_warnings.extend(map_warnings)
            install_urls, plan_warnings = plan_missing_installs(
                workflow_data,
                nodes_dir,
                resolved_map,
            )
            node_warnings.extend(plan_warnings)
    except Exception as exc:
        # The container already exists. Optional node work must not turn that
        # into a failed create (O-CN-01).
        node_warnings.append(f"custom node provisioning failed: {exc}")
        install_urls = []
    warning, started, instance = _finish_create_start(
        instance,
        start=start,
        install_urls=install_urls,
        node_warnings=node_warnings,
        layout=layout,
        docker=docker,
        gpus=gpus,
        port_in_use=port_in_use,
        max_concurrent=max_concurrent,
        use_env_limit=use_env_limit,
        node_installer=node_installer,
    )
    return ActionResult(
        instance=instance,
        started=started,
        warning=warning,
        warnings=node_warnings,
    )


def _finish_create_start(
    instance: Instance,
    *,
    start: bool,
    install_urls: list[str],
    node_warnings: list[str],
    layout: FleetLayout,
    docker: DockerCLI,
    gpus: list[Gpu],
    port_in_use,
    max_concurrent: int | None,
    use_env_limit: bool,
    node_installer: Callable | None,
) -> tuple[str | None, bool, Instance]:
    """Start when asked. Install missing workflow nodes, then reload.

    Git clones and zip extracts are already on the volume, so the first start
    loads those. A trusted Manager install happens while ComfyUI is up, so a
    successful install is followed by a restart. If the operator did not ask
    to leave the instance running, it is stopped again. Host mounts stay.
    """

    warning: str | None = None
    running = False

    def _start() -> None:
        nonlocal warning, running, instance
        result = start_instance(
            instance.name,
            layout=layout,
            docker=docker,
            gpus=gpus,
            port_in_use=port_in_use,
            max_concurrent=max_concurrent,
            use_env_limit=use_env_limit,
        )
        warning = result.warning
        instance = result.instance
        running = True

    if install_urls:
        try:
            _start()
        except FleetError as exc:
            if start:
                raise
            node_warnings.append(
                f"missing-node install skipped; could not start {instance.name}: {exc}"
            )
        if running:
            installer = node_installer or trusted_manager_install
            installed: list[str] = []
            try:
                install_warnings, installed = installer(
                    install_urls,
                    docker=docker,
                    name=instance.name,
                )
                node_warnings.extend(install_warnings)
            except Exception as exc:
                node_warnings.append(f"trusted install failed: {exc}")
                installed = []
            if installed:
                try:
                    restarted = restart_instance(
                        instance.name,
                        layout=layout,
                        docker=docker,
                        gpus=gpus,
                        port_in_use=port_in_use,
                        max_concurrent=max_concurrent,
                        use_env_limit=use_env_limit,
                    )
                    warning = restarted.warning
                    instance = restarted.instance
                    running = True
                except FleetError as exc:
                    node_warnings.append(f"restart after custom node install failed: {exc}")
                    if start:
                        _start()
                    else:
                        running = docker.status(instance.name) == "running"
            if not start and running:
                try:
                    stop_instance(instance.name, layout=layout, docker=docker)
                except FleetError as exc:
                    node_warnings.append(
                        f"custom nodes were installed but the instance stayed running: {exc}"
                    )
                else:
                    running = False
                    warning = None
    elif start:
        _start()
    return (warning if start else None), bool(start and running), instance


def start_instance(
    name: str,
    *,
    layout: FleetLayout,
    docker: DockerCLI,
    gpus: list[Gpu],
    port_in_use=None,
    max_concurrent: int | None = None,
    use_env_limit: bool = False,
) -> ActionResult:
    authorize("start")
    _require_name(name)
    if not gpus:
        raise FleetError(
            "nvidia-smi reported no usable GPU. Phase 1 will not start this instance."
        )
    instance = _require_instance(layout, name)
    known = {gpu.index for gpu in gpus}
    missing = [index for index in instance.gpus if index not in known]
    if missing:
        raise FleetError(
            f"GPU(s) {missing} were assigned at create but nvidia-smi no longer lists them. "
            "Changing GPUs requires recreate: stop the instance and run "
            f"comfyfleet create --workflow {instance.workflow_host_path} --force --gpus ..."
        )
    status = docker.status(name)
    if status == "running":
        return ActionResult(instance=instance, started=True, warning=None)
    in_use = _port_in_use(port_in_use, docker)
    reserved = _reserved_ports(layout, exclude=name)
    port = choose_port(reserved, preferred=instance.port, in_use=in_use)
    if status is None or port != instance.port:
        if status is not None:
            docker.remove(name)
        instance.port = port
        _write_metadata(layout, instance)
        docker.create(_create_args(layout, instance))
    limit = _resolve_limit(max_concurrent, use_env=use_env_limit)
    running = [item for item in docker.running_names() if item != name]
    running_after = len(running) + 1
    warning = _concurrency_warning(running_after, gpu_count=len(gpus), limit=limit)
    docker.update_restart(name, "unless-stopped")
    docker.start(name)
    return ActionResult(instance=instance, started=True, warning=warning)


def stop_instance(name: str, *, layout: FleetLayout, docker: DockerCLI) -> Instance:
    authorize("stop")
    _require_name(name)
    instance = _require_instance(layout, name)
    status = docker.status(name)
    if status is None:
        raise FleetError(
            f"instance {name!r} has metadata but no container. Nothing to stop."
        )
    if status != "running":
        return instance
    docker.stop(name)
    return instance


def force_stop_instance(name: str, *, layout: FleetLayout, docker: DockerCLI) -> Instance:
    """SIGKILL the instance container. This is not ``docker stop``."""

    authorize("force-stop")
    _require_name(name)
    instance = _require_instance(layout, name)
    status = docker.status(name)
    if status is None:
        raise FleetError(
            f"instance {name!r} has metadata but no container. Nothing to force-stop."
        )
    if status != "running":
        return instance
    docker.kill(name)
    return instance


def delete_instance(name: str, *, layout: FleetLayout, docker: DockerCLI) -> Instance:
    """Force-stop and remove this instance container, then drop its fleet record.

    Host files (workflow, input, output, custom nodes) are left in place.
    Only the named instance is removed. Other containers are not touched.
    """

    authorize("delete")
    _require_name(name)
    instance = _require_instance(layout, name)
    status = docker.status(name)
    if status == "running":
        docker.kill(name)
    if docker.status(name) is not None:
        docker.remove(name)
    _remove_metadata(layout, name)
    return instance


def update_instance_launch(
    name: str,
    launch: LaunchConfig | None,
    *,
    layout: FleetLayout,
    docker: DockerCLI,
    gpus: list[Gpu],
    port_in_use=None,
    max_concurrent: int | None = None,
    use_env_limit: bool = False,
) -> ActionResult:
    """Stop and recreate this instance so only Comfy argv changes.

    Name, host port, GPU set, image, CUDA line, workflow file, and mount
    paths stay. Apply does not read ``COMFYFLEET_INSTANCE_IMAGE`` again.
    Changing ``cu130`` / ``cu124`` is a create ``--force``, not this path.
    The previous container is removed before the replacement is created, so
    the name is not left duplicated.
    """

    authorize("update")
    _require_name(name)
    launch = _canonicalize_launch(launch)
    current = _require_instance(layout, name)
    status = docker.status(name)
    if status is None:
        raise FleetError(
            f"instance {name!r} has metadata but no container. "
            "Nothing to recreate."
        )
    was_running = status == "running"
    if was_running:
        docker.stop(name)
    if docker.status(name) is not None:
        docker.remove(name)
    updated = replace(current, launch=launch)
    _write_metadata(layout, updated)
    try:
        docker.create(_create_args(layout, updated))
    except Exception:
        _write_metadata(layout, current)
        if docker.status(name) is None:
            docker.create(_create_args(layout, current))
        raise
    if not was_running:
        return ActionResult(instance=updated, started=False, warning=None)
    return start_instance(
        name,
        layout=layout,
        docker=docker,
        gpus=gpus,
        port_in_use=port_in_use,
        max_concurrent=max_concurrent,
        use_env_limit=use_env_limit,
    )


def terminal_argv(
    name: str,
    *,
    layout: FleetLayout,
    docker: DockerCLI,
    argv_for=None,
) -> list[str]:
    """Argv for a shell in one running instance. The browser does not supply it."""

    from comfyfleet.terminal import terminal_exec_argv

    authorize("terminal")
    _require_name(name)
    instance = _require_instance(layout, name)
    if docker.status(instance.name) != "running":
        raise FleetError(
            f"instance {instance.name!r} is not running. Start it before opening a shell."
        )
    build = argv_for or terminal_exec_argv
    return list(build(instance.name))


def restart_instance(
    name: str,
    *,
    layout: FleetLayout,
    docker: DockerCLI,
    gpus: list[Gpu],
    port_in_use=None,
    max_concurrent: int | None = None,
    use_env_limit: bool = False,
) -> ActionResult:
    authorize("restart")
    status = docker.status(name)
    if status == "running":
        stop_instance(name, layout=layout, docker=docker)
    return start_instance(
        name,
        layout=layout,
        docker=docker,
        gpus=gpus,
        port_in_use=port_in_use,
        max_concurrent=max_concurrent,
        use_env_limit=use_env_limit,
    )


def list_instances(layout: FleetLayout, docker: DockerCLI) -> list[tuple[Instance, str]]:
    authorize("list")
    rows: list[tuple[Instance, str]] = []
    files_root = layout.files
    if not files_root.is_dir():
        return rows
    for child in sorted(path for path in files_root.iterdir() if path.is_dir()):
        meta = child / "comfyfleet.json"
        if not meta.is_file():
            continue
        instance = _read_metadata(meta)
        status = docker.status(instance.name) or "missing"
        rows.append((instance, status))
    return rows


def format_list(rows: list[tuple[Instance, str]]) -> str:
    header = ("NAME", "STATUS", "PORT", "URL", "GPUS", "CUDA", "WORKFLOW")
    host = list_url_host()
    body = []
    for instance, status in rows:
        body.append(
            (
                instance.name,
                status,
                str(instance.port),
                f"http://{host}:{instance.port}",
                ",".join(str(index) for index in instance.gpus),
                instance.cuda_tag or infer_cuda_tag(instance.image) or instance.image,
                instance.workflow_host_path,
            )
        )
    widths = [len(column) for column in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(cell.ljust(widths[index]) for index, cell in enumerate(header))]
    if not body:
        lines.append("(no instances)")
        return "\n".join(lines)
    for row in body:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def _canonicalize_launch(launch: LaunchConfig | None) -> LaunchConfig:
    if launch is None:
        return LaunchConfig()
    return parse_launch(
        vram=launch.vram,
        attention=launch.attention,
        flags=launch.flags,
        reserve_vram=launch.reserve_vram,
        vram_headroom=launch.vram_headroom,
        preview_method=launch.preview_method,
        preview_size=launch.preview_size,
        extra_args=launch.extra_args,
    )


def resolve_instance_image(image: str) -> str:
    """``COMFYFLEET_INSTANCE_IMAGE`` overrides the default instance tag.

    An explicit non-default ``--image`` is kept. Create pulls the resolved
    ref when the host engine does not already have it.
    """

    resolved, _tag = resolve_create_image(image)
    return resolved


def image_tag(image: str) -> str:
    """Tag portion of an image ref, ignoring a trailing digest."""

    name = image.strip().split("@", 1)[0]
    slash = name.rfind("/")
    tail = name[slash + 1 :]
    colon = tail.rfind(":")
    if colon < 0:
        return ""
    return tail[colon + 1 :]


def infer_cuda_tag(image: str) -> str:
    """Return ``cu130`` or ``cu124`` when the ref names that line, else ``""``.

    ``:latest`` is the cu130 alias. ``:phase1`` is the cu124 alias, except
    the digest published before that alias existed, which is the cu130 line.
    """

    ref = image.strip()
    if not ref:
        return ""
    if CU130_PUBLISHED_DIGEST in ref:
        return "cu130"
    tag = image_tag(ref)
    if tag in CUDA_TAGS:
        return tag
    if tag == "latest":
        return "cu130"
    if tag == "phase1":
        return "cu124"
    return ""


def _stored_cuda_tag(payload: dict, image: str) -> str:
    raw = payload.get("cuda_tag", "")
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raise FleetError("instance metadata field 'cuda_tag' must be a string")
    tag = raw.strip()
    if tag and tag not in CUDA_TAGS:
        raise FleetError(f"instance metadata cuda_tag {tag!r} is not cu130 or cu124")
    return tag or infer_cuda_tag(image)


def _install_default_image() -> tuple[str, str]:
    """Image and CUDA line chosen at install (or the cu130 default)."""

    env_image = os.environ.get("COMFYFLEET_INSTANCE_IMAGE", "").strip()
    env_tag = os.environ.get("COMFYFLEET_CUDA_TAG", "").strip()
    if env_tag and env_tag not in CUDA_TAGS:
        raise FleetError(
            f"COMFYFLEET_CUDA_TAG must be cu130 or cu124, got {env_tag!r}. "
            "cu130 needs a host driver that supports CUDA 13.0. "
            "cu124 needs a host driver that supports CUDA 12.4."
        )
    if env_image:
        inferred = infer_cuda_tag(env_image)
        if env_tag and inferred and env_tag != inferred:
            raise FleetError(
                f"COMFYFLEET_CUDA_TAG={env_tag} does not match "
                f"COMFYFLEET_INSTANCE_IMAGE ({inferred})."
            )
        return env_image, env_tag or inferred
    return DEFAULT_IMAGE, env_tag or DEFAULT_CUDA_TAG


def _local_image(cuda_tag: str) -> str:
    return image_for_cuda_tag(cuda_tag)


def resolve_create_image(
    image: str = DEFAULT_IMAGE,
    cuda_tag: str | None = None,
    instance_image: str | None = None,
    *,
    previous: Instance | None = None,
    reuse_previous: bool = False,
) -> tuple[str, str]:
    """Return ``(image ref, cuda_tag)`` for one create.

    ``instance_image`` or a non-default ``image`` is a full ref. ``cuda_tag``
    is ``cu130`` or ``cu124``. When neither is set, the install default is
    used (``COMFYFLEET_INSTANCE_IMAGE`` / ``COMFYFLEET_CUDA_TAG``, else
    ``ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130``).

    A ``--force`` recreate with no new line keeps a different CUDA line
    instead of swapping it because the install default changed. The same
    line still follows a new digest in ``COMFYFLEET_INSTANCE_IMAGE``.
    Create pulls the ref when it is missing on the host engine.
    """

    requested = (cuda_tag or "").strip()
    if requested and requested not in CUDA_TAGS:
        raise FleetError(
            f"cuda_tag must be cu130 or cu124, got {requested!r}. "
            "cu130 needs a host driver that supports CUDA 13.0. "
            "cu124 needs a host driver that supports CUDA 12.4. "
            "The wrong line can fail when the instance starts."
        )
    override = (instance_image or "").strip()
    explicit = image.strip() if image and image != DEFAULT_IMAGE else ""
    full = override or explicit
    if full:
        inferred = infer_cuda_tag(full)
        if requested and inferred and requested != inferred:
            raise FleetError(
                f"cuda_tag {requested} does not match image ref {full} ({inferred}). "
                "Pass one CUDA line, or a ref whose tag is that line."
            )
        return full, requested or inferred

    if reuse_previous and previous is not None and not requested:
        prev_tag = previous.cuda_tag or infer_cuda_tag(previous.image)
        env_image, env_tag = _install_default_image()
        if prev_tag and env_tag and prev_tag != env_tag:
            return previous.image, prev_tag
        if prev_tag and env_tag and prev_tag == env_tag:
            return env_image, env_tag
        if previous.image and previous.image not in {DEFAULT_IMAGE, "comfyfleet:phase1", env_image}:
            return previous.image, prev_tag
        return env_image, env_tag or prev_tag or DEFAULT_CUDA_TAG

    if requested:
        env_image, env_tag = _install_default_image()
        if env_image and env_tag == requested:
            return env_image, requested
        return _local_image(requested), requested

    return _install_default_image()


def list_url_host() -> str:
    raw = os.environ.get("COMFYFLEET_PUBLIC_HOST", "").strip()
    if not raw:
        return "0.0.0.0"
    from comfyfleet.public_host import open_host

    return open_host(None, "0.0.0.0", public_host=raw)


def _create_args(layout: FleetLayout, instance: Instance) -> list[str]:
    return build_create_args(
        name=instance.name,
        image=instance.image,
        port=instance.port,
        gpus=instance.gpus,
        models=str(layout.models),
        wildcards=str(layout.wildcards),
        custom_nodes=str(layout.custom_nodes(instance.name)),
        input_dir=str(layout.input_dir(instance.name)),
        output_dir=str(layout.output_dir(instance.name)),
        temp_dir=str(layout.temp_dir(instance.name)),
        instance_dir=str(layout.instance_dir(instance.name)),
        comfy_args=instance.launch.argv(),
        cuda_tag=instance.cuda_tag,
    )


def _prepare_dirs(layout: FleetLayout, name: str) -> None:
    try:
        ensure_wildcards_dir(layout)
    except OSError as exc:
        raise FleetError(
            f"cannot create {layout.wildcards}: {exc}. "
            "That directory is created only when it is missing; an existing directory is left alone."
        ) from exc
    try:
        ensure_instance_host_dirs(layout, name, MODEL_SUBDIRS)
    except OSError as exc:
        failed = getattr(exc, "filename", None) or layout.root
        raise FleetError(
            f"cannot create {failed}: {exc}. Required host paths: "
            f"{layout.models}, {layout.custom_nodes(name)}, "
            f"{layout.input_dir(name)}, {layout.output_dir(name)}, {layout.temp_dir(name)}."
        ) from exc


def _copy_workflow(source: Path, dest: Path) -> None:
    payload = source.read_bytes()
    temporary = dest.with_suffix(".json.tmp")
    temporary.write_bytes(payload)
    temporary.replace(dest)


def _write_metadata(layout: FleetLayout, instance: Instance) -> None:
    path = layout.metadata_file(instance.name)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(instance.to_json(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remove_metadata(layout: FleetLayout, name: str) -> None:
    path = layout.metadata_file(name)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _read_metadata(path: Path) -> Instance:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetError(f"cannot read instance metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FleetError(f"instance metadata is invalid: {path}")
    return Instance.from_json(payload, path)


def _load_if_present(layout: FleetLayout, name: str) -> Instance | None:
    path = layout.metadata_file(name)
    if not path.is_file():
        return None
    return _read_metadata(path)


def _require_instance(layout: FleetLayout, name: str) -> Instance:
    instance = _load_if_present(layout, name)
    if instance is None:
        raise FleetError(
            f"no instance named {name!r}. Create one with "
            "comfyfleet create --workflow /path/to/flow.json."
        )
    return instance


def _reserved_ports(layout: FleetLayout, *, exclude: str) -> set[int]:
    reserved: set[int] = set()
    for instance, _status in list_instances(layout, _StatusFreeDocker()):
        if instance.name == exclude:
            continue
        reserved.add(instance.port)
    return reserved


def _require_name(name: str) -> None:
    if not is_instance_name(name):
        raise FleetError(
            f"invalid instance name {name!r}. Names are lowercase [a-z0-9_-], "
            "start with a letter or digit, and are at most 63 characters."
        )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_limit(explicit: int | None, *, use_env: bool) -> int | None:
    if explicit is not None:
        return explicit
    if not use_env:
        return None
    raw = os.environ.get("COMFYFLEET_MAX_CONCURRENT")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise FleetError("COMFYFLEET_MAX_CONCURRENT must be an integer") from exc
    if value < 1:
        raise FleetError("COMFYFLEET_MAX_CONCURRENT must be >= 1")
    return value


def _concurrency_warning(running_after: int, *, gpu_count: int, limit: int | None) -> str | None:
    if limit is not None and running_after > limit:
        raise FleetError(
            f"refusing to start: {running_after} running instance(s) would exceed "
            f"COMFYFLEET_MAX_CONCURRENT={limit}."
        )
    if running_after > gpu_count:
        return (
            f"starting this instance would run {running_after} ComfyFleet container(s) "
            f"on {gpu_count} GPU(s). Create-many/run-few: stop another instance if the "
            "GPUs are overloaded. This is a warning, not a block."
        )
    return None


def _port_in_use(explicit, docker):
    """Use the caller's probe, or snapshot host listeners and published ports.

    Host CLI and the manager HTTP API both allocate here. A missing callback
    must still skip ports taken outside fleet metadata.
    """

    if explicit is not None:
        return explicit
    return make_port_in_use(docker)


class _StatusFreeDocker:
    """Used only while reading metadata so port reservation does not need Docker."""

    def status(self, _name: str) -> str | None:
        return None
