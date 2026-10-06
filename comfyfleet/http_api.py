"""Thin HTTP adapter over ``comfyfleet.control``.

This server does not create containers, assign ports, copy workflows, or
select GPUs itself. Those stay in ``comfyfleet.control`` and
``comfyfleet.gpu``. Fleet routes require a session cookie or
``Authorization: Bearer``. ``authorize()`` fails closed until that grant.

Contract: CONTROL_HTTP.md.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from email.parser import Parser
from email.policy import compat32
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

from comfyfleet import __version__
from comfyfleet.auth import (
    AuthError,
    LoginGuard,
    SessionStore,
    bearer_token,
    begin_http_request,
    end_http_request,
    grant_http_request,
    login_fail_delay,
    read_cookie,
    read_password,
    request_is_https,
    secrets_equal,
    session_cookie,
)
from comfyfleet.control import (
    Instance,
    authorize,
    create_instance,
    delete_instance,
    force_stop_instance,
    list_instances,
    start_instance,
    stop_instance,
    terminal_argv,
    update_instance_launch,
)
from comfyfleet.ownership import fix_owner
from comfyfleet.prune import prune_dangling_containers
from comfyfleet.terminal import accept_value, bridge_exec
from comfyfleet.errors import FleetError
from comfyfleet.gallery import (
    GalleryFile,
    GalleryMissing,
    delete_gallery_file,
    list_gallery,
    open_gallery_file,
    open_gallery_thumb,
)
from comfyfleet.gpu import Gpu
from comfyfleet.import_container import ImportService
from comfyfleet.launch import combine_extra_args, launch_from_json, parse_launch, split_flag_field
from comfyfleet.paths import FleetLayout
from comfyfleet.public_host import PUBLIC_HOST_ENV

DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_BIND_PORT = 9100
MAX_BODY_BYTES = 32 * 1024 * 1024

_JSON = "application/json; charset=utf-8"
_MISSING_WORKFLOW = (
    "workflow is required. Upload a workflow JSON file as multipart field "
    "'workflow', or pass 'workflow_path' to a .json file this process can read. "
    "There is no baked default workflow."
)
_CREATE_FIELDS = (
    "name",
    "workflow_path",
    "gpu",
    "gpus",
    "start",
    "force",
    "vram",
    "attention",
    "flags",
    "reserve_vram",
    "vram_headroom",
    "preview_method",
    "preview_size",
    "extra_args",
    "comfy_extra_args",
    "install_missing_from_workflow",
    "cuda_tag",
    "instance_image",
)
_FLOAT_FIELDS = {"reserve_vram", "vram_headroom", "preview_size"}
_AUTH_NOTE = (
    "Liveness only. This response has no fleet data. "
    "Fleet routes require a session cookie from POST /api/login "
    "or Authorization: Bearer. Trusted LAN is still recommended. "
    "This gate is not a full internet-hardening product; terminate TLS "
    "at a reverse proxy if you need HTTPS."
)
_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ComfyFleet</title>
</head>
<body>
  <p>ComfyFleet control API placeholder. The web UI files are not in this process.</p>
  <p>API: <a href="/api/health">/api/health</a>. See CONTROL_HTTP.md.</p>
  <p>Fleet routes require a session cookie or Authorization: Bearer. Trusted LAN is still recommended.</p>
</body>
</html>
"""
_BUILTIN_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign in — ComfyFleet</title>
  <style>
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #000; color: #f5f5f7; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    form { width: min(420px, calc(100% - 32px)); padding: 24px; border-radius: 22px;
      background: rgba(28, 28, 30, 0.72); border: 1px solid rgba(255,255,255,0.12); }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { color: rgba(235,235,245,0.62); }
    input, button { width: 100%; min-height: 50px; box-sizing: border-box; font: inherit; }
    input { margin: 8px 0 12px; padding: 0 14px; border-radius: 14px; border: 0; background: rgba(118,118,128,0.28); color: inherit; }
    button { border: 0; border-radius: 14px; background: #0a84ff; color: white; font-weight: 650; }
    .error { color: #ffd7d4; }
  </style>
</head>
<body>
  <form id="login-form" method="post" action="/api/login">
    <h1>ComfyFleet</h1>
    <p id="login-lead">Sign in with the manager password.</p>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <p id="login-error" class="error" hidden></p>
    <button type="submit">Sign in</button>
  </form>
  <script>
    const params = new URLSearchParams(location.search);
    const error = document.querySelector("#login-error");
    if (params.get("expired")) {
      error.hidden = false;
      error.textContent = "Session expired. Sign in again.";
    }
    document.querySelector("#login-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const password = document.querySelector("#password").value;
      const response = await fetch("/api/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ password })
      });
      let payload = null;
      try { payload = await response.json(); } catch (err) { payload = null; }
      if (response.ok && payload && payload.ok) {
        window.location.assign("/");
        return;
      }
      document.querySelector("#password").value = "";
      error.hidden = false;
      error.textContent = response.status === 429
        ? "Too many sign-in attempts. Wait a moment and try again."
        : "Invalid credentials.";
    });
  </script>
</body>
</html>
"""
_PUBLIC_FILES = {
    "/app.css",
    "/comfyfleet-logo-ships.jpg",
    "/favicon.ico",
    "/robots.txt",
}

_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


class HTTPStatusError(Exception):
    """A response this adapter produces before or around control."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class ApiContext:
    """Dependencies the adapter passes into ``comfyfleet.control``."""

    layout: FleetLayout
    docker: object
    detect_gpus: Callable[[], list[Gpu]]
    port_in_use: Callable[[int], bool] | None = None
    ui_dir: Path | None = None
    use_env_limit: bool = True
    host_fallback: str = "127.0.0.1"
    public_host: str | None = None
    password: str | None = None
    sessions: SessionStore | None = None
    login_guard: LoginGuard | None = None
    # Server-side only. The browser cannot replace this command.
    terminal_argv: Callable[[str], list[str]] | None = None
    # Tests inject these. Production leaves them unset.
    node_installer: Callable | None = None
    node_map: dict | None = None
    git_run: Callable | None = None
    importer: object | None = None


@dataclass
class Response:
    status: int
    body: bytes
    content_type: str
    headers: list[tuple[str, str]] = field(default_factory=list)
    stream_fd: int | None = None
    stream_start: int = 0
    stream_length: int | None = None


@dataclass
class _Upload:
    filename: str
    data: bytes


@dataclass
class _CreateForm:
    upload: _Upload | None
    name: str | None
    workflow_path: str | None
    gpu: str | None
    gpus: str | None
    start: bool
    force: bool
    vram: str | None = None
    attention: str | None = None
    flags: str | None = None
    reserve_vram: str | None = None
    vram_headroom: str | None = None
    preview_method: str | None = None
    preview_size: str | None = None
    extra_args: str | None = None
    custom_node_git_urls: list[str] = field(default_factory=list)
    custom_nodes_zips: list[tuple[str, bytes]] = field(default_factory=list)
    custom_nodes_zip_names: list[str] = field(default_factory=list)
    install_missing_from_workflow: bool = True
    cuda_tag: str | None = None
    instance_image: str | None = None


def resolve_ui_dir(explicit: str | None = None) -> Path | None:
    """Find a static UI directory. Missing is fine; the API still serves."""

    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FleetError(f"ui directory not found: {path}")
        return path
    for candidate in (Path.cwd() / "ui", Path(__file__).resolve().parent.parent / "ui"):
        if candidate.is_dir():
            return candidate
    return None


def serve(
    *,
    host: str = DEFAULT_BIND_HOST,
    port: int = DEFAULT_BIND_PORT,
    ui_dir: str | None = None,
    layout: FleetLayout | None = None,
    docker: object | None = None,
    detect_gpus: Callable[[], list[Gpu]] | None = None,
    port_in_use: Callable[[int], bool] | None = None,
    use_env_limit: bool = True,
) -> None:
    """Bind the control API and serve until interrupted.

    ``COMFYFLEET_PASSWORD`` must be a non-empty string. A missing or empty
    value raises ``FleetError`` before the socket is bound.
    """

    password = read_password()
    if not host or not str(host).strip():
        raise FleetError("bind host is required")
    if port < 1 or port > 65535:
        raise FleetError(f"port must be 1..65535, got {port}")
    if layout is None:
        layout = FleetLayout()
    if docker is None:
        from comfyfleet.docker import DockerCLI

        docker = DockerCLI()
    if detect_gpus is None:
        from comfyfleet.gpu import detect_gpus as default_detect_gpus

        detect_gpus = default_detect_gpus
    # port_in_use None: comfyfleet.control snapshots Docker-published ports
    # and host listeners. Do not bind-check inside this process only.
    context = ApiContext(
        layout=layout,
        docker=docker,
        detect_gpus=detect_gpus,
        port_in_use=port_in_use,
        ui_dir=resolve_ui_dir(ui_dir),
        use_env_limit=use_env_limit,
        password=password,
        sessions=SessionStore(),
        login_guard=LoginGuard(fail_delay_s=login_fail_delay()),
    )
    try:
        httpd = make_server(host, port, context)
    except OSError as exc:
        raise FleetError(f"cannot bind {host}:{port}: {exc}") from exc
    print(
        f"comfyfleet: control API on http://{host}:{port}/",
        file=sys.stderr,
    )
    print(
        "comfyfleet: auth required. Fleet routes need a session cookie "
        "or Authorization: Bearer. The password is not logged.",
        file=sys.stderr,
    )
    print(
        "comfyfleet: trusted LAN is still recommended. This gate is not a "
        "full internet-hardening product. Terminate TLS at a reverse proxy "
        "if you need HTTPS. Do not expose this port to the public internet.",
        file=sys.stderr,
    )
    print(
        "comfyfleet: Open Comfy is built in the browser from the page host and the "
        f"instance port. {PUBLIC_HOST_ENV} is not used for that link.",
        file=sys.stderr,
    )
    if context.ui_dir is None:
        print("comfyfleet: no ui/ directory; / is a placeholder.", file=sys.stderr)
    else:
        print(f"comfyfleet: static files from {context.ui_dir}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        raise
    finally:
        httpd.server_close()


def make_server(host: str, port: int, context: ApiContext) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            upgrade = (self.headers.get("Upgrade") or "").lower()
            if upgrade == "websocket" and _terminal_name(path) is not None:
                self._serve_terminal(path)
                return
            self._respond("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._respond("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            # Delete is POST /api/instances/{name}/delete. This does not remove anything.
            self._respond("DELETE")

        def log_message(self, fmt: str, *args) -> None:
            # Path only. Query strings and headers are omitted so a password
            # cannot land in the access log.
            path = urlsplit(self.path).path
            print(
                f"comfyfleet: {self.address_string()} {self.command} {path}",
                file=sys.stderr,
            )

        def _respond(self, method: str) -> None:
            response: Response | None = None
            try:
                body = _read_body(self)
                split = urlsplit(self.path)
                header_map = {key: value for key, value in self.headers.items()}
                client_ip = self.client_address[0] if self.client_address else ""
                response = dispatch(
                    context,
                    method,
                    split.path,
                    self.headers.get("Host"),
                    body,
                    self.headers.get("Content-Type"),
                    header_map,
                    client_ip,
                    split.query,
                )
            except HTTPStatusError as exc:
                response = _json(exc.status, {"ok": False, "error": exc.message})
            except AuthError as exc:
                response = _json(401, {"ok": False, "error": exc.message})
            except FleetError as exc:
                response = _json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                print(f"comfyfleet: internal error: {exc}", file=sys.stderr)
                response = _json(500, {"ok": False, "error": "internal error"})
            try:
                self._write_response(response)
            finally:
                if response is not None and response.stream_fd is not None:
                    os.close(response.stream_fd)
                    response.stream_fd = None

        def _write_response(self, response: Response) -> None:
            if response.stream_fd is not None:
                length = 0 if response.stream_length is None else response.stream_length
                self.send_response(response.status)
                self.send_header("Content-Type", response.content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("X-Content-Type-Options", "nosniff")
                if not any(key.lower() == "cache-control" for key, _value in response.headers):
                    self.send_header("Cache-Control", "no-store")
                for key, value in response.headers:
                    self.send_header(key, value)
                self.end_headers()
                remaining = length
                os.lseek(response.stream_fd, response.stream_start, os.SEEK_SET)
                while remaining > 0:
                    chunk = os.read(response.stream_fd, min(65536, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)
                return
            payload = response.body
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in response.headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _serve_terminal(self, path: str) -> None:
            """Proxy a shell. The Docker socket is not part of this response."""

            self.close_connection = True
            token = begin_http_request()
            try:
                name = _terminal_name(path)
                headers = {key: value for key, value in self.headers.items()}
                if name is None or not _request_authenticated(context, headers):
                    self._json_now(401, "unauthorized")
                    return
                grant_http_request()
                try:
                    argv = terminal_argv(
                        name,
                        layout=context.layout,
                        docker=context.docker,
                        argv_for=context.terminal_argv,
                    )
                except AuthError as exc:
                    self._json_now(401, exc.message)
                    return
                except FleetError as exc:
                    self._json_now(400, str(exc))
                    return
                key = (self.headers.get("Sec-WebSocket-Key") or "").strip()
                if not key:
                    self._json_now(400, "websocket upgrade requires Sec-WebSocket-Key")
                    return
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept_value(key))
                self.end_headers()
                self.wfile.flush()
                try:
                    bridge_exec(self.connection, argv)
                except (ConnectionError, OSError, FleetError) as exc:
                    print(f"comfyfleet: terminal closed: {exc}", file=sys.stderr)
            finally:
                end_http_request(token)

        def _json_now(self, status: int, message: str) -> None:
            body = json.dumps({"ok": False, "error": message}).encode("utf-8") + b"\n"
            self.send_response(status)
            self.send_header("Content-Type", _JSON)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    class ControlHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    return ControlHTTPServer((host, port), Handler)


def dispatch(
    context: ApiContext,
    method: str,
    path: str,
    host_header: str | None,
    body: bytes,
    content_type: str | None,
    headers: dict | None = None,
    client_ip: str | None = None,
    query: str = "",
) -> Response:
    """Route one request. Control errors propagate as ``FleetError``."""

    token = begin_http_request()
    try:
        try:
            return _route(
                context,
                method,
                path,
                host_header,
                body,
                content_type,
                headers,
                client_ip or "",
                query,
            )
        except AuthError as exc:
            return _json(401, {"ok": False, "error": exc.message})
    finally:
        end_http_request(token)


def _route(
    context: ApiContext,
    method: str,
    path: str,
    host_header: str | None,
    body: bytes,
    content_type: str | None,
    headers: dict | None,
    client_ip: str,
    query: str = "",
) -> Response:
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    if path == "/api/health":
        _require_method(method, "GET")
        return _health()
    if path == "/api/login":
        _require_method(method, "POST")
        return _login(context, body, content_type, headers, client_ip)
    if path == "/api/logout":
        _require_method(method, "POST")
        return _logout(context, headers)
    if path.startswith("/api/"):
        _require_fleet_auth(context, method, path, headers)
        grant_http_request()
        return _fleet(context, method, path, host_header, body, content_type, query, headers)
    if path in {"/login", "/login.html"} or path in _PUBLIC_FILES:
        if _request_authenticated(context, headers) and path in {"/login", "/login.html"}:
            return _redirect("/")
        if method != "GET":
            raise HTTPStatusError(405, "method not allowed")
        return _static(context, path)
    if not _request_authenticated(context, headers):
        if method != "GET":
            raise AuthError("unauthorized")
        cookie = read_cookie(_header(headers, "Cookie"))
        bearer = bearer_token(_header(headers, "Authorization"))
        if cookie and bearer is None:
            return _redirect("/login?expired=1")
        return _redirect("/login")
    if method != "GET":
        raise HTTPStatusError(405, "method not allowed")
    grant_http_request()
    return _static(context, path)


def _fleet(
    context: ApiContext,
    method: str,
    path: str,
    host_header: str | None,
    body: bytes,
    content_type: str | None,
    query: str = "",
    headers: dict | None = None,
) -> Response:
    del host_header  # Open Comfy does not use the request host or a pinned public host.
    if path == "/api/gallery" or path.startswith("/api/gallery/"):
        return _gallery(context, method, path, query, headers, body, content_type)
    if path == "/api/import" or path.startswith("/api/import/"):
        return _import_route(context, method, path, body, content_type)
    if path == "/api/host/fix-owner":
        _require_method(method, "POST")
        return _fix_owner(context, body, content_type)
    if path == "/api/host/prune-dangling":
        _require_method(method, "POST")
        return _prune_dangling(context)
    if path == "/api/gpus":
        _require_method(method, "GET")
        return _gpus(context)
    if path == "/api/instances":
        if method == "GET":
            return _list(context)
        if method == "POST":
            return _create(context, body, content_type)
        raise HTTPStatusError(405, "method not allowed")
    action = _instance_action(path)
    if action is not None:
        name, verb = action
        if verb == "terminal":
            _require_method(method, "GET")
            raise HTTPStatusError(400, "terminal requires a websocket upgrade")
        if verb == "delete":
            if method != "POST":
                raise HTTPStatusError(
                    405,
                    "method not allowed. Delete is POST /api/instances/{name}/delete. "
                    "That removes the container and the fleet record and leaves host mounts.",
                )
            return _delete(context, name)
        _require_method(method, "POST")
        if verb == "start":
            return _start(context, name)
        if verb == "stop":
            return _stop(context, name)
        if verb == "force-stop":
            return _force_stop(context, name)
        if verb == "launch":
            return _update_launch(context, name, body, content_type)
        raise HTTPStatusError(404, "not found")
    raise HTTPStatusError(404, "not found")


def _import_service(context: ApiContext) -> ImportService:
    if context.importer is None:
        context.importer = ImportService(context.layout, port_in_use=context.port_in_use)
    return context.importer  # type: ignore[return-value]


def _import_route(
    context: ApiContext,
    method: str,
    path: str,
    body: bytes,
    content_type: str | None,
) -> Response:
    service = _import_service(context)
    if path == "/api/import/containers":
        _require_method(method, "GET")
        return _json(200, {"ok": True, "containers": service.list_containers(context.docker)})
    if path == "/api/import/inspect":
        _require_method(method, "POST")
        payload = _import_body(body, content_type)
        return _json(200, service.inspect(context.docker, str(payload.get("container") or "")))
    if path == "/api/import/jobs":
        if method == "GET":
            return _json(200, {"ok": True, "jobs": service.list_jobs()})
        if method == "POST":
            payload = _import_body(body, content_type)
            job = service.create_job(context.docker, context.detect_gpus(), payload)
            return _json(200, {"ok": True, "job": service.public_job(job)})
        raise HTTPStatusError(405, "method not allowed")
    if path == "/api/import/active":
        _require_method(method, "GET")
        return _json(200, {"ok": True, "job": service.active_job()})
    if path == "/api/import/duplicates":
        if method != "GET":
            raise HTTPStatusError(405, "duplicates list is read-only")
        return _json(200, {"ok": True, "duplicates": service.duplicates.read()})
    prefix = "/api/import/jobs/"
    if path.startswith(prefix):
        job_id, _, verb = path[len(prefix) :].partition("/")
        if not _import_job_id(job_id):
            raise HTTPStatusError(404, "not found")
        if not verb:
            _require_method(method, "GET")
            return _json(200, {"ok": True, "job": service.public_job(service.get_job(job_id))})
        if verb == "log" and method == "GET":
            text = service.log_text(job_id)
            filename = f"{job_id}.log"
            return Response(
                200,
                text.encode("utf-8"),
                "text/plain; charset=utf-8",
                [("Content-Disposition", f'attachment; filename="{filename}"')],
            )
        _require_method(method, "POST")
        if verb == "start":
            payload = _import_body(body, content_type) if body.strip() else {}
            job = service.start_transfer(job_id, mode=payload.get("mode"))
            return _json(200, {"ok": True, "job": service.public_job(job)})
        if verb == "pause":
            return _json(200, {"ok": True, "job": service.public_job(service.pause(job_id))})
        if verb == "resume":
            job = service.resume(job_id, context.docker, context.detect_gpus())
            return _json(200, {"ok": True, "job": service.public_job(job)})
        if verb == "cancel":
            return _json(200, {"ok": True, "job": service.public_job(service.cancel(job_id))})
        if verb == "dismiss":
            return _json(200, {"ok": True, "job": service.public_job(service.dismiss(job_id))})
        if verb == "remove-old":
            return _json(200, {"ok": True, "job": service.public_job(service.remove_old(job_id, context.docker))})
    raise HTTPStatusError(404, "not found")


def _import_body(body: bytes, content_type: str | None) -> dict:
    if not body or not body.strip():
        return {}
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media not in {"", "application/json"}:
        raise FleetError("import body must be a JSON object")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FleetError(f"import body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FleetError("import body must be a JSON object")
    return payload


def _import_job_id(job_id: str) -> bool:
    return len(job_id) == 16 and job_id.startswith("imp-") and all(
        char in "0123456789abcdef" for char in job_id[4:]
    )


def _gallery(
    context: ApiContext,
    method: str,
    path: str,
    query: str,
    headers: dict | None,
    body: bytes,
    content_type: str | None,
) -> Response:
    try:
        if path == "/api/gallery":
            _require_method(method, "GET")
            return _gallery_list(context, query)
        if path == "/api/gallery/media":
            _require_method(method, "GET")
            return _gallery_media(context, query, headers)
        if path == "/api/gallery/thumb":
            _require_method(method, "GET")
            return _gallery_thumb(context, query)
        if path == "/api/gallery/delete":
            _require_method(method, "POST")
            return _gallery_delete(context, body, content_type)
    except GalleryMissing as exc:
        return _json(404, {"ok": False, "error": str(exc)})
    raise HTTPStatusError(404, "not found")


def _gallery_list(context: ApiContext, query: str) -> Response:
    offset = _gallery_int(query, "offset", 0)
    limit = _gallery_int(query, "limit", 48)
    payload = list_gallery(
        context.layout,
        context.docker,
        instance=_gallery_query_value(query, "instance"),
        offset=offset,
        limit=limit,
    )
    return _json(200, payload)


def _gallery_media(context: ApiContext, query: str, headers: dict | None) -> Response:
    instance, relative = _gallery_target(query)
    opened = open_gallery_file(context.layout, instance, relative)
    try:
        status, start, length = _byte_range(_header(headers, "Range"), opened.size)
    except Exception:
        os.close(opened.fd)
        raise
    attachment = _gallery_query_value(query, "download") == "1"
    mode = "attachment" if attachment else "inline"
    response = _file_response(opened, status, start, length, mode)
    return response


def _gallery_thumb(context: ApiContext, query: str) -> Response:
    instance, relative = _gallery_target(query)
    opened = open_gallery_thumb(context.layout, instance, relative)
    if isinstance(opened, GalleryFile):
        return _file_response(opened, 200, 0, opened.size, "inline")
    data, content_type = opened
    return Response(200, data, content_type)


def _gallery_delete(context: ApiContext, body: bytes, content_type: str | None) -> Response:
    instance, relative = _gallery_delete_body(body, content_type)
    payload = delete_gallery_file(context.layout, instance, relative)
    return _json(200, payload)


def _file_response(opened: GalleryFile, status: int, start: int, length: int, mode: str) -> Response:
    headers = [
        ("Accept-Ranges", "bytes"),
        ("Content-Disposition", _content_disposition(mode, opened.filename)),
        ("Cache-Control", "private, max-age=300"),
    ]
    if status == 206:
        end = start + length - 1 if length else start
        headers.append(("Content-Range", f"bytes {start}-{end}/{opened.size}"))
    return Response(
        status,
        b"",
        opened.content_type,
        headers=headers,
        stream_fd=opened.fd,
        stream_start=start,
        stream_length=length,
    )


def _gallery_target(query: str) -> tuple[str, str]:
    instance = _gallery_query_value(query, "instance")
    relative = _gallery_query_value(query, "path")
    if not instance or relative is None or relative == "":
        raise FleetError("gallery instance and path are required")
    return instance, relative


def _gallery_delete_body(body: bytes, content_type: str | None) -> tuple[str, str]:
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media != "application/json":
        raise FleetError("gallery delete requires application/json")
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FleetError("gallery delete is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FleetError("gallery delete requires a JSON object")
    instance = payload.get("instance")
    relative = payload.get("path")
    if not isinstance(instance, str) or not isinstance(relative, str):
        raise FleetError("gallery delete requires instance and path strings")
    return instance, relative


def _gallery_query_value(query: str, key: str) -> str | None:
    parsed = parse_qs(query, keep_blank_values=True)
    values = parsed.get(key)
    if not values:
        return None
    if len(values) != 1 or not isinstance(values[0], str):
        raise FleetError(f"invalid gallery {key}")
    return values[0]


def _gallery_int(query: str, key: str, default: int) -> int:
    text = _gallery_query_value(query, key)
    if text is None or text == "":
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise FleetError(f"invalid gallery {key}") from exc


def _byte_range(header: str | None, size: int) -> tuple[int, int, int]:
    """Status, start, and length for a single byte range."""

    if size < 0:
        size = 0
    if header is None or not str(header).strip():
        return 200, 0, size
    text = str(header).strip()
    if not text.lower().startswith("bytes=") or "," in text:
        raise HTTPStatusError(416, "range not satisfiable")
    spec = text.split("=", 1)[1].strip()
    if "-" not in spec:
        raise HTTPStatusError(416, "range not satisfiable")
    start_s, end_s = spec.split("-", 1)
    try:
        if start_s == "":
            suffix = int(end_s)
            if suffix <= 0 or size == 0:
                raise HTTPStatusError(416, "range not satisfiable")
            if suffix > size:
                suffix = size
            start = size - suffix
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
    except ValueError as exc:
        raise HTTPStatusError(416, "range not satisfiable") from exc
    if size == 0 or start < 0 or start >= size or end < start:
        raise HTTPStatusError(416, "range not satisfiable")
    end = min(end, size - 1)
    return 206, start, end - start + 1


def _content_disposition(mode: str, filename: str) -> str:
    base = filename.replace("\\", "/").split("/")[-1]
    base = "".join(ch for ch in base if ch not in {'"', "\r", "\n", "\x00"})
    if not base or base in {".", ".."}:
        base = "download"
    ascii_name = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in base) or "download"
    return f"{mode}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(base, safe='')}"


def _health() -> Response:
    return _json(
        200,
        {
            "ok": True,
            "service": "comfyfleet",
            "version": __version__,
            "auth": "required",
            "note": _AUTH_NOTE,
        },
    )


def _require_fleet_auth(context: ApiContext, method: str, path: str, headers: dict | None) -> None:
    """Grant is required. ``authorize()`` fails closed when it is not."""

    if _request_authenticated(context, headers):
        return
    action = _protected_action(method, path)
    cookie = read_cookie(_header(headers, "Cookie"))
    bearer = bearer_token(_header(headers, "Authorization"))
    message = "session expired" if cookie and bearer is None else "unauthorized"
    if action is not None:
        try:
            authorize(action)
        except AuthError:
            raise AuthError(message) from None
    raise AuthError(message)


def _protected_action(method: str, path: str) -> str | None:
    if path == "/api/host/fix-owner":
        return "fix-owner" if method == "POST" else None
    if path == "/api/host/prune-dangling":
        return "prune-dangling" if method == "POST" else None
    if path == "/api/gpus":
        return "list"
    if path == "/api/gallery" or path.startswith("/api/gallery/"):
        return "gallery"
    if path == "/api/import" or path.startswith("/api/import/"):
        return "import"
    if path == "/api/instances":
        if method == "POST":
            return "create"
        return "list"
    prefix = "/api/instances/"
    if path.startswith(prefix):
        _name, sep, verb = path[len(prefix) :].partition("/")
        if sep == "/" and verb == "launch":
            return "update"
        if sep == "/" and verb in {"start", "stop", "force-stop", "delete", "terminal"}:
            return verb
        return "list"
    return None


def _request_authenticated(context: ApiContext, headers: dict | None) -> bool:
    _ensure_auth_state(context)
    cookie = read_cookie(_header(headers, "Cookie"))
    if context.sessions is not None and context.sessions.valid(cookie):
        return True
    token = bearer_token(_header(headers, "Authorization"))
    if token is None:
        return False
    return secrets_equal(context.password, token)


def _ensure_auth_state(context: ApiContext) -> None:
    if context.sessions is None:
        context.sessions = SessionStore()
    if context.login_guard is None:
        context.login_guard = LoginGuard(fail_delay_s=0.25)


def _login(
    context: ApiContext,
    body: bytes,
    content_type: str | None,
    headers: dict | None,
    client_ip: str,
) -> Response:
    _ensure_auth_state(context)
    assert context.login_guard is not None
    assert context.sessions is not None
    key = client_ip or "unknown"
    secure = request_is_https(_header(headers, "X-Forwarded-Proto"))
    presented = _login_password(body, content_type)
    if context.login_guard.blocked(key):
        secrets_equal(context.password, presented or "")
        context.login_guard.pause()
        raise HTTPStatusError(429, "too many login attempts")
    if not secrets_equal(context.password, presented or ""):
        context.login_guard.record_failure(key)
        raise HTTPStatusError(401, "invalid credentials")
    context.login_guard.record_success(key)
    session_id = context.sessions.create()
    response = _json(200, {"ok": True})
    response.headers.append(("Set-Cookie", session_cookie(session_id, secure=secure)))
    return response


def _logout(context: ApiContext, headers: dict | None) -> Response:
    _ensure_auth_state(context)
    assert context.sessions is not None
    cookie = read_cookie(_header(headers, "Cookie"))
    context.sessions.revoke(cookie)
    secure = request_is_https(_header(headers, "X-Forwarded-Proto"))
    response = _json(200, {"ok": True})
    response.headers.append(("Set-Cookie", session_cookie("", secure=secure, clear=True)))
    return response


def _login_password(body: bytes, content_type: str | None) -> str | None:
    """Pull the password out of the body. Never include it in an error."""

    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media in {"", "application/json"}:
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("password")
        if not isinstance(value, str):
            return None
        return value
    if media == "application/x-www-form-urlencoded":
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return None
        parsed = parse_qs(text, keep_blank_values=True)
        values = parsed.get("password") or []
        if len(values) != 1 or not isinstance(values[0], str):
            return None
        return values[0]
    return None


def _redirect(location: str) -> Response:
    body = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>Sign in</title></head><body><p><a href=\"{location}\">Sign in</a></p></body></html>"
    ).encode("utf-8")
    response = Response(302, body, "text/html; charset=utf-8")
    response.headers.append(("Location", location))
    return response


def _header(headers: dict | None, name: str) -> str | None:
    if not headers:
        return None
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return None if value is None else str(value)
    return None


def _gpus(context: ApiContext) -> Response:
    try:
        found = context.detect_gpus()
    except FleetError as exc:
        raise HTTPStatusError(503, str(exc)) from exc
    return _json(
        200,
        {
            "ok": True,
            "gpus": [
                {"index": gpu.index, "name": gpu.name, "memory": gpu.memory}
                for gpu in found
            ],
        },
    )


def _list(context: ApiContext) -> Response:
    rows = list_instances(context.layout, context.docker)
    return _json(
        200,
        {
            "ok": True,
            "instances": [
                _instance_json(instance, status) for instance, status in rows
            ],
        },
    )


def _create(context: ApiContext, body: bytes, content_type: str | None) -> Response:
    form = _parse_create_form(body, content_type)
    if form.upload is None and not form.workflow_path:
        raise FleetError(_MISSING_WORKFLOW)
    if form.upload is not None and form.workflow_path:
        raise FleetError("pass either a workflow upload or workflow_path, not both.")
    if form.upload is not None:
        workflow = _materialize_upload(form.upload)
        cleanup = workflow.parent
    else:
        workflow = Path(form.workflow_path or "")
        cleanup = None
    try:
        result = create_instance(
            workflow,
            layout=context.layout,
            docker=context.docker,
            gpus=context.detect_gpus(),
            gpu=form.gpu,
            gpus_spec=form.gpus,
            interactive=False,
            prompt=None,
            cuda_tag=form.cuda_tag,
            instance_image=form.instance_image,
            start=form.start,
            force=form.force,
            port_in_use=context.port_in_use,
            use_env_limit=context.use_env_limit,
            launch=parse_launch(
                vram=form.vram,
                attention=form.attention,
                flags=split_flag_field(form.flags),
                reserve_vram=form.reserve_vram,
                vram_headroom=form.vram_headroom,
                preview_method=form.preview_method,
                preview_size=form.preview_size,
                extra_args=form.extra_args,
            ),
            name=form.name,
            custom_node_git_urls=form.custom_node_git_urls,
            custom_nodes_zip=_zip_payload(form.custom_nodes_zips),
            custom_nodes_zip_names=form.custom_nodes_zip_names or None,
            custom_nodes_zip_labels=[name for name, _data in form.custom_nodes_zips] or None,
            install_missing_from_workflow=form.install_missing_from_workflow,
            node_installer=context.node_installer,
            node_map=context.node_map,
            git_run=context.git_run,
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
    for item in result.warnings:
        print(f"comfyfleet: warning: {item}", file=sys.stderr)
    status = context.docker.status(result.instance.name) or "missing"
    return _json(
        200,
        {
            "ok": True,
            "started": result.started,
            "warning": result.warning,
            "warnings": list(result.warnings),
            "instance": _instance_json(result.instance, status),
        },
    )


def _start(context: ApiContext, name: str) -> Response:
    result = start_instance(
        name,
        layout=context.layout,
        docker=context.docker,
        gpus=context.detect_gpus(),
        port_in_use=context.port_in_use,
        use_env_limit=context.use_env_limit,
    )
    status = context.docker.status(result.instance.name) or "missing"
    return _json(
        200,
        {
            "ok": True,
            "started": result.started,
            "warning": result.warning,
            "instance": _instance_json(result.instance, status),
        },
    )


def _stop(context: ApiContext, name: str) -> Response:
    instance = stop_instance(name, layout=context.layout, docker=context.docker)
    status = context.docker.status(instance.name) or "missing"
    return _json(200, {"ok": True, "instance": _instance_json(instance, status)})


def _force_stop(context: ApiContext, name: str) -> Response:
    instance = force_stop_instance(name, layout=context.layout, docker=context.docker)
    status = context.docker.status(instance.name) or "missing"
    return _json(200, {"ok": True, "instance": _instance_json(instance, status)})


def _fix_owner_names(body: bytes, content_type: str | None) -> tuple[str | None, str | None]:
    """Optional user and group. An empty body leaves both for the default."""

    if not body or not body.strip():
        return None, None
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media not in {"", "application/json"}:
        raise FleetError("fix-owner body must be a JSON object with optional user and group")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FleetError(f"fix-owner body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FleetError("fix-owner body must be a JSON object with optional user and group")
    user = payload.get("user")
    group = payload.get("group")
    if user is not None and not isinstance(user, str):
        raise FleetError("fix-owner user must be a string")
    if group is not None and not isinstance(group, str):
        raise FleetError("fix-owner group must be a string")
    return user, group


def _fix_owner(context: ApiContext, body: bytes, content_type: str | None) -> Response:
    user, group = _fix_owner_names(body, content_type)
    result = fix_owner(context.layout, user=user, group=group)
    return _json(
        200,
        {
            "ok": True,
            "user": result.user,
            "group": result.group,
            "uid": result.uid,
            "gid": result.gid,
            "paths": list(result.paths),
        },
    )


def _prune_dangling(context: ApiContext) -> Response:
    result = prune_dangling_containers(context.docker)
    return _json(
        200,
        {
            "ok": True,
            "removed": list(result.removed),
            "kept_managed": list(result.kept_managed),
        },
    )


def _delete(context: ApiContext, name: str) -> Response:
    instance = delete_instance(name, layout=context.layout, docker=context.docker)
    return _json(200, {"ok": True, "deleted": instance.name})


def _update_launch(context: ApiContext, name: str, body: bytes, content_type: str | None) -> Response:
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media != "application/json":
        raise FleetError("launch update requires application/json")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FleetError("launch update is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FleetError("launch update must be a JSON object")
    result = update_instance_launch(
        name,
        launch_from_json(payload, path=name),
        layout=context.layout,
        docker=context.docker,
        gpus=context.detect_gpus(),
        port_in_use=context.port_in_use,
        use_env_limit=context.use_env_limit,
    )
    status = context.docker.status(result.instance.name) or "missing"
    return _json(
        200,
        {
            "ok": True,
            "started": result.started,
            "warning": result.warning,
            "instance": _instance_json(result.instance, status),
        },
    )


def _instance_json(instance: Instance, status: str) -> dict:
    return {
        "name": instance.name,
        "status": status,
        "port": instance.port,
        "gpus": list(instance.gpus),
        "image": instance.image,
        "cuda_tag": instance.cuda_tag,
        "launch": {**instance.launch.to_json(), "argv": instance.launch.argv()},
    }


def _terminal_name(path: str) -> str | None:
    action = None
    try:
        action = _instance_action(path)
    except FleetError:
        return None
    if action is None:
        return None
    name, verb = action
    if verb != "terminal":
        return None
    return name


def _instance_action(path: str) -> tuple[str, str] | None:
    prefix = "/api/instances/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    name, sep, verb = rest.partition("/")
    if sep != "/" or not name or not verb or "/" in verb:
        return None
    if verb not in {"start", "stop", "force-stop", "delete", "terminal", "launch"}:
        return None
    decoded = unquote(name)
    if (
        not decoded
        or "/" in decoded
        or "\\" in decoded
        or "\x00" in decoded
        or decoded in {".", ".."}
    ):
        raise FleetError(f"invalid instance name {decoded!r}")
    return decoded, verb


def _static(context: ApiContext, path: str) -> Response:
    if path in {"", "/"}:
        path = "/index.html"
    if path in {"/login", "/login.html"}:
        if context.ui_dir is not None:
            found = _safe_static(context.ui_dir, "/login.html")
            if found is not None:
                return Response(200, found.read_bytes(), "text/html; charset=utf-8")
        return Response(200, _BUILTIN_LOGIN_HTML.encode("utf-8"), "text/html; charset=utf-8")
    if context.ui_dir is not None:
        found = _safe_static(context.ui_dir, path)
        if found is not None:
            mime = _STATIC_TYPES.get(found.suffix.lower(), "application/octet-stream")
            return Response(200, found.read_bytes(), mime)
    if path == "/index.html":
        return Response(200, _PLACEHOLDER_HTML.encode("utf-8"), "text/html; charset=utf-8")
    raise HTTPStatusError(404, "not found")


def _safe_static(ui_dir: Path, url_path: str) -> Path | None:
    if not url_path.startswith("/") or "\\" in url_path or "\x00" in url_path:
        return None
    raw = unquote(url_path).lstrip("/")
    if not raw or "\x00" in raw:
        return None
    parts = Path(raw).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    root = ui_dir.resolve()
    candidate = (root.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _zip_payload(uploads: list[tuple[str, bytes]]) -> bytes | list[bytes] | None:
    """One archive stays ``bytes`` so the single-zip path is unchanged."""

    if not uploads:
        return None
    if len(uploads) == 1:
        return uploads[0][1]
    return [data for _name, data in uploads]


def _parse_create_form(body: bytes, content_type: str | None) -> _CreateForm:
    media = (content_type or "").split(";", 1)[0].strip().lower()
    zip_uploads: list[tuple[str, bytes]] = []
    zip_names: list[str] = []
    if body == b"" and media in {"", "application/json", "application/x-www-form-urlencoded"}:
        fields: dict[str, str] = {}
        git_urls: list[str] = []
        upload = None
    elif media == "application/json":
        fields, git_urls = _json_fields(body)
        upload = None
    elif media == "application/x-www-form-urlencoded":
        fields, git_urls = _urlencoded_fields(body)
        upload = None
    elif media == "multipart/form-data":
        fields, upload, git_urls, zip_uploads, zip_names = _multipart_fields(content_type or "", body)
    else:
        raise HTTPStatusError(
            415,
            "Content-Type must be application/json, multipart/form-data, "
            "or application/x-www-form-urlencoded",
        )
    return _CreateForm(
        upload=upload,
        name=_optional_str(fields.get("name")),
        workflow_path=_optional_str(fields.get("workflow_path")),
        gpu=_optional_str(fields.get("gpu")),
        gpus=_optional_str(fields.get("gpus")),
        start=_as_bool(fields.get("start"), default=False),
        force=_as_bool(fields.get("force"), default=False),
        vram=_optional_str(fields.get("vram")),
        attention=_optional_str(fields.get("attention")),
        flags=_optional_str(fields.get("flags")),
        reserve_vram=_optional_str(fields.get("reserve_vram")),
        vram_headroom=_optional_str(fields.get("vram_headroom")),
        preview_method=_optional_str(fields.get("preview_method")),
        preview_size=_optional_str(fields.get("preview_size")),
        extra_args=combine_extra_args(fields.get("extra_args"), fields.get("comfy_extra_args")),
        custom_node_git_urls=git_urls,
        custom_nodes_zips=zip_uploads,
        custom_nodes_zip_names=zip_names,
        install_missing_from_workflow=_as_bool(
            fields.get("install_missing_from_workflow"),
            default=True,
        ),
        cuda_tag=_optional_str(fields.get("cuda_tag")),
        instance_image=_optional_str(fields.get("instance_image")),
    )


def _json_fields(body: bytes) -> tuple[dict[str, str], list[str]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FleetError(f"request body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FleetError(
            "JSON body must be an object with workflow_path, gpu, gpus, start, force, "
            "optional cuda_tag (cu130 or cu124) or instance_image, "
            "and optional launch fields (vram, attention, flags, reserve_vram, "
            "vram_headroom, preview_method, preview_size, extra_args, comfy_extra_args)."
        )
    fields: dict[str, str] = {}
    git_urls = _git_url_values(payload.get("custom_node_git_urls"))
    for key in _CREATE_FIELDS:
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if key in {"flags", "comfy_extra_args"} and isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise FleetError(f"field {key!r} must be a string or a list of strings")
            fields[key] = ",".join(value) if key == "flags" else shlex.join(value)
            continue
        if isinstance(value, bool):
            fields[key] = "true" if value else "false"
        elif isinstance(value, int) and not isinstance(value, bool):
            fields[key] = str(value)
        elif isinstance(value, float) and key in _FLOAT_FIELDS:
            fields[key] = str(value)
        elif isinstance(value, str):
            fields[key] = value
        else:
            raise FleetError(f"field {key!r} must be a string, boolean, or integer")
    return fields, git_urls


def _urlencoded_fields(body: bytes) -> tuple[dict[str, str], list[str]]:
    from urllib.parse import parse_qs

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FleetError("form body is not UTF-8") from exc
    parsed = parse_qs(text, keep_blank_values=True)
    fields: dict[str, str] = {}
    git_urls = _git_url_values(parsed.get("custom_node_git_urls"))
    for key in _CREATE_FIELDS:
        values = parsed.get(key)
        if not values:
            continue
        if len(values) > 1:
            raise FleetError(f"duplicate form field {key!r}")
        fields[key] = values[0]
    return fields, git_urls


def _multipart_fields(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, str], _Upload | None, list[str], list[tuple[str, bytes]], list[str]]:
    boundary = _boundary(content_type)
    fields: dict[str, str] = {}
    upload: _Upload | None = None
    git_urls: list[str] = []
    zip_uploads: list[tuple[str, bytes]] = []
    zip_names: list[str] = []
    for headers, data in _multipart_parts(boundary, body):
        name = headers.get("name")
        if not name:
            continue
        filename = headers.get("filename")
        if filename:
            if name == "custom_nodes_zip":
                zip_uploads.append((_zip_filename(filename), data))
                continue
            if name == "custom_nodes_zip_name":
                raise FleetError("custom_nodes_zip_name must be a text field, not a file")
            safe = _upload_filename(filename)
            if name != "workflow":
                raise FleetError("upload the workflow JSON as the multipart field 'workflow'")
            if upload is not None:
                raise FleetError("duplicate workflow upload")
            upload = _Upload(filename=safe, data=data)
            continue
        if name == "custom_node_git_urls":
            try:
                git_urls.extend(_split_git_url_field(data.decode("utf-8")))
            except UnicodeDecodeError as exc:
                raise FleetError("form field 'custom_node_git_urls' is not UTF-8") from exc
            continue
        if name == "custom_nodes_zip_name":
            try:
                zip_names.append(data.decode("utf-8").strip())
            except UnicodeDecodeError as exc:
                raise FleetError("form field 'custom_nodes_zip_name' is not UTF-8") from exc
            continue
        if name == "custom_nodes_zip":
            # A text field is not a zip. Blank is a no-op. Non-file values are
            # not read from disk.
            if data.strip():
                raise FleetError(
                    "custom_nodes_zip must be a multipart file, not a text field"
                )
            continue
        if name not in _CREATE_FIELDS:
            continue
        if name in fields:
            raise FleetError(f"duplicate form field {name!r}")
        try:
            fields[name] = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FleetError(f"form field {name!r} is not UTF-8") from exc
    return fields, upload, git_urls, zip_uploads, zip_names


def _git_url_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _split_git_url_field(value)
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise FleetError("custom_node_git_urls must be a string or a list of strings")
            found.extend(_split_git_url_field(item))
        return found
    raise FleetError("custom_node_git_urls must be a string or a list of strings")


def _split_git_url_field(value: str) -> list[str]:
    """One field may hold several URLs separated by newlines or commas. Blanks drop out."""

    found: list[str] = []
    for line in value.replace(",", "\n").splitlines():
        text = line.strip()
        if text:
            found.append(text)
    return found


def _zip_filename(filename: str) -> str:
    base = filename.replace("\\", "/").split("/")[-1]
    if not base or base in {".", ".."} or "\x00" in base:
        raise FleetError("custom_nodes_zip filename is invalid")
    return base


def _boundary(content_type: str) -> bytes:
    message = Parser(policy=compat32).parsestr(f"Content-Type: {content_type}\n")
    value = message.get_param("boundary", header="Content-Type")
    if isinstance(value, tuple):
        value = value[-1]
    if not value or not isinstance(value, str):
        raise FleetError("multipart body is missing a boundary")
    try:
        return value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise FleetError("multipart boundary must be ASCII") from exc


def _multipart_parts(boundary: bytes, body: bytes):
    marker = b"--" + boundary
    chunks = body.split(marker)
    if len(chunks) < 2:
        raise FleetError("malformed multipart body")
    for chunk in chunks[1:]:
        if chunk.startswith(b"--"):
            break
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        elif chunk.startswith(b"\n"):
            chunk = chunk[1:]
        header_blob, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            header_blob, sep, data = chunk.partition(b"\n\n")
        if not sep:
            raise FleetError("malformed multipart body")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        elif data.endswith(b"\n"):
            data = data[:-1]
        yield _part_headers(header_blob), data


def _part_headers(header_blob: bytes) -> dict[str, str]:
    text = header_blob.decode("iso-8859-1")
    message = Parser(policy=compat32).parsestr(text + "\n")
    found: dict[str, str] = {}
    for key in ("name", "filename"):
        value = message.get_param(key, header="Content-Disposition")
        if isinstance(value, tuple):
            value = value[-1]
        if isinstance(value, str) and value != "":
            found[key] = value
    return found


def _upload_filename(filename: str) -> str:
    base = filename.replace("\\", "/").split("/")[-1]
    if not base or base in {".", ".."} or "\x00" in base or "/" in base:
        raise FleetError("uploaded workflow filename is invalid")
    if not base.lower().endswith(".json"):
        raise FleetError(
            f"uploaded workflow must be a .json file, got {base!r}. "
            "There is no baked default workflow."
        )
    return base


def _materialize_upload(upload: _Upload) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="comfyfleet-upload-"))
    path = directory / upload.filename
    try:
        path.write_bytes(upload.data)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return path


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise FleetError(f"expected true or false, got {value!r}")


def _require_method(method: str, allowed: str) -> None:
    if method != allowed:
        raise HTTPStatusError(405, "method not allowed")


def _json(status: int, payload: dict) -> Response:
    body = json.dumps(payload, indent=2).encode("utf-8") + b"\n"
    return Response(status, body, _JSON)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    raw = handler.headers.get("Content-Length")
    if raw is None or raw == "":
        return b""
    try:
        size = int(raw)
    except ValueError as exc:
        raise FleetError("Content-Length must be an integer") from exc
    if size < 0:
        raise FleetError("Content-Length must be >= 0")
    if size > MAX_BODY_BYTES:
        raise HTTPStatusError(413, "request body exceeds 32 MiB")
    data = handler.rfile.read(size)
    if len(data) != size:
        raise FleetError("request body ended before Content-Length")
    return data
