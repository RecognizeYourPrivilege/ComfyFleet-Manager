"""The control pages are a client of the HTTP API.

They are served by ``comfyfleet ui`` from ``ui/``. These tests do not
reimplement create, start, stop, or Docker.
"""

import json
import threading
import unittest
from pathlib import Path

from comfyfleet.auth import LoginGuard, SessionStore
from comfyfleet.http_api import ApiContext, make_server
from comfyfleet.paths import FleetLayout


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
PASSWORD = "test-password"


class IdleDocker:
    def __init__(self):
        self.calls = []

    def create(self, args):
        self.calls.append(("create", list(args)))

    def start(self, name):
        self.calls.append(("start", name))

    def stop(self, name):
        self.calls.append(("stop", name))

    def remove(self, name):
        self.calls.append(("rm", name))

    def update_restart(self, name, policy):
        self.calls.append(("update-restart", name, policy))

    def status(self, name):
        return None

    def running_names(self):
        return []


class UiContractTests(unittest.TestCase):
    def test_pages_call_the_published_api_only(self):
        html = (UI / "index.html").read_text(encoding="utf-8")
        login = (UI / "login.html").read_text(encoding="utf-8")
        script = (UI / "app.js").read_text(encoding="utf-8")
        css = (UI / "app.css").read_text(encoding="utf-8")
        self.assertIn("comfyfleet-logo-ships.jpg", html)
        self.assertIn("New instance", html)
        self.assertIn("Create &amp; start", html)
        self.assertIn("Trusted LAN", html)
        self.assertIn("Log out", html)
        self.assertIn('id="host-menu"', html)
        self.assertIn("Fix ownership", html)
        self.assertIn("Prune dangling containers", html)
        self.assertLess(html.index('id="host-menu"'), html.index('id="list"'))
        self.assertIn("/api/host/fix-owner", script)
        self.assertIn('id="confirm-user"', html)
        self.assertIn('id="confirm-group"', html)
        self.assertIn('id="confirm-fields" hidden', html)
        self.assertIn('placeholder="comfyuser"', html)
        self.assertIn("fields: true", script)
        self.assertIn("answer.user", script)
        self.assertIn("answer.group", script)
        self.assertIn("comfyuser", script)
        self.assertNotIn("comfyui:comfyui", script)
        self.assertIn("/home/ComfyFleet/files", script)
        self.assertIn("/home/ComfyFleet/custom_nodes_*", script)
        self.assertNotIn("/home/files", script)
        self.assertNotIn("/home/models", script)
        self.assertNotIn("/home/wildcards", script)
        self.assertIn("/api/host/prune-dangling", script)
        self.assertIn("comfyfleet.managed=true", script)
        self.assertIn("askConfirm", script)
        self.assertNotIn("not included", html.lower())
        self.assertIn("/api/login", login)
        self.assertIn('id="password"', login)
        self.assertIn('type="text"', login)
        self.assertIn('autocomplete="off"', login)
        self.assertNotIn('type="password"', login)
        self.assertNotIn("current-password", login)
        self.assertIn("is-revealed", login)
        self.assertIn("-webkit-text-security: disc", css)
        self.assertIn(".login-field input.is-revealed", css)
        self.assertIn('credentials: "same-origin"', login)
        self.assertIn("Invalid credentials.", login)
        self.assertIn("Session expired. Sign in again.", login)
        self.assertIn("Signing in…", login)
        self.assertIn("login-hero", login)
        self.assertIn("login-notice", login)
        self.assertIn("ComfyUI ports are not covered", login)
        self.assertIn("backdrop-filter", css)
        top_rule = css.split(".top {", 1)[1].split("}", 1)[0]
        self.assertIn("z-index: 30", top_rule)
        self.assertIn("--hit: 50px", css)
        self.assertIn(".login-field input:focus", css)
        self.assertIn("safe-area-inset-top", css)
        self.assertIn(".login-notice", css)
        self.assertNotIn("password", script)
        self.assertIn('instance.status === "running"', script)
        self.assertIn("/api/health", script)
        self.assertIn("/api/gpus", script)
        self.assertIn("/api/instances", script)
        self.assertIn('"/api/instances"', script)
        self.assertIn('body.append("workflow"', script)
        self.assertIn("workflow_path", script)
        self.assertIn('id="instance-name"', html)
        self.assertIn("Container name", html)
        self.assertIn("From the workflow file", html)
        self.assertIn('body.append("name"', script)
        self.assertIn('body.append("gpus"', script)
        self.assertIn('body.append("cuda_tag"', script)
        self.assertIn('name="cuda-tag"', html)
        self.assertIn('value="cu130"', html)
        self.assertIn('value="cu124"', html)
        self.assertIn("CUDA 13.0", html)
        self.assertIn("CUDA 12.4", html)
        self.assertIn("requires a recreate", html)
        self.assertIn("CUDA ${cudaText}", script)
        self.assertIn('body.append("vram"', script)
        self.assertIn('body.append("attention"', script)
        self.assertIn('body.append("flags"', script)
        self.assertIn('body.append("extra_args"', script)
        self.assertIn('name="vram"', html)
        self.assertIn('name="attention"', html)
        self.assertIn('value="--lowvram"', html)
        self.assertIn('value="--novram"', html)
        self.assertIn('value="--highvram"', html)
        self.assertIn('value="--use-pytorch-cross-attention"', html)
        self.assertIn('value="--disable-smart-memory"', html)
        self.assertIn('value="--disable-dynamic-vram"', html)
        self.assertIn('value="--cuda-malloc"', html)
        self.assertIn('value="--disable-xformers"', html)
        self.assertIn('value="--force-fp16"', html)
        self.assertIn('value="--fp8_e4m3fn-unet"', html)
        self.assertIn("flag-panel", html)
        self.assertIn("applied-flags", html)
        self.assertIn('id="comfy-flags"', html)
        self.assertIn("Advanced / ComfyUI flags", html)
        flags_at = html.index('id="comfy-flags"')
        flags_tag = html[html.rfind("<details", 0, flags_at):html.index(">", flags_at)]
        self.assertNotIn("open", flags_tag)
        self.assertIn('id="custom-node-git-urls"', html)
        self.assertIn('id="custom-nodes-zip"', html)
        self.assertIn('id="install-missing-from-workflow"', html)
        self.assertIn("checked", html[html.index('id="install-missing-from-workflow"'):html.index('id="install-missing-from-workflow"') + 80])
        self.assertIn("leave blank to skip", html.lower())
        self.assertIn("trash-btn", css)
        self.assertIn(".icon-actions", css)
        self.assertIn(".flag-editor[hidden]", css)
        self.assertIn(".action-icon.on", css)
        self.assertIn(".tone-green", css)
        self.assertIn(".tone-blue", css)
        self.assertIn(".action-glyph", css)
        self.assertIn(".flag-apply", css)
        self.assertIn(".flag-editor-body", css)
        self.assertNotIn("position: sticky", css)
        self.assertIn("flag-apply", script)
        self.assertIn("flag-editor-body", script)
        self.assertIn("listSignature", script)
        self.assertIn("window.scrollTo", script)
        self.assertIn("Reserve VRAM (GB)", script)
        self.assertIn("VRAM headroom (GB)", script)
        self.assertIn("--reserve-vram", script)
        self.assertIn("--vram-headroom", script)
        self.assertIn('className: "icon-actions"', script)
        self.assertIn("action-glyph", script)
        self.assertIn("tone-${tone}", script)
        self.assertIn('playIcon(), "green"', script)
        self.assertIn('forceStopIcon(), "orange"', script)
        self.assertIn('trashIcon(), "red"', script)
        self.assertNotIn("Copy URL", script)
        self.assertIn("Edit flags", script)
        self.assertIn("editor.hidden", script)
        self.assertIn("aria-expanded", script)
        self.assertNotIn('text: "Delete"', script)
        self.assertIn('body.append("custom_node_git_urls"', script)
        self.assertIn('id="custom-nodes-zip"', html)
        self.assertIn("multiple", html[html.index('id="custom-nodes-zip"') - 80:html.index('id="custom-nodes-zip"') + 80])
        self.assertIn("pyproject.toml", html)
        self.assertIn('body.append("custom_nodes_zip"', script)
        self.assertIn('body.append("custom_nodes_zip_name"', script)
        self.assertIn("Name from the zip", script)
        self.assertIn('body.append("install_missing_from_workflow"', script)
        self.assertIn('body.append("comfy_extra_args"', script)
        self.assertIn("payload.warnings", script)
        self.assertIn("flag-catalog", script)
        self.assertIn("data-section=\"caching\"", html)
        self.assertIn("data-section=\"precision\"", html)
        self.assertIn("--cache-none", html)
        self.assertIn("Remove ", script)
        self.assertIn("function addFlag", script)
        self.assertIn('text: "Apply"', script)
        self.assertIn("/launch", script)
        self.assertIn("same name, port, mounts, and workflow", script)
        self.assertIn("12GB", html)
        self.assertIn("RTX A2000", html)
        self.assertIn("--lowvram alone is a no-op", html)
        self.assertIn('value="--disable-dynamic-vram"', html)
        self.assertIn("Force stop", script)
        self.assertIn("force-stop", script)
        self.assertIn("Open Comfy", script)
        self.assertIn('id="tab-gallery"', html)
        self.assertIn("Open in ComfyUI", html)
        self.assertIn('id="lb-download"', html)
        self.assertIn("playsinline", html)
        self.assertIn("/api/gallery", script)
        self.assertIn("/api/gallery/delete", script)
        self.assertIn("Import container", html)
        self.assertIn('id="import-overlay"', html)
        self.assertIn('id="import-pill"', html)
        self.assertIn('id="import-dupes"', html)
        self.assertIn('aria-readonly="true"', html)
        self.assertIn("resumeImportOverlay", script)
        self.assertIn("/api/import/active", script)
        self.assertIn("/api/import/duplicates", script)
        self.assertIn("openImportOverlay", script)
        sheet_rule = css.split(".sheet {", 1)[1].split("}", 1)[0]
        self.assertIn("align-items: center", sheet_rule)
        self.assertIn("z-index: 90", sheet_rule)
        self.assertNotIn("flex-end", sheet_rule)
        self.assertIn("Delete file", script)
        self.assertIn("output folder on disk", script)
        self.assertIn("ArrowLeft", script)
        self.assertIn("ArrowRight", script)
        self.assertIn("touchstart", script)
        self.assertIn('loading = "lazy"', script)
        self.assertIn("gallery-tile", script)
        self.assertIn("location.hostname", script)
        self.assertIn("location.protocol", script)
        self.assertIn("instance.port", script)
        self.assertNotIn("instance.url", script)
        self.assertNotIn("127.0.0.1", script)
        self.assertNotIn("COMFYFLEET_PUBLIC_HOST", script)
        self.assertIn("terminal.html?name=", script)
        self.assertIn("Delete", script)
        self.assertIn("confirmDelete", script)
        self.assertIn("only that instance container", script)
        self.assertNotIn("docker.sock", script)
        self.assertNotIn("docker exec", script)
        terminal_js = (UI / "terminal.js").read_text(encoding="utf-8")
        self.assertIn("/api/instances/", terminal_js)
        self.assertIn("/terminal", terminal_js)
        self.assertNotIn("docker.sock", terminal_js)
        self.assertNotIn("docker exec", terminal_js)
        self.assertIn("submitCreate(false)", script)
        self.assertIn("submitCreate(true)", script)
        self.assertIn('credentials: "same-origin"', script)
        self.assertIn("/api/logout", script)
        self.assertIn("session expired", script)
        lowered = script.lower()
        for banned in (
            "docker create",
            "docker start",
            "subprocess",
            "nvidia-smi",
            "/opt/comfyui",
            "build_create_args",
        ):
            self.assertNotIn(banned, lowered)
        self.assertNotIn("http://", script)
        self.assertTrue((UI / "comfyfleet-logo-ships.jpg").is_file())

    def test_control_server_serves_the_ui(self):
        docker = IdleDocker()
        context = ApiContext(
            layout=FleetLayout(ROOT / "does-not-need-to-exist"),
            docker=docker,
            detect_gpus=lambda: [],
            port_in_use=lambda _port: False,
            ui_dir=UI,
            password=PASSWORD,
            sessions=SessionStore(),
            login_guard=LoginGuard(fail_delay_s=0),
        )
        httpd = make_server("127.0.0.1", 0, context)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        port = httpd.server_address[1]
        try:
            anon = _get(port, "/")
            self.assertEqual(anon.status, 302)
            self.assertIn("/login", anon.getheader("Location") or "")
            anon.read()
            login = _get(port, "/login")
            self.assertEqual(login.status, 200)
            login_body = login.read()
            self.assertIn(b"ComfyFleet", login_body)
            self.assertIn(b"password", login_body)
            logo = _get(port, "/comfyfleet-logo-ships.jpg")
            self.assertEqual(logo.status, 200)
            self.assertTrue(logo.read().startswith(b"\xff\xd8"))
            cookie = _login(port)
            page = _get(port, "/", cookie=cookie)
            self.assertEqual(page.status, 200)
            body = page.read()
            self.assertIn(b"ComfyFleet", body)
            self.assertIn(b"/comfyfleet-logo-ships.jpg", body)
            self.assertIn(b"New instance", body)
            self.assertNotIn(b"not included", body.lower())
            script = _get(port, "/app.js", cookie=cookie)
            self.assertEqual(script.status, 200)
            self.assertIn(b"/api/instances", script.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
        self.assertEqual(docker.calls, [])


def _get(port: int, path: str, cookie: str | None = None):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Cookie": cookie} if cookie else {}
    connection.request("GET", path, headers=headers)
    return connection.getresponse()


def _login(port: int) -> str:
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps({"password": PASSWORD}).encode("utf-8")
    connection.request(
        "POST",
        "/api/login",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = connection.getresponse()
    response.read()
    set_cookie = response.getheader("Set-Cookie") or ""
    connection.close()
    return set_cookie.split(";", 1)[0]


if __name__ == "__main__":
    unittest.main()
