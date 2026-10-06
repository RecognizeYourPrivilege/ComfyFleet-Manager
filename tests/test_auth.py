"""Auth gate: startup, Bearer, session cookie, health, logout."""

import http.client
import io
import json
import os
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from comfyfleet.auth import (
    DEFAULT_LOGIN_FAIL_DELAY_S,
    PASSWORD_ENV,
    AuthError,
    LoginGuard,
    SessionStore,
    begin_http_request,
    end_http_request,
    grant_http_request,
    read_password,
    secrets_equal,
)
from comfyfleet.cli import main
from comfyfleet.control import authorize
from comfyfleet.errors import FleetError
from comfyfleet.http_api import ApiContext, HTTPStatusError, dispatch, make_server, serve
from comfyfleet.paths import FleetLayout


PASSWORD = "test-password-value"
WRONG = "not-the-password"


class StartupTests(unittest.TestCase):
    def test_missing_password_refuses_to_start(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PASSWORD_ENV, None)
            with self.assertRaises(FleetError) as ctx:
                read_password()
            self.assertIn("COMFYFLEET_PASSWORD", str(ctx.exception))
            self.assertIn("Refusing to start", str(ctx.exception))
            self.assertIn("no open-LAN fallback", str(ctx.exception))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(["ui", "--host", "127.0.0.1", "--port", "9100"])
            self.assertEqual(code, 1)
            self.assertIn("COMFYFLEET_PASSWORD", stderr.getvalue())
            self.assertIn("Refusing to start", stderr.getvalue())
            with self.assertRaises(FleetError):
                serve(host="127.0.0.1", port=9100)

    def test_empty_and_whitespace_passwords_refuse_to_start(self):
        for value in ("", "   ", "\t"):
            with mock.patch.dict(os.environ, {PASSWORD_ENV: value}):
                with self.assertRaises(FleetError) as ctx:
                    read_password()
                self.assertIn("non-empty", str(ctx.exception))
                self.assertNotIn(value.strip(), str(ctx.exception)) if value.strip() else None

    def test_constant_time_compare_does_not_accept_a_different_length(self):
        self.assertTrue(secrets_equal("abc", "abc"))
        self.assertFalse(secrets_equal("abc", "abcd"))
        self.assertFalse(secrets_equal(None, "abc"))
        self.assertFalse(secrets_equal("", ""))


class AuthorizeTests(unittest.TestCase):
    def test_cli_path_allows_known_actions(self):
        authorize("list")

    def test_http_request_fails_closed_until_granted(self):
        token = begin_http_request()
        try:
            with self.assertRaises(AuthError) as ctx:
                authorize("list")
            self.assertEqual(ctx.exception.message, "unauthorized")
            grant_http_request()
            authorize("create")
        finally:
            end_http_request(token)
        authorize("stop")

    def test_unknown_action_is_still_an_error(self):
        with self.assertRaises(FleetError):
            authorize("wipe-host")


class HttpAuthTests(unittest.TestCase):
    def setUp(self):
        self.layout = FleetLayout(Path("/tmp/comfyfleet-auth-not-used"))
        self.docker_calls = []

        class Docker:
            def status(self, _name):
                return None

            def running_names(self):
                return []

        self.ctx = ApiContext(
            layout=self.layout,
            docker=Docker(),
            detect_gpus=lambda: [],
            port_in_use=lambda _port: False,
            ui_dir=Path(__file__).resolve().parents[1] / "ui",
            password=PASSWORD,
            sessions=SessionStore(),
            login_guard=LoginGuard(fail_delay_s=0, max_failures=3, window_s=60),
        )
        self.httpd = make_server("127.0.0.1", 0, self.ctx)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def _request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        merged = dict(headers or {})
        payload = body if body is not None else b""
        if body is not None and "Content-Length" not in merged:
            merged["Content-Length"] = str(len(payload))
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            connection.request(method, path, body=payload if body is not None else None, headers=merged)
            response = connection.getresponse()
            raw = response.read()
            status = response.status
            header_pairs = response.getheaders()
            time.sleep(0.05)
        connection.close()
        self.stderr = stderr.getvalue()
        return status, raw, header_pairs

    def _login(self, password, extra_headers=None):
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return self._request(
            "POST",
            "/api/login",
            body=json.dumps({"password": password}).encode("utf-8"),
            headers=headers,
        )

    def test_wrong_password_is_401_and_is_not_logged(self):
        status, raw, headers = self._login(WRONG)
        self.assertEqual(status, 401)
        payload = json.loads(raw.decode("utf-8"))
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "invalid credentials")
        self.assertNotIn(WRONG, raw.decode("utf-8"))
        self.assertNotIn(PASSWORD, raw.decode("utf-8"))
        self.assertNotIn(WRONG, self.stderr)
        self.assertNotIn(PASSWORD, self.stderr)
        self.assertNotIn("Set-Cookie", {key for key, _value in headers})

    def test_missing_credential_is_401_and_bearer_lists_instances(self):
        status, raw, _headers = self._request("GET", "/api/instances")
        self.assertEqual(status, 401)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["error"], "unauthorized")
        self.assertNotIn(b"portrait", raw)

        status, raw, _headers = self._request(
            "GET",
            "/api/instances",
            headers={"Authorization": f"Bearer {WRONG}"},
        )
        self.assertEqual(status, 401)

        status, raw, _headers = self._request(
            "GET",
            "/api/gpus",
            headers={"Authorization": "Bearer " + PASSWORD},
        )
        self.assertIn(status, (200, 503))
        if status == 503:
            self.assertNotIn(PASSWORD, raw.decode("utf-8"))

        status, raw, _headers = self._request(
            "GET",
            "/api/instances",
            headers={"Authorization": f"Bearer {PASSWORD}"},
        )
        self.assertEqual(status, 200, raw)
        payload = json.loads(raw.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["instances"], [])

    def test_cookie_session_and_logout(self):
        status, raw, headers = self._login(PASSWORD)
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw.decode("utf-8")), {"ok": True})
        cookies = [value for key, value in headers if key.lower() == "set-cookie"]
        self.assertEqual(len(cookies), 1)
        cookie = cookies[0]
        self.assertIn("comfyfleet_session=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertNotIn("Secure", cookie)
        self.assertNotIn(PASSWORD, cookie)
        pair = cookie.split(";", 1)[0]

        status, raw, _headers = self._request("GET", "/api/instances", headers={"Cookie": pair})
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw.decode("utf-8"))["instances"], [])

        status, _raw, secure_headers = self._login(
            PASSWORD,
            extra_headers={"X-Forwarded-Proto": "https"},
        )
        self.assertEqual(status, 200)
        secure = [value for key, value in secure_headers if key.lower() == "set-cookie"][0]
        self.assertIn("Secure", secure)

        anon, _body, location = self._request("GET", "/")
        self.assertEqual(anon, 302)
        locations = [value for key, value in location if key.lower() == "location"]
        self.assertEqual(locations, ["/login"])

        fleet, page, _headers = self._request("GET", "/", headers={"Cookie": pair})
        self.assertEqual(fleet, 200)
        self.assertIn(b"New instance", page)

        status, raw, cleared = self._request("POST", "/api/logout", headers={"Cookie": pair})
        self.assertEqual(status, 200, raw)
        self.assertTrue(json.loads(raw.decode("utf-8"))["ok"])
        clear = [value for key, value in cleared if key.lower() == "set-cookie"][0]
        self.assertIn("Max-Age=0", clear)

        status, raw, _headers = self._request("GET", "/api/instances", headers={"Cookie": pair})
        self.assertEqual(status, 401, raw)
        self.assertEqual(json.loads(raw.decode("utf-8"))["error"], "session expired")

        status, raw, _headers = self._request(
            "GET",
            "/api/instances",
            headers={"Authorization": f"Bearer {PASSWORD}"},
        )
        self.assertEqual(status, 200, raw)

    def test_health_without_auth(self):
        status, raw, _headers = self._request("GET", "/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(raw.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["auth"], "required")
        text = raw.decode("utf-8")
        self.assertNotIn(PASSWORD, text)
        self.assertNotIn("instances", payload)

    def test_login_page_is_public_and_shell_redirects(self):
        status, raw, _headers = self._request("GET", "/login")
        self.assertEqual(status, 200)
        self.assertIn(b"password", raw.lower())
        self.assertNotIn(PASSWORD.encode(), raw)
        status, raw, _headers = self._request("GET", "/app.css")
        self.assertEqual(status, 200)
        self.assertIn(b"backdrop-filter", raw)

    def test_login_rate_limit(self):
        for _ in range(3):
            status, _raw, _headers = self._login(WRONG)
            self.assertEqual(status, 401)
        status, raw, headers = self._login(PASSWORD)
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(raw.decode("utf-8"))["error"], "too many login attempts")
        self.assertNotIn(PASSWORD, raw.decode("utf-8"))
        self.assertFalse(any(key.lower() == "set-cookie" for key, _value in headers))

    def test_dispatch_does_not_echo_the_password(self):
        with self.assertRaises(HTTPStatusError) as ctx:
            dispatch(
                self.ctx,
                "POST",
                "/api/login",
                "127.0.0.1:9100",
                json.dumps({"password": WRONG}).encode("utf-8"),
                "application/json",
                client_ip="203.0.113.10",
            )
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.message, "invalid credentials")
        self.assertNotIn(WRONG, ctx.exception.message)
        self.assertNotIn(PASSWORD, ctx.exception.message)

    def test_default_login_delay_is_set(self):
        self.assertEqual(DEFAULT_LOGIN_FAIL_DELAY_S, 0.25)


if __name__ == "__main__":
    unittest.main()
