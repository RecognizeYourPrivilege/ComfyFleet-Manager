"""Shared-password gate for the manager HTTP API.

One operator secret, ``COMFYFLEET_PASSWORD``. The control server refuses to
start when it is missing or empty. Protected routes accept either:

- a session cookie issued by ``POST /api/login`` (in-memory, dies on restart)
- ``Authorization: Bearer <COMFYFLEET_PASSWORD>``

A local CLI call is not an HTTP request. ``authorize()`` allows that path.
The HTTP server sets a request scope and ``authorize()`` fails closed until
the request has presented a valid cookie or Bearer token.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import os
import secrets
import threading
import time

PASSWORD_ENV = "COMFYFLEET_PASSWORD"
COOKIE_NAME = "comfyfleet_session"
SESSION_TTL_S = 12 * 60 * 60
DEFAULT_LOGIN_FAIL_DELAY_S = 0.25
LOGIN_MAX_FAILURES = 8
LOGIN_WINDOW_S = 60.0

MISSING_PASSWORD_MESSAGE = (
    "COMFYFLEET_PASSWORD is required and must be non-empty. "
    "Refusing to start. Set it with docker run -e COMFYFLEET_PASSWORD=... "
    "or the compose environment. There is no open-LAN fallback."
)

_HTTP_AUTH: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "comfyfleet_http_auth",
    default=None,
)
_DUMMY_DIGEST = hashlib.sha256(b"comfyfleet-unset-password").digest()


class AuthError(Exception):
    """An HTTP request is not allowed to touch the fleet.

    This is not a ``FleetError``. The HTTP adapter maps it to 401.
    The message is ``unauthorized`` or ``session expired`` and never
    includes the password.
    """

    def __init__(self, message: str = "unauthorized"):
        super().__init__(message)
        self.message = message


def read_password(environ: dict[str, str] | None = None) -> str:
    """Return the configured password or raise ``FleetError``.

    Unset, ``""``, and whitespace-only values all refuse startup.
    The returned value is the raw environment string when it contains
    any non-whitespace character (leading or trailing spaces are kept).
    """

    from comfyfleet.errors import FleetError

    env = os.environ if environ is None else environ
    if PASSWORD_ENV not in env:
        raise FleetError(MISSING_PASSWORD_MESSAGE)
    value = env[PASSWORD_ENV]
    if value is None or value == "" or str(value).strip() == "":
        raise FleetError(MISSING_PASSWORD_MESSAGE)
    return str(value)


def login_fail_delay(environ: dict[str, str] | None = None) -> float:
    """Seconds to wait after a failed login. Default is a short delay."""

    from comfyfleet.errors import FleetError

    env = os.environ if environ is None else environ
    raw = env.get("COMFYFLEET_LOGIN_FAIL_DELAY", "").strip()
    if not raw:
        return DEFAULT_LOGIN_FAIL_DELAY_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise FleetError("COMFYFLEET_LOGIN_FAIL_DELAY must be a number of seconds") from exc
    if value < 0:
        raise FleetError("COMFYFLEET_LOGIN_FAIL_DELAY must be >= 0")
    return value


def secrets_equal(expected: str | None, presented: str) -> bool:
    """Best-effort constant-time compare. Does not log either value."""

    presented_digest = hashlib.sha256(presented.encode("utf-8")).digest()
    if not expected:
        hmac.compare_digest(presented_digest, _DUMMY_DIGEST)
        return False
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(presented_digest, expected_digest)


def begin_http_request() -> contextvars.Token:
    """Mark this thread as an HTTP request that is not yet authenticated."""

    return _HTTP_AUTH.set(False)


def grant_http_request() -> None:
    """Allow ``authorize()`` for the rest of this HTTP request."""

    _HTTP_AUTH.set(True)


def end_http_request(token: contextvars.Token) -> None:
    _HTTP_AUTH.reset(token)


def http_auth_state() -> bool | None:
    """``None`` outside HTTP, ``False`` until granted, ``True`` after."""

    return _HTTP_AUTH.get()


def bearer_token(header: str | None) -> str | None:
    """Return the token from ``Authorization: Bearer <token>``."""

    if header is None:
        return None
    text = header.strip()
    if not text:
        return None
    scheme, sep, rest = text.partition(" ")
    if sep != " " or scheme.lower() != "bearer":
        return None
    token = rest.strip()
    return token or None


def read_cookie(header: str | None, name: str = COOKIE_NAME) -> str | None:
    if not header:
        return None
    for part in header.split(";"):
        piece = part.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        if key.strip() == name:
            token = value.strip()
            return token or None
    return None


def request_is_https(forwarded_proto: str | None) -> bool:
    """True when a reverse proxy says this request arrived over HTTPS."""

    if not forwarded_proto:
        return False
    first = forwarded_proto.split(",", 1)[0].strip().lower()
    return first == "https"


def session_cookie(value: str, *, secure: bool, clear: bool = False) -> str:
    """HttpOnly session cookie. ``Secure`` only when the request is HTTPS."""

    shown = "" if clear else value
    parts = [f"{COOKIE_NAME}={shown}", "HttpOnly", "Path=/", "SameSite=Lax"]
    if clear:
        parts.append("Max-Age=0")
    else:
        parts.append(f"Max-Age={int(SESSION_TTL_S)}")
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


class SessionStore:
    """In-memory session ids. They die when this process exits."""

    def __init__(self, ttl_s: float = SESSION_TTL_S):
        self.ttl_s = ttl_s
        self._items: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._items[token] = time.monotonic() + self.ttl_s
        return token

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        now = time.monotonic()
        with self._lock:
            expiry = self._items.get(token)
            if expiry is None:
                return False
            if expiry <= now:
                self._items.pop(token, None)
                return False
            return True

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._items.pop(token, None)


class LoginGuard:
    """Small in-memory limit plus a delay on failed ``POST /api/login``."""

    def __init__(
        self,
        fail_delay_s: float = DEFAULT_LOGIN_FAIL_DELAY_S,
        max_failures: int = LOGIN_MAX_FAILURES,
        window_s: float = LOGIN_WINDOW_S,
    ):
        self.fail_delay_s = fail_delay_s
        self.max_failures = max_failures
        self.window_s = window_s
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key, time.monotonic())) >= self.max_failures

    def pause(self) -> None:
        if self.fail_delay_s > 0:
            time.sleep(self.fail_delay_s)

    def record_failure(self, key: str) -> None:
        self.pause()
        with self._lock:
            items = self._prune(key, time.monotonic())
            items.append(time.monotonic())
            self._fails[key] = items

    def record_success(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)

    def _prune(self, key: str, now: float) -> list[float]:
        items = [stamp for stamp in self._fails.get(key, []) if now - stamp < self.window_s]
        if items:
            self._fails[key] = items
        else:
            self._fails.pop(key, None)
        return items
