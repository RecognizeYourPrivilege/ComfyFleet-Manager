"""Hostname used in instance Open URLs.

``COMFYFLEET_PUBLIC_HOST`` wins when it is set to a hostname or IP. Otherwise
the request ``Host`` header is used when that value is safe to put in a link.
"""

from __future__ import annotations

import os

from comfyfleet.errors import FleetError

PUBLIC_HOST_ENV = "COMFYFLEET_PUBLIC_HOST"
_BIND_ANY = {"0.0.0.0", "::", "[::]", "*"}


def request_host(host_header: str | None, fallback: str = "127.0.0.1") -> str:
    """Hostname the caller used, without the control port.

    Browsers send ``Host``. The instance open URL uses that name so a phone
    that loaded ``http://lan-ip:9100/`` opens Comfy at ``http://lan-ip:<port>``.
    """

    if host_header is None or not host_header.strip():
        return fallback
    host = host_header.strip()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[: end + 1]
    if host.count(":") == 1:
        name, port = host.rsplit(":", 1)
        if port.isdigit() and name:
            return name
    return host


def configured_public_host() -> str | None:
    """Return ``COMFYFLEET_PUBLIC_HOST`` when set, or None when unset/blank.

    An invalid value raises ``FleetError`` so the manager fails at startup
    instead of emitting Open links the browser cannot use.
    """

    raw = os.environ.get(PUBLIC_HOST_ENV)
    if raw is None or not raw.strip():
        return None
    return require_public_host(raw)


def open_host(
    host_header: str | None,
    fallback: str = "127.0.0.1",
    public_host: str | None = None,
) -> str:
    """Host for an instance Open URL.

    ``public_host`` (from ``COMFYFLEET_PUBLIC_HOST``) replaces the request
    header. When it is omitted, the header is used only if it is a safe
    hostname or IP. Bind-all addresses and values with spaces or slashes
    fall through to ``fallback``.
    """

    if public_host is not None and public_host.strip():
        return require_public_host(public_host)
    candidate = request_host(host_header, "")
    safe = validated_host(candidate)
    if safe is None:
        return fallback
    return safe


def require_public_host(raw: str) -> str:
    host = request_host(raw.strip(), "")
    # 0.0.0.0 is a valid configured host: list and CLI URLs use it as written.
    # A request Host of 0.0.0.0 is still not used for the web UI Open link.
    if host == "0.0.0.0":
        return host
    safe = validated_host(host)
    if safe is None:
        raise FleetError(
            f"{PUBLIC_HOST_ENV} is not a usable hostname or IP ({raw.strip()!r}). "
            "Set it to 0.0.0.0 or the machine's LAN IP."
        )
    return safe


def validated_host(value: str) -> str | None:
    """A hostname or IP that can appear in an ``http://`` Open link."""

    text = value.strip()
    if not text or text in _BIND_ANY or len(text) > 253:
        return None
    if any(ord(char) < 33 or ord(char) == 127 for char in text):
        return None
    if any(char in text for char in "/\\@?#\"'"):
        return None
    return text
