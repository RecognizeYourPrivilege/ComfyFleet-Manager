#!/bin/bash
# ComfyFleet manager entrypoint.
# Starts the control HTTP server and web UI unless a comfyfleet subcommand is given.
# Instance containers are siblings on the host engine (mounted Docker socket).
set -euo pipefail

export COMFYFLEET_MANAGER=1

# Refuse to start with no shared secret. Do not print the value.
if [[ -z "${COMFYFLEET_PASSWORD:-}" || -z "${COMFYFLEET_PASSWORD//[[:space:]]/}" ]]; then
  echo "comfyfleet: COMFYFLEET_PASSWORD is required and must be non-empty." >&2
  echo "comfyfleet: refusing to start. Set -e COMFYFLEET_PASSWORD=... or the compose environment." >&2
  echo "comfyfleet: there is no open-LAN fallback." >&2
  exit 1
fi

if [[ -n "${DOCKER_HOST:-}" && "${DOCKER_HOST}" != unix://* ]]; then
  echo "comfyfleet: DOCKER_HOST=${DOCKER_HOST}. The client will use that endpoint instead of /var/run/docker.sock." >&2
else
  SOCKET="/var/run/docker.sock"
  if [[ "${DOCKER_HOST:-}" == unix://* ]]; then
    SOCKET="${DOCKER_HOST#unix://}"
    SOCKET="${SOCKET:-/var/run/docker.sock}"
  fi
  if [[ ! -S "${SOCKET}" ]]; then
    echo "comfyfleet: Docker socket ${SOCKET} is missing." >&2
    echo "comfyfleet: mount the host engine with -v /var/run/docker.sock:/var/run/docker.sock." >&2
    echo "comfyfleet: create, start, and stop need that socket. This image does not run a Docker daemon." >&2
  elif [[ ! -r "${SOCKET}" || ! -w "${SOCKET}" ]]; then
    echo "comfyfleet: permission denied on ${SOCKET}." >&2
    echo "comfyfleet: run the manager as root, or as a uid in the host docker group." >&2
    echo "comfyfleet: this socket is root-equivalent on the host. Trusted host only." >&2
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "comfyfleet: nvidia-smi is not available in the manager." >&2
  echo "comfyfleet: GPU probe runs nvidia-smi here so it lists host GPUs." >&2
  echo "comfyfleet: start the manager with --gpus all (the NVIDIA Container Toolkit injects the host binary and driver)." >&2
  echo "comfyfleet: this image does not ship a CUDA stack. Create and start fail until the probe succeeds." >&2
fi

if ! awk '$2 == "/home/ComfyFleet" { found = 1 } END { exit found ? 0 : 1 }' /proc/mounts; then
  echo "comfyfleet: /home/ComfyFleet is not a bind mount." >&2
  echo "comfyfleet: mount the host storage root with -v /home/ComfyFleet:/home/ComfyFleet so instance files, models, and workflow JSON are on the host." >&2
  echo "comfyfleet: sibling containers receive those same host paths. A /home/ComfyFleet that exists only inside the manager is not visible to them." >&2
fi

echo "comfyfleet: Open Comfy uses the browser host and the instance port. COMFYFLEET_PUBLIC_HOST is not used for that link." >&2

UI_DIR="${COMFYFLEET_UI_DIR:-/opt/comfyfleet/ui}"
BIND_HOST="${COMFYFLEET_BIND_HOST:-0.0.0.0}"
BIND_PORT="${COMFYFLEET_BIND_PORT:-9100}"

if [[ $# -eq 0 ]]; then
  exec comfyfleet ui --host "${BIND_HOST}" --port "${BIND_PORT}" --ui-dir "${UI_DIR}"
fi

exec comfyfleet "$@"
