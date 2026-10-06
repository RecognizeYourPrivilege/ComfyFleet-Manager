#!/usr/bin/env bash
# Pull the ComfyFleet manager image and start it.
# Instance images are not pulled unless --with-images is set.
# The manager pulls a missing instance image when an instance is created.
# Re-running replaces the comfyfleet-manager container and keeps /home/ComfyFleet.
set -euo pipefail

MANAGER_REPO="ghcr.io/recognizeyourprivilege/comfyfleet-manager"
INSTANCE_REPO="ghcr.io/recognizeyourprivilege/comfyfleet-images"
NAME="${COMFYFLEET_CONTAINER_NAME:-comfyfleet-manager}"
PORT="${COMFYFLEET_PUBLISH_PORT:-9100}"
MODE="run"
PULL_ONLY=0
CUDA_TAG=""
CUDA_TAG_EXPLICIT=0
WITH_IMAGES=""

usage() {
  cat <<'EOF'
Usage: install.sh [--cuda-tag cu130|cu124] [--with-images cu130|cu124|both] [--compose] [--pull-only] [--public-host HOST] [--password PASS] [--port PORT] [--name NAME]

Pull the ComfyFleet manager and start it. Does not pull an instance image
unless --with-images is set. The manager pulls a missing instance image
when an instance is created.

  COMFYFLEET_PASSWORD       required unless --pull-only. Prefer the environment
                            (a --password argument is visible in the process list).
  COMFYFLEET_PUBLIC_HOST    host written into comfyfleet list URLs and the URL
                            the CLI prints after create or start. Set it to
                            0.0.0.0 or the machine's LAN IP. Required unless
                            --pull-only. The web UI Open link uses the browser
                            host, not this variable.
  COMFYFLEET_CUDA_TAG       cu130 or cu124. Same as --cuda-tag. The flag wins.
  COMFYFLEET_INSTANCE_IMAGE full instance ref. Ignored when --cuda-tag or
                            COMFYFLEET_CUDA_TAG sets the line. Not pulled
                            unless --with-images asks for that tag.
  COMFYFLEET_INSTANCE_DIGEST / COMFYFLEET_MANAGER_DIGEST
                            sha256 pin for that image. Unset, install follows
                            the moving tags below.

Default manager image (always pulled):
  ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest

Instance image the manager is pointed at (not pulled by default):
  ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130
  ghcr.io/recognizeyourprivilege/comfyfleet-images:cu124   with --cuda-tag cu124

  --cuda-tag      cu130 (host driver CUDA 13.0, default) or cu124 (CUDA 12.4).
                  Points the manager at that comfyfleet-images tag.
                  Does not pull the instance image.
  --with-images   cu130, cu124, or both. Pulls only those instance tags.
  --pull-only     pull the manager (and any --with-images tags) and do not start it
  --compose       docker compose up -d instead of docker run
  --port          host port published to container 9100 (docker run only; default 9100)
  --name          manager container name (default comfyfleet-manager)

Re-running replaces the manager container. The same name, port 9100, and
/home/ComfyFleet mount are kept. The data directory is not deleted.

Examples:
  COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> ./install.sh
  COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> ./install.sh --cuda-tag cu124
  COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> ./install.sh --with-images cu130
  curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh \
    | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash
  curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh \
    | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash -s -- --with-images both
EOF
}

die() {
  echo "comfyfleet: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-tag)
      [[ $# -ge 2 ]] || die "--cuda-tag needs cu130 or cu124."
      if [[ "$2" == "both" ]]; then
        die "--cuda-tag does not take both. Pass --with-images both to pull both instance images."
      fi
      CUDA_TAG=$2
      CUDA_TAG_EXPLICIT=1
      shift 2
      ;;
    --with-images)
      [[ $# -ge 2 ]] || die "--with-images needs cu130, cu124, or both."
      WITH_IMAGES=$2
      shift 2
      ;;
    --compose)
      MODE="compose"
      shift
      ;;
    --pull-only)
      PULL_ONLY=1
      shift
      ;;
    --public-host)
      [[ $# -ge 2 ]] || die "--public-host needs a value."
      COMFYFLEET_PUBLIC_HOST=$2
      shift 2
      ;;
    --password)
      [[ $# -ge 2 ]] || die "--password needs a value."
      COMFYFLEET_PASSWORD=$2
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port needs a value."
      PORT=$2
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || die "--name needs a value."
      NAME=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -n "${WITH_IMAGES}" && "${WITH_IMAGES}" != "cu130" && "${WITH_IMAGES}" != "cu124" && "${WITH_IMAGES}" != "both" ]]; then
  die "--with-images must be cu130, cu124, or both, got ${WITH_IMAGES}."
fi

if [[ "${MODE}" == "compose" && "${PORT}" != "9100" ]]; then
  die "compose.yaml publishes 9100:9100. Omit --port or use docker run."
fi
if [[ "${MODE}" == "compose" && "${NAME}" != "comfyfleet-manager" ]]; then
  die "compose uses container name comfyfleet-manager. Omit --name or use docker run."
fi

digest_ref() {
  local repo="$1"
  local digest="$2"
  if [[ "${digest}" != sha256:* ]]; then
    digest="sha256:${digest}"
  fi
  printf '%s@%s\n' "${repo}" "${digest}"
}

image_line() {
  local ref="$1"
  if [[ -z "${ref}" ]]; then
    printf '%s\n' ""
    return
  fi
  case "${ref}" in
    *:cu124|*:cu124@*) printf '%s\n' "cu124" ;;
    *:cu130|*:cu130@*) printf '%s\n' "cu130" ;;
    *) printf '%s\n' "" ;;
  esac
}

choose_cuda_tag() {
  if [[ -z "${CUDA_TAG}" && -n "${COMFYFLEET_CUDA_TAG:-}" ]]; then
    CUDA_TAG="${COMFYFLEET_CUDA_TAG}"
    CUDA_TAG_EXPLICIT=1
  fi
  if [[ -z "${CUDA_TAG}" ]]; then
    if [[ -t 0 ]]; then
      echo "Instance CUDA line (match the host NVIDIA driver major):"
      echo "  cu130  host driver CUDA 13.0 (default)"
      echo "  cu124  host driver CUDA 12.4"
      read -r -p "CUDA tag [cu130]: " CUDA_TAG
    fi
    if [[ -z "${CUDA_TAG}" ]]; then
      CUDA_TAG="cu130"
    fi
  fi
  if [[ "${CUDA_TAG}" == "both" ]]; then
    die "--cuda-tag does not take both. Pass --with-images both to pull both instance images."
  fi
  if [[ "${CUDA_TAG}" != "cu130" && "${CUDA_TAG}" != "cu124" ]]; then
    die "CUDA tag must be cu130 or cu124 (host driver CUDA 13.0 or CUDA 12.4), got ${CUDA_TAG}."
  fi
}

choose_cuda_tag

if [[ -n "${COMFYFLEET_INSTANCE_DIGEST:-}" ]]; then
  INSTANCE_REF="$(digest_ref "${INSTANCE_REPO}" "${COMFYFLEET_INSTANCE_DIGEST}")"
elif [[ "${CUDA_TAG_EXPLICIT}" -eq 0 && -n "${COMFYFLEET_INSTANCE_IMAGE:-}" ]]; then
  INSTANCE_REF="${COMFYFLEET_INSTANCE_IMAGE}"
elif [[ -n "${COMFYFLEET_INSTANCE_IMAGE:-}" && "$(image_line "${COMFYFLEET_INSTANCE_IMAGE}")" == "${CUDA_TAG}" ]]; then
  INSTANCE_REF="${COMFYFLEET_INSTANCE_IMAGE}"
else
  INSTANCE_REF="${INSTANCE_REPO}:${CUDA_TAG}"
fi

if [[ -n "${COMFYFLEET_MANAGER_DIGEST:-}" ]]; then
  MANAGER_REF="$(digest_ref "${MANAGER_REPO}" "${COMFYFLEET_MANAGER_DIGEST}")"
elif [[ -n "${COMFYFLEET_MANAGER_IMAGE:-}" ]]; then
  MANAGER_REF="${COMFYFLEET_MANAGER_IMAGE}"
else
  MANAGER_REF="${MANAGER_REPO}:latest"
fi

export COMFYFLEET_CUDA_TAG="${CUDA_TAG}"
export COMFYFLEET_INSTANCE_IMAGE="${INSTANCE_REF}"
export COMFYFLEET_MANAGER_IMAGE="${MANAGER_REF}"

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    die "docker is not on PATH."
  fi
  if ! docker info >/dev/null 2>&1; then
    die "docker info failed. Start the Docker daemon and retry."
  fi
}

pull_ref() {
  local ref="$1"
  if ! docker pull "${ref}"; then
    echo "comfyfleet: docker pull failed for ${ref}." >&2
    echo "comfyfleet: the manager image is published by .github/workflows/publish-manager.yml on main." >&2
    echo "comfyfleet: until that run succeeds and the GHCR package is public, pull fails (denied or not found)." >&2
    return 1
  fi
}

pull_selected() {
  pull_ref "${MANAGER_REF}"
  case "${WITH_IMAGES}" in
    "")
      ;;
    cu130)
      pull_ref "${INSTANCE_REPO}:cu130"
      ;;
    cu124)
      pull_ref "${INSTANCE_REPO}:cu124"
      ;;
    both)
      pull_ref "${INSTANCE_REPO}:cu130"
      pull_ref "${INSTANCE_REPO}:cu124"
      ;;
  esac
  echo "comfyfleet: manager ${MANAGER_REF}"
  echo "comfyfleet: instance image ${INSTANCE_REF}"
  if [[ -n "${WITH_IMAGES}" ]]; then
    echo "comfyfleet: pulled instance images (${WITH_IMAGES})"
  else
    echo "comfyfleet: no instance image was pulled"
  fi
}

require_password() {
  if [[ -z "${COMFYFLEET_PASSWORD:-}" || -z "${COMFYFLEET_PASSWORD//[[:space:]]/}" ]]; then
    if [[ -t 0 ]]; then
      read -r -s -p "COMFYFLEET_PASSWORD: " COMFYFLEET_PASSWORD
      echo
    fi
  fi
  if [[ -z "${COMFYFLEET_PASSWORD:-}" || -z "${COMFYFLEET_PASSWORD//[[:space:]]/}" ]]; then
    die "COMFYFLEET_PASSWORD is required and must be non-empty. There is no open-LAN fallback."
  fi
  if [[ "${COMFYFLEET_PASSWORD}" == *$'\n'* ]]; then
    die "COMFYFLEET_PASSWORD must be a single line."
  fi
  export COMFYFLEET_PASSWORD
}

require_public_host() {
  if [[ -z "${COMFYFLEET_PUBLIC_HOST:-}" || -z "${COMFYFLEET_PUBLIC_HOST//[[:space:]]/}" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "COMFYFLEET_PUBLIC_HOST (0.0.0.0 or the machine's LAN IP): " COMFYFLEET_PUBLIC_HOST
    fi
  fi
  local host="${COMFYFLEET_PUBLIC_HOST:-}"
  if [[ -z "${host//[[:space:]]/}" ]]; then
    die "COMFYFLEET_PUBLIC_HOST is required. Set it to 0.0.0.0 or the machine's LAN IP."
  fi
  if ! public_host_ok "${host}"; then
    die "COMFYFLEET_PUBLIC_HOST is not a usable hostname or IP (${host})."
  fi
  export COMFYFLEET_PUBLIC_HOST="${host}"
}

public_host_ok() {
  local host="$1"
  [[ -n "${host}" && ${#host} -le 253 ]] || return 1
  # 0.0.0.0 is valid. It is what list and CLI URLs use when that is the setting.
  [[ "${host}" != "::" && "${host}" != "[::]" && "${host}" != "*" ]] || return 1
  [[ "${host}" != *[[:space:]]* && "${host}" != */* && "${host}" != *\\* ]] || return 1
  [[ "${host}" != *@* && "${host}" != *'#'* && "${host}" != *'?'* ]] || return 1
  [[ "${host}" != *'"'* && "${host}" != *\'* ]] || return 1
  return 0
}

remove_manager() {
  # Same container name as the previous manager. docker rm -f drops the
  # container only. It does not delete the /home/ComfyFleet bind mount.
  if docker container inspect "${NAME}" >/dev/null 2>&1; then
    echo "comfyfleet: replacing container ${NAME}. Files under /home/ComfyFleet are kept."
    docker rm -f "${NAME}" >/dev/null
  fi
}

script_dir() {
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    (cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  fi
}

write_compose() {
  local dest="$1"
  cat >"${dest}" <<'EOF'
# Generated by install.sh. Same contract as compose.yaml in the repo.
# Instance containers are not this service. The manager creates them with
# docker create --shm-size 8g (Compose form: shm_size: '8g').
services:
  manager:
    image: ${COMFYFLEET_MANAGER_IMAGE:?}
    container_name: comfyfleet-manager
    ports:
      - "9100:9100"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /home/ComfyFleet:/home/ComfyFleet
    environment:
      COMFYFLEET_PASSWORD: ${COMFYFLEET_PASSWORD:?Set COMFYFLEET_PASSWORD}
      COMFYFLEET_PUBLIC_HOST: ${COMFYFLEET_PUBLIC_HOST:?Set COMFYFLEET_PUBLIC_HOST}
      COMFYFLEET_INSTANCE_IMAGE: ${COMFYFLEET_INSTANCE_IMAGE:?}
      COMFYFLEET_CUDA_TAG: ${COMFYFLEET_CUDA_TAG:?}
      COMFYFLEET_BIND_HOST: 0.0.0.0
      COMFYFLEET_BIND_PORT: "9100"
    gpus: all
    restart: unless-stopped
EOF
}

start_run() {
  if [[ ! -S /var/run/docker.sock ]]; then
    echo "comfyfleet: /var/run/docker.sock is missing. The UI can still start; create needs the host socket." >&2
  fi
  remove_manager
  if ! docker run -d --name "${NAME}" \
    --restart unless-stopped \
    --gpus all \
    -p "${PORT}:9100" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /home/ComfyFleet:/home/ComfyFleet \
    -e "COMFYFLEET_PASSWORD=${COMFYFLEET_PASSWORD}" \
    -e "COMFYFLEET_PUBLIC_HOST=${COMFYFLEET_PUBLIC_HOST}" \
    -e "COMFYFLEET_INSTANCE_IMAGE=${COMFYFLEET_INSTANCE_IMAGE}" \
    -e "COMFYFLEET_CUDA_TAG=${COMFYFLEET_CUDA_TAG}" \
    -e COMFYFLEET_BIND_HOST=0.0.0.0 \
    -e COMFYFLEET_BIND_PORT=9100 \
    "${MANAGER_REF}"
  then
    echo "comfyfleet: docker run failed. --gpus all needs the NVIDIA Container Toolkit, and nvidia-smi must work on the host." >&2
    exit 1
  fi
  echo "comfyfleet: manager ${NAME} is starting."
  echo "comfyfleet: open http://${COMFYFLEET_PUBLIC_HOST}:${PORT}/"
}

start_compose() {
  if ! docker compose version >/dev/null 2>&1; then
    die "docker compose is not available. Re-run without --compose."
  fi
  local dir compose_file generated=0
  dir="$(script_dir || true)"
  if [[ -n "${dir}" && -f "${dir}/compose.yaml" ]]; then
    compose_file="${dir}/compose.yaml"
  else
    compose_file="$(mktemp)"
    generated=1
    write_compose "${compose_file}"
  fi
  remove_manager
  if ! docker compose -f "${compose_file}" up -d; then
    if [[ "${generated}" -eq 1 ]]; then
      rm -f "${compose_file}"
    fi
    echo "comfyfleet: docker compose up failed. gpus: all needs the NVIDIA Container Toolkit." >&2
    exit 1
  fi
  if [[ "${generated}" -eq 1 ]]; then
    rm -f "${compose_file}"
  fi
  echo "comfyfleet: manager is starting (compose, container ${NAME})."
  echo "comfyfleet: open http://${COMFYFLEET_PUBLIC_HOST}:9100/"
}

if [[ "${PULL_ONLY}" -eq 0 ]]; then
  require_password
  require_public_host
fi

require_docker
pull_selected

if [[ "${PULL_ONLY}" -eq 1 ]]; then
  echo "comfyfleet: pull-only done. Manager was not started."
  exit 0
fi

if [[ "${MODE}" == "compose" ]]; then
  start_compose
else
  start_run
fi
