#!/usr/bin/env bash
# Manager smoke: build the manager image, start it, check health and the UI.
# Create/start/stop against a real ComfyUI instance needs the instance image,
# a Docker socket, and a GPU. tests/test_manager.py covers that lifecycle with
# a fake engine so CI can run without a GPU.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"

if ! command -v docker >/dev/null 2>&1; then
  echo "manager-smoke: docker is not on PATH. Skipping the live container check."
  echo "manager-smoke: python -m unittest discover -s tests covers health, UI, create, start, and stop."
  exit 0
fi

tag="${COMFYFLEET_MANAGER_IMAGE:-ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest}"
name="comfyfleet-manager-smoke-$$"
port="${COMFYFLEET_SMOKE_PORT:-9100}"
password="${COMFYFLEET_SMOKE_PASSWORD:-smoke-not-a-real-secret}"

docker build -f Dockerfile.manager -t "${tag}" .

cleanup() {
  docker rm -f "${name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "${name}" -p "${port}:9100" \
  -e "COMFYFLEET_PASSWORD=${password}" \
  "${tag}" >/dev/null

ready=0
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:${port}/api/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "manager-smoke: control API did not become ready" >&2
  docker logs "${name}" >&2 || true
  exit 1
fi

echo "manager-smoke: health"
curl -fsS "http://127.0.0.1:${port}/api/health"
echo
echo "manager-smoke: login page"
curl -fsS "http://127.0.0.1:${port}/login" | grep -q "ComfyFleet"
curl -fsS "http://127.0.0.1:${port}/login" | grep -q "password"

unauth="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/instances" || true)"
echo "manager-smoke: unauthenticated /api/instances HTTP ${unauth}"
test "${unauth}" = "401"

echo "manager-smoke: signed-in ui"
curl -fsS -H "Authorization: Bearer ${password}" "http://127.0.0.1:${port}/" | grep -q "New instance"

gpu_code="$(curl -s -o /tmp/comfyfleet-smoke-gpus.json -w '%{http_code}' \
  -H "Authorization: Bearer ${password}" \
  "http://127.0.0.1:${port}/api/gpus" || true)"
echo "manager-smoke: /api/gpus HTTP ${gpu_code}"
if [[ "${gpu_code}" != "200" ]]; then
  grep -q "nvidia-smi" /tmp/comfyfleet-smoke-gpus.json
  echo "manager-smoke: GPU probe failed closed without host GPUs (expected in CI)."
fi

echo "manager-smoke: manager image is up, UI is served, health succeeded."
