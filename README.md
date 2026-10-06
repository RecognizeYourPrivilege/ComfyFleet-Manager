# ComfyFleet Manager

ComfyFleet Manager is a password-gated control panel that creates and runs ComfyUI containers on the host's Docker engine.

## Install with Docker

You need Docker, an NVIDIA driver, and the NVIDIA Container Toolkit (`docker run --gpus all` must work on the host).

The manager listens on `0.0.0.0` port 9100. `COMFYFLEET_PUBLIC_HOST` is the host written into `comfyfleet list` URLs and into the URL the CLI prints after create or start. Set it to `0.0.0.0` or the machine's LAN IP. The web UI Open link uses the browser's host, not this variable.

```bash
curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash
```

That pulls only `ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest`. Pass `--cuda-tag cu124` to point the manager at `ghcr.io/recognizeyourprivilege/comfyfleet-images:cu124` instead of `:cu130`. Optional instance-image pull (`cu130`, `cu124`, or `both`):

```bash
curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash -s -- --with-images cu130
curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash -s -- --with-images cu124
curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash -s -- --with-images both
```

## Update

```bash
curl -fsSL https://raw.githubusercontent.com/RecognizeYourPrivilege/ComfyFleet-Manager/main/install.sh | COMFYFLEET_PASSWORD='your-password' COMFYFLEET_PUBLIC_HOST=<ip-address> bash
```

That pulls `ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest` and replaces the `comfyfleet-manager` container. Files under `/home/ComfyFleet` stay.

## Free up space

```bash
docker image rm ghcr.io/recognizeyourprivilege/comfyfleet-manager-legacy:latest
```

Removes the previous manager image after switching. Does not delete `/home/ComfyFleet`. If Docker says the image is missing, it is already gone.

```bash
docker image prune -f
```

Removes dangling image layers (untagged images no container uses). Does not delete `/home/ComfyFleet`.

```bash
docker image prune -a -f
```

Removes images no container uses, including old ComfyFleet images. Images used by a running container stay. Does not delete `/home/ComfyFleet`. Unused instance images are pulled again the next time they are needed.

```bash
docker builder prune -af
```

Removes the Docker build cache. Does not delete images, containers, or `/home/ComfyFleet`.

```bash
docker container prune -f
```

Removes stopped containers. A stopped ComfyUI instance container is removed; its files under `/home/ComfyFleet` stay, but the container is gone. Do not run this if you still need those stopped instances.

Do not run `docker system prune --volumes` or `rm -rf /home/ComfyFleet`. Those can delete data.

## Features

- Password-gated web UI and HTTP API on port 9100
- Create, start, stop, force-stop, restart, and delete ComfyUI instances
- Each instance has its own port, GPU selection, and workflow JSON
- Default instance image `ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130`, or `:cu124` when that CUDA line is selected
- Pulls an instance image when create finds it missing on the host engine
- Shared models directory and per-instance input, output, temp, and custom nodes under `/home/ComfyFleet`
- Clone custom-node git URLs, extract zip packs, and install nodes the workflow is missing
- Per-instance launch options: VRAM mode, attention backend, and extra arguments
- Gallery of instance output files
- Browser terminal on a running instance
- Import an existing container into the fleet
- Fix ownership on the host storage directories
- Prune stopped containers that are not fleet instances
- Lists host GPUs with `nvidia-smi` (`--gpus all` on the manager)

## Build your own

```bash
docker build -f Dockerfile.manager -t ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest .
docker run -d --name comfyfleet-manager --restart unless-stopped --gpus all -p 9100:9100 -v /var/run/docker.sock:/var/run/docker.sock -v /home/ComfyFleet:/home/ComfyFleet -e COMFYFLEET_PASSWORD='your-password' -e COMFYFLEET_PUBLIC_HOST=<ip-address> -e COMFYFLEET_INSTANCE_IMAGE=ghcr.io/recognizeyourprivilege/comfyfleet-images:cu130 ghcr.io/recognizeyourprivilege/comfyfleet-manager:latest
```
