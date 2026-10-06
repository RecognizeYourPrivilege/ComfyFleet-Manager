"""Host paths under ``/home/ComfyFleet``.

Fleet data does not use other top-level directories in ``/home``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Subdirectories ComfyUI v0.37.4 looks for under models/.
MODEL_SUBDIRS = (
    "checkpoints",
    "configs",
    "loras",
    "vae",
    "text_encoders",
    "clip",
    "diffusion_models",
    "unet",
    "clip_vision",
    "style_models",
    "embeddings",
    "diffusers",
    "vae_approx",
    "controlnet",
    "t2i_adapter",
    "gligen",
    "upscale_models",
    "latent_upscale_models",
    "hypernetworks",
    "photomaker",
    "classifiers",
    "model_patches",
    "audio_encoders",
    "background_removal",
    "frame_interpolation",
    "geometry_estimation",
    "optical_flow",
    "detection",
    # Impact Pack SAM weights. The image does not bake them. First-run
    # download into this shared directory is the operator path.
    "sams",
)

CONTAINER_PORT = 8188
WORKFLOW_CONTAINER_PATH = "/opt/comfyfleet/instance/default_workflow.json"
# On-host storage root. models, wildcards, custom_nodes_*, and files
# are directories inside this path, never siblings of it.
HOST_ROOT = Path("/home/ComfyFleet")
# Path inside the instance container. The host directory is
# ``<HOST_ROOT>/wildcards``, bind-mounted here. Impact's impact-pack.ini
# is seeded with this value and no quotes.
WILDCARDS_CONTAINER = "/home/wildcards"
# Primary instance tags. cu130 is the default when the operator does not choose.
CUDA_TAGS = ("cu130", "cu124")
DEFAULT_CUDA_TAG = "cu130"
INSTANCE_IMAGE_REPO = "ghcr.io/recognizeyourprivilege/comfyfleet-images"
DEFAULT_IMAGE = f"{INSTANCE_IMAGE_REPO}:{DEFAULT_CUDA_TAG}"


def image_for_cuda_tag(cuda_tag: str) -> str:
    """GHCR ref for one instance CUDA line. The manager pulls it if it is missing."""

    return f"{INSTANCE_IMAGE_REPO}:{cuda_tag}"


# Published cu130 digest (GHCR run that built the CUDA 13.0 line). A ref that
# still says :phase1 but carries this digest is that cu130 image, not the
# later :phase1 alias of cu124.
CU130_PUBLISHED_DIGEST = (
    "sha256:cfa4afde856b909a8d3878688cb22eb3c65d17fe4e20efb22a959a3ce9890e75"
)


@dataclass(frozen=True)
class FleetLayout:
    """Host storage root. The CLI always uses ``/home/ComfyFleet``."""

    root: Path = HOST_ROOT

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def wildcards(self) -> Path:
        return self.root / "wildcards"

    @property
    def files(self) -> Path:
        return self.root / "files"

    def custom_nodes(self, name: str) -> Path:
        return self.root / f"custom_nodes_{name}"

    def instance_dir(self, name: str) -> Path:
        return self.files / name

    def workflow_file(self, name: str) -> Path:
        return self.instance_dir(name) / "default_workflow.json"

    def metadata_file(self, name: str) -> Path:
        return self.instance_dir(name) / "comfyfleet.json"

    def input_dir(self, name: str) -> Path:
        return self.instance_dir(name) / "input"

    def output_dir(self, name: str) -> Path:
        return self.instance_dir(name) / "output"

    def temp_dir(self, name: str) -> Path:
        return self.instance_dir(name) / "temp"
