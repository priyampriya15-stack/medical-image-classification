import platform
from fastapi import APIRouter
import torchvision

from app.core.config import settings
from app.models.registry import MODEL_SPECS, check_task_checkpoint
from app.schemas.health import HealthResponse
from app.utils.device import get_device_info

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/health/", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Health check endpoint returning real runtime information:
    system status, PyTorch version, Torchvision version, Python version,
    hardware execution device, CUDA availability, and loaded model checkpoint states.
    """
    device_info = get_device_info()
    models_status = {
        name: check_task_checkpoint(name)[0]
        for name in MODEL_SPECS.keys()
    }

    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=device_info["device"],
        cuda_available=device_info["cuda_available"],
        pytorch_version=device_info["pytorch_version"],
        torchvision_version=torchvision.__version__,
        python_version=platform.python_version(),
        models=models_status,
    )
