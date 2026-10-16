import torch
from app.core.config import settings


def get_device_info() -> dict:
    """Return dictionary of current device and PyTorch execution environment details."""
    cuda_available = torch.cuda.is_available()
    device = settings.get_device()

    return {
        "device": device.type,  # e.g. "cpu" or "cuda"
        "device_type": device.type,
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
    }
