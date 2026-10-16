import platform
from fastapi import APIRouter
import torch
import torchvision

from app.core.config import settings
from app.models.registry import MODEL_SPECS, check_task_checkpoint
from app.services.dataset_service import dataset_service

router = APIRouter(tags=["Project"])


@router.get("/project", summary="Get Medical Image Research Platform project metadata")
def get_project_metadata():
    """
    Retrieve project metadata, research title, academic dataset specifications,
    registered model architectures, runtime hardware target, and mandatory medical safety disclaimer.
    """
    ds_info = dataset_service.get_dataset_info()
    models_info = {}
    for name, spec in MODEL_SPECS.items():
        has_checkpoint, ckpt_path = check_task_checkpoint(name)
        models_info[name] = {
            "name": name,
            "display_name": spec["display_name"],
            "architecture": spec["architecture"],
            "checkpoint_available": has_checkpoint,
            "checkpoint_path": ckpt_path,
        }

    return {
        "title": "Medical Image Classification Using Transfer Learning: A Comparative Study of ResNet50 and EfficientNetB0 on PneumoniaMNIST",
        "short_name": "MedAI Research Platform",
        "version": settings.APP_VERSION,
        "purpose": "Academic and educational deep learning research",
        "disclaimer": "Not for clinical use. Model predictions are research outputs and not medical diagnoses.",
        "environment": {
            "device": settings.get_device().type,
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": platform.python_version(),
        },
        "dataset": {
            "name": ds_info.name,
            "num_classes": ds_info.num_classes,
            "classes": ds_info.classes,
            "available": ds_info.available,
            "total_samples": ds_info.total_samples,
            "splits": ds_info.splits.model_dump(),
            "image_dimensions": ds_info.image_dimensions,
            "channels": getattr(ds_info, "channels", 1),
        },
        "models": models_info,
    }
