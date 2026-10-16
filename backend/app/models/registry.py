import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch.nn as nn

from app.core.config import settings
from app.models.resnet50 import create_resnet50, count_parameters as count_resnet_params
from app.models.efficientnet_b0 import (
    create_efficientnet_b0,
    count_parameters as count_effnet_params,
)
from app.schemas.model import ModelInfo

logger = logging.getLogger(__name__)

# Model registry specification
MODEL_SPECS = {
    "resnet50": {
        "name": "resnet50",
        "display_name": "ResNet50",
        "architecture": "Deep Residual Network (50 layers) with Bottleneck residual blocks",
        "factory": create_resnet50,
        "count_fn": count_resnet_params,
        "default_checkpoint": "resnet50_best.pt",
    },
    "efficientnet_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNetB0",
        "architecture": "Compound-Scaled Convolutional Network with Mobile Inverted Bottleneck (MBConv)",
        "factory": create_efficientnet_b0,
        "count_fn": count_effnet_params,
        "default_checkpoint": "efficientnet_b0_best.pt",
    },
}

# Cache for calculated parameter counts so we don't recreate torch models on every HTTP request
_MODEL_METADATA_CACHE: Dict[str, Tuple[int, int]] = {}


def check_task_checkpoint(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a fine-tuned task-specific checkpoint exists in CHECKPOINT_DIR.
    A pretrained ImageNet model is NOT considered a task-trained medical checkpoint.
    """
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        return False, None

    checkpoint_file = settings.CHECKPOINT_DIR / spec["default_checkpoint"]
    if checkpoint_file.exists() and checkpoint_file.is_file() and checkpoint_file.stat().st_size > 0:
        return True, str(checkpoint_file)
    return False, None


def get_cached_param_counts(model_name: str) -> Tuple[int, int]:
    """Retrieve or dynamically calculate parameter counts from the model architecture."""
    if model_name in _MODEL_METADATA_CACHE:
        return _MODEL_METADATA_CACHE[model_name]

    spec = MODEL_SPECS[model_name]
    # Instantiate architecture once to dynamically calculate exact parameter counts
    model, _ = spec["factory"](num_classes=2, pretrained=False, freeze_backbone=True)
    total, trainable = spec["count_fn"](model)
    _MODEL_METADATA_CACHE[model_name] = (total, trainable)
    return total, trainable


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get dynamic metadata for a specific model."""
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        return None

    checkpoint_available, checkpoint_path = check_task_checkpoint(model_name)
    total_params, trainable_params = get_cached_param_counts(model_name)

    return ModelInfo(
        name=spec["name"],
        display_name=spec["display_name"],
        architecture=spec["architecture"],
        num_classes=2,
        pretrained=True,
        backbone_frozen=True,
        checkpoint_available=checkpoint_available,
        total_parameters=total_params,
        parameters=total_params,
        trainable_parameters=trainable_params,
        checkpoint_path=checkpoint_path,
        loaded_status=checkpoint_available,
        training_status="Trained" if checkpoint_available else "Untrained",
        device="cpu",
        version="1.0.0",
    )


def get_all_models_info() -> List[ModelInfo]:
    """Get metadata for all registered models in the registry."""
    infos = []
    for model_name in MODEL_SPECS.keys():
        info = get_model_info(model_name)
        if info:
            infos.append(info)
    return infos


def build_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> Tuple[nn.Module, bool]:
    """
    Instantiate model from factory.

    Raises:
        ValueError: If model_name is not registered.
    """
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        raise ValueError(
            f"Unsupported model '{model_name}'. Registered models: {list(MODEL_SPECS.keys())}"
        )

    return spec["factory"](
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
