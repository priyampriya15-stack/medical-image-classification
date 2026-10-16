import logging
from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)


def create_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> Tuple[nn.Module, bool]:
    """
    Factory function to create ResNet50 model adapted for transfer learning.

    Args:
        num_classes: Number of target classes for the new classification head.
        pretrained: Whether to attempt loading ImageNet pretrained weights.
        freeze_backbone: Whether to freeze feature extractor layers initially.

    Returns:
        Tuple of (model, weights_loaded_successfully)
    """
    weights_loaded = False
    weights = None

    if pretrained:
        try:
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            weights_loaded = True
            logger.info("ResNet50: Pretrained ImageNet weights loaded successfully.")
        except Exception as exc:
            logger.warning(
                "ResNet50: Could not download/load ImageNet weights (%s). Initializing architecture without pretrained weights.",
                exc,
            )
            model = resnet50(weights=None)
    else:
        model = resnet50(weights=None)

    # Freeze convolutional feature extraction backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully connected layer for target classification task
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Ensure the new classification head requires gradients
    for param in model.fc.parameters():
        param.requires_grad = True

    return model, weights_loaded


def unfreeze_resnet50_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters in ResNet50 to permit end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Dynamically compute total and trainable parameter counts directly from PyTorch model.
    Never hardcoded.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
