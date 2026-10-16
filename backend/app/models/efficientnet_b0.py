import logging
from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)


def create_efficientnet_b0(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> Tuple[nn.Module, bool]:
    """
    Factory function to create EfficientNetB0 model adapted for transfer learning.

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
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            weights_loaded = True
            logger.info("EfficientNetB0: Pretrained ImageNet weights loaded successfully.")
        except Exception as exc:
            logger.warning(
                "EfficientNetB0: Could not download/load ImageNet weights (%s). Initializing architecture without pretrained weights.",
                exc,
            )
            model = efficientnet_b0(weights=None)
    else:
        model = efficientnet_b0(weights=None)

    # Freeze convolutional feature extraction backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final linear classifier layer in classifier sequential block
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Ensure the new classification head requires gradients
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model, weights_loaded


def unfreeze_efficientnet_b0_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters in EfficientNetB0 to permit end-to-end fine-tuning."""
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
