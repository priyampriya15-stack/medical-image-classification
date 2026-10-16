import logging
from typing import List, Optional

from app.models.registry import (
    get_all_models_info,
    get_model_info,
    build_model,
    MODEL_SPECS,
)
from app.schemas.model import ModelInfo

logger = logging.getLogger(__name__)


class ModelService:
    """Service layer coordinating model registry, architecture definitions and checkpoints."""

    def get_models(self) -> List[ModelInfo]:
        """Return list of all registered transfer learning model architectures and checkpoint states."""
        return get_all_models_info()

    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get metadata for a single model by key.

        Args:
            model_name: e.g. 'resnet50' or 'efficientnet_b0'

        Returns:
            ModelInfo if found, None otherwise.
        """
        return get_model_info(model_name.lower().strip())

    def validate_model_name(self, model_name: str) -> bool:
        """Verify whether a model key is registered in MODEL_SPECS."""
        return model_name.lower().strip() in MODEL_SPECS


model_service = ModelService()
