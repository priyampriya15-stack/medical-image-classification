from fastapi import APIRouter, HTTPException, status
from app.schemas.model import ModelListResponse, ModelInfo
from app.services.model_service import model_service
from app.models.registry import MODEL_SPECS

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """
    List all supported transfer learning architectures (ResNet50, EfficientNetB0),
    including dynamic parameter counts and task checkpoint availability on disk.
    """
    models = model_service.get_models()
    return ModelListResponse(models=models)


@router.get("/{model_name}", response_model=ModelInfo)
def get_model(model_name: str) -> ModelInfo:
    """
    Retrieve architecture metadata, dynamic parameter counts, and checkpoint state for a specific model.

    Raises:
        HTTPException 404: If model_name is not registered.
    """
    clean_name = model_name.lower().strip()
    model_info = model_service.get_model(clean_name)
    if not model_info:
        valid_keys = list(MODEL_SPECS.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Supported architectures: {valid_keys}",
        )
    return model_info
