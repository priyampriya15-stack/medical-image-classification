import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import (
    prediction_service,
    InvalidImageError,
    ModelNotTrainedError,
    UnsupportedModelError,
    ProbabilityValidationError,
)
from app.services.gradcam_service import gradcam_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict pneumonia from a single chest radiograph",
    description=(
        "Upload a single chest radiograph (PNG, JPG, JPEG, WEBP) to obtain "
        "model prediction, class probabilities, and confidence from trained transfer backbones."
    ),
)
async def predict_image(
    image: Optional[UploadFile] = File(None, description="Uploaded chest radiograph image file"),
    file: Optional[UploadFile] = File(None, description="Uploaded image file (alias)"),
    model_name: str = Form("resnet50", description="Target model: 'resnet50' or 'efficientnet_b0'"),
):
    upload_file = image or file
    # Validate uploaded file presence
    if not upload_file or not upload_file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid or corrupted image file", "detail": "No image file provided."},
        )

    try:
        # Read file bytes in memory (no disk writing)
        file_bytes = await upload_file.read()

        response = prediction_service.predict(
            file_bytes=file_bytes,
            filename=upload_file.filename,
            model_name=model_name,
        )
        return response

    except InvalidImageError as exc:
        logger.warning("Image validation failed for '%s': %s", upload_file.filename, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid or corrupted image file", "detail": str(exc)},
        )

    except UnsupportedModelError as exc:
        logger.warning("Unsupported model requested '%s': %s", model_name, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Unsupported model", "detail": str(exc)},
        )

    except ModelNotTrainedError as exc:
        logger.warning("Prediction requested for untrained model '%s'", model_name)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Trained model checkpoint is not available. Model has not been trained yet.", "detail": str(exc)},
        )

    except ProbabilityValidationError as exc:
        logger.error("Probability validation invariant violated: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal inference validation error", "detail": str(exc)},
        )

    except Exception as exc:
        logger.error("Unexpected error during inference: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Prediction failure",
                "detail": "An internal error occurred during radiograph evaluation.",
            },
        )


@router.post(
    "/gradcam",
    summary="Compute Grad-CAM activation heatmap for a chest radiograph",
    description=(
        "Compute Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize "
        "salient attention regions for ResNet50 (layer4) or EfficientNetB0 (features[-1])."
    ),
)
async def generate_gradcam_endpoint(
    image: Optional[UploadFile] = File(None, description="Chest radiograph image file"),
    file: Optional[UploadFile] = File(None, description="Chest radiograph image file (alias)"),
    model_name: str = Form("resnet50", description="Target model: 'resnet50' or 'efficientnet_b0'"),
    opacity: float = Form(0.65, description="Overlay alpha blending ratio (0.0 - 1.0)"),
):
    upload_file = image or file
    if not upload_file or not upload_file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid or corrupted image file", "detail": "No image file provided."},
        )

    try:
        file_bytes = await upload_file.read()
        result = gradcam_service.generate_gradcam(
            file_bytes=file_bytes,
            filename=upload_file.filename,
            model_name=model_name,
            alpha=max(0.1, min(1.0, opacity)),
        )
        return result.to_dict()

    except InvalidImageError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid or corrupted image file", "detail": str(exc)},
        )
    except UnsupportedModelError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Unsupported model", "detail": str(exc)},
        )
    except ModelNotTrainedError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Trained model checkpoint is not available. Model has not been trained yet.", "detail": str(exc)},
        )
    except Exception as exc:
        logger.error("Grad-CAM generation failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Grad-CAM generation error", "detail": "Failed to generate attention heatmap."},
        )
