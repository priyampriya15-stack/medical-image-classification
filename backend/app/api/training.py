import logging
from typing import Dict
from fastapi import APIRouter, HTTPException, status

from app.schemas.training import (
    TrainingConfig,
    TrainingJobStatus,
    TrainingListResponse,
)
from app.services.training_service import training_service
from app.models.registry import MODEL_SPECS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training"])


@router.post(
    "/start",
    response_model=TrainingJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start model training / fine-tuning in background",
)
def start_training(config: TrainingConfig) -> TrainingJobStatus:
    """
    Initiate asynchronous transfer learning training on PneumoniaMNIST.
    Uses CPU-safe background worker execution with validation and checkpoint saving.
    """
    if config.model_name not in MODEL_SPECS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{config.model_name}'. Registered: {list(MODEL_SPECS.keys())}",
        )

    try:
        job = training_service.start_training(config)
        return job
    except RuntimeError as err:
        logger.warning("Training start rejected: %s", err)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )
    except Exception as exc:
        logger.exception("Unexpected error starting training: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate training session.",
        )


@router.get(
    "/status",
    response_model=TrainingListResponse,
    summary="Get status of all model training jobs",
)
def get_all_training_statuses() -> TrainingListResponse:
    """Return current or latest training status, progress, and validation telemetry for all models."""
    jobs = training_service.get_all_jobs()
    return TrainingListResponse(jobs=jobs)


@router.get(
    "/status/{model_name}",
    response_model=TrainingJobStatus,
    summary="Get training status for a specific model",
)
def get_model_training_status(model_name: str) -> TrainingJobStatus:
    """Return detailed training status, progress, epoch metrics, and checkpoint info for a model."""
    if model_name not in MODEL_SPECS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Registered: {list(MODEL_SPECS.keys())}",
        )

    job = training_service.get_job_status(model_name)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training record found for model '{model_name}'.",
        )
    return job


@router.post(
    "/stop/{model_name}",
    response_model=Dict[str, str],
    summary="Stop an ongoing training job",
)
def stop_training_job(model_name: str) -> Dict[str, str]:
    """Signal an active training worker to halt gracefully at the next epoch boundary."""
    if model_name not in MODEL_SPECS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found.",
        )

    stopped = training_service.stop_training(model_name)
    if not stopped:
        return {"status": "not_running", "message": f"No active training job running for '{model_name}'."}
    return {"status": "stopping", "message": f"Stop signal delivered for '{model_name}'."}
