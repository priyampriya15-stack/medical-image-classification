import logging
from typing import List
from fastapi import APIRouter, Query

from app.services.history_service import history_service, PredictionRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/history", response_model=List[PredictionRecord], summary="Get prediction history")
def get_predictions_history(limit: int = Query(20, ge=1, le=100)) -> List[PredictionRecord]:
    """
    Retrieve lightweight inference telemetry logs:
    timestamp, model_name, predicted_class, confidence, inference_time_ms.
    Does not permanently store uploaded medical images.
    """
    return history_service.get_predictions(limit=limit)
