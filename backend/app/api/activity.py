import logging
from typing import List
from fastapi import APIRouter, Query

from app.services.history_service import history_service, ActivityEvent, PredictionRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/activity", tags=["Activity"])


@router.get("", response_model=List[ActivityEvent], summary="Get recent research platform activities")
def get_recent_activity(limit: int = Query(20, ge=1, le=100)) -> List[ActivityEvent]:
    """
    Retrieve chronological activity events (training, evaluation, inference, grad-cam).
    """
    return history_service.get_activities(limit=limit)


@router.get("/predictions", response_model=List[PredictionRecord], summary="Get recent prediction metadata")
def get_prediction_history(limit: int = Query(20, ge=1, le=100)) -> List[PredictionRecord]:
    """
    Retrieve lightweight inference telemetry logs (metadata only, no raw images or PII).
    """
    return history_service.get_predictions(limit=limit)
