import logging
from fastapi import APIRouter, HTTPException, status

from app.analytics import analytics_service, AnalyticsSummaryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    summary="Get complete research analytics summary",
    description=(
        "Collects and synthesizes model training status, checkpoints, test evaluation metrics, "
        "and primary ranking (ROC-AUC) across ResNet50 and EfficientNetB0 without re-running models."
    ),
)
def get_analytics_summary() -> AnalyticsSummaryResponse:
    try:
        return analytics_service.get_summary()
    except Exception as exc:
        logger.error("Failed to compile analytics summary: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analytics summary.",
        )


@router.get(
    "/confusion-matrix",
    summary="Get confusion matrix for all models",
    description="Returns real confusion matrices for ResNet50 and EfficientNetB0 on PneumoniaMNIST.",
)
def get_confusion_matrix():
    try:
        return analytics_service.get_confusion_matrices()
    except Exception as exc:
        logger.error("Failed to retrieve confusion matrices: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve confusion matrices.",
        )


@router.get(
    "/roc",
    summary="Get ROC curve points and AUC for all models",
    description="Returns actual false_positive_rate, true_positive_rate, and roc_auc for both models.",
)
def get_roc_curve():
    try:
        return analytics_service.get_roc_data()
    except Exception as exc:
        logger.error("Failed to retrieve ROC data: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ROC data.",
        )


@router.get(
    "/precision-recall",
    summary="Get Precision-Recall curve points and average precision",
    description="Returns actual precision, recall, and average_precision for both models.",
)
def get_precision_recall():
    try:
        return analytics_service.get_precision_recall_data()
    except Exception as exc:
        logger.error("Failed to retrieve Precision-Recall data: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve Precision-Recall data.",
        )


@router.get(
    "/training-history",
    summary="Get real training convergence history for all models",
    description="Returns actual training history with epoch, train_loss, validation_loss, train_accuracy, validation_accuracy, learning_rate.",
)
def get_training_history():
    try:
        return analytics_service.get_all_training_histories()
    except Exception as exc:
        logger.error("Failed to retrieve training history: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training history.",
        )

