import logging
from fastapi import APIRouter, HTTPException, status

from app.evaluation.manager import evaluation_manager
from app.evaluation.schemas import (
    EvaluationJobStatus,
    EvaluationResult,
    ModelComparisonResponse,
)
from app.models.registry import MODEL_SPECS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("", summary="Get test evaluation results for all models")
@router.get("/", summary="Get test evaluation results for all models")
def get_all_evaluations():
    """
    Retrieve real test evaluation metrics for ResNet50 and EfficientNetB0.
    Calculated directly from real test-split inferences on PneumoniaMNIST.
    """
    out = {}
    for model_name in ["resnet50", "efficientnet_b0"]:
        res = evaluation_manager.get_model_result(model_name)
        if res:
            out[model_name] = {
                "model_name": model_name,
                "display_name": res.display_name,
                "dataset": res.dataset,
                "split": res.split,
                "num_samples": res.num_samples,
                "accuracy": res.metrics.accuracy,
                "precision": res.metrics.precision,
                "recall": res.metrics.recall,
                "sensitivity": res.metrics.sensitivity,
                "specificity": res.metrics.specificity,
                "f1": res.metrics.f1,
                "roc_auc": res.metrics.roc_auc,
                "confusion_matrix": res.confusion_matrix.raw_matrix,
                "evaluated_at": res.evaluated_at,
            }
        else:
            out[model_name] = None
    out["comparison"] = evaluation_manager.get_comparison().model_dump()
    return out


@router.post(
    "/{model_name}",
    response_model=EvaluationJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger test set evaluation for a trained model",
)
def evaluate_model(model_name: str) -> EvaluationJobStatus:
    """
    Launch deterministic evaluation of a trained model against the held-out PneumoniaMNIST test set.
    Requires an existing task checkpoint.
    """
    if model_name not in MODEL_SPECS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{model_name}'. Allowed architectures: {list(MODEL_SPECS.keys())}",
        )

    try:
        job = evaluation_manager.start_evaluation(model_name)
        return job
    except FileNotFoundError as fnf_err:
        logger.warning("Evaluation rejected for %s: %s", model_name, fnf_err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' has not been trained yet. No task checkpoint exists.",
        )
    except Exception as exc:
        logger.exception("Error starting evaluation for %s: %s", model_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate test-set evaluation: {str(exc)}",
        )


@router.get(
    "/status/{job_id}",
    response_model=EvaluationJobStatus,
    summary="Get status of an evaluation job",
)
def get_evaluation_status(job_id: str) -> EvaluationJobStatus:
    """Check progress, sample counts, and telemetry of an ongoing evaluation job."""
    job = evaluation_manager.get_job_status(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation job '{job_id}' not found.",
        )
    return job


@router.get(
    "/results/{model_name}",
    response_model=EvaluationResult,
    summary="Get latest evaluation report for a model",
)
def get_evaluation_results(model_name: str) -> EvaluationResult:
    """Retrieve full test-set evaluation metrics, confusion matrix, and ROC/PR data."""
    if model_name not in MODEL_SPECS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{model_name}'. Allowed: {list(MODEL_SPECS.keys())}",
        )

    result = evaluation_manager.get_model_result(model_name)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' has not been evaluated yet on the test set.",
        )
    return result


@router.get(
    "/comparison",
    response_model=ModelComparisonResponse,
    summary="Comparative test evaluation of all models",
)
def get_model_comparison() -> ModelComparisonResponse:
    """
    Retrieve side-by-side test-set evaluation metrics across ResNet50 and EfficientNetB0.
    Ranks the top model based on ROC-AUC.
    """
    return evaluation_manager.get_comparison()
