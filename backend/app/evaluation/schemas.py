from typing import List, Optional
from pydantic import BaseModel, Field


class ConfusionMatrixDetails(BaseModel):
    """Binary confusion matrix components and 2x2 grid."""
    true_negative: int = Field(..., description="Actual Normal correctly identified as Normal")
    false_positive: int = Field(..., description="Actual Normal incorrectly identified as Pneumonia")
    false_negative: int = Field(..., description="Actual Pneumonia incorrectly identified as Normal")
    true_positive: int = Field(..., description="Actual Pneumonia correctly identified as Pneumonia")
    raw_matrix: List[List[int]] = Field(..., description="[[TN, FP], [FN, TP]]")


class EvaluationMetrics(BaseModel):
    """Comprehensive diagnostic metrics evaluated on held-out test dataset."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall test accuracy: (TP + TN) / Total")
    precision: float = Field(..., ge=0.0, le=1.0, description="Positive predictive value: TP / (TP + FP)")
    recall: float = Field(..., ge=0.0, le=1.0, description="Sensitivity: TP / (TP + FN)")
    sensitivity: float = Field(..., ge=0.0, le=1.0, description="True positive rate (identical to recall)")
    specificity: float = Field(..., ge=0.0, le=1.0, description="True negative rate: TN / (TN + FP)")
    f1: float = Field(..., ge=0.0, le=1.0, description="Harmonic mean of precision and recall")
    roc_auc: float = Field(..., ge=0.0, le=1.0, description="Area under Receiver Operating Characteristic curve")


class RocCurveData(BaseModel):
    """ROC curve coordinates generated from positive-class softmax probabilities."""
    fpr: List[float] = Field(..., description="False positive rate coordinates")
    tpr: List[float] = Field(..., description="True positive rate coordinates")
    thresholds: List[float] = Field(..., description="Decision thresholds")


class PrecisionRecallCurveData(BaseModel):
    """Precision-Recall curve coordinates."""
    precision: List[float] = Field(..., description="Precision coordinates")
    recall: List[float] = Field(..., description="Recall coordinates")
    thresholds: List[float] = Field(..., description="Decision thresholds")


class EvaluationResult(BaseModel):
    """Complete evaluation report for a trained architecture on the test split."""
    model_name: str
    display_name: str
    dataset: str = "PneumoniaMNIST"
    split: str = "test"
    num_samples: int
    metrics: EvaluationMetrics
    confusion_matrix: ConfusionMatrixDetails
    roc_curve: RocCurveData
    precision_recall_curve: PrecisionRecallCurveData
    checkpoint_path: str
    evaluated_at: str


class EvaluationJobStatus(BaseModel):
    """Status and telemetry of an ongoing or completed test set evaluation job."""
    job_id: str
    model_name: str
    status: str = Field(..., description="'queued' | 'evaluating' | 'completed' | 'failed'")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    processed_samples: int = 0
    total_samples: int = 0
    elapsed_time: float = 0.0
    error: Optional[str] = None
    result: Optional[EvaluationResult] = None


class ModelComparisonItem(BaseModel):
    """Comparative test metrics for a single model."""
    model_name: str
    display_name: str
    evaluated: bool
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    evaluated_at: Optional[str] = None
    status_text: Optional[str] = None


class ModelComparisonResponse(BaseModel):
    """Comparative evaluation across ResNet50 and EfficientNetB0 architectures."""
    models: List[ModelComparisonItem]
    primary_metric: str = "roc_auc"
    best_model: Optional[str] = None
    comparison_summary: Optional[str] = None
