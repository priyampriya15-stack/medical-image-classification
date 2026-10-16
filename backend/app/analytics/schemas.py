from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProjectInfoSchema(BaseModel):
    name: str = "Medical Image Classification"
    dataset: str = "PneumoniaMNIST"
    num_classes: int = 2
    classes: List[str] = ["Normal", "Pneumonia"]


class BestModelSchema(BaseModel):
    name: str
    display_name: str
    primary_metric: str = "roc_auc"
    value: float


class ModelAnalyticsItem(BaseModel):
    name: str
    display_name: str
    training_status: str
    evaluation_status: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
    test_samples: Optional[int] = None
    evaluated_at: Optional[str] = None
    confusion_matrix: Optional[Dict[str, Any]] = None


class AnalyticsSummaryResponse(BaseModel):
    project: ProjectInfoSchema
    models: List[ModelAnalyticsItem]
    best_model: Optional[BestModelSchema] = None
    research_interpretation: Optional[str] = None
    generated_at: str
    disclaimer: str = (
        "Educational & Research Use Only. This model has not been clinically validated. "
        "Performance metrics are specific to the evaluated research dataset and should not "
        "be interpreted as clinical diagnostic performance."
    )
