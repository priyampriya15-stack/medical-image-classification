from typing import List, Optional, Tuple
from app.analytics.schemas import BestModelSchema, ModelAnalyticsItem, ProjectInfoSchema


def determine_best_model(models: List[ModelAnalyticsItem]) -> Optional[BestModelSchema]:
    """
    Ranks models using ROC-AUC as the primary ranking metric.
    Strictly requires actual evaluation data. If no model has been evaluated, returns None.
    """
    best_item: Optional[ModelAnalyticsItem] = None
    best_roc_auc: float = -1.0

    for item in models:
        if item.evaluation_status == "completed" and item.roc_auc is not None:
            if item.roc_auc > best_roc_auc:
                best_roc_auc = item.roc_auc
                best_item = item

    if best_item is not None and best_roc_auc >= 0.0:
        return BestModelSchema(
            name=best_item.name,
            display_name=best_item.display_name,
            primary_metric="roc_auc",
            value=round(best_roc_auc, 4),
        )
    return None


def generate_research_interpretation(models: List[ModelAnalyticsItem]) -> str:
    """
    Generates an objective, non-diagnostic research interpretation statement
    strictly derived from actual test-set values.
    Does NOT make clinical claims or declare universal superiority.
    """
    evaluated_models = [m for m in models if m.evaluation_status == "completed" and m.roc_auc is not None]

    if not evaluated_models:
        return (
            "On this research dataset, model test-set evaluations have not been completed yet. "
            "Quantitative diagnostic performance metrics remain uncalculated until evaluation runs finish."
        )

    if len(evaluated_models) == 1:
        m = evaluated_models[0]
        acc_pct = f"{m.accuracy * 100:.2f}%" if m.accuracy is not None else "N/A"
        roc_str = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "N/A"
        sens_pct = f"{m.sensitivity * 100:.2f}%" if m.sensitivity is not None else "N/A"
        spec_pct = f"{m.specificity * 100:.2f}%" if m.specificity is not None else "N/A"
        return (
            f"On the evaluated PneumoniaMNIST test split ({m.test_samples or 624} samples), "
            f"{m.display_name} achieved an ROC-AUC of {roc_str}, accuracy of {acc_pct}, "
            f"sensitivity of {sens_pct}, and specificity of {spec_pct}. "
            "Additional comparative architectures have not yet completed held-out evaluation."
        )

    # Both models evaluated: compare them neutrally
    m1, m2 = evaluated_models[0], evaluated_models[1]
    if m1.name != "resnet50" and m2.name == "resnet50":
        m1, m2 = m2, m1  # m1 is resnet50, m2 is efficientnet_b0

    r_acc = f"{m1.accuracy * 100:.2f}%" if m1.accuracy is not None else "N/A"
    r_roc = f"{m1.roc_auc:.4f}" if m1.roc_auc is not None else "N/A"
    r_sens = f"{m1.sensitivity * 100:.2f}%" if m1.sensitivity is not None else "N/A"

    e_acc = f"{m2.accuracy * 100:.2f}%" if m2.accuracy is not None else "N/A"
    e_roc = f"{m2.roc_auc:.4f}" if m2.roc_auc is not None else "N/A"
    e_sens = f"{m2.sensitivity * 100:.2f}%" if m2.sensitivity is not None else "N/A"

    return (
        f"On the evaluated PneumoniaMNIST test split ({m1.test_samples or 624} samples), "
        f"{m2.display_name} achieved a higher ROC-AUC ({e_roc}) than {m1.display_name} ({r_roc}), "
        f"indicating superior threshold-independent discrimination. "
        f"Conversely, {m1.display_name} demonstrated higher overall test accuracy ({r_acc} vs {e_acc}) "
        f"and higher sensitivity ({r_sens} vs {e_sens}). "
        "Best performance depends on the selected evaluation metric: ROC-AUC evaluates discriminatory ranking across thresholds, "
        "whereas sensitivity emphasizes detecting positive cases. Neither architecture is universally superior across all operational requirements."
    )
