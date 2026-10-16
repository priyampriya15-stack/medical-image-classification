from typing import Tuple, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from app.evaluation.schemas import (
    ConfusionMatrixDetails,
    EvaluationMetrics,
    RocCurveData,
    PrecisionRecallCurveData,
)


def calculate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Calculate binary confusion matrix values strictly:
    Negative Class (0) = Normal
    Positive Class (1) = Pneumonia

    Returns:
        (true_negative, false_positive, false_negative, true_positive)
    """
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    if tn + fp + fn + tp != total:
        raise ValueError(
            f"Confusion matrix sum mismatch: {tn}+{fp}+{fn}+{tp} != {total}"
        )

    return tn, fp, fn, tp


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[
    EvaluationMetrics,
    ConfusionMatrixDetails,
    RocCurveData,
    PrecisionRecallCurveData,
]:
    """
    Compute verified diagnostic evaluation metrics from actual test predictions and probabilities.

    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Pneumonia)
        y_pred: Hard predicted class labels (0 or 1)
        y_scores: Continuous softmax probabilities for the positive class (Pneumonia)

    Returns:
        Tuple of (EvaluationMetrics, ConfusionMatrixDetails, RocCurveData, PrecisionRecallCurveData)
    """
    y_true = np.asarray(y_true, dtype=int).flatten()
    y_pred = np.asarray(y_pred, dtype=int).flatten()
    y_scores = np.asarray(y_scores, dtype=float).flatten()

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    if len(y_true) != len(y_pred) or len(y_true) != len(y_scores):
        raise ValueError("Shape mismatch between true labels, predictions, and scores.")

    # 1. Confusion Matrix
    tn, fp, fn, tp = calculate_confusion_matrix(y_true, y_pred)
    total_samples = len(y_true)

    # 2. Strict Metric Calculations
    # Specificity = TN / (TN + FP)
    actual_normal = tn + fp
    specificity = float(tn / actual_normal) if actual_normal > 0 else 0.0

    # Sensitivity / Recall = TP / (TP + FN)
    actual_pneumonia = tp + fn
    sensitivity = float(tp / actual_pneumonia) if actual_pneumonia > 0 else 0.0
    recall = sensitivity

    # Precision = TP / (TP + FP)
    predicted_pneumonia = tp + fp
    precision = float(tp / predicted_pneumonia) if predicted_pneumonia > 0 else 0.0

    # Accuracy = (TP + TN) / Total
    accuracy = float((tp + tn) / total_samples)

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    # ROC-AUC using predicted probability for positive class
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception as exc:
        raise ValueError(f"Failed to calculate ROC-AUC score: {exc}")

    # Validate range invariants [0.0, 1.0]
    for name, val in [
        ("accuracy", accuracy),
        ("precision", precision),
        ("recall", recall),
        ("sensitivity", sensitivity),
        ("specificity", specificity),
        ("f1", f1),
        ("roc_auc", roc_auc),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Calculated metric '{name}'={val} is outside valid interval [0, 1].")

    metrics = EvaluationMetrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        f1=round(f1, 4),
        roc_auc=round(roc_auc, 4),
    )

    conf_matrix_details = ConfusionMatrixDetails(
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
        raw_matrix=[[tn, fp], [fn, tp]],
    )

    # 3. ROC Curve
    fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)
    roc_data = RocCurveData(
        fpr=[round(float(x), 4) for x in fpr],
        tpr=[round(float(x), 4) for x in tpr],
        thresholds=[round(float(x), 4) for x in roc_thresh],
    )

    # 4. Precision-Recall Curve
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_scores)
    pr_data = PrecisionRecallCurveData(
        precision=[round(float(x), 4) for x in pr_prec],
        recall=[round(float(x), 4) for x in pr_rec],
        thresholds=[round(float(x), 4) for x in pr_thresh],
    )

    return metrics, conf_matrix_details, roc_data, pr_data
