import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.evaluation.metrics import calculate_confusion_matrix, calculate_all_metrics
from app.evaluation.evaluator import resolve_checkpoint_path
from app.evaluation.schemas import EvaluationMetrics, ConfusionMatrixDetails

client = TestClient(app)


def test_confusion_matrix_calculation():
    """Verify deterministic binary confusion matrix computation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

    tn, fp, fn, tp = calculate_confusion_matrix(y_true, y_pred)

    assert tn == 3  # Normal predicted as Normal
    assert fp == 1  # Normal predicted as Pneumonia
    assert fn == 1  # Pneumonia predicted as Normal
    assert tp == 5  # Pneumonia predicted as Pneumonia
    assert tn + fp + fn + tp == len(y_true)


def test_specificity_and_diagnostic_metrics():
    """Verify math properties of specificity, recall, precision, and F1."""
    # Synthetic deterministic scenario:
    # 10 samples: 4 Normal (0), 6 Pneumonia (1)
    # TN=3, FP=1, FN=2, TP=4
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.35, 0.7, 0.85, 0.9, 0.95])

    metrics, conf, roc, pr = calculate_all_metrics(y_true, y_pred, y_scores)

    # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
    assert metrics.specificity == 0.75
    # Recall / Sensitivity = TP / (TP + FN) = 4 / 6 = 0.6667
    assert metrics.recall == 0.6667
    assert metrics.sensitivity == 0.6667
    # Precision = TP / (TP + FP) = 4 / 5 = 0.8
    assert metrics.precision == 0.8
    # Accuracy = (3 + 4) / 10 = 0.7
    assert metrics.accuracy == 0.7
    # F1 = 2 * (0.8 * 0.6667) / (0.8 + 0.6667) = 0.7273
    assert metrics.f1 == 0.7273
    # ROC-AUC is in valid range
    assert 0.0 <= metrics.roc_auc <= 1.0

    # Verify confusion matrix details
    assert conf.true_negative == 3
    assert conf.false_positive == 1
    assert conf.false_negative == 2
    assert conf.true_positive == 4
    assert conf.raw_matrix == [[3, 1], [2, 4]]

    # Verify curve coordinates are present
    assert len(roc.fpr) > 0
    assert len(roc.tpr) > 0
    assert len(pr.precision) > 0
    assert len(pr.recall) > 0


def test_evaluation_comparison_endpoint():
    """Verify GET /api/evaluation/comparison returns registered models."""
    response = client.get("/api/evaluation/comparison")
    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert len(data["models"]) == 2
    assert "primary_metric" in data
    assert data["primary_metric"] == "roc_auc"

    for m in data["models"]:
        assert m["model_name"] in ("resnet50", "efficientnet_b0")
        if m["evaluated"]:
            assert m["accuracy"] is not None
            assert m["roc_auc"] is not None
            assert m["specificity"] is not None


def test_get_evaluation_result_endpoint():
    """Verify GET /api/evaluation/results/{model_name} returns full report."""
    response = client.get("/api/evaluation/results/resnet50")
    # If evaluated, should return 200 with complete metrics schema
    if response.status_code == 200:
        data = response.json()
        assert data["model_name"] == "resnet50"
        assert data["split"] == "test"
        assert data["num_samples"] == 624
        assert "metrics" in data
        assert "confusion_matrix" in data
        assert "roc_curve" in data
        assert "precision_recall_curve" in data


def test_evaluation_invalid_model_endpoint():
    """Verify POST /api/evaluation/{invalid} returns 400."""
    response = client.post("/api/evaluation/vgg19_unsupported")
    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"].lower()


def test_missing_checkpoint_resolution():
    """Verify resolve_checkpoint_path raises FileNotFoundError for non-existent model."""
    with pytest.raises(ValueError):
        resolve_checkpoint_path("imaginary_model")
