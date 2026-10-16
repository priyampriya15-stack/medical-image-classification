import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.analytics import analytics_service, determine_best_model, generate_research_interpretation
from app.analytics.schemas import ModelAnalyticsItem

client = TestClient(app)


def test_analytics_summary_endpoint():
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200
    data = response.json()

    assert "project" in data
    assert data["project"]["name"] == "Medical Image Classification"
    assert data["project"]["dataset"] == "PneumoniaMNIST"
    assert data["project"]["num_classes"] == 2
    assert data["project"]["classes"] == ["Normal", "Pneumonia"]

    assert "models" in data
    assert len(data["models"]) >= 2
    model_names = [m["name"] for m in data["models"]]
    assert "resnet50" in model_names
    assert "efficientnet_b0" in model_names

    # Check metrics existence
    for m in data["models"]:
        assert "training_status" in m
        assert "evaluation_status" in m
        if m["evaluation_status"] == "completed":
            assert m["accuracy"] is not None
            assert m["precision"] is not None
            assert m["recall"] is not None
            assert m["sensitivity"] is not None
            assert m["specificity"] is not None
            assert m["f1"] is not None
            assert m["roc_auc"] is not None

    # Check best model is identified via ROC-AUC
    assert "best_model" in data
    if data["best_model"] is not None:
        assert data["best_model"]["primary_metric"] == "roc_auc"
        assert data["best_model"]["name"] in ["resnet50", "efficientnet_b0"]
        assert data["best_model"]["value"] > 0.5


def test_determine_best_model_none():
    # If no models are evaluated, best_model must be None
    models = [
        ModelAnalyticsItem(
            name="resnet50",
            display_name="ResNet50",
            training_status="not_started",
            evaluation_status="not_evaluated",
        ),
        ModelAnalyticsItem(
            name="efficientnet_b0",
            display_name="EfficientNetB0",
            training_status="not_started",
            evaluation_status="not_evaluated",
        ),
    ]
    best = determine_best_model(models)
    assert best is None


def test_determine_best_model_roc_auc_priority():
    # Model A has higher accuracy (0.95), but Model B has higher ROC-AUC (0.92 vs 0.88)
    models = [
        ModelAnalyticsItem(
            name="resnet50",
            display_name="ResNet50",
            training_status="completed",
            evaluation_status="completed",
            accuracy=0.95,
            roc_auc=0.88,
        ),
        ModelAnalyticsItem(
            name="efficientnet_b0",
            display_name="EfficientNetB0",
            training_status="completed",
            evaluation_status="completed",
            accuracy=0.85,
            roc_auc=0.92,
        ),
    ]
    best = determine_best_model(models)
    assert best is not None
    assert best.name == "efficientnet_b0"
    assert best.primary_metric == "roc_auc"
    assert best.value == 0.92


def test_research_interpretation_non_diagnostic():
    models = [
        ModelAnalyticsItem(
            name="resnet50",
            display_name="ResNet50",
            training_status="completed",
            evaluation_status="completed",
            accuracy=0.8317,
            sensitivity=0.9769,
            specificity=0.5897,
            roc_auc=0.8979,
            test_samples=624,
        ),
        ModelAnalyticsItem(
            name="efficientnet_b0",
            display_name="EfficientNetB0",
            training_status="completed",
            evaluation_status="completed",
            accuracy=0.8109,
            sensitivity=0.9615,
            specificity=0.5598,
            roc_auc=0.9068,
            test_samples=624,
        ),
    ]
    interp = generate_research_interpretation(models)
    assert "diagnos" not in interp.lower() or "non-diagnostic" in interp.lower()
    assert "EfficientNetB0" in interp
    assert "ResNet50" in interp
    assert "0.9068" in interp
    assert "0.8979" in interp
    assert "Best performance depends on the selected evaluation metric" in interp


def test_confusion_matrix_endpoint():
    response = client.get("/api/analytics/confusion-matrix")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert data["classes"] == ["Normal", "Pneumonia"]
    assert "resnet50" in data
    assert "efficientnet_b0" in data

    if data["resnet50"] is not None:
        cm = data["resnet50"]
        assert "matrix" in cm
        assert "true_positive" in cm
        assert "true_negative" in cm
        assert "false_positive" in cm
        assert "false_negative" in cm


def test_roc_endpoint():
    response = client.get("/api/analytics/roc")
    assert response.status_code == 200
    data = response.json()
    assert "resnet50" in data
    assert "efficientnet_b0" in data

    if data["resnet50"] is not None:
        roc = data["resnet50"]
        assert "false_positive_rate" in roc
        assert "true_positive_rate" in roc
        assert "roc_auc" in roc
        assert 0.0 <= roc["roc_auc"] <= 1.0


def test_precision_recall_endpoint():
    response = client.get("/api/analytics/precision-recall")
    assert response.status_code == 200
    data = response.json()
    assert "resnet50" in data
    assert "efficientnet_b0" in data

    if data["resnet50"] is not None:
        pr = data["resnet50"]
        assert "precision" in pr
        assert "recall" in pr
        assert "average_precision" in pr


def test_training_history_endpoint():
    response = client.get("/api/analytics/training-history")
    assert response.status_code == 200
    data = response.json()
    assert "resnet50" in data
    assert "efficientnet_b0" in data

    if data["resnet50"] is not None:
        hist = data["resnet50"]
        assert len(hist) > 0
        first_ep = hist[0]
        assert "epoch" in first_ep
        assert "train_loss" in first_ep
        assert "validation_loss" in first_ep
        assert "train_accuracy" in first_ep
        assert "validation_accuracy" in first_ep
        assert "learning_rate" in first_ep


def test_predictions_history_endpoint():
    response = client.get("/api/predictions/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_dataset_samples_endpoint():
    response = client.get("/api/dataset/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        sample = data[0]
        assert "id" in sample
        assert "imageUrl" in sample
        assert "classLabel" in sample
        assert sample["classLabel"] in ["Normal", "Pneumonia"]

