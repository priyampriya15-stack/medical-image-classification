import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.training import TrainingConfig, EpochMetric, TrainingJobStatus

client = TestClient(app)


def test_get_all_training_status():
    """Verify GET /api/train/status returns registered models with valid schema."""
    response = client.get("/api/train/status")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) == 2
    model_names = [j["model_name"] for j in data["jobs"]]
    assert "resnet50" in model_names
    assert "efficientnet_b0" in model_names


def test_get_single_model_training_status():
    """Verify GET /api/train/status/{model_name} returns proper status."""
    response = client.get("/api/train/status/resnet50")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "resnet50"
    assert data["display_name"] == "ResNet50"
    assert "status" in data
    assert "current_epoch" in data


def test_get_training_status_invalid_model():
    """Verify GET /api/train/status/{invalid} returns 404."""
    response = client.get("/api/train/status/vgg16_unknown")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_start_training_invalid_model():
    """Verify POST /api/train/start with invalid model name returns 400."""
    response = client.post(
        "/api/train/start",
        json={
            "model_name": "alexnet_unsupported",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "fine_tune": True,
        },
    )
    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"].lower()


def test_training_schema_validation():
    """Verify Pydantic models validate input boundaries."""
    # Test valid config
    config = TrainingConfig(
        model_name="resnet50",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer="AdamW",
        fine_tune=True,
    )
    assert config.model_name == "resnet50"
    assert config.epochs == 10

    # Test invalid epoch (< 1)
    with pytest.raises(Exception):
        TrainingConfig(
            model_name="resnet50",
            epochs=0,
            batch_size=32,
            learning_rate=0.001,
        )

    # Test invalid batch size (< 8)
    with pytest.raises(Exception):
        TrainingConfig(
            model_name="resnet50",
            epochs=5,
            batch_size=4,
            learning_rate=0.001,
        )
