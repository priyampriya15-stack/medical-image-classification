import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.history_service import history_service

client = TestClient(app)


def test_activity_recording_and_retrieval():
    event = history_service.record_activity(
        event_type="TRAINING_COMPLETED",
        model_name="resnet50",
        description="Test training completed",
        metadata={"val_acc": 0.95},
    )
    assert event.id.startswith("act-")
    assert event.event_type == "TRAINING_COMPLETED"
    assert event.model_name == "resnet50"

    response = client.get("/api/activity")
    assert response.status_code == 200
    activities = response.json()
    assert isinstance(activities, list)
    assert len(activities) > 0
    first = activities[0]
    assert "timestamp" in first
    assert "event_type" in first
    assert "model_name" in first
    assert "description" in first


def test_prediction_history_recording_and_retrieval():
    record = history_service.record_prediction(
        model_name="efficientnet_b0",
        predicted_class="Normal",
        confidence=0.8876,
        inference_time_ms=18.4,
    )
    assert record.id.startswith("pred-")
    assert record.model_name == "efficientnet_b0"
    assert record.predicted_class == "Normal"
    assert record.confidence == 0.8876

    response = client.get("/api/activity/predictions")
    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) > 0
    first = predictions[0]
    assert "timestamp" in first
    assert "model_name" in first
    assert "predicted_class" in first
    assert "confidence" in first
    assert "inference_time_ms" in first

    # Also test GET /api/predictions/history
    response_pred_hist = client.get("/api/predictions/history")
    assert response_pred_hist.status_code == 200
    p_data = response_pred_hist.json()
    assert isinstance(p_data, list)
    assert len(p_data) > 0
    assert "predicted_class" in p_data[0]

