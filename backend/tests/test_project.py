from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_project_endpoint():
    response = client.get("/api/project")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "PneumoniaMNIST" in data["title"]
    assert "disclaimer" in data
    assert "Not for clinical use" in data["disclaimer"]
    assert "environment" in data
    assert data["environment"]["device"] == "cpu"
    assert "dataset" in data
    assert data["dataset"]["num_classes"] == 2
    assert "models" in data
    assert "resnet50" in data["models"]
    assert "efficientnet_b0" in data["models"]


def test_training_alias_endpoint():
    response = client.get("/api/training/status")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) == 2


def test_evaluation_root_endpoint():
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "resnet50" in data
    assert "efficientnet_b0" in data
    assert "comparison" in data
    if data["resnet50"]:
        assert "accuracy" in data["resnet50"]
        assert "confusion_matrix" in data["resnet50"]
