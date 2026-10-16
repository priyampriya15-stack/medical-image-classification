from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "Medical Image Classification API"
    assert data["version"] == "1.0.0"
    assert data["device"] == "cpu"
    assert data["cuda_available"] is False
    assert "torchvision_version" in data
    assert "python_version" in data
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert "resnet50" in data["models"]
    assert "efficientnet_b0" in data["models"]
