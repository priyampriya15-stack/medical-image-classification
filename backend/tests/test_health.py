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
    assert data["models"]["resnet50"] is True
    assert data["models"]["efficientnet_b0"] is True


def test_health_root_alias():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Medical Image Classification API"
    assert data["models"]["resnet50"] is True
    assert data["models"]["efficientnet_b0"] is True


def test_root_endpoint():
    for path in ["/", "/api", "/api/"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Medical Image Classification API"
        assert data["health"] == "/api/health"

