from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_dataset_info_endpoint():
    response = client.get("/api/dataset/info")
    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "PneumoniaMNIST"
    assert data["available"] is True
    assert "classes" in data
    assert data["classes"] == ["Normal", "Pneumonia"]
    assert data["num_classes"] == 2
    assert "splits" in data
    assert "train" in data["splits"]
    assert "validation" in data["splits"]
    assert "test" in data["splits"]
    assert "image_dimensions" in data
    assert "28x28" in data["image_dimensions"]

    # When dataset archive is available, verify exact real counts
    assert data["splits"]["train"] == 4708
    assert data["splits"]["validation"] == 524
    assert data["splits"]["test"] == 624
    assert data["total_samples"] == 5856
    assert data["class_distribution"]["Normal"] == 1583
    assert data["class_distribution"]["Pneumonia"] == 4273


def test_dataset_samples_endpoint():
    response = client.get("/api/dataset/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0
    first = samples[0]
    assert "imageUrl" in first
    assert first["imageUrl"].startswith("data:image/png;base64,")
    assert first["classLabel"] in ["Normal", "Pneumonia"]
    assert "dimensions" in first


def test_dataset_root_endpoint():
    response = client.get("/api/dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PneumoniaMNIST"
    assert data["total_samples"] == 5856
    assert data["channels"] == 1
    assert data["train_size"] == 4708


def test_dataset_samples_paginated():
    response = client.get("/api/dataset/samples?split=test&page=1&page_size=5")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) == 5
    for s in samples:
        assert s["split"] == "test"
        assert s["imageUrl"].startswith("data:image/png;base64,")
        assert "sample_index" in s
        assert "label" in s

