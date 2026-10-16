from fastapi.testclient import TestClient
from app.main import app
from app.models.resnet50 import create_resnet50, count_parameters as count_resnet_params
from app.models.efficientnet_b0 import (
    create_efficientnet_b0,
    count_parameters as count_effnet_params,
)

client = TestClient(app)


def test_list_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    model_names = [m["name"] for m in data["models"]]
    assert "resnet50" in model_names
    assert "efficientnet_b0" in model_names

    for model_info in data["models"]:
        assert "total_parameters" in model_info
        assert "trainable_parameters" in model_info
        assert model_info["total_parameters"] > 0
        # When frozen, trainable parameters are strictly less than total
        # Checkpoint rule: checkpoint_available is a boolean reflecting presence of trained checkpoint
        assert isinstance(model_info["checkpoint_available"], bool)



def test_get_single_model_endpoint():
    response = client.get("/api/models/resnet50")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "resnet50"
    assert data["num_classes"] == 2

    response_eff = client.get("/api/models/efficientnet_b0")
    assert response_eff.status_code == 200
    data_eff = response_eff.json()
    assert data_eff["name"] == "efficientnet_b0"
    assert data_eff["num_classes"] == 2


def test_invalid_model_name():
    response = client.get("/api/models/nonexistent_architecture")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_resnet50_factory_and_parameter_counting():
    # Instantiate without downloading weights for fast deterministic test
    model, _ = create_resnet50(num_classes=2, pretrained=False, freeze_backbone=True)
    total, trainable = count_resnet_params(model)

    assert total > 20_000_000  # ResNet50 has ~23-25M parameters
    assert trainable == 2048 * 2 + 2  # Linear(2048, 2) weights and bias = 4098 params


def test_efficientnet_b0_factory_and_parameter_counting():
    model, _ = create_efficientnet_b0(num_classes=2, pretrained=False, freeze_backbone=True)
    total, trainable = count_effnet_params(model)

    assert total > 3_000_000  # EfficientNetB0 has ~4-5M parameters
    assert trainable == 1280 * 2 + 2  # Linear(1280, 2) weights and bias = 2562 params
