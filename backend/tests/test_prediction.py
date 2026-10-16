import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app.main import app
from app.services.prediction_service import (
    prediction_service,
    InvalidImageError,
    UnsupportedModelError,
    ModelNotTrainedError,
    ProbabilityValidationError,
)
from app.models.registry import MODEL_SPECS

client = TestClient(app)


def create_test_image_bytes(format_name: str = "PNG", size: tuple = (100, 100), color: int = 128) -> bytes:
    """Helper to generate in-memory valid image bytes."""
    buf = io.BytesIO()
    img = Image.new("L", size, color=color)
    img.save(buf, format=format_name)
    return buf.getvalue()


class DummyMockModel(nn.Module):
    """Deterministic dummy model for unit testing prediction logic without loading heavy weights."""
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits.repeat(x.size(0), 1)


# =====================================================================
# Unit Tests for PredictionService
# =====================================================================

def test_validate_and_load_image_valid_png():
    img_bytes = create_test_image_bytes("PNG", size=(64, 64))
    img, fmt, width, height = prediction_service.validate_and_load_image(img_bytes, "chest.png")
    assert fmt == "PNG"
    assert width == 64
    assert height == 64
    assert isinstance(img, Image.Image)


def test_validate_and_load_image_valid_jpeg():
    img_bytes = create_test_image_bytes("JPEG", size=(128, 128))
    img, fmt, width, height = prediction_service.validate_and_load_image(img_bytes, "scan.jpg")
    assert fmt == "JPEG"
    assert width == 128
    assert height == 128


def test_validate_and_load_image_invalid_extension():
    img_bytes = b"sample text content"
    with pytest.raises(InvalidImageError) as exc_info:
        prediction_service.validate_and_load_image(img_bytes, "report.pdf")
    assert "Unsupported file format" in str(exc_info.value)


def test_validate_and_load_image_corrupted():
    corrupted_bytes = b"not an image at all but with png extension"
    with pytest.raises(InvalidImageError) as exc_info:
        prediction_service.validate_and_load_image(corrupted_bytes, "corrupt.png")
    assert "Invalid or corrupted image file" in str(exc_info.value)


def test_validate_and_load_image_empty():
    with pytest.raises(InvalidImageError) as exc_info:
        prediction_service.validate_and_load_image(b"", "empty.png")
    assert "empty" in str(exc_info.value)


def test_preprocess_image_dimensions_and_channels():
    # 200x150 grayscale image
    pil_img = Image.new("L", (200, 150), color=100)
    tensor = prediction_service.preprocess_image(pil_img)

    # Must be (1, 3, 28, 28) on CPU
    assert tensor.shape == (1, 3, 28, 28)
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32


def test_model_cache_mechanism():
    prediction_service.clear_model_cache()
    assert len(prediction_service._cache) == 0

    # Inject dummy model into cache
    dummy = DummyMockModel(torch.tensor([[2.0, 5.0]]))
    prediction_service._cache["dummy_test"] = dummy
    assert "dummy_test" in prediction_service._cache

    prediction_service.clear_model_cache()
    assert len(prediction_service._cache) == 0


def test_prediction_with_mocked_model(monkeypatch):
    """Test inference, probability calculation, and confidence with mocked logits."""
    # Logits: [0.0, 2.1972] -> softmax yields [0.1, 0.9]
    dummy_model = DummyMockModel(torch.tensor([[0.0, 2.197224577]]))
    monkeypatch.setattr(prediction_service, "get_or_load_model", lambda name: dummy_model)

    img_bytes = create_test_image_bytes("PNG")
    result = prediction_service.predict(img_bytes, "sample.png", "resnet50")

    assert result.success is True
    assert result.prediction.predicted_class == "Pneumonia"
    assert result.prediction.class_index == 1
    assert pytest.approx(result.probabilities.Pneumonia, rel=1e-3) == 0.9
    assert pytest.approx(result.probabilities.Normal, rel=1e-3) == 0.1
    assert pytest.approx(result.prediction.confidence, rel=1e-3) == 0.9
    assert pytest.approx(result.probabilities.Normal + result.probabilities.Pneumonia, rel=1e-3) == 1.0
    assert result.inference_time_ms >= 0.0
    assert result.device == "cpu"


def test_prediction_class_mapping_normal(monkeypatch):
    """Test that higher index 0 logit maps to Normal."""
    # Logits: [3.0, 0.0] -> softmax yields ~0.9525 Normal, ~0.0474 Pneumonia
    dummy_model = DummyMockModel(torch.tensor([[3.0, 0.0]]))
    monkeypatch.setattr(prediction_service, "get_or_load_model", lambda name: dummy_model)

    img_bytes = create_test_image_bytes("JPEG")
    result = prediction_service.predict(img_bytes, "chest.jpeg", "efficientnet_b0")

    assert result.prediction.predicted_class == "Normal"
    assert result.prediction.class_index == 0
    assert result.prediction.confidence == pytest.approx(result.probabilities.Normal, rel=1e-3)
    assert result.probabilities.Normal > result.probabilities.Pneumonia


def test_missing_checkpoint_raises(monkeypatch):
    """Verify that if checkpoint cannot be resolved, ModelNotTrainedError is raised."""
    monkeypatch.setattr(prediction_service, "resolve_checkpoint", lambda name: (_ for _ in ()).throw(ModelNotTrainedError("Model has not been trained yet.")))
    prediction_service.clear_model_cache()

    img_bytes = create_test_image_bytes("PNG")
    with pytest.raises(ModelNotTrainedError):
        prediction_service.predict(img_bytes, "chest.png", "resnet50")


# =====================================================================
# API Endpoint Tests (POST /api/predict)
# =====================================================================

def test_api_predict_invalid_extension():
    response = client.post(
        "/api/predict",
        data={"model_name": "resnet50"},
        files={"image": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_api_predict_corrupted_image():
    response = client.post(
        "/api/predict",
        data={"model_name": "resnet50"},
        files={"image": ("corrupted.png", b"\x89PNG\r\n\x1a\ncorrupted_payload", "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Invalid or corrupted image file"


def test_api_predict_invalid_model():
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/predict",
        data={"model_name": "vgg16"},
        files={"image": ("chest.png", img_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported model" in data["error"]


def test_api_predict_untrained_model(monkeypatch):
    monkeypatch.setattr(
        prediction_service,
        "get_or_load_model",
        lambda name: (_ for _ in ()).throw(ModelNotTrainedError("Model has not been trained yet.")),
    )
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/api/predict",
        data={"model_name": "resnet50"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Model has not been trained yet." in data["error"]


# =====================================================================
# Integration / Smoke Test with Real Trained Checkpoints
# =====================================================================

def test_real_inference_resnet50():
    """Verify end-to-end inference using the actual trained ResNet50 checkpoint."""
    from app.services.dataset_service import dataset_service
    if not dataset_service.is_available():
        pytest.skip("Dataset archive not present")

    # Fetch 1 real test image from PneumoniaMNIST
    test_images, _ = dataset_service.load_split("test")
    raw_img = Image.fromarray(test_images[0])
    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/predict",
        data={"model_name": "resnet50"},
        files={"image": ("pneumoniamnist_test_0.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["success"] is True
    assert data["model"]["name"] == "resnet50"
    assert data["model"]["display_name"] == "ResNet50"
    assert data["prediction"]["class"] in ["Normal", "Pneumonia"]
    assert data["prediction"]["class_index"] in [0, 1]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0

    p_norm = data["probabilities"]["Normal"]
    p_pneu = data["probabilities"]["Pneumonia"]
    assert 0.0 <= p_norm <= 1.0
    assert 0.0 <= p_pneu <= 1.0
    assert pytest.approx(p_norm + p_pneu, rel=1e-3) == 1.0
    assert data["prediction"]["confidence"] == max(p_norm, p_pneu)

    assert data["image"]["filename"] == "pneumoniamnist_test_0.png"
    assert data["image"]["format"] == "PNG"
    assert data["inference_time_ms"] > 0
    assert data["device"] == "cpu"


def test_real_inference_efficientnet_b0():
    """Verify end-to-end inference using the actual trained EfficientNetB0 checkpoint."""
    from app.services.dataset_service import dataset_service
    if not dataset_service.is_available():
        pytest.skip("Dataset archive not present")

    test_images, _ = dataset_service.load_split("test")
    raw_img = Image.fromarray(test_images[1])
    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/predict",
        data={"model_name": "efficientnet_b0"},
        files={"image": ("pneumoniamnist_test_1.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["success"] is True
    assert data["model"]["name"] == "efficientnet_b0"
    assert data["model"]["display_name"] == "EfficientNetB0"
    assert data["prediction"]["class"] in ["Normal", "Pneumonia"]
    assert data["prediction"]["class_index"] in [0, 1]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0

    p_norm = data["probabilities"]["Normal"]
    p_pneu = data["probabilities"]["Pneumonia"]
    assert pytest.approx(p_norm + p_pneu, rel=1e-3) == 1.0
    assert data["prediction"]["confidence"] == max(p_norm, p_pneu)
    assert data["inference_time_ms"] > 0
    assert data["device"] == "cpu"
