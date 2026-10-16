import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.gradcam_service import gradcam_service
from app.services.dataset_service import dataset_service

client = TestClient(app)


def create_test_image_bytes(format_name: str = "PNG", size: tuple = (64, 64), color: int = 150) -> bytes:
    buf = io.BytesIO()
    img = Image.new("L", size, color=color)
    img.save(buf, format=format_name)
    return buf.getvalue()


def test_gradcam_invalid_model():
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/gradcam",
        data={"model_name": "vgg16"},
        files={"image": ("test.png", img_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["error"]


def test_gradcam_invalid_extension():
    response = client.post(
        "/api/gradcam",
        data={"model_name": "resnet50"},
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["error"]


def test_gradcam_resnet50_success():
    """Verify Grad-CAM produces real attention heatmaps and base64 PNGs on ResNet50."""
    if not dataset_service.is_available():
        pytest.skip("Dataset archive not present")

    test_images, _ = dataset_service.load_split("test")
    raw_img = Image.fromarray(test_images[0])
    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/gradcam",
        data={"model_name": "resnet50", "opacity": 0.65},
        files={"image": ("sample.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["success"] is True
    assert data["model_name"] == "resnet50"
    assert data["display_name"] == "ResNet50"
    assert data["target_layer"] == "layer4"
    assert data["predicted_class"] in ["Normal", "Pneumonia"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["original_image_base64"].startswith("data:image/png;base64,")
    assert data["heatmap_base64"].startswith("data:image/png;base64,")
    assert data["overlay_base64"].startswith("data:image/png;base64,")
    assert data["inference_time_ms"] > 0
    assert data["device"] == "cpu"


def test_gradcam_efficientnet_b0_success():
    """Verify Grad-CAM produces real attention heatmaps on EfficientNetB0 target layer features[-1]."""
    if not dataset_service.is_available():
        pytest.skip("Dataset archive not present")

    test_images, _ = dataset_service.load_split("test")
    raw_img = Image.fromarray(test_images[1])
    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/api/gradcam",
        data={"model_name": "efficientnet_b0", "opacity": 0.5},
        files={"image": ("sample.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["success"] is True
    assert data["model_name"] == "efficientnet_b0"
    assert data["display_name"] == "EfficientNetB0"
    assert data["target_layer"] == "features[-1]"
    assert data["predicted_class"] in ["Normal", "Pneumonia"]
    assert data["original_image_base64"].startswith("data:image/png;base64,")
    assert data["heatmap_base64"].startswith("data:image/png;base64,")
    assert data["overlay_base64"].startswith("data:image/png;base64,")
