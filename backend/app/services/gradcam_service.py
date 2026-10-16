import base64
import io
import logging
from pathlib import Path
import threading
import time
from typing import Optional, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from app.models.registry import MODEL_SPECS
from app.services.history_service import history_service
from app.services.prediction_service import (
    prediction_service,
    CLASS_NAMES,
    InvalidImageError,
    ModelNotTrainedError,
    UnsupportedModelError,
)

logger = logging.getLogger(__name__)


class GradCamResult:
    def __init__(
        self,
        model_name: str,
        display_name: str,
        target_layer: str,
        predicted_class: str,
        class_index: int,
        confidence: float,
        original_image_base64: str,
        heatmap_base64: str,
        overlay_base64: str,
        inference_time_ms: float,
    ):
        self.model_name = model_name
        self.display_name = display_name
        self.target_layer = target_layer
        self.predicted_class = predicted_class
        self.class_index = class_index
        self.confidence = confidence
        self.original_image_base64 = original_image_base64
        self.heatmap_base64 = heatmap_base64
        self.overlay_base64 = overlay_base64
        self.inference_time_ms = inference_time_ms

    def to_dict(self):
        return {
            "success": True,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "target_layer": self.target_layer,
            "predicted_class": self.predicted_class,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "heatmap": self.heatmap_base64,
            "heatmap_image": self.heatmap_base64,
            "heatmap_available": True,
            "overlay_image": self.overlay_base64,
            "original_image_base64": self.original_image_base64,
            "heatmap_base64": self.heatmap_base64,
            "overlay_base64": self.overlay_base64,
            "inference_time_ms": self.inference_time_ms,
            "device": "cpu",
        }


class GradCamService:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for model interpretability and attention localization.
    ResNet50: target layer = 'layer4'
    EfficientNetB0: target layer = 'features[-1]'
    """

    def __init__(self):
        self._lock = threading.Lock()

    def get_target_layer(self, model_name: str, model: nn.Module) -> Tuple[nn.Module, str]:
        if model_name == "resnet50":
            if hasattr(model, "layer4"):
                return model.layer4, "layer4"
            raise ValueError("ResNet50 does not possess 'layer4'")
        elif model_name == "efficientnet_b0":
            if hasattr(model, "features"):
                return model.features[-1], "features[-1]"
            raise ValueError("EfficientNetB0 does not possess 'features[-1]'")
        else:
            raise UnsupportedModelError(f"Unsupported model '{model_name}'.")

    def generate_gradcam(
        self,
        file_bytes: bytes,
        filename: str,
        model_name: str = "resnet50",
        target_class_idx: Optional[int] = None,
        alpha: float = 0.5,
    ) -> GradCamResult:
        model_name = model_name.strip().lower()
        if model_name not in MODEL_SPECS:
            raise UnsupportedModelError(f"Unsupported model '{model_name}'. Allowed: {list(MODEL_SPECS.keys())}")
        spec = MODEL_SPECS[model_name]

        # 1. Validate and load image
        pil_image, img_format, orig_w, orig_h = prediction_service.validate_and_load_image(
            file_bytes, filename
        )

        # 2. Preprocess input tensor at 224x224 for spatial feature map resolution
        img_rgb = pil_image.convert("RGB")
        img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
        img_np_float = np.array(img_resized, dtype=np.float32) / 255.0
        tensor_224 = torch.from_numpy(img_np_float).permute(2, 0, 1)
        norm_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(tensor_224)
        input_tensor = norm_tensor.unsqueeze(0).to(torch.device("cpu"))

        # 3. Load model
        model = prediction_service.get_or_load_model(model_name)

        start_time = time.perf_counter()

        with self._lock:
            # Locate target layer
            target_module, target_layer_name = self.get_target_layer(model_name, model)

            # Setup hooks
            activations = []
            gradients = []

            def forward_hook(module, inp, out):
                activations.append(out)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            handle_fwd = target_module.register_forward_hook(forward_hook)
            handle_bwd = target_module.register_full_backward_hook(backward_hook)

            try:
                # Enable gradient tracking for Grad-CAM
                model.zero_grad()
                input_tensor.requires_grad_(True)

                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).squeeze(0)

                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())
                predicted_class = CLASS_NAMES[pred_idx]

                class_for_cam = target_class_idx if target_class_idx is not None else pred_idx

                score = output[0, class_for_cam]
                score.backward()

                # Process gradients and activations
                grad = gradients[0].detach()  # (1, C, H, W)
                act = activations[0].detach()  # (1, C, H, W)

                # Global Average Pooling of gradients
                weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = torch.relu(cam)

                # Bilinearly interpolate CAM to 224x224
                cam_interp = torch.nn.functional.interpolate(
                    cam,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                cam_normalized = cam_interp[0, 0].cpu().numpy()
                cam_min, cam_max = float(cam_normalized.min()), float(cam_normalized.max())
                if cam_max - cam_min > 1e-8:
                    cam_normalized = (cam_normalized - cam_min) / (cam_max - cam_min)
                else:
                    cam_normalized = np.zeros_like(cam_normalized)

            finally:
                handle_fwd.remove()
                handle_bwd.remove()
                model.zero_grad()

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Prepare visualizations
        disp_size = (224, 224)
        orig_resized = pil_image.convert("RGB").resize(disp_size, Image.Resampling.BILINEAR)
        orig_np = np.array(orig_resized, dtype=np.float32) / 255.0

        # Apply Jet colormap using modern Matplotlib API
        import matplotlib
        cmap = matplotlib.colormaps["jet"]
        heatmap_colored = cmap(cam_normalized)[:, :, :3]  # drop alpha

        # Create overlay
        overlay_np = (1.0 - alpha) * orig_np + alpha * heatmap_colored
        overlay_np = np.clip(overlay_np, 0.0, 1.0)

        # Encode to Base64 PNGs
        def pil_to_base64(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        orig_b64 = pil_to_base64(orig_resized)
        heatmap_b64 = pil_to_base64(Image.fromarray((heatmap_colored * 255).astype(np.uint8)))
        overlay_b64 = pil_to_base64(Image.fromarray((overlay_np * 255).astype(np.uint8)))

        # Record activity event
        try:
            history_service.record_activity(
                event_type="GRADCAM_GENERATED",
                model_name=model_name,
                description=f"Generated Grad-CAM attention heatmap for {spec['display_name']} ({target_layer_name}). Attention target: {predicted_class}.",
                metadata={
                    "target_layer": target_layer_name,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "inference_time_ms": inference_time_ms,
                },
            )
        except Exception as h_err:
            logger.warning("Could not record Grad-CAM activity: %s", h_err)

        return GradCamResult(
            model_name=model_name,
            display_name=spec["display_name"],
            target_layer=target_layer_name,
            predicted_class=predicted_class,
            class_index=pred_idx,
            confidence=round(confidence, 4),
            original_image_base64=orig_b64,
            heatmap_base64=heatmap_b64,
            overlay_base64=overlay_b64,
            inference_time_ms=inference_time_ms,
        )


gradcam_service = GradCamService()
