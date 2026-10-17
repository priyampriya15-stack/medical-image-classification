import io
import logging
from pathlib import Path
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from app.core.config import settings
from app.models.registry import build_model, MODEL_SPECS
from app.schemas.prediction import (
    PredictionResponse,
    PredictionResult,
    ModelPredictionInfo,
    ImageInfo,
    ClassProbabilities,
)
from app.services.history_service import history_service

logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_PIL_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MODEL_INPUT_SIZE = (28, 28)
CLASS_NAMES = ["Normal", "Pneumonia"]

# Shared normalization matching ImageNet and PneumoniaMNIST training pipeline
IMAGE_NORMALIZATION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class PredictionException(Exception):
    """Base exception for prediction failures."""
    pass


class InvalidImageError(PredictionException):
    """Raised when uploaded file is invalid, corrupted, or unsupported."""
    pass


class ModelNotTrainedError(PredictionException):
    """Raised when the fine-tuned task checkpoint does not exist."""
    pass


class UnsupportedModelError(PredictionException):
    """Raised when an invalid model identifier is provided."""
    pass


class ProbabilityValidationError(PredictionException):
    """Raised when calculated probabilities violate mathematical constraints."""
    pass


class PredictionService:
    """
    Manages deterministic PyTorch inference on CPU for uploaded chest radiographs.
    Reuses cached model instances to prevent memory overhead and disk thrashing.
    """

    def __init__(self):
        self._cache: Dict[str, nn.Module] = {}
        self._lock = threading.Lock()

    def clear_model_cache(self) -> None:
        """Safely clear the in-memory PyTorch model cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared in-memory model cache for prediction service.")

    def resolve_checkpoint(self, model_name: str) -> Path:
        """
        Locate task-specific fine-tuned checkpoint.
        Checks nested path (checkpoints/resnet50/best.pt) and flat path (checkpoints/resnet50_best.pt).

        Raises:
            UnsupportedModelError: If model_name is not supported.
            ModelNotTrainedError: If no trained checkpoint exists.
        """
        spec = MODEL_SPECS.get(model_name)
        if not spec:
            raise UnsupportedModelError(
                f"Unsupported model '{model_name}'. Supported models: {list(MODEL_SPECS.keys())}"
            )

        checkpoint_filename = spec["default_checkpoint"]
        search_dirs = [
            settings.CHECKPOINT_DIR,
            settings.BASE_DIR / "checkpoints",
            Path.cwd() / "backend" / "checkpoints",
            Path.cwd() / "checkpoints",
            settings.BASE_DIR.parent / "backend" / "checkpoints",
        ]

        candidates = []
        for d in search_dirs:
            candidates.append(d / model_name / "best.pt")
            candidates.append(d / checkpoint_filename)

        for p in candidates:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return p.resolve()

        raise ModelNotTrainedError("Trained model checkpoint is not available. Model has not been trained yet.")

    def get_or_load_model(self, model_name: str) -> nn.Module:
        """
        Retrieve cached model or load checkpoint weights onto CPU.

        Raises:
            UnsupportedModelError: If model_name is invalid.
            ModelNotTrainedError: If checkpoint does not exist.
            ValueError: If checkpoint loading fails.
        """
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]

            # 1. Verify task-specific checkpoint exists
            checkpoint_path = self.resolve_checkpoint(model_name)
            logger.info("Loading model '%s' from task checkpoint: %s", model_name, checkpoint_path)

            device = torch.device("cpu")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
            except Exception as exc:
                logger.error("Failed to read checkpoint '%s': %s", checkpoint_path, exc)
                raise ValueError(f"Failed to load checkpoint for '{model_name}': {exc}")

            # Extract weights state_dict
            if isinstance(checkpoint, dict) and "model_name" in checkpoint:
                if checkpoint["model_name"] != model_name:
                    raise ValueError(
                        f"Checkpoint model mismatch: expected '{model_name}', found '{checkpoint['model_name']}'."
                    )
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # 2. Instantiate blank architecture (num_classes=2, no ImageNet downloads)
            model, _ = build_model(
                model_name=model_name,
                num_classes=2,
                pretrained=False,
                freeze_backbone=False,
            )

            # 3. Load trained weights strictly
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self._cache[model_name] = model
            logger.info("Successfully cached '%s' in memory for CPU inference.", model_name)
            return model

    def validate_and_load_image(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Tuple[Image.Image, str, int, int]:
        """
        Validate file extension, size, and actual PIL image content in memory.

        Returns:
            Tuple of (PIL Image, verified_format, width, height)

        Raises:
            InvalidImageError: If the file is unreadable, corrupted, or unsupported.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise InvalidImageError("Invalid or corrupted image file: File is empty.")

        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise InvalidImageError(
                f"File size exceeds maximum allowed limit of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB."
            )

        # Validate file extension
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise InvalidImageError(
                f"Unsupported file format '{suffix}'. Allowed formats: PNG, JPG, JPEG, WEBP."
            )

        # Validate actual image content with Pillow
        try:
            image_stream = io.BytesIO(file_bytes)
            img = Image.open(image_stream)
            img.load()  # Force reading entire image to catch corrupted data
        except Exception as exc:
            logger.warning("Corrupted image validation failed for '%s': %s", filename, exc)
            raise InvalidImageError("Invalid or corrupted image file")

        # Verify underlying image format
        img_format = (img.format or "").upper()
        if img_format not in ALLOWED_PIL_FORMATS and suffix not in ALLOWED_EXTENSIONS:
            raise InvalidImageError(
                f"Image content format '{img_format}' is not supported."
            )

        width, height = img.size
        if width <= 0 or height <= 0:
            raise InvalidImageError("Invalid image dimensions.")

        return img, (img_format or suffix.strip(".").upper()), width, height

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Transform PIL image to match the exact training/evaluation pipeline:
        1. Convert to RGB (3 channels)
        2. Resize to PneumoniaMNIST dimension (28, 28) with bilinear interpolation
        3. Convert uint8 [0, 255] to float tensor [0.0, 1.0]
        4. Normalize using ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        5. Add batch dimension: (1, 3, 28, 28)
        6. Device: CPU

        Returns:
            torch.Tensor of shape (1, 3, 28, 28) on CPU
        """
        # Convert to 3-channel RGB representation
        img_rgb = pil_image.convert("RGB")

        # Resize to input dimensions (28x28)
        img_resized = img_rgb.resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)

        # Convert to float numpy array normalized to [0, 1]
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # Permute HWC -> CHW: (28, 28, 3) -> (3, 28, 28)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Apply ImageNet normalization
        tensor = IMAGE_NORMALIZATION(tensor)

        # Add batch dimension: (1, 3, 28, 28)
        tensor = tensor.unsqueeze(0).to(torch.device("cpu"))

        return tensor

    def predict(
        self,
        file_bytes: bytes,
        filename: str,
        model_name: str,
    ) -> PredictionResponse:
        """
        Execute full end-to-end inference for a single radiograph:
        1. Validate model_name
        2. Validate image file
        3. Preprocess image
        4. Load model checkpoint from cache
        5. Run inference with torch.no_grad()
        6. Compute softmax probabilities
        7. Validate probability constraints
        8. Return structured PredictionResponse

        Raises:
            UnsupportedModelError: If model_name is invalid.
            ModelNotTrainedError: If checkpoint is missing.
            InvalidImageError: If image file is invalid or corrupted.
            ProbabilityValidationError: If math invariants fail.
        """
        # 1. Normalize model_name and validate existence in specs
        model_name = model_name.strip().lower()
        if model_name not in MODEL_SPECS:
            raise UnsupportedModelError(
                f"Unsupported model '{model_name}'. Allowed: {list(MODEL_SPECS.keys())}"
            )
        spec = MODEL_SPECS[model_name]

        # 2. Validate and load image
        pil_image, img_format, orig_width, orig_height = self.validate_and_load_image(
            file_bytes, filename
        )

        # 3. Preprocess image
        input_tensor = self.preprocess_image(pil_image)

        # 4. Get or load model from cache
        model = self.get_or_load_model(model_name)

        # 5. Execute inference on CPU with actual timing
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 6. Extract probabilities
        prob_normal = float(probabilities[0].item())
        prob_pneumonia = float(probabilities[1].item())

        # 7. Strict mathematical validation
        if not (0.0 <= prob_normal <= 1.0 and 0.0 <= prob_pneumonia <= 1.0):
            raise ProbabilityValidationError(
                f"Probabilities out of bounds: Normal={prob_normal}, Pneumonia={prob_pneumonia}"
            )

        prob_sum = prob_normal + prob_pneumonia
        if abs(prob_sum - 1.0) > 1e-3:
            raise ProbabilityValidationError(
                f"Probabilities do not sum to 1.0: sum={prob_sum}"
            )

        class_index = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[class_index]
        confidence = prob_pneumonia if class_index == 1 else prob_normal

        # Invariant: confidence must strictly match max(probabilities)
        max_prob = max(prob_normal, prob_pneumonia)
        if abs(confidence - max_prob) > 1e-4:
            raise ProbabilityValidationError(
                f"Confidence {confidence} does not match max probability {max_prob}"
            )

        # 8. Record prediction metadata in history service
        try:
            history_service.record_prediction(
                model_name=model_name,
                predicted_class=predicted_class,
                confidence=confidence,
                inference_time_ms=inference_time_ms,
            )
        except Exception as h_err:
            logger.warning("Could not record prediction event: %s", h_err)

        # 9. Log technical operational metrics (never log image data)
        logger.info(
            "Prediction executed successfully | Model: %s | Image: %s (%dx%d, %s) | "
            "Inference: %.2f ms | Device: cpu | Result: %s (Confidence: %.4f)",
            model_name,
            filename,
            orig_width,
            orig_height,
            img_format,
            inference_time_ms,
            predicted_class,
            confidence,
        )

        return PredictionResponse(
            success=True,
            model_name=model_name,
            predicted_class=predicted_class,
            class_name=predicted_class,
            class_index=class_index,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            device="cpu",
            model=ModelPredictionInfo(
                name=spec["name"],
                display_name=spec["display_name"],
            ),
            prediction=PredictionResult(
                predicted_class=predicted_class,
                class_index=class_index,
                confidence=confidence,
                inference_time_ms=inference_time_ms,
            ),
            probabilities=ClassProbabilities(
                Normal=prob_normal,
                Pneumonia=prob_pneumonia,
            ),
            image=ImageInfo(
                filename=Path(filename).name,
                format=img_format,
                width=orig_width,
                height=orig_height,
            ),
        )


# Singleton instance
prediction_service = PredictionService()
