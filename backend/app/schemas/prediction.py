from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, model_serializer


class ModelPredictionInfo(BaseModel):
    name: str
    display_name: str


class PredictionResult(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        data = handler(self)
        data["class"] = self.predicted_class
        return data


class ClassProbabilities(BaseModel):
    Normal: float
    Pneumonia: float


class ImageInfo(BaseModel):
    filename: str
    format: str
    width: int
    height: int


class PredictionResponse(BaseModel):
    success: bool = True
    model_name: str
    predicted_class: str
    class_name: Optional[str] = None
    class_index: int
    confidence: float
    inference_time_ms: float
    device: str = "cpu"

    # Backward-compatible structured fields
    model: ModelPredictionInfo
    prediction: PredictionResult
    probabilities: ClassProbabilities
    image: ImageInfo

    model_config = ConfigDict(populate_by_name=True)


class GradCamResponse(BaseModel):
    success: bool = True
    model_name: str
    display_name: str
    target_layer: str
    predicted_class: str
    class_index: int
    confidence: float
    heatmap_available: bool = True
    overlay_image: str
    overlay_base64: str
    original_image_base64: str
    heatmap_base64: str
    inference_time_ms: float
    device: str = "cpu"
