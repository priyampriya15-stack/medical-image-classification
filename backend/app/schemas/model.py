from typing import List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str = Field(description="Unique identifier key (e.g. resnet50, efficientnet_b0)")
    display_name: str = Field(description="Human readable title")
    architecture: str = Field(description="Detailed architectural family and building blocks")
    num_classes: int = Field(default=2, description="Target classification head capacity")
    pretrained: bool = Field(description="Whether ImageNet transfer weights are initialized")
    backbone_frozen: bool = Field(description="Whether convolutional feature extractor backbone is currently frozen")
    checkpoint_available: bool = Field(description="Whether a fine-tuned task-specific checkpoint exists on disk")
    total_parameters: int = Field(description="Total model parameters dynamically counted")
    parameters: int = Field(default=0, description="Alias for total parameter count")
    trainable_parameters: int = Field(description="Parameters requiring gradients dynamically counted")
    checkpoint_path: Optional[str] = Field(default=None, description="Filesystem location of fine-tuned weights if available")
    loaded_status: bool = Field(default=False, description="Whether model weights are loaded and ready in memory")
    training_status: str = Field(default="Untrained", description="Training status: 'Trained' or 'Untrained'")
    device: str = Field(default="cpu", description="Active runtime execution device")
    version: Optional[str] = Field(default="1.0.0", description="Model version")


class ModelListResponse(BaseModel):
    models: List[ModelInfo] = Field(description="List of registered architectures")
