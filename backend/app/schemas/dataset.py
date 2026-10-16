from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetSplits(BaseModel):
    train: int = Field(default=0, description="Number of training samples")
    validation: int = Field(default=0, description="Number of validation samples")
    test: int = Field(default=0, description="Number of testing samples")


class DatasetInfo(BaseModel):
    name: str = Field(default="PneumoniaMNIST", description="Dataset identifier")
    description: str = Field(
        default="PneumoniaMNIST pediatric chest radiograph benchmark for binary classification",
        description="Description of dataset and clinical scope",
    )
    available: bool = Field(description="Whether the dataset is downloaded and loaded in backend")
    num_classes: int = Field(default=2, description="Total number of discrete target classes")
    classes: List[str] = Field(default_factory=lambda: ["Normal", "Pneumonia"], description="Class label names")
    image_dimensions: str = Field(default="28x28 (Grayscale)", description="Native image resolution")
    image_size: str = Field(default="28x28 (Grayscale/1-channel)", description="Image size and channel info")
    channels: int = Field(default=1, description="Number of image channels: 1 (grayscale)")
    total_samples: int = Field(default=0, description="Total images across all splits")
    train_size: int = Field(default=0, description="Train split sample count")
    validation_size: int = Field(default=0, description="Validation split sample count")
    test_size: int = Field(default=0, description="Test split sample count")
    splits: DatasetSplits = Field(default_factory=DatasetSplits, description="Sample distribution per partition")
    class_distribution: Dict[str, int] = Field(default_factory=dict, description="Class balance across dataset")
    status_message: Optional[str] = Field(default=None, description="Detailed explanation of availability state")


# Alias for backward compatibility
DatasetInfoResponse = DatasetInfo
