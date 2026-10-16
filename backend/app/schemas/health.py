from typing import Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Operational status of the API")
    service: str = Field(description="Name of the service")
    version: str = Field(description="Software version")
    device: str = Field(description="Active execution device (e.g. CPU or CUDA)")
    cuda_available: bool = Field(description="Whether CUDA acceleration is available")
    pytorch_version: Optional[str] = Field(default=None, description="Installed PyTorch runtime version")
    torchvision_version: Optional[str] = Field(default=None, description="Installed Torchvision runtime version")
    python_version: Optional[str] = Field(default=None, description="Python interpreter version")
    models: Dict[str, bool] = Field(default_factory=dict, description="Model checkpoint / loaded status")
