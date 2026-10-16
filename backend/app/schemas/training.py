from typing import List, Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration parameters for model training / transfer learning."""
    model_name: str = Field(..., description="Target model name: 'resnet50' or 'efficientnet_b0'")
    epochs: int = Field(default=15, ge=1, le=100, description="Total training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Mini-batch size")
    learning_rate: float = Field(default=0.001, gt=0, le=0.1, description="Initial learning rate")
    optimizer: str = Field(default="AdamW", description="Optimizer type: 'Adam' or 'AdamW'")
    fine_tune: bool = Field(default=True, description="Whether to unfreeze backbone for domain adaptation")


class EpochMetric(BaseModel):
    """Metrics recorded at completion of a single training epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_duration_seconds: float


class TrainingJobStatus(BaseModel):
    """Current state, progress, and validation telemetry of a training job."""
    job_id: str
    model_name: str
    display_name: str
    status: str = Field(..., description="'idle' | 'queued' | 'running' | 'completed' | 'failed' | 'stopped'")
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0
    best_val_acc: Optional[float] = None
    best_epoch: Optional[int] = None
    current_train_loss: Optional[float] = None
    current_val_loss: Optional[float] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "AdamW"
    fine_tune: bool = True
    history: List[EpochMetric] = []
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    checkpoint_saved: bool = False
    checkpoint_path: Optional[str] = None


class TrainingListResponse(BaseModel):
    """List of all training jobs (active and completed)."""
    jobs: List[TrainingJobStatus]
