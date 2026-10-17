from pathlib import Path
from typing import Optional
import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Medical Image Classification API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    @field_validator("API_PREFIX", mode="after")
    @classmethod
    def clean_api_prefix(cls, v: str) -> str:
        if not v or v.strip() in ("", "/"):
            return "/api"
        return "/" + v.strip().strip("/")


    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Base filesystem paths relative to backend root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
    CHECKPOINT_DIR: Path = Path(__file__).resolve().parent.parent.parent / "checkpoints"
    OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent.parent / "outputs"

    # Optional manual device override ("cpu", "cuda", etc.)
    DEVICE: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def get_device(self) -> torch.device:
        """Resolve and return active torch device."""
        if self.DEVICE:
            return torch.device(self.DEVICE)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device_str(self) -> str:
        """Return human-readable device identifier."""
        device = self.get_device()
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(0)
            return f"CUDA ({device_name})"
        return "CPU"


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
