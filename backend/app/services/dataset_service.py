import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from app.core.config import settings
from app.schemas.dataset import DatasetInfoResponse, DatasetSplits

logger = logging.getLogger(__name__)

DATASET_FILENAME = "pneumoniamnist.npz"


class PneumoniaMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapper for PneumoniaMNIST arrays with PyTorch/torchvision transforms.
    Converts 1-channel grayscale to 3-channel representations for pretrained ImageNet transfer backbones.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
        as_rgb: bool = True,
    ):
        self.images = images
        self.labels = labels.squeeze()
        self.transform = transform
        self.as_rgb = as_rgb

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]  # shape: (28, 28), uint8
        label = int(self.labels[idx])

        # Convert uint8 numpy array to float tensor [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # shape: (1, 28, 28)

        # Replicate single channel to 3 channels if RGB input expected by ResNet/EfficientNet
        if self.as_rgb:
            img_tensor = img_tensor.repeat(3, 1, 1)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class DatasetService:
    """Service for managing PneumoniaMNIST dataset loading, metadata extraction and split preservation."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self._cached_info: Optional[DatasetInfoResponse] = None

    @property
    def dataset_path(self) -> Path:
        candidate_dirs = [
            self.data_dir,
            settings.DATA_DIR,
            settings.BASE_DIR / "data",
            Path.cwd() / "backend" / "data",
            Path.cwd() / "data",
            settings.BASE_DIR.parent / "backend" / "data",
        ]
        for d in candidate_dirs:
            p = d / DATASET_FILENAME
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return p.resolve()
        return (self.data_dir / DATASET_FILENAME).resolve()

    def is_available(self) -> bool:
        """Check if pneumoniamnist.npz exists and is readable."""
        return self.dataset_path.exists() and self.dataset_path.stat().st_size > 0

    def get_dataset_info(self) -> DatasetInfoResponse:
        """
        Load and return structured dataset metadata.
        Never fabricates counts; values are derived directly from the loaded NPZ archive.
        """
        if not self.is_available():
            return DatasetInfoResponse(
                name="PneumoniaMNIST",
                description=(
                    "The PneumoniaMNIST dataset is based on a prior collection of pediatric chest X-ray images. "
                    "Task: Binary classification between Normal and Pneumonia."
                ),
                available=False,
                num_classes=2,
                classes=["Normal", "Pneumonia"],
                image_size="28x28 (Grayscale/1-channel)",
                total_samples=0,
                splits=DatasetSplits(train=0, validation=0, test=0),
                class_distribution={},
                status_message="dataset_not_available: 'pneumoniamnist.npz' not found in data directory.",
            )

        if self._cached_info is not None:
            return self._cached_info

        try:
            # Load dataset archive
            data = np.load(self.dataset_path)

            train_count = len(data["train_images"])
            val_count = len(data["val_images"])
            test_count = len(data["test_images"])
            total_samples = train_count + val_count + test_count

            # Compute exact class distribution across all splits
            all_labels = np.concatenate(
                [data["train_labels"], data["val_labels"], data["test_labels"]]
            ).flatten()

            normal_count = int(np.sum(all_labels == 0))
            pneumonia_count = int(np.sum(all_labels == 1))

            h, w = data["train_images"].shape[1], data["train_images"].shape[2]
            image_size_str = f"{h}x{w} (Grayscale/1-channel)"

            info = DatasetInfoResponse(
                name="PneumoniaMNIST",
                description=(
                    "PneumoniaMNIST pediatric chest radiograph benchmark for binary classification "
                    "(Normal vs Pneumonia). Curated and standardized from MedMNIST v2."
                ),
                available=True,
                num_classes=2,
                classes=["Normal", "Pneumonia"],
                image_dimensions=f"{h}x{w}",
                image_size=image_size_str,
                channels=1,
                total_samples=total_samples,
                train_size=train_count,
                validation_size=val_count,
                test_size=test_count,
                splits=DatasetSplits(
                    train=train_count,
                    validation=val_count,
                    test=test_count,
                ),
                class_distribution={
                    "Normal": normal_count,
                    "Pneumonia": pneumonia_count,
                },
                status_message="PneumoniaMNIST dataset verified and loaded from disk archive.",
            )
            self._cached_info = info
            return info

        except Exception as exc:
            logger.error("Failed to load PneumoniaMNIST dataset archive: %s", exc)
            return DatasetInfoResponse(
                name="PneumoniaMNIST",
                description="PneumoniaMNIST pediatric chest X-ray benchmark",
                available=False,
                num_classes=2,
                classes=["Normal", "Pneumonia"],
                image_size="28x28",
                channels=1,
                total_samples=0,
                train_size=0,
                validation_size=0,
                test_size=0,
                splits=DatasetSplits(train=0, validation=0, test=0),
                class_distribution={},
                status_message=f"Error reading dataset archive: {str(exc)}",
            )

    def load_split(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images and labels for a specific partition ('train', 'val', or 'test').

        Raises:
            FileNotFoundError: If dataset archive is missing.
            ValueError: If split name is unrecognized.
        """
        if not self.is_available():
            raise FileNotFoundError(
                f"PneumoniaMNIST archive '{self.dataset_path}' not found."
            )

        valid_splits = {"train", "val", "validation", "test"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Allowed: {valid_splits}")

        data = np.load(self.dataset_path)
        prefix = "val" if split == "validation" else split

        images = data[f"{prefix}_images"]
        labels = data[f"{prefix}_labels"]
        return images, labels

    def get_sample_items(
        self,
        count_per_class: Optional[int] = None,
        split: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Extract real sample chest radiographs from the NPZ dataset as base64 PNGs."""
        if not self.is_available():
            return []

        import base64
        import io
        from PIL import Image

        samples: List[Dict[str, Any]] = []
        try:
            data = np.load(self.dataset_path)

            if split:
                # Specific split requested with pagination
                split_key = "val" if split.lower() in ("val", "validation") else split.lower()
                if f"{split_key}_images" not in data:
                    return []
                imgs = data[f"{split_key}_images"]
                lbls = data[f"{split_key}_labels"].flatten()

                start_idx = max(0, (page - 1) * page_size)
                end_idx = min(len(imgs), start_idx + page_size)

                for idx in range(start_idx, end_idx):
                    arr = imgs[idx]
                    lbl = int(lbls[idx])
                    class_name = "Pneumonia" if lbl == 1 else "Normal"
                    pil_img = Image.fromarray(arr).convert("RGB")
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                    samples.append({
                        "id": f"samp-{split_key}-{idx}",
                        "sample_index": int(idx),
                        "split": split_key,
                        "label": lbl,
                        "class_name": class_name,
                        "classLabel": class_name,
                        "imageUrl": b64_uri,
                        "dimensions": f"{arr.shape[0]}x{arr.shape[1]} (PneumoniaMNIST)",
                    })
                return samples

            # Default overview mode: extract count_per_class per split
            per_class = count_per_class if count_per_class is not None else 2
            for split_name in ["train", "val", "test"]:
                prefix = split_name
                imgs = data[f"{prefix}_images"]
                lbls = data[f"{prefix}_labels"].flatten()

                for class_val, class_name in [(0, "Normal"), (1, "Pneumonia")]:
                    indices = np.where(lbls == class_val)[0][:per_class]
                    for idx in indices:
                        arr = imgs[idx]
                        pil_img = Image.fromarray(arr).convert("RGB")
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        b64_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                        samples.append({
                            "id": f"samp-{split_name}-{class_name.lower()}-{idx}",
                            "sample_index": int(idx),
                            "split": split_name,
                            "label": int(class_val),
                            "class_name": class_name,
                            "classLabel": class_name,
                            "imageUrl": b64_uri,
                            "dimensions": f"{arr.shape[0]}x{arr.shape[1]} (PneumoniaMNIST)",
                        })
            return samples
        except Exception as exc:
            logger.warning("Could not extract real dataset samples: %s", exc)
            return []


dataset_service = DatasetService()

