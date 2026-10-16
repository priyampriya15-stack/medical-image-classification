import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


class ActivityEvent(BaseModel):
    id: str
    timestamp: str
    event_type: str  # TRAINING_COMPLETED, EVALUATION_COMPLETED, PREDICTION_COMPLETED, GRADCAM_GENERATED
    model_name: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    id: str
    timestamp: str
    model_name: str
    predicted_class: str
    confidence: float
    inference_time_ms: float


class HistoryService:
    """
    Manages lightweight, privacy-preserving research activity and prediction history.
    Stores metadata only: never permanently stores radiograph pixels or PII.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._storage_dir = storage_dir or (settings.OUTPUT_DIR / "history")
        self._activity_file = self._storage_dir / "activities.json"
        self._prediction_file = self._storage_dir / "predictions.json"
        self._activities: List[ActivityEvent] = []
        self._predictions: List[PredictionRecord] = []

        self._ensure_storage()
        self._load_persisted()
        self._seed_from_existing_outputs()

    def _ensure_storage(self) -> None:
        try:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Could not create history directory: %s", exc)

    def _load_persisted(self) -> None:
        with self._lock:
            if self._activity_file.exists():
                try:
                    with open(self._activity_file, "r") as f:
                        raw = json.load(f)
                        self._activities = [ActivityEvent(**item) for item in raw]
                except Exception as exc:
                    logger.warning("Failed to load activity history: %s", exc)

            if self._prediction_file.exists():
                try:
                    with open(self._prediction_file, "r") as f:
                        raw = json.load(f)
                        self._predictions = [PredictionRecord(**item) for item in raw]
                except Exception as exc:
                    logger.warning("Failed to load prediction history: %s", exc)

    def _save_activities(self) -> None:
        try:
            with open(self._activity_file, "w") as f:
                json.dump([item.model_dump() for item in self._activities], f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist activity history: %s", exc)

    def _save_predictions(self) -> None:
        try:
            with open(self._prediction_file, "w") as f:
                json.dump([item.model_dump() for item in self._predictions], f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist prediction history: %s", exc)

    def _seed_from_existing_outputs(self) -> None:
        """Seed real events from existing disk artifacts if activities are empty."""
        with self._lock:
            if self._activities:
                return

            seeded: List[ActivityEvent] = []

            # Check training histories
            for model_id, display_name in [("resnet50", "ResNet50"), ("efficientnet_b0", "EfficientNetB0")]:
                train_file = settings.OUTPUT_DIR / f"{model_id}_training_history.json"
                if train_file.exists():
                    try:
                        with open(train_file, "r") as f:
                            t_data = json.load(f)
                        best_acc = t_data.get("best_val_acc", 0.0)
                        epochs = t_data.get("total_epochs", 5)
                        seeded.append(
                            ActivityEvent(
                                id=f"act-train-{model_id}",
                                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                event_type="TRAINING_COMPLETED",
                                model_name=model_id,
                                description=f"Transfer learning training completed for {display_name} ({epochs} epochs, Best Val Acc: {best_acc * 100:.2f}%).",
                                metadata={"best_val_acc": best_acc, "epochs": epochs},
                            )
                        )
                    except Exception as exc:
                        logger.warning("Could not seed training event for %s: %s", model_id, exc)

            # Check evaluation results
            for model_id, display_name in [("resnet50", "ResNet50"), ("efficientnet_b0", "EfficientNetB0")]:
                eval_file = settings.OUTPUT_DIR / "evaluation" / model_id / "results.json"
                if eval_file.exists():
                    try:
                        with open(eval_file, "r") as f:
                            e_data = json.load(f)
                        metrics = e_data.get("metrics", {})
                        roc_auc = metrics.get("roc_auc", 0.0)
                        acc = metrics.get("accuracy", 0.0)
                        seeded.append(
                            ActivityEvent(
                                id=f"act-eval-{model_id}",
                                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                event_type="EVALUATION_COMPLETED",
                                model_name=model_id,
                                description=f"Test-set evaluation finished for {display_name} (ROC-AUC: {roc_auc:.4f}, Accuracy: {acc * 100:.2f}%).",
                                metadata={"roc_auc": roc_auc, "accuracy": acc},
                            )
                        )
                    except Exception as exc:
                        logger.warning("Could not seed evaluation event for %s: %s", model_id, exc)

            if seeded:
                self._activities.extend(seeded)
                self._save_activities()

    def record_activity(
        self,
        event_type: str,
        model_name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActivityEvent:
        event = ActivityEvent(
            id=f"act-{int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)}",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            event_type=event_type,
            model_name=model_name,
            description=description,
            metadata=metadata or {},
        )
        with self._lock:
            # Prepend to keep latest first, cap at 100
            self._activities.insert(0, event)
            self._activities = self._activities[:100]
            self._save_activities()
        return event

    def record_prediction(
        self,
        model_name: str,
        predicted_class: str,
        confidence: float,
        inference_time_ms: float,
    ) -> PredictionRecord:
        record = PredictionRecord(
            id=f"pred-{int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)}",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_name=model_name,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            inference_time_ms=round(inference_time_ms, 2),
        )
        with self._lock:
            self._predictions.insert(0, record)
            self._predictions = self._predictions[:100]
            self._save_predictions()

        # Also log activity event
        self.record_activity(
            event_type="PREDICTION_COMPLETED",
            model_name=model_name,
            description=f"Predicted {predicted_class} with {confidence * 100:.2f}% confidence ({inference_time_ms:.1f}ms) on CPU.",
            metadata={
                "predicted_class": predicted_class,
                "confidence": confidence,
                "inference_time_ms": inference_time_ms,
            },
        )
        return record

    def get_activities(self, limit: int = 20) -> List[ActivityEvent]:
        with self._lock:
            return list(self._activities[:limit])

    def get_predictions(self, limit: int = 20) -> List[PredictionRecord]:
        with self._lock:
            return list(self._predictions[:limit])


history_service = HistoryService()
