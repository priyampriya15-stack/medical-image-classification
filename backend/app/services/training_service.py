import datetime
import json
import logging
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from app.core.config import settings
from app.models.registry import build_model, MODEL_SPECS
from app.schemas.training import EpochMetric, TrainingConfig, TrainingJobStatus
from app.services.dataset_service import dataset_service, PneumoniaMNISTDataset
from app.services.history_service import history_service

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Manages asynchronous background training for transfer learning architectures
    (ResNet50 & EfficientNetB0) on PneumoniaMNIST.
    Ensures safe single-job execution on CPU to prevent CPU saturation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, TrainingJobStatus] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._initialize_from_disk()

    def _initialize_from_disk(self) -> None:
        """Scan checkpoints and output directory to restore state from previous runs."""
        for model_name, spec in MODEL_SPECS.items():
            checkpoint_filename = spec["default_checkpoint"]
            search_dirs = [
                settings.CHECKPOINT_DIR,
                settings.BASE_DIR / "checkpoints",
                Path.cwd() / "backend" / "checkpoints",
                Path.cwd() / "checkpoints",
                settings.BASE_DIR.parent / "backend" / "checkpoints",
            ]
            checkpoint_file = None
            for d in search_dirs:
                p1 = d / model_name / "best.pt"
                p2 = d / checkpoint_filename
                if p1.exists() and p1.stat().st_size > 0:
                    checkpoint_file = p1
                    break
                elif p2.exists() and p2.stat().st_size > 0:
                    checkpoint_file = p2
                    break

            history_file = settings.OUTPUT_DIR / f"{model_name}_training_history.json"

            if checkpoint_file is not None:
                # Load saved history if available
                history: List[EpochMetric] = []
                best_acc: Optional[float] = None
                best_ep: Optional[int] = None
                total_duration: float = 0.0
                config_dict: Dict[str, Any] = {}

                if history_file.exists():
                    try:
                        with open(history_file, "r") as f:
                            saved_data = json.load(f)
                            history = [EpochMetric(**m) for m in saved_data.get("history", [])]
                            best_acc = saved_data.get("best_val_acc")
                            best_ep = saved_data.get("best_epoch")
                            total_duration = float(saved_data.get("total_duration_seconds", 0.0))
                            config_dict.update(saved_data.get("config", {}))
                            if "total_epochs" in saved_data:
                                config_dict["epochs"] = saved_data["total_epochs"]
                    except Exception as err:
                        logger.warning("Could not read history for %s: %s", model_name, err)

                actual_total_epochs = config_dict.get("epochs") or len(history) or 0
                last_train_loss = history[-1].train_loss if history else None
                last_val_loss = history[-1].val_loss if history else None

                self._jobs[model_name] = TrainingJobStatus(
                    job_id=f"job-{model_name}",
                    model_name=model_name,
                    display_name=spec["display_name"],
                    status="completed",
                    current_epoch=len(history),
                    total_epochs=actual_total_epochs,
                    progress_percent=100.0 if actual_total_epochs > 0 else 0.0,
                    best_val_acc=best_acc,
                    best_epoch=best_ep,
                    current_train_loss=last_train_loss,
                    current_val_loss=last_val_loss,
                    learning_rate=config_dict.get("learning_rate", 0.001),
                    batch_size=config_dict.get("batch_size", 32),
                    optimizer=config_dict.get("optimizer", "AdamW"),
                    fine_tune=config_dict.get("fine_tune", True),
                    history=history,
                    elapsed_seconds=total_duration,
                    checkpoint_saved=True,
                    checkpoint_path=str(checkpoint_file),
                )
            else:
                self._jobs[model_name] = TrainingJobStatus(
                    job_id=f"job-{model_name}",
                    model_name=model_name,
                    display_name=spec["display_name"],
                    status="idle",
                    current_epoch=0,
                    total_epochs=0,
                    progress_percent=0.0,
                    best_val_acc=None,
                    best_epoch=None,
                    current_train_loss=None,
                    current_val_loss=None,
                    learning_rate=0.001,
                    batch_size=32,
                    optimizer="AdamW",
                    fine_tune=True,
                    history=[],
                    elapsed_seconds=0.0,
                    checkpoint_saved=False,
                    checkpoint_path=None,
                )


    def is_any_training_running(self) -> bool:
        """Check if any model is currently undergoing active training."""
        with self._lock:
            return any(job.status in ("running", "queued") for job in self._jobs.values())

    def get_job_status(self, model_name: str) -> Optional[TrainingJobStatus]:
        """Get status of training job for a given model."""
        with self._lock:
            return self._jobs.get(model_name)

    def get_all_jobs(self) -> List[TrainingJobStatus]:
        """Get status of all models' training jobs."""
        with self._lock:
            return list(self._jobs.values())

    def stop_training(self, model_name: str) -> bool:
        """Signal a currently running training job to halt."""
        with self._lock:
            stop_event = self._stop_events.get(model_name)
            job = self._jobs.get(model_name)
            if stop_event and job and job.status == "running":
                stop_event.set()
                job.status = "stopped"
                logger.info("Training stop signaled for model %s", model_name)
                return True
        return False

    def start_training(self, config: TrainingConfig) -> TrainingJobStatus:
        """
        Launch background training for the requested model configuration.

        Raises:
            ValueError: If model is not registered.
            RuntimeError: If dataset is missing or another job is currently active.
        """
        model_name = config.model_name
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Invalid model name '{model_name}'. Allowed: {list(MODEL_SPECS.keys())}")

        if not dataset_service.is_available():
            raise RuntimeError("Cannot start training: PneumoniaMNIST dataset archive is not available.")

        with self._lock:
            if any(job.status in ("running", "queued") for job in self._jobs.values()):
                raise RuntimeError(
                    "A training job is already actively running. Please wait for completion or stop it first."
                )

            stop_event = threading.Event()
            self._stop_events[model_name] = stop_event

            display_name = MODEL_SPECS[model_name]["display_name"]
            job_status = TrainingJobStatus(
                job_id=f"job-{model_name}-{int(time.time())}",
                model_name=model_name,
                display_name=display_name,
                status="queued",
                current_epoch=0,
                total_epochs=config.epochs,
                progress_percent=0.0,
                best_val_acc=None,
                best_epoch=None,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                optimizer=config.optimizer,
                fine_tune=config.fine_tune,
                history=[],
                start_time=datetime.datetime.utcnow().isoformat(),
                checkpoint_saved=False,
                checkpoint_path=None,
            )
            self._jobs[model_name] = job_status

        # Spawn worker thread
        thread = threading.Thread(
            target=self._run_training_loop,
            args=(model_name, config, stop_event),
            daemon=True,
            name=f"train-worker-{model_name}",
        )
        self._threads[model_name] = thread
        thread.start()

        return job_status

    def _run_training_loop(
        self,
        model_name: str,
        config: TrainingConfig,
        stop_event: threading.Event,
    ) -> None:
        """Worker thread executing the PyTorch training and validation cycles on CPU."""
        device = settings.get_device()
        start_time = time.time()
        logger.info(
            "Starting training worker for %s on device=%s, epochs=%d, batch_size=%d, lr=%f",
            model_name,
            device,
            config.epochs,
            config.batch_size,
            config.learning_rate,
        )

        with self._lock:
            self._jobs[model_name].status = "running"

        try:
            # 1. Load splits
            train_images, train_labels = dataset_service.load_split("train")
            val_images, val_labels = dataset_service.load_split("val")

            # Standard ImageNet normalization for transfer learning
            data_transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            train_ds = PneumoniaMNISTDataset(train_images, train_labels, transform=data_transform, as_rgb=True)
            val_ds = PneumoniaMNISTDataset(val_images, val_labels, transform=data_transform, as_rgb=True)

            train_loader = DataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            # 2. Build model architecture
            # If fine_tune is True, unfreeze backbone; otherwise freeze backbone
            model, weights_loaded = build_model(
                model_name=model_name,
                num_classes=2,
                pretrained=True,
                freeze_backbone=not config.fine_tune,
            )
            model = model.to(device)

            # 3. Configure optimizer
            if config.fine_tune:
                # Differential learning rates: conservative backbone lr, higher head lr
                backbone_params = []
                head_params = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "fc" in name or "classifier" in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

                param_groups = [
                    {"params": backbone_params, "lr": config.learning_rate * 0.1},
                    {"params": head_params, "lr": config.learning_rate},
                ]
            else:
                param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.learning_rate}]

            if config.optimizer.lower() == "adam":
                optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
            else:
                optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            best_epoch = 0
            best_checkpoint_path = settings.CHECKPOINT_DIR / MODEL_SPECS[model_name]["default_checkpoint"]
            history_list: List[EpochMetric] = []

            # 4. Epoch loop
            for epoch in range(1, config.epochs + 1):
                if stop_event.is_set():
                    logger.info("Training for %s interrupted by user stop event.", model_name)
                    with self._lock:
                        self._jobs[model_name].status = "stopped"
                    return

                epoch_start = time.time()

                # Train phase
                model.train()
                running_train_loss = 0.0
                train_correct = 0
                train_total = 0

                for inputs, targets in train_loader:
                    if stop_event.is_set():
                        return
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == targets).item()
                    train_total += targets.size(0)

                epoch_train_loss = running_train_loss / train_total if train_total > 0 else 0.0
                epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0

                # Validation phase
                model.eval()
                running_val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        running_val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == targets).item()
                        val_total += targets.size(0)

                epoch_val_loss = running_val_loss / val_total if val_total > 0 else 0.0
                epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
                epoch_dur = time.time() - epoch_start

                metric = EpochMetric(
                    epoch=epoch,
                    train_loss=round(epoch_train_loss, 4),
                    train_acc=round(epoch_train_acc, 4),
                    val_loss=round(epoch_val_loss, 4),
                    val_acc=round(epoch_val_acc, 4),
                    epoch_duration_seconds=round(epoch_dur, 2),
                )
                history_list.append(metric)

                # Check for best validation accuracy
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_epoch = epoch
                    # Save best checkpoint
                    torch.save(
                        {
                            "model_name": model_name,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": epoch_val_acc,
                            "val_loss": epoch_val_loss,
                            "train_acc": epoch_train_acc,
                            "train_loss": epoch_train_loss,
                            "config": config.model_dump(),
                        },
                        best_checkpoint_path,
                    )
                    logger.info(
                        "New best checkpoint saved for %s (Epoch %d, Val Acc: %.4f)",
                        model_name,
                        epoch,
                        epoch_val_acc,
                    )

                # Update job status
                progress = round((epoch / config.epochs) * 100, 1)
                elapsed = time.time() - start_time

                with self._lock:
                    job = self._jobs[model_name]
                    job.current_epoch = epoch
                    job.progress_percent = progress
                    job.best_val_acc = round(best_val_acc, 4)
                    job.best_epoch = best_epoch
                    job.current_train_loss = round(epoch_train_loss, 4)
                    job.current_val_loss = round(epoch_val_loss, 4)
                    job.history = list(history_list)
                    job.elapsed_seconds = round(elapsed, 1)
                    if best_checkpoint_path.exists():
                        job.checkpoint_saved = True
                        job.checkpoint_path = str(best_checkpoint_path)

                logger.info(
                    "[%s] Epoch %d/%d - Train Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Acc: %.4f (Best: %.4f @ Ep %d)",
                    model_name,
                    epoch,
                    config.epochs,
                    epoch_train_loss,
                    epoch_train_acc,
                    epoch_val_loss,
                    epoch_val_acc,
                    best_val_acc,
                    best_epoch,
                )

            # 5. Training successfully finished
            total_time = time.time() - start_time
            history_file = settings.OUTPUT_DIR / f"{model_name}_training_history.json"
            try:
                with open(history_file, "w") as f:
                    json.dump(
                        {
                            "model_name": model_name,
                            "total_epochs": config.epochs,
                            "best_val_acc": round(best_val_acc, 4),
                            "best_epoch": best_epoch,
                            "total_duration_seconds": round(total_time, 2),
                            "config": config.model_dump(),
                            "history": [m.model_dump() for m in history_list],
                        },
                        f,
                        indent=2,
                    )
            except Exception as exc:
                logger.warning("Could not persist history JSON for %s: %s", model_name, exc)

            with self._lock:
                final_job = self._jobs[model_name]
                final_job.status = "completed"
                final_job.progress_percent = 100.0
                final_job.end_time = datetime.datetime.utcnow().isoformat()
                final_job.elapsed_seconds = round(total_time, 1)
                final_job.checkpoint_saved = best_checkpoint_path.exists()
                final_job.checkpoint_path = str(best_checkpoint_path) if best_checkpoint_path.exists() else None

            try:
                display_name = MODEL_SPECS.get(model_name, {}).get("display_name", model_name)
                history_service.record_activity(
                    event_type="TRAINING_COMPLETED",
                    model_name=model_name,
                    description=f"Transfer learning training completed for {display_name} ({config.epochs} epochs, Best Val Acc: {best_val_acc * 100:.2f}%).",
                    metadata={"best_val_acc": best_val_acc, "epochs": config.epochs},
                )
            except Exception as h_err:
                logger.warning("Could not record training event: %s", h_err)

            logger.info("Training cycle for %s completed successfully in %.1fs.", model_name, total_time)

        except Exception as exc:
            logger.exception("Error during training loop for %s: %s", model_name, exc)
            with self._lock:
                failed_job = self._jobs[model_name]
                failed_job.status = "failed"
                failed_job.error_message = str(exc)
                failed_job.end_time = datetime.datetime.utcnow().isoformat()


training_service = TrainingService()
