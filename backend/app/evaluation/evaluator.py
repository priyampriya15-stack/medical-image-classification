import datetime
import json
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from app.core.config import settings
from app.evaluation.metrics import calculate_all_metrics
from app.evaluation.schemas import EvaluationResult
from app.models.registry import build_model, MODEL_SPECS
from app.services.dataset_service import dataset_service, PneumoniaMNISTDataset

logger = logging.getLogger(__name__)


def resolve_checkpoint_path(model_name: str) -> Path:
    """
    Locate the trained task-specific checkpoint for a model.
    Checks directory-nested (checkpoints/resnet50/best.pt) and flat paths (checkpoints/resnet50_best.pt).

    Raises:
        FileNotFoundError: If no valid task checkpoint exists on disk.
    """
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        raise ValueError(f"Unknown model architecture '{model_name}'.")

    checkpoint_filename = spec["default_checkpoint"]
    search_dirs = [
        settings.CHECKPOINT_DIR,
        settings.BASE_DIR / "checkpoints",
        Path.cwd() / "backend" / "checkpoints",
        Path.cwd() / "checkpoints",
        settings.BASE_DIR.parent / "backend" / "checkpoints",
    ]

    candidate_paths = []
    for d in search_dirs:
        candidate_paths.append(d / model_name / "best.pt")
        candidate_paths.append(d / checkpoint_filename)

    for path in candidate_paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path.resolve()

    raise FileNotFoundError(
        f"Model has not been trained yet. No task checkpoint found for '{model_name}' "
        f"in {settings.CHECKPOINT_DIR}."
    )


def load_model_for_evaluation(model_name: str, checkpoint_path: Path) -> nn.Module:
    """
    Load a trained PyTorch model on CPU for deterministic evaluation.

    Raises:
        ValueError: If checkpoint format or model architecture is invalid.
    """
    device = torch.device("cpu")
    logger.info("Loading evaluation checkpoint for %s from %s", model_name, checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint file '{checkpoint_path}': {exc}")

    # Validate checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_name" in checkpoint:
        chk_model_name = checkpoint["model_name"]
        if chk_model_name != model_name:
            raise ValueError(
                f"Checkpoint model mismatch: Expected '{model_name}', found '{chk_model_name}'."
            )
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Instantiate architecture cleanly without downloading ImageNet weights
    model, _ = build_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate_model_on_test_set(
    model_name: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> EvaluationResult:
    """
    Run full deterministic test-set evaluation for the specified architecture.

    Args:
        model_name: Target model name ('resnet50' or 'efficientnet_b0')
        progress_callback: Optional callback receiving (processed_samples, total_samples)

    Returns:
        EvaluationResult with verified metrics, confusion matrix, and ROC/PR curve points.
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model '{model_name}'. Allowed: {list(MODEL_SPECS.keys())}")

    if not dataset_service.is_available():
        raise RuntimeError("PneumoniaMNIST dataset archive is not available.")

    # 1. Resolve and verify checkpoint
    checkpoint_path = resolve_checkpoint_path(model_name)
    model = load_model_for_evaluation(model_name, checkpoint_path)

    # 2. Load held-out TEST split only
    test_images, test_labels = dataset_service.load_split("test")
    total_test_samples = len(test_images)

    # Deterministic normalization matching ImageNet expectations
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = PneumoniaMNISTDataset(
        test_images,
        test_labels,
        transform=test_transform,
        as_rgb=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 3. Inference loop
    all_targets = []
    all_preds = []
    all_pos_probs = []
    processed_count = 0

    device = torch.device("cpu")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # raw logits

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            # Class 1 is Pneumonia (positive class)
            pos_probs = probs[:, 1]

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_pos_probs.extend(pos_probs.cpu().numpy().tolist())

            processed_count += len(targets)
            if progress_callback:
                progress_callback(processed_count, total_test_samples)

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_scores = np.array(all_pos_probs, dtype=float)

    # 4. Compute verified diagnostic metrics
    metrics, conf_matrix, roc_curve, pr_curve = calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
    )

    # 5. Build and serialize result
    evaluated_at = datetime.datetime.utcnow().isoformat()
    result = EvaluationResult(
        model_name=model_name,
        display_name=MODEL_SPECS[model_name]["display_name"],
        dataset="PneumoniaMNIST",
        split="test",
        num_samples=total_test_samples,
        metrics=metrics,
        confusion_matrix=conf_matrix,
        roc_curve=roc_curve,
        precision_recall_curve=pr_curve,
        checkpoint_path=str(checkpoint_path),
        evaluated_at=evaluated_at,
    )

    # Save to backend/outputs/evaluation/{model_name}/results.json
    output_dir = settings.OUTPUT_DIR / "evaluation" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    with open(results_file, "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    logger.info(
        "Evaluation completed for %s on %d test samples. Saved to %s. Accuracy: %.4f, ROC-AUC: %.4f",
        model_name,
        total_test_samples,
        results_file,
        metrics.accuracy,
        metrics.roc_auc,
    )

    return result
