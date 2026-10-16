import datetime
import json
import logging
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional

from app.core.config import settings
from app.evaluation.evaluator import (
    evaluate_model_on_test_set,
    resolve_checkpoint_path,
)
from app.evaluation.schemas import (
    EvaluationJobStatus,
    EvaluationResult,
    ModelComparisonItem,
    ModelComparisonResponse,
)
from app.models.registry import MODEL_SPECS
from app.services.history_service import history_service

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    Coordinates asynchronous test-set evaluations and manages model comparisons.
    Ensures safe background execution and persistent result caching.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, EvaluationJobStatus] = {}
        self._results: Dict[str, EvaluationResult] = {}
        self._load_saved_results()

    def _load_saved_results(self) -> None:
        """Load any previously computed evaluation results from disk."""
        eval_base_dir = settings.OUTPUT_DIR / "evaluation"
        if not eval_base_dir.exists():
            return

        for model_name in MODEL_SPECS.keys():
            results_path = eval_base_dir / model_name / "results.json"
            if results_path.exists() and results_path.stat().st_size > 0:
                try:
                    with open(results_path, "r") as f:
                        data = json.load(f)
                        result = EvaluationResult(**data)
                        self._results[model_name] = result
                        logger.info("Loaded previous evaluation result for %s from %s", model_name, results_path)
                except Exception as exc:
                    logger.warning("Could not read evaluation file for %s: %s", model_name, exc)

    def get_job_status(self, job_id: str) -> Optional[EvaluationJobStatus]:
        """Retrieve evaluation job status by unique job_id."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_model_result(self, model_name: str) -> Optional[EvaluationResult]:
        """Retrieve the latest evaluated result for a given model."""
        with self._lock:
            return self._results.get(model_name)

    def start_evaluation(self, model_name: str) -> EvaluationJobStatus:
        """
        Schedule a test-set evaluation in the background.

        Raises:
            ValueError: If model_name is invalid.
            FileNotFoundError: If task checkpoint is missing.
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model architecture '{model_name}'. Allowed: {list(MODEL_SPECS.keys())}")

        # Pre-verify checkpoint exists to fail fast with a clean error
        resolve_checkpoint_path(model_name)

        with self._lock:
            # Check if job is currently evaluating
            for job in self._jobs.values():
                if job.model_name == model_name and job.status in ("queued", "evaluating"):
                    return job

            job_id = f"eval-{model_name}-{int(time.time())}"
            job_status = EvaluationJobStatus(
                job_id=job_id,
                model_name=model_name,
                status="queued",
                progress=0.0,
                processed_samples=0,
                total_samples=0,
                elapsed_time=0.0,
            )
            self._jobs[job_id] = job_status

        # Spawn worker thread
        thread = threading.Thread(
            target=self._run_evaluation_worker,
            args=(job_id, model_name),
            daemon=True,
            name=f"eval-worker-{model_name}",
        )
        thread.start()

        return job_status

    def _run_evaluation_worker(self, job_id: str, model_name: str) -> None:
        """Worker thread executing test-set inference and recording live progress."""
        start_time = time.time()
        logger.info("Starting test-set evaluation worker for %s (job=%s)", model_name, job_id)

        with self._lock:
            job = self._jobs[job_id]
            job.status = "evaluating"

        def progress_callback(processed: int, total: int) -> None:
            with self._lock:
                current_job = self._jobs.get(job_id)
                if current_job:
                    current_job.processed_samples = processed
                    current_job.total_samples = total
                    current_job.progress = round((processed / total) * 100, 1) if total > 0 else 0.0
                    current_job.elapsed_time = round(time.time() - start_time, 2)

        try:
            result = evaluate_model_on_test_set(
                model_name=model_name,
                progress_callback=progress_callback,
            )

            total_duration = time.time() - start_time

            with self._lock:
                completed_job = self._jobs[job_id]
                completed_job.status = "completed"
                completed_job.progress = 100.0
                completed_job.processed_samples = result.num_samples
                completed_job.total_samples = result.num_samples
                completed_job.elapsed_time = round(total_duration, 2)
                completed_job.result = result

                self._results[model_name] = result

            try:
                display_name = MODEL_SPECS.get(model_name, {}).get("display_name", model_name)
                history_service.record_activity(
                    event_type="EVALUATION_COMPLETED",
                    model_name=model_name,
                    description=f"Test-set evaluation finished for {display_name} (ROC-AUC: {result.metrics.roc_auc:.4f}, Accuracy: {result.metrics.accuracy * 100:.2f}%).",
                    metadata={
                        "roc_auc": result.metrics.roc_auc,
                        "accuracy": result.metrics.accuracy,
                        "f1": result.metrics.f1,
                    },
                )
            except Exception as h_err:
                logger.warning("Could not record evaluation event: %s", h_err)

            logger.info("Evaluation job %s for %s completed successfully in %.2fs.", job_id, model_name, total_duration)

        except Exception as exc:
            logger.exception("Evaluation failed for job %s (%s): %s", job_id, model_name, exc)
            with self._lock:
                failed_job = self._jobs[job_id]
                failed_job.status = "failed"
                failed_job.error = str(exc)
                failed_job.elapsed_time = round(time.time() - start_time, 2)

    def get_comparison(self) -> ModelComparisonResponse:
        """
        Generate comparison table across all models.
        Uses ROC-AUC as primary ranking metric.
        If a model is not evaluated, fields are None and marked 'Not evaluated'.
        """
        items: List[ModelComparisonItem] = []
        best_model: Optional[str] = None
        best_roc_auc: float = -1.0

        with self._lock:
            for model_name, spec in MODEL_SPECS.items():
                res = self._results.get(model_name)
                if res is not None:
                    items.append(
                        ModelComparisonItem(
                            model_name=model_name,
                            display_name=spec["display_name"],
                            evaluated=True,
                            accuracy=res.metrics.accuracy,
                            precision=res.metrics.precision,
                            recall=res.metrics.recall,
                            sensitivity=res.metrics.sensitivity,
                            specificity=res.metrics.specificity,
                            f1=res.metrics.f1,
                            roc_auc=res.metrics.roc_auc,
                            evaluated_at=res.evaluated_at,
                            status_text="Evaluated",
                        )
                    )
                    # ROC-AUC is preferred primary metric for clinical diagnostic ranking
                    if res.metrics.roc_auc > best_roc_auc:
                        best_roc_auc = res.metrics.roc_auc
                        best_model = model_name
                else:
                    items.append(
                        ModelComparisonItem(
                            model_name=model_name,
                            display_name=spec["display_name"],
                            evaluated=False,
                            status_text="Not evaluated",
                        )
                    )

        summary = (
            f"Comparison ranked by ROC-AUC primary metric. Top model: {MODEL_SPECS.get(best_model, {}).get('display_name', 'None')} "
            f"(ROC-AUC: {best_roc_auc:.4f})"
            if best_model
            else "No models have completed test-set evaluation yet."
        )

        return ModelComparisonResponse(
            models=items,
            primary_metric="roc_auc",
            best_model=best_model,
            comparison_summary=summary,
        )


evaluation_manager = EvaluationManager()
