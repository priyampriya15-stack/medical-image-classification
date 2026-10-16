import csv
import datetime
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.analytics.schemas import (
    AnalyticsSummaryResponse,
    BestModelSchema,
    ModelAnalyticsItem,
    ProjectInfoSchema,
)
from app.analytics.summary import determine_best_model, generate_research_interpretation
from app.core.config import settings
from app.evaluation.manager import evaluation_manager
from app.models.registry import MODEL_SPECS
from app.services.dataset_service import dataset_service
from app.services.model_service import model_service
from app.services.training_service import training_service

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Unified analytics service aggregating data across training runs,
    model checkpoints, evaluation benchmarks, and research telemetry.
    Never fabricates metrics; reuses stored execution artifacts.
    """

    def get_model_training_history(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored training history from disk or active service."""
        # 1. Check disk output first
        history_path = settings.OUTPUT_DIR / f"{model_name}_training_history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not read training history file for %s: %s", model_name, exc)

        # 2. Check active training manager job
        job = training_service.get_job_status(model_name)
        if job and job.history:
            return {
                "model_name": model_name,
                "total_epochs": job.total_epochs,
                "best_val_acc": job.best_val_acc,
                "best_epoch": job.best_epoch,
                "total_duration_seconds": job.elapsed_seconds,
                "history": [m.model_dump() for m in job.history],
            }

        return None

    def get_confusion_matrices(self) -> Dict[str, Any]:
        """Retrieve real confusion matrices for ResNet50 and EfficientNetB0."""
        result: Dict[str, Any] = {"classes": ["Normal", "Pneumonia"]}
        for model_name in MODEL_SPECS.keys():
            eval_res = evaluation_manager.get_model_result(model_name)
            if eval_res and eval_res.confusion_matrix:
                cm = eval_res.confusion_matrix
                result[model_name] = {
                    "matrix": cm.raw_matrix,
                    "raw_matrix": cm.raw_matrix,
                    "true_negative": cm.true_negative,
                    "false_positive": cm.false_positive,
                    "false_negative": cm.false_negative,
                    "true_positive": cm.true_positive,
                }
            else:
                result[model_name] = None
        return result

    def get_roc_data(self) -> Dict[str, Any]:
        """Retrieve actual ROC curve points and AUC for all models."""
        import math
        result: Dict[str, Any] = {}
        for model_name in MODEL_SPECS.keys():
            eval_res = evaluation_manager.get_model_result(model_name)
            if eval_res and eval_res.roc_curve:
                roc = eval_res.roc_curve
                sanitized_thresholds = [
                    1.0 if (t is None or math.isinf(t) or math.isnan(t)) else float(t)
                    for t in (roc.thresholds or [])
                ]
                result[model_name] = {
                    "false_positive_rate": roc.fpr,
                    "true_positive_rate": roc.tpr,
                    "fpr": roc.fpr,
                    "tpr": roc.tpr,
                    "thresholds": sanitized_thresholds,
                    "roc_auc": eval_res.metrics.roc_auc,
                }
            else:
                result[model_name] = None
        return result

    def get_precision_recall_data(self) -> Dict[str, Any]:
        """Retrieve actual Precision-Recall curve points and average precision for all models."""
        import math
        from sklearn.metrics import auc
        result: Dict[str, Any] = {}
        for model_name in MODEL_SPECS.keys():
            eval_res = evaluation_manager.get_model_result(model_name)
            if eval_res and eval_res.precision_recall_curve:
                pr = eval_res.precision_recall_curve
                avg_prec = None
                try:
                    if len(pr.recall) > 1 and len(pr.precision) > 1:
                        avg_prec = round(float(auc(pr.recall, pr.precision)), 4)
                except Exception as exc:
                    logger.warning("Could not calculate AUC for PR curve (%s): %s", model_name, exc)
                sanitized_pr_thresh = [
                    float(t) for t in (pr.thresholds or [])
                    if not (t is None or math.isinf(t) or math.isnan(t))
                ]
                result[model_name] = {
                    "precision": pr.precision,
                    "recall": pr.recall,
                    "thresholds": sanitized_pr_thresh,
                    "average_precision": avg_prec,
                }
            else:
                result[model_name] = None
        return result

    def get_all_training_histories(self) -> Dict[str, Any]:
        """Retrieve actual training histories containing loss, accuracy, and lr per epoch."""
        result: Dict[str, Any] = {}
        for model_name in MODEL_SPECS.keys():
            hist_data = self.get_model_training_history(model_name)
            if hist_data and "history" in hist_data:
                config = hist_data.get("config", {})
                lr = config.get("learning_rate", 0.001)
                epochs = []
                for item in hist_data["history"]:
                    t_loss = item.get("train_loss")
                    v_loss = item.get("val_loss")
                    t_acc = item.get("train_accuracy", item.get("train_acc"))
                    v_acc = item.get("validation_accuracy", item.get("val_acc"))
                    epochs.append({
                        "epoch": item.get("epoch"),
                        "train_loss": t_loss,
                        "validation_loss": v_loss,
                        "train_accuracy": t_acc,
                        "validation_accuracy": v_acc,
                        "train_acc": t_acc,
                        "val_acc": v_acc,
                        "learning_rate": item.get("learning_rate", lr),
                        "epoch_duration_seconds": item.get("epoch_duration_seconds"),
                    })
                result[model_name] = epochs
            else:
                result[model_name] = None
        return result

    def get_summary(self) -> AnalyticsSummaryResponse:
        """
        Assemble the unified research analytics summary for all models.
        Strictly reuses existing stored results without re-running training or evaluation.
        """
        items: List[ModelAnalyticsItem] = []

        for model_name, spec in MODEL_SPECS.items():
            model_info = model_service.get_model(model_name)
            eval_res = evaluation_manager.get_model_result(model_name)
            train_hist = self.get_model_training_history(model_name)

            training_status = "completed" if train_hist else "not_started"
            evaluation_status = "completed" if eval_res else "not_evaluated"

            item = ModelAnalyticsItem(
                name=model_name,
                display_name=spec["display_name"],
                training_status=training_status,
                evaluation_status=evaluation_status,
                total_parameters=model_info.total_parameters if model_info else None,
                trainable_parameters=model_info.trainable_parameters if model_info else None,
            )

            if eval_res:
                item.accuracy = eval_res.metrics.accuracy
                item.precision = eval_res.metrics.precision
                item.recall = eval_res.metrics.recall
                item.sensitivity = eval_res.metrics.sensitivity
                item.specificity = eval_res.metrics.specificity
                item.f1 = eval_res.metrics.f1
                item.roc_auc = eval_res.metrics.roc_auc
                item.test_samples = eval_res.num_samples
                item.evaluated_at = eval_res.evaluated_at
                item.confusion_matrix = eval_res.confusion_matrix.model_dump()

            items.append(item)

        best_model = determine_best_model(items)
        interpretation = generate_research_interpretation(items)

        return AnalyticsSummaryResponse(
            project=ProjectInfoSchema(),
            models=items,
            best_model=best_model,
            research_interpretation=interpretation,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def get_full_export_data(self) -> Dict[str, Any]:
        """
        Gathers comprehensive research results for JSON export.
        Includes dataset metadata, training convergence, and test evaluations.
        """
        summary = self.get_summary()
        d_info = dataset_service.get_dataset_info()

        models_data = {}
        for m_item in summary.models:
            m_name = m_item.name
            t_hist = self.get_model_training_history(m_name)
            e_res = evaluation_manager.get_model_result(m_name)

            models_data[m_name] = {
                "display_name": m_item.display_name,
                "parameters": {
                    "total": m_item.total_parameters,
                    "trainable": m_item.trainable_parameters,
                },
                "training": t_hist or {"status": m_item.training_status},
                "evaluation": e_res.model_dump() if e_res else {"status": m_item.evaluation_status},
            }

        return {
            "project": summary.project.model_dump(),
            "dataset": d_info.model_dump(),
            "summary": summary.model_dump(),
            "models_detail": models_data,
            "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "system_environment": {
                "hardware_device": "CPU",
                "framework": "PyTorch 2.14.1+cpu",
                "backend": "FastAPI",
            },
            "disclaimer": summary.disclaimer,
        }

    def get_csv_export(self) -> str:
        """
        Generates CSV format of model test evaluation metrics.
        Columns: model,accuracy,precision,recall,sensitivity,specificity,f1,roc_auc,test_samples
        """
        summary = self.get_summary()
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "model",
            "accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "test_samples",
        ])

        for m in summary.models:
            if m.evaluation_status == "completed":
                writer.writerow([
                    m.name,
                    f"{m.accuracy:.4f}" if m.accuracy is not None else "",
                    f"{m.precision:.4f}" if m.precision is not None else "",
                    f"{m.recall:.4f}" if m.recall is not None else "",
                    f"{m.sensitivity:.4f}" if m.sensitivity is not None else "",
                    f"{m.specificity:.4f}" if m.specificity is not None else "",
                    f"{m.f1:.4f}" if m.f1 is not None else "",
                    f"{m.roc_auc:.4f}" if m.roc_auc is not None else "",
                    m.test_samples or "",
                ])
            else:
                writer.writerow([m.name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

        return output.getvalue()

    def generate_pdf_report(self) -> bytes:
        """
        Generates a high-quality, professional academic research report PDF using ReportLab.
        Contains project metadata, training summaries, comparative test evaluation,
        confusion matrices, and research disclaimers.
        """
        summary = self.get_summary()
        d_info = dataset_service.get_dataset_info()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=14,
        )
        h2_style = ParagraphStyle(
            "ReportH2",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )
        body_bold = ParagraphStyle(
            "ReportBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold",
        )
        table_cell = ParagraphStyle(
            "ReportTableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1E293B"),
        )
        table_cell_bold = ParagraphStyle(
            "ReportTableCellBold",
            parent=table_cell,
            fontName="Helvetica-Bold",
        )
        disclaimer_style = ParagraphStyle(
            "ReportDisclaimer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#78350F"),
        )

        story = []

        # 1. Header Section
        story.append(Paragraph("Medical Image Classification Research Report", title_style))
        story.append(
            Paragraph(
                "A Comparative Study of ResNet50 and EfficientNetB0 on PneumoniaMNIST &bull; Academic Deep Learning",
                subtitle_style,
            )
        )
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=12))

        # 2. Project & Dataset Overview
        story.append(Paragraph("1. Project & Dataset Overview", h2_style))
        overview_data = [
            [
                Paragraph("<b>Investigation Title:</b>", table_cell),
                Paragraph("Transfer Learning Benchmark on PneumoniaMNIST", table_cell),
                Paragraph("<b>Execution Device:</b>", table_cell),
                Paragraph("CPU (PyTorch 2.14.1)", table_cell),
            ],
            [
                Paragraph("<b>Benchmark Dataset:</b>", table_cell),
                Paragraph(f"{d_info.name} (Total: {d_info.total_samples:,} images)", table_cell),
                Paragraph("<b>Target Classes:</b>", table_cell),
                Paragraph("Binary (0: Normal, 1: Pneumonia)", table_cell),
            ],
            [
                Paragraph("<b>Dataset Splits:</b>", table_cell),
                Paragraph(
                    f"Train: {d_info.splits.train:,} | Val: {d_info.splits.validation:,} | Test: {d_info.splits.test:,}",
                    table_cell,
                ),
                Paragraph("<b>Generated:</b>", table_cell),
                Paragraph(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), table_cell),
            ],
        ]
        t_overview = Table(overview_data, colWidths=[110, 160, 110, 160])
        t_overview.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(t_overview)
        story.append(Spacer(1, 10))

        # 3. Model Architecture Specifications
        story.append(Paragraph("2. Deep Learning Architectures", h2_style))
        arch_data = [
            [
                Paragraph("<b>Architecture</b>", table_cell_bold),
                Paragraph("<b>Total Parameters</b>", table_cell_bold),
                Paragraph("<b>Trainable (Fine-Tuning)</b>", table_cell_bold),
                Paragraph("<b>Training Status</b>", table_cell_bold),
                Paragraph("<b>Evaluation Status</b>", table_cell_bold),
            ]
        ]
        for m in summary.models:
            arch_data.append([
                Paragraph(f"<b>{m.display_name}</b>", table_cell),
                Paragraph(f"{m.total_parameters / 1e6:.1f}M" if m.total_parameters else "N/A", table_cell),
                Paragraph(f"{m.trainable_parameters:,}" if m.trainable_parameters else "N/A", table_cell),
                Paragraph(m.training_status.capitalize(), table_cell),
                Paragraph(m.evaluation_status.capitalize(), table_cell),
            ])

        t_arch = Table(arch_data, colWidths=[120, 100, 120, 100, 100])
        t_arch.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ])
        )
        story.append(t_arch)
        story.append(Spacer(1, 10))

        # 4. Training Convergence Summary
        story.append(Paragraph("3. Training Summary", h2_style))
        train_rows = [
            [
                Paragraph("<b>Model</b>", table_cell_bold),
                Paragraph("<b>Epochs</b>", table_cell_bold),
                Paragraph("<b>Best Epoch</b>", table_cell_bold),
                Paragraph("<b>Best Val Accuracy</b>", table_cell_bold),
                Paragraph("<b>Duration</b>", table_cell_bold),
                Paragraph("<b>Final Val Loss</b>", table_cell_bold),
            ]
        ]
        for m in summary.models:
            th = self.get_model_training_history(m.name)
            if th:
                h_list = th.get("history", [])
                final_val_loss = f"{h_list[-1].get('val_loss', 0.0):.4f}" if h_list else "N/A"
                dur = f"{th.get('total_duration_seconds', 0):.1f}s"
                best_acc = f"{th.get('best_val_acc', 0.0) * 100:.2f}%"
                train_rows.append([
                    Paragraph(f"<b>{m.display_name}</b>", table_cell),
                    Paragraph(str(th.get("total_epochs", 5)), table_cell),
                    Paragraph(str(th.get("best_epoch", "N/A")), table_cell),
                    Paragraph(best_acc, table_cell),
                    Paragraph(dur, table_cell),
                    Paragraph(final_val_loss, table_cell),
                ])
            else:
                train_rows.append([
                    Paragraph(f"<b>{m.display_name}</b>", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("Not trained", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("N/A", table_cell),
                ])

        t_train = Table(train_rows, colWidths=[120, 60, 70, 110, 80, 100])
        t_train.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
            ])
        )
        story.append(t_train)
        story.append(Spacer(1, 10))

        # 5. Held-Out Test Evaluation Comparison
        story.append(Paragraph("4. Held-Out Test Evaluation (N=624 Radiographs)", h2_style))
        eval_headers = [
            Paragraph("<b>Architecture</b>", table_cell_bold),
            Paragraph("<b>Accuracy</b>", table_cell_bold),
            Paragraph("<b>Precision</b>", table_cell_bold),
            Paragraph("<b>Sensitivity (Recall)</b>", table_cell_bold),
            Paragraph("<b>Specificity</b>", table_cell_bold),
            Paragraph("<b>F1-Score</b>", table_cell_bold),
            Paragraph("<b>ROC-AUC (Primary)</b>", table_cell_bold),
        ]
        eval_rows = [eval_headers]

        for m in summary.models:
            if m.evaluation_status == "completed":
                is_best_roc = summary.best_model and summary.best_model.name == m.name
                roc_cell = (
                    f"<b>{m.roc_auc:.4f} *</b>" if (m.roc_auc is not None and is_best_roc)
                    else f"{m.roc_auc:.4f}" if m.roc_auc is not None else "N/A"
                )
                eval_rows.append([
                    Paragraph(f"<b>{m.display_name}</b>", table_cell),
                    Paragraph(f"{m.accuracy * 100:.2f}%" if m.accuracy is not None else "N/A", table_cell),
                    Paragraph(f"{m.precision * 100:.2f}%" if m.precision is not None else "N/A", table_cell),
                    Paragraph(f"{m.sensitivity * 100:.2f}%" if m.sensitivity is not None else "N/A", table_cell),
                    Paragraph(f"{m.specificity * 100:.2f}%" if m.specificity is not None else "N/A", table_cell),
                    Paragraph(f"{m.f1 * 100:.2f}%" if m.f1 is not None else "N/A", table_cell),
                    Paragraph(roc_cell, table_cell_bold if is_best_roc else table_cell),
                ])
            else:
                eval_rows.append([
                    Paragraph(f"<b>{m.display_name}</b>", table_cell),
                    Paragraph("Not evaluated", table_cell),
                    Paragraph("Not evaluated", table_cell),
                    Paragraph("Not evaluated", table_cell),
                    Paragraph("Not evaluated", table_cell),
                    Paragraph("Not evaluated", table_cell),
                    Paragraph("Not evaluated", table_cell),
                ])

        t_eval = Table(eval_rows, colWidths=[110, 68, 68, 85, 68, 65, 76])
        t_eval.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF6FF")]),
            ])
        )
        story.append(t_eval)
        story.append(Paragraph("<font size='7.5'>* Indicates top performance on the primary ranking metric (ROC-AUC).</font>", body_style))
        story.append(Spacer(1, 10))

        # 6. Confusion Matrices
        story.append(Paragraph("5. Confusion Matrix Analysis", h2_style))
        cm_data = [
            [
                Paragraph("<b>Architecture</b>", table_cell_bold),
                Paragraph("<b>True Negative (TN)<br/>Normal correctly predicted</b>", table_cell),
                Paragraph("<b>False Positive (FP)<br/>Normal predicted Pneumonia</b>", table_cell),
                Paragraph("<b>False Negative (FN)<br/>Pneumonia predicted Normal</b>", table_cell),
                Paragraph("<b>True Positive (TP)<br/>Pneumonia correctly predicted</b>", table_cell),
                Paragraph("<b>Total Evaluated</b>", table_cell_bold),
            ]
        ]
        for m in summary.models:
            if m.confusion_matrix:
                c = m.confusion_matrix
                tot = c.get("true_negative", 0) + c.get("false_positive", 0) + c.get("false_negative", 0) + c.get("true_positive", 0)
                cm_data.append([
                    Paragraph(f"<b>{m.display_name}</b>", table_cell),
                    Paragraph(str(c.get("true_negative", "N/A")), table_cell),
                    Paragraph(str(c.get("false_positive", "N/A")), table_cell),
                    Paragraph(str(c.get("false_negative", "N/A")), table_cell),
                    Paragraph(str(c.get("true_positive", "N/A")), table_cell),
                    Paragraph(str(tot), table_cell_bold),
                ])
            else:
                cm_data.append([
                    Paragraph(f"<b>{m.display_name}</b>", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("N/A", table_cell),
                    Paragraph("N/A", table_cell),
                ])

        t_cm = Table(cm_data, colWidths=[110, 86, 86, 86, 86, 86])
        t_cm.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ])
        )
        story.append(t_cm)
        story.append(Spacer(1, 10))

        # 7. Research Interpretation (Non-diagnostic)
        story.append(Paragraph("6. Research Interpretation & Comparative Synthesis", h2_style))
        story.append(Paragraph(summary.research_interpretation or "N/A", body_style))
        story.append(Spacer(1, 10))

        # 8. Mandatory Medical Safety Disclaimer Box
        story.append(KeepTogether([
            Paragraph("<b>Academic & Medical Safety Disclaimer (Mandatory)</b>", h2_style),
            Table(
                [[
                    Paragraph(
                        "<b>Educational & Research Use Only:</b> This software and generated report are intended "
                        "exclusively for academic deep learning research and comparative model benchmarking. "
                        "<b>This model has not been clinically validated and is not approved as a medical diagnostic device.</b> "
                        "Performance metrics are specific strictly to the evaluated PneumoniaMNIST research dataset "
                        "and must not be interpreted as clinical diagnostic efficacy. All inference predictions and activation maps "
                        "represent experimental statistical correlations that cannot replace qualified professional medical judgment.",
                        disclaimer_style,
                    )
                ]],
                colWidths=[540],
                style=[
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D97706")),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ],
            )
        ]))

        doc.build(story)
        return buffer.getvalue()


analytics_service = AnalyticsService()
