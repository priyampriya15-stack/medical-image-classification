import json
import logging
from fastapi import APIRouter, HTTPException, Response, status

from app.analytics import analytics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["Export"])


@router.get(
    "/json",
    summary="Export complete research platform results as JSON",
    description="Returns comprehensive research metrics, training history, evaluation reports, and metadata in JSON format.",
)
def export_json(download: bool = True) -> Response:
    try:
        data = analytics_service.get_full_export_data()
        content = json.dumps(data, indent=2)

        headers = {}
        if download:
            headers["Content-Disposition"] = 'attachment; filename="medical_image_classification_results.json"'

        return Response(
            content=content,
            media_type="application/json",
            headers=headers,
        )
    except Exception as exc:
        logger.error("JSON export error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate research JSON export.",
        )


@router.get(
    "/csv",
    summary="Export comparative test evaluation metrics as CSV",
    description="Tabular CSV export containing accuracy, precision, recall, specificity, F1, and ROC-AUC.",
)
def export_csv() -> Response:
    try:
        csv_content = analytics_service.get_csv_export()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="medical_image_classification_results.csv"'
            },
        )
    except Exception as exc:
        logger.error("CSV export error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate research CSV export.",
        )


@router.get(
    "/report",
    summary="Generate and download PDF Research Report",
    description="Generates an academic research summary PDF using ReportLab with disclaimers, tables, and metrics.",
)
def export_pdf_report() -> Response:
    try:
        pdf_bytes = analytics_service.generate_pdf_report()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="medical_image_classification_report.pdf"'
            },
        )
    except Exception as exc:
        logger.error("PDF report generation error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF research report.",
        )
