import csv
import io
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.analytics import analytics_service

client = TestClient(app)


def test_export_json_endpoint():
    response = client.get("/api/export/json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert 'filename="medical_image_classification_results.json"' in response.headers.get("content-disposition", "")

    data = response.json()
    assert "project" in data
    assert "dataset" in data
    assert "summary" in data
    assert "models_detail" in data
    assert "disclaimer" in data
    assert "Educational & Research Use Only" in data["disclaimer"]


def test_export_csv_endpoint():
    response = client.get("/api/export/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert 'filename="medical_image_classification_results.csv"' in response.headers.get("content-disposition", "")

    reader = csv.reader(io.StringIO(response.text))
    rows = list(reader)
    assert len(rows) >= 3  # header + 2 models
    header = rows[0]
    expected_headers = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "test_samples",
    ]
    assert header == expected_headers

    models = [r[0] for r in rows[1:]]
    assert "resnet50" in models
    assert "efficientnet_b0" in models


def test_export_pdf_report_endpoint():
    response = client.get("/api/export/report")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert 'filename="medical_image_classification_report.pdf"' in response.headers.get("content-disposition", "")

    pdf_bytes = response.content
    assert len(pdf_bytes) > 1000
    # Standard PDF magic header check
    assert pdf_bytes.startswith(b"%PDF")
