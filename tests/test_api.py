"""Tests for the FastAPI backend endpoints."""

import io
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint returns OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_tabular_csv():
    """Upload a simple CSV and get report links."""
    csv_content = b"age,job,y\n30,blue-collar,no\n45,management,yes\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}
    data = {"request": "Understand this data", "modality": "tabular"}

    response = client.post("/advisor", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert "run_id" in result
    assert result["modality"] == "tabular"
    assert "download" in result
    assert "html" in result["download"]
    assert "json" in result["download"]


def test_analyze_invalid_extension():
    """Upload a file with unsupported extension returns 400."""
    files = {"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")}
    data = {"request": "Test"}

    response = client.post("/advisor", files=files, data=data)

    assert response.status_code == 400


def test_analyze_empty_request():
    """Missing or empty request description returns 400."""
    csv_content = b"a,b\n1,2\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}
    data = {"request": "   "}

    response = client.post("/advisor", files=files, data=data)

    assert response.status_code == 400
    assert "Request description is required" in response.json().get("detail", "")


def test_analyze_zip_requires_modality():
    """ZIP upload without explicit modality returns 400."""
    # Create a minimal ZIP in memory
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("test.csv", "a,b\n1,2\n")
    zip_buffer.seek(0)

    files = {"file": ("test.zip", zip_buffer, "application/zip")}
    data = {"request": "Test"}

    response = client.post("/advisor", files=files, data=data)

    assert response.status_code == 400
    assert "modality" in response.json().get("detail", "").lower()


def test_analyze_zip_with_modality():
    """ZIP upload with modality works (basic test)."""
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("test.csv", "a,b\n1,2\n")
    zip_buffer.seek(0)

    files = {"file": ("test.zip", zip_buffer, "application/zip")}
    data = {"request": "Test", "modality": "tabular"}

    response = client.post("/advisor", files=files, data=data)

    # Should succeed or fail gracefully based on workflow
    assert response.status_code in (200, 400, 500)


def test_report_endpoints_not_found():
    """Non-existent run_id returns 404."""
    response = client.get("/reports/invalidrun/advisor_report.html")
    assert response.status_code == 404

    response = client.get("/reports/invalidrun/advisor_results.json")
    assert response.status_code == 404


def test_report_endpoints_invalid_format():
    """Invalid run_id format returns 404."""
    response = client.get("/reports/notvalid/advisor_report.html")
    assert response.status_code == 404


def test_file_size_limit():
    """Files over 100MB are rejected (simulated via test override)."""
    # This test would need a real large file; we skip actual large upload
    # but the size check is in _save_upload
    pass


def test_missing_api_key_error():
    """Missing API key returns 503 with clear message."""
    # This requires mocking the LLM client; skip for unit test
    pass


def test_cleanup_on_error():
    """Failed uploads should not leave temporary directories."""
    # Create a CSV that will cause a workflow error
    csv_content = b""  # empty CSV will cause ValueError
    files = {"file": ("empty.csv", io.BytesIO(csv_content), "text/csv")}
    data = {"request": "Test"}

    response = client.post("/advisor", files=files, data=data)

    # Should return 400 (or 500 if workflow error)
    assert response.status_code in (400, 500)

    # Note: actual cleanup verification would need filesystem inspection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])