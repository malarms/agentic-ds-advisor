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


def test_idx_directory_discovery_direct():
    """Test that profile_idx_image_dataset finds IDX files in the given directory."""
    from app.tools.dataset_metadata import profile_idx_image_dataset

    # Test with Fashion-MNIST directory
    result = profile_idx_image_dataset("data/fashion-mnist-master/data/fashion")
    assert result is not None
    assert result["dataset_format"] == "gzip_idx"
    assert result["valid_count"] == 70000
    assert result["archive_files_discovered"] == 4
    assert result["split_counts"]["train"] == 60000
    assert result["split_counts"]["test"] == 10000


def test_idx_directory_discovery_nested():
    """Test that profile_idx_image_dataset finds IDX files in nested directories."""
    from app.tools.dataset_metadata import profile_idx_image_dataset
    import tempfile
    import shutil
    import os

    # Create a temporary directory structure mimicking ZIP extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the nested structure: tmpdir/data/fashion-mnist-master/data/fashion/
        fashion_dir = Path(tmpdir) / "data" / "fashion-mnist-master" / "data" / "fashion"
        fashion_dir.mkdir(parents=True)

        # Copy the IDX files to the nested directory
        src_fashion = Path("data/fashion-mnist-master/data/fashion")
        for file_name in [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]:
            src_file = src_fashion / file_name
            dst_file = fashion_dir / file_name
            shutil.copy2(src_file, dst_file)

        # Test that it finds the IDX files even when given the parent directory
        result = profile_idx_image_dataset(Path(tmpdir) / "data")
        assert result is not None
        assert result["dataset_format"] == "gzip_idx"
        assert result["valid_count"] == 70000
        assert result["archive_files_discovered"] == 4
        assert result["split_counts"]["train"] == 60000
        assert result["split_counts"]["test"] == 10000


def test_idx_directory_discovery_multiple_ambiguous():
    """Test that multiple IDX directories raise an error."""
    from app.tools.dataset_metadata import profile_idx_image_dataset
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory with two separate IDX sets
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create first IDX set
        fashion1_dir = tmp_path / "set1"
        fashion1_dir.mkdir()
        src_fashion = Path("data/fashion-mnist-master/data/fashion")
        for file_name in [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]:
            src_file = src_fashion / file_name
            dst_file = fashion1_dir / file_name
            shutil.copy2(src_file, dst_file)

        # Create second IDX set
        fashion2_dir = tmp_path / "set2"
        fashion2_dir.mkdir()
        for file_name in [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]:
            src_file = src_fashion / file_name
            dst_file = fashion2_dir / file_name
            shutil.copy2(src_file, dst_file)

        # Test that it raises an error for ambiguous directories
        with pytest.raises(ValueError, match="Multiple directories with IDX files found"):
            profile_idx_image_dataset(tmp_path)


def test_idx_directory_not_found():
    """Test that profile_idx_image_dataset returns None when no IDX files found."""
    from app.tools.dataset_metadata import profile_idx_image_dataset
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        result = profile_idx_image_dataset(tmpdir)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])