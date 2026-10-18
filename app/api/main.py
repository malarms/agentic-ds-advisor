"""FastAPI backend for the Agentic Data Science Advisor.

Wraps the existing run_advisor workflow to accept HTTP file uploads and
return generated HTML/JSON reports.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.agents.run_advisor import load_tabular_dataset
from app.agents.advisor_workflow import run_advisor
from app.tools.reporting import generate_html_report


app = FastAPI(
    title="Agentic Data Science Advisor API",
    description="Upload a dataset and get an exploratory analysis with ML strategy recommendations.",
    version="0.1.0",
)

# Allow local development from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for temporary reports
REPORTS_BASE = Path("reports")
REPORTS_BASE.mkdir(parents=True, exist_ok=True)

# Allowed extensions per modality (matches run_advisor.py logic)
TABULAR_EXT = {".csv", ".xlsx", ".xls", ".parquet"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
ZIP_EXT = {".zip"}

# Max upload size (100 MB)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024


def _safe_suffix(filename: str) -> str:
    """Extract a safe file suffix from the uploaded filename."""
    # Only use the final extension; ignore any path components
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    return suffix if suffix else ""


def _infer_modality(filename: str, explicit_modality: Optional[str]) -> Literal["tabular", "image", "audio"]:
    """Determine modality from explicit argument or file extension."""
    if explicit_modality:
        return explicit_modality

    suffix = _safe_suffix(filename)
    if suffix in TABULAR_EXT:
        return "tabular"
    if suffix in IMAGE_EXT:
        return "image"
    if suffix in AUDIO_EXT:
        return "audio"
    if suffix in ZIP_EXT:
        # ZIP files need explicit modality
        raise HTTPException(
            status_code=400,
            detail="ZIP uploads require explicit --modality parameter.",
        )
    raise HTTPException(
        status_code=400,
        detail=f"Cannot infer modality from extension: {suffix}. Use --modality to specify.",
    )


def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream the upload to disk with size limit."""
    size = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 100 MB).")
            out.write(chunk)


def _extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Safely extract a ZIP file, preventing path traversal."""
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Validate all paths are safe (no absolute paths, no ..)
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsafe path in ZIP archive: {member.filename}",
                )
        zf.extractall(extract_dir)

    # If the ZIP contains a single top-level directory, return that
    items = list(extract_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return extract_dir


@app.post("/advisor", response_class=JSONResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    request: str = Form(...),
    modality: Optional[Literal["tabular", "image", "audio"]] = Form(None),
    reports_dir: Optional[str] = Form(None),
) -> JSONResponse:
    """Analyze an uploaded dataset and return links to the generated reports."""
    if not request.strip():
        raise HTTPException(status_code=400, detail="Request description is required.")

    modality = _infer_modality(file.filename or "", modality)

    # Create a unique working directory for this request
    run_id = uuid.uuid4().hex[:8]
    work_dir = REPORTS_BASE / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    suffix = _safe_suffix(file.filename or "")
    input_path = None
    df = None

    try:
        if suffix in ZIP_EXT:
            # Handle ZIP upload: extract to a subdirectory
            zip_path = work_dir / f"upload{suffix}"
            _save_upload(file, zip_path)
            extract_dir = work_dir / "extracted"
            extract_dir.mkdir()
            input_path = str(_extract_zip(zip_path, extract_dir))
            # Clean up the ZIP after extraction
            zip_path.unlink(missing_ok=True)
        else:
            # Single file upload
            upload_path = work_dir / f"upload{suffix}"
            _save_upload(file, upload_path)
            input_path = str(upload_path)

        if modality == "tabular":
            # For tabular, we can load into a DataFrame and let the workflow use it
            df = load_tabular_dataset(input_path)

        # Run the existing advisory workflow
        result = run_advisor(
            request=request,
            df=df,
            input_path=input_path,
            modality=modality,
            reports_dir=str(work_dir),
        )

        # Generate HTML report (run_advisor does this, but ensure it exists)
        html_path = work_dir / "advisor_report.html"
        if not html_path.exists():
            generate_html_report(df=df, result=result, output_path=str(html_path))

        json_path = work_dir / "advisor_results.json"

        return JSONResponse(
            content={
                "status": "success",
                "run_id": run_id,
                "modality": modality,
                "summary": {
                    "findings": len(result.get("findings", [])),
                    "recommendations": len(result.get("recommendations", {}).get("recommendations", [])),
                    "advisory_status": result.get("advisory_status"),
                },
                "download": {
                    "html": f"/reports/{run_id}/advisor_report.html",
                    "json": f"/reports/{run_id}/advisor_results.json",
                },
            }
        )

    except HTTPException:
        # Clean up on HTTP errors
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except ValueError as e:
        error_msg = str(e)
        # Check if it's an API key configuration error
        if "FREELLMAPI_API_KEY" in error_msg or "API key" in error_msg:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503,
                detail="LLM service not configured. Set FREELLMAPI_API_KEY in environment.",
            )
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        # Log the full error for debugging (in production, use proper logging)
        print(f"Error processing request {run_id}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/reports/{run_id}/advisor_report.html", response_class=HTMLResponse)
async def get_html_report(run_id: str):
    """Serve the generated HTML report."""
    # Validate run_id format (8 hex chars)
    if not all(c in "0123456789abcdef" for c in run_id) or len(run_id) != 8:
        raise HTTPException(status_code=404, detail="Report not found.")

    html_path = REPORTS_BASE / run_id / "advisor_report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(html_path, media_type="text/html")


@app.get("/reports/{run_id}/advisor_results.json", response_class=JSONResponse)
async def get_json_report(run_id: str):
    """Serve the generated JSON results."""
    if not all(c in "0123456789abcdef" for c in run_id) or len(run_id) != 8:
        raise HTTPException(status_code=404, detail="Report not found.")

    json_path = REPORTS_BASE / run_id / "advisor_results.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON results not found.")
    return FileResponse(json_path, media_type="application/json")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/reports/{run_id}/advisor_report.html", response_class=HTMLResponse)
async def get_html_report(run_id: str):
    """Serve the generated HTML report."""
    html_path = REPORTS_BASE / run_id / "advisor_report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(html_path, media_type="text/html")


@app.get("/reports/{run_id}/advisor_results.json", response_class=JSONResponse)
async def get_json_report(run_id: str):
    """Serve the generated JSON results."""
    json_path = REPORTS_BASE / run_id / "advisor_results.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON results not found.")
    return FileResponse(json_path, media_type="application/json")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)