import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from app.agents.advisor_workflow import run_advisor
from app.tools.reporting import generate_html_report


def load_tabular_dataset(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        # Detect common CSV delimiters while respecting quoted fields.
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(65536)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        df = pd.read_csv(file_path, sep=delimiter, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(
            "Unsupported tabular format. Use .csv, .xlsx, .xls or .parquet."
        )

    if df.empty:
        raise ValueError("The dataset has no rows.")
    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a tabular dataset and generate an advisory report."
    )
    parser.add_argument("--file", required=True, help="Path to a dataset file or image/audio folder.")
    parser.add_argument("--modality", choices=["tabular", "image", "audio"], help="Input modality; auto-detected when omitted.")
    parser.add_argument("--request", required=True, help="What you want to understand or decide.")
    parser.add_argument("--reports-dir", default="reports", help="Output directory.")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.file).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    modality = args.modality
    df = None

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    audio_extensions = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    tabular_extensions = {".csv", ".xlsx", ".xls", ".parquet"}

    if modality is None:
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix in tabular_extensions:
                modality = "tabular"
            elif suffix in image_extensions:
                modality = "image"
            elif suffix in audio_extensions:
                modality = "audio"
            else:
                raise ValueError(f"Cannot infer modality from extension: {suffix}")
        else:
            suffixes = {
                item.suffix.lower()
                for item in input_path.rglob("*")
                if item.is_file()
            }
            has_images = bool(suffixes & image_extensions)
            has_audio = bool(suffixes & audio_extensions)
            if has_images and has_audio:
                raise ValueError(
                    "This folder contains both images and audio. "
                    "Specify --modality image or --modality audio."
                )
            if has_images:
                modality = "image"
            elif has_audio:
                modality = "audio"
            else:
                raise ValueError("Folder contains no supported image or audio files.")

    if modality == "tabular":
        df = load_tabular_dataset(str(input_path))
        print(f"Loaded {input_path}: {len(df)} rows, {len(df.columns)} columns")
    else:
        print(f"Loaded {modality} input: {input_path}")

    print("Running advisory workflow.")
    result = run_advisor(
        request=args.request,
        df=df,
        input_path=str(input_path),
        modality=modality,
        reports_dir=str(reports_dir),
    )

    json_path = reports_dir / "advisor_results.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    html_path = generate_html_report(
        df=df,
        result=result,
        output_path=str(reports_dir / "advisor_report.html"),
    )

    print("\n===== ADVISOR SUMMARY =====")
    print(f"Modality: {result['dataset']['modality']}")
    print(f"Findings: {len(result['findings'])}")
    print(f"Recommendations: {len(result['recommendations'].get('recommendations', []))}")
    print(f"JSON results: {json_path}")
    print(f"HTML report: {html_path}")

    failures = [
        item for item in result["agent_results"]
        if item.get("status") == "failed"
    ]
    if failures:
        print("\nAgent errors:")
        for failure in failures:
            print(f"- {failure['agent']}: {failure.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
