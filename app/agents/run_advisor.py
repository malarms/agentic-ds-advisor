import argparse
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
        df = pd.read_csv(file_path)
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
    parser.add_argument("--file", required=True, help="Path to CSV, Excel or Parquet file.")
    parser.add_argument("--request", required=True, help="What you want to understand or decide.")
    parser.add_argument("--reports-dir", default="reports", help="Output directory.")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_tabular_dataset(args.file)
    print(f"Loaded {args.file}: {len(df)} rows, {len(df.columns)} columns")
    print("Running advisory workflow. No models will be trained.")

    result = run_advisor(
        request=args.request,
        df=df,
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
    print("Models trained: No")
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
