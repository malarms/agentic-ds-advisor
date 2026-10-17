
import argparse
import json
from pathlib import Path

import pandas as pd

from app.agents.workflow import run_analysis


def json_default(value):
    """Convert common NumPy/Pandas values into JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run the agentic data science workflow on a CSV dataset."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--request",
        required=True,
        help="Describe the analysis you want the platform to perform.",
    )
    args = parser.parse_args()

    csv_path = Path(args.file)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The CSV file contains no data rows.")

    if len(df.columns) == 0:
        raise ValueError("The CSV file contains no columns.")

    print(f"Loaded dataset: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")

    output = run_analysis(
        request=args.request,
        df=df,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "analysis_results.json"

    report = {
        "dataset": str(csv_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "plan": output["plan"].model_dump(),
        "results": output["results"],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print("\n--- Analysis plan ---")
    print(output["plan"].model_dump_json(indent=2))

    if output["plan"].needs_clarification:
        print("\nClarification required:")
        print(output["plan"].clarification_question)
    else:
        print("\nExecuted result sections:")
        for section in output["results"]:
            print(f"- {section}")

    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
