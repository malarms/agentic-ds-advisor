import pandas as pd

from app.tools.reporting import generate_html_report


def test_report_contains_findings_recommendations_and_no_training_claim(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "monthly_charges": [10, 20, 30, 40],
            "contract_type": ["monthly", "annual", "monthly", "annual"],
        }
    )
    result = {
        "dataset": {
            "name": "test",
            "modality": "tabular",
            "objective": "Understand customer data",
            "row_count": 4,
            "column_count": 3,
            "columns": ["age", "monthly_charges", "contract_type"],
        },
        "findings": [
            {
                "agent": "dataset_analyst",
                "title": "Dataset structure",
                "observation": "The dataset has four rows.",
                "evidence": {"row_count": 4},
                "limitations": [],
            }
        ],
        "recommendations": {
            "task_interpretation": "Explore the data before choosing a task.",
            "recommendations": [
                {
                    "title": "Check the objective",
                    "approach": "Clarify the intended outcome first.",
                    "rationale": "The objective determines the task framing.",
                    "supporting_findings": ["Dataset structure"],
                    "validation_strategy": "Choose after the task is defined.",
                    "suggested_metrics": ["Choose metrics after task definition"],
                    "risks": ["The target is not confirmed."],
                    "next_steps": ["Clarify the objective."],
                }
            ],
            "assumptions": [],
            "clarification_questions": ["What outcome matters most?"],
        },
        "agent_results": [
            {
                "agent": "dataset_analyst",
                "status": "success",
                "summary": "Analysis completed.",
            }
        ],
        "training_performed": False,
    }

    output_path = tmp_path / "nested" / "report.html"
    generated_path = generate_html_report(df, result, str(output_path))

    assert output_path.exists()
    assert generated_path == str(output_path)
    html = output_path.read_text(encoding="utf-8")
    assert "Dataset analysis" in html
    assert "Dataset structure" in html
    assert "Check the objective" in html
    assert "Models trained" not in html
    assert "Plotly.newPlot" in html
    assert 'height:420px' in html
    assert 'src="https://cdn.plot.ly/plotly-' not in html


def test_report_escapes_user_supplied_html(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = {
        "dataset": {
            "modality": "tabular",
            "objective": "<script>alert('x')</script>",
            "row_count": 3,
            "column_count": 1,
            "columns": ["value"],
        },
        "findings": [],
        "recommendations": {"recommendations": []},
        "agent_results": [],
    }

    output_path = tmp_path / "report.html"
    generate_html_report(df, result, str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert "<script>alert('x')</script>" not in content
    assert "&lt;script&gt;" in content
