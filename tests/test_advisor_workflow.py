import json

import pandas as pd
import pytest

from app.agents.advisor_workflow import run_advisor


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
        assert "must never train" in prompt.lower() or "not to train models" in prompt.lower()
        return FakeResponse(
            json.dumps(
                {
                    "task_interpretation": "Understand patterns in customer churn.",
                    "recommendations": [
                        {
                            "title": "Establish a simple baseline",
                            "approach": "Start with a simple, interpretable baseline if a valid target is confirmed.",
                            "rationale": "A baseline gives a reference point before considering more complex methods.",
                            "supporting_findings": ["Dataset structure", "Data quality"],
                            "validation_strategy": "Use a stratified holdout or stratified cross-validation if the target is categorical and class counts allow it.",
                            "suggested_metrics": ["F1", "Precision", "Recall"],
                            "risks": ["The sample dataset is small.", "The target may not be suitable for prediction."],
                            "next_steps": ["Confirm the prediction target.", "Inspect class counts.", "Check for leakage."]
                        },
                        {
                            "title": "Check data suitability first",
                            "approach": "Inspect label quality, missingness and potential leakage before choosing a model.",
                            "rationale": "Data issues can invalidate downstream conclusions.",
                            "supporting_findings": ["Data quality"],
                            "validation_strategy": "Choose validation after confirming whether observations are independent and identically distributed.",
                            "suggested_metrics": ["Metric aligned with the business objective"],
                            "risks": ["The available data may not represent future cases."],
                            "next_steps": ["Confirm how the data was collected."]
                        }
                    ],
                    "assumptions": ["The objective may involve predicting churn; confirm before modelling."],
                    "clarification_questions": ["Is churn the intended prediction target?"]
                }
            )
        )


def test_advisor_returns_findings_recommendations_and_report(tmp_path):
    df = pd.DataFrame(
        {
            "age": [22, 35, 41, 29, 51, 33],
            "monthly_charges": [20, 50, 65, 30, 80, 42],
            "contract_type": ["monthly", "annual", "monthly", "annual", "monthly", "annual"],
            "churn": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )

    result = run_advisor(
        request="Help me understand this dataset and advise how churn could be analysed.",
        df=df,
        llm=FakeLLM(),
        reports_dir=str(tmp_path),
    )

    assert result["training_performed"] is False
    assert result["dataset"]["modality"] == "tabular"
    assert result["dataset"]["row_count"] == 6
    assert result["findings"]
    assert len(result["recommendations"]["recommendations"]) == 2
    assert result["report"]["training_performed"] is False

    agent_names = [item["agent"] for item in result["agent_results"]]
    assert agent_names == [
        "dataset_analyst",
        "eda_agent",
        "ml_strategy_agent",
        "report_agent",
    ]
    assert all(item["status"] == "success" for item in result["agent_results"])


def test_advisor_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="cannot be empty"):
        run_advisor("Analyse this dataset", pd.DataFrame())


def test_advisor_never_calls_training_utility(monkeypatch, tmp_path):
    from app.tools import model_training

    def forbidden_training(*args, **kwargs):
        raise AssertionError("The advisory workflow must never train models.")

    monkeypatch.setattr(model_training, "train_and_compare_models", forbidden_training)

    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "label": ["a", "b", "a", "b", "a", "b"],
        }
    )

    result = run_advisor(
        request="Recommend an appropriate approach for this dataset.",
        df=df,
        llm=FakeLLM(),
        reports_dir=str(tmp_path),
    )

    assert result["training_performed"] is False
