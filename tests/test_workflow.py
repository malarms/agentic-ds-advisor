
import json

import pandas as pd

from app.agents.workflow import run_analysis


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        return FakeResponse(self.response)


def make_fake_llm(steps):
    response = {
        "objective": "Profile the dataset and assess data quality.",
        "steps": steps,
        "target_column": None,
        "task": None,
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "This test checks the requested workflow steps.",
    }
    return FakeLLM(json.dumps(response))


def test_workflow_runs_requested_tools_only():
    df = pd.DataFrame(
        {
            "age": [22, 25, 30, 35],
            "city": ["Pune", "Mumbai", "Pune", "Delhi"],
            "income": [30000, 45000, 50000, 60000],
        }
    )

    result = run_analysis(
        request="Profile the dataset and check its data quality.",
        df=df,
        llm=make_fake_llm(["profile", "data_quality"]),
    )

    assert result["plan"].steps == ["profile", "data_quality"]
    assert "profile" in result["results"]
    assert "data_quality" in result["results"]
    assert "eda" not in result["results"]
    assert "ml" not in result["results"]


def test_workflow_returns_clarification_without_running_tools():
    response = {
        "objective": "Predict an unspecified target.",
        "steps": ["profile"],
        "target_column": None,
        "task": None,
        "needs_clarification": True,
        "clarification_question": "Which column should be predicted?",
        "rationale": "The prediction target is unspecified.",
    }

    df = pd.DataFrame({"age": [22, 25], "income": [30000, 45000]})

    result = run_analysis(
        request="Build a prediction model.",
        df=df,
        llm=FakeLLM(json.dumps(response)),
    )

    assert result["plan"].needs_clarification is True
    assert result["results"] == {}
    assert result["plan"].clarification_question is not None

def test_workflow_records_tool_failure_and_continues(monkeypatch):
    import app.agents.workflow as workflow

    def failing_profile(df):
        raise RuntimeError("Simulated profiling failure")

    monkeypatch.setattr(workflow, "profile_dataset", failing_profile)

    df = pd.DataFrame(
        {
            "age": [22, 25, 30, 35],
            "income": [30000, 45000, 50000, 60000],
        }
    )

    result = run_analysis(
        request="Profile the dataset and check data quality.",
        df=df,
        llm=make_fake_llm(["profile", "data_quality"]),
    )

    assert result["execution_status"]["profile"]["status"] == "failed"
    assert (
        "Simulated profiling failure"
        in result["execution_status"]["profile"]["error"]
    )

    # The independent data-quality tool should still execute.
    assert result["execution_status"]["data_quality"]["status"] == "success"
    assert "data_quality" in result["results"]

    # Failed results must not be represented as successful output.
    assert "profile" not in result["results"]
