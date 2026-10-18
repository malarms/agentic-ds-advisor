import json

import pandas as pd
import pytest

from app.agents.advisor_workflow import run_advisor


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
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
                            "validation_strategy": "Use stratified validation if the target is categorical and class counts allow it.",
                            "suggested_metrics": ["F1", "Precision", "Recall"],
                            "risks": ["The sample may be too small for reliable conclusions."],
                            "next_steps": ["Confirm the prediction target.", "Inspect class counts.", "Check for leakage."],
                        },
                        {
                            "title": "Check data suitability",
                            "approach": "Inspect label quality, missingness and potential leakage before selecting a method.",
                            "rationale": "Data issues can invalidate downstream conclusions.",
                            "supporting_findings": ["Data quality"],
                            "validation_strategy": "Choose validation after confirming how observations were collected.",
                            "suggested_metrics": ["Metrics aligned with the eventual task"],
                            "risks": ["The data may not represent future cases."],
                            "next_steps": ["Confirm the data collection process."],
                        },
                    ],
                    "assumptions": ["The objective may involve predicting churn; confirm before proceeding."],
                    "clarification_questions": ["Is churn the intended prediction target?"],
                }
            )
        )


class UnexpectedLLM:
    def invoke(self, prompt):
        raise AssertionError("The strategy agent should not have been called.")


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [22, 35, 41, 29, 51, 33],
            "monthly_charges": [20, 50, 65, 30, 80, 42],
            "contract_type": ["monthly", "annual", "monthly", "annual", "monthly", "annual"],
            "churn": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )


def test_advisor_returns_findings_recommendations_and_report(sample_df, tmp_path):
    result = run_advisor(
        request="Help me understand this dataset and advise how churn could be analysed.",
        df=sample_df,
        llm=FakeLLM(),
        reports_dir=str(tmp_path),
    )

    assert result["dataset"]["modality"] == "tabular"
    assert result["dataset"]["row_count"] == 6
    assert result["findings"]
    assert len(result["recommendations"]["recommendations"]) == 2
    assert result["advisory_status"] == "complete"
    assert result["report"]["advisory_status"] == "complete"

    agent_names = [item["agent"] for item in result["agent_results"]]
    assert agent_names == [
        "dataset_analyst",
        "eda_agent",
        "ml_strategy_agent",
        "report_agent",
    ]
    assert all(item["status"] == "success" for item in result["agent_results"])
    assert any("Evidence gate passed" in item for item in result["routing_decisions"])


def test_explicit_eda_only_request_skips_strategy(sample_df, tmp_path):
    result = run_advisor(
        request="EDA only. Show me the main distributions and relationships.",
        df=sample_df,
        llm=UnexpectedLLM(),
        reports_dir=str(tmp_path),
    )

    agent_names = [item["agent"] for item in result["agent_results"]]
    assert agent_names == ["dataset_analyst", "eda_agent", "report_agent"]
    assert result["recommendations"] == {}
    assert result["advisory_status"] == "analysis_only"


def test_profile_only_request_skips_eda_and_strategy(sample_df, tmp_path):
    result = run_advisor(
        request="Profile only. Summarise the schema and data quality.",
        df=sample_df,
        llm=UnexpectedLLM(),
        reports_dir=str(tmp_path),
    )

    agent_names = [item["agent"] for item in result["agent_results"]]
    assert agent_names == ["dataset_analyst", "report_agent"]
    assert result["recommendations"] == {}
    assert result["advisory_status"] == "analysis_only"


def test_failed_profiling_blocks_recommendations(monkeypatch, sample_df, tmp_path):
    import app.agents.advisor_workflow as workflow

    def fail_profile(df):
        raise RuntimeError("Synthetic profiling failure")

    monkeypatch.setattr(workflow, "profile_dataset", fail_profile)

    result = run_advisor(
        request="Analyse the dataset and recommend suitable approaches.",
        df=sample_df,
        llm=UnexpectedLLM(),
        reports_dir=str(tmp_path),
    )

    assert result["advisory_status"] == "blocked"
    assert result["recommendations"] == {}
    assert not any(
        item["agent"] == "ml_strategy_agent"
        for item in result["agent_results"]
    )
    assert any(
        item["agent"] == "dataset_analyst" and item["status"] == "failed"
        for item in result["agent_results"]
    )


def test_unknown_finding_reference_is_rejected(sample_df, tmp_path):
    class HallucinatingLLM:
        def invoke(self, prompt):
            payload = {
                "task_interpretation": "Analyse the dataset.",
                "recommendations": [
                    {
                        "title": "Unsupported recommendation",
                        "approach": "Use a method based on an invented finding.",
                        "rationale": "The referenced finding does not exist.",
                        "supporting_findings": ["Invented finding that was never observed"],
                        "validation_strategy": "Define after the task is confirmed.",
                        "suggested_metrics": ["To be determined"],
                        "risks": ["Insufficient evidence"],
                        "next_steps": ["Collect the necessary evidence."],
                    }
                ],
                "assumptions": [],
                "clarification_questions": [],
            }
            return FakeResponse(json.dumps(payload))

    result = run_advisor(
        request="Recommend a suitable approach for this dataset.",
        df=sample_df,
        llm=HallucinatingLLM(),
        reports_dir=str(tmp_path),
    )

    assert result["recommendations"] == {}
    assert result["advisory_status"] == "partial"
    assert any(
        item["agent"] == "ml_strategy_agent" and item["status"] == "failed"
        for item in result["agent_results"]
    )


def test_advisor_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="cannot be empty"):
        run_advisor("Analyse this dataset", pd.DataFrame())


def test_image_profile_only_runs_through_advisor(tmp_path):
    from PIL import Image

    Image.new("RGB", (20, 10), (10, 20, 30)).save(tmp_path / "sample.png")

    result = run_advisor(
        request="Profile only.",
        df=None,
        input_path=str(tmp_path),
        modality="image",
        llm=UnexpectedLLM(),
        reports_dir=str(tmp_path / "reports"),
    )

    assert result["dataset"]["modality"] == "image"
    assert result["dataset"]["row_count"] == 1
    assert result["advisory_status"] == "analysis_only"
    assert result["findings"][0]["evidence"]["valid_count"] == 1


def test_audio_profile_only_runs_through_advisor(tmp_path):
    import wave
    import numpy as np
    import soundfile as sf

    sample_rate = 8000
    samples = (
        np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate) * 0.4
    ).astype(np.float32)
    sf.write(tmp_path / "sample.wav", samples, sample_rate)

    result = run_advisor(
        request="Profile only.",
        df=None,
        input_path=str(tmp_path),
        modality="audio",
        llm=UnexpectedLLM(),
        reports_dir=str(tmp_path / "reports"),
    )

    assert result["dataset"]["modality"] == "audio"
    assert result["dataset"]["row_count"] == 1
    assert result["advisory_status"] == "analysis_only"


def test_evidence_truncation_records_original_count(sample_df, tmp_path):
    """Test that evidence truncation records the original item count."""
    from app.agents.advisor_workflow import _summarize_evidence_for_llm

    # Create evidence with more than 50 items
    evidence = {
        "profile": {"items": [{"name": f"item_{i}"} for i in range(100)]},
        "eda": {"items": [{"name": f"eda_{i}"} for i in range(75)]},
    }

    summary = _summarize_evidence_for_llm(evidence, max_items=50)

    assert len(summary["profile"]["items"]) == 50
    assert summary["profile"]["_items_truncated"] == "Showing first 50 of 100 items"
    assert len(summary["eda"]["items"]) == 50
    assert summary["eda"]["_items_truncated"] == "Showing first 50 of 75 items"


def test_evidence_truncation_noop_when_under_limit(sample_df, tmp_path):
    """Test that evidence is not modified when under the limit."""
    from app.agents.advisor_workflow import _summarize_evidence_for_llm

    evidence = {
        "profile": {"items": [{"name": f"item_{i}"} for i in range(10)]},
        "eda": {"items": [{"name": f"eda_{i}"} for i in range(5)]},
    }

    summary = _summarize_evidence_for_llm(evidence, max_items=50)

    assert len(summary["profile"]["items"]) == 10
    assert "_items_truncated" not in summary["profile"]
    assert len(summary["eda"]["items"]) == 5
    assert "_items_truncated" not in summary["eda"]


def test_evidence_truncation_handles_missing_keys(sample_df, tmp_path):
    """Test that truncation handles missing profile/eda or items keys gracefully."""
    from app.agents.advisor_workflow import _summarize_evidence_for_llm

    evidence = {
        "profile": {},
        "eda": {"items": [{"name": f"eda_{i}"} for i in range(100)]},
    }

    summary = _summarize_evidence_for_llm(evidence, max_items=50)

    assert "profile" in summary
    assert "_items_truncated" not in summary["profile"]
    assert len(summary["eda"]["items"]) == 50
    assert summary["eda"]["_items_truncated"] == "Showing first 50 of 100 items"


def test_invalid_finding_reference_rejected_in_strategy(sample_df, tmp_path):
    """Test that recommendations referencing non-existent findings are rejected."""
    class BadLLM:
        def invoke(self, prompt):
            import json
            return FakeResponse(json.dumps({
                "task_interpretation": "Test",
                "recommendations": [{
                    "title": "Bad Rec",
                    "approach": "Test",
                    "rationale": "Test",
                    "supporting_findings": ["Non-existent Finding Title"],
                    "validation_strategy": "Test",
                    "suggested_metrics": [],
                    "risks": [],
                    "next_steps": [],
                }],
                "assumptions": [],
                "clarification_questions": [],
            }))

    result = run_advisor(
        request="Recommend something.",
        df=sample_df,
        llm=BadLLM(),
        reports_dir=str(tmp_path),
    )

    assert result["recommendations"] == {}
    assert result["advisory_status"] == "partial"
    assert any(
        item["agent"] == "ml_strategy_agent" and item["status"] == "failed"
        for item in result["agent_results"]
    )


def test_missing_evidence_in_findings_handled(sample_df, tmp_path):
    """Test that findings without evidence dict don't break the workflow."""
    from app.agents.advisor_workflow import _summarize_evidence_for_llm

    evidence = {
        "profile": {"items": [{"name": f"item_{i}"} for i in range(100)]},
        "eda": None,  # Missing eda
    }

    summary = _summarize_evidence_for_llm(evidence, max_items=50)

    assert len(summary["profile"]["items"]) == 50
    assert summary["eda"] is None
