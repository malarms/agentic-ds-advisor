
import json

import pytest

from app.agents.planner import create_analysis_plan


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return FakeResponse(self.response)


def test_planner_creates_valid_eda_plan():
    response = {
        "objective": "Explore customer data",
        "steps": ["profile", "data_quality", "eda"],
        "target_column": None,
        "task": None,
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "The user wants to understand the dataset.",
    }

    llm = FakeLLM(json.dumps(response))

    plan = create_analysis_plan(
        request="Explore this dataset",
        columns=["age", "income", "city"],
        llm=llm,
    )

    assert plan.steps == ["profile", "data_quality", "eda"]
    assert plan.needs_clarification is False
    assert "AVAILABLE DATASET COLUMNS" in llm.last_prompt


def test_planner_rejects_unknown_step():
    response = {
        "objective": "Analyze data",
        "steps": ["execute_arbitrary_code"],
        "target_column": None,
        "task": None,
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "Test invalid step.",
    }

    llm = FakeLLM(json.dumps(response))

    with pytest.raises(ValueError, match="invalid analysis plan"):
        create_analysis_plan(
            request="Analyze data",
            columns=["age"],
            llm=llm,
        )


def test_invalid_target_requires_clarification():
    response = {
        "objective": "Predict churn",
        "steps": ["profile", "ml"],
        "target_column": "churn",
        "task": "classification",
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "Predict customer churn.",
    }

    llm = FakeLLM(json.dumps(response))

    plan = create_analysis_plan(
        request="Predict churn",
        columns=["age", "income"],
        llm=llm,
    )

    assert "ml" not in plan.steps
    assert plan.target_column is None
    assert plan.needs_clarification is True
    assert plan.clarification_question is not None


def test_empty_request_is_rejected():
    llm = FakeLLM("{}")

    with pytest.raises(ValueError, match="request cannot be empty"):
        create_analysis_plan(
            request=" ",
            columns=["age"],
            llm=llm,
        )

class SequenceFakeLLM:
    """Return different responses on successive calls."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses[self.call_count]
        self.call_count += 1
        return FakeResponse(response)


def test_planner_recovers_after_invalid_first_response():
    invalid_response = json.dumps({
        "objective": "Analyze data",
        "steps": ["run_arbitrary_code"],
        "target_column": None,
        "task": None,
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "Invalid step for testing.",
    })

    valid_response = json.dumps({
        "objective": "Explore customer data",
        "steps": ["profile", "data_quality", "eda"],
        "target_column": None,
        "task": None,
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "The user requested exploratory analysis.",
    })

    llm = SequenceFakeLLM([invalid_response, valid_response])

    plan = create_analysis_plan(
        request="Explore this dataset",
        columns=["age", "income", "city"],
        llm=llm,
    )

    assert llm.call_count == 2
    assert plan.steps == ["profile", "data_quality", "eda"]
    assert "PREVIOUS RESPONSE WAS INVALID" in llm.prompts[1]


def test_planner_stops_after_two_invalid_responses():
    invalid_response = json.dumps({
        "objective": "Analyze data",
        "steps": ["run_arbitrary_code"],
        "target_column": None,
        "task": None,
        "needs_clarification": False,
        "clarification_question": None,
        "rationale": "Invalid step for testing.",
    })

    llm = SequenceFakeLLM([invalid_response, invalid_response])

    with pytest.raises(ValueError, match="invalid analysis plan"):
        create_analysis_plan(
            request="Analyze data",
            columns=["age", "income"],
            llm=llm,
        )

    assert llm.call_count == 2
