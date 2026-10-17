
import json

from pydantic import ValidationError

from app.agents.llm_client import get_llm
from app.schemas.analysis_plan import AnalysisPlan


def create_analysis_plan(
    request: str,
    columns: list[str],
    llm=None,
) -> AnalysisPlan:
    """Ask an LLM to propose a validated data science plan."""

    if not request.strip():
        raise ValueError("The analysis request cannot be empty.")

    if not columns:
        raise ValueError("At least one dataset column is required.")

    if llm is None:
        llm = get_llm()

    prompt = f"""
You are the planning component of an agentic data science platform.

Return exactly one JSON object. Do not include Markdown fences or prose.

USER REQUEST:
{request}

AVAILABLE DATASET COLUMNS:
{json.dumps(columns)}

APPROVED STEPS:
- profile: inspect dataset dimensions, types and missing values
- data_quality: inspect missing values, duplicates and constant columns
- eda: calculate descriptive statistics, correlations and plots
- ml: train and compare supervised machine learning models

STRICT OUTPUT RULES:
1. "steps" must contain only approved step names.
2. "target_column" must be an exact column name from AVAILABLE DATASET COLUMNS or null.
3. "task" must be exactly "classification", "regression", or null.
4. If the user did not request supervised prediction, "task" and
   "target_column" must both be null, and "ml" must not be in "steps".
5. Never put a description, explanation, or analytical objective in "task".
6. If the prediction target or task is unclear, set "needs_clarification"
   to true, supply a "clarification_question", and omit "ml".
7. Do not claim to have inspected data or trained models.
8. Use "profile" and "data_quality" when appropriate.
9. Use "eda" when the user asks to explore or understand the data.
10. Keep steps in a sensible execution order.

Required JSON schema:
{{
  "objective": "string",
  "steps": ["profile", "data_quality"],
  "target_column": null,
  "task": null,
  "needs_clarification": false,
  "clarification_question": null,
  "rationale": "string"
}}
"""

    response = llm.invoke(prompt)
    content = response.content

    if not isinstance(content, str):
        raise ValueError("The LLM response must be a text JSON object.")

    try:
        plan = AnalysisPlan.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise ValueError(
            "The LLM returned an invalid analysis plan. "
            "Please retry the request."
        ) from exc

    # Validate target names independently of the LLM's claims.
    if plan.target_column is not None and plan.target_column not in columns:
        plan.steps = [step for step in plan.steps if step != "ml"]
        plan.target_column = None
        plan.task = None
        plan.needs_clarification = True
        plan.clarification_question = (
            "The proposed target column is not present in the dataset. "
            f"Available columns are: {', '.join(columns)}. "
            "Which column should be predicted?"
        )

    # Supervised ML requires both a valid target and task.
    if "ml" in plan.steps:
        if plan.target_column is None or plan.task is None:
            plan.steps = [step for step in plan.steps if step != "ml"]
            plan.needs_clarification = True

            if not plan.clarification_question:
                plan.clarification_question = (
                    "Which dataset column should the model predict, "
                    "and is this a classification or regression task?"
                )

    # A task or target without an ML step is inconsistent.
    if "ml" not in plan.steps:
        plan.target_column = None
        plan.task = None

    return plan
