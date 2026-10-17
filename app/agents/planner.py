
import json

from pydantic import ValidationError

from app.agents.llm_client import get_llm
from app.schemas.analysis_plan import AnalysisPlan


MAX_PLAN_ATTEMPTS = 2


def create_analysis_plan(
    request: str,
    columns: list[str],
    llm=None,
) -> AnalysisPlan:
    """Generate a validated plan, retrying invalid LLM responses once."""

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

STRICT RULES:
1. Use only the approved step names.
2. target_column must exactly match an available column or be null.
3. task must be "classification", "regression", or null.
4. If supervised prediction was not requested, omit "ml" and set
   target_column and task to null.
5. Never put a description or objective in the task field.
6. If the target or task is unclear, set needs_clarification to true,
   ask a clarification_question, and omit "ml".
7. Do not claim to have executed any tools or analyzed the data.
8. Include profile and data_quality when appropriate.
9. Include eda when the user requests exploratory analysis.
10. Keep steps in a sensible execution order.

Required JSON fields:
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

    current_prompt = prompt
    last_error = None

    for attempt in range(MAX_PLAN_ATTEMPTS):
        response = llm.invoke(current_prompt)
        content = response.content

        try:
            if not isinstance(content, str):
                raise ValueError(
                    "The LLM response must be a text JSON object."
                )

            plan = AnalysisPlan.model_validate_json(content)

        except (ValidationError, ValueError) as exc:
            last_error = exc

            if attempt + 1 < MAX_PLAN_ATTEMPTS:
                current_prompt = f"""
{prompt}

YOUR PREVIOUS RESPONSE WAS INVALID.
Validation error:
{str(exc)}

Return a corrected JSON object that follows every rule and field
requirement above. Return JSON only. Do not explain the correction.
"""
            continue

        break
    else:
        raise ValueError(
            f"The LLM returned an invalid analysis plan after "
            f"{MAX_PLAN_ATTEMPTS} attempts. Please retry the request."
        ) from last_error

    # Never allow the LLM to invent a target column.
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

    # Supervised ML requires a valid target and task.
    if "ml" in plan.steps:
        if plan.target_column is None or plan.task is None:
            plan.steps = [step for step in plan.steps if step != "ml"]
            plan.needs_clarification = True

            if not plan.clarification_question:
                plan.clarification_question = (
                    "Which dataset column should the model predict, "
                    "and is this a classification or regression task?"
                )

    # Do not retain irrelevant ML settings when ML is not requested.
    if "ml" not in plan.steps:
        plan.target_column = None
        plan.task = None

    return plan
