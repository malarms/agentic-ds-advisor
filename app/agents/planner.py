
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

Create an analysis plan for the user's request and dataset.

USER REQUEST:
{request}

AVAILABLE DATASET COLUMNS:
{json.dumps(columns)}

APPROVED STEPS:
- profile: inspect dataset dimensions, types and missing values
- data_quality: inspect missing values, duplicates and constant columns
- eda: calculate descriptive statistics, correlations and plots
- ml: train and compare supervised machine learning models

RULES:
1. Return exactly one JSON object, without Markdown fences.
2. Use only the approved step names.
3. Do not invent dataset columns or claim to have analyzed data.
4. Include profile and data_quality when appropriate.
5. Include eda when the user asks to explore or understand the data.
6. Include ml only when supervised prediction is requested.
7. For ML, identify an actual target column from the available columns.
8. Choose classification for categorical targets and regression for
   numeric targets when that interpretation is appropriate.
9. If the target or task is unclear, request clarification instead
   of guessing. Do not include ml until the ambiguity is resolved.
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

    # Never allow the LLM to invent a target column.
    if plan.target_column is not None:
        if plan.target_column not in columns:
            plan.steps = [step for step in plan.steps if step != "ml"]
            plan.target_column = None
            plan.task = None
            plan.needs_clarification = True
            plan.clarification_question = (
                "The proposed target column is not present in the dataset. "
                f"Available columns are: {', '.join(columns)}. "
                "Which column should be predicted?"
            )

    # Supervised ML must have a valid target and task.
    if "ml" in plan.steps:
        if plan.target_column is None or plan.task is None:
            plan.steps = [step for step in plan.steps if step != "ml"]
            plan.needs_clarification = True

            if not plan.clarification_question:
                plan.clarification_question = (
                    "Which dataset column should the model predict, "
                    "and is this a classification or regression task?"
                )

    return plan
