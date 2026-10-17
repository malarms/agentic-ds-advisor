
from typing import Literal, Optional

from pydantic import BaseModel, Field


AnalysisStep = Literal[
    "profile",
    "data_quality",
    "eda",
    "ml",
]


class AnalysisPlan(BaseModel):
    """Validated plan proposed by the LLM."""

    objective: str = Field(
        description="The user's analytical objective."
    )

    steps: list[AnalysisStep] = Field(
        description="Ordered analysis steps using approved tools."
    )

    target_column: Optional[str] = Field(
        default=None,
        description="Prediction target, if supervised ML is requested."
    )

    task: Optional[Literal["classification", "regression"]] = Field(
        default=None,
        description="ML task, if supervised ML is requested."
    )

    needs_clarification: bool = Field(
        default=False,
        description="Whether the user must clarify the request."
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask if essential information is missing."
    )

    rationale: str = Field(
        description="Brief explanation of the proposed plan."
    )
