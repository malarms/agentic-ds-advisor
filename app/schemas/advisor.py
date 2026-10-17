from typing import Any, Literal

from pydantic import BaseModel, Field


Modality = Literal["tabular", "image", "audio"]


class DatasetSpec(BaseModel):
    name: str
    modality: Modality = "tabular"
    objective: str
    row_count: int | None = None
    column_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisFinding(BaseModel):
    agent: str
    title: str
    observation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    title: str
    approach: str
    rationale: str
    supporting_findings: list[str] = Field(default_factory=list)
    validation_strategy: str
    suggested_metrics: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class RecommendationBundle(BaseModel):
    task_interpretation: str
    recommendations: list[Recommendation]
    assumptions: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    agent: str
    status: Literal["success", "failed", "skipped"]
    summary: str = ""
    findings: list[AnalysisFinding] = Field(default_factory=list)
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)
