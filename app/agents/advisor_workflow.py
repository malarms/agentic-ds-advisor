from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.agents.llm_client import get_llm
from app.schemas.advisor import (
    AgentResult,
    AnalysisFinding,
    DatasetSpec,
    RecommendationBundle,
)
from app.tools.data_quality import analyze_data_quality
from app.tools.eda import perform_eda
from app.tools.profiling import profile_dataset


class AdvisorState(TypedDict, total=False):
    request: str
    df: pd.DataFrame
    dataset: DatasetSpec
    llm: Any
    reports_dir: str
    profile: dict[str, Any]
    data_quality: dict[str, Any]
    eda: dict[str, Any]
    findings: list[dict[str, Any]]
    recommendations: dict[str, Any]
    agent_results: list[dict[str, Any]]
    report: dict[str, Any]


def _record(state: AdvisorState, result: AgentResult) -> dict[str, Any]:
    results = list(state.get("agent_results", []))
    results.append(result.model_dump(mode="json"))
    return {"agent_results": results}


def _supervisor(state: AdvisorState) -> dict[str, Any]:
    """Route the initial implementation through the tabular advisory path."""
    df = state["df"]
    request = state["request"].strip()
    if not request:
        raise ValueError("Please describe what you want to learn from this dataset.")
    if df.empty or len(df.columns) == 0:
        raise ValueError("The dataset must contain at least one row and one column.")

    spec = DatasetSpec(
        name="in-memory dataset",
        modality="tabular",
        objective=request,
        row_count=len(df),
        column_count=len(df.columns),
        columns=[str(c) for c in df.columns],
    )
    return {
        "dataset": spec,
        "findings": [],
        "agent_results": [],
        "recommendations": {},
    }


def _dataset_analyst(state: AdvisorState) -> dict[str, Any]:
    try:
        profile = profile_dataset(state["df"])
        quality = analyze_data_quality(state["df"])
        findings = [
            AnalysisFinding(
                agent="dataset_analyst",
                title="Dataset structure",
                observation=(
                    f"The dataset contains {len(state['df'])} rows and "
                    f"{len(state['df'].columns)} columns."
                ),
                evidence=profile,
            ),
            AnalysisFinding(
                agent="dataset_analyst",
                title="Data quality",
                observation="Computed data-quality checks for missing values, duplicates and constant columns.",
                evidence=quality,
            ),
        ]
        update = {
            "profile": profile,
            "data_quality": quality,
            "findings": [f.model_dump(mode="json") for f in findings],
        }
        result = AgentResult(
            agent="dataset_analyst",
            status="success",
            summary="Dataset profile and data-quality checks completed.",
            findings=findings,
        )
        update.update(_record({**state, **update}, result))
        return update
    except Exception as exc:
        update = _record(
            state,
            AgentResult(
                agent="dataset_analyst",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        update["findings"] = list(state.get("findings", []))
        return update


def _eda_agent(state: AdvisorState) -> dict[str, Any]:
    try:
        output_dir = state.get("reports_dir", "reports")
        eda = perform_eda(state["df"], output_dir=output_dir)
        findings = list(state.get("findings", []))
        findings.append(
            AnalysisFinding(
                agent="eda_agent",
                title="Exploratory analysis",
                observation="Computed numeric summaries, categorical distributions and available correlations.",
                evidence=eda,
                limitations=[
                    "EDA describes observed data; it does not establish causation.",
                    "Small or unrepresentative datasets may not support reliable generalisation.",
                ],
            ).model_dump(mode="json")
        )
        result = AgentResult(
            agent="eda_agent",
            status="success",
            summary="Exploratory analysis completed.",
            findings=[AnalysisFinding.model_validate(findings[-1])],
            artifacts=[
                str(p)
                for p in Path(output_dir).glob("*")
                if p.is_file()
            ],
        )
        update = {"eda": eda, "findings": findings}
        update.update(_record({**state, **update}, result))
        return update
    except Exception as exc:
        update = _record(
            state,
            AgentResult(
                agent="eda_agent",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        return update


def _strategy_agent(state: AdvisorState) -> dict[str, Any]:
    """Recommend strategies only. This node must never train or evaluate models."""
    llm = state.get("llm") or get_llm()
    evidence = {
        "dataset": state["dataset"].model_dump(mode="json"),
        "profile": state.get("profile", {}),
        "data_quality": state.get("data_quality", {}),
        "eda": state.get("eda", {}),
        "findings": state.get("findings", []),
    }
    prompt = f"""
You are the ML Strategy Agent in a data-science advisory system.
Your job is to recommend approaches, NOT to train models, run experiments,
claim performance results, or invent dataset statistics.

User objective:
{state['request']}

Measured dataset evidence (treat as the only source of dataset-specific facts):
{evidence}

Return only a valid JSON object matching this structure:
{{
  "task_interpretation": "what the objective appears to require",
  "recommendations": [
    {{
      "title": "short recommendation title",
      "approach": "specific candidate method or family of methods",
      "rationale": "why it fits the objective and observed evidence",
      "supporting_findings": ["refer to actual finding titles or observations"],
      "validation_strategy": "appropriate validation approach and why",
      "suggested_metrics": ["metrics appropriate to the task"],
      "risks": ["risks or assumptions to check"],
      "next_steps": ["concrete actions before modelling"]
    }}
  ],
  "assumptions": ["explicit assumptions; do not present them as facts"],
  "clarification_questions": ["questions only if important information is missing"]
}}

Rules:
- Give 2 to 4 useful recommendations, ordered by priority.
- Do not assume a target column exists or that the task is supervised.
- If the objective or target is unclear, say so and ask focused questions.
- Mention baselines and trade-offs, not just model names.
- Do not claim that any model was trained or that any metric was achieved.
- Do not invent citations. Keep advice proportional to the evidence.
- Return JSON only, without Markdown fences.
"""
    response = llm.invoke(prompt)
    content = response.content
    if not isinstance(content, str):
        raise ValueError("The strategy agent must return a text JSON object.")
    bundle = RecommendationBundle.model_validate_json(content)
    update = {"recommendations": bundle.model_dump(mode="json")}
    result = AgentResult(
        agent="ml_strategy_agent",
        status="success",
        summary=f"Generated {len(bundle.recommendations)} strategy recommendations; no models were trained.",
    )
    update.update(_record({**state, **update}, result))
    return update


def _report_agent(state: AdvisorState) -> dict[str, Any]:
    report = {
        "dataset": state["dataset"].model_dump(mode="json"),
        "findings": state.get("findings", []),
        "recommendations": state.get("recommendations", {}),
        "agent_results": state.get("agent_results", []),
        "training_performed": False,
    }
    result = AgentResult(
        agent="report_agent",
        status="success",
        summary="Assembled the advisory report payload.",
    )
    update = {"report": report}
    update.update(_record({**state, **update}, result))
    return update


def build_advisor_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("dataset_analyst", _dataset_analyst)
    graph.add_node("eda_agent", _eda_agent)
    graph.add_node("ml_strategy_agent", _strategy_agent)
    graph.add_node("report_agent", _report_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "dataset_analyst")
    graph.add_edge("dataset_analyst", "eda_agent")
    graph.add_edge("eda_agent", "ml_strategy_agent")
    graph.add_edge("ml_strategy_agent", "report_agent")
    graph.add_edge("report_agent", END)
    return graph.compile()


def run_advisor(
    request: str,
    df: pd.DataFrame,
    llm=None,
    reports_dir: str = "reports",
) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The dataset cannot be empty.")
    graph = build_advisor_graph()
    final_state = graph.invoke(
        {
            "request": request,
            "df": df,
            "llm": llm,
            "reports_dir": reports_dir,
            "findings": [],
            "agent_results": [],
        }
    )
    return {
        "dataset": final_state.get("dataset").model_dump(mode="json"),
        "findings": final_state.get("findings", []),
        "recommendations": final_state.get("recommendations", {}),
        "agent_results": final_state.get("agent_results", []),
        "report": final_state.get("report", {}),
        "training_performed": False,
    }
