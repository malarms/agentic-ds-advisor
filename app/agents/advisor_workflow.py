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
    run_eda: bool
    run_strategy: bool
    routing_decisions: list[str]
    profile: dict[str, Any]
    data_quality: dict[str, Any]
    eda: dict[str, Any]
    findings: list[dict[str, Any]]
    recommendations: dict[str, Any]
    agent_results: list[dict[str, Any]]
    advisory_status: str
    report: dict[str, Any]


def _record(state: AdvisorState, result: AgentResult) -> dict[str, Any]:
    results = list(state.get("agent_results", []))
    results.append(result.model_dump(mode="json"))
    return {"agent_results": results}


def _supervisor(state: AdvisorState) -> dict[str, Any]:
    """Validate the request and select the analysis path."""
    df = state["df"]
    request = state["request"].strip()

    if not request:
        raise ValueError("Please describe what you want to learn from this dataset.")
    if df.empty or len(df.columns) == 0:
        raise ValueError("The dataset must contain at least one row and one column.")

    lowered = request.lower()
    profile_only = any(
        phrase in lowered
        for phrase in (
            "only profile",
            "profile only",
            "only data quality",
            "data quality only",
        )
    )
    eda_only = any(
        phrase in lowered
        for phrase in (
            "only eda",
            "eda only",
            "exploratory analysis only",
            "only exploratory analysis",
        )
    )

    # Full advisory analysis is the default. Explicitly narrow requests
    # can skip stages that are not relevant to the user's stated objective.
    run_eda = not profile_only
    run_strategy = not profile_only and not eda_only

    decisions = ["Run dataset profiling and data-quality analysis."]
    if run_eda:
        decisions.append("Run exploratory data analysis.")
    else:
        decisions.append("Skip EDA because the request explicitly limits scope.")
    if run_strategy:
        decisions.append("Consider recommendations after the evidence gate.")
    else:
        decisions.append("Skip ML strategy recommendations because the request limits scope.")

    spec = DatasetSpec(
        name="in-memory dataset",
        modality="tabular",
        objective=request,
        row_count=len(df),
        column_count=len(df.columns),
        columns=[str(column) for column in df.columns],
    )

    return {
        "dataset": spec,
        "run_eda": run_eda,
        "run_strategy": run_strategy,
        "routing_decisions": decisions,
        "findings": [],
        "agent_results": [],
        "recommendations": {},
        "advisory_status": "in_progress",
    }


def _dataset_analyst(state: AdvisorState) -> dict[str, Any]:
    try:
        df = state["df"]
        profile = profile_dataset(df)
        quality = analyze_data_quality(df)

        findings = [
            AnalysisFinding(
                agent="dataset_analyst",
                title="Dataset structure",
                observation=(
                    f"The dataset contains {len(df)} rows and "
                    f"{len(df.columns)} columns."
                ),
                evidence=profile,
            ),
            AnalysisFinding(
                agent="dataset_analyst",
                title="Data quality",
                observation=(
                    "Data-quality checks were computed for missing values, "
                    "duplicates, constant columns and detected issues."
                ),
                evidence=quality,
            ),
        ]
        update = {
            "profile": profile,
            "data_quality": quality,
            "findings": [finding.model_dump(mode="json") for finding in findings],
        }
        update.update(
            _record(
                {**state, **update},
                AgentResult(
                    agent="dataset_analyst",
                    status="success",
                    summary="Dataset profile and data-quality checks completed.",
                    findings=findings,
                ),
            )
        )
        return update
    except Exception as exc:
        update = _record(
            state,
            AgentResult(
                agent="dataset_analyst",
                status="failed",
                summary="Dataset profiling or data-quality analysis failed.",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        update["advisory_status"] = "blocked"
        return update


def _eda_agent(state: AdvisorState) -> dict[str, Any]:
    try:
        output_dir = state.get("reports_dir", "reports")
        eda = perform_eda(state["df"], output_dir=output_dir)

        finding = AnalysisFinding(
            agent="eda_agent",
            title="Exploratory analysis",
            observation=(
                "Numeric summaries, categorical distributions and available "
                "correlations were computed."
            ),
            evidence=eda,
            limitations=[
                "EDA describes the observed sample; it does not establish causation.",
                "A small or unrepresentative dataset may not support generalisation.",
            ],
        )
        findings = list(state.get("findings", []))
        findings.append(finding.model_dump(mode="json"))

        artifacts = [
            str(path)
            for path in Path(output_dir).glob("*")
            if path.is_file()
        ]
        update = {"eda": eda, "findings": findings}
        update.update(
            _record(
                {**state, **update},
                AgentResult(
                    agent="eda_agent",
                    status="success",
                    summary="Exploratory analysis completed.",
                    findings=[finding],
                    artifacts=artifacts,
                ),
            )
        )
        return update
    except Exception as exc:
        update = _record(
            state,
            AgentResult(
                agent="eda_agent",
                status="failed",
                summary="Exploratory analysis failed.",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        update["advisory_status"] = "blocked"
        return update


def _evidence_gate(state: AdvisorState) -> dict[str, Any]:
    """Decide whether the collected evidence supports further recommendations."""
    results = state.get("agent_results", [])
    status_by_agent = {
        result["agent"]: result["status"]
        for result in results
    }

    required_agents = ["dataset_analyst"]
    if state.get("run_eda", True):
        required_agents.append("eda_agent")

    failed = [
        name for name in required_agents
        if status_by_agent.get(name) != "success"
    ]

    if failed:
        decisions = list(state.get("routing_decisions", []))
        decisions.append(
            "Evidence gate blocked recommendations because required analysis "
            f"did not succeed: {', '.join(failed)}."
        )
        return {
            "run_strategy": False,
            "advisory_status": "blocked",
            "routing_decisions": decisions,
        }

    if not state.get("run_strategy", True):
        return {"advisory_status": "analysis_only"}

    if not state.get("findings"):
        decisions = list(state.get("routing_decisions", []))
        decisions.append("Evidence gate blocked recommendations because no findings were available.")
        return {
            "run_strategy": False,
            "advisory_status": "blocked",
            "routing_decisions": decisions,
        }

    decisions = list(state.get("routing_decisions", []))
    decisions.append("Evidence gate passed; the strategy agent may use the collected findings.")
    return {
        "advisory_status": "in_progress",
        "routing_decisions": decisions,
    }


def _route_after_dataset_analyst(state: AdvisorState) -> str:
    status = state.get("agent_results", [])[-1].get("status")
    if status != "success":
        return "evidence_gate"
    return "eda_agent" if state.get("run_eda", True) else "evidence_gate"


def _route_after_evidence_gate(state: AdvisorState) -> str:
    if state.get("run_strategy", False) and state.get("advisory_status") != "blocked":
        return "ml_strategy_agent"
    return "report_agent"


def _strategy_agent(state: AdvisorState) -> dict[str, Any]:
    try:
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
Recommend suitable analytical and ML approaches based on the supplied evidence.
Do not claim that a method has been tested or that any performance has been measured.
Never invent dataset statistics or unsupported findings.

User objective:
{state['request']}

Measured dataset evidence:
{evidence}

Return only a valid JSON object matching this structure:
{{
  "task_interpretation": "what the objective appears to require",
  "recommendations": [
    {{
      "title": "short recommendation title",
      "approach": "specific candidate method or family of methods",
      "rationale": "why it fits the objective and observed evidence",
      "supporting_findings": ["exact finding titles or observations from the evidence"],
      "validation_strategy": "appropriate validation approach and why",
      "suggested_metrics": ["metrics appropriate to the task"],
      "risks": ["risks or assumptions to check"],
      "next_steps": ["concrete next actions"]
    }}
  ],
  "assumptions": ["explicit assumptions, not facts"],
  "clarification_questions": ["focused questions if essential information is missing"]
}}

Rules:
- Provide 2 to 4 useful recommendations, ordered by priority.
- Do not assume the task is supervised or that a target exists.
- If the objective or target is unclear, ask focused questions and state assumptions.
- Include baselines and trade-offs, not just algorithm names.
- Do not report experimental results or invent citations.
- Return JSON only, without Markdown fences.
"""
        response = llm.invoke(prompt)
        if not isinstance(response.content, str):
            raise ValueError("The strategy agent must return a text JSON object.")

        bundle = RecommendationBundle.model_validate_json(response.content)

        # Accept an exact title, observation, or exact title-plus-observation
        # combination. Every reference must still match a generated finding.
        findings = state.get("findings", [])

        def is_supported_reference(reference: str) -> bool:
            for finding in findings:
                title = str(finding.get("title", ""))
                observation = str(finding.get("observation", ""))

                if reference == title or reference == observation:
                    return True

                if (
                    title
                    and observation
                    and reference == f"{title}: {observation}"
                ):
                    return True

            return False

        for recommendation in bundle.recommendations:
            unknown = [
                reference
                for reference in recommendation.supporting_findings
                if not is_supported_reference(reference)
            ]
            if unknown:
                raise ValueError(
                    "Recommendation references unsupported findings: "
                    + ", ".join(unknown)
                )

        update = {"recommendations": bundle.model_dump(mode="json")}
        update.update(
            _record(
                {**state, **update},
                AgentResult(
                    agent="ml_strategy_agent",
                    status="success",
                    summary=(
                        f"Generated {len(bundle.recommendations)} "
                        "evidence-checked recommendations."
                    ),
                ),
            )
        )
        return update
    except Exception as exc:
        update = _record(
            state,
            AgentResult(
                agent="ml_strategy_agent",
                status="failed",
                summary="Recommendation generation failed.",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        update["recommendations"] = {}
        update["advisory_status"] = "partial"
        return update


def _report_agent(state: AdvisorState) -> dict[str, Any]:
    results = list(state.get("agent_results", []))
    statuses = [item.get("status") for item in results]
    recommendation_status = next(
        (
            item.get("status")
            for item in reversed(results)
            if item.get("agent") == "ml_strategy_agent"
        ),
        None,
    )

    if state.get("advisory_status") == "blocked":
        final_status = "blocked"
    elif recommendation_status == "failed" or "failed" in statuses:
        final_status = "partial"
    elif state.get("run_strategy", False) and state.get("recommendations"):
        final_status = "complete"
    else:
        final_status = "analysis_only"

    report = {
        "dataset": state["dataset"].model_dump(mode="json"),
        "findings": state.get("findings", []),
        "recommendations": state.get("recommendations", {}),
        "agent_results": results,
        "advisory_status": final_status,
        "routing_decisions": state.get("routing_decisions", []),
    }
    result = AgentResult(
        agent="report_agent",
        status="success",
        summary=f"Report assembled with status: {final_status}.",
    )
    update = {
        "report": report,
        "advisory_status": final_status,
    }
    update.update(_record({**state, **update}, result))
    return update


def build_advisor_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("dataset_analyst", _dataset_analyst)
    graph.add_node("eda_agent", _eda_agent)
    graph.add_node("evidence_gate", _evidence_gate)
    graph.add_node("ml_strategy_agent", _strategy_agent)
    graph.add_node("report_agent", _report_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "dataset_analyst")
    graph.add_conditional_edges(
        "dataset_analyst",
        _route_after_dataset_analyst,
        {
            "eda_agent": "eda_agent",
            "evidence_gate": "evidence_gate",
        },
    )
    graph.add_edge("eda_agent", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate",
        _route_after_evidence_gate,
        {
            "ml_strategy_agent": "ml_strategy_agent",
            "report_agent": "report_agent",
        },
    )
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
        "dataset": final_state["dataset"].model_dump(mode="json"),
        "findings": final_state.get("findings", []),
        "recommendations": final_state.get("recommendations", {}),
        "agent_results": final_state.get("agent_results", []),
        "report": final_state.get("report", {}),
        "advisory_status": final_state.get("advisory_status", "unknown"),
        "routing_decisions": final_state.get("routing_decisions", []),
    }
