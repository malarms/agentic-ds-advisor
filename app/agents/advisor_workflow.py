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
from app.tools.multimodal import (
    profile_image_dataset,
    profile_audio_dataset,
    perform_image_eda,
    perform_audio_eda,
)


class AdvisorState(TypedDict, total=False):
    request: str
    df: pd.DataFrame | None
    input_path: str
    modality: str
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
    """Validate input and select the appropriate modality-specific path."""
    df = state.get("df")
    input_path = state.get("input_path")
    modality = state.get("modality", "tabular")
    request = state["request"].strip()

    if not request:
        raise ValueError("Please describe what you want to learn from this dataset.")

    if modality == "tabular":
        if df is None or df.empty or len(df.columns) == 0:
            raise ValueError("The tabular dataset must contain rows and columns.")
    elif modality in {"image", "audio"}:
        if not input_path or not Path(input_path).exists():
            raise ValueError("Image/audio analysis requires an existing input path.")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    lowered = request.lower()
    profile_only = any(
        phrase in lowered
        for phrase in (
            "only profile", "profile only", "only data quality", "data quality only"
        )
    )
    eda_only = any(
        phrase in lowered
        for phrase in (
            "only eda", "eda only", "exploratory analysis only",
            "only exploratory analysis",
        )
    )

    run_eda = not profile_only
    run_strategy = not profile_only and not eda_only

    decisions = [f"Use the {modality} analysis adapter."]
    decisions.append("Run dataset profiling and data-quality analysis.")
    if run_eda:
        decisions.append("Run modality-specific exploratory analysis.")
    else:
        decisions.append("Skip EDA because the request explicitly limits scope.")
    if run_strategy:
        decisions.append("Consider recommendations after the evidence gate.")
    else:
        decisions.append("Skip strategy recommendations because the request limits scope.")

    dataset = DatasetSpec(
        name=Path(input_path).name if input_path else "in-memory dataset",
        modality=modality,
        objective=request,
        row_count=len(df) if df is not None else None,
        column_count=len(df.columns) if df is not None else None,
        columns=[str(column) for column in df.columns] if df is not None else [],
        metadata={"source_path": input_path} if input_path else {},
    )

    return {
        "dataset": dataset,
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
        modality = state["dataset"].modality
        df = state.get("df")

        if modality == "tabular":
            profile = profile_dataset(df)
            quality = analyze_data_quality(df)
            item_count = len(df)
            structure_observation = (
                f"The dataset contains {len(df)} rows and {len(df.columns)} columns."
            )
            quality_observation = (
                "Data-quality checks were computed for missing values, duplicates, "
                "constant columns and detected issues."
            )
        elif modality == "image":
            profile = profile_image_dataset(state["input_path"])
            quality = {
                "files_discovered": profile["files_discovered"],
                "valid_files": profile["valid_count"],
                "invalid_files": profile["invalid_files"],
            }
            item_count = profile["valid_count"]
            structure_observation = (
                f"The image dataset contains {item_count} readable images."
            )
            quality_observation = (
                f"Image validation found {profile['invalid_count']} unreadable "
                f"files out of {profile['files_discovered']} discovered."
            )
        else:
            profile = profile_audio_dataset(state["input_path"])
            quality = {
                "files_discovered": profile["files_discovered"],
                "valid_files": profile["valid_count"],
                "invalid_files": profile["invalid_files"],
            }
            item_count = profile["valid_count"]
            structure_observation = (
                f"The audio dataset contains {item_count} readable audio files."
            )
            quality_observation = (
                f"Audio validation found {profile['invalid_count']} unreadable "
                f"files out of {profile['files_discovered']} discovered."
            )

        dataset = state["dataset"].model_copy(
            update={
                "row_count": item_count,
                "metadata": {
                    **state["dataset"].metadata,
                    "profile_summary": {
                        key: value for key, value in profile.items()
                        if key not in {"items", "invalid_files"}
                    },
                },
            }
        )

        findings = [
            AnalysisFinding(
                agent="dataset_analyst",
                title="Dataset structure",
                observation=structure_observation,
                evidence=profile,
            ),
            AnalysisFinding(
                agent="dataset_analyst",
                title="Data quality",
                observation=quality_observation,
                evidence=quality,
                limitations=[
                    "File-level checks do not establish semantic correctness or label quality."
                ],
            ),
        ]

        update = {
            "dataset": dataset,
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
                    summary=f"{modality.title()} profiling and validation completed.",
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
                summary="Dataset profiling or validation failed.",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        update["advisory_status"] = "blocked"
        return update


def _eda_agent(state: AdvisorState) -> dict[str, Any]:
    try:
        output_dir = state.get("reports_dir", "reports")
        modality = state["dataset"].modality

        if modality == "tabular":
            eda = perform_eda(state["df"], output_dir=output_dir)
        elif modality == "image":
            eda = perform_image_eda(state["input_path"], output_dir=output_dir)
        else:
            eda = perform_audio_eda(state["input_path"], output_dir=output_dir)

        finding = AnalysisFinding(
            agent="eda_agent",
            title="Exploratory analysis",
            observation=(
                f"Modality-specific exploratory analysis was computed for "
                f"{eda.get('item_count', eda.get('row_count', 0))} items."
            ),
            evidence=eda,
            limitations=[
                "Descriptive measurements do not establish causation.",
                "File-level statistics do not establish semantic content or label quality.",
            ],
        )
        findings = list(state.get("findings", []))
        findings.append(finding.model_dump(mode="json"))

        artifacts = [
            str(file)
            for file in Path(output_dir).glob("*")
            if file.is_file()
        ]
        update = {"eda": eda, "findings": findings}
        update.update(
            _record(
                {**state, **update},
                AgentResult(
                    agent="eda_agent",
                    status="success",
                    summary=f"{modality.title()} exploratory analysis completed.",
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


def _summarize_evidence_for_llm(evidence: dict[str, Any], max_items: int = 50) -> dict[str, Any]:
    """Summarize evidence to fit within LLM context limits."""
    import copy

    summary = copy.deepcopy(evidence)

    # Trim profile items if present and too large
    if "profile" in summary and isinstance(summary["profile"], dict):
        profile = summary["profile"]
        if "items" in profile and isinstance(profile["items"], list):
            if len(profile["items"]) > max_items:
                profile["items"] = profile["items"][:max_items]
                profile["_items_truncated"] = f"Showing first {max_items} of {len(profile['items'])} items"

    # Trim eda items if present and too large
    if "eda" in summary and isinstance(summary["eda"], dict):
        eda = summary["eda"]
        if "items" in eda and isinstance(eda["items"], list):
            if len(eda["items"]) > max_items:
                eda["items"] = eda["items"][:max_items]
                eda["_items_truncated"] = f"Showing first {max_items} of {len(eda['items'])} items"

    return summary


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

        # Summarize evidence to avoid LLM context limits with large datasets
        evidence = _summarize_evidence_for_llm(evidence)

        valid_finding_titles = [
            str(finding.get("title", ""))
            for finding in state.get("findings", [])
            if finding.get("title")
        ]

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
      "supporting_findings": ["exact finding titles copied verbatim from the valid titles list"],
      "validation_strategy": "appropriate validation approach and why",
      "suggested_metrics": ["metrics appropriate to the task"],
      "risks": ["risks or assumptions to check"],
      "next_steps": ["concrete next actions"]
    }}
  ],
  "assumptions": ["explicit assumptions, not facts"],
  "clarification_questions": ["focused questions if essential information is missing"]
}}

Valid finding titles (copy these exactly; do not cite observations or invent references):
{valid_finding_titles}

Rules:
- Every supporting_findings entry must exactly match one title in the valid finding titles list.
- Use finding titles only, never paraphrased observations.
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

        # Recommendations must cite exact generated finding titles.
        # Do not accept paraphrases or arbitrary observation text.
        valid_finding_titles = {
            str(finding.get("title", ""))
            for finding in state.get("findings", [])
            if finding.get("title")
        }

        def is_supported_reference(reference: str) -> bool:
            return reference in valid_finding_titles

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
    df: pd.DataFrame | None = None,
    llm=None,
    reports_dir: str = "reports",
    input_path: str | None = None,
    modality: str = "tabular",
) -> dict[str, Any]:
    """Run advisory analysis on a DataFrame or an image/audio input path."""
    if modality == "tabular":
        if df is None:
            raise ValueError("Tabular analysis requires a DataFrame.")
        if df.empty or len(df.columns) == 0:
            raise ValueError("The dataset cannot be empty.")
    elif modality not in {"image", "audio"}:
        raise ValueError(f"Unsupported modality: {modality}")
    elif not input_path:
        raise ValueError(f"{modality.title()} analysis requires input_path.")

    graph = build_advisor_graph()
    initial_state = {
        "request": request,
        "df": df,
        "input_path": input_path or "",
        "modality": modality,
        "llm": llm,
        "reports_dir": reports_dir,
        "findings": [],
        "agent_results": [],
    }
    final_state = graph.invoke(initial_state)

    return {
        "dataset": final_state["dataset"].model_dump(mode="json"),
        "findings": final_state.get("findings", []),
        "recommendations": final_state.get("recommendations", {}),
        "agent_results": final_state.get("agent_results", []),
        "report": final_state.get("report", {}),
        "advisory_status": final_state.get("advisory_status", "unknown"),
        "routing_decisions": final_state.get("routing_decisions", []),
    }
