
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.agents.planner import create_analysis_plan
from app.schemas.analysis_plan import AnalysisPlan
from app.tools.profiling import profile_dataset
from app.tools.data_quality import analyze_data_quality
from app.tools.eda import perform_eda
from app.tools.model_training import train_and_compare_models


class AnalysisState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    request: str
    df: pd.DataFrame
    plan: AnalysisPlan
    results: dict[str, Any]


def build_analysis_graph(llm=None):
    """Build and compile the data science workflow."""

    def plan_node(state: AnalysisState) -> dict:
        df = state["df"]

        plan = create_analysis_plan(
            request=state["request"],
            columns=[str(column) for column in df.columns],
            llm=llm,
        )

        return {
            "plan": plan,
            "results": {},
        }

    def profile_node(state: AnalysisState) -> dict:
        plan = state["plan"]

        if plan.needs_clarification or "profile" not in plan.steps:
            return {}

        results = dict(state.get("results", {}))
        results["profile"] = profile_dataset(state["df"])

        return {"results": results}

    def data_quality_node(state: AnalysisState) -> dict:
        plan = state["plan"]

        if plan.needs_clarification or "data_quality" not in plan.steps:
            return {}

        results = dict(state.get("results", {}))
        results["data_quality"] = analyze_data_quality(state["df"])

        return {"results": results}

    def eda_node(state: AnalysisState) -> dict:
        plan = state["plan"]

        if plan.needs_clarification or "eda" not in plan.steps:
            return {}

        results = dict(state.get("results", {}))
        results["eda"] = perform_eda(state["df"])

        return {"results": results}

    def ml_node(state: AnalysisState) -> dict:
        plan = state["plan"]

        if plan.needs_clarification or "ml" not in plan.steps:
            return {}

        if plan.target_column is None or plan.task is None:
            return {}

        results = dict(state.get("results", {}))
        results["ml"] = train_and_compare_models(
            df=state["df"],
            target_column=plan.target_column,
            task=plan.task,
        )

        return {"results": results}

    graph = StateGraph(AnalysisState)

    graph.add_node("plan", plan_node)
    graph.add_node("profile", profile_node)
    graph.add_node("data_quality", data_quality_node)
    graph.add_node("eda", eda_node)
    graph.add_node("ml", ml_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "profile")
    graph.add_edge("profile", "data_quality")
    graph.add_edge("data_quality", "eda")
    graph.add_edge("eda", "ml")
    graph.add_edge("ml", END)

    return graph.compile()


def run_analysis(
    request: str,
    df: pd.DataFrame,
    llm=None,
) -> dict:
    """Run the workflow and return its plan and tool results."""

    if df.empty:
        raise ValueError("The dataset cannot be empty.")

    graph = build_analysis_graph(llm=llm)

    final_state = graph.invoke(
        {
            "request": request,
            "df": df,
            "results": {},
        }
    )

    return {
        "plan": final_state["plan"],
        "results": final_state.get("results", {}),
    }
