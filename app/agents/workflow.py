
from typing import Any, Callable, TypedDict

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
    execution_status: dict[str, dict[str, str]]


def execute_step(
    state: AnalysisState,
    step: str,
    operation: Callable[[], Any],
) -> dict:
    """Run one requested tool and record its success or failure."""

    results = dict(state.get("results", {}))
    statuses = dict(state.get("execution_status", {}))
    plan = state["plan"]

    if plan.needs_clarification:
        statuses[step] = {
            "status": "blocked_by_clarification",
        }
        return {
            "results": results,
            "execution_status": statuses,
        }

    if step not in plan.steps:
        statuses[step] = {"status": "skipped"}
        return {
            "results": results,
            "execution_status": statuses,
        }

    try:
        results[step] = operation()
        statuses[step] = {"status": "success"}
    except Exception as exc:
        # Record the failure explicitly and let independent steps continue.
        statuses[step] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "results": results,
        "execution_status": statuses,
    }


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
            "execution_status": {},
        }

    def profile_node(state: AnalysisState) -> dict:
        return execute_step(
            state,
            "profile",
            lambda: profile_dataset(state["df"]),
        )

    def data_quality_node(state: AnalysisState) -> dict:
        return execute_step(
            state,
            "data_quality",
            lambda: analyze_data_quality(state["df"]),
        )

    def eda_node(state: AnalysisState) -> dict:
        return execute_step(
            state,
            "eda",
            lambda: perform_eda(state["df"]),
        )

    def ml_node(state: AnalysisState) -> dict:
        plan = state["plan"]

        def train_models():
            if plan.target_column is None or plan.task is None:
                raise ValueError(
                    "ML requires a target column and a task."
                )

            return train_and_compare_models(
                df=state["df"],
                target_column=plan.target_column,
                task=plan.task,
            )

        return execute_step(state, "ml", train_models)

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
    """Run the workflow and return the plan, results and statuses."""

    if df.empty:
        raise ValueError("The dataset cannot be empty.")

    graph = build_analysis_graph(llm=llm)

    final_state = graph.invoke(
        {
            "request": request,
            "df": df,
            "results": {},
            "execution_status": {},
        }
    )

    return {
        "plan": final_state["plan"],
        "results": final_state.get("results", {}),
        "execution_status": final_state.get("execution_status", {}),
    }
