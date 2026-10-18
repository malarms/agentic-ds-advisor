import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.offline import get_plotlyjs


def _safe(value: Any) -> str:
    """Escape dynamic text before inserting it into HTML."""
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return html.escape(str(value))


def _chart_html(fig) -> str:
    """Render a responsive Plotly chart without duplicating its JS library."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height="420px",
    )


def _build_charts(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Build general-purpose EDA charts with optional target-aware views."""
    charts: list[tuple[str, str]] = []
    if df is None or df.empty:
        return charts

    # Missingness
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    if len(missing) and (missing > 0).any():
        missing_df = missing[missing > 0].rename("Missing (%)").reset_index()
        missing_df.columns = ["Column", "Missing (%)"]
        fig = px.bar(
            missing_df, x="Column", y="Missing (%)",
            title="Missing values by column", text_auto=".1f",
        )
        charts.append(("Missingness", _chart_html(fig)))

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    # Numeric distributions; keep original scale and add log1p views where valid.
    for column in numeric_df.columns[:6]:
        values = numeric_df[column].dropna()
        if values.empty:
            continue

        frame = pd.DataFrame({str(column): values})
        fig = px.histogram(
            frame, x=str(column), title=f"Distribution: {column}",
            marginal="box", opacity=0.85,
        )
        charts.append((f"Distribution — {column}", _chart_html(fig)))

        # Log1p is valid only when all observed values are non-negative.
        if len(values) > 20 and values.min() >= 0 and values.skew() > 1:
            transformed = pd.DataFrame({
                f"log1p({column})": __import__("numpy").log1p(values)
            })
            fig = px.histogram(
                transformed,
                x=f"log1p({column})",
                title=f"Skew-adjusted distribution: {column}",
                labels={f"log1p({column})": f"log1p({column})"},
            )
            charts.append((f"Log-scaled view — {column}", _chart_html(fig)))

    # Categorical counts. Always include a likely target, even if it appears late.
    categorical_columns = list(categorical_df.columns)
    target_column = next(
        (name for name in ("y", "target", "label", "outcome", "subscribed")
         if name in df.columns and name in categorical_columns),
        None,
    )
    selected_categories = categorical_columns[:4]
    if target_column and target_column not in selected_categories:
        selected_categories = selected_categories[:3] + [target_column]

    for column in selected_categories:
        counts = (
            df[column].fillna("(Missing)").astype(str)
            .value_counts().head(12)
            .rename_axis("Category").reset_index(name="Count")
        )
        if counts.empty:
            continue
        fig = px.bar(
            counts.sort_values("Count"), x="Count", y="Category",
            orientation="h", title=f"Category counts: {column}", text="Count",
        )
        charts.append((f"Categories — {column}", _chart_html(fig)))

    # Target-aware charts: only when the target has a small number of classes.
    if target_column:
        target_values = df[target_column].fillna("(Missing)").astype(str)
        target_counts = (
            target_values.value_counts().rename_axis("Class")
            .reset_index(name="Count")
        )
        fig = px.bar(
            target_counts, x="Class", y="Count", text="Count",
            title=f"Target class distribution: {target_column}",
        )
        charts.append(("Target class distribution", _chart_html(fig)))

        classes = target_values.unique()
        positive_class = next(
            (value for value in ("yes", "true", "1", "positive")
             if value in {str(item).lower() for item in classes}),
            None,
        )
        if positive_class is not None:
            actual_positive = next(
                item for item in classes if str(item).lower() == positive_class
            )
            positive = (target_values == actual_positive).astype(int)

            candidate_columns = [
                column for column in categorical_columns
                if column != target_column and df[column].nunique(dropna=False) <= 20
            ][:5]

            for column in candidate_columns:
                frame = pd.DataFrame({
                    "Category": df[column].fillna("(Missing)").astype(str),
                    "Positive": positive,
                })
                rates = (
                    frame.groupby("Category", dropna=False)
                    .agg(Records=("Positive", "size"),
                         Positive_Rate=("Positive", "mean"))
                    .reset_index()
                )
                # Avoid visually noisy rates from very small groups.
                rates = rates[rates["Records"] >= max(20, int(len(df) * 0.005))]
                if rates.empty:
                    continue
                rates["Positive rate (%)"] = rates["Positive_Rate"] * 100
                fig = px.bar(
                    rates.sort_values("Positive rate (%)"),
                    x="Positive rate (%)", y="Category", orientation="h",
                    text=rates.sort_values("Positive rate (%)")["Positive rate (%)"].round(1),
                    hover_data={"Records": True, "Positive rate (%)": ":.2f"},
                    title=f"{actual_positive} rate by {column}",
                )
                charts.append((f"Target rate — {column}", _chart_html(fig)))

    # Handle the Bank Marketing pdays sentinel as a special case.
    if "pdays" in df.columns and pd.api.types.is_numeric_dtype(df["pdays"]):
        pdays = pd.to_numeric(df["pdays"], errors="coerce")
        status = pd.Series(
            pdays.map(lambda value: "No previous contact (-1)"
                      if value == -1 else
                      "Previous contact recorded" if pd.notna(value) and value >= 0
                      else "Other / missing"),
            name="Previous-contact status",
        )
        status_counts = (
            status.value_counts().rename_axis("Status").reset_index(name="Records")
        )
        fig = px.bar(
            status_counts, x="Status", y="Records", text="Records",
            title="Previous-contact status (pdays)",
        )
        charts.append(("Previous-contact status", _chart_html(fig)))

        prior_days = pdays[pdays >= 0]
        if not prior_days.empty:
            fig = px.histogram(
                pd.DataFrame({"Days since previous contact": prior_days}),
                x="Days since previous contact",
                title="Days since previous contact (excluding -1 sentinel)",
            )
            charts.append(("Previous-contact day distribution", _chart_html(fig)))

    # Correlation heatmap for numeric columns.
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Numeric feature correlations",
        )
        charts.append(("Correlations", _chart_html(fig)))

    return charts


def _build_modality_charts(result: dict[str, Any]) -> list[tuple[str, str]]:
    """Build charts from measured image/audio item metadata."""
    eda = result.get("report", {}).get("eda") or {}
    if not eda:
        eda = next(
            (
                item.get("evidence", {})
                for item in result.get("findings", [])
                if item.get("title") == "Exploratory analysis"
            ),
            {},
        )

    charts = []
    modality = result.get("dataset", {}).get("modality")

    if modality == "image":
        items = eda.get("items", [])
        if items:
            frame = pd.DataFrame(items)
            if {"width", "height"}.issubset(frame.columns):
                frame["dimensions"] = (
                    frame["width"].astype(str) + " × " + frame["height"].astype(str)
                )
                counts = frame["dimensions"].value_counts().head(15).rename_axis(
                    "Dimensions"
                ).reset_index(name="Images")
                fig = px.bar(
                    counts.sort_values("Images"),
                    x="Images",
                    y="Dimensions",
                    orientation="h",
                    title="Image dimension distribution",
                )
                charts.append(("Image dimensions", _chart_html(fig)))

            if "file_size_bytes" in frame.columns:
                fig = px.histogram(
                    frame,
                    x="file_size_bytes",
                    title="Image file sizes",
                    labels={"file_size_bytes": "File size (bytes)"},
                )
                charts.append(("Image file sizes", _chart_html(fig)))

    elif modality == "audio":
        items = eda.get("items", [])
        if items:
            frame = pd.DataFrame(items)
            if "duration_seconds" in frame.columns:
                fig = px.histogram(
                    frame,
                    x="duration_seconds",
                    title="Audio duration distribution",
                    labels={"duration_seconds": "Duration (seconds)"},
                )
                charts.append(("Audio durations", _chart_html(fig)))

            if {"sample_rate_hz", "channels"}.issubset(frame.columns):
                counts = frame.groupby(
                    ["sample_rate_hz", "channels"]
                ).size().reset_index(name="Files")
                counts["Configuration"] = (
                    counts["sample_rate_hz"].astype(str) + " Hz, "
                    + counts["channels"].astype(str) + " channel(s)"
                )
                fig = px.bar(
                    counts,
                    x="Configuration",
                    y="Files",
                    title="Audio recording configurations",
                )
                charts.append(("Audio configurations", _chart_html(fig)))

            if "rms_amplitude" in frame.columns:
                fig = px.histogram(
                    frame,
                    x="rms_amplitude",
                    title="Audio RMS amplitude distribution",
                )
                charts.append(("Audio amplitude", _chart_html(fig)))

    return charts


def generate_html_report(
    df: pd.DataFrame,
    result: dict[str, Any],
    output_path: str = "reports/advisor_report.html",
) -> str:
    """Write a visual HTML report and return the output path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset = result.get("dataset", {})
    findings = result.get("findings", [])
    recommendations_bundle = result.get("recommendations", {})
    recommendations = recommendations_bundle.get("recommendations", [])
    agents = result.get("agent_results", [])

    chart_sections = []
    charts = _build_charts(df) if df is not None else _build_modality_charts(result)
    for index, (title, chart) in enumerate(charts, start=1):
        chart_sections.append(
            f"""
            <section class="chart-card">
              <h3>{_safe(title)}</h3>
              {chart}
            </section>
            """
        )

    # Bundle Plotly once so the report also works offline.
    plotly_script = (
        f"<script>{get_plotlyjs()}</script>" if charts else ""
    )

    finding_sections = []
    for finding in findings:
        evidence = finding.get("evidence", {})
        finding_sections.append(
            f"""
            <article class="finding-card">
              <p class="eyebrow">{_safe(finding.get("agent", "Analysis"))}</p>
              <h3>{_safe(finding.get("title", "Finding"))}</h3>
              <p>{_safe(finding.get("observation", ""))}</p>
              <details>
                <summary>View supporting evidence</summary>
                <pre>{_safe(evidence)}</pre>
              </details>
              <p class="muted">Limitations: {_safe(finding.get("limitations", []))}</p>
            </article>
            """
        )

    recommendation_sections = []
    for index, recommendation in enumerate(recommendations, start=1):
        recommendation_sections.append(
            f"""
            <article class="recommendation-card">
              <p class="eyebrow">RECOMMENDATION {index}</p>
              <h3>{_safe(recommendation.get("title", "Suggested approach"))}</h3>
              <p><strong>Approach</strong><br>{_safe(recommendation.get("approach", ""))}</p>
              <p><strong>Why it may fit</strong><br>{_safe(recommendation.get("rationale", ""))}</p>
              <p><strong>Supporting findings</strong><br>{_safe(recommendation.get("supporting_findings", []))}</p>
              <p><strong>Validation strategy</strong><br>{_safe(recommendation.get("validation_strategy", ""))}</p>
              <p><strong>Suggested metrics</strong><br>{_safe(recommendation.get("suggested_metrics", []))}</p>
              <p><strong>Risks and assumptions to check</strong><br>{_safe(recommendation.get("risks", []))}</p>
              <p><strong>Next steps</strong><br>{_safe(recommendation.get("next_steps", []))}</p>
            </article>
            """
        )

    agent_rows = []
    for agent in agents:
        agent_rows.append(
            f"<tr><td>{_safe(agent.get('agent'))}</td>"
            f"<td>{_safe(agent.get('status'))}</td>"
            f"<td>{_safe(agent.get('summary') or agent.get('error', ''))}</td></tr>"
        )

    html_document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{plotly_script}
<title>Data Science Advisory Report</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172033;
  --muted: #657087;
  --line: #e1e6ef;
  --panel: #ffffff;
  --canvas: #f5f7fb;
  --accent: #3559c7;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--canvas); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.6;
}}
header {{ background: #101a31; color: white; padding: 42px max(22px, calc((100% - 1100px)/2)); }}
header h1 {{ margin: 8px 0; font-size: clamp(28px, 4vw, 42px); line-height: 1.15; }}
header p {{ color: #d6deef; margin: 8px 0; }}
main {{ max-width: 1100px; margin: 28px auto; padding: 0 20px 50px; }}
section {{ margin: 28px 0; }}
h2 {{ font-size: 24px; margin-bottom: 14px; }}
h3 {{ margin-top: 0; line-height: 1.35; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
.card, .chart-card, .finding-card, .recommendation-card {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 20px; min-width: 0; box-shadow: 0 3px 12px rgba(20,35,70,.035);
}}
.metric {{ font-size: 28px; font-weight: 750; letter-spacing: -.5px; }}
.muted, .eyebrow {{ color: var(--muted); }}
.eyebrow {{ font-size: 11px; font-weight: 750; letter-spacing: 1.2px; text-transform: uppercase; }}
.chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
.chart-card {{ overflow: hidden; }}
.chart-card h3 {{ font-size: 16px; }}
.recommendation-card {{ margin: 14px 0; border-left: 4px solid var(--accent); }}
.finding-card {{ margin: 14px 0; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f2f4f8; padding: 12px; border-radius: 8px; font-size: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ background: #edf1f8; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }}
.notice {{ border: 1px solid #d8e2fb; background: #eef3ff; padding: 15px 18px; border-radius: 12px; }}
footer {{ color: var(--muted); font-size: 12px; padding: 20px 0; }}
@media (max-width: 480px) {{ .chart-grid {{ grid-template-columns: 1fr; }} main {{ padding: 0 12px 30px; }} }}
</style>
</head>
<body>
<header>
  <p>AGENTIC DATA SCIENCE ADVISOR</p>
  <h1>Dataset analysis & ML strategy</h1>
  <p>{_safe(dataset.get("objective", "Exploratory analysis and advisory report"))}</p>
  <p>Generated {_safe(pd.Timestamp.now().strftime("%d %b %Y, %H:%M"))}</p>
</header>
<main>
  <section>
    <h2>Dataset overview</h2>
    <div class="grid">
      <div class="card"><div class="muted">Modality</div><div class="metric">{_safe(dataset.get("modality", "unknown")).title()}</div></div>
      <div class="card"><div class="muted">Rows / items</div><div class="metric">{_safe(dataset.get("row_count") if dataset.get("row_count") is not None else (len(df) if df is not None else 0))}</div></div>
      <div class="card"><div class="muted">Columns / features</div><div class="metric">{_safe(dataset.get("column_count") if dataset.get("column_count") is not None else (len(df.columns) if df is not None else "N/A"))}</div></div>
      
    </div>
    <p class="muted">Columns: {_safe(dataset.get("columns") if dataset.get("columns") is not None else (list(map(str, df.columns)) if df is not None else []))}</p>
  </section>

  <section>
    <h2>Exploratory data analysis</h2>
    <p class="muted">Charts below are computed directly from the supplied dataset. They describe the observed sample and do not establish causation.</p>
    <div class="chart-grid">
      {''.join(chart_sections) if chart_sections else '<div class="card">No suitable charts could be generated for this dataset.</div>'}
    </div>
  </section>

  <section>
    <h2>Key findings</h2>
    {''.join(finding_sections) if finding_sections else '<div class="card">No findings were returned.</div>'}
  </section>

  <section>
    <h2>ML strategy recommendations</h2>
    <div class="notice">
      <strong>Advisory report.</strong> Recommendations provide methodological guidance and do not represent measured model performance.
      <p><strong>Task interpretation:</strong> {_safe(recommendations_bundle.get("task_interpretation", "Not specified"))}</p>
      <p><strong>Assumptions:</strong> {_safe(recommendations_bundle.get("assumptions", []))}</p>
      <p><strong>Clarification questions:</strong> {_safe(recommendations_bundle.get("clarification_questions", []))}</p>
    </div>
    {''.join(recommendation_sections) if recommendation_sections else '<div class="card">No recommendations were returned.</div>'}
  </section>

  <section>
    <h2>Agent execution</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Agent</th><th>Status</th><th>Details</th></tr></thead>
        <tbody>{''.join(agent_rows)}</tbody>
      </table>
    </div>
  </section>

  <footer>
    Generated by Agentic Multimodal Data Science Advisor. Recommendations should be reviewed alongside their evidence, assumptions and limitations.
    Review the evidence and assumptions before using them for consequential decisions.
  </footer>
</main>
</body>
</html>
"""
    destination.write_text(html_document, encoding="utf-8")
    return str(destination)
