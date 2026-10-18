import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio


def _safe(value: Any) -> str:
    """Escape dynamic text before inserting it into HTML."""
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return html.escape(str(value))


def _chart_html(fig) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )


def _build_charts(df: pd.DataFrame) -> list[tuple[str, str]]:
    charts: list[tuple[str, str]] = []

    # Missingness across columns
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    if len(missing) and (missing > 0).any():
        missing_df = missing[missing > 0].rename("Missing (%)").reset_index()
        missing_df.columns = ["Column", "Missing (%)"]
        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing (%)",
            title="Missing values by column",
            text_auto=".1f",
        )
        charts.append(("Missingness", _chart_html(fig)))

    # Numeric distributions
    numeric_columns = list(df.select_dtypes(include="number").columns[:6])
    for column in numeric_columns:
        values = df[column].dropna()
        if values.empty:
            continue
        fig = px.histogram(
            pd.DataFrame({str(column): values}),
            x=str(column),
            title=f"Distribution: {column}",
            marginal="box",
            opacity=0.85,
        )
        charts.append((f"Distribution — {column}", _chart_html(fig)))

    # Categorical distributions
    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
    ][:4]

    for column in categorical_columns:
        counts = (
            df[column]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
            .head(12)
            .rename_axis("Category")
            .reset_index(name="Count")
        )
        if counts.empty:
            continue
        fig = px.bar(
            counts.sort_values("Count"),
            x="Count",
            y="Category",
            orientation="h",
            title=f"Category counts: {column}",
            text="Count",
        )
        charts.append((f"Categories — {column}", _chart_html(fig)))

    # Correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric feature correlations",
        )
        charts.append(("Correlations", _chart_html(fig)))

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
    for index, (title, chart) in enumerate(_build_charts(df), start=1):
        chart_sections.append(
            f"""
            <section class="chart-card">
              <h3>{_safe(title)}</h3>
              {chart}
            </section>
            """
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
      <div class="card"><div class="muted">Rows / items</div><div class="metric">{_safe(dataset.get("row_count", len(df)))}</div></div>
      <div class="card"><div class="muted">Columns / features</div><div class="metric">{_safe(dataset.get("column_count", len(df.columns)))}</div></div>
      <div class="card"><div class="muted">Models trained</div><div class="metric">0</div></div>
    </div>
    <p class="muted">Columns: {_safe(dataset.get("columns", list(map(str, df.columns))))}</p>
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
      <strong>Advisory only.</strong> This report recommends approaches and evaluation strategies. It does not train models or report model performance.
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
    Generated by Agentic Multimodal Data Science Advisor. Recommendations are suggestions, not validated model results.
    Review the evidence and assumptions before using them for consequential decisions.
  </footer>
</main>
</body>
</html>
"""
    destination.write_text(html_document, encoding="utf-8")
    return str(destination)
