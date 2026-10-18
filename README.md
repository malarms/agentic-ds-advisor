# Agentic Data Science Advisor

An LLM-driven platform that automatically profiles datasets, performs exploratory data analysis (EDA), identifies useful findings, and provides ML strategy recommendations. Generates interactive HTML reports.

**Supported Modalities:** Tabular (CSV, Excel, Parquet), Image (PNG, JPEG, BMP, TIFF, WebP, IDX/GZIP), Audio (WAV, FLAC, OGG, AIFF)

## Current Features

- **Multi-modal dataset profiling** — Automatic detection and profiling for tabular, image, and audio data
- **Data-quality analysis** — Missing values, duplicates, constant columns, invalid files
- **Modality-specific EDA** — Distributions, charts, and statistics tailored to each modality
- **ML Strategy recommendations** — Evidence-based approach suggestions with validation strategies
- **Interactive HTML reports** — Plotly-powered visualizations, findings, and recommendations
- **Multi-agent workflow** — LangGraph-based orchestration with evidence gates
- **Evidence provenance** — Findings tagged as measured, metadata, methodology, or assumption
- **Automated tests** — 30 tests covering all modalities, truncation, provenance, and error handling

## Technology Stack

- **Core:** Python 3.12+, Pandas, NumPy
- **ML/Analysis:** Scikit-learn, SciPy
- **Visualization:** Plotly (interactive HTML), Matplotlib/Seaborn
- **Agent Orchestration:** LangGraph, LangChain
- **Image:** Pillow
- **Audio:** soundfile (libsndfile)
- **Testing:** Pytest

## Quick Start

### Local Development

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Set up API key (required for ML strategy recommendations)
echo "FREELLMAPI_API_KEY=your-key-here" > .env
echo "FREELLMAPI_BASE_URL=http://localhost:3001/v1" >> .env  # optional, defaults to this
echo "FREELLMAPI_MODEL=auto" >> .env  # optional, defaults to this

# Run tests
python -m pytest tests/ -v
```

### Using Docker

```bash
# Build and run tests
docker compose up --build app

# Run advisor on a dataset
docker compose run --rm advisor \
  --file /app/data/fashion-mnist-master/data/fashion \
  --modality image \
  --request "Recommend ML approaches for image classification" \
  --reports-dir /app/reports/fashion-mnist
```

## Usage

### Command Line Interface

```bash
# Tabular dataset (CSV, Excel, Parquet)
python -m app.agents.run_advisor \
  --file data/bank+marketing/bank/bank-full.csv \
  --modality tabular \
  --request "Predict term deposit subscription" \
  --reports-dir reports/bank

# Image dataset (standard files or IDX/GZIP like MNIST/Fashion-MNIST)
python -m app.agents.run_advisor \
  --file data/fashion-mnist-master/data/fashion \
  --modality image \
  --request "Recommend ML approaches for image classification" \
  --reports-dir reports/fashion

# Audio dataset
python -m app.agents.run_advisor \
  --file data/ESC-50-master \
  --modality audio \
  --request "Recommend approaches for environmental sound classification" \
  --reports-dir reports/esc50

# Profile-only (skip EDA and strategy)
python -m app.agents.run_advisor \
  --file data/bank+marketing/bank/bank-full.csv \
  --modality tabular \
  --request "Only profile this dataset" \
  --reports-dir reports/bank-profile
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--file` | Yes | Path to dataset file or directory |
| `--modality` | No | `tabular`, `image`, or `audio` (auto-detected if omitted) |
| `--request` | Yes | Natural language description of what you want to learn |
| `--reports-dir` | No | Output directory (default: `reports`) |

### Output

Each run generates:
- `advisor_results.json` — Structured results for programmatic use
- `advisor_report.html` — Interactive HTML report with charts, findings, and recommendations

Open the HTML report in a browser to explore:
- Dataset overview with key metrics
- Exploratory data analysis charts
- Key findings with supporting evidence
- ML strategy recommendations with rationale and next steps
- Agent execution log

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    run_advisor.py (CLI)                     │
│  • Loads data • Detects modality • Calls workflow          │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  advisor_workflow.py (LangGraph)            │
│  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌───────────┐ │
│  │Supervisor│─▶│Dataset Analyst│─▶│EDA Agent│─▶│Evidence   │ │
│  │          │  │• Profiling   │  │• EDA    │  │Gate       │ │
│  └──────────┘  │• Quality     │  │         │  └─────┬─────┘ │
│                └──────────────┘  └─────────┘        │       │
│                        ▲                     ┌───────▼─────┐│
│                        │                     │ML Strategy  ││
│                        └────────────────────▶│Agent (LLM)  ││
│                                              └───────┬─────┘│
│                                                      │       │
│                                              ┌──────▼──────┐│
│                                              │Report Agent ││
│                                              └─────────────┘│
└─────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │  Image   │ │  Tabular │ │  Audio   │
            │ Profiling│ │ Profiling│ │ Profiling│
            │ + EDA    │ │ + EDA    │ │ + EDA    │
            └──────────┘ └──────────┘ └──────────┘
                    │            │            │
                    ▼            ▼            ▼
            ┌──────────────────────────────────┐
            │      reporting.py (HTML)         │
            └──────────────────────────────────┘
```

### Key Components

| Module | Purpose |
|--------|---------|
| `app/agents/run_advisor.py` | CLI entry point, data loading, modality detection |
| `app/agents/advisor_workflow.py` | LangGraph multi-agent workflow |
| `app/tools/multimodal.py` | Image & audio profiling + EDA |
| `app/tools/dataset_metadata.py` | IDX/GZIP reader (MNIST/Fashion-MNIST) |
| `app/tools/profiling.py` | Tabular profiling |
| `app/tools/data_quality.py` | Data quality checks |
| `app/tools/eda.py` | Tabular EDA |
| `app/tools/reporting.py` | HTML report generation |

## Supported Modalities

| Modality | Supported Inputs | Analysis Capabilities |
|---|---|---|
| **Tabular** | CSV, Excel, Parquet | Dataset profiling, data-quality checks, distributions, correlations, and target-aware EDA |
| **Image** | PNG, JPEG, BMP, TIFF, WebP, supported IDX/GZIP formats | Image statistics, dimensions, dataset splits, class distributions, and visualizations where supported |
| **Audio** | WAV, FLAC, OGG, AIFF | Audio metadata, duration, configuration, amplitude statistics, and EDA where supported |

## Demo Datasets

The following datasets have been tested end-to-end:

- **Fashion-MNIST:** 70,000 images in IDX/GZIP format
- **Bank Marketing:** 45,211 rows in CSV format
- **ESC-50:** 2,000 WAV audio files

## Limitations and Scope

- **No model training:** The platform profiles datasets, performs EDA, and recommends ML approaches. It does not train or evaluate models.
- **IDX compatibility:** IDX support targets standard MNIST/Fashion-MNIST IDX formats; custom variants may not be supported.
- **Audio metadata:** Class labels and dataset folds require a compatible metadata CSV when needed for the analysis.
- **LLM context limits:** Evidence supplied to the strategy agent is capped at 50 items to manage context size. The generated JSON and HTML reports retain full details; the strategy agent receives a summarized version with truncation noted.
- **Planned integrations:** FastAPI, PostgreSQL, and RAG are not currently implemented.

## Project Structure

```
agentic-ds-ml/
├── app/
│   ├── agents/           # LangGraph workflow & CLI
│   │   ├── run_advisor.py
│   │   ├── advisor_workflow.py
│   │   ├── llm_client.py
│   │   └── check_llm.py
│   ├── tools/            # Analysis & reporting
│   │   ├── multimodal.py         # Image/audio profiling + EDA
│   │   ├── dataset_metadata.py   # IDX/GZIP reader (MNIST/Fashion-MNIST)
│   │   ├── profiling.py          # Tabular profiling
│   │   ├── data_quality.py       # Data quality checks
│   │   ├── eda.py                # Tabular EDA
│   │   └── reporting.py          # HTML report generation
│   ├── schemas/          # Pydantic models (with provenance)
│   └── api/              # (planned) FastAPI endpoints
├── data/                 # Sample datasets (not committed)
├── reports/              # Generated HTML/JSON reports
├── tests/                # 30 pytest tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

All 30 tests pass (25 original + 5 new regression tests for evidence truncation, provenance, and finding references).

### Adding New Modalities

1. Add profiling/EDA functions in `app/tools/multimodal.py`
2. Update modality detection in `app/agents/run_advisor.py`
3. Add routing in `app/agents/advisor_workflow.py` supervisor
4. Add chart generation in `app/tools/reporting.py`

## License

MIT License — see LICENSE file for details.

## Security

Never commit API keys, passwords, environment files, or private datasets. The `.gitignore` excludes `.env`, `data/`, and `reports/`.