# Agentic Data Science Advisor

An LLM-driven platform that automatically profiles datasets, performs exploratory data analysis (EDA), and provides ML strategy recommendations. Supports tabular, image, and audio data.

## What It Does

Upload a dataset and get:
- **Automated profiling** — structure, types, missing values, duplicates, quality checks
- **Modality-specific EDA** — distributions, correlations, image statistics, audio metadata
- **Evidence-grounded ML recommendations** — approach, rationale, validation strategy, risks, and next steps
- **Interactive HTML reports** — Plotly visualizations, findings with provenance, downloadable JSON

All without writing profiling code or boilerplate.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI / FastAPI Entry Point                  │
│  • Loads data • Detects modality • Calls workflow              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph Multi-Agent Workflow                │
│  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌─────────────┐ │
│  │Supervisor│─▶│Dataset Analyst│─▶│EDA Agent│─▶│Evidence Gate│ │
│  │          │  │• Profiling   │  │• EDA    │  │             │ │
│  └──────────┘  │• Quality     │  │         │  └──────┬──────┘ │
│                └──────────────┘  └─────────┘         │       │
│                        ▲                     ┌────────▼─────┐│
│                        │                     │ML Strategy   ││
│                        └────────────────────▶│Agent (LLM)   ││
│                                              └──────┬───────┘│
│                                                     │       │
│                                             ┌───────▼───────┐│
│                                             │Report Agent   ││
│                                             └───────────────┘│
└─────────────────────────────────────────────────────────────────┘
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
            │      Reporting (HTML + JSON)     │
            └──────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Type | Responsibility |
|-------|------|----------------|
| **Supervisor** | Deterministic | Validates input, detects modality, routes workflow |
| **Dataset Analyst** | Deterministic | Profiles data, computes quality metrics, creates findings |
| **EDA Agent** | Deterministic | Runs modality-specific exploratory analysis |
| **Evidence Gate** | Deterministic | Validates evidence sufficiency before strategy step |
| **ML Strategy Agent** | LLM-dependent | Generates evidence-grounded ML recommendations |
| **Report Agent** | Deterministic | Assembles HTML + JSON report with provenance tags |

**Failure Handling:** If any deterministic agent fails, the Evidence Gate blocks the Strategy Agent and produces a `blocked` or `partial` report with error details.

## Features

- **Multi-modal profiling** — Tabular (CSV/Excel/Parquet), Image (PNG/JPEG/IDX/GZIP), Audio (WAV/FLAC/OGG/AIFF)
- **Data quality analysis** — Missing values, duplicates, constant columns, invalid files
- **Modality-specific EDA** — Distributions, correlations, image statistics, audio metadata
- **Evidence-grounded ML recommendations** — Approach, rationale, validation strategy, risks, next steps
- **Evidence provenance** — Findings tagged as measured, metadata, methodology, or assumption
- **Interactive HTML reports** — Plotly charts, collapsible evidence, downloadable JSON
- **Multi-agent workflow** — LangGraph orchestration with evidence gates
- **FastAPI backend + web UI** — Drag-and-drop upload, modality selection, results display
- **Docker support** — Containerized API and CLI

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Core | Python 3.12+, Pandas, NumPy |
| ML/Analysis | Scikit-learn, SciPy |
| Visualization | Plotly, Matplotlib/Seaborn |
| Agent Orchestration | LangGraph, LangChain |
| Image | Pillow |
| Audio | soundfile (libsndfile) |
| API | FastAPI, Uvicorn |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |

## Quick Start

### Prerequisites
- Python 3.12+
- Docker (optional)
- FreeLLMAPI-compatible API key for ML strategy recommendations

### Local Development

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Set up API key (required for ML strategy recommendations)
echo "FREELLMAPI_API_KEY=your-key-here" > .env
echo "FREELLMAPI_BASE_URL=http://localhost:3001/v1" >> .env
echo "FREELLMAPI_MODEL=auto" >> .env

# Run tests
python -m pytest tests/ -v
```

### Using Docker

```bash
# Build and run tests
docker compose up --build app

# Start the web API
docker compose up --build api
# Access at http://localhost:8000

# Run advisor CLI on a dataset
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

### Web Interface

Start the API server and open `http://localhost:8000`:
```bash
docker compose up --build api
# or locally: python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
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
- `advisor_report.html` — Interactive HTML report with charts, findings, recommendations

## Supported Modalities

| Modality | Supported Inputs | Analysis Capabilities |
|----------|------------------|----------------------|
| **Tabular** | CSV, Excel, Parquet | Profiling, quality checks, distributions, correlations, target-aware EDA |
| **Image** | PNG, JPEG, BMP, TIFF, WebP, IDX/GZIP | Statistics, dimensions, splits, class distributions, visualizations |
| **Audio** | WAV, FLAC, OGG, AIFF | Metadata, duration, configuration, amplitude statistics, EDA |

## Demo Datasets (not included)

- **Fashion-MNIST** — 70,000 images in IDX/GZIP format
- **Bank Marketing** — 45,211 rows CSV
- **ESC-50** — 2,000 WAV audio files

> **Note:** Datasets are not included. Download separately and place in `data/`.

## Scope & Future Improvements

- **No model training** — The platform profiles data, performs EDA, and recommends ML approaches. It does not train or evaluate models.
- **IDX support** — Standard MNIST/Fashion-MNIST IDX formats; custom variants may not work.
- **Audio metadata** — Class labels and folds require a compatible metadata CSV.
- **LLM context limits** — Evidence capped at 50 items per collection. Full details retained in JSON/HTML reports; strategy agent receives summarized version.
- **Large image directories** — 300+ individual image files may exceed LLM context even after truncation. Compressed IDX format (Fashion-MNIST) works around this.
- **No model training/evaluation** — Platform provides analysis and recommendations only.
- **Planned integrations** — FastAPI, PostgreSQL, RAG are not currently implemented beyond basic API endpoints.

## Project Structure

```
agentic-ds-ml/
├── app/
│   ├── agents/           # LangGraph workflow & CLI
│   │   ├── run_advisor.py
│   │   ├── advisor_workflow.py
│   │   ├── llm_client.py
│   │   └── check_llm.py
│   ├── api/              # FastAPI backend
│   │   └── main.py
│   ├── tools/            # Analysis & reporting
│   │   ├── multimodal.py         # Image/audio profiling + EDA
│   │   ├── dataset_metadata.py   # IDX/GZIP reader (MNIST/Fashion-MNIST)
│   │   ├── profiling.py          # Tabular profiling
│   │   ├── data_quality.py       # Data quality checks
│   │   ├── eda.py                # Tabular EDA
│   │   └── reporting.py          # HTML report generation
│   ├── schemas/          # Pydantic models (with provenance)
│   └── static/           # Web frontend (HTML/CSS/JS)
├── data/                 # Sample datasets (not committed)
├── reports/              # Generated HTML/JSON reports
├── tests/                # 45 pytest tests
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

All 45 tests pass (30 original + 15 API tests covering truncation, provenance, and finding references).

### Adding New Modalities

1. Add profiling/EDA functions in `app/tools/multimodal.py`
2. Update modality detection in `app/agents/run_advisor.py`
3. Add routing in `app/agents/advisor_workflow.py` supervisor
4. Add chart generation in `app/tools/reporting.py`

## Security

- Never commit API keys, passwords, environment files, or private datasets. The `.gitignore` excludes `.env`, `data/`, and `reports/`.
- The API does not expose API keys in responses or logs.
- File uploads limited to 100 MB with extension validation.
- ZIP extraction includes path traversal protection.