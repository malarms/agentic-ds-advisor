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
- **Automated tests** — 25 tests covering all modalities and workflows

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
echo "OPENAI_API_KEY=your-key-here" > .env

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

## Supported Datasets (Validated)

| Modality | Dataset | Size | Status |
|----------|---------|------|--------|
| **Image** | Fashion-MNIST (IDX/GZIP) | 70,000 images | ✅ Full workflow |
| **Image** | Standard PNG/JPEG folders | Variable | ✅ Full workflow |
| **Tabular** | Bank Marketing (CSV) | 45,211 rows | ✅ Full workflow |
| **Audio** | ESC-50 (WAV) | 2,000 files | ✅ Full workflow |

## Known Limitations

1. **LLM Context Limits** — Very large datasets (e.g., 2000+ audio files) have their evidence trimmed to 50 items for the strategy agent. Full details remain in JSON output and HTML report.

2. **No Model Training** — This platform provides analysis and recommendations only. It does not train or evaluate models.

3. **IDX Format** — Only standard MNIST/Fashion-MNIST IDX3/IDX1 format is supported (gzipped). Custom IDX variants may not work.

4. **Audio Metadata** — Audio analysis requires a corresponding CSV metadata file (e.g., `meta/esc50.csv`) for class labels and folds.

5. **API Key Required** — ML strategy recommendations require an OpenAI-compatible API key in `.env`.

6. **No FastAPI/PostgreSQL/RAG** — These planned features are not yet implemented (see `app/api/`, `app/database/`, `app/rag/` directories are empty).

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
│   │   ├── dataset_metadata.py   # IDX reader
│   │   ├── profiling.py          # Tabular profiling
│   │   ├── data_quality.py       # Quality checks
│   │   ├── eda.py                # Tabular EDA
│   │   └── reporting.py          # HTML generation
│   ├── schemas/          # Pydantic models
│   └── api/              # (planned) FastAPI endpoints
├── data/                 # Sample datasets (not committed)
├── reports/              # Generated HTML/JSON reports
├── tests/                # 25 pytest tests
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

### Adding New Modalities

1. Add profiling/EDA functions in `app/tools/multimodal.py`
2. Update modality detection in `app/agents/run_advisor.py`
3. Add routing in `app/agents/advisor_workflow.py` supervisor
4. Add chart generation in `app/tools/reporting.py`

## License

MIT License — see LICENSE file for details.

## Security

Never commit API keys, passwords, environment files, or private datasets. The `.gitignore` excludes `.env`, `data/`, and `reports/`.