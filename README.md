# Agentic Data Science & ML Platform

An LLM-driven platform designed to orchestrate data science workflows,
including dataset profiling, data-quality analysis, exploratory data
analysis, machine learning experiments, and report generation.

## Current Features

- Dataset profiling and missing-value analysis
- Duplicate-row and constant-column detection
- Exploratory data analysis and histogram generation
- Classification and regression baseline model comparison
- Automated tests for core components

## Technology Stack

- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Pytest

## Planned Features

- LangGraph-based agent orchestration
- LangChain tool calling and structured LLM outputs
- Retrieval-Augmented Generation (RAG)
- FastAPI and PostgreSQL
- Docker deployment

## Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

Run tests:

    python -m pytest -q

## Project Status

Under active development.

## Security

Never commit API keys, passwords, environment files, or private datasets.
