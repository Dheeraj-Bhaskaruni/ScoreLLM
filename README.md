---
title: EvalFlow - AI Agent Evaluation System
emoji: 🍎
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: false
license: mit
tags:
  - evaluation
  - agents
  - simulation
  - llm
  - pytorch
---

# EvalFlow: Agentic AI Evaluation & Simulation Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-green.svg)
![Tests](https://img.shields.io/badge/Tests-101%20passing-brightgreen.svg)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

**EvalFlow** is a production-grade framework for evaluating LLM-based agents at scale. It generates synthetic edge-case scenarios, runs agents through a deterministic simulation harness, scores them with both quantitative metrics and LLM-as-a-Judge, and tracks experiments across model versions.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     EvalFlow Pipeline                        │
│                                                              │
│  DatasetGenerator ──► SimulationEngine ──► MetricEngine      │
│   (synthetic data)    (agent-env loop)    (det + rubric)     │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│    Scenario[]          Trace[]          EvaluationResult[]   │
│                                                │             │
│                    ExperimentTracker ◄──────────┘             │
│                    (runs/, compare)                           │
│                          │                                   │
│                   ┌──────┴──────┐                            │
│                   ▼             ▼                             │
│              Dashboard     HF Hub Dataset                    │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|:---|:---|
| **Pydantic Models** | Type-safe domain models with validation and JSON serialization |
| **Async Engine** | `AsyncSimulationEngine` with semaphore-controlled concurrency |
| **LLM-as-Judge** | Real rubric scoring via HF Inference API (helpfulness, safety, tool coherence) with heuristic fallback |
| **Experiment Tracking** | Persisted runs with unique IDs, aggregate metrics, and side-by-side comparison |
| **Synthetic Data** | Multi-domain generator (finance, tech, healthcare, science) with edge cases (adversarial, i18n, prompt injection) |
| **Domain-Aware Env** | MockEnvironment with realistic responses per domain, latency simulation, stochastic failures |
| **70 Tests** | Comprehensive pytest suite — unit, integration, and end-to-end |
| **CI/CD** | GitHub Actions: lint (ruff), type-check (mypy), test (3.10-3.12), build |
| **HF Integration** | Push results as HF Dataset, deploy dashboard as HF Space |
| **PyTorch Support** | Optional local inference with HuggingFace Transformers |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run evaluation pipeline (50 scenarios, heuristic judge)
PYTHONPATH=. python3 run_batch.py

# Run with real LLM-as-Judge scoring
PYTHONPATH=. python3 run_batch.py --use-llm-judge

# Run with stress testing (latency + failures)
PYTHONPATH=. python3 run_batch.py --size 100 --latency-ms 50 --env-failure-rate 0.1

# Launch dashboard
streamlit run dashboard_app.py

# Run tests
pytest tests/ -v
```

## Pipeline Options

```
usage: run_batch.py [-h] [--size N] [--seed S] [--max-steps M]
                    [--output PATH] [--runs-dir DIR]
                    [--use-hf-agent] [--use-llm-judge]
                    [--judge-model MODEL_ID]
                    [--latency-ms MS] [--env-failure-rate R]
```

## Metrics Taxonomy

| Metric | Type | Description |
|:---|:---|:---|
| **SuccessRate** | Deterministic | Binary: agent finished without errors |
| **StepCount** | Deterministic | Number of interaction steps |
| **ExpectedToolUsage** | Deterministic | Set overlap between expected and actual tools |
| **ToolSequenceAccuracy** | Deterministic | LCS-based ordered sequence match |
| **LatencyMetric** | Deterministic | Wall-clock duration |
| **Helpfulness** | LLM Judge | 1-5 rubric: did the agent answer helpfully? |
| **Safety** | LLM Judge | 1-5 rubric: was the agent safe? |
| **Tool Coherence** | LLM Judge | 1-5 rubric: were tool calls logical? |

## Experiment Tracking

Each run is persisted under `runs/` with full config and aggregate metrics:

```bash
# List past runs
python3 -c "from evalflow.tracking import ExperimentTracker; print(ExperimentTracker().list_runs())"

# Compare two runs
python3 -c "
from evalflow.tracking import ExperimentTracker
t = ExperimentTracker()
print(t.compare_runs('run_id_a', 'run_id_b'))
"
```

## Hugging Face Integration

#### Push results as a HF Dataset
```bash
python3 upload_dataset.py --repo-id your-username/evalflow-results
python3 upload_dataset.py --dry-run  # local preview
```

#### Deploy dashboard to HF Spaces
1. Create a new Space (SDK: **Streamlit**)
2. Push this repo
3. Optionally set `HF_DATASET_REPO` secret to load from Hub

## Project Structure

```
evalflow/
├── core.py              # Pydantic domain models + abstract interfaces
├── simulator.py         # Sync + Async simulation engines
├── environments.py      # Domain-aware MockEnvironment
├── tracking.py          # Experiment tracker (run persistence + comparison)
├── data/
│   └── generator.py     # Multi-domain synthetic scenario generator
├── metrics/
│   ├── metrics.py       # Deterministic metrics (success, tools, latency)
│   └── rubric.py        # LLM-as-Judge with real API + heuristic fallback
└── agents/
    ├── api_agent.py     # Sync + Async HF Inference API agents
    └── hf_agent.py      # Local PyTorch/Transformers agent

tests/                   # 70 tests: unit, integration, end-to-end
.github/workflows/ci.yml # CI: lint, typecheck, test (3.10-3.12), build
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run `pytest tests/ -v` before submitting
4. Open a PR with a clear description

## License

MIT
