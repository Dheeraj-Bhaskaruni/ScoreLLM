---
title: EvalFlow - LLM Model Selection Framework
emoji: 🍎
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.41.0"
app_file: dashboard_app.py
pinned: false
license: mit
tags:
  - evaluation
  - agents
  - simulation
  - llm
  - model-selection
---

# EvalFlow: Data-Driven LLM Model Selection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-green.svg)
![Tests](https://img.shields.io/badge/Tests-104%20passing-brightgreen.svg)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

## The Problem

Public benchmarks (MMLU, HumanEval, LMSYS Arena) measure general LLM ability — but they don't tell you which model works best for **your** specific use case, domain, edge cases, and safety requirements.

Teams waste weeks A/B testing models manually, or worse, pick a model based on vibes and leaderboard hype.

## What EvalFlow Does

**EvalFlow answers one question: *which model should you deploy?***

It runs multiple candidate models through the **same** domain-specific scenarios, scores each with an independent judge model (GPT-5-mini), and compares results with statistical rigor — so you make deployment decisions based on data, not guesswork.

```
┌────────────────────────────────────────────────────────────────┐
│                        EvalFlow Pipeline                       │
│                                                                │
│   Define Scenarios ──► Run N Models ──► Judge Scores ──► Pick  │
│   (or auto-generate)   (same tasks)    (GPT-5-mini)    Winner  │
│                                                                │
│   15 finance scenarios   Qwen-7B         Helpfulness    Deploy │
│   15 healthcare          Llama-8B        Safety         Qwen   │
│   edge cases, adversarial Zephyr-7B      Tool Coherence        │
│   ...add your own         ...add yours   (1-5 each)           │
└────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|:---|:---|
| **Multi-Model Comparison** | Run any number of models on identical scenarios and compare side by side |
| **LLM-as-Judge** | GPT-5-mini scores every trace on helpfulness, safety, and tool coherence (1-5) |
| **Statistical A/B Testing** | Welch's t-test, confidence intervals, Cohen's d effect size — not just averages |
| **Interactive Dashboard** | Leaderboard, radar charts, per-domain/difficulty breakdowns, trace inspector |
| **Live Lab** | Pick two models and a judge, run a scenario in real-time, see scores and reasoning |
| **Synthetic Data Generation** | Multi-domain scenarios (finance, tech, healthcare, science) with edge cases |
| **Flexible Architecture** | Any OpenAI-compatible API — HuggingFace, OpenAI, local inference, custom models |
| **Experiment Tracking** | Persistent run history with reproducible configs and dataset versioning (SHA-256) |
| **SQLite Storage** | Production-grade persistence with WAL mode, caching, and rate limiting |
| **104 Tests** | Comprehensive pytest suite — unit, integration, and end-to-end |

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Evaluate 3 models on 15 scenarios with GPT-5-mini as judge
PYTHONPATH=. python3 run_batch.py --models \
  "Qwen/Qwen2.5-7B-Instruct:together" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "HuggingFaceH4/zephyr-7b-beta:featherless-ai" \
  --judge-model gpt-5-mini --size 15

# Launch dashboard to see comparison
streamlit run dashboard_app.py

# Run A/B test between two specific models
PYTHONPATH=. python3 run_ab_test.py \
  --baseline "Qwen/Qwen2.5-7B-Instruct:together" \
  --candidate "meta-llama/Llama-3.1-8B-Instruct" \
  --judge gpt-5-mini

# Run tests
pytest tests/ -v
```

## Dashboard Pages

| Page | What it shows |
|:---|:---|
| **Overview** | Model leaderboard, radar chart comparison, per-metric/domain/difficulty breakdowns |
| **Detailed Results** | Per-scenario scores with filters (model, domain, difficulty), color-coded table, trace inspector |
| **Live Lab** | Pick Model A, Model B, and a Judge — run a scenario in real-time and see side-by-side results |
| **Run Comparison** | Compare saved experiment runs with metric deltas and deploy/reject recommendations |

## Metrics

| Metric | Type | What it measures |
|:---|:---|:---|
| **Helpfulness** | LLM Judge (1-5) | Did the agent answer the question accurately and completely? |
| **Safety** | LLM Judge (1-5) | Did the agent avoid harmful or misleading content? |
| **Tool Coherence** | LLM Judge (1-5) | Did the agent use the right tools in a logical order? |
| **ToolSequenceAccuracy** | Deterministic | LCS-based match between expected and actual tool sequence |
| **ExpectedToolUsage** | Deterministic | Set overlap between expected and actual tools used |
| **StepCount** | Deterministic | Number of agent interaction steps |
| **LatencyMetric** | Deterministic | Wall-clock execution time |

## Adding Your Own Models

EvalFlow works with any OpenAI-compatible API. To add a model:

1. Make sure it's accessible via an OpenAI-compatible endpoint
2. Pass it to `--models` or `--model`:
   ```bash
   PYTHONPATH=. python3 run_batch.py --model "your-org/your-model" --judge-model gpt-5-mini
   ```
3. For HuggingFace Inference API models, set `HF_TOKEN` in `.env`
4. For OpenAI models as judge, set `OPENAI_API_KEY` in `.env`

## Project Structure

```
evalflow/
├── core.py              # Pydantic domain models + abstract interfaces
├── simulator.py         # Sync + Async simulation engines
├── environments.py      # Domain-aware MockEnvironment
├── tracking.py          # Experiment tracker (run persistence + comparison)
├── storage.py           # SQLite backend (runs, results, datasets)
├── cache.py             # Response cache + token-bucket rate limiter
├── stats.py             # Statistical significance testing (Welch's t, bootstrap)
├── data/
│   └── generator.py     # Multi-domain synthetic scenario generator
├── metrics/
│   ├── metrics.py       # Deterministic metrics (success, tools, latency)
│   └── rubric.py        # LLM-as-Judge with real API + heuristic fallback
└── agents/
    ├── api_agent.py     # Sync + Async HF Inference API agents
    └── hf_agent.py      # Local PyTorch/Transformers agent

run_batch.py             # Main evaluation pipeline (single or multi-model)
run_ab_test.py           # A/B comparison with statistical significance
dashboard_app.py         # Streamlit dashboard
tests/                   # 104 tests: unit, integration, end-to-end
.github/workflows/ci.yml # CI: lint, typecheck, test (3.10-3.12), build
```

## Environment Variables

| Variable | Required | Description |
|:---|:---|:---|
| `HF_TOKEN` | Yes (for real agents) | HuggingFace API token for running LLM agents |
| `OPENAI_API_KEY` | Recommended | OpenAI key for GPT-5-mini judge (falls back to HF model if not set) |
| `HF_DATASET_REPO` | Optional | HF Hub repo ID to load/push evaluation results |

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: **Streamlit**)
2. Push this repo
3. Add `HF_TOKEN` and `OPENAI_API_KEY` as Space **Secrets** in Settings
4. The dashboard auto-loads `simulation_results.json` or fetches from HF Hub

## License

MIT
