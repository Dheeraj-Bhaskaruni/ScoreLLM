---
title: ScoreLLM
emoji: 📊
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
  - llm
  - model-selection
  - finetuning
---

# ScoreLLM

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-green.svg)
![Tests](https://img.shields.io/badge/Tests-104%20passing-brightgreen.svg)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

## Why I Built This

I was finetuning language models for lab research and needed a systematic way to compare model variants — base vs. finetuned, different architectures, different sizes — on our domain-specific tasks. Public benchmarks (MMLU, HumanEval) didn't help because they measure general ability, not how a model handles *our* scenarios, edge cases, and safety requirements.

So I built **ScoreLLM**: a framework that runs multiple candidate models through identical domain-specific scenarios, scores each with an independent judge model (GPT-5-mini), and produces a data-driven comparison with statistical significance testing.

## What It Does

**Given N candidate models and a set of scenarios, ScoreLLM tells you which model to deploy.**

```
┌────────────────────────────────────────────────────────────────┐
│                     ScoreLLM Pipeline                   │
│                                                                │
│   Define Scenarios ──► Run N Models ──► Judge Scores ──► Pick  │
│   (or auto-generate)   (same tasks)    (GPT-5-mini)    Winner  │
│                                                                │
│   finance, healthcare    Qwen-7B         Helpfulness           │
│   edge cases, adversarial Llama-8B       Safety                │
│   your custom scenarios   Zephyr-7B      Tool Coherence        │
│   ...any domain           finetuned v2   (1-5 each)            │
└────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|:---|:---|
| **Multi-Model Comparison** | Run any number of models on identical scenarios — compare base vs. finetuned, or N candidates |
| **LLM-as-Judge** | GPT-5-mini scores every trace on helpfulness, safety, and tool coherence (1-5) |
| **Statistical A/B Testing** | Welch's t-test, confidence intervals, Cohen's d effect size — not just averages |
| **Interactive Dashboard** | Leaderboard, radar charts, per-domain/difficulty breakdowns, trace inspector |
| **Live Lab** | Pick two models and a judge, run a scenario in real-time, see scores and reasoning |
| **Synthetic Data Generation** | Multi-domain scenarios (finance, tech, healthcare, science) with edge cases |
| **Flexible Architecture** | Any OpenAI-compatible API — HuggingFace, OpenAI, local models, finetuned checkpoints |
| **Experiment Tracking** | Persistent run history with reproducible configs and dataset versioning (SHA-256) |
| **SQLite Storage** | Production-grade persistence with WAL mode, caching, and rate limiting |
| **104 Tests** | Comprehensive pytest suite — unit, integration, end-to-end |

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Compare 3 models on 15 scenarios with GPT-5-mini as judge
PYTHONPATH=. python3 run_batch.py --models \
  "Qwen/Qwen2.5-7B-Instruct:together" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "HuggingFaceH4/zephyr-7b-beta:featherless-ai" \
  --judge-model gpt-5-mini --size 15

# Compare a base model vs your finetuned version
PYTHONPATH=. python3 run_ab_test.py \
  --baseline "base-model-id" \
  --candidate "your-finetuned-model-id" \
  --judge gpt-5-mini

# Launch dashboard
streamlit run dashboard_app.py

# Run tests
pytest tests/ -v
```

## Dashboard

| Page | What it shows |
|:---|:---|
| **Overview** | Model leaderboard, radar chart comparison, per-metric/domain/difficulty breakdowns |
| **Detailed Results** | Per-scenario scores with filters (model, domain, difficulty), color-coded table, trace inspector |
| **Live Lab** | Pick Model A, Model B, and a Judge — run a scenario in real-time with side-by-side results |
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

Works with any OpenAI-compatible API:

```bash
# HuggingFace Inference API
PYTHONPATH=. python3 run_batch.py --model "your-org/your-model" --judge-model gpt-5-mini

# Local model served via vLLM/TGI
PYTHONPATH=. python3 run_batch.py --model "local-model-name"

# Finetuned checkpoint on HF Hub
PYTHONPATH=. python3 run_batch.py --model "your-username/finetuned-v2"
```

Set `HF_TOKEN` in `.env` for HuggingFace models, `OPENAI_API_KEY` for OpenAI judge.

## Project Structure

```
evalflow/                    # Core Python package
├── core.py                  # Pydantic domain models + abstract interfaces
├── simulator.py             # Sync + Async simulation engines
├── environments.py          # Domain-aware MockEnvironment
├── tracking.py              # Experiment tracker (run persistence + comparison)
├── storage.py               # SQLite backend (runs, results, datasets)
├── cache.py                 # Response cache + token-bucket rate limiter
├── stats.py                 # Statistical significance testing (Welch's t, bootstrap)
├── data/
│   └── generator.py         # Multi-domain synthetic scenario generator
├── metrics/
│   ├── metrics.py           # Deterministic metrics (success, tools, latency)
│   └── rubric.py            # LLM-as-Judge with real API + heuristic fallback
└── agents/
    ├── api_agent.py         # Sync + Async HF Inference API agents
    └── hf_agent.py          # Local PyTorch/Transformers agent

run_batch.py                 # Main evaluation pipeline (single or multi-model)
run_ab_test.py               # A/B comparison with statistical significance
dashboard_app.py             # Streamlit dashboard
tests/                       # 104 tests
.github/workflows/ci.yml     # CI: lint, typecheck, test (3.10-3.12), build
```

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: **Streamlit**)
2. Push this repo
3. Add `HF_TOKEN` and `OPENAI_API_KEY` as Space **Secrets** in Settings
4. The dashboard auto-loads `simulation_results.json`

## License

MIT
