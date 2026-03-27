# Resume Bullet Points: Applied ML Engineer - EvalFlow Project

**Role:** Applied ML Engineer — AI/ML Evaluation & Model Selection
**Framing:** EvalFlow solves the model selection problem: given N candidate LLMs and domain-specific requirements, which model should you deploy?

---

### Option 1: Direct Project Entry

**Project: EvalFlow — Data-Driven LLM Model Selection Framework**

*   **Built a model selection framework** that evaluates multiple LLM agents across domain-specific scenarios using LLM-as-a-Judge scoring (GPT-5-mini), statistical A/B testing, and multi-model comparison — replacing subjective benchmark-chasing with data-driven deployment decisions.
*   **Designed multi-model evaluation pipeline** comparing 3+ models (Qwen-7B, Llama-3.1-8B, Zephyr-7B) on identical scenarios with an independent judge, producing per-model leaderboards, radar profiles, and per-domain quality breakdowns.
*   **Implemented statistical significance testing** for model comparison using Welch's t-test with bootstrap fallback, confidence intervals, and Cohen's d effect size — ensuring observed differences aren't noise.
*   **Built an interactive Streamlit dashboard** with model leaderboard, overlaid radar charts, grouped metric comparisons by domain/difficulty, scatter plots, trace inspector, and live A/B testing lab with real-time judge scoring.
*   **Designed a synthetic data pipeline** generating edge-case scenarios across 6 domains (finance, healthcare, tech, science) with adversarial, prompt injection, and multi-hop categories for comprehensive agent stress testing.
*   **Built production infrastructure:** SQLite storage backend with WAL mode, disk-backed response cache with TTL expiry, token-bucket rate limiter, dataset versioning via SHA-256 hashing, and async simulation engine with semaphore-controlled concurrency.
*   **Delivered with 104 pytest tests**, GitHub Actions CI (lint, type-check, test across Python 3.10-3.12), and Hugging Face Spaces deployment.

### Option 2: Integrated into Work Experience

*   **Engineered a model selection platform** that compares LLM agents across domain-specific scenarios using LLM-as-a-Judge scoring — enabling teams to pick the right model based on quality, safety, and efficiency data instead of public benchmarks.
*   **Defined an 8-metric evaluation taxonomy** combining deterministic scoring (tool sequence accuracy via LCS, step count) with qualitative LLM-as-a-Judge rubrics (helpfulness, safety, tool coherence) and statistical significance testing.
*   **Built multi-model comparison infrastructure** running N candidate models on identical test suites, with per-domain/difficulty breakdowns, radar profiles, and automated deploy/reject recommendations backed by p-values and effect sizes.
*   **Designed a domain-aware simulation environment** with realistic response banks across 5 domains, AST-safe arithmetic evaluation, configurable latency simulation, and stochastic failure injection for reliability testing.

### One-Liner (for conversations/interviews)

> "I built a framework that helps teams pick the right LLM for their use case — it runs multiple models through the same domain-specific scenarios, scores them with an independent judge model, and compares results statistically so you deploy based on data, not benchmark hype."

### Skills to Highlight

*   **Systems:** LLM Evaluation, Model Selection, A/B Testing, Simulation Harnesses, Experiment Tracking
*   **ML/Stats:** LLM-as-a-Judge, Welch's t-test, Bootstrap CI, Cohen's d, Multi-metric Scoring
*   **Tools:** Python, Pydantic v2, PyTorch, HuggingFace (Transformers, Hub, Spaces, Inference API), OpenAI API, Streamlit, Plotly, SQLite, pytest, GitHub Actions, asyncio
