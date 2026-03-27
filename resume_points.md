# Resume Bullet Points: Model Eval Flow

**Context:** Built during lab research involving finetuning LLMs and developing RFID-based agents. Needed a systematic way to compare model variants (base vs. finetuned, different architectures) on domain-specific tasks before deployment.

---

### Option 1: Direct Project Entry

**Project: Model Eval Flow — LLM Model Selection Framework**

*   **Built a model selection framework** to compare LLM variants during finetuning and RFID agent development — runs multiple models through identical domain-specific scenarios, scores with an independent judge (GPT-5-mini), and produces statistically rigorous comparison results.
*   **Designed multi-model evaluation pipeline** comparing 3+ models (Qwen-7B, Llama-3.1-8B, Zephyr-7B) on identical scenarios, producing per-model leaderboards, radar profiles, and per-domain quality breakdowns to identify the best model for each use case.
*   **Implemented statistical A/B testing** using Welch's t-test with bootstrap fallback, confidence intervals, and Cohen's d effect size — ensuring observed quality differences between model variants aren't noise.
*   **Built an interactive Streamlit dashboard** with model leaderboard, overlaid radar charts, grouped metric comparisons by domain/difficulty, scatter plots, trace inspector, and live A/B testing lab with real-time judge scoring.
*   **Designed synthetic data pipeline** generating edge-case scenarios across 6 domains (finance, healthcare, tech, science) with adversarial, prompt injection, and multi-hop categories for comprehensive agent stress testing.
*   **Built production infrastructure:** SQLite storage backend with WAL mode, disk-backed response cache with TTL expiry, token-bucket rate limiter, dataset versioning via SHA-256 hashing, and async simulation engine with semaphore-controlled concurrency.
*   **Delivered with 104 pytest tests**, GitHub Actions CI (lint, type-check, test across Python 3.10-3.12), and Hugging Face Spaces deployment.

### Option 2: Integrated into Work Experience

*   **Engineered a model selection platform** for lab research — systematically compared base vs. finetuned LLM variants on domain-specific tasks using LLM-as-a-Judge scoring, replacing ad-hoc manual testing with automated, reproducible evaluation.
*   **Defined an 8-metric evaluation taxonomy** combining deterministic scoring (tool sequence accuracy via LCS, step count) with qualitative LLM-as-a-Judge rubrics (helpfulness, safety, tool coherence) and statistical significance testing.
*   **Built multi-model comparison infrastructure** supporting any OpenAI-compatible endpoint, enabling direct comparison of HuggingFace models, finetuned checkpoints, and local inference within the same evaluation harness.
*   **Applied to RFID agent development** — used the framework to evaluate and select the best-performing model for RFID-based agentic workflows, validating tool-use accuracy and safety before deployment.

### One-Liner (for conversations/interviews)

> "I was finetuning models for lab work and building RFID agents, and I needed a systematic way to compare model variants — so I built a framework that runs multiple models through the same scenarios, scores them with an independent judge, and tells you which one to deploy based on data."

### Skills to Highlight

*   **Systems:** LLM Evaluation, Model Selection, Finetuning Validation, A/B Testing, Simulation Harnesses
*   **ML/Stats:** LLM-as-a-Judge, Welch's t-test, Bootstrap CI, Cohen's d, Multi-metric Scoring
*   **Tools:** Python, Pydantic v2, PyTorch, HuggingFace (Transformers, Hub, Spaces, Inference API), OpenAI API, Streamlit, Plotly, SQLite, pytest, GitHub Actions, asyncio
