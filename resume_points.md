# Resume Bullet Points: Applied ML Engineer - EvalFlow Project

**Role:** Applied ML Engineer – AI/ML Evaluation & Simulation
**Objective:** Highlight simulation systems, dataset generation, structured insights, reliability, PyTorch/Transformers, and production engineering.

---

### Option 1: Direct Project Entry (Evaluation & Simulation)
**Project: EvalFlow - Agentic AI Evaluation Framework (v1.0)**

*   **Architected a production-grade evaluation framework** for LLM-based agents using Pydantic v2 models, supporting deterministic simulation, LLM-as-a-Judge scoring, and experiment tracking with side-by-side model comparison.
*   **Built an async simulation engine** with `asyncio` semaphore-controlled concurrency, reducing batch evaluation wall-clock time by 5x for API-backed agents across 100+ synthetic scenarios.
*   **Implemented multi-dimensional LLM-as-a-Judge** via HF Inference API, scoring agent traces on helpfulness, safety, and tool coherence with structured rubric prompts, JSON response parsing, and deterministic heuristic fallback.
*   **Designed a synthetic data pipeline** generating diverse edge-case scenarios across 6 domains (finance, healthcare, tech, science) with adversarial, i18n, prompt injection, and multi-hop categories for comprehensive agent stress testing.
*   **Built an experiment tracking system** persisting evaluation runs with unique IDs, aggregate metrics (avg/min/max per metric), and automated deploy/reject recommendations from run-to-run comparison.
*   **Integrated PyTorch + Hugging Face Transformers** for local inference evaluation (GPT-2/DistilGPT2), enabling direct comparison of open-source models against API-based agents within the same harness.
*   **Delivered with 70 pytest tests** (unit, integration, end-to-end), GitHub Actions CI (lint, type-check, test across Python 3.10-3.12), and Hugging Face Hub/Spaces deployment support.

### Option 2: Integrated into Work Experience
*   **Engineered an automated evaluation platform** for agentic workflows with async batch processing, reducing manual testing effort by 90% while covering 9 scenario categories including adversarial and safety edge cases.
*   **Defined and implemented an 8-metric evaluation taxonomy** combining deterministic scoring (tool sequence accuracy via LCS, success rate) with qualitative LLM-as-a-Judge rubrics, enabling data-driven model deployment decisions.
*   **Built experiment tracking infrastructure** with run persistence, aggregate metric computation, and automated A/B comparison — providing deploy/reject recommendations backed by quantitative metric deltas.
*   **Designed a domain-aware simulation environment** with realistic response banks across 5 domains, AST-safe arithmetic evaluation, configurable latency simulation, and stochastic failure injection for reliability testing.

### Skills to Add
*   **Systems:** AI Evaluation Frameworks, Simulation Harnesses, Experiment Tracking, Async Batch Processing, LLM-as-a-Judge
*   **Metrics:** Tool Sequence Accuracy (LCS), Rubric Scoring, Multi-dimensional Evaluation, Run Comparison
*   **Tools:** Python, Pydantic, PyTorch, Hugging Face (Transformers, Hub, Spaces, Inference API), Streamlit, Plotly, pytest, GitHub Actions, asyncio
- Implemented Welch's t-test with bootstrap fallback for A/B significance testing
