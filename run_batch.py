#!/usr/bin/env python3
"""
run_batch.py — Main evaluation pipeline.

Generates synthetic scenarios, runs real LLM agents through the simulation
engine, evaluates with deterministic + LLM-as-Judge metrics, and persists
results with full experiment tracking.

Usage:
    PYTHONPATH=. python3 run_batch.py                          # Real LLM (if HF_TOKEN set)
    PYTHONPATH=. python3 run_batch.py --size 100 --seed 42     # Larger run
    PYTHONPATH=. python3 run_batch.py --mock                   # Force mock agents (no API)
    PYTHONPATH=. python3 run_batch.py --model Qwen/Qwen2.5-7B-Instruct:together
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import List

from dotenv import load_dotenv

from evalflow.core import (
    Agent,
    EvaluationResult,
    MetricResult,
    RunConfig,
    Scenario,
    SimulationTrace,
    StepResult,
    ToolCall,
)
from evalflow.data.generator import DatasetGenerator
from evalflow.environments import MockEnvironment
from evalflow.metrics.metrics import (
    ExpectedToolUsage,
    LatencyMetric,
    MetricEngine,
    StepCount,
    SuccessRate,
    ToolSequenceAccuracy,
)
from evalflow.metrics.rubric import RubricMetric
from evalflow.simulator import SimulationEngine
from evalflow.tracking import ExperimentTracker

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stochastic fallback agent (only used when --mock or no API token)
# ---------------------------------------------------------------------------

class StochasticAgent(Agent):
    """Fallback agent with tunable error rate — used when no API token is available."""

    def __init__(self, error_rate: float = 0.15, seed: int | None = None):
        self._error_rate = error_rate
        self._rng = random.Random(seed)

    @property
    def agent_id(self) -> str:
        return f"StochasticAgent(err={self._error_rate})"

    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        if self._rng.random() < self._error_rate:
            return ToolCall(tool_name="bad_tool", arguments={"reason": "hallucinated"})

        if len(history) == 0:
            query = current_observation.split(":")[-1].strip()[:80] or "general query"
            return ToolCall(tool_name="search", arguments={"query": query}, raw_output=f"Action: search")
        elif len(history) == 1 and self._rng.random() > 0.4:
            return ToolCall(tool_name="calculate", arguments={"expression": "100 * 1.05"}, raw_output="Action: calculate")
        else:
            return ToolCall(tool_name="done", arguments={"answer": "Based on the data retrieved, the result is 105.0"}, raw_output="Action: done")


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class EvalEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    tracker = ExperimentTracker(runs_dir=args.runs_dir)
    hf_token = os.getenv("HF_TOKEN")
    use_real_llm = bool(hf_token) and not args.mock

    # ── 1. Resolve agent + judge ──────────────────────────────────────────
    openai_client = None
    if use_real_llm:
        try:
            from openai import OpenAI
            openai_client = OpenAI(
                base_url="https://router.huggingface.co/v1/",
                api_key=hf_token,
            )
            # Quick connectivity check
            openai_client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            logger.info("Connected to HF Inference API (model: %s)", args.model)
        except Exception as e:
            logger.warning("HF API not reachable (%s) — falling back to mock agents", e)
            use_real_llm = False
            openai_client = None

    if use_real_llm:
        from evalflow.agents.api_agent import HFApiAgent
        agent = HFApiAgent(model_id=args.model, api_token=hf_token)
        agent_id = agent.agent_id
        model_name = args.model
    else:
        if not args.mock:
            logger.warning("No HF_TOKEN found — using mock StochasticAgent. Set HF_TOKEN in .env for real LLM runs.")
        agent = StochasticAgent(error_rate=0.15, seed=args.seed)
        agent_id = agent.agent_id
        model_name = "mock-stochastic"

    # ── 2. Config ─────────────────────────────────────────────────────────
    config = RunConfig(
        agent_id=agent_id,
        model_name=model_name,
        agent_config={"model": args.model if use_real_llm else "stochastic", "seed": args.seed},
        max_steps=args.max_steps,
        dataset_size=args.size,
        seed=args.seed,
    )
    run = tracker.start_run(config)
    logger.info("Run ID: %s | Agent: %s", config.run_id, agent_id)

    # ── 3. Generate dataset ───────────────────────────────────────────────
    logger.info("Generating %d synthetic scenarios (seed=%s)...", args.size, args.seed)
    generator = DatasetGenerator(seed=args.seed)
    scenarios = generator.generate_synthetic_dataset(size=args.size)
    domains = sorted({s.domain for s in scenarios})
    logger.info("Generated %d scenarios across domains: %s", len(scenarios), domains)

    # ── 4. Run simulation ─────────────────────────────────────────────────
    env = MockEnvironment(
        latency_ms=args.latency_ms,
        failure_rate=args.env_failure_rate,
        seed=args.seed,
    )
    engine = SimulationEngine(environment=env, max_steps=args.max_steps)

    logger.info("Running %d scenarios with %s...", len(scenarios), agent_id)
    t_start = time.time()
    traces = engine.run_batch(
        agent, scenarios,
        on_progress=lambda i, n, name: logger.info("  [%d/%d] %s", i + 1, n, name),
    )
    t_sim = time.time() - t_start
    logger.info("Simulation complete in %.1fs", t_sim)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    logger.info("Evaluating traces...")

    # Judge client — prefer OpenAI if key available, else use HF client
    openai_key = os.getenv("OPENAI_API_KEY")
    judge_client = None
    if openai_key and args.judge_model.startswith("gpt-"):
        try:
            from openai import OpenAI
            judge_client = OpenAI(api_key=openai_key)
            logger.info("Judge using OpenAI API (model: %s)", args.judge_model)
        except Exception as e:
            logger.warning("OpenAI judge setup failed (%s), trying HF", e)
    if judge_client is None and use_real_llm:
        judge_client = openai_client
    judge_source = f"LLM judge ({args.judge_model})" if judge_client else "heuristic fallback"
    logger.info("Rubric scoring via: %s", judge_source)

    deterministic_metrics = MetricEngine([
        SuccessRate(),
        StepCount(),
        ExpectedToolUsage(),
        ToolSequenceAccuracy(),
        LatencyMetric(),
    ])

    rubric_metrics = [
        RubricMetric(name="helpfulness", client=judge_client, model_id=args.judge_model),
        RubricMetric(name="safety", client=judge_client, model_id=args.judge_model),
        RubricMetric(name="tool_coherence", client=judge_client, model_id=args.judge_model),
    ]

    results: List[EvaluationResult] = []
    for idx, (trace, scenario) in enumerate(zip(traces, scenarios)):
        det_scores = deterministic_metrics.evaluate_trace(trace, scenario)
        metrics = {k: MetricResult(name=k, score=v) for k, v in det_scores.items()}

        for rubric in rubric_metrics:
            mr = rubric.evaluate_with_detail(trace, scenario)
            metrics[mr.name] = mr

        results.append(EvaluationResult(scenario=scenario, trace=trace, metrics=metrics))

        if (idx + 1) % 10 == 0:
            logger.info("  Evaluated %d/%d", idx + 1, len(scenarios))

    # ── 6. Persist ────────────────────────────────────────────────────────
    run = tracker.finish_run(run, results)

    # Flat JSON for dashboard
    legacy_data = []
    for r in results:
        legacy_metrics = {}
        for k, m in r.metrics.items():
            legacy_metrics[k] = m.score if isinstance(m, MetricResult) else float(m)
            if isinstance(m, MetricResult) and m.explanation:
                legacy_metrics[f"{k}_reason"] = m.explanation
        legacy_data.append({
            "scenario": r.scenario.model_dump(),
            "trace": r.trace.model_dump(),
            "metrics": legacy_metrics,
        })

    with open(args.output, "w") as f:
        json.dump(legacy_data, f, indent=2, cls=EvalEncoder)

    # ── 7. Summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RUN COMPLETE: %s", config.run_id)
    logger.info("  Model: %s", model_name)
    logger.info("  Scenarios: %d | Passed: %d | Failed: %d", run.total_scenarios, run.completed, run.failed)
    logger.info("  Duration: %.1fs", run.duration_seconds)
    logger.info("  Aggregate Metrics:")
    for k, v in sorted(run.aggregate_metrics.items()):
        if k.startswith("avg_"):
            logger.info("    %s: %.4f", k, v)
    logger.info("  Results: %s", args.output)
    logger.info("  Run file: %s/%s.json", args.runs_dir, config.run_id)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EvalFlow Batch Evaluation Pipeline")
    parser.add_argument("--size", type=int, default=20, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per scenario")
    parser.add_argument("--output", type=str, default="simulation_results.json", help="Output JSON path")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Experiment tracking directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct:together", help="HF model ID for the agent")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct:together", help="Model ID for LLM judge")
    parser.add_argument("--mock", action="store_true", help="Force mock agents (skip real LLM even if HF_TOKEN exists)")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="Simulated env latency (ms)")
    parser.add_argument("--env-failure-rate", type=float, default=0.0, help="Env stochastic failure rate (0-1)")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
