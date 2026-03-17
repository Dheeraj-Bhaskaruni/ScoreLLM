#!/usr/bin/env python3
"""
run_ab_test.py — A/B model comparison pipeline.

Runs two different LLM agents (baseline vs candidate) on the SAME set of
scenarios, scores both with a SEPARATE stronger judge model, and produces
a side-by-side comparison with a deploy/reject recommendation.

This is how real production model evaluation works at companies like Apple,
Google, and OpenAI — you never ship a model without comparing it against
the current production version on structured benchmarks.

Usage:
    # Compare two models (judge defaults to a separate model)
    PYTHONPATH=. python3 run_ab_test.py \
        --baseline "HuggingFaceH4/zephyr-7b-beta:featherless-ai" \
        --candidate "Qwen/Qwen2.5-7B-Instruct:together" \
        --judge "Qwen/Qwen2.5-7B-Instruct:together" \
        --size 15

    # With a stronger judge (ideal setup)
    PYTHONPATH=. python3 run_ab_test.py \
        --baseline "mistralai/Mistral-7B-Instruct-v0.3:together" \
        --candidate "Qwen/Qwen2.5-7B-Instruct:together" \
        --judge "meta-llama/Llama-3.3-70B-Instruct:together" \
        --size 15
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv

from evalflow.agents.api_agent import HFApiAgent
from evalflow.core import EvaluationResult, MetricResult, RunConfig
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
from evalflow.stats import compare_ab_scores, format_stat_table
from evalflow.storage import StorageBackend
from evalflow.tracking import ExperimentTracker

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class EvalEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


def evaluate_agent(
    agent: HFApiAgent,
    scenarios: list,
    env: MockEnvironment,
    engine: SimulationEngine,
    rubric_metrics: list[RubricMetric],
    det_metrics: MetricEngine,
    tracker: ExperimentTracker,
    model_name: str,
    label: str,
    seed: int,
) -> tuple[list[EvaluationResult], str]:
    """Run a single agent through the full eval pipeline, return results + run_id."""

    config = RunConfig(
        agent_id=agent.agent_id,
        model_name=model_name,
        agent_config={"label": label},
        max_steps=engine.max_steps,
        dataset_size=len(scenarios),
        seed=seed,
    )
    run = tracker.start_run(config)
    logger.info("[%s] Run ID: %s | Model: %s", label, config.run_id, model_name)

    # Simulate
    logger.info("[%s] Running %d scenarios...", label, len(scenarios))
    t0 = time.time()
    traces = engine.run_batch(
        agent, scenarios,
        on_progress=lambda i, n, name: logger.info("  [%s %d/%d] %s", label, i + 1, n, name),
    )
    logger.info("[%s] Simulation done in %.1fs", label, time.time() - t0)

    # Evaluate
    logger.info("[%s] Scoring with LLM judge...", label)
    results: List[EvaluationResult] = []
    for trace, scenario in zip(traces, scenarios):
        det_scores = det_metrics.evaluate_trace(trace, scenario)
        metrics = {k: MetricResult(name=k, score=v) for k, v in det_scores.items()}
        for rubric in rubric_metrics:
            mr = rubric.evaluate_with_detail(trace, scenario)
            metrics[mr.name] = mr
        results.append(EvaluationResult(scenario=scenario, trace=trace, metrics=metrics))

    run = tracker.finish_run(run, results)
    return results, config.run_id


def collect_per_scenario_scores(
    results: List[EvaluationResult],
) -> Dict[str, List[float]]:
    """Extract per-scenario metric scores into {metric_name: [scores]}."""
    scores: Dict[str, List[float]] = {}
    for r in results:
        for k, m in r.metrics.items():
            val = m.score if isinstance(m, MetricResult) else float(m)
            scores.setdefault(k, []).append(val)
    return scores


def print_comparison(
    comparison: Dict,
    results_a: list,
    results_b: list,
    dataset_hash: str = "",
) -> None:
    """Pretty-print the A/B comparison with statistical significance."""
    print("\n" + "=" * 70)
    print("A/B TEST RESULTS")
    print("=" * 70)

    run_a = comparison["run_a"]
    run_b = comparison["run_b"]
    print(f"\n  Baseline (A):  {run_a['model']}  [{run_a['run_id']}]")
    print(f"  Candidate (B): {run_b['model']}  [{run_b['run_id']}]")
    if dataset_hash:
        print(f"  Dataset:       {dataset_hash} ({len(results_a)} scenarios)")

    print(f"\n{'Metric':<30} {'Baseline':>10} {'Candidate':>10} {'Delta':>10} {'Result':>10}")
    print("-" * 70)

    for metric, data in sorted(comparison["metric_deltas"].items()):
        if not metric.startswith("avg_"):
            continue
        name = metric.replace("avg_", "")
        delta_str = f"{data['delta']:+.4f}"
        tag = "better" if data["improved"] else ("worse" if data["delta"] < -0.01 else "same")
        print(f"  {name:<28} {data['run_a']:>10.4f} {data['run_b']:>10.4f} {delta_str:>10} {tag:>10}")

    print(f"\n  RECOMMENDATION: {comparison['recommendation']}")
    print("=" * 70)

    # --- Statistical Significance ---
    scores_a = collect_per_scenario_scores(results_a)
    scores_b = collect_per_scenario_scores(results_b)
    stat_results = compare_ab_scores(scores_a, scores_b)

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (alpha=0.05)")
    print("=" * 70)
    print(format_stat_table(stat_results))

    sig_count = sum(1 for r in stat_results.values() if r.significant)
    total = len(stat_results)
    print(f"\n  {sig_count}/{total} metrics show statistically significant differences.")
    for name, r in sorted(stat_results.items()):
        if r.significant:
            winner = "Candidate (B)" if r.delta > 0 else "Baseline (A)"
            print(f"    {name}: {winner} wins (p={r.p_value:.4f}, effect={r.effect_size:+.2f} [{_effect_label(r.effect_size)}])")
    print("=" * 70)

    # Show per-scenario breakdown for interesting cases
    print("\nPer-Scenario Highlights:")
    print("-" * 70)
    for ra, rb in zip(results_a, results_b):
        h_a = ra.metrics.get("helpfulness")
        h_b = rb.metrics.get("helpfulness")
        score_a = h_a.score if h_a else 0
        score_b = h_b.score if h_b else 0

        if abs(score_a - score_b) >= 1.0:
            print(f"\n  {ra.scenario.name} [{ra.scenario.category}]")
            print(f"    Baseline:  helpfulness={score_a}/5")
            if h_a and h_a.explanation:
                print(f"      Judge: {h_a.explanation[:100]}")
            print(f"    Candidate: helpfulness={score_b}/5")
            if h_b and h_b.explanation:
                print(f"      Judge: {h_b.explanation[:100]}")


def _effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "trivial"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def main():
    parser = argparse.ArgumentParser(description="EvalFlow A/B Model Comparison")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline model ID (current prod)")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate model ID (new model)")
    parser.add_argument("--judge", type=str, default="Qwen/Qwen2.5-7B-Instruct:together",
                        help="Judge model ID (should be stronger than both agents)")
    parser.add_argument("--size", type=int, default=15, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=10, metavar="N", help="Max steps per scenario")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Experiment tracking directory")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not hf_token:
        print("ERROR: HF_TOKEN not found in .env — required for running agents.")
        sys.exit(1)

    # ── Setup clients ─────────────────────────────────────────────────────
    from openai import OpenAI

    # Agent client — always HF
    agent_client_url = "https://router.huggingface.co/v1/"

    # Judge client — OpenAI if key provided, else HF
    if openai_key and args.judge.startswith("gpt-"):
        judge_client = OpenAI(api_key=openai_key)  # Uses api.openai.com
        logger.info("Judge using OpenAI API (model: %s)", args.judge)
    else:
        judge_client = OpenAI(base_url=agent_client_url, api_key=hf_token)
        if openai_key and not args.judge.startswith("gpt-"):
            logger.info("OPENAI_API_KEY found but judge model '%s' is not a GPT model — using HF API", args.judge)
        logger.info("Judge using HF Inference API (model: %s)", args.judge)

    # Verify connectivity
    logger.info("Verifying API connectivity...")
    hf_client = OpenAI(base_url=agent_client_url, api_key=hf_token)
    for model_id in [args.baseline, args.candidate]:
        try:
            hf_client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": "ping"}], max_tokens=5,
            )
            logger.info("  Agent model %s — OK", model_id)
        except Exception as e:
            logger.error("  Agent model %s — FAILED: %s", model_id, e)
            sys.exit(1)
    try:
        judge_client.chat.completions.create(
            model=args.judge, messages=[{"role": "user", "content": "ping"}], max_tokens=5,
        )
        logger.info("  Judge model %s — OK", args.judge)
    except Exception as e:
        logger.error("  Judge model %s — FAILED: %s", args.judge, e)
        sys.exit(1)

    tracker = ExperimentTracker(runs_dir=args.runs_dir)
    storage = StorageBackend()

    # Generate ONE dataset — both agents get the EXACT same scenarios
    logger.info("Generating %d scenarios (seed=%d)...", args.size, args.seed)
    generator = DatasetGenerator(seed=args.seed)
    scenarios = generator.generate_synthetic_dataset(size=args.size)

    # Dataset versioning — hash the scenario set for reproducibility
    dataset_hash = generator.compute_dataset_hash(scenarios)
    logger.info("Dataset hash: %s (%d scenarios)", dataset_hash, len(scenarios))
    storage.insert_dataset(
        dataset_hash=dataset_hash,
        size=len(scenarios),
        seed=args.seed,
        domains=sorted({s.domain for s in scenarios}),
        scenarios=[s.model_dump() for s in scenarios],
    )

    env = MockEnvironment(seed=args.seed)
    engine = SimulationEngine(environment=env, max_steps=args.max_steps)

    # Deterministic metrics (shared)
    det_metrics = MetricEngine([
        SuccessRate(), StepCount(), ExpectedToolUsage(),
        ToolSequenceAccuracy(), LatencyMetric(),
    ])

    # Rubric metrics — using the JUDGE model (separate from agents)
    rubric_metrics = [
        RubricMetric(name="helpfulness", client=judge_client, model_id=args.judge),
        RubricMetric(name="safety", client=judge_client, model_id=args.judge),
        RubricMetric(name="tool_coherence", client=judge_client, model_id=args.judge),
    ]

    logger.info("Judge model: %s (separate from both agents)", args.judge)

    # ── Run Agent A (Baseline) ────────────────────────────────────────────
    agent_a = HFApiAgent(model_id=args.baseline, api_token=hf_token)
    results_a, run_id_a = evaluate_agent(
        agent_a, scenarios, env, engine, rubric_metrics, det_metrics,
        tracker, args.baseline, "BASELINE", args.seed,
    )

    # ── Run Agent B (Candidate) ───────────────────────────────────────────
    agent_b = HFApiAgent(model_id=args.candidate, api_token=hf_token)
    results_b, run_id_b = evaluate_agent(
        agent_b, scenarios, env, engine, rubric_metrics, det_metrics,
        tracker, args.candidate, "CANDIDATE", args.seed,
    )

    # ── Compare ───────────────────────────────────────────────────────────
    comparison = tracker.compare_runs(run_id_a, run_id_b)
    comparison["dataset_hash"] = dataset_hash
    print_comparison(comparison, results_a, results_b, dataset_hash=dataset_hash)

    # Save comparison
    out_path = f"ab_comparison_{run_id_a}_vs_{run_id_b}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison saved to %s", out_path)

    # Persist to SQLite
    for run_id, results, model_name in [
        (run_id_a, results_a, args.baseline),
        (run_id_b, results_b, args.candidate),
    ]:
        storage.insert_run(
            run_id=run_id,
            agent_id=f"HFApiAgent({model_name})",
            model_name=model_name,
            config={"judge": args.judge, "seed": args.seed, "size": args.size},
            status="completed",
            dataset_hash=dataset_hash,
        )
        storage.insert_results(
            run_id=run_id,
            results=[
                {
                    "scenario": r.scenario.model_dump(),
                    "trace": r.trace.model_dump(),
                    "metrics": {k: m.model_dump() for k, m in r.metrics.items()},
                }
                for r in results
            ],
        )
    logger.info("Results persisted to SQLite (evalflow.db)")


if __name__ == "__main__":
    main()
