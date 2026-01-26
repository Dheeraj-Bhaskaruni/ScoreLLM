"""
evalflow.tracking — Lightweight experiment tracking.

Persists each run as a JSON file under runs/ with full config, aggregate
metrics, and per-scenario results.  Supports comparing two runs side-by-side.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .core import (
    EvaluationResult,
    MetricResult,
    RunConfig,
    RunStatus,
    RunSummary,
)

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = "runs"


class _RunEncoder(json.JSONEncoder):
    """Handles Pydantic models and enums."""

    def default(self, obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return super().default(obj)


class ExperimentTracker:
    """
    Tracks evaluation runs and persists them as structured JSON.

    Usage:
        tracker = ExperimentTracker()
        run = tracker.start_run(config)
        # ... run evaluation ...
        tracker.finish_run(run, results)
        tracker.compare_runs(run_id_a, run_id_b)
    """

    def __init__(self, runs_dir: str = DEFAULT_RUNS_DIR):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, config: RunConfig) -> RunSummary:
        summary = RunSummary(config=config, status=RunStatus.RUNNING)
        self._save(summary)
        logger.info("Started run %s", config.run_id)
        return summary

    def finish_run(
        self,
        summary: RunSummary,
        results: List[EvaluationResult],
    ) -> RunSummary:
        summary.status = RunStatus.COMPLETED
        summary.results = results
        summary.total_scenarios = len(results)
        summary.completed = sum(1 for r in results if r.trace.error is None)
        summary.failed = summary.total_scenarios - summary.completed
        summary.duration_seconds = round(time.time() - summary.config.created_at, 2)
        summary.aggregate_metrics = self._compute_aggregates(results)
        self._save(summary)
        logger.info(
            "Finished run %s — %d/%d succeeded in %.1fs",
            summary.config.run_id,
            summary.completed,
            summary.total_scenarios,
            summary.duration_seconds,
        )
        return summary

    def fail_run(self, summary: RunSummary, error: str) -> RunSummary:
        summary.status = RunStatus.FAILED
        summary.duration_seconds = round(time.time() - summary.config.created_at, 2)
        self._save(summary)
        logger.error("Run %s failed: %s", summary.config.run_id, error)
        return summary

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def load_run(self, run_id: str) -> RunSummary:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run {run_id} not found at {path}")
        with open(path) as f:
            data = json.load(f)
        return RunSummary.model_validate(data)

    def list_runs(self) -> List[Dict]:
        """Return lightweight summaries of all runs (no per-scenario results)."""
        runs = []
        for p in sorted(self.runs_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                runs.append({
                    "run_id": data["config"]["run_id"],
                    "agent_id": data["config"].get("agent_id", ""),
                    "model_name": data["config"].get("model_name", ""),
                    "status": data.get("status", ""),
                    "total": data.get("total_scenarios", 0),
                    "completed": data.get("completed", 0),
                    "failed": data.get("failed", 0),
                    "duration_s": data.get("duration_seconds", 0),
                    "metrics": data.get("aggregate_metrics", {}),
                })
            except Exception as e:
                logger.warning("Skipping corrupt run file %s: %s", p, e)
        return runs

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Dict:
        """
        Side-by-side comparison of two runs.
        Returns a dict with per-metric deltas and a recommendation.
        """
        a = self.load_run(run_id_a)
        b = self.load_run(run_id_b)

        comparison: Dict = {
            "run_a": {"run_id": run_id_a, "agent": a.config.agent_id, "model": a.config.model_name},
            "run_b": {"run_id": run_id_b, "agent": b.config.agent_id, "model": b.config.model_name},
            "metric_deltas": {},
            "recommendation": "",
        }

        all_keys = set(a.aggregate_metrics) | set(b.aggregate_metrics)
        improvements = 0
        regressions = 0

        for key in sorted(all_keys):
            val_a = a.aggregate_metrics.get(key, 0.0)
            val_b = b.aggregate_metrics.get(key, 0.0)
            delta = round(val_b - val_a, 4)
            comparison["metric_deltas"][key] = {
                "run_a": val_a,
                "run_b": val_b,
                "delta": delta,
                "improved": delta > 0,
            }
            if delta > 0.01:
                improvements += 1
            elif delta < -0.01:
                regressions += 1

        if regressions == 0 and improvements > 0:
            comparison["recommendation"] = "DEPLOY: Run B is strictly better."
        elif regressions > 0 and improvements > regressions:
            comparison["recommendation"] = "REVIEW: Run B is better on most metrics but has regressions."
        elif regressions > improvements:
            comparison["recommendation"] = "REJECT: Run B regressed on more metrics than it improved."
        else:
            comparison["recommendation"] = "NEUTRAL: No significant difference."

        return comparison

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self, summary: RunSummary) -> None:
        path = self.runs_dir / f"{summary.config.run_id}.json"
        with open(path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, cls=_RunEncoder)

    @staticmethod
    def _compute_aggregates(results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}

        # Collect all metric names across all results
        all_names: set = set()
        for r in results:
            all_names.update(r.metrics.keys())

        aggregates: Dict[str, float] = {}
        for name in sorted(all_names):
            values = []
            for r in results:
                if name in r.metrics:
                    m = r.metrics[name]
                    score = m.score if isinstance(m, MetricResult) else float(m)
                    values.append(score)
            if values:
                aggregates[f"avg_{name}"] = round(sum(values) / len(values), 4)
                aggregates[f"min_{name}"] = round(min(values), 4)
                aggregates[f"max_{name}"] = round(max(values), 4)

        return aggregates
