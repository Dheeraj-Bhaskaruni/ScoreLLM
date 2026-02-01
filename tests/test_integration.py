"""Integration tests — full pipeline end-to-end."""
import json
import tempfile
from pathlib import Path

import pytest

from evalflow.core import EvaluationResult, MetricResult, RunConfig
from evalflow.data.generator import DatasetGenerator
from evalflow.environments import MockEnvironment
from evalflow.metrics.metrics import (
    ExpectedToolUsage,
    MetricEngine,
    StepCount,
    SuccessRate,
    ToolSequenceAccuracy,
)
from evalflow.metrics.rubric import RubricMetric
from evalflow.simulator import SimulationEngine
from evalflow.tracking import ExperimentTracker
from tests.conftest import DeterministicAgent


class TestFullPipeline:
    """End-to-end: generate -> simulate -> evaluate -> track."""

    def test_pipeline(self, tmp_path):
        # 1. Generate
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=10)
        assert len(scenarios) == 10

        # 2. Simulate
        env = MockEnvironment(seed=42)
        engine = SimulationEngine(environment=env, max_steps=5)
        agent = DeterministicAgent(["search", "calculate", "done"])
        traces = engine.run_batch(agent, scenarios)
        assert len(traces) == 10

        # 3. Evaluate
        metric_engine = MetricEngine([SuccessRate(), StepCount(), ExpectedToolUsage()])
        rubric = RubricMetric(name="helpfulness")  # Heuristic fallback

        results = []
        for trace, scenario in zip(traces, scenarios):
            det_scores = metric_engine.evaluate_trace(trace, scenario)
            metrics = {k: MetricResult(name=k, score=v) for k, v in det_scores.items()}
            rubric_result = rubric.evaluate_with_detail(trace, scenario)
            metrics["helpfulness"] = rubric_result
            results.append(EvaluationResult(scenario=scenario, trace=trace, metrics=metrics))

        assert len(results) == 10
        assert all(r.metrics["SuccessRate"].score == 1.0 for r in results)

        # 4. Track
        tracker = ExperimentTracker(runs_dir=str(tmp_path / "runs"))
        config = RunConfig(run_id="integration-test", agent_id="DeterministicAgent", seed=42)
        run = tracker.start_run(config)
        run = tracker.finish_run(run, results)

        assert run.total_scenarios == 10
        assert run.completed == 10
        assert "avg_SuccessRate" in run.aggregate_metrics

        # 5. Verify persistence
        loaded = tracker.load_run("integration-test")
        assert loaded.config.run_id == "integration-test"
        assert loaded.total_scenarios == 10

    def test_pipeline_with_failures(self, tmp_path):
        """Ensure pipeline handles agent failures gracefully."""
        from tests.conftest import CrashingAgent

        gen = DatasetGenerator(seed=1)
        scenarios = gen.generate_synthetic_dataset(size=5, include_edge_cases=False)

        env = MockEnvironment()
        engine = SimulationEngine(environment=env, max_steps=5)
        agent = CrashingAgent(crash_on_step=0)
        traces = engine.run_batch(agent, scenarios)

        # All should have errors
        assert all(t.error is not None for t in traces)

        metric_engine = MetricEngine([SuccessRate()])
        results = []
        for trace, scenario in zip(traces, scenarios):
            scores = metric_engine.evaluate_trace(trace, scenario)
            metrics = {k: MetricResult(name=k, score=v) for k, v in scores.items()}
            results.append(EvaluationResult(scenario=scenario, trace=trace, metrics=metrics))

        # All should fail
        assert all(r.metrics["SuccessRate"].score == 0.0 for r in results)

        tracker = ExperimentTracker(runs_dir=str(tmp_path / "runs"))
        config = RunConfig(run_id="crash-test", agent_id="CrashingAgent")
        run = tracker.start_run(config)
        run = tracker.finish_run(run, results)
        assert run.failed == 5

    def test_serialization_roundtrip(self):
        """Ensure results can be serialized to JSON and back."""
        scenario = scenarios = DatasetGenerator(seed=42).generate_synthetic_dataset(size=1)[0]
        trace = SimulationEngine(
            environment=MockEnvironment(),
        ).run_scenario(
            DeterministicAgent(["search", "done"]),
            scenario,
        )
        result = EvaluationResult(
            scenario=scenario,
            trace=trace,
            metrics={"test": MetricResult(name="test", score=3.5, explanation="good")},
        )

        # Serialize
        data = result.model_dump()
        json_str = json.dumps(data)

        # Deserialize
        loaded = EvaluationResult.model_validate(json.loads(json_str))
        assert loaded.scenario.name == scenario.name
        assert loaded.metrics["test"].score == 3.5
