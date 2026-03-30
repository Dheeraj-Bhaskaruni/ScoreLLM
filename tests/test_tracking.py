"""Tests for evalflow.tracking — experiment tracker."""

import pytest

from evalflow.core import (
    EvaluationResult,
    MetricResult,
    RunConfig,
    RunStatus,
    Scenario,
    SimulationTrace,
)
from evalflow.tracking import ExperimentTracker


@pytest.fixture
def tmp_tracker(tmp_path):
    return ExperimentTracker(runs_dir=str(tmp_path / "runs"))


@pytest.fixture
def sample_results():
    scenario = Scenario(
        id="s1",
        name="Test",
        description="Test",
        initial_context="Test",
        expected_tool_sequence=["search"],
    )
    trace = SimulationTrace(
        scenario_id="s1",
        agent_id="test",
        start_time=100.0,
        end_time=102.0,
        final_output="answer",
    )
    return [
        EvaluationResult(
            scenario=scenario,
            trace=trace,
            metrics={"SuccessRate": MetricResult(name="SuccessRate", score=1.0)},
        )
    ]


class TestExperimentTracker:
    def test_start_and_finish_run(self, tmp_tracker, sample_results):
        config = RunConfig(run_id="test-run-1", agent_id="test-agent")
        run = tmp_tracker.start_run(config)

        assert run.status == RunStatus.RUNNING
        assert run.config.run_id == "test-run-1"

        run = tmp_tracker.finish_run(run, sample_results)
        assert run.status == RunStatus.COMPLETED
        assert run.total_scenarios == 1
        assert run.completed == 1
        assert run.failed == 0

    def test_fail_run(self, tmp_tracker):
        config = RunConfig(run_id="fail-run", agent_id="test")
        run = tmp_tracker.start_run(config)
        run = tmp_tracker.fail_run(run, "Something broke")
        assert run.status == RunStatus.FAILED

    def test_load_run(self, tmp_tracker, sample_results):
        config = RunConfig(run_id="load-test", agent_id="test")
        run = tmp_tracker.start_run(config)
        tmp_tracker.finish_run(run, sample_results)

        loaded = tmp_tracker.load_run("load-test")
        assert loaded.config.run_id == "load-test"
        assert loaded.status == RunStatus.COMPLETED

    def test_load_missing_run_raises(self, tmp_tracker):
        with pytest.raises(FileNotFoundError):
            tmp_tracker.load_run("nonexistent")

    def test_list_runs(self, tmp_tracker, sample_results):
        for i in range(3):
            config = RunConfig(run_id=f"run-{i}", agent_id=f"agent-{i}")
            run = tmp_tracker.start_run(config)
            tmp_tracker.finish_run(run, sample_results)

        runs = tmp_tracker.list_runs()
        assert len(runs) == 3
        assert all("run_id" in r for r in runs)

    def test_compare_runs_deploy(self, tmp_tracker):
        # Run A: worse
        config_a = RunConfig(run_id="run-a", agent_id="baseline")
        run_a = tmp_tracker.start_run(config_a)
        results_a = [
            EvaluationResult(
                scenario=Scenario(id="s1", name="T", description="T", initial_context="T"),
                trace=SimulationTrace(
                    scenario_id="s1", agent_id="baseline", start_time=0, end_time=1, final_output="a"
                ),
                metrics={"SuccessRate": MetricResult(name="SuccessRate", score=0.5)},
            )
        ]
        tmp_tracker.finish_run(run_a, results_a)

        # Run B: better
        config_b = RunConfig(run_id="run-b", agent_id="candidate")
        run_b = tmp_tracker.start_run(config_b)
        results_b = [
            EvaluationResult(
                scenario=Scenario(id="s1", name="T", description="T", initial_context="T"),
                trace=SimulationTrace(
                    scenario_id="s1", agent_id="candidate", start_time=0, end_time=1, final_output="b"
                ),
                metrics={"SuccessRate": MetricResult(name="SuccessRate", score=1.0)},
            )
        ]
        tmp_tracker.finish_run(run_b, results_b)

        comparison = tmp_tracker.compare_runs("run-a", "run-b")
        assert "DEPLOY" in comparison["recommendation"]

    def test_compare_runs_reject(self, tmp_tracker):
        config_a = RunConfig(run_id="run-good", agent_id="good")
        run_a = tmp_tracker.start_run(config_a)
        results_a = [
            EvaluationResult(
                scenario=Scenario(id="s1", name="T", description="T", initial_context="T"),
                trace=SimulationTrace(scenario_id="s1", agent_id="good", start_time=0, end_time=1, final_output="ok"),
                metrics={"SuccessRate": MetricResult(name="SuccessRate", score=1.0)},
            )
        ]
        tmp_tracker.finish_run(run_a, results_a)

        config_b = RunConfig(run_id="run-bad", agent_id="bad")
        run_b = tmp_tracker.start_run(config_b)
        results_b = [
            EvaluationResult(
                scenario=Scenario(id="s1", name="T", description="T", initial_context="T"),
                trace=SimulationTrace(
                    scenario_id="s1", agent_id="bad", start_time=0, end_time=1, final_output=None, error="crash"
                ),
                metrics={"SuccessRate": MetricResult(name="SuccessRate", score=0.0)},
            )
        ]
        tmp_tracker.finish_run(run_b, results_b)

        comparison = tmp_tracker.compare_runs("run-good", "run-bad")
        assert "REJECT" in comparison["recommendation"]

    def test_aggregate_metrics(self, tmp_tracker):
        config = RunConfig(run_id="agg-test", agent_id="test")
        run = tmp_tracker.start_run(config)
        results = [
            EvaluationResult(
                scenario=Scenario(id=f"s{i}", name="T", description="T", initial_context="T"),
                trace=SimulationTrace(
                    scenario_id=f"s{i}", agent_id="test", start_time=0, end_time=1, final_output="ok"
                ),
                metrics={"SuccessRate": MetricResult(name="SuccessRate", score=float(i % 2))},
            )
            for i in range(4)
        ]
        run = tmp_tracker.finish_run(run, results)

        assert "avg_SuccessRate" in run.aggregate_metrics
        assert run.aggregate_metrics["avg_SuccessRate"] == 0.5
        assert run.aggregate_metrics["min_SuccessRate"] == 0.0
        assert run.aggregate_metrics["max_SuccessRate"] == 1.0
