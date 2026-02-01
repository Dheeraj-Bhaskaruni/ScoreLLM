"""Tests for evalflow.metrics."""
import pytest

from evalflow.core import MetricResult, Scenario, SimulationTrace, StepResult, ToolCall
from evalflow.metrics.metrics import (
    ExpectedToolUsage,
    LatencyMetric,
    MetricEngine,
    StepCount,
    SuccessRate,
    ToolSequenceAccuracy,
)
from evalflow.metrics.rubric import RubricMetric, _parse_judge_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(
    tools: list[str],
    final_output: str | None = "answer",
    error: str | None = None,
) -> SimulationTrace:
    steps = []
    for i, tool in enumerate(tools):
        steps.append(StepResult(
            step_id=i,
            input_state=f"obs_{i}",
            action=ToolCall(tool_name=tool, arguments={}),
            output_observation=f"result_{i}",
        ))
    return SimulationTrace(
        scenario_id="t1",
        agent_id="test",
        start_time=100.0,
        end_time=102.5,
        steps=steps,
        final_output=final_output,
        error=error,
    )


@pytest.fixture
def scenario_with_expected():
    return Scenario(
        id="s1",
        name="Test",
        description="Test scenario",
        initial_context="Test",
        expected_tool_sequence=["search", "calculate"],
    )


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

class TestSuccessRate:
    def test_success(self, scenario_with_expected):
        trace = _make_trace(["search", "done"], final_output="42")
        assert SuccessRate().evaluate(trace, scenario_with_expected) == 1.0

    def test_failure_no_output(self, scenario_with_expected):
        trace = _make_trace(["search"], final_output=None)
        assert SuccessRate().evaluate(trace, scenario_with_expected) == 0.0

    def test_failure_with_error(self, scenario_with_expected):
        trace = _make_trace(["search"], error="crash")
        assert SuccessRate().evaluate(trace, scenario_with_expected) == 0.0


class TestStepCount:
    def test_counts_steps(self, scenario_with_expected):
        trace = _make_trace(["search", "calculate", "done"])
        assert StepCount().evaluate(trace, scenario_with_expected) == 3.0

    def test_zero_steps(self, scenario_with_expected):
        trace = _make_trace([])
        assert StepCount().evaluate(trace, scenario_with_expected) == 0.0


class TestExpectedToolUsage:
    def test_full_overlap(self, scenario_with_expected):
        trace = _make_trace(["search", "calculate", "done"])
        assert ExpectedToolUsage().evaluate(trace, scenario_with_expected) == 1.0

    def test_partial_overlap(self, scenario_with_expected):
        trace = _make_trace(["search", "done"])
        assert ExpectedToolUsage().evaluate(trace, scenario_with_expected) == 0.5

    def test_no_overlap(self, scenario_with_expected):
        trace = _make_trace(["writer", "done"])
        assert ExpectedToolUsage().evaluate(trace, scenario_with_expected) == 0.0

    def test_no_expected_tools(self):
        scenario = Scenario(name="t", description="t", initial_context="t")
        trace = _make_trace(["search"])
        assert ExpectedToolUsage().evaluate(trace, scenario) == 1.0


class TestToolSequenceAccuracy:
    def test_exact_match(self, scenario_with_expected):
        trace = _make_trace(["search", "calculate", "done"])
        assert ToolSequenceAccuracy().evaluate(trace, scenario_with_expected) == 1.0

    def test_partial_sequence(self, scenario_with_expected):
        trace = _make_trace(["search", "done"])
        assert ToolSequenceAccuracy().evaluate(trace, scenario_with_expected) == 0.5

    def test_reversed_sequence(self, scenario_with_expected):
        trace = _make_trace(["calculate", "search", "done"])
        # LCS is still 1 (either search or calculate matches in order)
        # Actually LCS of [search,calculate] vs [calculate,search] = 1
        assert ToolSequenceAccuracy().evaluate(trace, scenario_with_expected) == 0.5

    def test_no_expected(self):
        scenario = Scenario(name="t", description="t", initial_context="t")
        trace = _make_trace(["search"])
        assert ToolSequenceAccuracy().evaluate(trace, scenario) == 1.0


class TestLatencyMetric:
    def test_returns_duration(self, scenario_with_expected):
        trace = _make_trace(["search"])
        assert LatencyMetric().evaluate(trace, scenario_with_expected) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Metric Engine
# ---------------------------------------------------------------------------

class TestMetricEngine:
    def test_evaluates_all_metrics(self, scenario_with_expected):
        engine = MetricEngine([SuccessRate(), StepCount(), ExpectedToolUsage()])
        trace = _make_trace(["search", "calculate", "done"])
        results = engine.evaluate_trace(trace, scenario_with_expected)

        assert "SuccessRate" in results
        assert "StepCount" in results
        assert "ExpectedToolUsage" in results
        assert results["SuccessRate"] == 1.0

    def test_detailed_returns_metric_results(self, scenario_with_expected):
        engine = MetricEngine([SuccessRate()])
        trace = _make_trace(["done"])
        results = engine.evaluate_trace_detailed(trace, scenario_with_expected)
        assert isinstance(results["SuccessRate"], MetricResult)


# ---------------------------------------------------------------------------
# Rubric metric (heuristic fallback)
# ---------------------------------------------------------------------------

class TestRubricMetricFallback:
    def test_heuristic_success(self, scenario_with_expected):
        rubric = RubricMetric(name="helpfulness")  # No client = heuristic
        trace = _make_trace(["search", "calculate", "done"])
        result = rubric.evaluate_with_detail(trace, scenario_with_expected)

        assert 1.0 <= result.score <= 5.0
        assert result.metadata["source"] == "heuristic_fallback"
        assert result.explanation  # Should have an explanation

    def test_heuristic_failure(self, scenario_with_expected):
        trace = _make_trace(["bad_tool"], final_output=None, error="crash")
        rubric = RubricMetric(name="helpfulness")
        result = rubric.evaluate_with_detail(trace, scenario_with_expected)
        assert result.score <= 2.0


class TestJudgeResponseParser:
    def test_clean_json(self):
        raw = '{"score": 4, "explanation": "Good answer"}'
        parsed = _parse_judge_response(raw)
        assert parsed["score"] == 4
        assert "Good answer" in parsed["explanation"]

    def test_json_in_markdown(self):
        raw = '```json\n{"score": 5, "explanation": "Perfect"}\n```'
        parsed = _parse_judge_response(raw)
        assert parsed["score"] == 5

    def test_bare_score(self):
        raw = "Score: 3. The answer was okay."
        parsed = _parse_judge_response(raw)
        assert parsed["score"] == 3

    def test_unparseable(self):
        raw = "I cannot evaluate this."
        parsed = _parse_judge_response(raw)
        assert parsed["score"] == 0
        assert "PARSE_FAILURE" in parsed["explanation"]
