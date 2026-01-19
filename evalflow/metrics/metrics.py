"""
evalflow.metrics.metrics — Deterministic metrics for agent trace evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from ..core import MetricResult, Scenario, SimulationTrace


class Metric(ABC):
    """Base class for all evaluation metrics."""

    @abstractmethod
    def evaluate(self, trace: SimulationTrace, scenario: Scenario) -> float:
        ...

    @property
    def metric_name(self) -> str:
        return self.__class__.__name__


class SuccessRate(Metric):
    """Binary: 1.0 if the agent finished with a final answer and no errors, else 0.0."""

    def evaluate(self, trace: SimulationTrace, scenario: Scenario) -> float:
        if trace.error:
            return 0.0
        return 1.0 if trace.final_output is not None else 0.0


class StepCount(Metric):
    """Returns the number of steps taken."""

    def evaluate(self, trace: SimulationTrace, scenario: Scenario) -> float:
        return float(len(trace.steps))


class ExpectedToolUsage(Metric):
    """
    Jaccard-style overlap between expected and actual tool sets.
    Returns fraction of expected tools that were actually called.
    """

    def evaluate(self, trace: SimulationTrace, scenario: Scenario) -> float:
        if not scenario.expected_tool_sequence:
            return 1.0
        expected = set(scenario.expected_tool_sequence)
        actual = set(trace.tool_sequence)
        if not expected:
            return 1.0
        return len(actual & expected) / len(expected)


class ToolSequenceAccuracy(Metric):
    """
    Ordered sequence match — stricter than set overlap.
    Returns longest-common-subsequence ratio against expected sequence.
    """

    def evaluate(self, trace: SimulationTrace, scenario: Scenario) -> float:
        if not scenario.expected_tool_sequence:
            return 1.0
        expected = scenario.expected_tool_sequence
        actual = trace.tool_sequence

        # LCS via DP
        m, n = len(expected), len(actual)
        if m == 0:
            return 1.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected[i - 1] == actual[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n] / m


class LatencyMetric(Metric):
    """Returns the total simulation duration in seconds."""

    def evaluate(self, trace: SimulationTrace, scenario: Scenario) -> float:
        return round(trace.duration, 4)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MetricEngine:
    """Runs a suite of metrics over a trace and returns structured results."""

    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def evaluate_trace(self, trace: SimulationTrace, scenario: Scenario) -> Dict[str, float]:
        """Legacy dict-based interface for backward compatibility."""
        return {m.metric_name: m.evaluate(trace, scenario) for m in self.metrics}

    def evaluate_trace_detailed(
        self, trace: SimulationTrace, scenario: Scenario
    ) -> Dict[str, MetricResult]:
        """Returns MetricResult objects with scores + metadata."""
        results: Dict[str, MetricResult] = {}
        for m in self.metrics:
            score = m.evaluate(trace, scenario)
            results[m.metric_name] = MetricResult(name=m.metric_name, score=score)
        return results
