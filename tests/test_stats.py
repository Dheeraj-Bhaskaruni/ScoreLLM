"""Tests for evalflow.stats — Welch t-test, bootstrap, and A/B comparison."""
import pytest
from evalflow.stats import (
    StatTestResult,
    bootstrap_test,
    compare_ab_scores,
    format_stat_table,
    welch_t_test,
)


class TestWelchTTest:
    def test_identical_distributions_not_significant(self):
        a = [3.0, 3.0, 3.0, 3.0, 3.0]
        b = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = welch_t_test(a, b, "test_metric")
        assert not result.significant
        assert result.delta == 0.0
        assert result.verdict == "NO_DIFFERENCE"

    def test_clearly_different_distributions(self):
        a = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
        b = [5.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 4.0]
        result = welch_t_test(a, b, "helpfulness")
        assert result.significant
        assert result.delta > 0
        assert result.verdict == "B_BETTER"
        assert result.p_value < 0.01

    def test_b_worse_than_a(self):
        a = [5.0, 4.0, 5.0, 5.0, 4.0, 5.0]
        b = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
        result = welch_t_test(a, b, "safety")
        assert result.significant
        assert result.delta < 0
        assert result.verdict == "A_BETTER"

    def test_effect_size_computed(self):
        a = [1.5, 2.0, 2.5, 1.0, 2.0, 2.5, 1.5, 2.0]
        b = [4.0, 4.5, 3.5, 4.0, 5.0, 4.5, 4.0, 3.5]
        result = welch_t_test(a, b, "metric")
        assert result.effect_size > 0
        assert abs(result.effect_size) > 0.8  # large effect

    def test_confidence_interval_covers_delta(self):
        a = [2.0, 3.0, 2.5, 3.0, 2.0, 2.5, 3.0, 2.0]
        b = [4.0, 3.5, 4.0, 4.5, 3.5, 4.0, 4.0, 3.5]
        result = welch_t_test(a, b, "metric")
        assert result.ci_lower <= result.delta <= result.ci_upper

    def test_small_sample_uses_bootstrap(self):
        a = [1.0, 2.0]
        b = [4.0, 5.0]
        result = welch_t_test(a, b, "metric")
        assert result.method == "bootstrap"


class TestBootstrapTest:
    def test_returns_stat_result(self):
        a = [1.0, 2.0, 3.0, 2.0, 1.0]
        b = [4.0, 5.0, 4.0, 5.0, 4.0]
        result = bootstrap_test(a, b, "metric")
        assert isinstance(result, StatTestResult)
        assert result.method == "bootstrap"

    def test_empty_input(self):
        result = bootstrap_test([], [1.0, 2.0], "metric")
        assert result.p_value == 1.0
        assert not result.significant


class TestCompareABScores:
    def test_multi_metric_comparison(self):
        scores_a = {"helpfulness": [2.0, 3.0, 2.0], "safety": [5.0, 5.0, 5.0]}
        scores_b = {"helpfulness": [4.0, 5.0, 4.0], "safety": [5.0, 5.0, 5.0]}
        results = compare_ab_scores(scores_a, scores_b)
        assert "helpfulness" in results
        assert "safety" in results
        assert results["helpfulness"].delta > 0

    def test_missing_metric_handled(self):
        scores_a = {"helpfulness": [2.0, 3.0]}
        scores_b = {"helpfulness": [4.0, 5.0], "new_metric": [3.0, 4.0]}
        results = compare_ab_scores(scores_a, scores_b)
        assert "new_metric" in results


class TestFormatStatTable:
    def test_produces_readable_output(self):
        scores_a = {"helpfulness": [2.0, 3.0, 2.0, 3.0]}
        scores_b = {"helpfulness": [4.0, 5.0, 4.0, 5.0]}
        results = compare_ab_scores(scores_a, scores_b)
        table = format_stat_table(results)
        assert "helpfulness" in table
        assert "p-value" in table
        assert "Effect" in table
