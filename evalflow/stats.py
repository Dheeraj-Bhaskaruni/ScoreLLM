"""
evalflow.stats — Statistical significance testing for A/B model comparisons.

Provides confidence intervals, p-values, and effect sizes to determine whether
observed metric differences are statistically meaningful or just noise.
Uses scipy when available, falls back to bootstrap estimation otherwise.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class StatTestResult:
    """Result of a statistical significance test between two groups."""
    metric_name: str
    mean_a: float
    mean_b: float
    delta: float
    std_a: float
    std_b: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float  # Cohen's d
    n_a: int
    n_b: int
    significant: bool  # p < alpha
    method: str  # "welch_t" or "bootstrap"

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "NO_DIFFERENCE"
        return "B_BETTER" if self.delta > 0 else "A_BETTER"


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cohens_d(mean_a: float, mean_b: float, std_a: float, std_b: float, n_a: int, n_b: int) -> float:
    """Compute Cohen's d effect size with pooled standard deviation."""
    if n_a + n_b < 4:
        return 0.0
    pooled = math.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0
    return (mean_b - mean_a) / pooled


def welch_t_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "metric",
    alpha: float = 0.05,
) -> StatTestResult:
    """
    Welch's t-test (unequal variances) — the standard for comparing two sample means.
    Requires scipy for exact p-values; falls back to bootstrap if unavailable.
    """
    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = _mean(scores_a), _mean(scores_b)
    std_a, std_b = _std(scores_a), _std(scores_b)
    delta = mean_b - mean_a
    effect = _cohens_d(mean_a, mean_b, std_a, std_b, n_a, n_b)

    if not HAS_SCIPY or n_a < 3 or n_b < 3:
        return bootstrap_test(scores_a, scores_b, metric_name, alpha)

    t_stat, p_value = scipy_stats.ttest_ind(scores_a, scores_b, equal_var=False)

    # Confidence interval for the difference in means
    se = math.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b) if n_a > 0 and n_b > 0 else 0
    # Welch-Satterthwaite degrees of freedom
    if se > 0:
        df_num = (std_a ** 2 / n_a + std_b ** 2 / n_b) ** 2
        df_den = (std_a ** 2 / n_a) ** 2 / max(n_a - 1, 1) + (std_b ** 2 / n_b) ** 2 / max(n_b - 1, 1)
        df = df_num / df_den if df_den > 0 else 1
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
        ci_lower = delta - t_crit * se
        ci_upper = delta + t_crit * se
    else:
        ci_lower = ci_upper = delta

    return StatTestResult(
        metric_name=metric_name,
        mean_a=round(mean_a, 4),
        mean_b=round(mean_b, 4),
        delta=round(delta, 4),
        std_a=round(std_a, 4),
        std_b=round(std_b, 4),
        p_value=round(float(p_value), 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        effect_size=round(effect, 4),
        n_a=n_a,
        n_b=n_b,
        significant=float(p_value) < alpha,
        method="welch_t",
    )


def bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "metric",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> StatTestResult:
    """
    Non-parametric bootstrap test — works with any sample size and distribution.
    Estimates the sampling distribution of the mean difference via resampling.
    """
    rng = random.Random(seed)
    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = _mean(scores_a), _mean(scores_b)
    std_a, std_b = _std(scores_a), _std(scores_b)
    observed_delta = mean_b - mean_a
    effect = _cohens_d(mean_a, mean_b, std_a, std_b, n_a, n_b)

    if n_a == 0 or n_b == 0:
        return StatTestResult(
            metric_name=metric_name, mean_a=mean_a, mean_b=mean_b,
            delta=observed_delta, std_a=std_a, std_b=std_b,
            p_value=1.0, ci_lower=0.0, ci_upper=0.0, effect_size=0.0,
            n_a=n_a, n_b=n_b, significant=False, method="bootstrap",
        )

    # Bootstrap: resample with replacement and compute mean differences
    boot_deltas = []
    for _ in range(n_bootstrap):
        sample_a = [rng.choice(scores_a) for _ in range(n_a)]
        sample_b = [rng.choice(scores_b) for _ in range(n_b)]
        boot_deltas.append(_mean(sample_b) - _mean(sample_a))

    boot_deltas.sort()

    # p-value: proportion of bootstrap samples where delta <= 0 (one-sided)
    # Convert to two-sided by doubling
    count_le_zero = sum(1 for d in boot_deltas if d <= 0)
    p_one_sided = count_le_zero / n_bootstrap
    p_value = min(2 * min(p_one_sided, 1 - p_one_sided), 1.0)

    # Confidence interval (percentile method)
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    ci_lower = boot_deltas[lower_idx]
    ci_upper = boot_deltas[min(upper_idx, n_bootstrap - 1)]

    return StatTestResult(
        metric_name=metric_name,
        mean_a=round(mean_a, 4),
        mean_b=round(mean_b, 4),
        delta=round(observed_delta, 4),
        std_a=round(std_a, 4),
        std_b=round(std_b, 4),
        p_value=round(p_value, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        effect_size=round(effect, 4),
        n_a=n_a,
        n_b=n_b,
        significant=p_value < alpha,
        method="bootstrap",
    )


def compare_ab_scores(
    scores_a: Dict[str, List[float]],
    scores_b: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, StatTestResult]:
    """
    Run statistical tests across all metrics for an A/B comparison.

    Args:
        scores_a: {metric_name: [per-scenario scores]} for model A
        scores_b: {metric_name: [per-scenario scores]} for model B
        alpha: significance level (default 0.05)

    Returns:
        {metric_name: StatTestResult}
    """
    results = {}
    all_metrics = set(scores_a.keys()) | set(scores_b.keys())

    for metric in sorted(all_metrics):
        a = scores_a.get(metric, [])
        b = scores_b.get(metric, [])
        results[metric] = welch_t_test(a, b, metric_name=metric, alpha=alpha)

    return results


def format_stat_table(stat_results: Dict[str, StatTestResult]) -> str:
    """Format statistical results as a readable ASCII table."""
    header = (
        f"{'Metric':<22} {'Mean A':>8} {'Mean B':>8} {'Delta':>8} "
        f"{'95% CI':>16} {'p-value':>9} {'Effect':>8} {'Sig?':>6}"
    )
    lines = [header, "-" * len(header)]

    for name, r in sorted(stat_results.items()):
        ci_str = f"[{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]"
        sig_str = "YES" if r.significant else "no"
        effect_str = _effect_label(r.effect_size)
        lines.append(
            f"  {name:<20} {r.mean_a:>8.3f} {r.mean_b:>8.3f} {r.delta:>+8.3f} "
            f"{ci_str:>16} {r.p_value:>9.4f} {effect_str:>8} {sig_str:>6}"
        )

    return "\n".join(lines)


def _effect_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "trivial"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"

__all__ = ["welch_t_test", "bootstrap_test", "compare_ab_scores", "format_stat_table", "StatTestResult"]
