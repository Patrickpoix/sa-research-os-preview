# 中文阅读入口：这是跨市场研究共享的统计原语 owner，集中实现 return moments、Deflated
# Sharpe、HAC mean、FDR 等可审计计算。调用者必须显式提供 observations、annualization/
# periods-per-year、trial count 等语义；本模块只计算统计量，不自行决定某市场的 PASS threshold。
# NaN/样本不足等边界保持显式结果或错误，避免“默认 0”把不可测量伪装成失败/成功。
"""Auditable statistical measures shared by market research pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew


@dataclass(frozen=True)
class ReturnMoments:
    observations: int
    periods_per_year: float
    annualized_sharpe: float
    skewness: float
    pearson_kurtosis: float


@dataclass(frozen=True)
class DeflatedSharpeResult:
    probability: float
    observed_annualized_sharpe: float
    expected_max_annualized_sharpe: float
    trial_count: int
    observations: int
    periods_per_year: float
    skewness: float
    pearson_kurtosis: float
    trial_sharpe_mean: float | None = None
    trial_sharpe_std: float | None = None
    trial_distribution_source: str = "standard_normal_prior"
    authoritative: bool = False


@dataclass(frozen=True)
class HACMeanResult:
    estimate: float
    standard_error: float
    t_statistic: float
    p_value: float
    lag: int
    observations: int
    long_run_variance: float


@dataclass(frozen=True)
class FDRResult:
    family_id: str
    alpha: float
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    method: str = "benjamini_hochberg"


@dataclass(frozen=True)
class BlockBootstrapResult:
    estimate: float
    confidence_interval: tuple[float, float]
    confidence: float
    block_length: int
    resamples: int
    seed: int
    observations: int


def _finite_vector(values: Sequence[float], *, field: str, minimum: int = 1) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.ndim != 1 or len(vector) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} observations")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{field} must contain only finite values")
    return vector


def hac_mean_inference(values: Sequence[float], *, lag: int) -> HACMeanResult:
    """Estimate a mean with a Bartlett-kernel Newey-West standard error."""
    vector = _finite_vector(values, field="values", minimum=2)
    observations = len(vector)
    if isinstance(lag, bool) or not isinstance(lag, int) or not 0 <= lag < observations:
        raise ValueError("lag must be an integer in [0, observations)")

    estimate = float(np.mean(vector))
    centered = vector - estimate
    long_run_variance = float(np.dot(centered, centered) / observations)
    for offset in range(1, lag + 1):
        autocovariance = float(
            np.dot(centered[offset:], centered[:-offset]) / observations
        )
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * autocovariance
    if long_run_variance < 0 and math.isclose(long_run_variance, 0.0, abs_tol=1e-15):
        long_run_variance = 0.0
    if long_run_variance <= 0 or not math.isfinite(long_run_variance):
        raise ValueError("HAC long-run variance must be finite and positive")
    standard_error = math.sqrt(long_run_variance / observations)
    t_statistic = estimate / standard_error
    return HACMeanResult(
        estimate=estimate,
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value=float(2.0 * norm.sf(abs(t_statistic))),
        lag=lag,
        observations=observations,
        long_run_variance=long_run_variance,
    )


def fdr_benjamini_hochberg(
    p_values: Sequence[float], *, alpha: float, family_id: str
) -> FDRResult:
    """Adjust one explicitly named hypothesis family with Benjamini-Hochberg."""
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError("family_id is required")
    family = family_id.strip()
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and between zero and one")
    values = _finite_vector(p_values, field="p_values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be between zero and one")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    ordered_adjusted = np.minimum.accumulate((ordered * len(values) / ranks)[::-1])[::-1]
    ordered_adjusted = np.minimum(ordered_adjusted, 1.0)
    adjusted = np.empty_like(ordered_adjusted)
    adjusted[order] = ordered_adjusted
    rejected = adjusted <= alpha
    return FDRResult(
        family_id=family,
        alpha=float(alpha),
        adjusted_p_values=tuple(float(value) for value in adjusted),
        rejected=tuple(bool(value) for value in rejected),
    )


def block_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    block_length: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> BlockBootstrapResult:
    """Return a reproducible circular block-bootstrap interval for the mean."""
    vector = _finite_vector(values, field="values", minimum=2)
    observations = len(vector)
    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or not 1 <= block_length <= observations
    ):
        raise ValueError("block_length must be an integer in [1, observations]")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError("resamples must be an integer of at least two")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    generator = np.random.default_rng(seed)
    blocks_per_sample = math.ceil(observations / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        starts = generator.integers(0, observations, size=blocks_per_sample)
        sample_indexes = ((starts[:, None] + offsets) % observations).reshape(-1)
        means[index] = float(np.mean(vector[sample_indexes[:observations]]))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return BlockBootstrapResult(
        estimate=float(np.mean(vector)),
        confidence_interval=(float(lower), float(upper)),
        confidence=float(confidence),
        block_length=block_length,
        resamples=resamples,
        seed=seed,
        observations=observations,
    )


def return_moments_from_nav(
    nav: Sequence[float], dates: Sequence[date]
) -> ReturnMoments:
    """Calculate return moments with annualization derived from elapsed time."""
    values = np.asarray(nav, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError("NAV and date sequences must be one-dimensional and aligned")
    if len(dates) >= 2 and any(
        dates[index] >= dates[index + 1] for index in range(len(dates) - 1)
    ):
        raise ValueError("dates must be strictly increasing")
    if len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        return ReturnMoments(0, float("nan"), float("nan"), float("nan"), float("nan"))

    returns = np.diff(values) / values[:-1]
    observations = int(len(returns))
    elapsed_days = max((dates[-1] - dates[0]).days, 1)
    periods_per_year = observations / (elapsed_days / 365.2425)
    standard_deviation = float(np.std(returns, ddof=0))
    annualized_sharpe = (
        float(np.mean(returns) / standard_deviation * math.sqrt(periods_per_year))
        if standard_deviation > 0
        else float("nan")
    )
    skewness = float(skew(returns, bias=False)) if observations >= 3 else float("nan")
    pearson_kurtosis = (
        float(kurtosis(returns, fisher=False, bias=False))
        if observations >= 4
        else float("nan")
    )
    return ReturnMoments(
        observations=observations,
        periods_per_year=float(periods_per_year),
        annualized_sharpe=annualized_sharpe,
        skewness=skewness,
        pearson_kurtosis=pearson_kurtosis,
    )


def _expected_max_standard_normal(trial_count: int) -> float:
    if trial_count < 1:
        raise ValueError("trial_count must be positive")
    if trial_count == 1:
        return 0.0
    euler_mascheroni = 0.5772156649015329
    first = norm.ppf(1.0 - 1.0 / trial_count)
    second = norm.ppf(1.0 - 1.0 / (trial_count * math.e))
    return float((1.0 - euler_mascheroni) * first + euler_mascheroni * second)


def _expected_max_empirical(values: np.ndarray, draws: int) -> float:
    """Exact expected maximum for iid draws from an empirical distribution."""
    if values.ndim != 1 or len(values) == 0 or draws < 1:
        raise ValueError("empirical trial values and draw count are required")
    ordered = np.sort(values.astype(np.float64, copy=False))
    denominator = float(len(ordered))
    upper = np.power(np.arange(1, len(ordered) + 1, dtype=np.float64) / denominator, draws)
    lower = np.power(np.arange(0, len(ordered), dtype=np.float64) / denominator, draws)
    return float(np.sum(ordered * (upper - lower)))


def deflated_sharpe_probability(
    moments: ReturnMoments,
    *,
    trial_count: int,
    trial_sharpes: Sequence[float] | None = None,
) -> DeflatedSharpeResult:
    """Return the probability that Sharpe exceeds the multiple-test benchmark.

    The calculation follows the Bailey-Lopez de Prado deflated Sharpe form. The
    reported value is a probability, not a Sharpe difference, so a threshold of
    0.5 has an explicit and stable meaning.
    """
    if moments.observations < 2:
        raise ValueError("deflated Sharpe requires at least two return observations")
    values = (
        moments.periods_per_year,
        moments.annualized_sharpe,
        moments.skewness,
        moments.pearson_kurtosis,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("deflated Sharpe requires finite return moments")
    if moments.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if (
        isinstance(trial_count, bool)
        or not isinstance(trial_count, Integral)
        or int(trial_count) < 1
    ):
        raise ValueError("trial_count must be positive")
    trial_count = int(trial_count)
    expected_standard_normal = _expected_max_standard_normal(trial_count)
    if trial_sharpes is None:
        # Backward-compatible prior for legacy callers.  This is useful for
        # diagnostics but is not an authoritative multiple-testing result,
        # because the empirical trial Sharpe distribution is unavailable.
        trial_mean = 0.0
        trial_std = math.sqrt(
            moments.periods_per_year / (moments.observations - 1)
        )
        distribution_source = "standard_normal_prior"
        authoritative = False
        expected_annualized = trial_mean + trial_std * expected_standard_normal
    else:
        trial_values = np.asarray(tuple(trial_sharpes), dtype=np.float64)
        if trial_values.ndim != 1 or len(trial_values) != trial_count:
            raise ValueError("trial_sharpes length must equal trial_count")
        if len(trial_values) == 0 or np.any(~np.isfinite(trial_values)):
            raise ValueError("trial_sharpes must contain finite values")
        trial_mean = float(np.mean(trial_values))
        trial_std = float(np.std(trial_values, ddof=1)) if len(trial_values) > 1 else 0.0
        distribution_source = "empirical"
        authoritative = True
        expected_annualized = _expected_max_empirical(trial_values, trial_count)
    observed_period = moments.annualized_sharpe / math.sqrt(moments.periods_per_year)
    expected_period = expected_annualized / math.sqrt(moments.periods_per_year)
    variance_adjustment = (
        1.0
        - moments.skewness * observed_period
        + ((moments.pearson_kurtosis - 1.0) / 4.0) * observed_period**2
    )
    if variance_adjustment <= 0 or not math.isfinite(variance_adjustment):
        raise ValueError("deflated Sharpe variance adjustment is not positive")
    statistic = (
        (observed_period - expected_period)
        * math.sqrt(moments.observations - 1)
        / math.sqrt(variance_adjustment)
    )
    return DeflatedSharpeResult(
        probability=float(norm.cdf(statistic)),
        observed_annualized_sharpe=moments.annualized_sharpe,
        expected_max_annualized_sharpe=float(expected_annualized),
        trial_count=trial_count,
        observations=moments.observations,
        periods_per_year=moments.periods_per_year,
        skewness=moments.skewness,
        pearson_kurtosis=moments.pearson_kurtosis,
        trial_sharpe_mean=float(trial_mean),
        trial_sharpe_std=float(trial_std),
        trial_distribution_source=distribution_source,
        authoritative=authoritative,
    )
