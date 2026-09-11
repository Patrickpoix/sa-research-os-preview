# 中文阅读入口：这是 研究 的回归测试，验证“研究 / 统计”相关契约；fixture 与断言只用于证明行为，不是生产逻辑的第二 owner。 主要入口：`test_deflated_sharpe_is_a_probability_and_penalizes_more_trials`、`test_deflated_sharpe_rejects_insufficient_observations`、`test_return_moments_record_calendar_annualization`、`test_return_moments_reject_non_monotonic_dates`。
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from alpha_research_os.research.statistics import (
    ReturnMoments,
    block_bootstrap_mean_ci,
    deflated_sharpe_probability,
    fdr_benjamini_hochberg,
    hac_mean_inference,
    return_moments_from_nav,
)


def _nav_from_returns(returns: np.ndarray) -> np.ndarray:
    return np.concatenate(([100.0], 100.0 * np.cumprod(1.0 + returns)))


def test_deflated_sharpe_is_a_probability_and_penalizes_more_trials() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.015, 0.04, size=180)
    dates = tuple(date(2010 + index // 12, index % 12 + 1, 1) for index in range(181))
    moments = return_moments_from_nav(_nav_from_returns(returns), dates)

    few_trials = deflated_sharpe_probability(moments, trial_count=5)
    many_trials = deflated_sharpe_probability(moments, trial_count=500)

    assert 0.0 <= many_trials.probability <= 1.0
    assert many_trials.probability < few_trials.probability
    assert many_trials.expected_max_annualized_sharpe > 0


def test_deflated_sharpe_rejects_insufficient_observations() -> None:
    moments = return_moments_from_nav(
        np.asarray([100.0, 101.0]),
        (date(2024, 1, 1), date(2024, 2, 1)),
    )

    with pytest.raises(ValueError, match="at least two return observations"):
        deflated_sharpe_probability(moments, trial_count=10)


@pytest.mark.parametrize("trial_count", [True, 1.5, 0])
def test_deflated_sharpe_requires_a_positive_integer_trial_count(
    trial_count: object,
) -> None:
    moments = ReturnMoments(
        observations=10,
        periods_per_year=252.0,
        annualized_sharpe=1.0,
        skewness=0.0,
        pearson_kurtosis=3.0,
    )

    with pytest.raises(ValueError, match="trial_count must be positive"):
        deflated_sharpe_probability(moments, trial_count=trial_count)  # type: ignore[arg-type]


def test_return_moments_record_calendar_annualization() -> None:
    returns = np.asarray([0.01, -0.005, 0.02, 0.0] * 6)
    dates = tuple(date(2022 + index // 12, index % 12 + 1, 1) for index in range(25))

    moments = return_moments_from_nav(_nav_from_returns(returns), dates)

    assert moments.observations == 24
    assert moments.periods_per_year == pytest.approx(12.0, rel=0.05)
    assert np.isfinite(moments.annualized_sharpe)
    assert np.isfinite(moments.skewness)
    assert np.isfinite(moments.pearson_kurtosis)


def test_return_moments_reject_non_monotonic_dates() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        return_moments_from_nav(
            np.asarray([100.0, 101.0, 102.0]),
            (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 3)),
        )


def test_deflated_sharpe_uses_empirical_trial_sharpe_dispersion() -> None:
    moments = ReturnMoments(
        observations=252,
        periods_per_year=252.0,
        annualized_sharpe=1.5,
        skewness=0.0,
        pearson_kurtosis=3.0,
    )

    concentrated = deflated_sharpe_probability(
        moments,
        trial_count=4,
        trial_sharpes=(0.1, 0.1, 0.1, 0.1),
    )
    dispersed = deflated_sharpe_probability(
        moments,
        trial_count=4,
        trial_sharpes=(-2.0, -1.0, 0.0, 5.0),
    )

    assert dispersed.expected_max_annualized_sharpe > concentrated.expected_max_annualized_sharpe
    assert dispersed.probability < concentrated.probability
    assert dispersed.trial_distribution_source == "empirical"


def test_hac_mean_uses_declared_lag_and_detects_serial_dependence() -> None:
    values = np.repeat(np.linspace(-0.02, 0.04, 20), 3)

    independent = hac_mean_inference(values, lag=0)
    serially_robust = hac_mean_inference(values, lag=2)

    assert serially_robust.estimate == pytest.approx(float(np.mean(values)))
    assert serially_robust.standard_error > independent.standard_error
    assert serially_robust.lag == 2
    assert 0.0 <= serially_robust.p_value <= 1.0


def test_hac_mean_rejects_nonfinite_or_out_of_range_lag() -> None:
    with pytest.raises(ValueError, match="finite"):
        hac_mean_inference((0.01, float("nan")), lag=0)
    with pytest.raises(ValueError, match="lag"):
        hac_mean_inference((0.01, 0.02), lag=2)


def test_fdr_adjusts_one_named_family_in_original_order() -> None:
    result = fdr_benjamini_hochberg(
        (0.01, 0.04, 0.03, 0.20), alpha=0.05, family_id="cn_ci11_candidates"
    )

    assert result.family_id == "cn_ci11_candidates"
    assert result.adjusted_p_values == pytest.approx(
        (0.04, 0.0533333333, 0.0533333333, 0.20)
    )
    assert result.rejected == (True, False, False, False)


def test_fdr_rejects_unnamed_or_invalid_families() -> None:
    with pytest.raises(ValueError, match="family_id"):
        fdr_benjamini_hochberg((0.01,), alpha=0.05, family_id="")
    with pytest.raises(ValueError, match="family_id"):
        fdr_benjamini_hochberg((0.01,), alpha=0.05, family_id=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between zero and one"):
        fdr_benjamini_hochberg((1.1,), alpha=0.05, family_id="family")


def test_block_bootstrap_is_reproducible_and_records_resampling_contract() -> None:
    values = np.asarray([0.01, 0.02, -0.01, 0.03, 0.00] * 8)

    first = block_bootstrap_mean_ci(
        values, block_length=5, resamples=200, confidence=0.95, seed=17
    )
    second = block_bootstrap_mean_ci(
        values, block_length=5, resamples=200, confidence=0.95, seed=17
    )

    assert first == second
    assert first.confidence_interval[0] <= first.estimate <= first.confidence_interval[1]
    assert first.block_length == 5
    assert first.resamples == 200
    assert first.seed == 17


def test_block_bootstrap_rejects_invalid_block_or_missing_seed_contract() -> None:
    with pytest.raises(ValueError, match="block_length"):
        block_bootstrap_mean_ci(
            (0.01, 0.02), block_length=3, resamples=10, confidence=0.95, seed=1
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        block_bootstrap_mean_ci(
            (0.01, 0.02), block_length=1, resamples=10, confidence=0.95, seed=-1
        )
