# 中文阅读入口：这是 研究 的回归测试，验证“cn / 交易日历”相关契约；fixture 与断言只用于证明行为，不是生产逻辑的第二 owner。 主要入口：`test_weekends_are_non_trading`、`test_verified_spring_festival_weekdays_are_closed`、`test_verified_spring_festival_boundary_sessions_are_open`、`test_verified_qingming_weekdays_are_closed`。
"""Boundary tests for the verified CN trading calendar contract."""
from datetime import date

import pytest

from alpha_research_os.datasets.cn.trading_calendar import (
    VERIFIED_END_YEAR,
    VERIFIED_START_YEAR,
    count_trading_days,
    is_trading_day,
    next_trading_day,
)


@pytest.mark.parametrize(
    "value",
    [
        date(2024, 6, 1),
        date(2024, 6, 2),
        date(2025, 2, 8),
        date(2026, 2, 14),
    ],
)
def test_weekends_are_non_trading(value: date) -> None:
    assert is_trading_day(value) is False


@pytest.mark.parametrize(
    "value",
    [
        date(2010, 2, 15),
        date(2017, 1, 27),
        date(2022, 1, 31),
        date(2024, 2, 9),
        date(2024, 2, 16),
        date(2025, 1, 28),
        date(2025, 2, 4),
        date(2026, 2, 16),
        date(2026, 2, 23),
    ],
)
def test_verified_spring_festival_weekdays_are_closed(value: date) -> None:
    assert is_trading_day(value) is False


@pytest.mark.parametrize(
    "value",
    [
        date(2010, 2, 12),
        date(2017, 1, 26),
        date(2022, 1, 28),
        date(2024, 2, 7),
        date(2024, 2, 8),
        date(2025, 2, 5),
        date(2026, 2, 24),
    ],
)
def test_verified_spring_festival_boundary_sessions_are_open(value: date) -> None:
    assert is_trading_day(value) is True


@pytest.mark.parametrize(
    "value",
    [
        date(2013, 4, 4),
        date(2013, 4, 5),
        date(2023, 4, 5),
        date(2024, 4, 4),
        date(2024, 4, 5),
        date(2025, 4, 4),
        date(2026, 4, 6),
    ],
)
def test_verified_qingming_weekdays_are_closed(value: date) -> None:
    assert is_trading_day(value) is False


def test_qingming_is_not_approximated_as_fixed_april_4_and_5() -> None:
    assert is_trading_day(date(2023, 4, 4)) is True
    assert is_trading_day(date(2024, 4, 8)) is True


@pytest.mark.parametrize(
    "value",
    [
        date(2024, 10, 1),
        date(2024, 10, 7),
        date(2025, 10, 1),
        date(2025, 10, 8),
        date(2026, 10, 1),
        date(2026, 10, 7),
    ],
)
def test_national_day_weekdays_are_closed(value: date) -> None:
    assert is_trading_day(value) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2024, 6, 3), date(2024, 6, 4)),
        (date(2024, 5, 31), date(2024, 6, 3)),
        (date(2024, 2, 8), date(2024, 2, 19)),
        (date(2024, 9, 30), date(2024, 10, 8)),
        (date(2024, 4, 3), date(2024, 4, 8)),
        (date(2024, 12, 31), date(2025, 1, 2)),
        (date(2025, 1, 27), date(2025, 2, 5)),
        (date(2026, 2, 13), date(2026, 2, 24)),
    ],
)
def test_next_trading_day_uses_verified_sessions(value: date, expected: date) -> None:
    assert next_trading_day(value) == expected


def test_count_trading_days_uses_verified_holidays() -> None:
    assert count_trading_days(date(2024, 6, 3), date(2024, 6, 7)) == 5
    assert count_trading_days(date(2024, 6, 3), date(2024, 6, 14)) == 9
    assert count_trading_days(date(2024, 2, 5), date(2024, 2, 23)) == 9


def test_count_trading_days_rejects_reverse_interval() -> None:
    with pytest.raises(ValueError, match="end must not precede start"):
        count_trading_days(date(2024, 1, 3), date(2024, 1, 2))


@pytest.mark.parametrize(
    "value",
    [
        date(VERIFIED_START_YEAR - 1, 12, 31),
        date(VERIFIED_END_YEAR + 1, 1, 1),
    ],
)
def test_dates_outside_verified_coverage_fail_closed(value: date) -> None:
    with pytest.raises(ValueError, match="outside verified coverage"):
        is_trading_day(value)


def test_next_session_crossing_beyond_verified_coverage_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside verified coverage"):
        next_trading_day(date(VERIFIED_END_YEAR, 12, 31))
