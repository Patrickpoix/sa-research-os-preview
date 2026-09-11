"""CN 已验证交易日历 owner（2010–2026）。

中文阅读入口：模块加载时先校验每个覆盖年份和工作日 closure 表，再由
``_require_verified_date`` 拦住范围外输入；公开函数分别完成“是否交易日、下一个交易日、
区间交易日计数”。周末固定不交易，工作日节假日来自已核验的发布/价格证据；范围外必须
fail-closed，不能用普通 weekday 近似。本模块是纯日期规则，不负责 price-limit/execution policy，
也不产生 provider、数据库或文件副作用。
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


VERIFIED_START_YEAR = 2010
VERIFIED_END_YEAR = 2026

# A 股收盘时刻；labels 的 label_available_at 和 factors 的 factor available_at
# 共用同一个 observation-date 收盘口径，不能各自定义不同的时刻。
CN_MARKET_CLOSE_SUFFIX = "T15:00:00+08:00"

# Only weekday closures are listed.  Saturdays and Sundays are handled by
# ``is_trading_day`` and therefore do not need to be duplicated here.
_WEEKDAY_CLOSURES_BY_YEAR: dict[int, str] = {
    2010: """
        01-01 02-15 02-16 02-17 02-18 02-19 04-05 05-03 06-14 06-15
        06-16 09-22 09-23 09-24 10-01 10-04 10-05 10-06 10-07
    """,
    2011: """
        01-03 02-02 02-03 02-04 02-07 02-08 04-04 04-05 05-02 06-06
        09-12 10-03 10-04 10-05 10-06 10-07
    """,
    2012: """
        01-02 01-03 01-23 01-24 01-25 01-26 01-27 04-02 04-03 04-04
        04-30 05-01 06-22 10-01 10-02 10-03 10-04 10-05
    """,
    2013: """
        01-01 01-02 01-03 02-11 02-12 02-13 02-14 02-15 04-04 04-05
        04-29 04-30 05-01 06-10 06-11 06-12 09-19 09-20 10-01 10-02
        10-03 10-04 10-07
    """,
    2014: """
        01-01 01-31 02-03 02-04 02-05 02-06 04-07 05-01 05-02 06-02
        09-08 10-01 10-02 10-03 10-06 10-07
    """,
    2015: """
        01-01 01-02 02-18 02-19 02-20 02-23 02-24 04-06 05-01 06-22
        09-03 09-04 10-01 10-02 10-05 10-06 10-07
    """,
    2016: """
        01-01 02-08 02-09 02-10 02-11 02-12 04-04 05-02 06-09 06-10
        09-15 09-16 10-03 10-04 10-05 10-06 10-07
    """,
    2017: """
        01-02 01-27 01-30 01-31 02-01 02-02 04-03 04-04 05-01 05-29
        05-30 10-02 10-03 10-04 10-05 10-06
    """,
    2018: """
        01-01 02-15 02-16 02-19 02-20 02-21 04-05 04-06 04-30 05-01
        06-18 09-24 10-01 10-02 10-03 10-04 10-05 12-31
    """,
    2019: """
        01-01 02-04 02-05 02-06 02-07 02-08 04-05 05-01 05-02 05-03
        06-07 09-13 10-01 10-02 10-03 10-04 10-07
    """,
    2020: """
        01-01 01-24 01-27 01-28 01-29 01-30 01-31 04-06 05-01 05-04
        05-05 06-25 06-26 10-01 10-02 10-05 10-06 10-07 10-08
    """,
    2021: """
        01-01 02-11 02-12 02-15 02-16 02-17 04-05 05-03 05-04 05-05
        06-14 09-20 09-21 10-01 10-04 10-05 10-06 10-07
    """,
    2022: """
        01-03 01-31 02-01 02-02 02-03 02-04 04-04 04-05 05-02 05-03
        05-04 06-03 09-12 10-03 10-04 10-05 10-06 10-07
    """,
    2023: """
        01-02 01-23 01-24 01-25 01-26 01-27 04-05 05-01 05-02 05-03
        06-22 06-23 09-29 10-02 10-03 10-04 10-05 10-06
    """,
    2024: """
        01-01 02-09 02-12 02-13 02-14 02-15 02-16 04-04 04-05 05-01
        05-02 05-03 06-10 09-16 09-17 10-01 10-02 10-03 10-04 10-07
    """,
    2025: """
        01-01 01-28 01-29 01-30 01-31 02-03 02-04 04-04 05-01 05-02
        05-05 06-02 10-01 10-02 10-03 10-06 10-07 10-08
    """,
    2026: """
        01-01 01-02 02-16 02-17 02-18 02-19 02-20 02-23 04-06 05-01
        05-04 05-05 06-19 09-25 10-01 10-02 10-05 10-06 10-07
    """,
}


def _build_closure_set() -> frozenset[date]:
    closures: set[date] = set()
    expected_years = set(range(VERIFIED_START_YEAR, VERIFIED_END_YEAR + 1))
    # 先确认年度覆盖完整，再解析日期并拒绝把周末重复写进工作日 closure 表。
    if set(_WEEKDAY_CLOSURES_BY_YEAR) != expected_years:
        raise RuntimeError("CN trading calendar year coverage is incomplete")
    for year, month_days in _WEEKDAY_CLOSURES_BY_YEAR.items():
        for month_day in month_days.split():
            value = date.fromisoformat(f"{year}-{month_day}")
            if value.weekday() >= 5:
                raise RuntimeError(f"weekend duplicated in CN weekday closures: {value}")
            closures.add(value)
    return frozenset(closures)


CN_HOLIDAYS = _build_closure_set()


def _require_verified_date(value: date) -> None:
    """在所有查询前执行类型与证据覆盖范围的 fail-closed 校验。"""
    if not isinstance(value, date):
        raise TypeError(f"expected datetime.date, got {type(value).__name__}")
    if not VERIFIED_START_YEAR <= value.year <= VERIFIED_END_YEAR:
        raise ValueError(
            "CN trading calendar date is outside verified coverage: "
            f"{value} not in {VERIFIED_START_YEAR}-{VERIFIED_END_YEAR}"
        )


def is_trading_day(value: date) -> bool:
    """判断日期是否属于已验证的 A 股交易 session。"""
    _require_verified_date(value)
    return value.weekday() < 5 and value not in CN_HOLIDAYS


def next_trading_day(value: date) -> date:
    """返回输入日期之后的下一个已验证 A 股交易 session。"""
    _require_verified_date(value)
    cursor = value + timedelta(days=1)
    while True:
        # 每次递增都重新校验范围，避免在验证窗口末端静默越界或退化成 weekday 近似。
        _require_verified_date(cursor)
        if is_trading_day(cursor):
            return cursor
        cursor += timedelta(days=1)


def count_trading_days(start: date, end: date) -> int:
    """统计闭区间内的已验证 A 股交易 session 数量。"""
    _require_verified_date(start)
    _require_verified_date(end)
    if end < start:
        raise ValueError("end must not precede start")
    count = 0
    cursor = start
    while cursor <= end:
        # 这是 inclusive interval；先判断当天，再前进到下一自然日。
        if is_trading_day(cursor):
            count += 1
        cursor += timedelta(days=1)
    return count


def market_close_available_at(dates: pd.Series) -> pd.Series:
    """将日期列映射到 observation-date 收盘时刻，作为 PIT available_at。"""
    normalized = pd.to_datetime(dates, errors="raise")
    formatted = normalized.dt.strftime(f"%Y-%m-%d{CN_MARKET_CLOSE_SUFFIX}")
    return formatted.where(normalized.notna(), None)
