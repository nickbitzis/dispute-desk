"""Business-day math for network deadlines.

Network deadlines count business days: weekends and bank holidays do not count. The holiday list is the US
Federal Reserve calendar computed by rule, so it works for any year. When a fixed-date holiday falls on a weekend
it is observed on the nearest weekday (Saturday to Friday, Sunday to Monday), as the Fed does for Sunday.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 6:
        return d + timedelta(days=1)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    fixed = [date(year, 1, 1), date(year, 6, 19), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)]
    floating = [
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents Day
        _last_weekday(year, 5, 0),  # Memorial Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 10, 0, 2),  # Columbus Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
    ]
    return frozenset([_observed(d) for d in fixed] + floating)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def add_business_days(start: date, n: int) -> date:
    """The date n business days after start. Day 0 is start itself, even if it is not a business day."""
    if n < 0:
        raise ValueError("n must be >= 0")
    d = start
    left = n
    while left > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            left -= 1
    return d


def business_days_between(start: date, end: date) -> int:
    """Business days in (start, end]. Negative when end is before start."""
    if end < start:
        return -business_days_between(end, start)
    count = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return count
