from datetime import date, timedelta

import pytest

from disputedesk.bizdays import add_business_days, business_days_between, holidays, is_business_day


def test_weekend_is_not_a_business_day():
    assert not is_business_day(date(2026, 11, 7))  # Saturday
    assert not is_business_day(date(2026, 11, 8))
    assert is_business_day(date(2026, 11, 6))


def test_add_skips_weekend():
    assert add_business_days(date(2026, 11, 6), 1) == date(2026, 11, 9)  # Friday + 1 = Monday


def test_add_skips_thanksgiving():
    assert date(2026, 11, 26) in holidays(2026)
    assert add_business_days(date(2026, 11, 25), 1) == date(2026, 11, 27)


def test_saturday_holiday_observed_on_friday():
    assert date(2026, 7, 3) in holidays(2026)  # July 4 2026 is a Saturday
    assert not is_business_day(date(2026, 7, 3))


def test_sunday_holiday_observed_on_monday():
    assert date(2027, 7, 5) in holidays(2027)  # July 4 2027 is a Sunday


def test_floating_holidays_2026():
    h = holidays(2026)
    assert date(2026, 1, 19) in h  # MLK
    assert date(2026, 5, 25) in h  # Memorial Day, last Monday
    assert date(2026, 9, 7) in h  # Labor Day


def test_zero_days_is_start_and_negative_raises():
    assert add_business_days(date(2026, 11, 7), 0) == date(2026, 11, 7)
    with pytest.raises(ValueError):
        add_business_days(date(2026, 11, 7), -1)


def test_twenty_business_day_deadline_over_thanksgiving():
    # 20 business days from Nov 2 2026 skips 4 weekends, Veterans Day, and Thanksgiving.
    assert add_business_days(date(2026, 11, 2), 20) == date(2026, 12, 2)


def test_between_is_antisymmetric():
    a, b = date(2026, 3, 2), date(2026, 3, 16)
    assert business_days_between(a, b) == 10
    assert business_days_between(b, a) == -10
    assert business_days_between(a, a) == 0


@pytest.mark.parametrize("n", [1, 5, 10, 20, 30])
def test_add_and_between_round_trip(n):
    d = date(2025, 11, 3)
    for _ in range(60):
        assert business_days_between(d, add_business_days(d, n)) == n
        d += timedelta(days=5)
