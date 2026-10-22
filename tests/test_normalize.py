from datetime import UTC, date, datetime

import pytest

from normalize import (
    celsius_to_fahrenheit,
    latest,
    parse_date,
    parse_datetime,
    to_fahrenheit,
    within_days,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-02-24T10:12:00Z", datetime(2026, 2, 24, 10, 12, tzinfo=UTC)),
        ("2026-02-24T05:12:00-05:00", datetime(2026, 2, 24, 10, 12, tzinfo=UTC)),
        ("2026-02-24", datetime(2026, 2, 24, tzinfo=UTC)),
        (None, None),
        ("", None),
        ("pending OR availability", None),
    ],
)
def test_parse_datetime(value: str | None, expected: datetime | None) -> None:
    assert parse_datetime(value) == expected


def test_parse_date_drops_time() -> None:
    assert parse_date("2026-02-24T23:59:00Z") == date(2026, 2, 24)


def test_temperature_conversion() -> None:
    assert celsius_to_fahrenheit(38.0) == pytest.approx(100.4)
    assert to_fahrenheit(37.9, "C") == pytest.approx(100.22)
    assert to_fahrenheit(99.1, "F") == 99.1


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (date(2026, 2, 11), True),   # exactly 30 days: inclusive
        (date(2026, 2, 10), False),  # 31 days
        (date(2026, 3, 13), True),   # same day
        (date(2026, 3, 14), False),  # after the procedure
    ],
)
def test_within_days_is_inclusive(event: date, expected: bool) -> None:
    assert within_days(event, date(2026, 3, 13), 30) is expected


def test_latest_ignores_undated_and_prefers_later_on_tie() -> None:
    at = datetime(2026, 3, 1, 8, tzinfo=UTC)
    items = [("old", datetime(2026, 2, 1, tzinfo=UTC)), ("first", at), ("undated", None), ("recheck", at)]

    assert latest(items, key=lambda x: x[1]) == ("recheck", at)
    assert latest([("undated", None)], key=lambda x: x[1]) is None
