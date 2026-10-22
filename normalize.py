"""Pure helpers for dates, units, and reading selection used by the rules."""

from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from typing import TypeVar

T = TypeVar("T")


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 date or date-time into an aware UTC datetime.

    Date-only values become midnight UTC. Unparseable or empty values return None
    so callers can treat them as missing data instead of crashing.
    """

    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_date(value: str | None) -> date | None:
    parsed = parse_datetime(value)
    return parsed.date() if parsed else None


def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def to_fahrenheit(value: float, unit: str) -> float:
    return celsius_to_fahrenheit(value) if unit == "C" else value


def days_before(event: date, reference: date) -> int:
    """Whole days from `event` to `reference`; negative if `event` is later."""

    return (reference - event).days


def within_days(event: date, reference: date, max_days: int) -> bool:
    """True if `event` is on or before `reference` and at most `max_days` earlier.

    The boundary is inclusive: exactly `max_days` before still counts.
    """

    return 0 <= days_before(event, reference) <= max_days


def latest(items: Iterable[T], key: Callable[[T], datetime | None]) -> T | None:
    """Return the item with the newest timestamp, ignoring undated items.

    Ties keep the later item in input order, so a recheck listed after the
    original reading at the same timestamp wins.
    """

    best: T | None = None
    best_at: datetime | None = None
    for item in items:
        at = key(item)
        if at is not None and (best_at is None or at >= best_at):
            best, best_at = item, at
    return best
