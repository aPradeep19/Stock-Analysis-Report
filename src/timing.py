"""
When a piece of news (or a rating change) came out relative to the last two closes.

Shared by the "why it's moving" explainer, which only lets fresh items explain a move, and the
FinBERT sentiment score, which weights fresh items most.
"""

from datetime import datetime
from datetime import time as clock_time
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_CLOSE = clock_time(16, 0)

SINCE_CLOSE = "since last close"
LAST_SESSION = "during last session"
OLDER = "older"
UNKNOWN = "unknown"


def close_times(candles):
    """(last close, the close before it) as New York datetimes, or (None, None) without history."""
    if candles is None or len(candles) < 2:
        return None, None
    return tuple(datetime.combine(candles.index[i].date(), REGULAR_CLOSE, MARKET_TZ) for i in (-1, -2))


def timing(unix_seconds, last_close, prev_close):
    """SINCE_CLOSE, LAST_SESSION or OLDER for a Unix timestamp; UNKNOWN without a time or history."""
    if not unix_seconds or last_close is None:
        return UNKNOWN
    when = datetime.fromtimestamp(unix_seconds, MARKET_TZ)
    if when > last_close:
        return SINCE_CLOSE
    if when > prev_close:
        return LAST_SESSION
    return OLDER
