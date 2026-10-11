"""
SYNTHETIC test data. None of these prices, headlines or dates are real.

Builds inputs shaped exactly like what main.py collects from Finnhub and yfinance, so the
analyzer, formatter, PDF and dispatcher can be tested without network access or API keys.
"""
from datetime import datetime

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"


def make_candles(start_price, daily_vol, sessions=276, seed=0, last_return=None,
                 last_volume_mult=1.0, end="2026-09-30", drift=0.0005):
    """A random-walk OHLCV DataFrame indexed by trading day, like FinnhubFetcher.get_historical_candles."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=sessions)
    returns = rng.normal(drift, daily_vol, sessions)
    if last_return is not None:
        returns[-1] = last_return
    close = start_price * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0, daily_vol / 2, sessions))
    volume = rng.integers(5_000_000, 9_000_000, sessions).astype(float)
    volume[-1] *= last_volume_mult
    return pd.DataFrame({
        "open": close * (1 - spread / 2),
        "high": close * (1 + spread),
        "low": close * (1 - spread),
        "close": close,
        "volume": volume,
    }, index=index)


def make_snapshot(candles, gap_pct, as_of="2026-10-01 07:52"):
    return {
        "price": float(candles["close"].iloc[-1] * (1 + gap_pct / 100)),
        "as_of": pd.Timestamp(as_of, tz=MARKET_TZ).to_pydatetime(),
        "session": "pre-market",
        "premarket_volume": 250_000,
        "traded_today": True,
    }


def sample_pipeline():
    """Eight tickers with a mix of calm, big-gap, high-volume and breakdown setups."""
    specs = {
        # symbol: (start, daily_vol, seed, last_return, vol_mult, gap_pct, name, industry, mcap_m)
        "AAA": (120, 0.020, 1, -0.071, 3.2, -4.6, "Alpha Semiconductor (synthetic)", "Semiconductors", 210_000),
        "BBB": (45, 0.015, 2, 0.012, 1.1, 3.4, "Beta Networks (synthetic)", "Communications", 38_000),
        "CCC": (300, 0.012, 3, 0.004, 0.9, 0.3, "Gamma Cloud (synthetic)", "Technology", 1_250_000),
        "DDD": (8, 0.060, 4, 0.05, 1.3, 4.1, "Delta Biotech (synthetic)", "Biotechnology", 900),
        "EEE": (60, 0.025, 5, -0.01, 1.0, -0.8, "Epsilon Media (synthetic)", "Media", 22_000),
        "FFF": (90, 0.018, 6, 0.002, 0.8, 0.0, "Zeta Components (synthetic)", "Electrical Equipment", 95_000),
    }
    pipeline = {}
    for i, (symbol, (start, vol, seed, last_ret, vmult, gap, name, industry, mcap)) in enumerate(specs.items()):
        candles = make_candles(start, vol, seed=seed, last_return=last_ret, last_volume_mult=vmult)
        pipeline[symbol] = {
            "quote": {"current_price": float(candles["close"].iloc[-1]), "change_pct": last_ret * 100},
            "candles": candles,
            "news": [
                {"headline": f"{symbol} shares plunge after guidance miss & analyst downgrade" if gap < -2 else
                             f"{symbol} jumps as quarterly results beat estimates",
                 "url": "https://example.com/news/1", "source": "Example Wire",
                 "summary": "", "datetime": int(datetime(2026, 9, 30, 18).timestamp())},
                {"headline": f"<b>{symbol}</b> mission update: what investors should know",
                 "url": "javascript:alert(1)", "source": "Example Daily",
                 "summary": "", "datetime": int(datetime(2026, 9, 29, 12).timestamp())},
            ],
            "premarket": make_snapshot(candles, gap),
            "fundamentals": {"name": name, "industry": industry, "market_cap_m": mcap,
                             "forward_pe": 15.0 + i * 5, "beta": 0.9 + i * 0.2,
                             "dividend_yield": None, "short_pct_float": 4.0 + i * 6, "short_ratio": 1.5 + i,
                             "short_change_pct": 8.0, "call_oi": 120_000 - i * 15_000, "put_oi": 60_000 + i * 10_000,
                             "put_call_oi_ratio": (60_000 + i * 10_000) / (120_000 - i * 15_000),
                             "options_window_days": 30},
        }
    # One ticker whose price history failed
    pipeline["ZZZ"] = {"quote": {}, "candles": pd.DataFrame(), "news": [], "premarket": {}, "fundamentals": {}}
    return pipeline


def sample_market():
    spy = make_candles(560, 0.009, seed=20)
    qqq = make_candles(480, 0.012, seed=21)
    vix = make_candles(17, 0.05, seed=22, drift=0.0)
    return {
        "SPY": {"candles": spy, "premarket": make_snapshot(spy, -0.3)},
        "QQQ": {"candles": qqq, "premarket": make_snapshot(qqq, -0.5)},
        "^VIX": {"candles": vix, "premarket": {}},
    }


def sample_earnings():
    return {"BBB": {"date": "2026-10-05", "hour": "amc", "eps_estimate": 0.42}}


def sample_movers():
    return {
        "day_gainers": [{"symbol": "GGG", "change_pct": 18.2}, {"symbol": "HHH", "change_pct": 12.5}],
        "day_losers": [{"symbol": "LLL", "change_pct": -15.1}, {"symbol": "MMM", "change_pct": -9.8}],
        "most_actives": [{"symbol": "NNN", "change_pct": 1.2}, {"symbol": "OOO", "change_pct": -2.4}],
    }
