"""Tests for the alert rules. Each test builds a tiny synthetic history with one known property."""
import unittest

import numpy as np
import pandas as pd

from src.analyzer import PUT_CALL_LEVELS, SHORT_INTEREST_LEVELS, QuantitativeAnalyzer, level_of
from tests.fixtures import make_candles

THRESHOLDS = {
    "price_change_pct": 5.0, "premarket_gap_pct": 3.0, "unusual_move_sigma": 2.0, "min_move_pct": 1.5,
    "volume_surge_multiplier": 2.0, "rsi_oversold": 30, "rsi_overbought": 70, "rsi_bullish": 50,
    "near_52w_pct": 3.0, "earnings_lookahead_days": 7,
}


def one_stock(candles, gap_pct=None, news=None, fundamentals=None):
    last = float(candles["close"].iloc[-1])
    premarket = {} if gap_pct is None else {"price": last * (1 + gap_pct / 100), "session": "pre-market"}
    return {"X": {"quote": {}, "candles": candles, "news": news or [], "premarket": premarket,
                  "fundamentals": fundamentals or {}}}


def move_alerts(result):
    return [a for a in result["X"]["alerts"] if a["kind"] == "move"]


class VolumeTests(unittest.TestCase):
    def test_volume_ratio_uses_previous_20_sessions(self):
        candles = make_candles(100, 0.01, sessions=60, seed=1)
        candles["volume"] = 1_000_000.0
        candles.iloc[-1, candles.columns.get_loc("volume")] = 3_000_000.0
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(one_stock(candles))
        # If today's bar were inside its own average, the ratio would be ~2.73 instead of 3.0
        self.assertAlmostEqual(result["X"]["metrics"]["volume_ratio"], 3.0)
        self.assertTrue(any(a["kind"] == "volume" for a in result["X"]["alerts"]))


class MoveRuleTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = QuantitativeAnalyzer(THRESHOLDS)

    def test_big_gap_on_calm_stock_alerts(self):
        candles = make_candles(100, 0.01, seed=2, last_return=0.0)
        alerts = move_alerts(self.analyzer.analyze(one_stock(candles, gap_pct=-4.0)))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["direction"], "down")
        self.assertIn("Gapping down 4.0% pre-market", alerts[0]["text"])

    def test_routine_gap_on_volatile_stock_is_ignored(self):
        # Typical move ~6%: a 4% gap is a normal day for this stock
        candles = make_candles(10, 0.06, seed=3, last_return=0.0)
        self.assertEqual(move_alerts(self.analyzer.analyze(one_stock(candles, gap_pct=4.0))), [])

    def test_unusual_move_below_fixed_threshold_alerts(self):
        # Typical move ~1%: a 2.5% gap is under the 3% bar but 2.5x normal
        candles = make_candles(100, 0.01, seed=4, last_return=0.0)
        alerts = move_alerts(self.analyzer.analyze(one_stock(candles, gap_pct=2.5)))
        self.assertEqual(len(alerts), 1)
        self.assertIn("its typical daily move", alerts[0]["text"])

    def test_tiny_move_on_very_calm_stock_is_ignored(self):
        # 1% is >3x a 0.3% typical move, but below min_move_pct (1.5%)
        candles = make_candles(100, 0.003, seed=5, last_return=0.0)
        self.assertEqual(move_alerts(self.analyzer.analyze(one_stock(candles, gap_pct=1.0))), [])


class TrendTests(unittest.TestCase):
    def test_close_below_200_day_average(self):
        index = pd.bdate_range(end="2026-09-30", periods=260)
        close = np.full(260, 100.0)
        close[:-1] += np.linspace(-1, 1, 259)   # gentle slope so averages are well-defined
        close[-1] = 90.0
        candles = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                                "close": close, "volume": 1_000_000.0}, index=index)
        alerts = QuantitativeAnalyzer(THRESHOLDS).analyze(one_stock(candles))["X"]["alerts"]
        texts = [a["text"] for a in alerts]
        self.assertTrue(any("below its 200-day average" in t for t in texts), texts)
        self.assertTrue(any("below its 50-day average" in t for t in texts), texts)


class SentimentTests(unittest.TestCase):
    def test_whole_word_matching(self):
        analyzer = QuantitativeAnalyzer(THRESHOLDS)
        news = [{"headline": "Company mission statement updated"},        # "miss" inside a word: neutral
                {"headline": "Shares plunge after downgrade"},             # bearish
                {"headline": "Stock falls as guidance misses estimates"}]  # bearish
        result = analyzer._analyze_sentiment(news)
        self.assertEqual((result["positive"], result["negative"], result["neutral"]), (0, 2, 1))
        self.assertEqual(result["label"], "Bearish")


class EarningsAndErrorsTests(unittest.TestCase):
    def test_upcoming_earnings_alert(self):
        candles = make_candles(100, 0.01, seed=6)
        date = (pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(one_stock(candles), earnings={"X": {"date": date, "hour": "amc"}})
        events = [a for a in result["X"]["alerts"] if a["kind"] == "event"]
        self.assertEqual(len(events), 1)
        self.assertIn("after the close", events[0]["text"])

    def test_missing_history_is_reported_not_raised(self):
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(
            {"X": {"quote": {}, "candles": pd.DataFrame(), "news": [], "premarket": {}}})
        self.assertEqual(result["X"], {"error": "No price history returned"})


class PositioningTests(unittest.TestCase):
    """Forward P/E, short interest and options open interest."""

    def analyze(self, **fundamentals):
        candles = make_candles(100, 0.01, seed=6)
        return QuantitativeAnalyzer(THRESHOLDS).analyze(one_stock(candles, fundamentals=fundamentals))["X"]

    def positioning(self, result):
        return [a for a in result["alerts"] if a["kind"] == "positioning"]

    def test_heavily_shorted_stock_gets_a_minor_alert(self):
        result = self.analyze(short_pct_float=29.4, short_ratio=7.6, short_change_pct=4.8)
        signal = next(t for t in result["triggers"] if t.startswith("Short interest"))
        self.assertIn("29.4% of float (very high), 7.6 days to cover (high), +5% vs. prior report. A crowded short",
                      signal)
        self.assertEqual(result["metrics"]["short_level"], "Very high")
        alerts = self.positioning(result)
        self.assertEqual(len(alerts), 1)
        self.assertIn("Heavily shorted", alerts[0]["text"])
        self.assertLess(alerts[0]["severity"], 1.0)  # page only, never the header

    def test_high_short_interest_that_is_hard_to_cover_alerts(self):
        result = self.analyze(short_pct_float=14.0, short_ratio=12.0)
        self.assertEqual(result["metrics"]["days_to_cover_level"], "Very high")
        self.assertEqual(len(self.positioning(result)), 1)

    def test_normal_short_interest_is_a_signal_only(self):
        result = self.analyze(short_pct_float=1.3, short_ratio=2.3)
        self.assertTrue(any(t.startswith("Short interest 1.3% of float (low), 2.3 days to cover (moderate).")
                            for t in result["triggers"]))
        self.assertEqual(self.positioning(result), [])

    def test_level_boundaries(self):
        analyzer = QuantitativeAnalyzer(THRESHOLDS)
        names = [level_of(v, analyzer.short_levels, SHORT_INTEREST_LEVELS)[0] for v in (4.9, 5, 10, 19.9, 20)]
        self.assertEqual(names, ["Low", "Moderate", "High", "High", "Very high"])
        names = [level_of(v, analyzer.put_call_levels, PUT_CALL_LEVELS)[0] for v in (0.3, 0.6, 1.0, 1.3, 2.0)]
        self.assertEqual(names, ["Very bullish", "Bullish", "Neutral", "Cautious", "Bearish"])

    def test_bad_level_config_is_rejected(self):
        with self.assertRaises(ValueError):
            QuantitativeAnalyzer({**THRESHOLDS, "short_interest_levels_pct": [5, 10]})

    def test_call_heavy_options_lean_bullish(self):
        result = self.analyze(call_oi=60_000, put_oi=20_000, put_call_oi_ratio=1 / 3, options_window_days=30)
        self.assertTrue(any(t.startswith("Open interest, next 30 days: 60K calls vs. 20K puts. "
                                         "Put/call 0.33 (very bullish): call-heavy") for t in result["triggers"]))
        self.assertIn("3.0 calls open for every put", self.positioning(result)[0]["text"])

    def test_put_heavy_options_lean_bearish(self):
        result = self.analyze(call_oi=10_000, put_oi=20_000, put_call_oi_ratio=2.0)
        self.assertEqual(result["metrics"]["put_call_level"], "Bearish")
        self.assertEqual(self.positioning(result)[0]["direction"], "down")

    def test_neutral_options_are_a_signal_only(self):
        result = self.analyze(call_oi=10_000, put_oi=10_000, put_call_oi_ratio=1.0)
        self.assertEqual(result["metrics"]["put_call_level"], "Neutral")
        self.assertEqual(self.positioning(result), [])

    def test_thin_option_chain_is_not_flagged(self):
        result = self.analyze(call_oi=300, put_oi=20, put_call_oi_ratio=20 / 300)
        self.assertEqual(self.positioning(result), [])
        self.assertNotIn("put_call_level", result["metrics"])

    def test_most_calls_expiring_this_week_alerts(self):
        result = self.analyze(call_oi_week=133_410, call_oi_week_pct=64.3, week_end="2026-10-16")
        self.assertIn("133K calls (64% of near-term calls) expire by Fri Oct 16 (monthly expiration)",
                      result["triggers"])
        alerts = self.positioning(result)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["direction"], "caution")
        self.assertLess(alerts[0]["severity"], 1.0)

    def test_few_calls_expiring_is_a_signal_only(self):
        result = self.analyze(call_oi_week=20_000, call_oi_week_pct=25.0, week_end="2026-10-23")
        self.assertIn("20K calls (25% of near-term calls) expire by Fri Oct 23", result["triggers"])
        self.assertEqual(self.positioning(result), [])

    def test_thin_weekly_calls_are_not_flagged(self):
        result = self.analyze(call_oi_week=400, call_oi_week_pct=90.0, week_end="2026-10-23")
        self.assertEqual(self.positioning(result), [])

    def test_institutional_ownership_signal(self):
        result = self.analyze(institutional_pct=71.4)
        self.assertIn("Institutions hold 71.4% of shares", result["triggers"])
        self.assertEqual(self.positioning(result), [])

    def test_net_debt(self):
        result = self.analyze(total_debt_m=10_200, total_cash_m=8_040, operating_cashflow_m=5_260)
        self.assertAlmostEqual(result["metrics"]["net_debt_m"], 2_160)
        self.assertIn("Net debt $2.2B (debt $10.2B, cash $8.0B)", result["triggers"])
        self.assertNotIn("cash_runway_months", result["metrics"])  # cash-flow positive: no runway

    def test_net_cash(self):
        result = self.analyze(total_debt_m=38_860, total_cash_m=62_470)
        self.assertIn("Net cash $23.6B (debt $38.9B, cash $62.5B)", result["triggers"])

    def test_short_cash_runway_alerts(self):
        result = self.analyze(total_debt_m=170, total_cash_m=30, operating_cashflow_m=-140)
        self.assertAlmostEqual(result["metrics"]["cash_runway_months"], 30 / 140 * 12)
        self.assertEqual(result["metrics"]["cash_runway_level"], "Short")
        self.assertTrue(any(t.startswith("Burning $140M a year from operations: cash lasts ~3 months at that "
                                         "rate (short)") for t in result["triggers"]))
        alerts = [a for a in result["alerts"] if a["kind"] == "balance_sheet"]
        self.assertEqual(len(alerts), 1)
        self.assertIn("Cash runway ~3 months: $30M of cash, burning $140M a year", alerts[0]["text"])
        self.assertLess(alerts[0]["severity"], 1.0)  # page only, never the header

    def test_long_cash_runway_is_a_signal_only(self):
        result = self.analyze(total_debt_m=1, total_cash_m=140, operating_cashflow_m=-30)
        self.assertEqual(result["metrics"]["cash_runway_level"], "Long")
        self.assertTrue(any("cash lasts ~4.7 years" in t for t in result["triggers"]))
        self.assertEqual([a for a in result["alerts"] if a["kind"] == "balance_sheet"], [])

    def test_forward_pe_uses_last_close_and_12_month_eps(self):
        result = self.analyze(forward_eps=2.0, forward_pe=99.0)  # 99.0 = Yahoo's fallback, ignored
        self.assertAlmostEqual(result["metrics"]["forward_pe"], result["metrics"]["last_close"] / 2.0)

    def test_negative_forward_pe_is_blanked(self):
        for fundamentals in ({"forward_eps": -1.5}, {"forward_pe": -113.4}):  # computed, and Yahoo's fallback
            result = self.analyze(**fundamentals)
            self.assertIsNone(result["metrics"]["forward_pe"])
            self.assertIn("No forward P/E: analysts expect a loss over the next 12 months", result["triggers"])


class RankingTests(unittest.TestCase):
    def test_stock_with_more_problems_ranks_first(self):
        results = {
            "ONE": {"alerts": [{"text": "a", "direction": "down", "severity": 1.5, "kind": "move"}], "metrics": {}},
            "MANY": {"alerts": [{"text": "b", "direction": "down", "severity": 1.4, "kind": "move"},
                                {"text": "c", "direction": "down", "severity": 1.3, "kind": "volume"}], "metrics": {}},
            "MINOR": {"alerts": [{"text": "d", "direction": "up", "severity": 0.6, "kind": "momentum"}], "metrics": {}},
        }
        ranked = QuantitativeAnalyzer.rank_stocks(results, min_severity=1.0)
        self.assertEqual([r["symbol"] for r in ranked], ["MANY", "ONE"])  # MINOR stays off the header


if __name__ == "__main__":
    unittest.main()


class SessionTests(unittest.TestCase):
    """Manual runs after the open must compare against the right close and say which session it is."""

    def test_unfinished_daily_bar_is_dropped_until_4pm(self):
        from datetime import datetime
        from src.fetcher import MARKET_TZ, FinnhubFetcher
        candles = make_candles(100, 0.01, sessions=30, end="2026-10-01")
        morning = datetime(2026, 10, 1, 7, 50, tzinfo=MARKET_TZ)
        evening = datetime(2026, 10, 1, 18, 0, tzinfo=MARKET_TZ)
        self.assertEqual(FinnhubFetcher.drop_unfinished_bar(candles, morning).index[-1].date().isoformat(), "2026-09-30")
        self.assertEqual(FinnhubFetcher.drop_unfinished_bar(candles, evening).index[-1].date().isoformat(), "2026-10-01")
        yesterday_only = candles.iloc[:-1]
        self.assertEqual(len(FinnhubFetcher.drop_unfinished_bar(yesterday_only, morning)), len(yesterday_only))

    def test_after_hours_move_is_labeled_after_hours(self):
        candles = make_candles(100, 0.01, seed=7, last_return=0.0)
        data = one_stock(candles, gap_pct=-4.0)
        data["X"]["premarket"]["session"] = "after-hours"
        alerts = move_alerts(QuantitativeAnalyzer(THRESHOLDS).analyze(data))
        self.assertEqual(len(alerts), 1)
        self.assertIn("Trading down 4.0% after hours", alerts[0]["text"])


class MissingDataTests(unittest.TestCase):
    def test_trailing_row_without_prices_is_ignored(self):
        # Yahoo can add a price-less row (e.g. a dividend on a day that hasn't traded yet)
        candles = make_candles(100, 0.01, sessions=276, seed=8, end="2026-09-30")
        blank = pd.DataFrame({"open": [np.nan], "high": [np.nan], "low": [np.nan], "close": [np.nan],
                              "volume": [np.nan]}, index=[pd.Timestamp("2026-10-01")])
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(one_stock(pd.concat([candles, blank])))
        self.assertNotIn("error", result["X"])
        self.assertAlmostEqual(result["X"]["metrics"]["last_close"], float(candles["close"].iloc[-1]))
