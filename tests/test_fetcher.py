"""Tests for the Yahoo-sourced fundamentals and options data (yfinance is mocked: no network)."""
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from src.fetcher import MARKET_TZ, FinnhubFetcher


def chain(call_oi, put_oi):
    return SimpleNamespace(calls=pd.DataFrame({"openInterest": call_oi}),
                           puts=pd.DataFrame({"openInterest": put_oi}))


class YahooFundamentalsTests(unittest.TestCase):
    def test_short_interest_is_converted_to_percent(self):
        fiscal_year_end = datetime.now(MARKET_TZ) + timedelta(days=73)  # 20% of this fiscal year left
        info = {"forwardPE": 15.1, "sharesShort": 120, "sharesShortPriorMonth": 100,
                "shortPercentOfFloat": 0.1504, "shortRatio": 1.94, "marketCap": 2_000_000_000,
                "heldPercentInstitutions": 0.71398, "totalDebt": 10_200_000_000, "totalCash": 8_040_000_000,
                "operatingCashflow": -140_000_000,
                "nextFiscalYearEnd": int(fiscal_year_end.timestamp())}
        estimates = pd.DataFrame({"avg": [2.5, 2.7, 10.0, 15.0]}, index=["0q", "+1q", "0y", "+1y"])
        with mock.patch("src.fetcher.yf.Ticker", return_value=SimpleNamespace(info=info, earnings_estimate=estimates)):
            data = FinnhubFetcher("key").get_yahoo_fundamentals("X")
        self.assertAlmostEqual(data["forward_eps"], 0.2 * 10.0 + 0.8 * 15.0, places=2)
        self.assertIsNone(data["forward_pe"])  # Yahoo's figure only used without estimates
        self.assertAlmostEqual(data["short_pct_float"], 15.04)
        self.assertAlmostEqual(data["short_change_pct"], 20.0)
        self.assertAlmostEqual(data["institutional_pct"], 71.398)
        self.assertEqual((data["total_debt_m"], data["total_cash_m"], data["operating_cashflow_m"]),
                         (10_200, 8_040, -140))
        self.assertEqual(data["market_cap_m"], 2000)
        self.assertIsNone(data["beta"])  # missing fields stay None so Finnhub's values win

    def test_falls_back_to_yahoo_forward_pe_without_estimates(self):
        ticker = SimpleNamespace(info={"forwardPE": 15.1}, earnings_estimate=pd.DataFrame())
        with mock.patch("src.fetcher.yf.Ticker", return_value=ticker):
            data = FinnhubFetcher("key").get_yahoo_fundamentals("X")
        self.assertIsNone(data["forward_eps"])
        self.assertEqual(data["forward_pe"], 15.1)
        self.assertIsNone(data["institutional_pct"])

    def test_ntm_eps_blend(self):
        today = datetime(2026, 10, 6).date()
        blend = FinnhubFetcher.blend_ntm_eps
        # NVDA-style: fiscal year ends Jan 25, ~30% of it left
        self.assertAlmostEqual(blend(9.31, 15.80, datetime(2027, 1, 25).date(), today),
                               111 / 365 * 9.31 + 254 / 365 * 15.80)
        self.assertEqual(blend(9.31, 15.80, datetime(2028, 6, 1).date(), today), 9.31)   # weight capped at 1
        self.assertEqual(blend(9.31, 15.80, datetime(2026, 9, 1).date(), today), 15.80)  # year already over
        self.assertEqual(blend(None, 15.80, None, today), 15.80)
        self.assertIsNone(blend(None, float("nan"), None, today))


class OptionsInterestTests(unittest.TestCase):
    def test_sums_only_expirations_inside_the_window(self):
        today = datetime.now(MARKET_TZ).date()
        soon, later, far = (today + timedelta(days=d) for d in (3, 20, 90))
        chains = {str(soon): chain([100, 50, None], [40]), str(later): chain([200], [60, 10]),
                  str(far): chain([10_000], [10_000])}
        ticker = mock.Mock(options=tuple(chains))
        ticker.option_chain.side_effect = chains.get
        with mock.patch("src.fetcher.yf.Ticker", return_value=ticker):
            data = FinnhubFetcher("key").get_options_interest("X", days_ahead=30)
        self.assertEqual(data["call_oi"], 350)
        self.assertEqual(data["put_oi"], 110)
        self.assertAlmostEqual(data["put_call_oi_ratio"], 110 / 350)

    def test_counts_calls_expiring_by_friday(self):
        today = datetime.now(MARKET_TZ).date()
        friday = today + timedelta(days=(4 - today.weekday()) % 7)
        chains = {str(friday): chain([300], [10]), str(friday + timedelta(days=7)): chain([100], [10])}
        ticker = mock.Mock(options=tuple(chains))
        ticker.option_chain.side_effect = chains.get
        with mock.patch("src.fetcher.yf.Ticker", return_value=ticker):
            data = FinnhubFetcher("key").get_options_interest("X", days_ahead=30)
        self.assertEqual(friday.weekday(), 4)
        self.assertEqual(data["week_end"], friday.isoformat())
        self.assertEqual(data["call_oi_week"], 300)
        self.assertAlmostEqual(data["call_oi_week_pct"], 75.0)

    def test_no_listed_options(self):
        with mock.patch("src.fetcher.yf.Ticker", return_value=mock.Mock(options=())):
            self.assertEqual(FinnhubFetcher("key").get_options_interest("X"), {})


if __name__ == "__main__":
    unittest.main()
