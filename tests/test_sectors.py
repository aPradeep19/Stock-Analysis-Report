"""Tests for automatic sector grouping, news merging and analyst-rating fetching (no network)."""

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.fetcher import FinnhubFetcher, merge_news
from src.sectors import assign_sectors, fund_from_label, sector_context

NY = ZoneInfo("America/New_York")


def closes_from_returns(returns, index):
    return pd.Series(100 * np.cumprod(1 + returns), index=index)


class SectorGroupingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.index = pd.bdate_range(end="2026-09-30", periods=250)
        n = len(self.index)
        self.market = rng.normal(0, 0.01, n)
        self.chips = rng.normal(0, 0.012, n)  # what chip stocks share beyond the market
        self.software = rng.normal(0, 0.012, n)
        funds = {
            "SPY": self.market,
            "SMH": self.market + self.chips,
            "IGV": self.market + self.software,
            "XLK": self.market + 0.5 * self.chips + 0.5 * self.software,
        }
        self.funds = pd.DataFrame({k: closes_from_returns(v, self.index) for k, v in funds.items()})
        self.rng = rng

    def stock(self, *components, noise=0.01):
        returns = self.market + sum(components) + self.rng.normal(0, noise, len(self.index))
        return closes_from_returns(returns, self.index)

    def test_clear_price_match_wins_over_the_label(self):
        # Labeled like a hardware/internet company, but trades with chips.
        groups = assign_sectors(
            {"CHIP": self.stock(self.chips)},
            self.funds,
            {"CHIP": ("Communication Services", "Internet Content")},
        )
        self.assertEqual((groups["CHIP"]["fund"], groups["CHIP"]["method"]), ("SMH", "price match"))

    def test_weak_match_falls_back_to_the_label(self):
        groups = assign_sectors(
            {"BIO": self.stock(noise=0.05)}, self.funds, {"BIO": ("Healthcare", "Biotechnology")}
        )
        self.assertEqual((groups["BIO"]["fund"], groups["BIO"]["method"]), ("XBI", "label"))

    def test_short_history_uses_the_label(self):
        young = self.stock(self.chips).iloc[-30:]
        groups = assign_sectors({"NEW": young}, self.funds, {"NEW": ("Technology", "Software - Application")})
        self.assertEqual((groups["NEW"]["fund"], groups["NEW"]["method"]), ("IGV", "label"))

    def test_override_beats_everything(self):
        groups = assign_sectors({"CHIP": self.stock(self.chips)}, self.funds, {}, overrides={"chip": "xlu"})
        self.assertEqual((groups["CHIP"]["fund"], groups["CHIP"]["method"]), ("XLU", "override"))

    def test_unplaceable_stock_is_left_out(self):
        groups = assign_sectors({"ODD": self.stock(noise=0.05)}, self.funds, {})
        self.assertNotIn("ODD", groups)

    def test_label_mapping(self):
        self.assertEqual(fund_from_label("Technology", "Semiconductors"), "SMH")
        self.assertEqual(fund_from_label("Technology", "Software - Infrastructure"), "IGV")
        self.assertEqual(fund_from_label("Industrials", "Specialty Industrial Machinery"), "XLI")
        self.assertIsNone(fund_from_label(None, None))

    def test_context_lists_same_group_peers(self):
        groups = {
            "NVDA": {"fund": "SMH", "name": "Semiconductor ETF"},
            "MU": {"fund": "SMH", "name": "Semiconductor ETF"},
            "NOW": {"fund": "IGV", "name": "Software ETF"},
        }
        moves = {
            "NVDA": {"gap_pct": -2.8, "last_session_pct": 1.0},
            "MU": {"gap_pct": -4.0, "last_session_pct": 2.0},
            "NOW": {"gap_pct": 0.5, "last_session_pct": 0.1},
        }
        context = sector_context(groups, {"SMH": {"gap_pct": -3.1, "last_session_pct": -1.0}}, moves)
        self.assertEqual(context["NVDA"]["stock"], moves["NVDA"])  # its own moves, for the comparison
        self.assertEqual(context["NVDA"]["peers"], [("MU", -4.0)])
        self.assertEqual(context["NVDA"]["gap_pct"], -3.1)
        self.assertNotIn("NOW", context)  # its fund's prices weren't available


class NewsMergeTests(unittest.TestCase):
    def test_duplicates_dropped_and_newest_first(self):
        finnhub = [
            {
                "headline": "Chipmaker beats estimates",
                "url": "https://a.com/1?utm_source=newsletter",
                "datetime": 100,
            },
            {"headline": "Old story", "url": "https://a.com/2", "datetime": 50},
        ]
        yahoo = [
            {
                "headline": "Chipmaker BEATS estimates!",
                "url": "https://b.com/9",
                "datetime": 101,
            },  # same story
            {"headline": "Different", "url": "https://a.com/1", "datetime": 90},  # same URL
            {"headline": "Overnight press release", "url": "https://b.com/3", "datetime": 200},
        ]
        merged = merge_news(finnhub, yahoo, limit=10)
        self.assertEqual(
            [a["headline"] for a in merged],
            ["Overnight press release", "Chipmaker BEATS estimates!", "Old story"],
        )

    def test_links_differing_only_by_query_are_different_articles(self):
        # Finnhub's links are all https://finnhub.io/api/news?id=...; only the id differs.
        finnhub = [
            {"headline": f"Story {i}", "url": f"https://finnhub.io/api/news?id={i}", "datetime": i}
            for i in range(5)
        ]
        self.assertEqual(len(merge_news(finnhub)), 5)

    def test_limit(self):
        items = [{"headline": f"h{i}", "url": f"u{i}", "datetime": i} for i in range(20)]
        self.assertEqual(len(merge_news(items, limit=10)), 10)


class YahooFeedTests(unittest.TestCase):
    def test_reads_current_and_older_news_formats(self):
        raw = [
            {
                "id": "1",
                "content": {
                    "title": "Nested-format headline",
                    "pubDate": "2026-09-30T23:00:00Z",
                    "summary": "Summary text",
                    "provider": {"displayName": "Business Wire"},
                    "canonicalUrl": {"url": "https://example.com/n"},
                },
            },
            {
                "title": "Flat-format headline",
                "providerPublishTime": 1790000000,
                "publisher": "Reuters",
                "link": "https://example.com/f",
            },
            {"content": {"title": "", "pubDate": "2026-09-30T22:00:00Z"}},  # no headline: skipped
        ]
        with mock.patch("src.fetcher.yf.Ticker", return_value=SimpleNamespace(get_news=lambda count: raw)):
            news = FinnhubFetcher.get_yahoo_news("X")
        self.assertEqual([a["headline"] for a in news], ["Nested-format headline", "Flat-format headline"])
        self.assertEqual(news[0]["source"], "Business Wire")
        self.assertEqual(news[0]["url"], "https://example.com/n")
        self.assertEqual(news[0]["datetime"], int(pd.Timestamp("2026-09-30T23:00:00Z").timestamp()))
        self.assertEqual(news[1]["source"], "Reuters")

    def test_yahoo_outage_returns_nothing(self):
        def down(count):
            raise RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***")

        with (
            mock.patch("src.fetcher.yf.Ticker", return_value=SimpleNamespace(get_news=down)),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(FinnhubFetcher.get_yahoo_news("X"), [])

    def test_rating_changes_since_cutoff(self):
        table = pd.DataFrame(
            {
                "Firm": ["Melius Research", "DA Davidson", "Old Firm"],
                "ToGrade": ["Hold", "Buy", "Buy"],
                "FromGrade": ["Buy", "Buy", ""],
                "Action": ["down", "main", "init"],
                "priorPriceTarget": [315.0, 300.0, 0.0],
                "currentPriceTarget": [185.0, 200.0, 0.0],
            },
            index=pd.DatetimeIndex(
                ["2026-10-01 11:00:00", "2026-09-30 21:30:00", "2026-09-20 12:00:00"], name="GradeDate"
            ),  # UTC, as Yahoo serves them
        )
        since = datetime(2026, 9, 29, 16, 0, tzinfo=NY)
        with mock.patch("src.fetcher.yf.Ticker", return_value=SimpleNamespace(upgrades_downgrades=table)):
            changes = FinnhubFetcher.get_rating_changes("X", since)
        self.assertEqual([c["firm"] for c in changes], ["Melius Research", "DA Davidson"])
        self.assertEqual((changes[0]["prior_target"], changes[0]["target"]), (315.0, 185.0))
        self.assertEqual(changes[0]["action"], "down")


if __name__ == "__main__":
    unittest.main()
