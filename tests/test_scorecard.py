"""Tests for yesterday's scorecard: logging, history reading and scoring (temp files, synthetic candles)."""

import json
import os
import tempfile
import unittest
from datetime import date

import pandas as pd

from src.scorecard import (
    alert_rows,
    build_scorecard,
    earnings_line,
    load_history,
    summary_line,
    write_pending,
)

TODAY = date(2026, 10, 5)  # a Monday; the last logged report is Friday Oct 2


def candles(closes):
    """Daily bars from {"2026-10-01": close, ...}; open/high/low aren't used by the scoring."""
    index = pd.DatetimeIndex(list(closes))
    values = list(closes.values())
    return pd.DataFrame(
        {"open": values, "high": values, "low": values, "close": values, "volume": 1e6}, index=index
    )


def row(symbol="AAA", day="2026-10-02", direction="up", last_close=100.0, price=104.0, **extra):
    base = {
        "date": day,
        "symbol": symbol,
        "kind": "move",
        "direction": direction,
        "severity": 1.5,
        "text": f"{symbol} alert",
        "session": "pre-market",
        "last_close": last_close,
        "price": price,
        "gap_pct": (price / last_close - 1) * 100,
        "typical_move_pct": 2.0,
        "benchmark": "QQQ",
        "benchmark_last_close": 400.0,
        "benchmark_price": 400.0,
        "benchmark_gap_pct": 0.0,
    }
    return {**base, **extra}


QQQ = candles({"2026-09-30": 398.0, "2026-10-01": 400.0, "2026-10-02": 402.0})  # +0.5% from 400


class ScoringTests(unittest.TestCase):
    def score(self, history, stock_closes, status="ok"):
        data = {"QQQ": QQQ, **{s: candles(c) for s, c in stock_closes.items()}}
        return build_scorecard(history, status, data, TODAY)

    def test_held_gap(self):
        card = self.score([row()], {"AAA": {"2026-10-01": 100.0, "2026-10-02": 106.0}})
        result = card["rows"][0]
        self.assertTrue(result["held"])  # closed above the 100 prior close
        self.assertAlmostEqual(result["follow_pct"], (106 / 104 - 1) * 100)
        self.assertAlmostEqual(result["vs_bench_pct"], (106 / 104 - 1) * 100 - 0.5)
        self.assertEqual(card["summary"]["held"], 1)
        self.assertEqual(
            summary_line(card),
            "Fri Oct 2: 1 of 1 directional alert held; avg follow-through +1.9% (+1.4% vs. Nasdaq).",
        )

    def test_filled_gap(self):
        # Gapped down to 96 from 100, but closed back above the prior close.
        card = self.score(
            [row(direction="down", price=96.0)], {"AAA": {"2026-10-01": 100.0, "2026-10-02": 101.0}}
        )
        result = card["rows"][0]
        self.assertFalse(result["held"])
        self.assertAlmostEqual(result["follow_pct"], -(101 / 96 - 1) * 100)  # moved against a "down" call
        self.assertAlmostEqual(result["vs_bench_pct"], -(101 / 96 - 1) * 100 + 0.5)

    def test_missing_bar_is_skipped_not_guessed(self):
        # Friday's bar isn't in the history (e.g. a holiday or a data gap).
        card = self.score(
            [row(), row("BBB")],
            {"AAA": {"2026-10-01": 100.0}, "BBB": {"2026-10-01": 100.0, "2026-10-02": 99.0}},
        )
        self.assertEqual([r["symbol"] for r in card["rows"]], ["BBB"])
        self.assertEqual(card["skipped"], 1)
        self.assertIn("1 not scored (no session data)", summary_line(card))

    def test_no_history(self):
        card = self.score([], {})
        self.assertEqual(card["status"], "none")
        self.assertEqual(summary_line(card), "No prior alerts to score.")

    def test_todays_rows_are_not_scored(self):
        card = self.score([row(day=TODAY.isoformat())], {"AAA": {"2026-10-02": 100.0}})
        self.assertEqual(card["status"], "none")

    def test_unreadable_history(self):
        card = self.score([], {}, status="error")
        self.assertEqual(summary_line(card), "Alert history unavailable today.")

    def test_only_the_latest_report_and_only_directional_alerts(self):
        history = [
            row(day="2026-10-01"),  # older report: ignored
            row(),
            row("CCC", direction="caution", kind="rsi"),  # not directional: ignored
        ]
        card = self.score(
            history, {"AAA": {"2026-10-01": 100.0, "2026-10-02": 106.0}, "CCC": {"2026-10-02": 1.0}}
        )
        self.assertEqual(card["date"], "2026-10-02")
        self.assertEqual(len(card["rows"]), 1)
        self.assertEqual(card["skipped"], 0)

    def test_after_hours_alert_is_judged_on_the_next_session(self):
        history = [row(day="2026-10-01", session="after-hours", last_close=100.0, price=104.0)]
        card = self.score(history, {"AAA": {"2026-09-30": 95.0, "2026-10-01": 100.0, "2026-10-02": 103.0}})
        self.assertAlmostEqual(card["rows"][0]["close"], 103.0)  # Oct 2, not Oct 1
        self.assertTrue(card["rows"][0]["held"])

    def test_earnings_reaction_vs_typical_move(self):
        history = [
            row("AAA", kind="event", direction="caution", earnings_date="2026-10-01", earnings_hour="amc"),
            row("BBB", kind="event", direction="caution", earnings_date="2026-10-02", earnings_hour="bmo"),
            row(
                "CCC", kind="event", direction="caution", earnings_date="2026-09-30", earnings_hour="bmo"
            ),  # older
        ]
        closes = {"2026-09-30": 100.0, "2026-10-01": 100.0, "2026-10-02": 108.0}
        card = self.score(history, {"AAA": closes, "BBB": closes, "CCC": closes})
        reactions = {r["symbol"]: r for r in card["earnings"]}
        self.assertEqual(set(reactions), {"AAA", "BBB"})  # both reacted in Friday's session
        self.assertAlmostEqual(reactions["AAA"]["move_pct"], 8.0)
        self.assertAlmostEqual(reactions["AAA"]["multiple"], 4.0)
        self.assertEqual(
            earnings_line(reactions["AAA"]),
            "AAA earnings (after Thu's close): moved +8.0% vs. a typical 2.0% (4.0x)",
        )

    def test_hit_rates_need_20_report_days(self):
        days = pd.bdate_range("2026-09-01", periods=21)
        closes = {d.strftime("%Y-%m-%d"): 100.0 + i for i, d in enumerate(days)}
        bench = candles({d.strftime("%Y-%m-%d"): 400.0 for d in days})
        history = [row(day=d, last_close=c - 1, price=c - 0.5) for d, c in list(closes.items())[1:]]
        data = {"QQQ": bench, "AAA": candles(closes)}

        few = build_scorecard(history[:19], "ok", data, date(2026, 10, 30))
        self.assertIsNone(few["hit_rates"])
        many = build_scorecard(history, "ok", data, date(2026, 10, 30))
        self.assertEqual(many["days_logged"], 20)
        self.assertEqual(many["hit_rates"][0]["kind"], "Price move")
        self.assertEqual(many["hit_rates"][0]["scored"], 20)
        self.assertEqual(many["hit_rates"][0]["held_pct"], 100.0)


class HistoryFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "history", "alerts.jsonl")
        self.pending = os.path.join(self.tmp.name, "output", "alerts_pending.jsonl")

    def test_first_run_has_no_file(self):
        self.assertEqual(load_history(self.path), ([], "missing"))

    def test_corrupt_lines_are_skipped(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(row())
                + "\n{not json\n\n"
                + json.dumps({"no": "date"})
                + "\n"
                + json.dumps(row("BBB"))
                + "\n"
            )
        from unittest import mock

        with mock.patch("builtins.print"):
            rows, status = load_history(self.path)
        self.assertEqual(status, "ok")
        self.assertEqual([r["symbol"] for r in rows], ["AAA", "BBB"])

    def test_pending_rows_skip_dates_already_logged(self):
        self.assertEqual(write_pending([row()], [], self.pending), 1)
        with open(self.pending, encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["symbol"], "AAA")
        os.remove(self.pending)
        self.assertEqual(write_pending([row()], [row()], self.pending), 0)  # a second send that day
        self.assertFalse(os.path.exists(self.pending))
        self.assertEqual(write_pending([], [], self.pending), 0)

    def test_rows_from_a_real_report_are_complete_and_json_safe(self):
        from tests.test_report import build_report

        _, ctx, _ = build_report()
        market = {"QQQ": {"last_close": 400.0, "premarket_price": 398.0, "gap_pct": -0.5}}
        results = {s["symbol"]: s for s in ctx["stocks"]}
        rows = alert_rows(ctx["ranked"], results, market, date(2026, 10, 1), "QQQ")
        self.assertEqual(len(rows), sum(len(item["alerts"]) for item in ctx["ranked"]))
        first = json.loads(json.dumps(rows[0]))  # survives a JSON round trip (no numpy types)
        for field in (
            "date",
            "symbol",
            "kind",
            "direction",
            "severity",
            "text",
            "last_close",
            "price",
            "gap_pct",
            "benchmark_gap_pct",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["date"], "2026-10-01")
        self.assertEqual(first["benchmark_price"], 398.0)
        earnings = [r for r in rows if r["kind"] == "event"]
        self.assertTrue(all(r.get("earnings_date") for r in earnings))


if __name__ == "__main__":
    unittest.main()
