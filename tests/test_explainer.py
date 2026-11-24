"""Tests for the 'why it's moving' explainer. The Anthropic client is always faked: no network, no cost."""

import json
import os
import tempfile
import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

from src.explainer import NO_CATALYST, NO_NEWS_YET, explain_moves
from tests.fixtures import make_candles

NY = ZoneInfo("America/New_York")
TODAY = date(2026, 10, 1)
# make_candles(end="2026-09-30"): last close Wed Sep 30 4 PM ET, the one before Tue Sep 29 4 PM ET.
AFTER_CLOSE = int(datetime(2026, 9, 30, 19, 0, tzinfo=NY).timestamp())  # since last close
DURING_SESSION = int(datetime(2026, 9, 30, 10, 0, tzinfo=NY).timestamp())  # during last session
OLD = int(datetime(2026, 9, 25, 10, 0, tzinfo=NY).timestamp())  # older


def ranked(*symbols, overnight=True):
    """Ranked items; overnight=True means their only alert is a pre-market gap."""
    alert = (
        {"text": "Gapping down 6.0% pre-market", "window": "overnight"}
        if overnight
        else {"text": "Closed down 7% last session"}
    )
    return [{"symbol": s, "alerts": [dict(alert)]} for s in symbols]


def article(symbol, i, when):
    return {
        "headline": f"{symbol} headline {i}",
        "summary": "word " * 200,  # ~1000 characters, must be trimmed
        "source": f"Wire {i}",
        "url": f"https://example.com/{symbol}/{i}",
        "datetime": when,
    }


def stock_data(symbol, times=(AFTER_CLOSE,) * 7, ratings=None):
    return {
        "candles": make_candles(100, 0.01, sessions=30, end="2026-09-30"),
        "news": [article(symbol, i, when) for i, when in enumerate(times)],
        "ratings": ratings or [],
    }


def fake_client(text=None, error=None, stop_reason="end_turn"):
    """A stand-in client whose messages.create returns `text` as the model's JSON (or raises)."""
    create = mock.Mock()
    if error is not None:
        create.side_effect = error
    else:
        thinking = SimpleNamespace(type="thinking", thinking="")
        create.return_value = SimpleNamespace(
            stop_reason=stop_reason, content=[thinking, SimpleNamespace(type="text", text=text)]
        )
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def reply(**entries):
    def cites(n):
        return [] if n == -1 else (n if isinstance(n, list) else [n])

    return json.dumps({s: {"explanation": text, "evidence": cites(n)} for s, (text, n) in entries.items()})


class ExplainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def explain(self, client, items, data, config=None, sectors=None):
        return explain_moves(
            items, data, config, sector_context=sectors, client=client, cache_dir=self.tmp.name, today=TODAY
        )

    def sent(self, client):
        kwargs = client.beta.messages.create.call_args.kwargs
        return kwargs, {s["ticker"]: s for s in json.loads(kwargs["messages"][0]["content"])}

    # -- normal path ---------------------------------------------------------------------------
    def test_normal_response(self):
        client = fake_client(
            reply(AAA=("Shares fell after the company cut its outlook.", 2), BBB=("Unsupported.", -1))
        )
        data = {"AAA": stock_data("AAA"), "BBB": stock_data("BBB")}
        lines = self.explain(client, ranked("AAA", "BBB"), data)

        self.assertEqual(
            lines["AAA"],
            {
                "text": "Shares fell after the company cut its outlook.",
                "source": "Wire 2",
                "url": "https://example.com/AAA/2",
            },
        )
        # Fresh news existed but didn't explain the move.
        self.assertEqual(lines["BBB"], {"text": NO_CATALYST, "source": None, "url": None})

        kwargs, payload = self.sent(client)
        self.assertEqual(client.beta.messages.create.call_count, 1)  # one batched call
        self.assertEqual(kwargs["model"], "claude-sonnet-5-5")  # the default
        self.assertEqual(kwargs["output_config"]["effort"], "medium")
        # Opted into server-side fallbacks, so a declined request is re-run instead of lost.
        self.assertEqual(
            (kwargs["fallbacks"], kwargs["betas"]), ("default", ["server-side-fallback-2026-07-01"])
        )
        self.assertEqual(set(kwargs["output_config"]["format"]["schema"]["properties"]), {"AAA", "BBB"})
        stock = payload["AAA"]
        self.assertEqual(stock["last_close"], "Wed 2026-09-30 16:00 ET")
        self.assertEqual(stock["move_to_explain"], "move since the last close (pre-market)")
        self.assertEqual(len(stock["evidence"]), 5)  # capped at 5 articles
        self.assertTrue(all(len(e["summary"]) <= 300 for e in stock["evidence"]))
        self.assertTrue(all(e["timing"] == "since last close" for e in stock["evidence"]))

    # -- failures ------------------------------------------------------------------------------
    def test_malformed_json_gives_no_lines(self):
        with mock.patch("builtins.print"):
            lines = self.explain(
                fake_client('{"AAA": {"sentence": "cut off mid'), ranked("AAA"), {"AAA": stock_data("AAA")}
            )
        self.assertEqual(lines, {})
        self.assertEqual(os.listdir(self.tmp.name), [])  # nothing cached, so the next run retries

    def test_truncated_or_refused_response_gives_no_lines(self):
        for reason in ("max_tokens", "refusal"):
            with mock.patch("builtins.print"):
                lines = self.explain(
                    fake_client("{}", stop_reason=reason), ranked("AAA"), {"AAA": stock_data("AAA")}
                )
            self.assertEqual(lines, {}, reason)

    def test_timeout_gives_no_lines(self):
        with mock.patch("builtins.print") as printed:
            lines = self.explain(
                fake_client(error=TimeoutError("read timed out")), ranked("AAA"), {"AAA": stock_data("AAA")}
            )
        self.assertEqual(lines, {})
        self.assertIn("skipped", printed.call_args.args[0])

    def test_sdk_timeout_error_gives_no_lines(self):
        try:
            import anthropic
            import httpx2
        except ImportError:
            self.skipTest("anthropic SDK not installed")
        error = anthropic.APITimeoutError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        with mock.patch("builtins.print"):
            self.assertEqual(
                self.explain(fake_client(error=error), ranked("AAA"), {"AAA": stock_data("AAA")}), {}
            )

    def test_dry_run_without_key_skips_quietly(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), mock.patch("builtins.print") as printed:
            lines = explain_moves(
                ranked("AAA"), {"AAA": stock_data("AAA")}, cache_dir=self.tmp.name, today=TODAY, quiet=True
            )
        self.assertEqual(lines, {})
        printed.assert_not_called()

    # -- validation ----------------------------------------------------------------------------
    def test_advice_and_bad_citations_are_dropped(self):
        client = fake_client(
            reply(
                AAA=("Analysts say investors should buy the dip.", 0),
                BBB=("Shares rose on a new contract.", 9),
            )
        )
        data = {"AAA": stock_data("AAA"), "BBB": stock_data("BBB")}
        self.assertEqual(self.explain(client, ranked("AAA", "BBB"), data), {})

    def test_factual_rating_and_sell_off_wording_is_kept(self):
        client = fake_client(
            reply(
                AAA=("Melius Research downgraded the stock from Buy to Hold and cut its target.", 0),
                BBB=("Shares fell in a sector-wide sell-off after a rival's warning.", 0),
            )
        )
        data = {"AAA": stock_data("AAA"), "BBB": stock_data("BBB")}
        lines = self.explain(client, ranked("AAA", "BBB"), data)
        self.assertIn("from Buy to Hold", lines["AAA"]["text"])
        self.assertIn("sell-off", lines["BBB"]["text"])

    # -- freshness -----------------------------------------------------------------------------
    def test_nothing_fresh_skips_the_model(self):
        client = fake_client("{}")
        data = {"AAA": stock_data("AAA", times=(DURING_SESSION, OLD))}  # nothing since the close
        lines = self.explain(client, ranked("AAA"), data)
        self.assertEqual(lines["AAA"]["text"], NO_NEWS_YET)
        client.beta.messages.create.assert_not_called()

    def test_older_news_cannot_explain_a_gap(self):
        client = fake_client(reply(AAA=("Shares fell after last week's recall.", 1)))
        data = {"AAA": stock_data("AAA", times=(AFTER_CLOSE, OLD))}
        _, payload = None, None
        lines = self.explain(client, ranked("AAA"), data)
        _, payload = self.sent(client)
        self.assertEqual([e["timing"] for e in payload["AAA"]["evidence"]], ["since last close", "older"])
        self.assertEqual(lines["AAA"]["text"], NO_CATALYST)  # cited the old item: replaced

    def test_last_session_news_can_explain_a_last_session_move(self):
        client = fake_client(reply(AAA=("Shares fell after a guidance cut during the session.", 0)))
        data = {"AAA": stock_data("AAA", times=(DURING_SESSION,))}
        lines = self.explain(client, ranked("AAA", overnight=False), data)
        self.assertEqual(lines["AAA"]["source"], "Wire 0")

    def test_fresh_items_are_kept_when_older_ones_crowd_the_feed(self):
        client = fake_client(reply(AAA=("Shares fell on the overnight news.", 0)))
        times = (OLD,) * 8 + (AFTER_CLOSE,)  # the only fresh article is the oldest-sorted
        self.explain(client, ranked("AAA"), {"AAA": stock_data("AAA", times=times)})
        _, payload = self.sent(client)
        self.assertEqual(payload["AAA"]["evidence"][0]["timing"], "since last close")

    # -- ratings and sector evidence -----------------------------------------------------------
    def test_rating_change_is_evidence(self):
        rating = {
            "datetime": AFTER_CLOSE,
            "firm": "Melius Research",
            "action": "down",
            "from_grade": "Buy",
            "to_grade": "Hold",
            "prior_target": 315.0,
            "target": 185.0,
        }
        client = fake_client(reply(AAA=("Melius Research downgraded the stock to Hold.", 0)))
        lines = self.explain(client, ranked("AAA"), {"AAA": stock_data("AAA", times=(), ratings=[rating])})
        _, payload = self.sent(client)
        self.assertEqual(
            payload["AAA"]["evidence"][0]["text"],
            "Melius Research downgraded AAA from Buy to Hold; price target cut from $315 to $185",
        )
        self.assertEqual(lines["AAA"]["source"], "Melius Research (analyst rating)")

    def test_sector_move_is_evidence_even_without_news(self):
        sector = {
            "fund": "SMH",
            "name": "Semiconductor ETF",
            "gap_pct": -3.1,
            "last_session_pct": -1.2,
            "peers": [("MU", -4.0), ("NVDA", None)],
        }
        client = fake_client(
            reply(
                AAA=("Fell with chip stocks; the semiconductor ETF is down 3.1% pre-market.", 1)
            )  # item 1: the sector line follows the one article
        )
        lines = self.explain(
            client, ranked("AAA"), {"AAA": stock_data("AAA", times=(OLD,))}, sectors={"AAA": sector}
        )
        _, payload = self.sent(client)
        sector_item = payload["AAA"]["evidence"][-1]
        self.assertEqual(
            sector_item["text"],
            "Semiconductor ETF (SMH): -3.1% pre-market, -1.2% last session. "
            "Watchlist peers in the same group, pre-market: MU -4.0%.",
        )
        self.assertEqual(lines["AAA"]["source"], "Market data (SMH)")

    def test_no_fresh_news_fallback_says_so(self):
        sector = {
            "fund": "SMH",
            "name": "Semiconductor ETF",
            "gap_pct": 0.2,
            "last_session_pct": 0.1,
            "peers": [],
        }
        client = fake_client(reply(AAA=(NO_CATALYST, -1)))
        lines = self.explain(
            client, ranked("AAA"), {"AAA": stock_data("AAA", times=(OLD,))}, sectors={"AAA": sector}
        )
        self.assertEqual(lines["AAA"]["text"], NO_NEWS_YET)  # only old news: "not out yet", not "no catalyst"

    # -- caching -------------------------------------------------------------------------------
    def test_same_day_rerun_only_asks_about_unexplained_stocks(self):
        data = {"AAA": stock_data("AAA"), "BBB": stock_data("BBB")}
        first = fake_client(reply(AAA=("Cut its outlook.", 0), BBB=(NO_CATALYST, -1)))
        self.explain(first, ranked("AAA", "BBB"), data)

        second = fake_client(reply(BBB=("Won a contract.", 1)))
        lines = self.explain(second, ranked("AAA", "BBB"), data)
        _, payload = self.sent(second)
        self.assertEqual(list(payload), ["BBB"])  # AAA's explanation was cached; BBB's fallback wasn't
        self.assertEqual(
            (lines["AAA"]["text"], lines["BBB"]["text"]), ("Cut its outlook.", "Won a contract.")
        )

        third = fake_client("{}")
        self.explain(third, ranked("AAA", "BBB"), data)
        third.beta.messages.create.assert_not_called()  # everything explained and cached: no call, no bill

    # -- several sources, comparisons, cache versioning ------------------------------------------
    def test_every_cited_source_is_credited(self):
        rating = {
            "datetime": AFTER_CLOSE,
            "firm": "Melius Research",
            "action": "up",
            "from_grade": "Hold",
            "to_grade": "Buy",
        }
        sector = {
            "fund": "SMH",
            "name": "Semiconductor ETF",
            "gap_pct": 2.0,
            "last_session_pct": 1.0,
            "stock": {"gap_pct": 6.0},
            "peers": [],
        }
        client = fake_client(
            reply(
                AAA=(
                    "Shares jumped after a raised outlook. Melius upgraded the stock to Buy, and chip stocks also rose.",
                    [0, 2, 3],
                )
            )
        )
        data = {"AAA": stock_data("AAA", times=(AFTER_CLOSE, AFTER_CLOSE), ratings=[rating])}
        lines = self.explain(client, ranked("AAA"), data, sectors={"AAA": sector})
        self.assertEqual(
            lines["AAA"]["source"], "Wire 0 · Melius Research (analyst rating) · Market data (SMH)"
        )
        self.assertEqual(lines["AAA"]["url"], "https://example.com/AAA/0")  # first cited article

    def test_one_stale_citation_spoils_the_line(self):
        client = fake_client(reply(AAA=("Fell on the outlook cut and last week's recall.", [0, 1])))
        lines = self.explain(client, ranked("AAA"), {"AAA": stock_data("AAA", times=(AFTER_CLOSE, OLD))})
        self.assertEqual(lines["AAA"]["text"], NO_CATALYST)

    def test_sector_item_states_how_the_stock_compares(self):
        cases = {
            31.5: "outpaced its sector fund (2.1x",
            15.0: "in line with its sector fund (1.0x",
            3.0: "lagged its sector fund (0.2x",
            -2.0: "moved against its sector fund (-0.1x",
        }
        for stock_move, words in cases.items():
            sector = {
                "fund": "SMH",
                "name": "Semiconductor ETF",
                "gap_pct": None,
                "last_session_pct": 15.0,
                "stock": {"last_session_pct": stock_move},
                "peers": [],
            }
            client = fake_client(reply(AAA=(NO_CATALYST, -1)))
            self.explain(
                client,
                ranked("AAA", overnight=False),
                {"AAA": stock_data("AAA", times=())},
                sectors={"AAA": sector},
            )
            _, payload = self.sent(client)
            self.assertIn(
                f"AAA {stock_move:+.1f}% last session: {words}", payload["AAA"]["evidence"][-1]["text"]
            )

    def test_lines_from_another_model_or_prompt_version_are_not_reused(self):
        data = {"AAA": stock_data("AAA")}
        with open(os.path.join(self.tmp.name, "explanations_2026-10-01.json"), "w", encoding="utf-8") as f:
            json.dump({"AAA": {"text": "Old Haiku line.", "source": "Wire 0", "url": None}}, f)  # old format
        client = fake_client(reply(AAA=("Fresh Sonnet line.", 0)))
        lines = self.explain(client, ranked("AAA"), data)
        self.assertEqual(lines["AAA"]["text"], "Fresh Sonnet line.")
        client.beta.messages.create.assert_called_once()

        again = fake_client("{}")
        self.explain(again, ranked("AAA"), data, {"model": "claude-haiku-5-5"})  # different model
        again.beta.messages.create.assert_called_once()

    def test_after_hours_run_names_the_session(self):
        sector = {
            "fund": "SMH",
            "name": "Semiconductor ETF",
            "gap_pct": 1.0,
            "last_session_pct": 15.0,
            "stock": {"gap_pct": 0.8},
            "peers": [("MU", -0.3)],
        }
        data = {"AAA": {**stock_data("AAA"), "premarket": {"session": "after-hours"}}}
        client = fake_client(reply(AAA=(NO_CATALYST, -1)))
        self.explain(client, ranked("AAA"), data, sectors={"AAA": sector})
        _, payload = self.sent(client)
        text = payload["AAA"]["evidence"][-1]["text"]
        self.assertIn("+1.0% after hours", text)
        self.assertIn("AAA +0.8% after hours: in line with", text)
        self.assertIn("peers in the same group, after hours: MU -0.3%", text)
        self.assertNotIn("pre-market", text)
        self.assertEqual(payload["AAA"]["move_to_explain"], "move since the last close (after hours)")

    def test_saved_lines_need_the_same_last_close(self):
        data = {"AAA": stock_data("AAA")}
        first = fake_client(reply(AAA=("Line written against Wednesday's close.", 0)))
        self.explain(first, ranked("AAA"), data)

        # Same day, but the price history now ends a session earlier (e.g. Yahoo was late with a
        # bar): the saved line describes a different move, so it's asked again.
        shorter = {"AAA": {**data["AAA"], "candles": data["AAA"]["candles"].iloc[:-1]}}
        second = fake_client(reply(AAA=("Line written against Tuesday's close.", 0)))
        lines = self.explain(second, ranked("AAA"), shorter)
        second.beta.messages.create.assert_called_once()
        self.assertEqual(lines["AAA"]["text"], "Line written against Tuesday's close.")
        self.assertNotIn("basis", lines["AAA"])  # bookkeeping stays in the cache file


if __name__ == "__main__":
    unittest.main()
