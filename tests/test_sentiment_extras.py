"""Tests for the analysis built on FinBERT scores: strongest story, news vs. price, tone shift,
sector / watchlist mood, sentiment hit rates, and tone in the explainer. No model is downloaded."""

import json
import tempfile
import unittest
from datetime import date

import pandas as pd

from src.analyzer import QuantitativeAnalyzer
from src.scorecard import sentiment_hits_line, sentiment_rows, sentiment_scorecard, tone_history
from src.sentiment import divergence, news_mood, shift, stock_sentiment, strongest_story
from tests.fixtures import make_candles
from tests.test_analyzer import THRESHOLDS
from tests.test_sentiment import CANDLES, FRESH, OLD, scored


def finbert(score, label, relevant=3):
    return {"score": score, "label": label, "relevant": relevant, "method": "finbert"}


class StrongestStoryTests(unittest.TestCase):
    def test_clearest_relevant_story(self):
        articles = [
            scored("Acme flat", 0.2, 0.1),
            scored("Acme plunges", 0.02, 0.95),
            scored("Acme beats", 0.8, 0.1),
        ]
        for a in articles:
            a["sentiment"]["counted"] = True
        best = strongest_story(articles, {})
        self.assertEqual((best["headline"], best["label"]), ("Acme plunges", "negative"))
        self.assertAlmostEqual(best["net"], -0.93)

    def test_nothing_clear_enough(self):
        self.assertIsNone(strongest_story([scored("Acme flat", 0.3, 0.1)], {}))

    def test_part_of_the_stock_score(self):
        result = stock_sentiment(
            [scored("Acme beats", 0.9, 0.05), scored("Acme raises", 0.6, 0.2)],
            "ACME",
            "Acme Corp",
            CANDLES,
            {},
        )
        self.assertEqual(result["strongest"]["headline"], "Acme beats")


class DivergenceTests(unittest.TestCase):
    def test_moves_against_the_news(self):
        line = divergence(8.3, "last session", finbert(-0.4, "Bearish"), {})
        self.assertEqual(
            line,
            "News vs. price: up 8.3% last session despite bearish coverage (score -0.40 over 3 stories naming the stock)",
        )
        self.assertIn(
            "down 5.0% pre-market despite bullish",
            divergence(-5.0, "pre-market", finbert(0.5, "Bullish"), {}),
        )

    def test_moves_without_news(self):
        self.assertIn(
            "with no stories naming the stock", divergence(6.0, "pre-market", finbert(0.0, "Neutral", 0), {})
        )
        self.assertIn(
            "without a clear news lead (mixed", divergence(-6.0, "last session", finbert(0.05, "Mixed"), {})
        )

    def test_aligned_or_keyword_based_is_quiet(self):
        self.assertIsNone(divergence(6.0, "pre-market", finbert(0.5, "Bullish"), {}))
        self.assertIsNone(divergence(-6.0, "pre-market", finbert(-0.5, "Bearish"), {}))
        self.assertIsNone(
            divergence(6.0, "pre-market", {"score": -2, "label": "Bearish", "method": "keywords"}, {})
        )
        self.assertIsNone(divergence(None, "pre-market", finbert(-0.5, "Bearish"), {}))


class MoveWindowTests(unittest.TestCase):
    """A move is only compared with news from its own window."""

    def news(self):
        from tests.test_sentiment import SESSION

        articles = [
            scored("Acme gives back its rally", 0.02, 0.95, when=FRESH),  # after the close
            scored("Acme slips after the bell", 0.05, 0.9, when=FRESH),
            scored("Acme soars on a contract win", 0.95, 0.02, when=SESSION),  # during the session
        ]
        for a in articles:
            a["sentiment"]["counted"] = True
        return articles

    def test_gap_uses_news_since_the_close(self):
        from src.sentiment import window_sentiment

        w = window_sentiment(self.news(), CANDLES, "overnight", {})
        self.assertEqual(
            (w["relevant"], w["label"], w["window_words"]), (2, "Bearish", "since the last close")
        )
        line = divergence(6.0, "pre-market", w, {})
        self.assertEqual(
            line,
            "News vs. price: up 6.0% pre-market despite bearish coverage "
            "(score -0.89 over 2 stories naming the stock since the last close)",
        )

    def test_session_move_ignores_this_mornings_news(self):
        from src.sentiment import window_sentiment

        w = window_sentiment(self.news(), CANDLES, "last session", {})
        self.assertEqual(w["relevant"], 1)  # only the in-session story, not this morning's two
        # Up 30% on a session whose own story was strongly positive: nothing to flag, even though
        # this morning's stories about the pullback are negative.
        self.assertIsNone(divergence(30.0, "last session", w, {}))
        # The same session read against the whole day's news would have looked contradictory.
        whole_day = window_sentiment(self.news(), CANDLES, "overnight", {})
        self.assertIn("despite bearish coverage", divergence(30.0, "last session", whole_day, {}))

    def test_no_stories_in_the_window(self):
        from src.sentiment import window_sentiment

        # The feed reaches back past the session (an older article), and none from it names the stock.
        unrelated_old = scored("Sector roundup", 0.1, 0.1, when=OLD)
        unrelated_old["sentiment"]["counted"] = False
        news = [a for a in self.news() if a["datetime"] == FRESH] + [unrelated_old]
        w = window_sentiment(news, CANDLES, "last session", {})
        self.assertEqual(
            divergence(-8.0, "last session", w, {}),
            "News vs. price: down 8.0% last session with no stories naming the stock during that session",
        )

    def test_no_claim_when_the_feed_doesnt_reach_the_window(self):
        from src.sentiment import window_sentiment

        # Every article is from after the close: the session's stories have scrolled out of the feed.
        w = window_sentiment([a for a in self.news() if a["datetime"] == FRESH], CANDLES, "last session", {})
        self.assertFalse(w["covered"])
        self.assertIsNone(divergence(30.0, "last session", w, {}))


class ShiftTests(unittest.TestCase):
    def test_clear_shift_from_the_recent_average(self):
        line = shift(finbert(-0.38, "Bearish"), [0.5, 0.4, 0.36], {})
        self.assertEqual(
            line,
            "News tone turned more negative: -0.38 today vs. a +0.42 average over the previous 3 report days",
        )

    def test_needs_enough_history_and_a_big_enough_change(self):
        self.assertIsNone(shift(finbert(-0.4, "Bearish"), [0.5, 0.4], {}))  # only 2 days
        self.assertIsNone(shift(finbert(0.3, "Bullish"), [0.1, 0.0, 0.2], {}))  # +0.2 change
        self.assertIsNone(
            shift(finbert(-0.4, "Bearish", relevant=0), [0.5, 0.4, 0.5], {})
        )  # no stories today

    def test_window_uses_the_latest_days(self):
        line = shift(finbert(0.6, "Bullish"), [-0.9, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0], {"shift_window_days": 5})
        self.assertIn("+0.00 average over the previous 5", line)


class MoodTests(unittest.TestCase):
    def results(self):
        shared = scored("Chip stocks plunge", 0.02, 0.95)
        shared["url"] = "https://example.com/chips"

        def stock(*articles):
            for a in articles:
                a["sentiment"]["counted"] = True
            return {"sentiment_detail": {"method": "finbert"}, "all_news": list(articles)}

        return {
            "NVDA": stock(dict(shared, sentiment=dict(shared["sentiment"])), scored("NVDA beats", 0.9, 0.05)),
            "MU": stock(dict(shared, sentiment=dict(shared["sentiment"])), scored("MU plunges", 0.05, 0.9)),
            "NOW": stock(scored("NOW wins deal", 0.9, 0.05)),
        }

    def test_watchlist_and_groups(self):
        groups = {
            "NVDA": {"fund": "SMH", "name": "Semiconductor ETF"},
            "MU": {"fund": "SMH", "name": "Semiconductor ETF"},
            "NOW": {"fund": "IGV", "name": "Software ETF"},
        }
        mood = news_mood(self.results(), groups, {})
        self.assertEqual(mood["watchlist"]["stories"], 4)  # the shared chip story counts once
        self.assertEqual([g["fund"] for g in mood["groups"]], ["SMH"])  # IGV has only one stock
        chips = mood["groups"][0]
        self.assertEqual((chips["name"], chips["stories"], chips["negative"]), ("Semiconductor", 3, 2))
        self.assertEqual(chips["label"], "Bearish")

    def test_none_without_finbert(self):
        self.assertIsNone(news_mood({"AAA": {"sentiment_detail": {"method": "keywords"}}}, {}, {}))


class AnalyzerSignalTests(unittest.TestCase):
    def test_divergence_and_shift_signals(self):
        candles = make_candles(100, 0.01, seed=2, last_return=0.0, end="2026-09-30")
        last = float(candles["close"].iloc[-1])
        news = [scored("Acme plunges on a probe", 0.02, 0.95), scored("Acme faces a lawsuit", 0.05, 0.85)]
        data = {
            "ACME": {
                "quote": {},
                "candles": candles,
                "news": news,
                "premarket": {"price": last * 1.06, "session": "pre-market"},  # gaps UP on bad news
                "fundamentals": {"name": "Acme Corp"},
            }
        }
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(data, sentiment_history={"ACME": [0.5, 0.4, 0.45]})[
            "ACME"
        ]
        kinds = {s["kind"]: s["text"] for s in result["signals"]}
        self.assertIn("up 6.0% pre-market despite bearish coverage", kinds["news_divergence"])
        self.assertIn("News tone turned more negative", kinds["news_shift"])
        self.assertEqual(result["metrics"]["news_divergence"], kinds["news_divergence"])
        self.assertFalse(any(a["kind"].startswith("news") for a in result["alerts"]))  # signals, not alerts
        self.assertEqual(len(result["all_news"]), 2)


class SentimentHistoryTests(unittest.TestCase):
    def test_rows_only_for_finbert_days(self):
        results = {
            "AAA": {
                "sentiment_detail": finbert(0.42, "Bullish"),
                "metrics": {"premarket_price": 101.0, "last_close": 100.0},
            },
            "BBB": {
                "sentiment_detail": {"score": 2, "label": "Bullish", "method": "keywords"},
                "metrics": {},
            },
        }
        rows = sentiment_rows(results, date(2026, 10, 1))
        self.assertEqual([r["symbol"] for r in rows], ["AAA"])
        self.assertEqual(
            (rows[0]["score"], rows[0]["price"], rows[0]["session"]), (0.42, 101.0, "pre-market")
        )
        json.dumps(rows)

    def test_tone_history_oldest_first_before_today(self):
        rows = [
            {"date": "2026-10-02", "symbol": "AAA", "score": 0.2},
            {"date": "2026-10-01", "symbol": "AAA", "score": 0.1},
            {"date": "2026-10-03", "symbol": "AAA", "score": 0.9},  # today: excluded
        ]
        self.assertEqual(tone_history(rows, date(2026, 10, 3)), {"AAA": [0.1, 0.2]})

    def test_hit_rates_after_20_days(self):
        days = pd.bdate_range("2026-09-01", periods=22)
        closes = {d: 100.0 + i for i, d in enumerate(days)}  # rises every day
        candles = {
            "AAA": pd.DataFrame({"close": list(closes.values())}, index=pd.DatetimeIndex(list(closes)))
        }
        rows = []
        for i, d in enumerate(days[:-1]):
            label = "Bullish" if i % 2 == 0 else "Bearish"
            rows.append({"date": d.strftime("%Y-%m-%d"), "symbol": "AAA", "label": label, "price": 100.0 + i - 0.5,
                         "session": "pre-market"})  # fmt: skip
        few = sentiment_scorecard(rows[:19], candles, "2026-12-01")
        self.assertIsNone(few["sentiment_hits"])
        card = sentiment_scorecard(rows, candles, "2026-12-01")
        hits = {h["label"]: h for h in card["sentiment_hits"]}
        self.assertEqual(hits["Bullish"]["hit_pct"], 100.0)  # every Bullish day closed higher
        self.assertEqual(hits["Bearish"]["hit_pct"], 0.0)
        # 11 Bullish days, but the first has no earlier bar to compare with, so 10 are scored.
        self.assertIn("Bullish news was followed by a gain 100% of the time (n=10", sentiment_hits_line(card))


class ExplainerToneTests(unittest.TestCase):
    def test_articles_carry_finbert_tone(self):
        from tests.test_explainer import fake_client, ranked, reply, stock_data

        data = {"AAA": stock_data("AAA", times=(FRESH, OLD))}
        data["AAA"]["news"][0]["sentiment"] = {"positive": 0.05, "negative": 0.9, "neutral": 0.05,
                                               "label": "negative", "counted": True}  # fmt: skip
        client = fake_client(reply(AAA=("Shares fell on the news.", 0)))
        from src.explainer import explain_moves

        with tempfile.TemporaryDirectory() as tmp:
            explain_moves(ranked("AAA"), data, {}, client=client, cache_dir=tmp, today=date(2026, 10, 1))
        payload = json.loads(client.beta.messages.create.call_args.kwargs["messages"][0]["content"])
        first, second = payload[0]["evidence"][0], payload[0]["evidence"][1]
        self.assertEqual(
            (first["tone"], first["tone_score"], first["names_stock"]), ("negative", -0.85, True)
        )
        self.assertNotIn("tone", second)  # unscored article: no tone fields


if __name__ == "__main__":
    unittest.main()
