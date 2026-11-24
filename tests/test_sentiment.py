"""Tests for FinBERT news sentiment. The classifier is always faked: the model is never downloaded."""

import builtins
import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

from src.analyzer import QuantitativeAnalyzer
from src.sentiment import article_text, is_relevant, load_classifier, name_terms, score_news, stock_sentiment
from tests.fixtures import make_candles
from tests.test_analyzer import THRESHOLDS

NY = ZoneInfo("America/New_York")
# make_candles(end="2026-09-30"): last close Wed Sep 30 4 PM ET, the one before Tue Sep 29 4 PM ET.
FRESH = int(datetime(2026, 9, 30, 19, 0, tzinfo=NY).timestamp())  # since the last close
SESSION = int(datetime(2026, 9, 30, 11, 0, tzinfo=NY).timestamp())  # during the last session
OLD = int(datetime(2026, 9, 25, 11, 0, tzinfo=NY).timestamp())  # older
CANDLES = make_candles(100, 0.01, sessions=30, end="2026-09-30")


def scored(headline, positive=0.0, negative=0.0, when=FRESH, summary=""):
    """An article as score_news leaves it."""
    neutral = max(0.0, 1.0 - positive - negative)
    probs = {"positive": positive, "negative": negative, "neutral": neutral}
    return {
        "headline": headline,
        "summary": summary,
        "datetime": when,
        "sentiment": {**probs, "label": max(probs, key=probs.get)},
    }


class FakeClassifier:
    """Scores by keyword: 'beats' -> positive, 'plunges' -> negative, else neutral. Records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        out = []
        for text in texts:
            p, n = (0.9, 0.05) if "beats" in text else (0.05, 0.9) if "plunges" in text else (0.1, 0.1)
            out.append([{"label": "positive", "score": p}, {"label": "negative", "score": n},
                        {"label": "neutral", "score": 1 - p - n}])  # fmt: skip
        return out


class ScoringPassTests(unittest.TestCase):
    def pipeline(self):
        shared = {"headline": "Chipmaker beats estimates", "summary": ""}
        return {
            "AAA": {"news": [dict(shared), {"headline": "AAA plunges on recall", "summary": ""}]},
            "BBB": {"news": [dict(shared)]},  # same text as AAA's first article
            "CCC": {"news": []},
        }

    def test_one_batched_pass_with_duplicate_texts_scored_once(self):
        data, fake = self.pipeline(), FakeClassifier()
        self.assertTrue(score_news(data, {"input": "headline"}, fake))
        self.assertEqual(len(fake.calls), 1)
        texts, kwargs = fake.calls[0]
        self.assertEqual(sorted(texts), ["AAA plunges on recall", "Chipmaker beats estimates"])
        self.assertEqual((kwargs["truncation"], kwargs["max_length"]), (True, 512))
        first = data["AAA"]["news"][0]["sentiment"]
        self.assertEqual(first["label"], "positive")
        self.assertAlmostEqual(first["positive"], 0.9)
        self.assertEqual(data["AAA"]["news"][1]["sentiment"]["label"], "negative")
        self.assertEqual(data["BBB"]["news"][0]["sentiment"]["label"], "positive")

    def test_scoring_failure_leaves_articles_unscored(self):
        data = self.pipeline()

        def broken(texts, **kwargs):
            raise RuntimeError("out of memory")

        with mock.patch("builtins.print") as printed:
            self.assertFalse(score_news(data, {}, broken))
        self.assertNotIn("sentiment", data["AAA"]["news"][0])
        self.assertEqual(printed.call_count, 1)  # one warning

    def test_disabled(self):
        self.assertFalse(score_news(self.pipeline(), {"enabled": False}, FakeClassifier()))

    def test_missing_transformers_returns_none_with_one_warning(self):
        real_import = builtins.__import__

        def no_transformers(name, *args, **kwargs):
            if name.startswith("transformers"):
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", no_transformers), mock.patch("builtins.print") as printed:
            self.assertIsNone(load_classifier({}))
        self.assertEqual(printed.call_count, 1)
        self.assertIn("keyword sentiment", printed.call_args.args[0])


class InputTextTests(unittest.TestCase):
    def test_headline_plus_first_sentences(self):
        article = {
            "headline": "Acme beats estimates",
            "summary": "Revenue rose 20%. Margins widened! Guidance raised.",
        }
        self.assertEqual(article_text(article, "headline"), "Acme beats estimates")
        self.assertEqual(article_text(article), "Acme beats estimates. Revenue rose 20%. Margins widened!")

    def test_teaser_summaries_and_headings_are_cleaned(self):
        teaser = {"headline": "Is Acme cheap?", "summary": "Today's big story: Nvidia added $954 billion."}
        self.assertEqual(article_text(teaser), "Is Acme cheap?")
        heading = {"headline": "Acme falls", "summary": "Key TakeawaysAcme fell 13% on Monday. More text."}
        self.assertEqual(article_text(heading, sentences=1), "Acme falls. Acme fell 13% on Monday.")

    def test_summary_repeating_the_headline_isnt_doubled(self):
        article = {"headline": "Acme beats estimates", "summary": "Acme beats estimates."}
        self.assertEqual(article_text(article), "Acme beats estimates")


class RelevanceTests(unittest.TestCase):
    def relevant(self, headline, symbol, name, aliases=None, summary=""):
        return is_relevant({"headline": headline, "summary": summary}, name_terms(symbol, name, aliases))

    def test_ticker_and_name_without_legal_endings(self):
        self.assertTrue(self.relevant("NVDA slides after the open", "NVDA", "NVIDIA Corp"))
        self.assertTrue(self.relevant("Nvidia euphoria wears off", "NVDA", "NVIDIA Corp"))
        self.assertTrue(self.relevant("Analysts like Micron here", "MU", "Micron Technology, Inc."))
        self.assertTrue(self.relevant("Nebius signs a cloud deal", "NBIS", "Nebius Group N.V. Class A"))
        self.assertFalse(self.relevant("U.S. chip stocks fall premarket", "NBIS", "Nebius Group N.V."))

    def test_ticker_must_be_in_capitals(self):
        self.assertFalse(self.relevant("Stocks are falling now", "NOW", "ServiceNow, Inc."))
        self.assertTrue(self.relevant("NOW beats on subscriptions", "NOW", "ServiceNow, Inc."))
        self.assertTrue(self.relevant("ServiceNow beats on subscriptions", "NOW", "ServiceNow, Inc."))

    def test_name_match_in_summary_counts(self):
        self.assertTrue(
            self.relevant("Premarket movers", "MU", "Micron Technology", summary="Micron leads losers")
        )

    def test_generic_first_words_dont_match_alone(self):
        self.assertFalse(self.relevant("General Motors cuts guidance", "GE", "General Electric Company"))
        self.assertTrue(self.relevant("General Electric cuts guidance", "GE", "General Electric Company"))

    def test_aliases(self):
        self.assertFalse(self.relevant("Google unveils a new model", "GOOGL", "Alphabet Inc."))
        self.assertTrue(self.relevant("Google unveils a new model", "GOOGL", "Alphabet Inc.", ["Google"]))


class StockScoreTests(unittest.TestCase):
    def score(self, articles, config=None, name="Acme Corp"):
        return stock_sentiment(articles, "ACME", name, CANDLES, config or {})

    def test_bullish_and_bearish(self):
        bull = self.score([scored("Acme beats", 0.9, 0.05), scored("Acme raises guidance", 0.8, 0.1)])
        self.assertEqual((bull["label"], bull["method"], bull["positive"]), ("Bullish", "finbert", 2))
        bear = self.score([scored("Acme plunges", 0.05, 0.9), scored("Acme cut to sell", 0.1, 0.7)])
        self.assertEqual(bear["label"], "Bearish")
        self.assertLess(bear["score"], -0.25)

    def test_mixed_when_confident_stories_offset(self):
        mixed = self.score(
            [scored("Acme wins a big contract", 0.9, 0.05), scored("Acme faces a probe", 0.05, 0.9)]
        )
        self.assertEqual(mixed["label"], "Mixed")
        self.assertEqual((mixed["positive"], mixed["negative"]), (1, 1))

    def test_neutral(self):
        flat = self.score(
            [scored("Acme holds its meeting", 0.1, 0.1), scored("Acme names a director", 0.2, 0.1)]
        )
        self.assertEqual(flat["label"], "Neutral")

    def test_fewer_than_two_relevant_articles_is_neutral(self):
        one = self.score(
            [scored("Acme soars on a buyout", 0.95, 0.01), scored("Chip stocks soar", 0.95, 0.01)]
        )
        self.assertEqual((one["label"], one["relevant"], one["total"]), ("Neutral", 1, 2))

    def test_unmatched_articles_dont_count_but_are_marked(self):
        articles = [
            scored("Acme beats", 0.9, 0.05),
            scored("Acme raises", 0.9, 0.05),
            scored("Rival plunges", 0.0, 0.95),
        ]
        result = self.score(articles)
        self.assertEqual(result["label"], "Bullish")
        self.assertEqual(result["negative"], 0)
        self.assertEqual([a["sentiment"]["counted"] for a in articles], [True, True, False])

    def test_fresh_news_outweighs_older_news(self):
        fresh_good = [scored("Acme beats", 0.9, 0.0, when=FRESH), scored("Acme plunges", 0.0, 0.9, when=OLD)]
        result = self.score(fresh_good)
        # (1.0 * 0.9 + 0.3 * -0.9) / 1.3
        self.assertAlmostEqual(result["score"], round((0.9 - 0.27) / 1.3, 3))
        self.assertEqual(result["label"], "Bullish")

        custom = self.score(fresh_good, {"weights": {"since last close": 1.0, "older": 1.0}})
        self.assertAlmostEqual(custom["score"], 0.0)

    def test_session_weight(self):
        result = self.score(
            [scored("Acme beats", 0.8, 0.0, when=SESSION), scored("Acme flat", 0.0, 0.0, when=FRESH)]
        )
        self.assertAlmostEqual(result["score"], round(0.6 * 0.8 / 1.6, 3))

    def test_unscored_articles_give_none(self):
        self.assertIsNone(self.score([{"headline": "Acme beats"}]))
        self.assertIsNone(self.score([]))


class AnalyzerFallbackTests(unittest.TestCase):
    def one_stock(self, news, name="Acme Corp"):
        return {
            "ACME": {
                "quote": {},
                "candles": CANDLES,
                "news": news,
                "premarket": {},
                "fundamentals": {"name": name},
            }
        }

    def test_finbert_score_when_articles_are_scored(self):
        news = [scored("Acme beats", 0.9, 0.05), scored("Acme raises guidance", 0.85, 0.05)]
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(self.one_stock(news))["ACME"]
        self.assertEqual((result["sentiment"], result["sentiment_detail"]["method"]), ("Bullish", "finbert"))

    def test_keyword_fallback_when_transformers_isnt_installed(self):
        news = [{"headline": "Acme shares plunge after downgrade"}, {"headline": "Acme falls on a probe"}]
        real_import = builtins.__import__

        def no_transformers(name, *args, **kwargs):
            if name.startswith("transformers"):
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        data = self.one_stock(news)
        with mock.patch("builtins.__import__", no_transformers), mock.patch("builtins.print"):
            classifier = load_classifier({})
        self.assertIsNone(classifier)  # main.py then skips score_news; articles stay unscored
        result = QuantitativeAnalyzer(THRESHOLDS).analyze(data)["ACME"]
        self.assertEqual(result["sentiment_detail"]["method"], "keywords")
        self.assertEqual(result["sentiment"], "Bearish")


if __name__ == "__main__":
    unittest.main()
