"""Tests for the email formatter, the PDF, and the dispatcher (no network, nothing is sent)."""

import os
import smtplib
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.analyzer import QuantitativeAnalyzer
from src.dispatcher import EmailDispatcher, parse_recipients
from src.formatter import ReportFormatter, _most_common_session, fmt_market_cap, fmt_pct, fmt_volume
from tests.fixtures import (
    sample_earnings,
    sample_market,
    sample_movers,
    sample_pipeline,
    stress_context,
    stress_earnings,
    stress_pipeline,
    stress_scorecard,
)
from tests.test_analyzer import THRESHOLDS


def build_report(pipeline=None, earnings=None, explanations=None, extras=False):
    """
    The sample (or given) watchlist run through analyzer and formatter. `extras` adds every
    optional line at full length: a 'why' per ranked stock and a busy economic calendar.
    """
    analyzer = QuantitativeAnalyzer(THRESHOLDS)
    pipeline = pipeline if pipeline is not None else sample_pipeline()
    earnings = sample_earnings() if earnings is None else earnings
    market = analyzer.analyze_market(sample_market())
    results = analyzer.analyze(pipeline, market, earnings)
    ranked = analyzer.rank_stocks(results)
    formatter = ReportFormatter({"title": "Pre-Market Brief"}, THRESHOLDS)
    when = pd.Timestamp("2026-10-01 07:53", tz="America/New_York").to_pydatetime()
    calendar, card = [], None
    if extras:
        explanations, calendar = stress_context(ranked)
        card = stress_scorecard(ranked)
    ctx = formatter.build_context(
        results,
        ranked,
        market,
        sample_movers(),
        when,
        economic_calendar=calendar,
        earnings=earnings,
        explanations=explanations,
        scorecard=card,
    )
    return formatter, ctx, pipeline


class FormatterTests(unittest.TestCase):
    def setUp(self):
        self.formatter, self.ctx, self.pipeline = build_report()

    def test_outside_text_is_escaped_and_bad_links_dropped(self):
        html = self.formatter.render_html(self.ctx)
        self.assertIn("&lt;b&gt;AAA&lt;/b&gt;", html)  # headline markup shown as text
        self.assertNotIn("<b>AAA</b>", html)
        self.assertNotIn("javascript:", html)  # unsafe link removed
        self.assertIn("href='https://example.com/news/1'", html)

    def test_header_lists_flagged_stocks_and_errors(self):
        html = self.formatter.render_html(self.ctx)
        self.assertIn("Big &amp; concerning moves", html)
        self.assertIn("No price history returned", html)  # ZZZ shown, not silently dropped
        self.assertEqual(self.ctx["ranked"][0]["symbol"], "AAA")

    def test_subject_names_top_movers(self):
        subject = self.formatter.subject(self.ctx)
        self.assertTrue(subject.startswith("Pre-Market Brief · Thu Oct 1 · AAA ▼4.6%"), subject)

    def test_subject_when_nothing_alerts(self):
        ctx = dict(self.ctx, ranked=[])
        self.assertTrue(self.formatter.subject(ctx).endswith("No major moves"))

    def test_text_version_has_every_stock(self):
        text = self.formatter.render_text(self.ctx)
        for symbol in ("AAA", "BBB", "CCC", "ZZZ"):
            self.assertIn(symbol, text)

    def test_positioning_metrics_are_shown(self):
        text = self.formatter.render_text(self.ctx)
        self.assertIn("Forward P/E (12 mo): 15.0", text)
        self.assertIn("Short interest: 4.0% of float · Low", text)
        self.assertIn("Days to cover: 1.5 · Low", text)
        self.assertIn("Call open interest: 120.0K", text)
        self.assertIn("Put/call ratio: 0.50 · Bullish", text)
        html = self.formatter.render_html(self.ctx)
        self.assertIn("Short interest", html)
        self.assertIn("Put/call ratio", html)

    def test_positioning_guide_follows_config(self):
        guide = dict(
            ReportFormatter({}, {**THRESHOLDS, "short_interest_levels_pct": [4, 8, 15]}).positioning_guide()
        )
        ranges = [(r, name) for r, name, _ in guide["Short interest (% of float)"]]
        self.assertEqual(
            ranges,
            [("under 4%", "Low"), ("4%–8%", "Moderate"), ("8%–15%", "High"), ("15% and up", "Very high")],
        )
        self.assertEqual(len(guide["Put/call open interest"]), 5)
        text = self.formatter.render_text(self.ctx)
        self.assertIn("READING SHORT INTEREST, OPTIONS AND CASH RUNWAY", text)
        self.assertEqual(
            [name for _, name, _ in guide["Cash runway (months)"]], ["Short", "Moderate", "Long"]
        )
        self.assertIn("not how fast the business is growing", text)
        self.assertIn("Reading short interest, options and cash runway", self.formatter.render_html(self.ctx))

    def test_number_formatting(self):
        self.assertEqual(fmt_pct(4.56), "+4.6%")
        self.assertEqual(fmt_pct(None), "—")
        self.assertEqual(fmt_volume(34_400_000), "34.4M")
        self.assertEqual(fmt_market_cap(2_100_000), "$2.1T")


def render_pdf(formatter, ctx, pipeline, tmp):
    """
    Builds the PDF and returns (page texts, {page index: [24pt heading text]}).

    Stock pages are found by their 24pt ticker heading, the only text set that large: plain text
    matching would also hit the ticker's row in the watchlist table.
    """
    from pypdf import PdfReader

    from src.pdf_report import NewsletterPDF

    path = NewsletterPDF(formatter).build(ctx, pipeline, os.path.join(tmp, "report.pdf"))
    texts, headings = [], {}
    for i, page in enumerate(PdfReader(path).pages):

        def visit(text, cm, tm, font, size, i=i):
            if size * tm[0] >= 23.5 and text.strip():
                headings.setdefault(i, []).append(text.strip())

        texts.append(page.extract_text(visitor_text=visit))
    return texts, headings


class PdfTests(unittest.TestCase):
    def assert_structure(self, formatter, ctx, pipeline):
        with tempfile.TemporaryDirectory() as tmp:
            texts, headings = render_pdf(formatter, ctx, pipeline, tmp)
        symbols = [s["symbol"] for s in ctx["stocks"]]

        # Each stock's ticker heading is on exactly one page, in report order, one stock per page.
        pages_by_symbol = {sym: [i for i, h in headings.items() if sym in h] for sym in symbols}
        for sym, pages in pages_by_symbol.items():
            self.assertEqual(len(pages), 1, f"{sym} heading on pages {pages}")
        stock_pages = [pages_by_symbol[sym][0] for sym in symbols]
        header_pages = stock_pages[0]
        self.assertEqual(stock_pages, list(range(header_pages, header_pages + len(symbols))))

        self.assertLessEqual(header_pages, 2)
        self.assertEqual(len(texts), header_pages + len(symbols) + 1)

        self.assertIn("Big & concerning moves", texts[0])
        header_text = " ".join(texts[:header_pages])
        for item in ctx["ranked"]:  # collapsed if need be, but never dropped
            self.assertIn(item["symbol"], header_text)
        self.assertIn("Watchlist at a glance", header_text)
        self.assertNotIn("How alerts work", header_text)
        self.assertIn("How alerts work", texts[-1])
        self.assertIn("Reading short interest, options and cash runway", texts[-1])
        return texts, stock_pages

    def test_pdf_structure_on_sample(self):
        formatter, ctx, pipeline = build_report()
        texts, _ = self.assert_structure(formatter, ctx, pipeline)
        self.assertNotIn("Watchlist at a glance", texts[0])  # page 1 stays the front page

    def test_pdf_structure_on_full_watchlist_stress_case(self):
        pipeline = stress_pipeline()
        formatter, ctx, _ = build_report(pipeline, stress_earnings())
        self.assertEqual(len(ctx["stocks"]), 11)
        self.assertGreaterEqual(len(ctx["ranked"]), 9)
        self.assert_structure(formatter, ctx, pipeline)

    def test_pdf_structure_with_every_extra_line(self):
        """'Why' lines, economic releases, earnings this week and the 10-yr tile, all at once."""
        for name, pipeline, earnings in (
            ("sample", sample_pipeline(), None),
            ("stress", stress_pipeline(), stress_earnings()),
        ):
            with self.subTest(name):
                formatter, ctx, _ = build_report(pipeline, earnings, extras=True)
                self.assertTrue(all(item["why"] for item in ctx["ranked"]))
                self.assertEqual(len(ctx["economic_events"]), 5)
                texts, stock_pages = self.assert_structure(formatter, ctx, pipeline)
                self.assertIn("10-yr Treasury", texts[0])
                self.assertIn("Today (Central):", texts[0])
                self.assertIn("Why:", texts[0])
                self.assertIn("Yesterday's scorecard", texts[1])  # header page 2
                self.assertIn("directional alerts held", texts[1])
                self.assertIn("What's moving it", texts[stock_pages[0]])

    def test_stock_page_skips_signals_repeated_in_key_metrics(self):
        formatter, ctx, pipeline = build_report()
        texts, stock_pages = self.assert_structure(formatter, ctx, pipeline)
        page = texts[stock_pages[0]]
        self.assertIn("Institutional ownership", page)  # the metric row stays
        self.assertNotIn("Institutions hold", page)  # its signal twin doesn't
        self.assertNotIn("Open interest, next 30 days", page)
        self.assertIn("day average", page)  # moving-average signals add information, so they stay

    def test_signals_are_trimmed_least_useful_first(self):
        from src.pdf_report import _signal_trims

        signals = [("rsi", "r"), ("sma_20", "a20"), ("sma_50", "a50"), (None, "untagged"), ("sma_20", "b20")]
        trims = [[text for _, text in shown] for shown in _signal_trims(signals)]
        self.assertEqual(trims[0], ["r", "a20", "a50", "untagged", "b20"])
        self.assertEqual(trims[1], ["r", "a50", "untagged", "b20"])  # 20-day averages go first
        self.assertEqual(trims[3], ["a50", "untagged"])  # then RSI
        self.assertEqual(trims[-2], ["untagged"])  # untagged signals are kept longest
        self.assertEqual(trims[-1], [])

    def test_headlines_are_cut_to_two_lines(self):
        from src.pdf_report import CONTENT_W, HEADLINE_MAX_LINES, NewsletterPDF, _line_count

        pipeline = stress_pipeline()
        formatter, ctx, _ = build_report(pipeline, stress_earnings())
        lines = NewsletterPDF(formatter)._headlines(ctx["stocks"][0], ctx)[1:]
        self.assertEqual(len(lines), 3)
        for paragraph in lines:
            self.assertLessEqual(_line_count(paragraph, CONTENT_W), HEADLINE_MAX_LINES)
            self.assertIn("…", paragraph.text)

    def test_preview_shows_first_page_only(self):
        from PIL import Image

        from src.pdf_report import NewsletterPDF
        from src.preview import render_preview

        formatter, ctx, pipeline = build_report()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = NewsletterPDF(formatter).build(ctx, pipeline, os.path.join(tmp, "report.pdf"))
            png = render_preview(pdf, os.path.join(tmp, "preview.png"))
            with Image.open(png) as im:
                # A single letter page is 11/8.5 = 1.29x as tall as wide
                self.assertAlmostEqual(im.height / im.width, 11 / 8.5, delta=0.05)


class DispatcherTests(unittest.TestCase):
    def test_parse_recipients(self):
        raw = "a@x.com, b@y.com; A@x.com\nnot-an-email  c@z.com"
        self.assertEqual(parse_recipients(raw), ["a@x.com", "b@y.com", "c@z.com"])

    def test_message_hides_recipients_and_attaches_pdf(self):
        dispatcher = EmailDispatcher("me@gmail.com", "abcd efgh ijkl mnop", ["a@x.com", "b@y.com"])
        self.assertEqual(dispatcher.app_password, "abcdefghijklmnop")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
        try:
            msg = dispatcher.build_message("Subject", "<p>hi</p>", "hi", attachments=[f.name])
        finally:
            os.unlink(f.name)
        raw = msg.as_string()
        self.assertEqual(msg["To"], "me@gmail.com")
        self.assertNotIn("a@x.com", raw)  # no recipient address anywhere in the headers/body
        self.assertIsNone(msg["Bcc"])
        types = [part.get_content_type() for part in msg.walk()]
        self.assertIn("text/plain", types)
        self.assertIn("text/html", types)
        self.assertIn("application/pdf", types)

    def test_send_puts_everyone_in_the_envelope(self):
        sent = {}

        class FakeSMTP:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def login(self, user, password):
                sent["login"] = (user, password)

            def send_message(self, msg, from_addr=None, to_addrs=None):
                sent["to"] = to_addrs
                return {}

        dispatcher = EmailDispatcher("me@gmail.com", "pw", ["a@x.com", "me@gmail.com"])
        with mock.patch.object(smtplib, "SMTP_SSL", FakeSMTP):
            self.assertTrue(dispatcher.send("s", "<p>h</p>", "t"))
        self.assertEqual(sent["to"], ["me@gmail.com", "a@x.com"])  # sender once, no duplicates

    def test_send_reports_bad_login(self):
        class FailingSMTP:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def login(self, user, password):
                raise smtplib.SMTPAuthenticationError(535, b"bad")

        dispatcher = EmailDispatcher("me@gmail.com", "pw", ["a@x.com"])
        with mock.patch.object(smtplib, "SMTP_SSL", FailingSMTP):
            self.assertFalse(dispatcher.send("s", "<p>h</p>", "t"))


if __name__ == "__main__":
    unittest.main()


class SessionLabelTests(unittest.TestCase):
    def test_after_hours_run_relabels_columns(self):
        formatter, ctx, _ = build_report()
        for stock in ctx["stocks"]:
            stock["metrics"]["premarket_session"] = "after-hours"
        ctx = formatter.build_context(
            {s["symbol"]: {k: v for k, v in s.items() if k != "symbol"} for s in ctx["stocks"]},
            ctx["ranked"],
            ctx["market"],
            ctx["movers"],
            ctx["generated_at"],
        )
        html = formatter.render_html(ctx)
        self.assertIn("After-hrs", html)
        self.assertIn("after-hours price vs. today", html)
        self.assertNotIn(">Pre-mkt<", html)

    def test_session_ties_go_to_the_later_session(self):
        # Previously decided by set iteration order, which varies with string hash seeds.
        self.assertEqual(_most_common_session(["pre-market", "regular"]), "regular")
        self.assertEqual(_most_common_session(["regular", "pre-market"]), "regular")
        self.assertEqual(_most_common_session(["after-hours", "regular", "regular"]), "regular")
        self.assertEqual(_most_common_session([]), "pre-market")


def internal_links(path):
    """{page index: [target page index, ...]} for every in-document link, in drawing order."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    page_of = {page.indirect_reference.idnum: i for i, page in enumerate(reader.pages)}
    links = {}
    for i, page in enumerate(reader.pages):
        for annot in page.get("/Annots") or []:
            annot = annot.get_object()
            dest = annot.get("/Dest") or (annot.get("/A") or {}).get("/D")
            if annot.get("/Subtype") == "/Link" and dest is not None and not isinstance(dest, str):
                links.setdefault(i, []).append(page_of[dest[0].idnum])
    return links


class JumpLinkTests(unittest.TestCase):
    def test_pdf_tickers_jump_to_their_pages_and_back(self):
        from src.pdf_report import NewsletterPDF

        for name, pipeline, earnings in (
            ("sample", sample_pipeline(), None),
            ("stress", stress_pipeline(), stress_earnings()),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                formatter, ctx, _ = build_report(pipeline, earnings, extras=True)
                path = NewsletterPDF(formatter).build(ctx, pipeline, os.path.join(tmp, "r.pdf"))
                texts, headings = render_pdf(formatter, ctx, pipeline, tmp)
                page_of = {h[0]: i for i, h in headings.items()}
                links = internal_links(path)

                # Page 2's watchlist table: one link per stock, to that stock's page, in table order.
                self.assertEqual(links[1], [page_of[s["symbol"]] for s in ctx["stocks"]])
                # Page 1: every flagged stock links to its page, in rank order.
                self.assertEqual(links[0], [page_of[item["symbol"]] for item in ctx["ranked"]])
                # Every stock page links back to the watchlist on page 2.
                for symbol in (s["symbol"] for s in ctx["stocks"]):
                    self.assertEqual(links[page_of[symbol]], [1], symbol)

    def test_email_tickers_jump_to_their_sections_and_back(self):
        formatter, ctx, _ = build_report()
        html = formatter.render_html(ctx)
        self.assertIn("id='watchlist'", html)
        for stock in ctx["stocks"]:
            anchor = f"stock-{stock['symbol']}"
            self.assertIn(f"id='{anchor}'", html)
            self.assertGreaterEqual(html.count(f"href='#{anchor}'"), 1)  # watchlist table
        for item in ctx["ranked"]:
            self.assertEqual(html.count(f"href='#stock-{item['symbol']}'"), 2)  # table + flagged list
        self.assertEqual(html.count("href='#watchlist'"), len(ctx["stocks"]))

    def test_anchor_names_are_safe(self):
        from src.formatter import anchor_id

        self.assertEqual(anchor_id("BRK.B"), "stock-BRK-B")
        self.assertEqual(anchor_id("^VIX"), "stock--VIX")


class SentimentDisplayTests(unittest.TestCase):
    """FinBERT tone markers and labels in the email, text version and PDF (fake classifier)."""

    def scored_report(self):
        from src.sentiment import score_news

        def fake(texts, **kwargs):  # "plunge" negative, "beats" positive, the rest neutral
            out = []
            for t in texts:
                p, n = (0.9, 0.05) if "beats" in t else (0.05, 0.9) if "plunge" in t else (0.1, 0.1)
                out.append([{"label": "positive", "score": p}, {"label": "negative", "score": n},
                            {"label": "neutral", "score": 1 - p - n}])  # fmt: skip
            return out

        pipeline = sample_pipeline()
        pipeline["AAA"]["news"].append(
            {
                "headline": "Chip stocks plunge across the board",
                "summary": "",
                "source": "Wire",
                "datetime": 1,
            }
        )
        self.assertTrue(score_news(pipeline, {"input": "headline"}, fake))
        return build_report(pipeline)

    def test_markers_and_labels_in_email_and_text(self):
        formatter, ctx, _ = self.scored_report()
        aaa = next(s for s in ctx["stocks"] if s["symbol"] == "AAA")
        self.assertEqual(aaa["sentiment_detail"]["method"], "finbert")
        html, text = formatter.render_html(ctx), formatter.render_text(ctx)
        self.assertIn("title='negative'>–</b>", html)  # sign + colour, not colour alone
        self.assertIn("title='neutral'>·</b>", html)
        self.assertIn("[–] AAA shares plunge after guidance miss", text)
        self.assertNotIn("keyword count", html)
        # Only the two articles naming AAA count: one strongly negative, one neutral -> Bearish.
        self.assertIn("sentiment Bearish (0 positive, 1 negative, 1 neutral)", html)

    def test_unrelated_headlines_are_marked_not_counted(self):
        formatter, ctx, _ = self.scored_report()
        aaa = next(s for s in ctx["stocks"] if s["symbol"] == "AAA")
        # The sector roundup is AAA's third headline, and it doesn't name AAA.
        self.assertEqual((aaa["sentiment_detail"]["relevant"], aaa["sentiment_detail"]["total"]), (2, 3))
        self.assertIn("[–] Chip stocks plunge across the board (not counted)", formatter.render_text(ctx))
        self.assertRegex(
            formatter.render_html(ctx),
            r"Chip stocks plunge across the board<span[^>]*> — Wire · [^<]* not counted",
        )

    def test_mood_news_checks_and_strongest_story(self):
        formatter, ctx, pipeline = self.scored_report()
        aaa = next(s for s in ctx["stocks"] if s["symbol"] == "AAA")
        aaa["metrics"]["news_divergence"] = "News vs. price: down 4.6% pre-market despite bullish coverage"
        ctx["news_mood"] = {
            "watchlist": {"name": "Watchlist", "label": "Bearish", "score": -0.31, "stories": 9, "positive": 1,
                          "negative": 5},
            "groups": [{"name": "Semiconductor", "fund": "SMH", "label": "Bearish", "score": -0.5, "stories": 4}],
        }  # fmt: skip
        html, text = formatter.render_html(ctx), formatter.render_text(ctx)
        mood = "watchlist Bearish (score -0.31; 5 of 9 stories negative, 1 positive) · Semiconductor (SMH) Bearish"
        self.assertIn(mood, text)
        self.assertIn("News tone (FinBERT):</span> " + mood, html)
        self.assertIn("News check: News vs. price: down 4.6% pre-market despite bullish coverage", text)
        self.assertIn("News check", html)
        self.assertIn("(strongest story)", text)  # AAA's plunge headline is its clearest story
        with tempfile.TemporaryDirectory() as tmp:
            texts, _ = render_pdf(formatter, ctx, pipeline, tmp)
        self.assertIn("News tone (FinBERT):", " ".join(texts[0].split()))
        self.assertIn("strongest story", " ".join(" ".join(texts).split()))

    def test_keyword_fallback_is_labelled(self):
        formatter, ctx, _ = build_report()  # unscored articles: keyword method
        self.assertIn("keyword count, FinBERT unavailable", formatter.render_html(ctx))

    def test_pdf_markers_and_finbert_credit(self):
        from src.pdf_report import NewsletterPDF

        formatter, ctx, pipeline = self.scored_report()
        with tempfile.TemporaryDirectory() as tmp:
            texts, _ = render_pdf(formatter, ctx, pipeline, tmp)
            NewsletterPDF(formatter).build(ctx, pipeline, os.path.join(tmp, "r.pdf"))
        self.assertIn("News sentiment", texts[-1])
        self.assertIn("FinBERT", texts[-1])
        self.assertIn("news sentiment by FinBERT (ProsusAI/finbert)", formatter.render_html(ctx))
        aaa_page = next(t for t in texts if "Alpha Semiconductor" in t and "Key metrics" in t)
        page = " ".join(aaa_page.split())
        self.assertIn("sentiment Bearish (0 positive, 1 negative, 1 neutral)", page)
        self.assertIn("Wire · not counted", page)


class ScorecardRenderTests(unittest.TestCase):
    def test_no_history_message_everywhere(self):
        formatter, ctx, pipeline = build_report()
        self.assertIn("No prior alerts to score.", formatter.render_html(ctx))
        self.assertIn("No prior alerts to score.", formatter.render_text(ctx))
        with tempfile.TemporaryDirectory() as tmp:
            texts, _ = render_pdf(formatter, ctx, pipeline, tmp)
        self.assertIn("No prior alerts to score.", texts[1])

    def test_full_scorecard_in_email_and_text(self):
        formatter, ctx, _ = build_report(extras=True)
        html, text = formatter.render_html(ctx), formatter.render_text(ctx)
        earnings = "AAA earnings (after Tue's close): moved +8.1% vs. a typical 2.4% (3.4x)"
        self.assertIn(earnings, text)
        self.assertIn(earnings.replace("'", "&#x27;"), html)  # escaped like all outside text
        for rendered in (html, text):
            self.assertIn("directional alerts held; avg follow-through +0.6% (+0.2% vs. Nasdaq)", rendered)
            self.assertIn("Hit rates over 23 report days", rendered)
            self.assertIn("Held: the session closed on the flagged side of the prior close.", rendered)
        self.assertIn("vs. Nasdaq</th>", html)


class WhyLineTests(unittest.TestCase):
    """Model output is outside text: it must be escaped exactly like a headline."""

    HOSTILE = "Shares <b>jumped</b> after Q3 beat & guidance <script>x</script>"

    def build(self):
        symbol = build_report()[1]["ranked"][0]["symbol"]
        why = {symbol: {"text": self.HOSTILE, "source": "Wire <i>", "url": "https://example.com"}}
        return build_report(explanations=why)

    def test_html_escapes_model_output(self):
        formatter, ctx, _ = self.build()
        html = formatter.render_html(ctx)
        self.assertIn("Shares &lt;b&gt;jumped&lt;/b&gt; after Q3 beat &amp; guidance &lt;script&gt;", html)
        self.assertIn("Wire &lt;i&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<b>jumped</b>", html)

    def test_text_version_shows_it_verbatim(self):
        formatter, ctx, _ = self.build()
        self.assertIn(f"Why: {self.HOSTILE} — Wire <i>", formatter.render_text(ctx))

    def test_pdf_shows_markup_as_text(self):
        formatter, ctx, pipeline = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            texts, _ = render_pdf(formatter, ctx, pipeline, tmp)
        self.assertIn("<b>jumped</b>", " ".join(texts[0].split()))  # shown literally, not as bold

    def test_no_explanations_means_no_lines(self):
        formatter, ctx, _ = build_report()
        self.assertNotIn("Why:", formatter.render_html(ctx))
        self.assertNotIn("Why:", formatter.render_text(ctx))


class MarketContextTests(unittest.TestCase):
    def test_ten_year_yield_in_either_yahoo_scale(self):
        from src.analyzer import _yield_pct

        self.assertAlmostEqual(_yield_pct(4.918), 4.918)
        self.assertAlmostEqual(_yield_pct(49.18), 4.918)

    def test_ten_year_tile_in_basis_points(self):
        formatter, ctx, _ = build_report()
        tnx = ctx["market"]["^TNX"]
        self.assertTrue(3 < tnx["yield_pct"] < 6)
        self.assertIn(" bp last session", formatter.render_html(ctx))
        self.assertIn("10-yr ", formatter.render_text(ctx))

    def test_only_todays_releases_in_time_order(self):
        from datetime import date

        from src.formatter import todays_releases

        entries = [
            {"date": "2026-10-01", "time": "13:00", "event": "FOMC decision"},
            {"date": date(2026, 10, 1), "time": "07:30", "event": "CPI (Sep)"},  # YAML parses dates
            {"date": "2026-10-02", "time": "07:30", "event": "Jobs report"},
            {"date": "2026-10-01", "time": "7.30am", "event": "Bad time"},
            {"time": "08:00", "event": "No date"},
        ]
        with mock.patch("builtins.print"):
            releases = todays_releases(entries, date(2026, 10, 1))
        self.assertEqual(
            releases,
            [{"time": "7:30 AM", "event": "CPI (Sep)"}, {"time": "1:00 PM", "event": "FOMC decision"}],
        )

    def test_earnings_this_week_runs_through_friday(self):
        from datetime import date

        from src.formatter import earnings_this_week

        earnings = {
            "AAA": {"date": "2026-10-02", "hour": "amc"},  # Friday
            "BBB": {"date": "2026-10-01", "hour": "bmo"},  # today
            "CCC": {"date": "2026-10-05", "hour": "bmo"},  # next Monday: excluded
            "DDD": {"date": "2026-09-30", "hour": "amc"},  # yesterday: excluded
        }
        labels = [e["label"] for e in earnings_this_week(earnings, date(2026, 10, 1))]
        self.assertEqual(labels, ["BBB today before open", "AAA Fri after close"])
        # Run on a Saturday, "this week" means the coming one.
        self.assertEqual([e["symbol"] for e in earnings_this_week(earnings, date(2026, 10, 3))], ["CCC"])


class HeadlineDateTests(unittest.TestCase):
    def test_headline_dates_use_new_york_time(self):
        formatter, ctx, _ = build_report()
        # 01:30 UTC on Oct 2 is 9:30 PM ET on Oct 1; a UTC runner used to print "Oct 2".
        late_evening = int(pd.Timestamp("2026-10-02 01:30", tz="UTC").timestamp())
        ctx["stocks"][0]["top_news"] = [
            {"headline": "Late story", "source": "Wire", "datetime": late_evening}
        ]
        html = formatter.render_html(ctx)
        self.assertIn("Wire · Oct 1", html)
        self.assertNotIn("Wire · Oct 2", html)
