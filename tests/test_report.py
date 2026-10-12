"""Tests for the email formatter, the PDF, and the dispatcher (no network, nothing is sent)."""
import os
import smtplib
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.analyzer import QuantitativeAnalyzer
from src.dispatcher import EmailDispatcher, parse_recipients
from src.formatter import ReportFormatter, fmt_market_cap, fmt_pct, fmt_volume
from tests.fixtures import sample_earnings, sample_market, sample_movers, sample_pipeline
from tests.test_analyzer import THRESHOLDS


def build_report():
    analyzer = QuantitativeAnalyzer(THRESHOLDS)
    pipeline = sample_pipeline()
    market = analyzer.analyze_market(sample_market())
    results = analyzer.analyze(pipeline, market, sample_earnings())
    ranked = analyzer.rank_stocks(results)
    formatter = ReportFormatter({"title": "Pre-Market Brief"}, THRESHOLDS)
    when = pd.Timestamp("2026-10-01 07:53", tz="America/New_York").to_pydatetime()
    return formatter, formatter.build_context(results, ranked, market, sample_movers(), when), pipeline


class FormatterTests(unittest.TestCase):
    def setUp(self):
        self.formatter, self.ctx, self.pipeline = build_report()

    def test_outside_text_is_escaped_and_bad_links_dropped(self):
        html = self.formatter.render_html(self.ctx)
        self.assertIn("&lt;b&gt;AAA&lt;/b&gt;", html)      # headline markup shown as text
        self.assertNotIn("<b>AAA</b>", html)
        self.assertNotIn("javascript:", html)              # unsafe link removed
        self.assertIn("href='https://example.com/news/1'", html)

    def test_header_lists_flagged_stocks_and_errors(self):
        html = self.formatter.render_html(self.ctx)
        self.assertIn("Big &amp; concerning moves", html)
        self.assertIn("No price history returned", html)    # ZZZ shown, not silently dropped
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
        guide = dict(ReportFormatter({}, {**THRESHOLDS, "short_interest_levels_pct": [4, 8, 15]}).positioning_guide())
        ranges = [(r, name) for r, name, _ in guide["Short interest (% of float)"]]
        self.assertEqual(ranges, [("under 4%", "Low"), ("4%–8%", "Moderate"), ("8%–15%", "High"),
                                  ("15% and up", "Very high")])
        self.assertEqual(len(guide["Put/call open interest"]), 5)
        text = self.formatter.render_text(self.ctx)
        self.assertIn("READING SHORT INTEREST AND OPTIONS", text)
        self.assertIn("not how fast the business is growing", text)
        self.assertIn("Reading short interest and options", self.formatter.render_html(self.ctx))

    def test_number_formatting(self):
        self.assertEqual(fmt_pct(4.56), "+4.6%")
        self.assertEqual(fmt_pct(None), "—")
        self.assertEqual(fmt_volume(34_400_000), "34.4M")
        self.assertEqual(fmt_market_cap(2_100_000), "$2.1T")


class PdfTests(unittest.TestCase):
    def test_pdf_has_header_page_stock_pages_and_guide_page(self):
        from pypdf import PdfReader
        from src.pdf_report import NewsletterPDF
        formatter, ctx, pipeline = build_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = NewsletterPDF(formatter).build(ctx, pipeline, os.path.join(tmp, "report.pdf"))
            reader = PdfReader(path)
            self.assertGreaterEqual(len(reader.pages), 2 + len(ctx["stocks"]))
            first_page = reader.pages[0].extract_text()
            self.assertIn("Big & concerning moves", first_page)
            self.assertNotIn("Reading short interest and options", first_page)
            self.assertIn(ctx["stocks"][-1]["symbol"], reader.pages[-2].extract_text())
            self.assertIn("Reading short interest and options", reader.pages[-1].extract_text())

    def test_preview_shows_first_two_pages(self):
        from PIL import Image
        from src.pdf_report import NewsletterPDF
        from src.preview import render_preview
        formatter, ctx, pipeline = build_report()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = NewsletterPDF(formatter).build(ctx, pipeline, os.path.join(tmp, "report.pdf"))
            png = render_preview(pdf, os.path.join(tmp, "preview.png"))
            with Image.open(png) as im:
                # A letter page is 11/8.5 = 1.29x as tall as wide; two pages plus the gap is just over 2.59x
                self.assertAlmostEqual(im.height / im.width, 2 * 11 / 8.5, delta=0.05)


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
        self.assertNotIn("a@x.com", raw)          # no recipient address anywhere in the headers/body
        self.assertIsNone(msg["Bcc"])
        types = [part.get_content_type() for part in msg.walk()]
        self.assertIn("text/plain", types)
        self.assertIn("text/html", types)
        self.assertIn("application/pdf", types)

    def test_send_puts_everyone_in_the_envelope(self):
        sent = {}

        class FakeSMTP:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def login(self, user, password): sent["login"] = (user, password)
            def send_message(self, msg, from_addr=None, to_addrs=None):
                sent["to"] = to_addrs
                return {}

        dispatcher = EmailDispatcher("me@gmail.com", "pw", ["a@x.com", "me@gmail.com"])
        with mock.patch.object(smtplib, "SMTP_SSL", FakeSMTP):
            self.assertTrue(dispatcher.send("s", "<p>h</p>", "t"))
        self.assertEqual(sent["to"], ["me@gmail.com", "a@x.com"])  # sender once, no duplicates

    def test_send_reports_bad_login(self):
        class FailingSMTP:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def login(self, user, password): raise smtplib.SMTPAuthenticationError(535, b"bad")

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
        ctx = formatter.build_context({s["symbol"]: {k: v for k, v in s.items() if k != "symbol"} for s in ctx["stocks"]},
                                      ctx["ranked"], ctx["market"], ctx["movers"], ctx["generated_at"])
        html = formatter.render_html(ctx)
        self.assertIn("After-hrs", html)
        self.assertIn("after-hours price vs. today", html)
        self.assertNotIn(">Pre-mkt<", html)
