"""
PDF Newsletter 

Page 1 is the header page: market backdrop, the big & concerning moves, and the whole
watchlist at a glance. After that, every stock gets its own page with a chart, its key
metrics, its alerts and signals, and the latest headlines.

Design decisions:
1. Built from the same `context` as the email (src/formatter.py), so the two always agree.
2. reportlab's "platypus" layout engine: we hand it a list of blocks (paragraphs, tables,
   images, page breaks) and it handles page flow. If the header page overflows because many
   stocks alerted, it simply continues on page 2.
3. reportlab's built-in Helvetica can't draw symbols like ▲ ▼, so the PDF uses colored
   bullets, +/- signs and words (Up / Down / Watch) instead. Nothing depends on color alone.
4. Pure pip installs (reportlab, matplotlib): no system libraries, so it runs the same on
   Windows/WSL and on GitHub's Linux servers.
"""

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from src.charts import price_volume_chart
from src.formatter import (DOWN_TEXT, HAIRLINE, INK, INK_2, MUTED, UP_TEXT, fmt_money, fmt_num,
                           fmt_pct, fmt_ratio, safe_url, session_labels)

TONE_PDF = {
    "down": {"color": DOWN_TEXT, "label": "DOWN"},
    "up": {"color": UP_TEXT, "label": "UP"},
    "caution": {"color": "#a86e00", "label": "WATCH"},
}
PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


def _styles():
    base = dict(fontName="Helvetica", textColor=colors.HexColor(INK), leading=13, fontSize=9.5)
    return {
        "kicker": ParagraphStyle("kicker", **{**base, "fontSize": 8, "textColor": colors.HexColor(INK_2)}),
        "title": ParagraphStyle("title", **{**base, "fontName": "Helvetica-Bold", "fontSize": 22, "leading": 26}),
        "sub": ParagraphStyle("sub", **{**base, "textColor": colors.HexColor(INK_2)}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Helvetica-Bold", "fontSize": 12.5, "leading": 16,
                                       "spaceBefore": 14, "spaceAfter": 6}),
        "h3": ParagraphStyle("h3", **{**base, "fontName": "Helvetica-Bold", "fontSize": 10, "spaceBefore": 8,
                                       "spaceAfter": 3}),
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle("small", **{**base, "fontSize": 8, "leading": 10.5, "textColor": colors.HexColor(INK_2)}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.5, "leading": 11}),
        "cell_r": ParagraphStyle("cell_r", **{**base, "fontSize": 8.5, "leading": 11, "alignment": TA_RIGHT}),
        "ticker": ParagraphStyle("ticker", **{**base, "fontName": "Helvetica-Bold", "fontSize": 24, "leading": 28}),
        "price": ParagraphStyle("price", **{**base, "fontName": "Helvetica-Bold", "fontSize": 18, "leading": 22,
                                             "alignment": TA_RIGHT}),
        "bullet": ParagraphStyle("bullet", **{**base, "leftIndent": 10, "bulletIndent": 0, "spaceAfter": 1.5}),
        # Hanging indent for lines that start with their own (colored) bullet character
        "hang": ParagraphStyle("hang", **{**base, "leftIndent": 10, "firstLineIndent": -10, "spaceAfter": 2}),
    }


def _pct_markup(value, decimals=1, bold=True):
    """A signed percentage in green/red. The sign (+/-) carries the meaning; color reinforces it."""
    if value is None:
        return "—"
    color = INK_2 if abs(value) < 0.05 else (UP_TEXT if value > 0 else DOWN_TEXT)
    text = fmt_pct(value, decimals=decimals)
    return f"<font color='{color}'>{'<b>' if bold else ''}{text}{'</b>' if bold else ''}</font>"


class NewsletterPDF:
    def __init__(self, formatter):
        # Reuse the email formatter's metric list so the PDF and email show the same numbers
        self.formatter = formatter
        self.s = _styles()

    def build(self, ctx: dict, pipeline_data: dict, path: str) -> str:
        doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=MARGIN, bottomMargin=MARGIN + 10,
                                title=f"{ctx['title']} — {ctx['date_label']}", author="Stock-Analysis-Report")
        story = self._header_page(ctx)
        for stock in ctx["stocks"]:
            story.append(PageBreak())
            story += self._stock_page(stock, ctx, pipeline_data.get(stock["symbol"], {}))

        def footer(canvas, _doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor(MUTED))
            canvas.drawString(MARGIN, MARGIN * 0.6,
                              f"{ctx['title']} · {ctx['date_label']} · Automated analysis, not investment advice")
            canvas.drawRightString(PAGE_W - MARGIN, MARGIN * 0.6, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        return path

    # ------------------------------------------------------------------ #
    # Page 1: the header page
    # ------------------------------------------------------------------ #
    def _header_page(self, ctx):
        s = self.s
        as_of = ctx["as_of"]
        as_of_text = f"Prices as of {as_of:%I:%M %p} ET".replace(" 0", " ") if as_of else "No pre-market trades yet"
        story = [
            Paragraph(escape(ctx["title"].upper()), s["kicker"]),
            Paragraph(escape(ctx["date_label"]), s["title"]),
            Paragraph(f"{as_of_text} · {len(ctx['stocks'])} stocks tracked · "
                      f"{len(ctx['ranked'])} need attention", s["sub"]),
        ]

        # Market backdrop: three stat tiles
        market = ctx["market"]
        tiles = []
        for symbol, name in (("SPY", "S&amp;P 500 (SPY)"), ("QQQ", "Nasdaq-100 (QQQ)"), ("^VIX", "VIX fear gauge")):
            data = market.get(symbol)
            if not data:
                continue
            if symbol == "^VIX":
                value = fmt_num(data.get("last_close"))
                change = data.get("last_session_pct")
                # Rising fear is bad news, so flip the color (but keep the true sign)
                color = INK_2 if change is None else (DOWN_TEXT if change > 0 else UP_TEXT)
                sub = f"<font color='{color}'><b>{fmt_pct(change)}</b></font> last session"
            else:
                value = fmt_money(data.get("premarket_price") or data.get("last_close"))
                sub = f"{_pct_markup(data.get('gap_pct'))} {ctx['labels']['long']}"
            tiles.append([Paragraph(name, s["small"]),
                          Paragraph(f"<b>{value}</b>", ParagraphStyle("v", parent=s["body"], fontSize=15, leading=19)),
                          Paragraph(sub, s["small"])])
        if tiles:
            story.append(Paragraph("Market backdrop", s["h2"]))
            table = Table([tiles], colWidths=[CONTENT_W / len(tiles)] * len(tiles))
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            story.append(table)

        # Big & concerning moves
        story.append(Paragraph("Big &amp; concerning moves", s["h2"]))
        if not ctx["ranked"]:
            story.append(Paragraph(f"Nothing crossed your thresholds this morning. All {len(ctx['stocks'])} "
                                   f"stocks are trading within their normal ranges.", s["body"]))
        for item in ctx["ranked"]:
            tone = TONE_PDF[item["tone"]]
            name = f" <font color='{INK_2}'>{escape(item.get('name') or '')}</font>" if item.get("name") else ""
            gap = item.get("gap_pct")
            head = Table([[Paragraph(f"<b>{escape(item['symbol'])}</b>{name}  "
                                     f"<font size='7.5' color='{tone['color']}'><b>{tone['label']}</b></font>", s["body"]),
                           Paragraph(f"{_pct_markup(gap)} {ctx['labels']['short']}" if gap is not None else "", s["cell_r"])]],
                         colWidths=[CONTENT_W * 0.72 - 12, CONTENT_W * 0.28])
            head.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                      ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            bullets = [Paragraph(escape(a["text"]), s["bullet"], bulletText="•") for a in item["alerts"]]
            block = Table([[[head] + bullets]], colWidths=[CONTENT_W])
            block.setStyle(TableStyle([
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(tone["color"])),
                ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story += [KeepTogether(block), Spacer(1, 6)]

        # Watchlist at a glance (the PDF has room for all eight columns)
        story.append(Paragraph("Watchlist at a glance", s["h2"]))
        header = ["Ticker", ctx["labels"]["col"], "Gap", "Last close", "Last day", "RSI", "Vol/avg", "vs 200d"]
        rows = [[Paragraph(f"<b>{h}</b>", s["cell"] if i == 0 else s["cell_r"]) for i, h in enumerate(header)]]
        for stock in ctx["stocks"]:
            m = stock["metrics"]
            rows.append([
                Paragraph(f"<b>{escape(stock['symbol'])}</b>", s["cell"]),
                Paragraph(fmt_money(m.get("premarket_price")), s["cell_r"]),
                Paragraph(_pct_markup(m.get("gap_pct")), s["cell_r"]),
                Paragraph(fmt_money(m.get("last_close")), s["cell_r"]),
                Paragraph(_pct_markup(m.get("last_session_pct"), bold=False), s["cell_r"]),
                Paragraph(fmt_num(m.get("rsi_14"), 0), s["cell_r"]),
                Paragraph(fmt_ratio(m.get("volume_ratio")), s["cell_r"]),
                Paragraph(_pct_markup(m.get("pct_vs_sma_200"), decimals=0, bold=False), s["cell_r"]),
            ])
        for err in ctx["errors"]:
            rows.append([Paragraph(f"<b>{escape(err['symbol'])}</b>", s["cell"]),
                         Paragraph(escape(err["error"]), s["small"])] + [""] * 6)
        widths = [0.9, 0.95, 0.85, 1.0, 0.85, 0.6, 0.75, 0.75]
        scale = CONTENT_W / (sum(widths) * inch)
        table = Table(rows, colWidths=[w * inch * scale for w in widths], repeatRows=1)
        style = [("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        for r in range(len(ctx["stocks"]) + 1, len(rows)):
            style.append(("SPAN", (1, r), (-1, r)))
        table.setStyle(TableStyle(style))
        story.append(table)

        # Market movers outside the watchlist
        movers = ctx["movers"]
        if movers and any(movers.values()):
            story.append(Paragraph("Outside your watchlist", s["h2"]))
            story.append(Paragraph("Biggest US movers in the last regular session (Yahoo Finance screens)", s["small"]))
            cols = []
            for key, title in (("day_gainers", "Top gainers"), ("day_losers", "Top losers"), ("most_actives", "Most active")):
                lines = [Paragraph(f"<b>{title}</b>", s["cell"])]
                for q in movers.get(key, []):
                    lines.append(Paragraph(f"{escape(q.get('symbol') or '')}  {_pct_markup(q.get('change_pct'), bold=False)}",
                                           s["cell"]))
                cols.append(lines)
            table = Table([cols], colWidths=[CONTENT_W / 3] * 3)
            table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                       ("TOPPADDING", (0, 0), (-1, -1), 6)]))
            story.append(table)

        # What each short-interest / options level means (same table as the email)
        story.append(Paragraph("Reading short interest and options", s["h2"]))
        story.append(Paragraph(escape(self.formatter.positioning_note()), s["small"]))
        for title, rows in self.formatter.positioning_guide():
            table = Table([[Paragraph(escape(r), s["small"]), Paragraph(f"<b>{escape(name)}</b>", s["small"]),
                            Paragraph(escape(meaning), s["small"])] for r, name, meaning in rows],
                          colWidths=[CONTENT_W * 0.13, CONTENT_W * 0.13, CONTENT_W * 0.74])
            table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
            story.append(KeepTogether([Paragraph(f"<b>{escape(title)}</b>", s["cell"]), table]))

        t = ctx["thresholds"]
        story += [Spacer(1, 14), Paragraph(
            f"How alerts work: a pre-market gap of ±{t.get('premarket_gap_pct', 3.0):g}% or a last-session move of "
            f"±{t.get('price_change_pct', 5.0):g}% (when it's at least a normal-sized move for that stock), or any move of "
            f"{t.get('unusual_move_sigma', 2.0):g}x the stock's typical daily move; volume at "
            f"{t.get('volume_surge_multiplier', 2.0):g}x its 20-day average; RSI outside "
            f"{t.get('rsi_oversold', 30)}–{t.get('rsi_overbought', 70)}; a close through the 50- or 200-day average; "
            f"a 52-week high or low; earnings within {t.get('earnings_lookahead_days', 7)} days. Data: Finnhub and "
            f"Yahoo Finance.", s["small"])]
        return story

    # ------------------------------------------------------------------ #
    # One page per stock
    # ------------------------------------------------------------------ #
    def _stock_page(self, stock, ctx, raw):
        s = self.s
        m = stock["metrics"]
        name_bits = " · ".join(escape(b) for b in (m.get("name"), m.get("industry")) if b)
        price = m.get("premarket_price") or m.get("last_close")
        gap = m.get("gap_pct")
        top = Table([[
            [Paragraph(escape(stock["symbol"]), s["ticker"]), Paragraph(name_bits or "&nbsp;", s["sub"])],
            [Paragraph(fmt_money(price), s["price"]),
             Paragraph(f"{_pct_markup(gap)} {session_labels(m.get('premarket_session'))['long']}"
                       if gap is not None else "last close", s["cell_r"])],
        ]], colWidths=[CONTENT_W * 0.65, CONTENT_W * 0.35])
        top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story = [top, Spacer(1, 8)]

        candles = raw.get("candles")
        if candles is not None and not candles.empty:
            png = price_volume_chart(candles, m)
            story.append(Image(png, width=CONTENT_W, height=CONTENT_W * 3.3 / 7.3, kind="proportional"))

        # Left column: every key metric. Right column: alerts + signals.
        metric_rows = [[Paragraph(escape(k), s["small"]), Paragraph(escape(v), s["cell_r"])]
                       for k, v in self.formatter._metric_rows(stock)]
        metrics_table = Table(metric_rows, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.24])
        metrics_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
                                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        left = [Paragraph("Key metrics", s["h3"]), metrics_table]

        right = [Paragraph("Alerts", s["h3"])]
        if stock["alerts"]:
            for a in stock["alerts"]:
                color = TONE_PDF[a["direction"]]["color"]
                # bulletText can't hold markup, so the colored bullet is written into the text itself
                right.append(Paragraph(f"<font color='{color}'>•</font>&nbsp;&nbsp;{escape(a['text'])}", s["hang"]))
        else:
            right.append(Paragraph("No alerts: everything within normal ranges.", s["small"]))
        right.append(Paragraph("Signals", s["h3"]))
        for t in stock["triggers"]:
            right.append(Paragraph(escape(t), ParagraphStyle("sig", parent=s["bullet"], fontSize=8.5, leading=11,
                                                             textColor=colors.HexColor(INK_2)), bulletText="•"))

        columns = Table([[left, right]], colWidths=[CONTENT_W * 0.48, CONTENT_W * 0.52])
        columns.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                     ("RIGHTPADDING", (0, 0), (0, -1), 14), ("RIGHTPADDING", (1, 0), (1, -1), 0)]))
        story += [Spacer(1, 4), columns]

        news = stock.get("top_news", [])[:ctx["news_per_stock"]]
        if news:
            sd = stock.get("sentiment_detail", {})
            story.append(Paragraph(
                f"Headlines <font name='Helvetica' size='8.5' color='{INK_2}'>· sentiment {escape(stock['sentiment'])} "
                f"({sd.get('positive', 0)} positive, {sd.get('negative', 0)} negative, "
                f"{sd.get('neutral', 0)} neutral)</font>", s["h3"]))
            for article in news:
                headline = escape(article.get("headline") or "Untitled")
                url = safe_url(article.get("url"))
                link = f"<link href='{url}' color='{INK}'><u>{headline}</u></link>" if url else headline
                source = escape(article.get("source") or "")
                story.append(Paragraph(f"{link} <font size='7.5' color='{MUTED}'>{('— ' + source) if source else ''}</font>",
                                       ParagraphStyle("news", parent=s["bullet"], fontSize=9), bulletText="•"))
        return story
