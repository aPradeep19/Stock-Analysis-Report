"""
The PDF newsletter.

Layout, fixed so readers always know where to look:
  page 1     front page: title, market backdrop, big & concerning moves (also the README preview)
  page 2     watchlist at a glance, yesterday's scorecard, market movers
  one page per stock: chart, key metrics, alerts and signals, headlines
  last page  how alerts work, and what each positioning level means

Every page is fitted before it's handed to reportlab: each section offers versions from richest
to most compact, they're measured with wrap(), and the richest set that fits is used. Pages never
spill, and text is never made smaller to make room.

- Built from the same context as the email (src/formatter.py), so the two always agree.
- The built-in Helvetica has no glyphs for ▲ ▼, so the PDF relies on signs, colored bullets
  and words (UP / DOWN / WATCH) instead of the email's arrows.
- reportlab and matplotlib are pure pip installs with no system libraries, so the PDF renders
  the same locally and on GitHub's Linux runners.
"""

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src import scorecard as scorecard_text
from src.charts import price_volume_chart
from src.formatter import (
    DOWN_TEXT,
    HAIRLINE,
    INK,
    INK_2,
    LINK,
    MUTED,
    SENTIMENT_NOTE,
    UP_TEXT,
    WATCHLIST_ANCHOR,
    anchor_id,
    fmt_bp,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_ratio,
    is_strongest,
    mood_line,
    safe_url,
    sentiment_summary,
    session_labels,
    strongest_note,
    tone_mark,
)

TONE_PDF = {
    "down": {"color": DOWN_TEXT, "label": "DOWN"},
    "up": {"color": UP_TEXT, "label": "UP"},
    "caution": {"color": "#a86e00", "label": "WATCH"},
}

PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch
TOP_MARGIN = MARGIN
BOTTOM_MARGIN = MARGIN + 10  # leaves room for the footer line
CONTENT_W = PAGE_W - 2 * MARGIN
# build() uses an unpadded frame exactly CONTENT_W wide, so what's measured here is what gets laid
# out, and paragraphs line up with full-width tables. 2pt of slack keeps rounding from pushing an
# exactly-fitting page onto the next one.
FRAME_H = PAGE_H - TOP_MARGIN - BOTTOM_MARGIN - 2

CHART_ASPECT = 3.3 / 7.3  # height / width of price_volume_chart()'s native figure
CHART_MIN_SCALE = 0.6  # the chart shrinks to this before any signal is trimmed
CHART_LAST_RESORT_SCALE = 0.45  # only once every signal is gone; below this the chart is dropped
HEADLINE_MAX_LINES = 2

# Signals whose figure already appears in Key metrics or Alerts, so the PDF page skips them.
REDUNDANT_SIGNALS = frozenset(
    {
        "premarket",
        "last_session",
        "volume",
        "short_interest",
        "open_interest",
        "calls_expiring",
        "ownership",
        "net_debt",
    }
)
# When a page is still too tall, signals go in this order, least informative first. Kinds not
# listed (including untagged ones) are treated as the most valuable and go last.
SIGNAL_TRIM_ORDER = ("sma_20", "rsi", "sma_200", "sma_50", "valuation", "cash_runway")

# wrap() needs a canvas for some flowables (KeepTogether, tables of paragraphs); measuring never
# draws, so a throwaway in-memory one is enough.
_MEASURE_CANVAS = Canvas(io.BytesIO())


def _styles():
    base = dict(fontName="Helvetica", textColor=colors.HexColor(INK), leading=13, fontSize=9.5)
    return {
        "kicker": ParagraphStyle("kicker", **{**base, "fontSize": 8, "textColor": colors.HexColor(INK_2)}),
        "title": ParagraphStyle(
            "title", **{**base, "fontName": "Helvetica-Bold", "fontSize": 22, "leading": 26}
        ),
        "sub": ParagraphStyle("sub", **{**base, "textColor": colors.HexColor(INK_2)}),
        "h2": ParagraphStyle(
            "h2",
            **{
                **base,
                "fontName": "Helvetica-Bold",
                "fontSize": 12.5,
                "leading": 16,
                "spaceBefore": 14,
                "spaceAfter": 6,
            },
        ),
        "h3": ParagraphStyle(
            "h3", **{**base, "fontName": "Helvetica-Bold", "fontSize": 10, "spaceBefore": 8, "spaceAfter": 3}
        ),
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle(
            "small", **{**base, "fontSize": 8, "leading": 10.5, "textColor": colors.HexColor(INK_2)}
        ),
        "small_r": ParagraphStyle(
            "small_r",
            **{
                **base,
                "fontSize": 8,
                "leading": 10.5,
                "textColor": colors.HexColor(INK_2),
                "alignment": TA_RIGHT,
            },
        ),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.5, "leading": 11}),
        "cell_r": ParagraphStyle("cell_r", **{**base, "fontSize": 8.5, "leading": 11, "alignment": TA_RIGHT}),
        "ticker": ParagraphStyle(
            "ticker", **{**base, "fontName": "Helvetica-Bold", "fontSize": 24, "leading": 28}
        ),
        "price": ParagraphStyle(
            "price",
            **{**base, "fontName": "Helvetica-Bold", "fontSize": 18, "leading": 22, "alignment": TA_RIGHT},
        ),
        "bullet": ParagraphStyle(
            "bullet", **{**base, "leftIndent": 10, "bulletIndent": 0, "spaceAfter": 1.5}
        ),
        "note": ParagraphStyle(
            "note",
            **{**base, "fontSize": 8.5, "leading": 11, "textColor": colors.HexColor(INK_2), "spaceBefore": 4},
        ),
        # The 'why it's moving' text is something to read, not a footnote: full ink, near body size.
        "why": ParagraphStyle("why", **{**base, "fontSize": 9, "leading": 12, "spaceBefore": 3}),
        # Hanging indent for lines that carry their own colored bullet in the text.
        "hang": ParagraphStyle("hang", **{**base, "leftIndent": 10, "firstLineIndent": -10, "spaceAfter": 2}),
    }


def _ticker_link(symbol):
    """Bold ticker that jumps to the stock's own page (an internal PDF link)."""
    return f'<a href="#{anchor_id(symbol)}" color="{LINK}"><b>{escape(symbol)}</b></a>'


def _pct_markup(value, decimals=1, bold=True):
    """Signed percentage as reportlab markup, green/red like the email's pct_color()."""
    if value is None:
        return "—"
    color = INK_2 if abs(value) < 0.05 else (UP_TEXT if value > 0 else DOWN_TEXT)
    text = fmt_pct(value, decimals=decimals)
    return f"<font color='{color}'>{'<b>' if bold else ''}{text}{'</b>' if bold else ''}</font>"


# ---------------------------------------------------------------------- #
# Measuring and fitting
# ---------------------------------------------------------------------- #
def _height(flowables, width=CONTENT_W):
    """
    Height a list of flowables takes in a frame `width` wide, spacing included.

    Slightly pessimistic on purpose: a frame drops the first flowable's spaceBefore at the top
    of a page, which this counts anyway.
    """
    total = 0.0
    for flowable in flowables:
        # KeepTogether reports a huge sentinel height from wrap(); measure what it holds instead.
        parts = flowable._content if isinstance(flowable, KeepTogether) else [flowable]
        for part in parts:
            _, h = part.wrapOn(_MEASURE_CANVAS, width, FRAME_H * 10)
            total += h + part.getSpaceBefore() + part.getSpaceAfter()
    return total


def _fit_sections(sections, budget=FRAME_H):
    """
    Lays sections out top to bottom within `budget` points.

    Each section is a list of versions (each a list of flowables), richest first. A section gets
    the richest version that fits the space left after the sections above it, minus the room
    the sections below need at their most compact, so an early section can't crowd out a later
    one. New sections only need to describe their versions; nothing here counts rows by hand.
    """
    smallest = [_height(versions[-1]) for _, versions in sections]
    story = []
    remaining = budget
    for i, (name, versions) in enumerate(sections):
        reserved = sum(smallest[i + 1 :])
        for version in versions:
            height = _height(version)
            if height <= remaining - reserved:
                break
        else:
            print(f"    [WARN] PDF: '{name}' doesn't fit its page even at its most compact; it will spill.")
        story += version
        remaining -= height
    return story


def _line_count(paragraph, width):
    paragraph.wrapOn(_MEASURE_CANVAS, width, FRAME_H * 10)
    return len(paragraph.blPara.lines)


def _clip(text, length):
    """`text` cut to about `length` characters at a word boundary when one is close, plus '…'."""
    cut = text[:length].rstrip()
    space = cut.rfind(" ")
    if 0 < space and length - space < 15:
        cut = cut[:space]
    return cut.rstrip(" ,;:-–—") + "…"


def _fit_lines(make, text, width, max_lines):
    """
    make(text) -> Paragraph, with `text` (plain, unescaped) shortened until the paragraph is at
    most `max_lines` lines at `width`. Text is cut, never shrunk.
    """
    paragraph = make(text)
    if _line_count(paragraph, width) <= max_lines:
        return paragraph
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _line_count(make(_clip(text, mid)), width) <= max_lines:
            lo = mid
        else:
            hi = mid - 1
    return make(_clip(text, lo))


def _page_signals(stock):
    """(kind, text) for the signals worth showing on the PDF page, in the analyzer's order."""
    signals = stock.get("signals")
    if signals is None:  # results built without kinds: show everything, trim it last
        signals = [{"text": text, "kind": None} for text in stock.get("triggers", [])]
    return [(s["kind"], s["text"]) for s in signals if s["kind"] not in REDUNDANT_SIGNALS]


def _signal_trims(signals):
    """Yields the signal list, then shorter and shorter versions of it, down to empty."""
    yield list(signals)
    rank = {kind: i for i, kind in enumerate(SIGNAL_TRIM_ORDER)}
    order = sorted(range(len(signals)), key=lambda i: rank.get(signals[i][0], len(rank)))
    dropped = set()
    for index in order:
        dropped.add(index)
        yield [s for i, s in enumerate(signals) if i not in dropped]


def _tight(table, **padding):
    """Applies zero padding (or the given overrides) plus top alignment to a layout table."""
    pad = {"LEFTPADDING": 0, "RIGHTPADDING": 0, "TOPPADDING": 0, "BOTTOMPADDING": 0, **padding}
    table.setStyle(
        TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")] + [(k, (0, 0), (-1, -1), v) for k, v in pad.items()])
    )
    return table


class NewsletterPDF:
    def __init__(self, formatter):
        # The email formatter supplies the metric rows and positioning guide, so the PDF can't
        # show different numbers or wording.
        self.formatter = formatter
        self.s = _styles()

    def build(self, ctx: dict, pipeline_data: dict, path: str) -> str:
        """Writes the PDF to `path` and returns it. `pipeline_data` supplies raw candles for charts."""
        doc = BaseDocTemplate(
            path,
            pagesize=letter,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title=f"{ctx['title']} — {ctx['date_label']}",
            author="Stock-Analysis-Report",
        )
        story = self._front_page(ctx) + [PageBreak()] + self._overview_page(ctx)
        for stock in ctx["stocks"]:
            story.append(PageBreak())
            story += self._stock_page(stock, ctx, pipeline_data.get(stock["symbol"], {}))
        story.append(PageBreak())
        story += self._guide_page(ctx)

        def footer(canvas, _doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor(MUTED))
            canvas.drawString(
                MARGIN,
                MARGIN * 0.6,
                f"{ctx['title']} · {ctx['date_label']} · Automated analysis, not investment advice",
            )
            canvas.drawRightString(PAGE_W - MARGIN, MARGIN * 0.6, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        # No frame padding (reportlab's default is 6pt a side); see FRAME_H.
        frame = Frame(
            MARGIN,
            BOTTOM_MARGIN,
            CONTENT_W,
            PAGE_H - TOP_MARGIN - BOTTOM_MARGIN,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=footer)])
        doc.build(story)
        return path

    # ------------------------------------------------------------------ #
    # Page 1: front page (src/preview.py turns it into the README image)
    # ------------------------------------------------------------------ #
    def _front_page(self, ctx):
        return _fit_sections(
            [
                ("title", [self._masthead(ctx)]),
                ("market backdrop", [self._backdrop(ctx)]),
                ("big & concerning moves", self._ranked_versions(ctx)),
            ]
        )

    def _masthead(self, ctx):
        s = self.s
        as_of = ctx["as_of"]
        as_of_text = (
            f"Prices as of {as_of:%I:%M %p} ET".replace(" 0", " ") if as_of else "No pre-market trades yet"
        )
        return [
            # Plain-text fields still go through escape(): reportlab parses Paragraph text as markup.
            Paragraph(escape(ctx["title"].upper()), s["kicker"]),
            Paragraph(escape(ctx["date_label"]), s["title"]),
            Paragraph(
                f"{as_of_text} · {len(ctx['stocks'])} stocks tracked · {len(ctx['ranked'])} need attention",
                s["sub"],
            ),
        ]

    def _backdrop(self, ctx):
        s = self.s
        market = ctx["market"]
        tiles = []
        for symbol, name in (
            ("SPY", "S&amp;P 500 (SPY)"),
            ("QQQ", "Nasdaq-100 (QQQ)"),
            ("^VIX", "VIX fear gauge"),
            ("^TNX", "10-yr Treasury"),
        ):
            data = market.get(symbol)
            if not data:
                continue
            if symbol == "^TNX":
                value = fmt_pct(data.get("yield_pct"), signed=False, decimals=2)
                # Neutral ink: a rising yield isn't simply good or bad news the way a price move is.
                sub = f"<b>{fmt_bp(data.get('last_session_bp'))}</b> last session"
            elif symbol == "^VIX":
                value = fmt_num(data.get("last_close"))
                change = data.get("last_session_pct")
                # A rising VIX means rising fear, so color it like a loss (the sign stays true).
                color = INK_2 if change is None else (DOWN_TEXT if change > 0 else UP_TEXT)
                sub = f"<font color='{color}'><b>{fmt_pct(change)}</b></font> last session"
            else:
                value = fmt_money(data.get("premarket_price") or data.get("last_close"))
                sub = f"{_pct_markup(data.get('gap_pct'))} {ctx['labels']['long']}"
            tiles.append(
                [
                    Paragraph(name, s["small"]),
                    Paragraph(
                        f"<b>{value}</b>", ParagraphStyle("v", parent=s["body"], fontSize=15, leading=19)
                    ),
                    Paragraph(sub, s["small"]),
                ]
            )
        notes = [
            Paragraph(f"<font color='{INK}'><b>{label}:</b></font> {escape(text)}", s["note"])
            for label, text in (
                ("Today (Central)", self.formatter.releases_line(ctx)),
                ("Earnings this week", self.formatter.earnings_week_line(ctx)),
                ("News tone (FinBERT)", mood_line(ctx.get("news_mood"))),
            )
            if text
        ]
        if not tiles and not notes:
            return []
        story = [Paragraph("Market backdrop", s["h2"])]
        if tiles:
            table = Table([tiles], colWidths=[CONTENT_W / len(tiles)] * len(tiles))
            table.setStyle(
                TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])
            )
            story.append(table)
        return story + notes

    def _why(self, item, width, max_lines, label="Why:", style="why"):
        """The 'Why: …  — Sources' text, cut to `max_lines`, or None when there isn't one."""
        parts = self.formatter.why_parts(item)
        if not parts:
            return None
        text, source = parts
        prefix = f"<b>{label}</b> " if label else ""
        suffix = f" <font size='7.5' color='{MUTED}'>— {escape(source)}</font>" if source else ""
        return _fit_lines(
            lambda t: Paragraph(f"{prefix}{escape(t)}{suffix}", self.s[style]),
            text,
            width,
            max_lines,
        )

    def _ranked_versions(self, ctx):
        """
        Versions of "Big & concerning moves": the first k ranked stocks get their full alert list
        and the rest one line each (with up to two lines of 'why'), for k = all, all - 1, ..., 0.
        The last resort cuts every 'why' to one line. Every ranked stock appears in every
        version; only the detail shrinks, and the most urgent stocks keep theirs longest.
        """
        s = self.s
        heading = Paragraph("Big &amp; concerning moves", s["h2"])
        if not ctx["ranked"]:
            note = Paragraph(
                f"Nothing crossed your thresholds this morning. All {len(ctx['stocks'])} "
                f"stocks are trading within their normal ranges.",
                s["body"],
            )
            return [[heading, note]]

        full = [[KeepTogether(self._ranked_block(item, ctx)), Spacer(1, 6)] for item in ctx["ranked"]]
        short = [[self._ranked_line(item, ctx), Spacer(1, 4)] for item in ctx["ranked"]]
        versions = []
        for k in range(len(full), -1, -1):
            version = [heading]
            for i in range(len(full)):
                version += full[i] if i < k else short[i]
            versions.append(version)
        shortest = [[self._ranked_line(item, ctx, why_lines=1), Spacer(1, 4)] for item in ctx["ranked"]]
        versions.append([heading] + [flowable for row in shortest for flowable in row])
        return versions

    def _ranked_frame(self, content, tone, top=4, bottom=4):
        """One-cell table that draws a stock's colored rule down its left edge."""
        block = Table([[content]], colWidths=[CONTENT_W])
        block.setStyle(
            TableStyle(
                [
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(tone["color"])),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), top),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), bottom),
                ]
            )
        )
        return block

    def _ranked_block(self, item, ctx):
        s = self.s
        tone = TONE_PDF[item["tone"]]
        inner_w = CONTENT_W - 16  # the frame's 10pt left and 6pt right padding
        name = f" <font color='{INK_2}'>{escape(item.get('name') or '')}</font>" if item.get("name") else ""
        gap = item.get("gap_pct")
        head = Table(
            [
                [
                    Paragraph(
                        f"{_ticker_link(item['symbol'])}{name}  "
                        f"<font size='7.5' color='{tone['color']}'><b>{tone['label']}</b></font>",
                        s["body"],
                    ),
                    Paragraph(
                        f"{_pct_markup(gap)} {ctx['labels']['short']}" if gap is not None else "",
                        s["cell_r"],
                    ),
                ]
            ],
            colWidths=[inner_w * 0.72, inner_w * 0.28],
        )
        _tight(head, BOTTOMPADDING=2)
        bullets = [Paragraph(escape(a["text"]), s["bullet"], bulletText="•") for a in item["alerts"]]
        why = self._why(item, inner_w, 3)
        return self._ranked_frame([head] + bullets + ([why] if why else []), tone)

    def _ranked_line(self, item, ctx, why_lines=2):
        """The compact form: ticker, tone, gap and top alert on one line, then the 'why'."""
        s = self.s
        tone = TONE_PDF[item["tone"]]
        gap = item.get("gap_pct")
        gap_text = f"{_pct_markup(gap)} {ctx['labels']['short']}  ·  " if gap is not None else ""
        prefix = (
            f"{_ticker_link(item['symbol'])}  "
            f"<font size='7.5' color='{tone['color']}'><b>{tone['label']}</b></font>  {gap_text}"
        )
        line = _fit_lines(
            lambda text: Paragraph(prefix + escape(text), s["body"]),
            item["alerts"][0]["text"],
            CONTENT_W - 16,
            1,
        )
        why = self._why(item, CONTENT_W - 16, why_lines)
        return self._ranked_frame([line, why] if why else line, tone, top=3, bottom=3)

    # ------------------------------------------------------------------ #
    # Page 2: the whole watchlist, yesterday's scorecard and the wider market
    # ------------------------------------------------------------------ #
    def _overview_page(self, ctx):
        # Sections fill the page in this order; each falls back to a more compact version when
        # the ones above leave too little room. New sections slot in as (name, versions) pairs.
        return _fit_sections(
            [
                ("watchlist at a glance", [self._watchlist(ctx)]),
                ("yesterday's scorecard", self._scorecard_versions(ctx)),
                ("outside your watchlist", self._movers_versions(ctx)),
            ]
        )

    def _pdf_table(self, header, rows, widths, left_columns=1):
        """A plain data table in the report's style; `widths` are relative and fill the page."""
        s = self.s
        scale = CONTENT_W / sum(widths)

        def cell(text, i, bold=False):
            style = s["cell"] if i < left_columns else s["cell_r"]
            return Paragraph(f"<b>{escape(text)}</b>" if bold else escape(text), style)

        data = [[cell(h, i, bold=True) for i, h in enumerate(header)]]
        data += [[cell(c, i) for i, c in enumerate(row)] for row in rows]
        table = Table(data, colWidths=[w * scale for w in widths], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return table

    def _scorecard_versions(self, ctx):
        """
        Yesterday's scorecard, richest first: every scored alert plus hit rates, then fewer
        rows (the most severe alerts are kept), then without hit rates, then the summary alone.
        """
        s = self.s
        card = ctx["scorecard"]
        heading = Paragraph("Yesterday's scorecard", s["h2"])
        summary = Paragraph(escape(scorecard_text.summary_line(card)), s["body"])
        if card.get("status") != "ok":
            return [[heading, summary]]

        footnote = Paragraph(escape(scorecard_text.footnote(card)), s["small"])
        hits_line = scorecard_text.sentiment_hits_line(card)
        sentiment_hits = [Spacer(1, 4), Paragraph(escape(hits_line), s["cell"])] if hits_line else []
        earnings = [
            Paragraph(escape(scorecard_text.earnings_line(item)), s["cell"]) for item in card["earnings"]
        ]
        header, rows = self.formatter.scorecard_table(card)
        hit_rates = []
        if card["hit_rates"]:
            hit_header, hit_rows = self.formatter.hit_rate_table(card)
            hit_rates = [
                Paragraph(f"Hit rates over {card['days_logged']} report days", s["h3"]),
                self._pdf_table(hit_header, hit_rows, [1.6, 0.7, 0.7, 1.3, 1.3]),
            ]

        def version(row_count, with_hit_rates):
            parts = [heading, summary]
            if rows and row_count:
                parts += [
                    Spacer(1, 4),
                    self._pdf_table(header, rows[:row_count], [0.8, 1.8, 1.0, 1.0, 0.6, 1.2, 1.1], 2),
                ]
                if row_count < len(rows):
                    parts.append(Paragraph(f"…and {len(rows) - row_count} more", s["small"]))
            parts += (
                earnings + (hit_rates if with_hit_rates else []) + sentiment_hits + [Spacer(1, 4), footnote]
            )
            return parts

        versions = [version(n, True) for n in range(len(rows), min(len(rows), 3) - 1, -1)]
        versions += [version(min(len(rows), 3), False), [heading, summary]]
        return versions

    def _watchlist(self, ctx):
        s = self.s
        # The page is wide enough for two more columns than the email's phone-sized table.
        header = [
            "Ticker",
            ctx["labels"]["col"],
            "Gap",
            "Last close",
            "Last day",
            "RSI",
            "Vol/avg",
            "vs 200d",
        ]
        rows = [
            [Paragraph(f"<b>{h}</b>", s["cell"] if i == 0 else s["cell_r"]) for i, h in enumerate(header)]
        ]
        for stock in ctx["stocks"]:
            m = stock["metrics"]
            rows.append(
                [
                    Paragraph(_ticker_link(stock["symbol"]), s["cell"]),
                    Paragraph(fmt_money(m.get("premarket_price")), s["cell_r"]),
                    Paragraph(_pct_markup(m.get("gap_pct")), s["cell_r"]),
                    Paragraph(fmt_money(m.get("last_close")), s["cell_r"]),
                    Paragraph(_pct_markup(m.get("last_session_pct"), bold=False), s["cell_r"]),
                    Paragraph(fmt_num(m.get("rsi_14"), 0), s["cell_r"]),
                    Paragraph(fmt_ratio(m.get("volume_ratio")), s["cell_r"]),
                    Paragraph(_pct_markup(m.get("pct_vs_sma_200"), decimals=0, bold=False), s["cell_r"]),
                ]
            )
        for err in ctx["errors"]:
            rows.append(
                [
                    Paragraph(f"<b>{escape(err['symbol'])}</b>", s["cell"]),
                    Paragraph(escape(err["error"]), s["small"]),
                ]
                + [""] * 6
            )
        # Relative column widths, scaled to fill the content width.
        widths = [0.9, 0.95, 0.85, 1.0, 0.85, 0.6, 0.75, 0.75]
        scale = CONTENT_W / (sum(widths) * inch)
        table = Table(rows, colWidths=[w * inch * scale for w in widths], repeatRows=1)
        style = [
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
        # Error rows come after the header and stock rows; their message spans the data columns.
        for r in range(len(ctx["stocks"]) + 1, len(rows)):
            style.append(("SPAN", (1, r), (-1, r)))
        table.setStyle(TableStyle(style))
        # Named anchor: every stock page's "Back to watchlist" link lands here.
        return [Paragraph(f'<a name="{WATCHLIST_ANCHOR}"/>Watchlist at a glance', s["h2"]), table]

    def _movers_versions(self, ctx):
        """Market movers with every row, then one fewer row per screen, down to one."""
        s = self.s
        movers = ctx["movers"]
        if not movers or not any(movers.values()):
            return [[]]
        screens = (
            ("day_gainers", "Top gainers"),
            ("day_losers", "Top losers"),
            ("most_actives", "Most active"),
        )
        most = max(len(movers.get(key, [])) for key, _ in screens)
        versions = []
        for rows in range(most, 0, -1):
            cols = []
            for key, title in screens:
                lines = [Paragraph(f"<b>{title}</b>", s["cell"])]
                for q in movers.get(key, [])[:rows]:
                    lines.append(
                        Paragraph(
                            f"{escape(q.get('symbol') or '')}  {_pct_markup(q.get('change_pct'), bold=False)}",
                            s["cell"],
                        )
                    )
                cols.append(lines)
            table = Table([cols], colWidths=[CONTENT_W / 3] * 3)
            _tight(table, TOPPADDING=6)
            versions.append(
                [
                    Paragraph("Outside your watchlist", s["h2"]),
                    Paragraph(
                        "Biggest US movers in the last regular session (Yahoo Finance screens)", s["small"]
                    ),
                    table,
                ]
            )
        return versions

    # ------------------------------------------------------------------ #
    # One page per stock
    # ------------------------------------------------------------------ #
    def _stock_page(self, stock, ctx, raw):
        """
        Exactly one page: heading, chart, key metrics beside alerts and signals, headlines.

        Everything except the chart has a natural height, so the chart gets whatever is left. If
        that's under CHART_MIN_SCALE of full size, the least useful signals are dropped one at a
        time until it isn't. Key metrics, alerts and headlines are never trimmed.
        """
        m = stock["metrics"]
        why = self._why(stock, CONTENT_W, 4, label=None, style="body")
        # 'What's moving it' sits under the heading and is measured with it.
        top = [self._stock_heading(stock)]
        if why:
            top += [Paragraph("What's moving it", self.s["h3"]), why]
        left = self._metrics_column(stock)
        alerts = self._alert_lines(stock)
        news = self._headlines(stock, ctx)

        candles = raw.get("candles")
        has_chart = candles is not None and not candles.empty
        full_chart = CONTENT_W * CHART_ASPECT if has_chart else 0.0
        min_chart = full_chart * CHART_MIN_SCALE

        for shown in _signal_trims(_page_signals(stock)):
            columns = self._columns(left, alerts, [text for _, text in shown])
            text_flowables = top + [Spacer(1, 8), Spacer(1, 4), columns] + news
            room = FRAME_H - _height(text_flowables)
            if room >= min_chart:
                break

        chart_height = min(full_chart, room)
        if has_chart and chart_height < full_chart * CHART_LAST_RESORT_SCALE:
            print(f"    [WARN] PDF: no room for {stock['symbol']}'s chart; leaving it out.")
            has_chart = False
        if room < 0:
            print(f"    [WARN] PDF: {stock['symbol']}'s page is {-room:.0f}pt too tall and will spill.")

        story = top + [Spacer(1, 8)]
        if has_chart:
            png = price_volume_chart(candles, m)
            story.append(Image(png, width=CONTENT_W, height=chart_height, kind="proportional"))
        return story + [Spacer(1, 4), columns] + news

    def _stock_heading(self, stock):
        s = self.s
        m = stock["metrics"]
        name_bits = " · ".join(escape(b) for b in (m.get("name"), m.get("industry")) if b)
        price = m.get("premarket_price") or m.get("last_close")
        gap = m.get("gap_pct")
        top = Table(
            [
                [
                    [
                        # Named anchor: the ticker links on pages 1 and 2 jump here.
                        Paragraph(
                            f'<a name="{anchor_id(stock["symbol"])}"/>{escape(stock["symbol"])}', s["ticker"]
                        ),
                        Paragraph(name_bits or "&nbsp;", s["sub"]),
                    ],
                    [
                        Paragraph(fmt_money(price), s["price"]),
                        Paragraph(
                            f"{_pct_markup(gap)} {session_labels(m.get('premarket_session'))['long']}"
                            if gap is not None
                            else "last close",
                            s["cell_r"],
                        ),
                        Paragraph(
                            f'<a href="#{WATCHLIST_ANCHOR}" color="{LINK}">Back to watchlist</a>',
                            s["small_r"],
                        ),
                    ],
                ]
            ],
            colWidths=[CONTENT_W * 0.65, CONTENT_W * 0.35],
        )
        return _tight(top, TOPPADDING=3, BOTTOMPADDING=3)

    def _metrics_column(self, stock):
        s = self.s
        rows = [
            [Paragraph(escape(k), s["small"]), Paragraph(escape(v), s["cell_r"])]
            for k, v in self.formatter._metric_rows(stock)
        ]
        table = Table(rows, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.24])
        table.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return [Paragraph("Key metrics", s["h3"]), table]

    def _alert_lines(self, stock):
        s = self.s
        lines = [Paragraph("Alerts", s["h3"])]
        if not stock["alerts"]:
            return lines + [Paragraph("No alerts: everything within normal ranges.", s["small"])]
        for a in stock["alerts"]:
            color = TONE_PDF[a["direction"]]["color"]
            # bulletText can't carry markup, so the colored bullet goes in the text itself.
            lines.append(
                Paragraph(f"<font color='{color}'>•</font>&nbsp;&nbsp;{escape(a['text'])}", s["hang"])
            )
        return lines

    def _columns(self, left, alerts, signals):
        """Key metrics on the left; alerts, then any signals, on the right."""
        s = self.s
        right = list(alerts)
        if signals:
            style = ParagraphStyle(
                "sig", parent=s["bullet"], fontSize=8.5, leading=11, textColor=colors.HexColor(INK_2)
            )
            right.append(Paragraph("Signals", s["h3"]))
            right += [Paragraph(escape(text), style, bulletText="•") for text in signals]
        columns = Table([[left, right]], colWidths=[CONTENT_W * 0.48, CONTENT_W * 0.52])
        columns.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, -1), 14),
                    ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ]
            )
        )
        return columns

    def _headlines(self, stock, ctx):
        """Headlines block, each headline (with its source) capped at HEADLINE_MAX_LINES lines."""
        s = self.s
        news = stock.get("top_news", [])[: ctx["news_per_stock"]]
        if not news:
            return []
        summary = sentiment_summary(stock.get("sentiment_detail"), stock["sentiment"])
        story = [
            Paragraph(
                f"Headlines <font name='Helvetica' size='8.5' color='{INK_2}'>· sentiment {escape(summary)}</font>",
                s["h3"],
            )
        ]
        base = ParagraphStyle("news", parent=s["bullet"], fontSize=9)
        extra = strongest_note(stock, news)
        if extra:  # the clearest-toned relevant story isn't among the latest: show it too
            news = news + [{**extra, "sentiment": {"label": extra["label"], "counted": True}}]
        for article in news:
            url = safe_url(article.get("url"))
            mark = tone_mark(article)
            # The tone sign replaces the plain bullet, in its colour, so it costs no width.
            bullet, style = "•", base
            if mark:
                bullet = mark[0]
                style = ParagraphStyle(
                    f"news_{article['sentiment']['label']}",
                    parent=base,
                    bulletColor=colors.HexColor(mark[1]),
                    bulletFontName="Helvetica-Bold",
                )
            meta = [escape(article.get("source") or "")]
            if mark and not mark[2]:
                meta.append("not counted")
            if is_strongest(article, stock):
                meta.append("strongest story")
            meta = " · ".join(m for m in meta if m)
            suffix = f" <font size='7.5' color='{MUTED}'>— {meta}</font>" if meta else ""

            def make(text, url=url, suffix=suffix, style=style, bullet=bullet):
                headline = escape(text)
                link = f"<link href='{url}' color='{INK}'><u>{headline}</u></link>" if url else headline
                return Paragraph(link + suffix, style, bulletText=bullet)

            story.append(
                _fit_lines(make, article.get("headline") or "Untitled", CONTENT_W, HEADLINE_MAX_LINES)
            )
        return story

    # ------------------------------------------------------------------ #
    # Last page: how alerts work, and what each positioning level means
    # ------------------------------------------------------------------ #
    def _guide_page(self, ctx):
        s = self.s
        t = ctx["thresholds"]
        how_alerts_work = (
            f"A pre-market gap of ±{t.get('premarket_gap_pct', 3.0):g}% or a last-session move of "
            f"±{t.get('price_change_pct', 5.0):g}% (when it's at least a normal-sized move for that stock), "
            f"or any move of {t.get('unusual_move_sigma', 2.0):g}x the stock's typical daily move; volume at "
            f"{t.get('volume_surge_multiplier', 2.0):g}x its 20-day average; RSI outside "
            f"{t.get('rsi_oversold', 30)}–{t.get('rsi_overbought', 70)}; a close through the 50- or 200-day "
            f"average; a 52-week high or low; earnings within {t.get('earnings_lookahead_days', 7)} days. "
            f"Data: Finnhub and Yahoo Finance."
        )
        story = [
            Paragraph("How alerts work", s["h2"]),
            Paragraph(escape(how_alerts_work), s["body"]),
            Paragraph("News sentiment", s["h3"]),
            Paragraph(escape(SENTIMENT_NOTE), s["small"]),
            Paragraph("Reading short interest, options and cash runway", s["h2"]),
            Paragraph(escape(self.formatter.positioning_note()), s["body"]),
        ]
        for title, rows in self.formatter.positioning_guide():
            table = Table(
                [
                    [
                        Paragraph(escape(r), s["cell"]),
                        Paragraph(f"<b>{escape(name)}</b>", s["cell"]),
                        Paragraph(escape(meaning), s["cell"]),
                    ]
                    for r, name, meaning in rows
                ],
                colWidths=[CONTENT_W * 0.14, CONTENT_W * 0.14, CONTENT_W * 0.72],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(KeepTogether([Paragraph(escape(title), s["h3"]), table]))
        if _height(story) > FRAME_H:
            print("    [WARN] PDF: the guide page is taller than one page and will spill.")
        return story
