"""
Report Formatter 

Turns the analyzer's output into the newsletter: an HTML email, a plain-text twin, and a
subject line. The PDF (src/pdf_report.py) is built from the SAME `context` dictionary, so the
email and the PDF can never disagree about a number.

Design decisions:
1. build_context() does all the sorting/choosing ONCE. The renderers only lay things out.
2. Email HTML uses tables + inline styles. Gmail and Outlook ignore most modern CSS
   (flexbox, grid, <style> blocks in some clients), so this is the reliable way to get
   the same layout everywhere.
3. Every piece of outside text (headlines, company names, URLs) goes through html.escape().
   A headline containing "<" or "&" would otherwise break the email, and a malicious link
   could inject markup.
4. Color never carries meaning alone: every red/green number also has a +/- sign, an arrow,
   or a word ("Down", "Up", "Watch").
"""

import html
from datetime import datetime

from src.analyzer import DAYS_TO_COVER_LEVELS, PUT_CALL_LEVELS, RUNWAY_LEVELS, SHORT_INTEREST_LEVELS, runway_text

# Palette (shared with the PDF). Status colors are reserved for good/bad meaning.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
PAGE_BG = "#f9f9f7"
CARD_BG = "#ffffff"
UP_TEXT = "#006300"      # gains (text-safe green)
DOWN_TEXT = "#d03b3b"    # losses (status critical)
CAUTION = "#fab219"      # watch items (status warning, always paired with a label)

TONE = {
    "down": {"color": DOWN_TEXT, "arrow": "▼", "label": "Down"},
    "up": {"color": UP_TEXT, "arrow": "▲", "label": "Up"},
    "caution": {"color": "#a86e00", "arrow": "●", "label": "Watch"},
}
# No quotes around "Segoe UI": the font list sits inside style='...' attributes, and a quote
# there would end the attribute early. Unquoted multi-word font names are valid CSS.
# Labels for the "latest price" column, which depends on when the report runs. The scheduled
# 7 AM Central (8 AM ET) run is always "pre-market"; a manual run at 2 PM or 6 PM gets honest labels instead.
SESSION_LABELS = {
    "pre-market": {"col": "Pre-mkt", "short": "pre-mkt", "long": "pre-market", "metric": "Pre-market",
                   "gap_note": "pre-market price vs. last close"},
    "regular": {"col": "Now", "short": "today", "long": "so far today", "metric": "Price now",
                "gap_note": "current price vs. yesterday's close"},
    "after-hours": {"col": "After-hrs", "short": "after-hrs", "long": "after hours", "metric": "After hours",
                    "gap_note": "after-hours price vs. today's close"},
}


def session_labels(session):
    return SESSION_LABELS.get(session or "pre-market", SESSION_LABELS["pre-market"])


FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif"


# ---------------------------------------------------------------------- #
# Number formatting helpers (shared with the PDF)
# ---------------------------------------------------------------------- #
def fmt_money(value, decimals=2):
    return "—" if value is None else f"${value:,.{decimals}f}"


def fmt_pct(value, signed=True, decimals=1):
    if value is None:
        return "—"
    return f"{value:+.{decimals}f}%" if signed else f"{value:.{decimals}f}%"


def fmt_ratio(value):
    return "—" if value is None else f"{value:.1f}x"


def fmt_num(value, decimals=1):
    return "—" if value is None else f"{value:.{decimals}f}"


def fmt_volume(value):
    if value is None:
        return "—"
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:.1f}{suffix}"
    return f"{value:.0f}"


def fmt_market_cap(millions):
    """Finnhub gives market cap in millions of dollars."""
    if millions is None:
        return "—"
    dollars = millions * 1e6
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if dollars >= size:
            return f"${dollars / size:.1f}{suffix}"
    return f"${dollars:,.0f}"


def pct_color(value):
    if value is None or abs(value) < 0.05:
        return INK_2
    return UP_TEXT if value > 0 else DOWN_TEXT


def arrow(value):
    if value is None or abs(value) < 0.05:
        return ""
    return "▲ " if value > 0 else "▼ "


def safe_url(url):
    """Only http(s) links make it into the email; anything else (javascript:, data:) is dropped."""
    if url and url.lower().startswith(("http://", "https://")):
        return html.escape(url, quote=True)
    return None


def e(text):
    """Short alias: escape any outside text before it goes into HTML."""
    return html.escape(str(text)) if text is not None else ""


class ReportFormatter:
    def __init__(self, report_config: dict = None, thresholds: dict = None):
        self.config = report_config or {}
        self.thresholds = thresholds or {}
        self.title = self.config.get("title", "Pre-Market Brief")
        self.news_per_stock = self.config.get("news_per_stock", 3)

    # ------------------------------------------------------------------ #
    # 1. Context: every decision about WHAT to show, made once
    # ------------------------------------------------------------------ #
    def build_context(self, results: dict, ranked: list, market: dict = None,
                      movers: dict = None, generated_at: datetime = None) -> dict:
        generated_at = generated_at or datetime.now()

        stocks = []
        errors = []
        for symbol, result in results.items():
            if "error" in result:
                errors.append({"symbol": symbol, "error": result["error"]})
                continue
            stocks.append({"symbol": symbol, **result})

        # Stock pages run biggest pre-market move first (stocks without a pre-market trade last),
        # so the order of the email matches the order of urgency.
        def sort_key(stock):
            gap = stock["metrics"].get("gap_pct")
            return (gap is None, -abs(gap) if gap is not None else 0)
        stocks.sort(key=sort_key)

        # Latest pre-market timestamp across the watchlist = "prices as of"
        times = [s["metrics"].get("premarket_as_of") for s in stocks if s["metrics"].get("premarket_as_of")]
        as_of = max(times) if times else None

        # Which session do most of the latest prices come from? Drives every column label.
        sessions = [s["metrics"].get("premarket_session") for s in stocks if s["metrics"].get("premarket_session")]
        session = max(set(sessions), key=sessions.count) if sessions else "pre-market"

        return {
            "title": self.title,
            "generated_at": generated_at,
            "date_label": f"{generated_at:%A, %B} {generated_at.day}, {generated_at:%Y}",
            "short_date": f"{generated_at:%a %b} {generated_at.day}",
            "as_of": as_of,
            "session": session,
            "labels": session_labels(session),
            "market": market or {},
            "ranked": ranked,
            "stocks": stocks,
            "errors": errors,
            "movers": movers or {},
            "thresholds": self.thresholds,
            "news_per_stock": self.news_per_stock,
        }

    def subject(self, ctx: dict) -> str:
        ranked = ctx["ranked"]
        if not ranked:
            return f"{ctx['title']} · {ctx['short_date']} · No major moves"
        parts = []
        for item in ranked[:2]:
            gap = item.get("gap_pct")
            if gap is not None and abs(gap) >= 0.05:
                parts.append(f"{item['symbol']} {'▲' if gap > 0 else '▼'}{abs(gap):.1f}%")
            else:
                parts.append(item["symbol"])
        extra = f" +{len(ranked) - 2} more" if len(ranked) > 2 else ""
        return f"{ctx['title']} · {ctx['short_date']} · {', '.join(parts)}{extra}"

    # ------------------------------------------------------------------ #
    # 2. HTML email
    # ------------------------------------------------------------------ #
    def render_html(self, ctx: dict, pdf_attached: bool = False) -> str:
        sections = [
            self._html_header(ctx),
            self._html_market(ctx),
            self._html_headline_alerts(ctx),
            self._html_glance_table(ctx),
            "".join(self._html_stock(stock, ctx) for stock in ctx["stocks"]),
            self._html_movers(ctx),
            self._html_positioning_guide(),
            self._html_footer(ctx, pdf_attached),
        ]
        body = "".join(s for s in sections if s)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta name='color-scheme' content='light'>"
            f"<title>{e(ctx['title'])}</title></head>"
            f"<body style='margin:0;padding:0;background:{PAGE_BG};'>"
            f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' style='background:{PAGE_BG};'>"
            "<tr><td align='center' style='padding:16px 8px;'>"
            f"<table role='presentation' width='640' cellpadding='0' cellspacing='0' "
            f"style='width:100%;max-width:640px;font-family:{FONT};color:{INK};'>"
            f"{body}</table></td></tr></table></body></html>"
        )

    def _card(self, inner: str, padding="20px 24px", extra_style="") -> str:
        return (f"<tr><td style='background:{CARD_BG};border:1px solid {HAIRLINE};border-radius:8px;"
                f"padding:{padding};{extra_style}'>{inner}</td></tr>"
                "<tr><td style='height:12px;line-height:12px;font-size:0;'>&nbsp;</td></tr>")

    def _h2(self, text: str, note: str = "") -> str:
        note_html = f"<div style='font-size:13px;font-weight:400;color:{INK_2};margin-top:2px;'>{e(note)}</div>" if note else ""
        return f"<div style='font-size:17px;font-weight:600;margin:0 0 12px 0;'>{e(text)}{note_html}</div>"

    def _html_header(self, ctx):
        as_of = ctx["as_of"]
        as_of_text = f"Prices as of {as_of:%I:%M %p} ET".replace(" 0", " ") if as_of else "No pre-market trades yet"
        n = len(ctx["stocks"])
        inner = (
            f"<div style='font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:{INK_2};'>{e(ctx['title'])}</div>"
            f"<div style='font-size:24px;font-weight:600;margin-top:4px;'>{e(ctx['date_label'])}</div>"
            f"<div style='font-size:13px;color:{INK_2};margin-top:6px;'>{e(as_of_text)} · {n} stock{'s' if n != 1 else ''} tracked · "
            f"{len(ctx['ranked'])} need{'s' if len(ctx['ranked']) == 1 else ''} attention</div>"
        )
        return self._card(inner)

    def _html_market(self, ctx):
        market = ctx["market"]
        names = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "^VIX": "VIX fear gauge"}
        cells = []
        for symbol in ("SPY", "QQQ", "^VIX"):
            data = market.get(symbol)
            if not data:
                continue
            if symbol == "^VIX":
                main = fmt_num(data.get("last_close"))
                sub_val = data.get("last_session_pct")
                sub = f"{arrow(sub_val)}{fmt_pct(sub_val)} last session"
                # Rising VIX = rising fear, so it is colored like a loss
                color = pct_color(-sub_val if sub_val is not None else None)
            else:
                gap = data.get("gap_pct")
                main = fmt_money(data.get("premarket_price") or data.get("last_close"))
                sub = f"{arrow(gap)}{fmt_pct(gap)} {ctx['labels']['long']}" if gap is not None else \
                    f"{fmt_pct(data.get('last_session_pct'))} last session"
                color = pct_color(gap if gap is not None else data.get("last_session_pct"))
            cells.append(
                f"<td width='33%' valign='top' style='padding:4px 8px 4px 0;'>"
                f"<div style='font-size:12px;color:{INK_2};'>{e(names[symbol])}</div>"
                f"<div style='font-size:18px;font-weight:600;margin-top:2px;'>{e(main)}</div>"
                f"<div style='font-size:13px;color:{color};margin-top:2px;'>{e(sub)}</div></td>")
        if not cells:
            return ""
        inner = (self._h2("Market backdrop")
                 + f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0'><tr>{''.join(cells)}</tr></table>")
        return self._card(inner)

    def _html_headline_alerts(self, ctx):
        ranked = ctx["ranked"]
        if not ranked:
            inner = (self._h2("Big & concerning moves")
                     + f"<div style='font-size:14px;color:{INK_2};'>Nothing crossed your thresholds this morning. "
                       f"All {len(ctx['stocks'])} stocks are trading within their normal ranges.</div>")
            return self._card(inner)

        rows = []
        for item in ranked:
            tone = TONE[item["tone"]]
            gap = item.get("gap_pct")
            gap_html = (f"<span style='color:{pct_color(gap)};font-weight:600;'>{arrow(gap)}{fmt_pct(gap)}</span>"
                        f"<span style='color:{INK_2};'> {ctx['labels']['short']}</span>") if gap is not None else ""
            name = f"<span style='color:{INK_2};font-weight:400;'> {e(item.get('name') or '')}</span>" if item.get("name") else ""
            alerts = "".join(
                f"<li style='margin:3px 0;'>{e(a['text'])}</li>" for a in item["alerts"])
            rows.append(
                f"<tr><td style='border-left:4px solid {tone['color']};padding:8px 0 8px 12px;'>"
                f"<div style='font-size:15px;font-weight:600;'>{e(item['symbol'])}{name}"
                f"<span style='float:right;font-size:13px;font-weight:400;'>{gap_html}</span></div>"
                f"<div style='font-size:12px;color:{tone['color']};font-weight:600;margin-top:2px;'>"
                f"{tone['arrow']} {tone['label']}</div>"
                f"<ul style='margin:6px 0 0 0;padding-left:18px;font-size:14px;line-height:1.45;'>{alerts}</ul>"
                f"</td></tr><tr><td style='height:10px;line-height:10px;font-size:0;'>&nbsp;</td></tr>")
        inner = (self._h2("Big & concerning moves", "Ranked by how far each move is past its threshold")
                 + f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0'>{''.join(rows)}</table>")
        return self._card(inner)

    def _html_glance_table(self, ctx):
        th_style = (f"font-size:11px;font-weight:600;color:{INK_2};text-align:right;padding:6px 4px;"
                    f"border-bottom:1px solid {HAIRLINE};white-space:nowrap;")
        td_style = f"font-size:13px;text-align:right;padding:7px 3px;border-bottom:1px solid {HAIRLINE};white-space:nowrap;"
        # Six columns is the most that fits a phone screen without sideways scrolling.
        headers = ["Ticker", ctx["labels"]["col"], "Gap", "Last day", "RSI", "Vol/avg"]
        head = "".join(
            f"<th style='{th_style}{'text-align:left;' if i == 0 else ''}'>{h}</th>" for i, h in enumerate(headers))
        rows = []
        for stock in ctx["stocks"]:
            m = stock["metrics"]
            rsi = m.get("rsi_14")
            rsi_note = ""
            if rsi is not None and (rsi > self.thresholds.get("rsi_overbought", 70) or rsi < self.thresholds.get("rsi_oversold", 30)):
                rsi_note = "font-weight:600;"
            vol = m.get("volume_ratio")
            vol_note = "font-weight:600;" if vol is not None and vol >= self.thresholds.get("volume_surge_multiplier", 2.0) else ""
            cells = [
                f"<td style='{td_style}text-align:left;font-weight:600;'>{e(stock['symbol'])}</td>",
                f"<td style='{td_style}'>{fmt_money(m.get('premarket_price'))}</td>",
                f"<td style='{td_style}color:{pct_color(m.get('gap_pct'))};font-weight:600;'>{fmt_pct(m.get('gap_pct'))}</td>",
                f"<td style='{td_style}color:{pct_color(m.get('last_session_pct'))};'>{fmt_pct(m.get('last_session_pct'))}</td>",
                f"<td style='{td_style}{rsi_note}'>{fmt_num(rsi, 0)}</td>",
                f"<td style='{td_style}{vol_note}'>{fmt_ratio(vol)}</td>",
            ]
            rows.append(f"<tr>{''.join(cells)}</tr>")
        for err in ctx["errors"]:
            rows.append(f"<tr><td style='{td_style}text-align:left;font-weight:600;'>{e(err['symbol'])}</td>"
                        f"<td colspan='5' style='{td_style}text-align:left;color:{INK_2};'>{e(err['error'])}</td></tr>")
        inner = (self._h2("Watchlist at a glance", f"Gap = {ctx['labels']['gap_note']} · bold = past a threshold")
                 + "<div style='overflow-x:auto;'>"
                 f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' style='border-collapse:collapse;'>"
                 f"<tr>{head}</tr>{''.join(rows)}</table></div>")
        return self._card(inner)

    def _metric_rows(self, stock):
        """The key-metric pairs shown on each stock's section (and its PDF page)."""
        m = stock["metrics"]
        hi, lo = m.get("high_52w"), m.get("low_52w")
        earnings = "—"
        if m.get("earnings_date"):
            when = {"bmo": " (before open)", "amc": " (after close)"}.get(m.get("earnings_hour"), "")
            earnings = f"{m['earnings_date']}{when}"
        def with_level(value, key):
            return value if value == "—" or not m.get(key) else f"{value} · {m[key]}"

        short_pct = m.get("short_pct_float")
        short_interest = "—" if short_pct is None else f"{fmt_pct(short_pct, signed=False)} of float"
        inst_pct = m.get("institutional_pct")
        net_debt = m.get("net_debt_m")
        if net_debt is None:
            net_debt_text = "—"
        elif net_debt > 0:
            net_debt_text = fmt_market_cap(net_debt)
        else:
            net_debt_text = f"−{fmt_market_cap(-net_debt)} (net cash)" if net_debt < 0 else "$0"
        months = m.get("cash_runway_months")
        if months is not None:
            runway = with_level(runway_text(months), "cash_runway_level")
        elif (m.get("operating_cashflow_m") or 0) > 0:
            runway = "Cash-flow positive"
        else:
            runway = "—"
        return [
            (session_labels(m.get("premarket_session"))["metric"],
             f"{fmt_money(m.get('premarket_price'))}  ({fmt_pct(m.get('gap_pct'))})"),
            ("Last close", f"{fmt_money(m.get('last_close'))}  ({fmt_pct(m.get('last_session_pct'))})"),
            ("Typical daily move", fmt_pct(m.get("typical_move_pct"), signed=False)),
            ("Volume vs 20-day avg", fmt_ratio(m.get("volume_ratio"))),
            ("RSI (14-day)", fmt_num(m.get("rsi_14"))),
            ("50-day / 200-day avg", f"{fmt_money(m.get('sma_50'))} / {fmt_money(m.get('sma_200'))}"),
            ("52-week range", f"{fmt_money(lo)} – {fmt_money(hi)}"),
            ("From 52-week high", fmt_pct(m.get("pct_from_high_52w"))),
            ("1-month return", f"{fmt_pct(m.get('return_1m_pct'))}  ({fmt_pct(m.get('relative_1m_pct'))} vs market)"),
            ("Market cap", fmt_market_cap(m.get("market_cap_m"))),
            ("Forward P/E (12 mo)", fmt_num(m.get("forward_pe"))),
            ("Beta", fmt_num(m.get("beta"), 2)),
            ("Net debt", net_debt_text),
            ("Cash runway", runway),
            ("Short interest", with_level(short_interest, "short_level")),
            ("Days to cover", with_level(fmt_num(m.get("short_ratio")), "days_to_cover_level")),
            ("Call open interest", fmt_volume(m.get("call_oi"))),
            ("Calls expiring this week", "—" if m.get("call_oi_week") is None else
             f"{fmt_volume(m['call_oi_week'])}  ({fmt_pct(m.get('call_oi_week_pct'), signed=False)})"),
            ("Put/call ratio", with_level(fmt_num(m.get("put_call_oi_ratio"), 2), "put_call_level")),
            ("Institutional ownership", "—" if inst_pct is None else f"{fmt_pct(inst_pct, signed=False)} of shares"),
            ("Next earnings", earnings),
        ]

    def _html_stock(self, stock, ctx):
        m = stock["metrics"]
        name_bits = [b for b in (m.get("name"), m.get("industry")) if b]
        name_line = f"<div style='font-size:13px;color:{INK_2};margin-top:2px;'>{e(' · '.join(name_bits))}</div>" if name_bits else ""
        gap = m.get("gap_pct")
        label = session_labels(m.get("premarket_session"))
        price_line = (f"<div style='float:right;text-align:right;'>"
                      f"<div style='font-size:18px;font-weight:600;'>{fmt_money(m.get('premarket_price') or m.get('last_close'))}</div>"
                      f"<div style='font-size:13px;color:{pct_color(gap)};font-weight:600;'>{arrow(gap)}{fmt_pct(gap)} "
                      f"<span style='color:{INK_2};font-weight:400;'>{label['short'] if gap is not None else ''}</span></div></div>")

        # The email shows the metrics people scan first, one per row (readable on a phone).
        # The PDF page shows the full set from _metric_rows().
        email_keys = (label["metric"], "Last close", "Typical daily move", "Volume vs 20-day avg",
                      "RSI (14-day)", "52-week range", "1-month return", "Forward P/E (12 mo)",
                      "Short interest", "Put/call ratio", "Institutional ownership",
                      "Net debt", "Cash runway", "Next earnings")
        pairs = [(k, v) for k, v in self._metric_rows(stock) if k in email_keys and v != "—"]
        metric_rows = "".join(
            f"<tr><td style='font-size:13px;color:{INK_2};padding:5px 8px 5px 0;border-bottom:1px solid {HAIRLINE};'>{e(k)}</td>"
            f"<td style='font-size:13px;text-align:right;padding:5px 0;border-bottom:1px solid {HAIRLINE};'>{e(v)}</td></tr>"
            for k, v in pairs)

        alerts_html = ""
        if stock["alerts"]:
            items = "".join(
                f"<li style='margin:3px 0;'><span style='color:{TONE[a['direction']]['color']};font-weight:600;'>"
                f"{TONE[a['direction']]['arrow']}</span> {e(a['text'])}</li>" for a in stock["alerts"])
            alerts_html = (f"<div style='font-size:13px;font-weight:600;margin:14px 0 4px;'>Alerts</div>"
                           f"<ul style='margin:0;padding-left:18px;font-size:14px;line-height:1.45;list-style:none;padding-left:0;'>{items}</ul>")

        sd = stock.get("sentiment_detail", {})
        news_items = []
        for article in stock.get("top_news", [])[:ctx["news_per_stock"]]:
            url = safe_url(article.get("url"))
            headline = e(article.get("headline") or "Untitled")
            link = f"<a href='{url}' style='color:{INK};'>{headline}</a>" if url else headline
            meta_bits = [article.get("source")]
            if article.get("datetime"):
                ts = datetime.fromtimestamp(article["datetime"])
                meta_bits.append(f"{ts:%b} {ts.day}")
            meta = " · ".join(b for b in meta_bits if b)
            meta_html = f"<span style='color:{MUTED};font-size:12px;'> — {e(meta)}</span>" if meta else ""
            news_items.append(f"<li style='margin:4px 0;'>{link}{meta_html}</li>")
        news_html = ""
        if news_items:
            news_html = (f"<div style='font-size:13px;font-weight:600;margin:14px 0 4px;'>Headlines "
                         f"<span style='font-weight:400;color:{INK_2};'>· sentiment {e(stock['sentiment'])} "
                         f"({sd.get('positive', 0)} positive, {sd.get('negative', 0)} negative, {sd.get('neutral', 0)} neutral)</span></div>"
                         f"<ul style='margin:0;padding-left:18px;font-size:13px;line-height:1.45;'>{''.join(news_items)}</ul>")

        inner = (
            f"<div style='overflow:hidden;'>{price_line}"
            f"<div style='font-size:20px;font-weight:600;'>{e(stock['symbol'])}</div>{name_line}</div>"
            f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' style='margin-top:12px;border-collapse:collapse;'>{metric_rows}</table>"
            f"{alerts_html}{news_html}"
        )
        # page-break-before: when the saved HTML is printed, each stock starts a new page
        return self._card(inner, extra_style="page-break-before:always;")

    def _html_movers(self, ctx):
        movers = ctx["movers"]
        if not movers or not any(movers.values()):
            return ""
        titles = {"day_gainers": "Top gainers", "day_losers": "Top losers", "most_actives": "Most active"}
        cols = []
        for key in ("day_gainers", "day_losers", "most_actives"):
            rows = "".join(
                f"<tr><td style='font-size:13px;font-weight:600;padding:3px 0;'>{e(q.get('symbol'))}</td>"
                f"<td style='font-size:13px;text-align:right;color:{pct_color(q.get('change_pct'))};padding:3px 0;'>"
                f"{fmt_pct(q.get('change_pct'))}</td></tr>"
                for q in movers.get(key, []))
            cols.append(f"<td width='33%' valign='top' style='padding-right:12px;'>"
                        f"<div style='font-size:12px;color:{INK_2};margin-bottom:4px;'>{titles[key]}</div>"
                        f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0'>{rows}</table></td>")
        inner = (self._h2("Outside your watchlist", "Biggest US movers in the last regular session (Yahoo Finance screens)")
                 + f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0'><tr>{''.join(cols)}</tr></table>")
        return self._card(inner)

    # ------------------------------------------------------------------ #
    # Positioning guide: what each short-interest / options level means
    # ------------------------------------------------------------------ #
    POSITIONING_NOTE = ("These show how traders are betting, not how fast the business is growing: they hint at "
                        "near-term volatility and sentiment. Short interest is reported twice a month; open "
                        "interest counts option contracts expiring in the next {window} days, and calls "
                        "expiring this week are flagged when they're {expiring}% or more of those. Institutional "
                        "ownership comes from funds' quarterly 13F filings, so it moves in steps as filings arrive "
                        "(mostly mid-Feb, May, Aug and Nov). It can top 100% because lent-out shares get counted "
                        "twice. Forward P/E uses analysts' earnings estimates for the next 12 months. Net debt is debt "
                        "minus cash (negative means more cash than debt). Cash runway is cash divided by the cash "
                        "the business burned over the last 12 months, shown only for companies losing cash.")

    def positioning_guide(self):
        """
        [(title, [(range text, level name, meaning), ...]), ...] for the email, text and PDF.
        Ranges come from config.yaml, so the guide always matches the rules being applied.
        """
        t = self.thresholds

        def ranges(bounds, fmt):
            edges = [fmt(b) for b in bounds]
            return ([f"under {edges[0]}"] + [f"{a}–{b}" for a, b in zip(edges, edges[1:])]
                    + [f"{edges[-1]} and up"])

        sections = (
            ("Short interest (% of float)", t.get("short_interest_levels_pct", [5, 10, 20]),
             lambda v: f"{v:g}%", SHORT_INTEREST_LEVELS),
            ("Days to cover", t.get("days_to_cover_levels", [2, 5, 10]), lambda v: f"{v:g}", DAYS_TO_COVER_LEVELS),
            ("Put/call open interest", t.get("put_call_levels", [0.5, 0.8, 1.2, 1.5]), lambda v: f"{v:g}",
             PUT_CALL_LEVELS),
            ("Cash runway (months)", t.get("cash_runway_levels_months", [12, 24]), lambda v: f"{v:g}",
             RUNWAY_LEVELS),
        )
        guide = []
        for title, bounds, fmt, levels in sections:
            rows = [(r, name, meaning[0].upper() + meaning[1:])
                    for r, (name, meaning) in zip(ranges(sorted(bounds), fmt), levels)]
            guide.append((title, rows))
        return guide

    def positioning_note(self):
        return self.POSITIONING_NOTE.format(window=self.thresholds.get("options_window_days", 30),
                                            expiring=f"{self.thresholds.get('calls_expiring_week_pct', 50):g}")

    def _html_positioning_guide(self):
        cell = f"font-size:12px;color:{INK_2};padding:3px 8px 3px 0;vertical-align:top;border-bottom:1px solid {HAIRLINE};"
        tables = []
        for title, rows in self.positioning_guide():
            body = "".join(f"<tr><td style='{cell}white-space:nowrap;'>{e(r)}</td>"
                           f"<td style='{cell}white-space:nowrap;font-weight:600;color:{INK};'>{e(name)}</td>"
                           f"<td style='{cell}'>{e(meaning)}</td></tr>" for r, name, meaning in rows)
            tables.append(f"<div style='font-size:13px;font-weight:600;margin:10px 0 2px;'>{e(title)}</div>"
                          f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
                          f"style='border-collapse:collapse;'>{body}</table>")
        inner = (self._h2("Reading short interest, options and cash runway", self.positioning_note()) + "".join(tables))
        return self._card(inner)

    def _html_footer(self, ctx, pdf_attached):
        t = ctx["thresholds"]
        rules = (f"Alerts fire on a pre-market gap of ±{t.get('premarket_gap_pct', 3.0):g}% or a last-session move of "
                 f"±{t.get('price_change_pct', 5.0):g}%, or any move of at least {t.get('unusual_move_sigma', 2.0):g}x the "
                 f"stock's typical daily move; volume at {t.get('volume_surge_multiplier', 2.0):g}x its 20-day average; "
                 f"RSI outside {t.get('rsi_oversold', 30)}–{t.get('rsi_overbought', 70)}; a close through the 50- or 200-day "
                 f"average; a 52-week high or low; earnings within {t.get('earnings_lookahead_days', 7)} days.")
        pdf_note = "The full newsletter is attached as a PDF. " if pdf_attached else ""
        return (f"<tr><td style='padding:4px 8px 24px;font-size:12px;line-height:1.5;color:{INK_2};'>"
                f"{pdf_note}{e(rules)}<br><br>Data: Finnhub (quotes, news, fundamentals, earnings) and Yahoo Finance "
                f"(price history, pre-market trades, forward P/E, short interest, options, institutional ownership, debt and cash). "
                f"Automated analysis for information only, not investment advice."
                f"</td></tr>")

    # ------------------------------------------------------------------ #
    # 3. Plain-text twin (shown by clients that block HTML; also helps spam scoring)
    # ------------------------------------------------------------------ #
    def render_text(self, ctx: dict, pdf_attached: bool = False) -> str:
        lines = [f"{ctx['title'].upper()} — {ctx['date_label']}", ""]

        market = ctx["market"]
        if market:
            parts = []
            for symbol in ("SPY", "QQQ"):
                if symbol in market:
                    parts.append(f"{symbol} {fmt_pct(market[symbol].get('gap_pct'))} {ctx['labels']['short']}")
            if "^VIX" in market:
                parts.append(f"VIX {fmt_num(market['^VIX'].get('last_close'))}")
            lines += ["MARKET: " + " | ".join(parts), ""]

        lines.append("BIG & CONCERNING MOVES")
        if not ctx["ranked"]:
            lines.append(f"  Nothing crossed your thresholds. All {len(ctx['stocks'])} stocks are in normal ranges.")
        for item in ctx["ranked"]:
            tone = TONE[item["tone"]]
            lines.append(f"  {item['symbol']} [{tone['label']}] {fmt_pct(item.get('gap_pct'))} {ctx['labels']['short']}")
            lines += [f"    - {a['text']}" for a in item["alerts"]]
        lines.append("")

        lines.append("WATCHLIST")
        lines.append(f"  {'Ticker':<7}{ctx['labels']['col']:>10}{'Gap':>8}{'Close':>10}{'RSI':>6}{'Vol/avg':>9}")
        for stock in ctx["stocks"]:
            m = stock["metrics"]
            lines.append(f"  {stock['symbol']:<7}{fmt_money(m.get('premarket_price')):>10}{fmt_pct(m.get('gap_pct')):>8}"
                         f"{fmt_money(m.get('last_close')):>10}{fmt_num(m.get('rsi_14'), 0):>6}{fmt_ratio(m.get('volume_ratio')):>9}")
        for err in ctx["errors"]:
            lines.append(f"  {err['symbol']:<7}{err['error']}")
        lines.append("")

        for stock in ctx["stocks"]:
            lines.append(f"{stock['symbol']} — {stock['metrics'].get('name') or ''}".rstrip(" —"))
            lines += [f"  {k}: {v}" for k, v in self._metric_rows(stock)]
            lines += [f"  ! {a['text']}" for a in stock["alerts"]]
            for article in stock.get("top_news", [])[:ctx["news_per_stock"]]:
                url = article.get("url") if safe_url(article.get("url")) else ""
                lines.append(f"  * {article.get('headline')} {url}".rstrip())
            lines.append("")

        lines.append("READING SHORT INTEREST, OPTIONS AND CASH RUNWAY")
        lines.append(f"  {self.positioning_note()}")
        for title, rows in self.positioning_guide():
            lines.append(f"  {title}")
            lines += [f"    {r:<14}{name:<14}{meaning}" for r, name, meaning in rows]
        lines.append("")

        if pdf_attached:
            lines.append("The full newsletter is attached as a PDF.")
        lines.append("Automated analysis for information only, not investment advice.")
        return "\n".join(lines)
