"""
Indicators, signals and alerts.

For each stock the analyzer produces three layers:
  metrics   raw numbers, used by the tables, the PDF page and the tests
  triggers  plain-English lines describing every indicator (always shown on the stock's page)
  alerts    only what crossed a threshold, each with a direction and a severity

Severity is relative to the threshold: 1.0 means "just crossed it", 2.0 means "twice as
extreme". The report header only shows alerts at or above `header_min_severity` (config.yaml),
so minor notes stay on the stock's own page instead of crowding the top of the email.
"""

import bisect
import re
from datetime import date

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

UP, DOWN, CAUTION = "up", "down", "caution"
SESSION_WORDS = {"pre-market": "pre-market", "regular": "so far today", "after-hours": "after hours"}


# ---------------------------------------------------------------------- #
# Level vocabularies
#
# The numeric boundaries live in config.yaml; the words live here, so the report's guide and
# each stock's signals always describe a level the same way. These describe how traders are
# positioned, not how the business is doing: they hint at near-term volatility and sentiment.
# ---------------------------------------------------------------------- #
SHORT_INTEREST_LEVELS = (
    ("Low", "Few bets against the stock; the price mostly follows the business."),
    ("Moderate", "Some skepticism; normal for fast-growing or volatile stocks."),
    (
        "High",
        "A sizable bet on a decline. Moves get bigger both ways, and good news can force shorts to buy back.",
    ),
    (
        "Very high",
        "A crowded short: squeeze potential on good news, but many traders expect growth or "
        "the valuation to disappoint.",
    ),
)
DAYS_TO_COVER_LEVELS = (
    ("Low", "Shorts could exit within a couple of days of normal trading."),
    ("Moderate", "Shorts would need most of a week to exit."),
    ("High", "Exiting takes over a week, so a rally can feed on itself as shorts buy back."),
    ("Very high", "Shorts are boxed in: classic short-squeeze conditions if the news turns good."),
)
PUT_CALL_LEVELS = (
    ("Very bullish", "call-heavy, traders positioned for gains. At extremes it can also mean froth."),
    ("Bullish", "more calls than puts, typical for a stock people expect to grow."),
    ("Neutral", "bets roughly balanced."),
    ("Cautious", "more puts, from holders hedging or traders betting on a drop."),
    (
        "Bearish",
        "put-heavy, strong demand for downside protection. Contrarians sometimes read extreme "
        "pessimism as a sign of a bottom.",
    ),
)
RUNWAY_LEVELS = (
    (
        "Short",
        "under a year of cash at the current burn. Expect a share sale or new debt soon, which can "
        "dilute holders or add interest costs.",
    ),
    ("Moderate", "a year or two of cash. A raise is likely before the business pays for itself."),
    ("Long", "two years or more of cash, so no near-term pressure to raise money."),
)


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
def _pct(new, old):
    """Percent change from old to new, or None if either side is missing (or old is zero)."""
    if new is None or old is None or pd.isna(new) or pd.isna(old) or old == 0:
        return None
    return float((new - old) / old * 100)


def _num(value):
    """Plain float from a pandas/numpy scalar, with NaN mapped to None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


TNX_SYMBOL = "^TNX"  # 10-year Treasury yield index


def _yield_pct(value):
    """
    ^TNX as a percent. Yahoo quotes it directly (4.92 = 4.92%) as of late 2026, but historically
    quoted ten times that (49.2); no real 10-year yield is anywhere near 20%, so that's the cut.
    """
    if value is None:
        return None
    return value / 10 if value > 20 else value


def level_of(value, bounds, levels):
    """The (name, meaning) a value falls in. Bounds [5, 10, 20] split values into <5, 5-10, 10-20, 20+."""
    return levels[bisect.bisect_right(bounds, value)]


def _count(value):
    """Compact count for signal text: 106425 -> '106K'."""
    for size, suffix in ((1e6, "M"), (1e3, "K")):
        if value >= size:
            return f"{value / size:.{0 if value >= size * 10 else 1}f}{suffix}"
    return f"{value:,.0f}"


def _dollars(millions):
    """Compact unsigned dollar amount from millions: 2160 -> '$2.2B', 140 -> '$140M'."""
    millions = abs(millions)
    for size, suffix in ((1e6, "T"), (1e3, "B")):
        if millions >= size:
            return f"${millions / size:.1f}{suffix}"
    return f"${millions:.0f}M" if millions >= 1 else f"${millions * 1000:.0f}K"


def runway_text(months):
    """'~3 months' or '~4.7 years'. Shared with the formatter so the signal and metric row agree."""
    if months < 1:
        return "under a month"
    return f"~{months:.0f} months" if months < 24 else f"~{months / 12:.1f} years"


class _SignalLog(list):
    """
    The trigger lines, plus a parallel list naming what each line is about ('rsi', 'sma_50', ...).

    Renderers that already show a figure elsewhere (the PDF's Key metrics table) use the kind to
    skip lines that would repeat it, without matching on wording that may change.
    """

    def __init__(self):
        super().__init__()
        self.kinds = []

    def add(self, kind, text):
        self.append(text)
        self.kinds.append(kind)


class QuantitativeAnalyzer:
    # Whole-word matching, so "miss" doesn't fire on "mission" or "record" on "recorder".
    BULLISH_WORDS = [
        "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps", "jumped",
        "rally", "rallies", "rallied", "beat", "beats", "upgrade", "upgrades", "upgraded",
        "record", "raises", "raised", "outperform", "outperforms", "bullish", "tops", "breakout",
    ]  # fmt: skip
    BEARISH_WORDS = [
        "miss", "misses", "missed", "drop", "drops", "dropped", "plunge", "plunges", "plunged",
        "downgrade", "downgrades", "downgraded", "lawsuit", "sues", "sued", "falls", "fell",
        "slump", "slumps", "slumped", "cuts", "probe", "investigation", "recall", "bearish",
        "selloff", "sell-off", "warns", "tumble", "tumbles", "tumbled", "sinks", "sank",
    ]  # fmt: skip
    _BULL_RE = re.compile(r"\b(" + "|".join(map(re.escape, BULLISH_WORDS)) + r")\b")
    _BEAR_RE = re.compile(r"\b(" + "|".join(map(re.escape, BEARISH_WORDS)) + r")\b")

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

        self.price_pct_limit = self.thresholds.get("price_change_pct", 5.0)
        self.gap_pct_limit = self.thresholds.get("premarket_gap_pct", 3.0)
        self.sigma_limit = self.thresholds.get("unusual_move_sigma", 2.0)
        self.min_move_pct = self.thresholds.get("min_move_pct", 1.5)
        self.vol_multiplier = self.thresholds.get("volume_surge_multiplier", 2.0)
        self.rsi_oversold = self.thresholds.get("rsi_oversold", 30)
        self.rsi_overbought = self.thresholds.get("rsi_overbought", 70)
        self.rsi_bullish = self.thresholds.get("rsi_bullish", 50)
        self.near_52w_pct = self.thresholds.get("near_52w_pct", 3.0)
        self.earnings_days = self.thresholds.get("earnings_lookahead_days", 7)
        self.short_levels = self._levels("short_interest_levels_pct", [5, 10, 20], SHORT_INTEREST_LEVELS)
        self.days_to_cover_levels = self._levels("days_to_cover_levels", [2, 5, 10], DAYS_TO_COVER_LEVELS)
        self.put_call_levels = self._levels("put_call_levels", [0.5, 0.8, 1.2, 1.5], PUT_CALL_LEVELS)
        self.options_min_oi = self.thresholds.get("options_min_open_interest", 1000)
        self.calls_expiring_pct = self.thresholds.get("calls_expiring_week_pct", 50)
        self.runway_levels = self._levels("cash_runway_levels_months", [12, 24], RUNWAY_LEVELS)

    def _levels(self, key, default, levels):
        """Level boundaries from config; there must be exactly one fewer boundary than levels."""
        bounds = sorted(self.thresholds.get(key, default))
        if len(bounds) != len(levels) - 1:
            names = " / ".join(name for name, _ in levels)
            raise ValueError(
                f"config.yaml '{key}' needs {len(levels) - 1} numbers (levels: {names}), got {bounds}"
            )
        return bounds

    # ------------------------------------------------------------------ #
    # Indicators
    # ------------------------------------------------------------------ #
    def _compute_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds indicator columns to a copy of the candles. The raw candles are left untouched
        because the charts reuse them.
        """
        df = df.copy()

        # Baseline is the 20 sessions *before* each bar. Including the bar in its own average
        # pulls the ratio toward 1.0 and hides genuine surges.
        df["vol_sma_20"] = df["volume"].rolling(window=20).mean().shift(1)

        df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

        macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # "Typical move" = stdev of the previous 20 daily returns. It's what turns "moved 4%" into
        # "moved 2.5x its normal amount". Shifted for the same reason as the volume baseline: a
        # huge day shouldn't inflate its own yardstick.
        df["return_pct"] = df["close"].pct_change() * 100
        df["typical_move_pct"] = df["return_pct"].rolling(window=20).std().shift(1)

        # 52-week range over 252 sessions, using intraday highs/lows like quote sites do.
        # min_periods lets recently listed stocks still get a (shorter) range.
        df["high_52w"] = df["high"].rolling(window=252, min_periods=200).max()
        df["low_52w"] = df["low"].rolling(window=252, min_periods=200).min()

        return df

    def _analyze_sentiment(self, news: list) -> dict:
        """
        Classifies each article as positive, negative or neutral by counting bullish vs. bearish
        words in its headline and summary. The overall label needs a margin of two articles, so
        one ambiguous headline can't flip a stock to Bullish or Bearish on its own.
        """
        if not news:
            return {"score": 0, "label": "Neutral", "positive": 0, "negative": 0, "neutral": 0}

        positive = negative = neutral = 0
        for article in news:
            text = f"{article.get('headline') or ''} {article.get('summary') or ''}".lower()
            bull_hits = len(self._BULL_RE.findall(text))
            bear_hits = len(self._BEAR_RE.findall(text))
            if bull_hits > bear_hits:
                positive += 1
            elif bear_hits > bull_hits:
                negative += 1
            else:
                neutral += 1

        score = positive - negative
        if score >= 2:
            label = "Bullish"
        elif score <= -2:
            label = "Bearish"
        else:
            label = "Neutral"
        return {
            "score": score,
            "label": label,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        }

    # ------------------------------------------------------------------ #
    # Market backdrop
    # ------------------------------------------------------------------ #
    def analyze_market(self, market_data: dict) -> dict:
        """
        Summarizes the benchmarks (SPY, QQQ) and the VIX. Each stock's move is shown next to
        these so a stock-specific move can be told apart from the whole market moving.
        """
        summary = {}
        for symbol, data in (market_data or {}).items():
            candles = data.get("candles", pd.DataFrame())
            snapshot = data.get("premarket", {}) or {}
            if candles is None or candles.empty:
                continue

            last_close = _num(candles["close"].iloc[-1])
            prev_close = _num(candles["close"].iloc[-2]) if len(candles) > 1 else None
            month_ago = _num(candles["close"].iloc[-22]) if len(candles) > 22 else None
            summary[symbol] = {
                "last_close": last_close,
                "last_session_pct": _pct(last_close, prev_close),
                "premarket_price": snapshot.get("price"),
                "gap_pct": _pct(snapshot.get("price"), last_close),
                "return_1m_pct": _pct(last_close, month_ago),
            }
            if symbol == TNX_SYMBOL:
                yield_now, yield_prev = _yield_pct(last_close), _yield_pct(prev_close)
                summary[symbol]["yield_pct"] = yield_now
                # Yields move in basis points (0.01%), and a % change of a yield means little.
                summary[symbol]["last_session_bp"] = (
                    (yield_now - yield_prev) * 100 if None not in (yield_now, yield_prev) else None
                )
        return summary

    # ------------------------------------------------------------------ #
    # Watchlist
    # ------------------------------------------------------------------ #
    def analyze(self, pipeline_data: dict, market: dict = None, earnings: dict = None) -> dict:
        """
        Runs every rule over the watchlist.

        Returns {symbol: result}. A result is either {"error": reason} when the price history is
        unusable, or a dict with 'metrics', 'triggers', 'alerts' (most severe first), sentiment
        and the top headlines. 'signals' repeats 'triggers' as {"text", "kind"} dicts for renderers
        that filter by topic.
        """
        market = market or {}
        earnings = earnings or {}
        # QQQ is the closest benchmark for a tech-heavy watchlist; SPY is the fallback.
        benchmark = market.get("QQQ") or market.get("SPY") or {}

        return {
            symbol: self._analyze_symbol(data, benchmark, earnings.get(symbol))
            for symbol, data in pipeline_data.items()
        }

    def _analyze_symbol(self, data: dict, benchmark: dict, earnings_event: dict | None) -> dict:
        quote = data.get("quote", {}) or {}
        raw_candles = data.get("candles", pd.DataFrame())
        news = data.get("news", []) or []
        snapshot = data.get("premarket", {}) or {}
        fundamentals = data.get("fundamentals", {}) or {}

        # Repeat the fetcher's cleanup in case candles come from elsewhere (tests, a future source).
        if raw_candles is not None and not raw_candles.empty:
            raw_candles = raw_candles.dropna(subset=["close"])
        if raw_candles is None or raw_candles.empty or len(raw_candles) < 2:
            return {"error": "No price history returned"}

        df = self._compute_technicals(raw_candles)
        latest_bar = df.iloc[-1]  # most recent *finished* session
        prior_bar = df.iloc[-2]  # needed to detect crossovers

        last_close = _num(latest_bar["close"])
        typical = _num(latest_bar["typical_move_pct"])

        # Prefer the candle-derived move so it matches the chart and the other indicators;
        # Finnhub's quote is only a fallback.
        last_session_pct = _num(latest_bar["return_pct"])
        if last_session_pct is None:
            last_session_pct = quote.get("change_pct")
        last_sigma = last_session_pct / typical if (last_session_pct is not None and typical) else None

        premarket_price = snapshot.get("price")
        gap_pct = _pct(premarket_price, last_close)
        gap_sigma = gap_pct / typical if (gap_pct is not None and typical) else None

        volume_ratio = None
        if pd.notna(latest_bar["vol_sma_20"]) and latest_bar["vol_sma_20"] > 0:
            volume_ratio = float(latest_bar["volume"] / latest_bar["vol_sma_20"])

        rsi = _num(latest_bar["rsi_14"])
        sma = {n: _num(latest_bar[f"sma_{n}"]) for n in (20, 50, 200)}
        high_52w, low_52w = _num(latest_bar["high_52w"]), _num(latest_bar["low_52w"])

        return_1m = _pct(last_close, _num(df["close"].iloc[-22])) if len(df) > 22 else None
        bench_1m = benchmark.get("return_1m_pct")
        relative_1m = (return_1m - bench_1m) if (return_1m is not None and bench_1m is not None) else None

        metrics = {
            "last_close": last_close,
            "last_session_pct": last_session_pct,
            "last_session_sigma": last_sigma,
            "premarket_price": premarket_price,
            "premarket_session": snapshot.get("session"),
            "premarket_as_of": snapshot.get("as_of"),
            "premarket_volume": snapshot.get("premarket_volume"),
            "gap_pct": gap_pct,
            "gap_sigma": gap_sigma,
            "typical_move_pct": typical,
            "volume": _num(latest_bar["volume"]),
            "volume_ratio": volume_ratio,
            "rsi_14": rsi,
            "sma_20": sma[20],
            "sma_50": sma[50],
            "sma_200": sma[200],
            "pct_vs_sma_50": _pct(last_close, sma[50]),
            "pct_vs_sma_200": _pct(last_close, sma[200]),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_high_52w": _pct(last_close, high_52w),
            "pct_from_low_52w": _pct(last_close, low_52w),
            "macd": _num(latest_bar["macd"]),
            "macd_signal": _num(latest_bar["macd_signal"]),
            "return_1m_pct": return_1m,
            "relative_1m_pct": relative_1m,
            **fundamentals,
        }

        triggers = _SignalLog()  # descriptive lines, always shown on the stock's page
        alerts = []  # threshold breaches, eligible for the header

        # Order matters: alerts are sorted by severity below, and the sort is stable, so ties
        # keep the order these rules run in.
        self._gap_signals(snapshot, premarket_price, gap_pct, gap_sigma, benchmark, triggers, alerts)
        self._last_session_signals(last_session_pct, last_sigma, benchmark, triggers, alerts)
        self._volume_signals(volume_ratio, last_session_pct, triggers, alerts)
        self._rsi_signals(rsi, triggers, alerts)
        self._moving_average_signals(last_close, sma, prior_bar, triggers, alerts)
        self._range_signals(last_close, latest_bar, high_52w, low_52w, alerts)
        self._macd_signals(metrics, prior_bar, alerts)
        self._valuation_signals(metrics, last_close, triggers)
        self._short_interest_signals(metrics, triggers, alerts)
        self._options_signals(metrics, triggers, alerts)
        self._ownership_signals(metrics, triggers)
        self._balance_sheet_signals(metrics, triggers, alerts)
        self._earnings_signals(earnings_event, metrics, alerts)

        sentiment = self._analyze_sentiment(news)
        alerts.sort(key=lambda a: a["severity"], reverse=True)

        return {
            "price": last_close,
            "change_pct": last_session_pct,
            "metrics": metrics,
            "triggers": list(triggers),
            "signals": [{"text": t, "kind": k} for t, k in zip(triggers, triggers.kinds, strict=True)],
            "alerts": alerts,
            "sentiment": sentiment["label"],
            "sentiment_detail": sentiment,
            "top_news": news[:3],
        }

    # ------------------------------------------------------------------ #
    # Price moves
    # ------------------------------------------------------------------ #
    def _move_alert(self, pct, sigma, fixed_limit, what, when, market_pct=None):
        """
        Shared "is this move big?" rule for both the pre-market gap and the last session.

        Fires when EITHER:
          - the move is unusual for THIS stock: >= sigma_limit x its typical daily move, and at
            least min_move_pct so very calm stocks don't alert on 0.8% wiggles; OR
          - it crosses the fixed threshold AND is at least a normal-sized move for the stock
            (>= 1x typical). Without that second condition a biotech that swings 6% a day would
            alert on every routine 5% move.
        """
        if pct is None:
            return None

        big_relative = sigma is not None and abs(sigma) >= self.sigma_limit and abs(pct) >= self.min_move_pct
        big_fixed = abs(pct) >= fixed_limit and (sigma is None or abs(sigma) >= 1.0)
        if not (big_fixed or big_relative):
            return None

        severity = abs(pct) / fixed_limit
        if sigma is not None:
            severity = max(severity, abs(sigma) / self.sigma_limit)

        direction = UP if pct > 0 else DOWN
        verb = "up" if pct > 0 else "down"
        details = []
        if sigma is not None:
            details.append(f"{abs(sigma):.1f}x its typical daily move")
        if market_pct is not None:
            details.append(f"market {market_pct:+.1f}%")
        detail_text = f" ({'; '.join(details)})" if details else ""

        return {
            "text": f"{what} {verb} {abs(pct):.1f}% {when}{detail_text}",
            "direction": direction,
            "severity": round(float(severity), 2),
            "kind": "move",
        }

    def _gap_signals(self, snapshot, premarket_price, gap_pct, gap_sigma, benchmark, triggers, alerts):
        if gap_pct is None:
            triggers.add("premarket", "No pre-market trades yet today")
            return

        # The scheduled run sees pre-market trades, but manual runs later in the day see the
        # regular or after-hours session, so the wording follows whichever session it was.
        session = snapshot.get("session") or "pre-market"
        when = SESSION_WORDS.get(session, session)
        triggers.add("premarket", f"Trading at ${premarket_price:.2f} {when}, {gap_pct:+.2f}% vs. last close")

        what = "Gapping" if session == "pre-market" else "Trading"
        alert = self._move_alert(gap_pct, gap_sigma, self.gap_pct_limit, what, when, benchmark.get("gap_pct"))
        if alert:
            # Marks a move since the last close, which only news from after that close can explain.
            alert["window"] = "overnight"
            alerts.append(alert)

    def _last_session_signals(self, last_session_pct, last_sigma, benchmark, triggers, alerts):
        if last_session_pct is None:
            return

        direction = "Up" if last_session_pct > 0 else "Down"
        triggers.add("last_session", f"Last session: {last_session_pct:+.2f}% ({direction})")
        alert = self._move_alert(
            last_session_pct,
            last_sigma,
            self.price_pct_limit,
            "Closed",
            "last session",
            benchmark.get("last_session_pct"),
        )
        if alert:
            alerts.append(alert)

    def _volume_signals(self, volume_ratio, last_session_pct, triggers, alerts):
        if volume_ratio is None:
            return

        triggers.add("volume", f"Volume: {volume_ratio:.1f}x the 20-day average")
        if volume_ratio >= self.vol_multiplier:
            move = f" on a {last_session_pct:+.1f}% day" if last_session_pct is not None else ""
            alerts.append(
                {
                    "text": f"Volume {volume_ratio:.1f}x its 20-day average{move}",
                    # Heavy volume confirms whichever way the price went.
                    "direction": DOWN if (last_session_pct or 0) < 0 else UP,
                    "severity": round(volume_ratio / self.vol_multiplier, 2),
                    "kind": "volume",
                }
            )

    # ------------------------------------------------------------------ #
    # Momentum and trend
    # ------------------------------------------------------------------ #
    def _rsi_signals(self, rsi, triggers, alerts):
        if rsi is None:
            return

        if rsi > self.rsi_overbought:
            triggers.add("rsi", f"Potentially overbought (14 day RSI: {rsi:.1f})")
            alerts.append(
                {
                    "text": f"RSI {rsi:.0f}: overbought, stretched after a run-up",
                    "direction": CAUTION,
                    "kind": "rsi",
                    "severity": round(1 + (rsi - self.rsi_overbought) / 20, 2),
                }
            )
        elif rsi >= self.rsi_bullish:
            triggers.add("rsi", f"Bullish momentum (14 day RSI: {rsi:.1f})")
        elif rsi >= self.rsi_oversold:
            triggers.add("rsi", f"Bearish/weak momentum (14 day RSI: {rsi:.1f})")
        else:
            triggers.add("rsi", f"Potentially oversold (14 day RSI: {rsi:.1f})")
            alerts.append(
                {
                    "text": f"RSI {rsi:.0f}: oversold, selling may be exhausted",
                    "direction": CAUTION,
                    "kind": "rsi",
                    "severity": round(1 + (self.rsi_oversold - rsi) / 20, 2),
                }
            )

    def _moving_average_signals(self, last_close, sma, prior_bar, triggers, alerts):
        """Where the price sits vs. each moving average, fresh crossovers, and golden/death crosses."""
        prior_close = _num(prior_bar["close"])
        for n, cross_weight in ((20, 0.5), (50, 1.0), (200, 1.5)):
            level, prior_level = sma[n], _num(prior_bar[f"sma_{n}"])
            if level is None:
                continue

            side = "above" if last_close > level else "below"
            triggers.add(f"sma_{n}", f"Price {side} {n}-day average (${level:.2f})")
            if prior_level is None:
                continue

            crossed_up = prior_close <= prior_level and last_close > level
            crossed_down = prior_close >= prior_level and last_close < level
            if crossed_up or crossed_down:
                # Breaking DOWN through an average is weighted a little heavier: it's the move
                # holders most need to see.
                weight = cross_weight * (1.0 if crossed_down else 0.85)
                new_side = "above" if crossed_up else "below"
                alerts.append(
                    {
                        "text": f"Closed {new_side} its {n}-day average (${level:.2f})",
                        "direction": UP if crossed_up else DOWN,
                        "kind": "trend",
                        "severity": round(weight, 2),
                    }
                )

        # Golden / death cross: the 50-day crossing the 200-day.
        if (
            None not in (sma[50], sma[200])
            and pd.notna(prior_bar["sma_50"])
            and pd.notna(prior_bar["sma_200"])
        ):
            was_above = prior_bar["sma_50"] >= prior_bar["sma_200"]
            now_above = sma[50] >= sma[200]
            if was_above != now_above:
                alerts.append(
                    {
                        "text": (
                            "Golden cross: 50-day average moved above the 200-day"
                            if now_above
                            else "Death cross: 50-day average fell below the 200-day"
                        ),
                        "direction": UP if now_above else DOWN,
                        "kind": "trend",
                        "severity": 1.5,
                    }
                )

    def _range_signals(self, last_close, latest_bar, high_52w, low_52w, alerts):
        if high_52w is None or low_52w is None:
            return

        day_high = _num(latest_bar["high"]) or last_close
        day_low = _num(latest_bar["low"]) or last_close
        if day_high >= high_52w:
            alerts.append(
                {
                    "text": f"New 52-week high (${high_52w:.2f})",
                    "direction": UP,
                    "kind": "range",
                    "severity": 1.2,
                }
            )
        elif day_low <= low_52w:
            alerts.append(
                {
                    "text": f"New 52-week low (${low_52w:.2f})",
                    "direction": DOWN,
                    "kind": "range",
                    "severity": 1.5,
                }
            )
        else:
            from_high = _pct(last_close, high_52w)
            from_low = _pct(last_close, low_52w)
            if from_high is not None and abs(from_high) <= self.near_52w_pct:
                alerts.append(
                    {
                        "text": f"Within {abs(from_high):.1f}% of its 52-week high",
                        "direction": UP,
                        "kind": "range",
                        "severity": 0.6,
                    }
                )
            elif from_low is not None and from_low <= self.near_52w_pct:
                alerts.append(
                    {
                        "text": f"Within {from_low:.1f}% of its 52-week low",
                        "direction": DOWN,
                        "kind": "range",
                        "severity": 0.7,
                    }
                )

    def _macd_signals(self, metrics, prior_bar, alerts):
        """MACD crossing its signal line means short-term momentum has flipped direction."""
        if (
            None in (metrics["macd"], metrics["macd_signal"])
            or not pd.notna(prior_bar["macd"])
            or not pd.notna(prior_bar["macd_signal"])
        ):
            return

        was_above = prior_bar["macd"] >= prior_bar["macd_signal"]
        now_above = metrics["macd"] >= metrics["macd_signal"]
        if was_above != now_above:
            alerts.append(
                {
                    "text": f"MACD turned {'up' if now_above else 'down'}: short-term momentum flipped",
                    "direction": UP if now_above else DOWN,
                    "kind": "momentum",
                    "severity": 0.6,
                }
            )

    # ------------------------------------------------------------------ #
    # Fundamentals and positioning
    # ------------------------------------------------------------------ #
    def _valuation_signals(self, metrics, last_close, triggers):
        # Price forward P/E off the same last close as the rest of the page. A 'forward_pe' that
        # arrives in the fundamentals is Yahoo's fallback, used only when there are no estimates.
        forward_eps = metrics.get("forward_eps")
        if forward_eps and last_close:
            metrics["forward_pe"] = last_close / forward_eps

        # A negative P/E (expected loss) isn't a meaningful multiple, so blank it and say why.
        forward_pe = metrics.get("forward_pe")
        if forward_pe is not None and forward_pe < 0:
            metrics["forward_pe"] = None
            triggers.add("valuation", "No forward P/E: analysts expect a loss over the next 12 months")

    def _short_interest_signals(self, metrics, triggers, alerts):
        short_pct = metrics.get("short_pct_float")
        if short_pct is None:
            return

        short_level, short_meaning = level_of(short_pct, self.short_levels, SHORT_INTEREST_LEVELS)
        metrics["short_level"] = short_level

        days = metrics.get("short_ratio")
        days_level = None
        if days is not None:
            days_level = level_of(days, self.days_to_cover_levels, DAYS_TO_COVER_LEVELS)[0]
            metrics["days_to_cover_level"] = days_level

        change = metrics.get("short_change_pct")
        detail = (f", {days:.1f} days to cover ({days_level.lower()})" if days is not None else "") + (
            f", {change:+.0f}% vs. prior report" if change is not None else ""
        )
        triggers.add(
            "short_interest",
            f"Short interest {short_pct:.1f}% of float ({short_level.lower()}){detail}. {short_meaning}",
        )

        # Squeeze setup: a crowded short, or a large one that would take more than ~2 weeks of
        # normal volume to buy back.
        if short_level == "Very high" or (short_level == "High" and days_level == "Very high"):
            days_text = f", {days:.1f} days to cover" if days is not None else ""
            alerts.append(
                {
                    "text": (
                        f"Heavily shorted: {short_pct:.1f}% of float{days_text}. "
                        "Good news can trigger a short squeeze; bad news tends to hit harder"
                    ),
                    "direction": CAUTION,
                    "kind": "positioning",
                    "severity": 0.7,
                }
            )

    def _options_signals(self, metrics, triggers, alerts):
        call_oi, put_oi = metrics.get("call_oi"), metrics.get("put_oi")
        ratio = metrics.get("put_call_oi_ratio")
        if call_oi is not None and put_oi is not None:
            window = metrics.get("options_window_days", 30)
            text = f"Open interest, next {window} days: {_count(call_oi)} calls vs. {_count(put_oi)} puts"

            # A lopsided ratio on a thinly traded chain means little, so require some size first.
            if ratio is not None and call_oi + put_oi >= self.options_min_oi:
                pc_level, pc_meaning = level_of(ratio, self.put_call_levels, PUT_CALL_LEVELS)
                metrics["put_call_level"] = pc_level
                text += f". Put/call {ratio:.2f} ({pc_level.lower()}): {pc_meaning}"
                if pc_level == "Very bullish":
                    alerts.append(
                        {
                            "text": f"Options lean bullish: {1 / ratio:.1f} calls open for every put",
                            "direction": UP,
                            "kind": "positioning",
                            "severity": 0.6,
                        }
                    )
                elif pc_level == "Bearish":
                    alerts.append(
                        {
                            "text": (
                                f"Options lean bearish: {ratio:.1f} puts open for every call "
                                "(hedging or bets on a drop)"
                            ),
                            "direction": DOWN,
                            "kind": "positioning",
                            "severity": 0.6,
                        }
                    )
            elif ratio is not None:
                text += f" (put/call {ratio:.2f}; too few contracts to read much into)"
            triggers.add("open_interest", text)

        # When most near-dated calls expire at once, traders close or roll them and dealers unwind
        # their hedges. That can pin the price near big strikes beforehand and free it up after.
        week_calls, week_pct = metrics.get("call_oi_week"), metrics.get("call_oi_week_pct")
        if week_calls is not None and week_pct is not None and metrics.get("week_end"):
            friday = date.fromisoformat(metrics["week_end"])
            # The third Friday (day 15-21) is the standard monthly expiration.
            monthly = " (monthly expiration)" if 15 <= friday.day <= 21 else ""
            when = f"{friday:%a %b} {friday.day}{monthly}"
            text = f"{_count(week_calls)} calls ({week_pct:.0f}% of near-term calls) expire by {when}"
            triggers.add("calls_expiring", text)
            if week_pct >= self.calls_expiring_pct and week_calls >= self.options_min_oi:
                alerts.append(
                    {
                        "text": (
                            f"{text}: expect hedging flows and possible pinning near big strikes, "
                            "then a freer move once they roll off"
                        ),
                        "direction": CAUTION,
                        "kind": "positioning",
                        "severity": 0.6,
                    }
                )

    def _ownership_signals(self, metrics, triggers):
        inst_pct = metrics.get("institutional_pct")
        if inst_pct is not None:
            triggers.add("ownership", f"Institutions hold {inst_pct:.1f}% of shares")

    def _balance_sheet_signals(self, metrics, triggers, alerts):
        # Gross debt alone says little; what matters is the cash held against it and whether the
        # business generates cash or burns it.
        debt, cash = metrics.get("total_debt_m"), metrics.get("total_cash_m")
        op_cash = metrics.get("operating_cashflow_m")

        if debt is not None and cash is not None:
            net_debt = debt - cash
            metrics["net_debt_m"] = net_debt
            label = "Net debt" if net_debt > 0 else "Net cash"
            triggers.add(
                "net_debt", f"{label} {_dollars(net_debt)} (debt {_dollars(debt)}, cash {_dollars(cash)})"
            )

        if op_cash is not None and op_cash < 0 and cash is not None:
            months = cash / -op_cash * 12
            runway_level, runway_meaning = level_of(months, self.runway_levels, RUNWAY_LEVELS)
            metrics["cash_runway_months"] = months
            metrics["cash_runway_level"] = runway_level
            triggers.add(
                "cash_runway",
                f"Burning {_dollars(op_cash)} a year from operations: cash lasts {runway_text(months)} "
                f"at that rate ({runway_level.lower()}), {runway_meaning}",
            )
            if runway_level == "Short":
                alerts.append(
                    {
                        "text": (
                            f"Cash runway {runway_text(months)}: {_dollars(cash)} of cash, burning "
                            f"{_dollars(op_cash)} a year. A share sale or new debt is likely soon"
                        ),
                        "direction": CAUTION,
                        "kind": "balance_sheet",
                        "severity": 0.8,
                    }
                )

    def _earnings_signals(self, event, metrics, alerts):
        if not event or not event.get("date"):
            return

        today = pd.Timestamp.now(tz="America/New_York").date()
        days_until = (pd.Timestamp(event["date"]).date() - today).days
        metrics["earnings_date"] = event["date"]
        metrics["earnings_hour"] = event.get("hour")
        metrics["earnings_days"] = days_until

        if 0 <= days_until <= self.earnings_days:
            when = {"bmo": "before the open", "amc": "after the close"}.get(event.get("hour"))
            day = pd.Timestamp(event["date"])
            if days_until == 0:
                relative = "today"
            elif days_until == 1:
                relative = "tomorrow"
            else:
                relative = f"in {days_until} days"
            alerts.append(
                {
                    "text": (
                        f"Earnings {day:%a %b} {day.day}{', ' + when if when else ''} ({relative}): "
                        "expect a bigger-than-usual move"
                    ),
                    "direction": CAUTION,
                    "kind": "event",
                    # Sooner is more urgent: 1.0 a week out, 2.0 on the day.
                    "severity": round(1 + (self.earnings_days - days_until) / max(self.earnings_days, 1), 2),
                }
            )

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #
    @staticmethod
    def rank_stocks(results: dict, min_severity: float = 1.0) -> list:
        """
        Builds the report header: every stock with at least one alert at or above
        `min_severity`, most urgent first. Alerts stay grouped under their stock, so a stock with
        three problems reads as one story rather than three scattered lines.

        A stock's tone comes from its strongest directional alert: 'down' (red) or 'up' (green),
        or 'caution' (amber) when only non-directional alerts such as RSI or earnings fired.
        """
        ranked = []
        for symbol, result in results.items():
            headline = [a for a in result.get("alerts", []) if a["severity"] >= min_severity]
            if not headline:
                continue

            directional = [a for a in headline if a["direction"] in (UP, DOWN)]
            tone = directional[0]["direction"] if directional else CAUTION
            # Worst alert plus 30% of the rest, so several simultaneous problems outrank a single
            # borderline one.
            score = headline[0]["severity"] + 0.3 * sum(a["severity"] for a in headline[1:])
            ranked.append(
                {
                    "symbol": symbol,
                    "name": result.get("metrics", {}).get("name"),
                    "tone": tone,
                    "score": round(score, 2),
                    "top_severity": headline[0]["severity"],
                    "gap_pct": result.get("metrics", {}).get("gap_pct"),
                    "alerts": headline,
                }
            )
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked
