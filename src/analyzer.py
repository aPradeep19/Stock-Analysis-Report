import bisect
import re
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Every alert carries a severity score. 1.0 means "just crossed its threshold";
# 2.0 means "twice as extreme as the threshold". The report's header page only shows
# alerts at or above `header_min_severity` (config.yaml), so minor notes stay on the
# stock's own page instead of crowding the top of the email.
UP, DOWN, CAUTION = "up", "down", "caution"
SESSION_WORDS = {"pre-market": "pre-market", "regular": "so far today", "after-hours": "after hours"}


def _pct(new, old):
    """Percent change from old to new, or None if either side is missing."""
    if new is None or old is None or pd.isna(new) or pd.isna(old) or old == 0:
        return None
    return float((new - old) / old * 100)


def _num(value):
    """Converts pandas/numpy numbers to plain floats and NaN to None (easier to format and to test)."""
    if value is None or pd.isna(value):
        return None
    return float(value)


# Positioning levels. The boundaries live in config.yaml; the words live here, so the report's
# guide and each stock's signals always say the same thing. These describe how traders are
# betting, not how fast the business is growing: they hint at near-term volatility and sentiment.
SHORT_INTEREST_LEVELS = (
    ("Low", "Few bets against the stock; the price mostly follows the business."),
    ("Moderate", "Some skepticism; normal for fast-growing or volatile stocks."),
    ("High", "A sizable bet on a decline. Moves get bigger both ways, and good news can force shorts to buy back."),
    ("Very high", "A crowded short: squeeze potential on good news, but many traders expect growth or "
                  "the valuation to disappoint."),
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
    ("Bearish", "put-heavy, strong demand for downside protection. Contrarians sometimes read extreme "
                "pessimism as a sign of a bottom."),
)


def level_of(value, bounds, levels):
    """The (name, meaning) a value falls in. Bounds [5, 10, 20] split values into <5, 5-10, 10-20, 20+."""
    return levels[bisect.bisect_right(bounds, value)]


def _count(value):
    """Short form for big counts in signal text: 106425 -> '106K'."""
    for size, suffix in ((1e6, "M"), (1e3, "K")):
        if value >= size:
            return f"{value / size:.{0 if value >= size * 10 else 1}f}{suffix}"
    return f"{value:,.0f}"


class QuantitativeAnalyzer:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

        # Load risk boundaries from config
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

    def _levels(self, key, default, levels):
        """Reads a list of level boundaries from config; it needs one fewer number than there are levels."""
        bounds = sorted(self.thresholds.get(key, default))
        if len(bounds) != len(levels) - 1:
            names = " / ".join(name for name, _ in levels)
            raise ValueError(f"config.yaml '{key}' needs {len(levels) - 1} numbers (levels: {names}), got {bounds}")
        return bounds

    # ------------------------------------------------------------------ #
    # Indicator math
    # ------------------------------------------------------------------ #
    def _compute_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard quantitative indicators to the historical DataFrame.
        Works on a copy so the raw candles in pipeline_data stay untouched (charts reuse them).
        """
        df = df.copy()

        # 1. Volume baseline: average of the 20 sessions BEFORE each bar.
        #    The original version included the current bar in its own average, which pulls the
        #    ratio toward 1.0 and hides surges. shift(1) compares today against a clean baseline.
        df['vol_sma_20'] = df['volume'].rolling(window=20).mean().shift(1)

        # 2. 14-Period Relative Strength Index (RSI)
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()

        # 3. Price Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # 4. MACD (12/26/9): a momentum-turn detector. When the MACD line crosses its signal
        #    line, short-term momentum has flipped direction.
        macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # 5. Daily % returns and the stock's "typical move": the standard deviation of its last
        #    20 daily returns. This is what turns "moved 4%" into "moved 2.5x its normal amount".
        #    shift(1) again, so a huge day doesn't inflate its own yardstick.
        df['return_pct'] = df['close'].pct_change() * 100
        df['typical_move_pct'] = df['return_pct'].rolling(window=20).std().shift(1)

        # 6. 52-week range from the last 252 trading sessions (one year of trading days).
        #    Highs/lows use the intraday high/low, like every quote site does.
        df['high_52w'] = df['high'].rolling(window=252, min_periods=200).max()
        df['low_52w'] = df['low'].rolling(window=252, min_periods=200).min()

        return df

    # Bullish/bearish vocabularies. Whole-word matching (\b) avoids false hits such as
    # "miss" inside "mission" or "record" inside "recorder", which the substring version caught.
    BULLISH_WORDS = [
        "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps", "jumped",
        "rally", "rallies", "rallied", "beat", "beats", "upgrade", "upgrades", "upgraded",
        "record", "raises", "raised", "outperform", "outperforms", "bullish", "tops", "breakout",
    ]
    BEARISH_WORDS = [
        "miss", "misses", "missed", "drop", "drops", "dropped", "plunge", "plunges", "plunged",
        "downgrade", "downgrades", "downgraded", "lawsuit", "sues", "sued", "falls", "fell",
        "slump", "slumps", "slumped", "cuts", "probe", "investigation", "recall", "bearish",
        "selloff", "sell-off", "warns", "tumble", "tumbles", "tumbled", "sinks", "sank",
    ]
    _BULL_RE = re.compile(r"\b(" + "|".join(map(re.escape, BULLISH_WORDS)) + r")\b")
    _BEAR_RE = re.compile(r"\b(" + "|".join(map(re.escape, BEARISH_WORDS)) + r")\b")

    def _analyze_sentiment(self, news: list) -> dict:
        """
        Scores each article as positive, negative or neutral by counting bullish vs bearish
        words in its headline and summary. The label needs a margin of 2 articles, so a single
        ambiguous headline can't flip a stock to "Bullish" or "Bearish" on its own.
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
        return {"score": score, "label": label,
                "positive": positive, "negative": negative, "neutral": neutral}

    # ------------------------------------------------------------------ #
    # Alert rules
    # ------------------------------------------------------------------ #
    def _move_alert(self, pct, sigma, fixed_limit, what, when, market_pct=None):
        """
        Shared rule for "is this price move big?" (used for the pre-market gap AND yesterday's move).

        Fires when EITHER:
          - it is unusually large for THIS stock: >= 2x its typical daily move (sigma) and at
            least min_move_pct, so super-calm stocks don't alert on 0.8% wiggles, OR
          - it crosses the fixed threshold (e.g. 5%) AND is at least a normal-sized move for the
            stock (>= 1x typical). That second condition stops a biotech that swings 6% a day from
            alerting on every routine 5% move.
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
        return {"text": f"{what} {verb} {abs(pct):.1f}% {when}{detail_text}",
                "direction": direction, "severity": round(float(severity), 2), "kind": "move"}

    def analyze_market(self, market_data: dict) -> dict:
        """
        Summarizes the benchmark ETFs (SPY = S&P 500, QQQ = Nasdaq-100) and the VIX "fear gauge".
        Every stock's move is reported next to this, so you can tell a stock-specific move from
        the whole market moving.
        """
        summary = {}
        for symbol, data in (market_data or {}).items():
            candles = data.get("candles", pd.DataFrame())
            snapshot = data.get("premarket", {}) or {}
            if candles is None or candles.empty:
                continue
            last_close = _num(candles['close'].iloc[-1])
            prev_close = _num(candles['close'].iloc[-2]) if len(candles) > 1 else None
            entry = {
                "last_close": last_close,
                "last_session_pct": _pct(last_close, prev_close),
                "premarket_price": snapshot.get("price"),
                "gap_pct": _pct(snapshot.get("price"), last_close),
                "return_1m_pct": _pct(last_close, _num(candles['close'].iloc[-22])) if len(candles) > 22 else None,
            }
            summary[symbol] = entry
        return summary

    def analyze(self, pipeline_data: dict, market: dict = None, earnings: dict = None) -> dict:
        """
        Evaluates the entire watchlist against the configuration thresholds.

        Returns {symbol: result} where each result holds three layers:
          'metrics'  - raw numbers (for tables, the PDF page, and tests)
          'triggers' - plain-English descriptions of every indicator (always present)
          'alerts'   - only the things that crossed a threshold, each with a direction and severity
        """
        market = market or {}
        earnings = earnings or {}
        # QQQ is the closest benchmark for this tech-heavy watchlist; fall back to SPY.
        benchmark = market.get("QQQ") or market.get("SPY") or {}

        results = {}
        for symbol, data in pipeline_data.items():
            quote = data.get("quote", {}) or {}
            raw_candles = data.get("candles", pd.DataFrame())
            news = data.get("news", []) or []
            snapshot = data.get("premarket", {}) or {}
            fundamentals = data.get("fundamentals", {}) or {}

            # Skip if the price history failed; record why so the report can say so
            # Same cleanup as the fetcher, in case candles arrive from somewhere else (tests,
            # a future data source): rows without a closing price are dropped.
            if raw_candles is not None and not raw_candles.empty:
                raw_candles = raw_candles.dropna(subset=["close"])
            if raw_candles is None or raw_candles.empty or len(raw_candles) < 2:
                results[symbol] = {"error": "No price history returned"}
                continue

            df = self._compute_technicals(raw_candles)
            latest_bar = df.iloc[-1]  # The most recent closed trading session
            prior_bar = df.iloc[-2]   # The session before it, used to detect crossovers

            last_close = _num(latest_bar['close'])
            typical = _num(latest_bar['typical_move_pct'])

            # Last session's move. Finnhub's quote and the candles agree; prefer the candle math
            # so the number always matches the chart and the other indicators.
            last_session_pct = _num(latest_bar['return_pct'])
            if last_session_pct is None:
                last_session_pct = quote.get("change_pct")
            last_sigma = last_session_pct / typical if (last_session_pct is not None and typical) else None

            # Pre-market gap vs. last close
            premarket_price = snapshot.get("price")
            gap_pct = _pct(premarket_price, last_close)
            gap_sigma = gap_pct / typical if (gap_pct is not None and typical) else None

            volume_ratio = None
            if pd.notna(latest_bar['vol_sma_20']) and latest_bar['vol_sma_20'] > 0:
                volume_ratio = float(latest_bar['volume'] / latest_bar['vol_sma_20'])

            rsi = _num(latest_bar['rsi_14'])
            sma = {n: _num(latest_bar[f'sma_{n}']) for n in (20, 50, 200)}
            high_52w, low_52w = _num(latest_bar['high_52w']), _num(latest_bar['low_52w'])

            return_1m = _pct(last_close, _num(df['close'].iloc[-22])) if len(df) > 22 else None
            bench_1m = benchmark.get("return_1m_pct")

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
                "volume": _num(latest_bar['volume']),
                "volume_ratio": volume_ratio,
                "rsi_14": rsi,
                "sma_20": sma[20], "sma_50": sma[50], "sma_200": sma[200],
                "pct_vs_sma_50": _pct(last_close, sma[50]),
                "pct_vs_sma_200": _pct(last_close, sma[200]),
                "high_52w": high_52w, "low_52w": low_52w,
                "pct_from_high_52w": _pct(last_close, high_52w),
                "pct_from_low_52w": _pct(last_close, low_52w),
                "macd": _num(latest_bar['macd']), "macd_signal": _num(latest_bar['macd_signal']),
                "return_1m_pct": return_1m,
                "relative_1m_pct": (return_1m - bench_1m) if (return_1m is not None and bench_1m is not None) else None,
                **fundamentals,
            }

            triggers = []  # descriptive lines, always shown on the stock's page
            alerts = []    # threshold breaches, ranked into the header

            # ---- Pre-market gap ------------------------------------------------
            if gap_pct is not None:
                # The 7 AM Central run sees pre-market trades. Manual runs later in the day see the
                # regular session or after-hours trades, so the wording follows the session.
                session = snapshot.get("session") or "pre-market"
                when = SESSION_WORDS.get(session, session)
                triggers.append(f"Trading at ${premarket_price:.2f} {when}, {gap_pct:+.2f}% vs. last close")
                alert = self._move_alert(gap_pct, gap_sigma, self.gap_pct_limit,
                                         "Gapping" if session == "pre-market" else "Trading",
                                         when, benchmark.get("gap_pct"))
                if alert:
                    alerts.append(alert)
            else:
                triggers.append("No pre-market trades yet today")

            # ---- Daily Price Volatility (last session) ------------------------
            if last_session_pct is not None:
                direction = "Up" if last_session_pct > 0 else "Down"
                triggers.append(f"Last session: {last_session_pct:+.2f}% ({direction})")
                alert = self._move_alert(last_session_pct, last_sigma, self.price_pct_limit,
                                         "Closed", "last session", benchmark.get("last_session_pct"))
                if alert:
                    alerts.append(alert)

            # ---- Volume Anomaly -----------------------------------------------
            if volume_ratio is not None:
                triggers.append(f"Volume: {volume_ratio:.1f}x the 20-day average")
                if volume_ratio >= self.vol_multiplier:
                    move = f" on a {last_session_pct:+.1f}% day" if last_session_pct is not None else ""
                    alerts.append({
                        "text": f"Volume {volume_ratio:.1f}x its 20-day average{move}",
                        # Heavy volume confirms whichever way the price went
                        "direction": DOWN if (last_session_pct or 0) < 0 else UP,
                        "severity": round(volume_ratio / self.vol_multiplier, 2), "kind": "volume"})

            # ---- RSI Extremes -------------------------------------------------
            if rsi is not None:
                if rsi > self.rsi_overbought:
                    triggers.append(f"Potentially overbought (14 day RSI: {rsi:.1f})")
                    alerts.append({"text": f"RSI {rsi:.0f}: overbought, stretched after a run-up",
                                   "direction": CAUTION, "kind": "rsi",
                                   "severity": round(1 + (rsi - self.rsi_overbought) / 20, 2)})
                elif rsi >= self.rsi_bullish:
                    triggers.append(f"Bullish momentum (14 day RSI: {rsi:.1f})")
                elif rsi >= self.rsi_oversold:
                    triggers.append(f"Bearish/weak momentum (14 day RSI: {rsi:.1f})")
                else:
                    triggers.append(f"Potentially oversold (14 day RSI: {rsi:.1f})")
                    alerts.append({"text": f"RSI {rsi:.0f}: oversold, selling may be exhausted",
                                   "direction": CAUTION, "kind": "rsi",
                                   "severity": round(1 + (self.rsi_oversold - rsi) / 20, 2)})

            # ---- Price vs Moving Averages (+ fresh crossovers) -----------------
            prior_close = _num(prior_bar['close'])
            for n, cross_weight in ((20, 0.5), (50, 1.0), (200, 1.5)):
                level, prior_level = sma[n], _num(prior_bar[f'sma_{n}'])
                if level is None:
                    continue
                side = "above" if last_close > level else "below"
                triggers.append(f"Price {side} {n}-day average (${level:.2f})")
                if prior_level is None:
                    continue
                crossed_up = prior_close <= prior_level and last_close > level
                crossed_down = prior_close >= prior_level and last_close < level
                if crossed_up or crossed_down:
                    # A break DOWN through a long average is weighted a bit heavier: it's the
                    # move people holding the stock most need to see.
                    weight = cross_weight * (1.0 if crossed_down else 0.85)
                    alerts.append({
                        "text": f"Closed {'above' if crossed_up else 'below'} its {n}-day average (${level:.2f})",
                        "direction": UP if crossed_up else DOWN, "kind": "trend",
                        "severity": round(weight, 2)})

            # ---- Golden / death cross (50-day crossing the 200-day) ------------
            if None not in (sma[50], sma[200]) and pd.notna(prior_bar['sma_50']) and pd.notna(prior_bar['sma_200']):
                was_above = prior_bar['sma_50'] >= prior_bar['sma_200']
                now_above = sma[50] >= sma[200]
                if was_above != now_above:
                    alerts.append({
                        "text": "Golden cross: 50-day average moved above the 200-day" if now_above
                                else "Death cross: 50-day average fell below the 200-day",
                        "direction": UP if now_above else DOWN, "kind": "trend", "severity": 1.5})

            # ---- 52-week range --------------------------------------------------
            if high_52w is not None and low_52w is not None:
                day_high = _num(latest_bar['high']) or last_close
                day_low = _num(latest_bar['low']) or last_close
                if day_high >= high_52w:
                    alerts.append({"text": f"New 52-week high (${high_52w:.2f})", "direction": UP,
                                   "kind": "range", "severity": 1.2})
                elif day_low <= low_52w:
                    alerts.append({"text": f"New 52-week low (${low_52w:.2f})", "direction": DOWN,
                                   "kind": "range", "severity": 1.5})
                else:
                    from_high = _pct(last_close, high_52w)
                    from_low = _pct(last_close, low_52w)
                    if from_high is not None and abs(from_high) <= self.near_52w_pct:
                        alerts.append({"text": f"Within {abs(from_high):.1f}% of its 52-week high",
                                       "direction": UP, "kind": "range", "severity": 0.6})
                    elif from_low is not None and from_low <= self.near_52w_pct:
                        alerts.append({"text": f"Within {from_low:.1f}% of its 52-week low",
                                       "direction": DOWN, "kind": "range", "severity": 0.7})

            # ---- MACD momentum turn ---------------------------------------------
            if None not in (metrics["macd"], metrics["macd_signal"]) and pd.notna(prior_bar['macd']) \
                    and pd.notna(prior_bar['macd_signal']):
                was_above = prior_bar['macd'] >= prior_bar['macd_signal']
                now_above = metrics["macd"] >= metrics["macd_signal"]
                if was_above != now_above:
                    alerts.append({"text": f"MACD turned {'up' if now_above else 'down'}: short-term momentum flipped",
                                   "direction": UP if now_above else DOWN, "kind": "momentum", "severity": 0.6})

            # ---- Valuation: forward P/E (next 12 months) -------------------------
            # Priced off the same last close as the rest of the page. forward_pe from the
            # fundamentals is only Yahoo's fallback, used when there are no analyst estimates.
            forward_eps = metrics.get("forward_eps")
            if forward_eps and last_close:
                metrics["forward_pe"] = last_close / forward_eps
            # A negative P/E (expected loss) isn't a meaningful valuation, so the field is blanked
            # and the signal says why instead.
            forward_pe = metrics.get("forward_pe")
            if forward_pe is not None and forward_pe < 0:
                metrics["forward_pe"] = None
                triggers.append("No forward P/E: analysts expect a loss over the next 12 months")

            # ---- Short interest (reported twice a month) -----------------------
            short_pct = metrics.get("short_pct_float")
            if short_pct is not None:
                short_level, short_meaning = level_of(short_pct, self.short_levels, SHORT_INTEREST_LEVELS)
                metrics["short_level"] = short_level
                days = metrics.get("short_ratio")
                days_level = None
                if days is not None:
                    days_level = level_of(days, self.days_to_cover_levels, DAYS_TO_COVER_LEVELS)[0]
                    metrics["days_to_cover_level"] = days_level
                change = metrics.get("short_change_pct")
                detail = (f", {days:.1f} days to cover ({days_level.lower()})" if days is not None else "") + \
                         (f", {change:+.0f}% vs. prior report" if change is not None else "")
                triggers.append(f"Short interest {short_pct:.1f}% of float ({short_level.lower()}){detail}. "
                                f"{short_meaning}")
                # Squeeze setup: a crowded short, or a big one that would take shorts more than
                # ~2 weeks of normal volume to buy back
                if short_level == "Very high" or (short_level == "High" and days_level == "Very high"):
                    alerts.append({"text": f"Heavily shorted: {short_pct:.1f}% of float"
                                           + (f", {days:.1f} days to cover" if days is not None else "")
                                           + ". Good news can trigger a short squeeze; bad news tends to hit harder",
                                   "direction": CAUTION, "kind": "positioning", "severity": 0.7})

            # ---- Options open interest (near-dated) ----------------------------
            call_oi, put_oi = metrics.get("call_oi"), metrics.get("put_oi")
            ratio = metrics.get("put_call_oi_ratio")
            if call_oi is not None and put_oi is not None:
                window = metrics.get("options_window_days", 30)
                text = f"Open interest, next {window} days: {_count(call_oi)} calls vs. {_count(put_oi)} puts"
                # A lopsided ratio on a thinly traded chain means little, so it needs some volume
                if ratio is not None and call_oi + put_oi >= self.options_min_oi:
                    pc_level, pc_meaning = level_of(ratio, self.put_call_levels, PUT_CALL_LEVELS)
                    metrics["put_call_level"] = pc_level
                    text += f". Put/call {ratio:.2f} ({pc_level.lower()}): {pc_meaning}"
                    if pc_level == "Very bullish":
                        alerts.append({"text": f"Options lean bullish: {1 / ratio:.1f} calls open for every put",
                                       "direction": UP, "kind": "positioning", "severity": 0.6})
                    elif pc_level == "Bearish":
                        alerts.append({"text": f"Options lean bearish: {ratio:.1f} puts open for every call "
                                               f"(hedging or bets on a drop)",
                                       "direction": DOWN, "kind": "positioning", "severity": 0.6})
                elif ratio is not None:
                    text += f" (put/call {ratio:.2f}; too few contracts to read much into)"
                triggers.append(text)

            # ---- Upcoming earnings (event risk) ---------------------------------
            event = earnings.get(symbol)
            if event and event.get("date"):
                days_until = (pd.Timestamp(event["date"]).date() - pd.Timestamp.now(tz="America/New_York").date()).days
                metrics["earnings_date"] = event["date"]
                metrics["earnings_hour"] = event.get("hour")
                metrics["earnings_days"] = days_until
                if 0 <= days_until <= self.earnings_days:
                    when = {"bmo": "before the open", "amc": "after the close"}.get(event.get("hour"))
                    day = pd.Timestamp(event["date"])
                    relative = "today" if days_until == 0 else ("tomorrow" if days_until == 1 else f"in {days_until} days")
                    alerts.append({
                        "text": f"Earnings {day:%a %b} {day.day}{', ' + when if when else ''} ({relative}): "
                                f"expect a bigger-than-usual move",
                        "direction": CAUTION, "kind": "event",
                        # Sooner = more urgent: 1.0 a week out, 2.0 on the day
                        "severity": round(1 + (self.earnings_days - days_until) / max(self.earnings_days, 1), 2)})

            sentiment = self._analyze_sentiment(news)
            alerts.sort(key=lambda a: a["severity"], reverse=True)

            results[symbol] = {
                "price": last_close,
                "change_pct": last_session_pct,
                "metrics": metrics,
                "triggers": triggers,
                "alerts": alerts,
                "sentiment": sentiment["label"],
                "sentiment_detail": sentiment,
                "top_news": news[:3],
            }

        return results

    @staticmethod
    def rank_stocks(results: dict, min_severity: float = 1.0) -> list:
        """
        Builds the report's header: every stock with at least one alert at or above
        `min_severity`, most urgent stock first. Alerts stay grouped under their stock, so a
        stock with three problems reads as one story instead of three scattered lines.

        The stock's overall tone comes from its strongest up/down alert:
        'down' (red), 'up' (green), or 'caution' (amber) when only RSI/earnings fired.
        """
        ranked = []
        for symbol, result in results.items():
            headline = [a for a in result.get("alerts", []) if a["severity"] >= min_severity]
            if not headline:
                continue
            directional = [a for a in headline if a["direction"] in (UP, DOWN)]
            tone = directional[0]["direction"] if directional else CAUTION
            # Stock score = its worst alert + 30% of the rest, so a stock with several
            # problems at once outranks one with a single borderline alert.
            score = headline[0]["severity"] + 0.3 * sum(a["severity"] for a in headline[1:])
            ranked.append({
                "symbol": symbol,
                "name": result.get("metrics", {}).get("name"),
                "tone": tone,
                "score": round(score, 2),
                "top_severity": headline[0]["severity"],
                "gap_pct": result.get("metrics", {}).get("gap_pct"),
                "alerts": headline,
            })
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked
