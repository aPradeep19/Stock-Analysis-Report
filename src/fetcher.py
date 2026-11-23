"""
Market data access.

Finnhub (free tier) supplies quotes, news, company profiles, a few fundamentals and the earnings
calendar. Everything Finnhub's free tier lacks or gets wrong comes from yfinance instead: daily
price history, extended-hours trades, forward estimates, short interest, options open interest,
balance-sheet figures and the market-mover screens.
"""

import time
from datetime import datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = clock_time(9, 30)
REGULAR_CLOSE = clock_time(16, 0)


class FinnhubFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    def _get(self, endpoint: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        params["token"] = self.api_key

        response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=10)

        # The free tier allows 60 calls/minute; waiting out the window is simpler than queueing.
        if response.status_code == 429:
            print("\n[WARN] Rate limit reached. Sleeping for 60 seconds...")
            time.sleep(60)
            return self._get(endpoint, params)

        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Prices
    # ------------------------------------------------------------------ #
    def get_quote(self, symbol: str) -> dict:
        """
        Latest regular-session quote from Finnhub.

        Finnhub's short keys: c = last price, d = $ change, dp = % change,
        h/l/o = day high/low/open, pc = previous close.
        """
        data = self._get("/quote", {"symbol": symbol})
        return {
            "current_price": data.get("c"),
            "change": data.get("d"),
            "change_pct": data.get("dp"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "prev_close": data.get("pc"),
        }

    def get_historical_candles(self, symbol: str, days_back: int = 90) -> pd.DataFrame:
        """
        Daily OHLCV bars from yfinance (Finnhub's candle endpoint is paywalled).

        Returns a tz-naive, date-indexed DataFrame with lowercase columns, or an empty
        DataFrame on any failure.
        """
        try:
            ticker = yf.Ticker(symbol)

            # yfinance's `end` is exclusive, so ask for tomorrow to include today's bar.
            # drop_unfinished_bar() removes it again if the session hasn't closed yet.
            now = datetime.now(MARKET_TZ)
            end_date = now.date() + timedelta(days=1)
            start_date = now.date() - timedelta(days=days_back)

            df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            if df.empty:
                return pd.DataFrame()

            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df = df[["open", "high", "low", "close", "volume"]]

            # Yahoo occasionally emits a price-less row (e.g. a dividend dated on a day that hasn't
            # traded yet). Without a close it's useless, so drop it, and backfill any missing
            # open/high/low from the close so downstream comparisons never see NaN.
            df = df.dropna(subset=["close"])
            for column in ("open", "high", "low"):
                df[column] = df[column].fillna(df["close"])
            df["volume"] = df["volume"].fillna(0)
            df.index = df.index.tz_localize(None)

            return self.drop_unfinished_bar(df.sort_index(), now)

        except Exception as e:
            print(f"[WARN] Failed to fetch historical data for {symbol} via yfinance: {e}")
            return pd.DataFrame()

    @staticmethod
    def drop_unfinished_bar(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """
        Drops today's bar until the 4 PM close, so "last close" always means a finished session:
          - 8:00 AM run -> yesterday's close (the pre-market gap is measured from it)
          - 2:00 PM run -> yesterday's close (the move "so far today" is measured from it)
          - 6:00 PM run -> today's close (the after-hours move is measured from it)
        """
        if not df.empty and df.index[-1].date() == now.date() and now.time() < REGULAR_CLOSE:
            return df.iloc[:-1]
        return df

    def get_premarket_snapshot(self, symbol: str) -> dict:
        """
        Latest price traded *today*, including extended hours.

        Finnhub's free /quote only tracks the 9:30-4:00 session, so at 8 AM it still shows
        yesterday's close. Yahoo's 1-minute bars with prepost=True include pre-market trades
        (4:00-9:30 AM ET).

        Returns:
          price             last traded price today (None if nothing has traded yet)
          as_of             timestamp of that trade, New York time
          session           "pre-market", "regular" or "after-hours"
          premarket_volume  shares traded today before 9:30 AM
          traded_today      False on weekends/holidays; main.py uses it to skip the email
        """
        empty = {
            "price": None,
            "as_of": None,
            "session": None,
            "premarket_volume": None,
            "traded_today": False,
        }

        try:
            # Around 4 AM, Yahoo's "1d" window can still point at yesterday, so pull a few days
            # and filter to today ourselves.
            bars = yf.Ticker(symbol).history(period="5d", interval="1m", prepost=True)
        except Exception as e:
            print(f"[WARN] Failed to fetch pre-market data for {symbol} via yfinance: {e}")
            return empty

        if bars.empty:
            return empty

        # Normalize to New York time so "today" and "before 9:30" mean the same thing whether
        # this runs locally or on GitHub's UTC runners.
        index = bars.index
        index = index.tz_localize(MARKET_TZ) if index.tz is None else index.tz_convert(MARKET_TZ)
        bars = bars.set_axis(index)

        today = datetime.now(MARKET_TZ).date()
        todays_bars = bars[bars.index.date == today]
        if todays_bars.empty:
            return empty  # weekend, holiday, or before 4 AM

        last_ts = todays_bars.index[-1]
        if last_ts.time() < REGULAR_OPEN:
            session = "pre-market"
        elif last_ts.time() < REGULAR_CLOSE:
            session = "regular"
        else:
            session = "after-hours"

        premarket_bars = todays_bars[todays_bars.index.time < REGULAR_OPEN]

        return {
            "price": float(todays_bars["Close"].iloc[-1]),
            "as_of": last_ts.to_pydatetime(),
            "session": session,
            "premarket_volume": int(premarket_bars["Volume"].sum()),
            "traded_today": True,
        }

    # ------------------------------------------------------------------ #
    # News and company data
    # ------------------------------------------------------------------ #
    def get_recent_news(self, symbol: str, days_back: int = 3) -> list:
        """The 10 most recent company headlines, newest first."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        data = self._get("/company-news", {"symbol": symbol, "from": start_date, "to": end_date})

        # Finnhub is usually newest-first already, but not guaranteed, and leading with a stale
        # headline would be misleading. 'datetime' is a Unix timestamp in seconds.
        data = sorted(data, key=lambda article: article.get("datetime", 0), reverse=True)

        return [
            {
                "headline": article.get("headline"),
                "url": article.get("url"),
                "summary": article.get("summary"),
                "source": article.get("source"),
                "datetime": article.get("datetime"),
            }
            for article in data[:10]
        ]

    def get_key_metrics(self, symbol: str) -> dict:
        """
        A few fundamentals from Finnhub's free /stock/metric endpoint.

        It returns ~100 fields whose names vary a little between companies, so every lookup is
        a .get() and the formatter shows "—" for anything missing.
        """
        data = self._get("/stock/metric", {"symbol": symbol, "metric": "all"})
        metric = data.get("metric", {})
        return {
            "market_cap_m": metric.get("marketCapitalization"),  # Finnhub reports this in $ millions
            "beta": metric.get("beta"),
            "dividend_yield": metric.get(
                "currentDividendYieldTTM", metric.get("dividendYieldIndicatedAnnual")
            ),
        }

    def get_company_profile(self, symbol: str) -> dict:
        """Name and industry, so the report can say "NVIDIA Corp · Semiconductors" rather than just "NVDA"."""
        data = self._get("/stock/profile2", {"symbol": symbol})
        return {
            "name": data.get("name"),
            "industry": data.get("finnhubIndustry"),
            "market_cap_m": data.get("marketCapitalization"),
        }

    def get_yahoo_fundamentals(self, symbol: str) -> dict:
        """
        Forward earnings, short interest, ownership and balance-sheet figures from Yahoo.

        Forward EPS:
          'forward_eps' is consensus EPS for the next twelve months (see blend_ntm_eps); the
          analyzer divides the last close by it. Yahoo's own 'forwardPE' is only a fallback for
          when there are no estimates: it divides by the next *fiscal year's* EPS, which can be
          nearly two years out (NVDA in October -> the year ending January 2028), so it reads too
          low for growing companies. Its 'forwardEps' has also been seen disagreeing with Yahoo's
          own estimates table (RDDT).

        Short interest (exchange-reported twice a month, so up to ~2 weeks stale):
          short_pct_float   shares short as a % of the tradable float
          short_ratio       days to cover: shares short / average daily volume
          short_change_pct  change vs. the prior report

        'institutional_pct' is the % of shares held by institutions, from quarterly 13F filings.

        Debt and cash are from the latest quarterly balance sheet; operating cash flow is
        trailing twelve months. Yahoo's 'freeCashflow' is deliberately ignored: it's computed
        oddly (GEV's came out above its operating cash flow), so the analyzer's runway uses
        operating cash flow instead.

        Market cap and beta are included as fallbacks for when Finnhub has none.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        shares_short, prior = info.get("sharesShort"), info.get("sharesShortPriorMonth")
        short_pct = info.get("shortPercentOfFloat")
        institutional = info.get("heldPercentInstitutions")
        market_cap = info.get("marketCap")

        def millions(key):
            value = info.get(key)
            return value / 1e6 if value is not None else None

        forward_eps = None
        try:
            estimates = ticker.earnings_estimate
            avg = estimates["avg"] if estimates is not None and "avg" in estimates else {}
            fiscal_year_end = info.get("nextFiscalYearEnd")  # end of the year Yahoo labels "0y"
            forward_eps = self.blend_ntm_eps(
                avg.get("0y"),
                avg.get("+1y"),
                datetime.fromtimestamp(fiscal_year_end, MARKET_TZ).date() if fiscal_year_end else None,
                datetime.now(MARKET_TZ).date(),
            )
        except Exception as e:
            print(f"    [WARN] {symbol} analyst estimates unavailable, using Yahoo's forward P/E: {e}")

        return {
            "forward_eps": forward_eps,
            "forward_pe": None if forward_eps is not None else info.get("forwardPE"),
            "shares_short": shares_short,
            "short_pct_float": short_pct * 100 if short_pct is not None else None,
            "short_ratio": info.get("shortRatio"),
            "short_change_pct": (shares_short / prior - 1) * 100 if shares_short and prior else None,
            "institutional_pct": institutional * 100 if institutional is not None else None,
            "total_debt_m": millions("totalDebt"),
            "total_cash_m": millions("totalCash"),
            "operating_cashflow_m": millions("operatingCashflow"),
            "market_cap_m": market_cap / 1e6 if market_cap else None,
            "beta": info.get("beta"),
        }

    @staticmethod
    def blend_ntm_eps(eps_this_year, eps_next_year, fiscal_year_end, today):
        """
        Next-twelve-months EPS: whatever is left of this fiscal year plus the matching slice of
        next year. With 3 months left in the year, NTM = 25% of this year + 75% of next.

        Falls back to whichever single estimate exists, or None if neither does.
        """
        this_year = None if eps_this_year is None or pd.isna(eps_this_year) else float(eps_this_year)
        next_year = None if eps_next_year is None or pd.isna(eps_next_year) else float(eps_next_year)
        if this_year is None or next_year is None or fiscal_year_end is None:
            return next_year if next_year is not None else this_year

        weight = min(max((fiscal_year_end - today).days / 365, 0.0), 1.0)
        return weight * this_year + (1 - weight) * next_year

    def get_options_interest(self, symbol: str, days_ahead: int = 30) -> dict:
        """
        Total call and put open interest for expirations in the next `days_ahead` days.

        Only near-dated contracts are counted: that's where short-term positioning shows up,
        while longer-dated ones are dominated by large hedges. Yahoo refreshes open interest once
        a day, after the prior session settles.

        Calls expiring by this Friday are also tallied separately ('call_oi_week') so the report
        can say how much call positioning rolls off this week. Using "on or before Friday" also
        catches Thursday expirations in holiday weeks and Monday-Thursday dailies.
        """
        ticker = yf.Ticker(symbol)
        today = datetime.now(MARKET_TZ).date()
        cutoff = today + timedelta(days=days_ahead)
        # This Friday, or next Friday when run on a weekend.
        week_end = today + timedelta(days=(4 - today.weekday()) % 7)
        expirations = [d for d in ticker.options if datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        if not expirations:
            return {}

        call_oi = put_oi = call_oi_week = 0
        for expiry in expirations:
            chain = ticker.option_chain(expiry)
            calls = int(chain.calls["openInterest"].fillna(0).sum())
            call_oi += calls
            put_oi += int(chain.puts["openInterest"].fillna(0).sum())
            if datetime.strptime(expiry, "%Y-%m-%d").date() <= week_end:
                call_oi_week += calls

        return {
            "call_oi": call_oi,
            "put_oi": put_oi,
            "put_call_oi_ratio": put_oi / call_oi if call_oi else None,
            "options_window_days": days_ahead,
            "call_oi_week": call_oi_week,
            "call_oi_week_pct": call_oi_week / call_oi * 100 if call_oi else None,
            "week_end": week_end.isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Market-wide data
    # ------------------------------------------------------------------ #
    def get_upcoming_earnings(self, symbols: list, days_ahead: int = 14) -> dict:
        """
        Earnings dates for `symbols` within the next `days_ahead` days.

        Pulls the whole market calendar in one call and filters locally, rather than one call
        per ticker, so the 60 calls/minute limit doesn't tighten as the watchlist grows.

        Returns {symbol: {"date": "2026-10-08", "hour": "amc", "eps_estimate": 1.23}}, where
        'hour' is Finnhub's code: "bmo" = before market open, "amc" = after market close.
        """
        today = datetime.now(MARKET_TZ).date()
        data = self._get(
            "/calendar/earnings",
            {
                "from": today.strftime("%Y-%m-%d"),
                "to": (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
            },
        )

        wanted = {s.upper() for s in symbols}
        upcoming = {}
        for row in data.get("earningsCalendar", []) or []:
            symbol = (row.get("symbol") or "").upper()
            if symbol in wanted and symbol not in upcoming:
                upcoming[symbol] = {
                    "date": row.get("date"),
                    "hour": row.get("hour"),
                    "eps_estimate": row.get("epsEstimate"),
                }
        return upcoming

    def get_market_movers(self, count: int = 5) -> dict:
        """
        Biggest US gainers, losers and most-traded stocks from Yahoo's predefined screens.

        These surface big moves outside the watchlist. The screens reflect the last *regular*
        session, so on an 8 AM run they describe yesterday, and the report labels them that way.
        """
        movers = {}
        for key in ("day_gainers", "day_losers", "most_actives"):
            try:
                result = yf.screen(key, count=count)
                quotes = result.get("quotes", []) if isinstance(result, dict) else []
            except Exception as e:
                print(f"[WARN] Market movers screen '{key}' failed: {e}")
                quotes = []

            movers[key] = [
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("shortName") or q.get("longName"),
                    "price": q.get("regularMarketPrice"),
                    "change_pct": q.get("regularMarketChangePercent"),
                    "volume": q.get("regularMarketVolume"),
                    "avg_volume": q.get("averageDailyVolume3Month"),
                }
                for q in quotes[:count]
            ]
        return movers
