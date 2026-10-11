import time
import requests
import pandas as pd
from datetime import datetime, timedelta, time as clock_time
from zoneinfo import ZoneInfo
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
        
        response = requests.get(
            f"{self.base_url}{endpoint}",
            params=params,
            timeout=10
        )
        
        if response.status_code == 429:
            print("\n[WARN] Rate limit reached. Sleeping for 60 seconds...")
            time.sleep(60)
            return self._get(endpoint, params)
            
        response.raise_for_status()
        return response.json()

    def get_quote(self, symbol: str) -> dict:
        """
        Retrieves real-time/latest end-of-day price action.
        
        Finnhub keys:
          'c': Current close/last price
          'd': Dollar change
          'dp': Percentage change
          'h': Daily High, 'l': Daily Low, 'o': Daily Open, 'pc': Previous Close
        """
        
        data = self._get("/quote", {"symbol": symbol})
        return {
            "current_price": data.get("c"),
            "change": data.get("d"),
            "change_pct": data.get("dp"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "prev_close": data.get("pc")
        }

    def get_historical_candles(self, symbol: str, days_back: int = 90) -> pd.DataFrame:
        """
        Fetches daily OHLCV bars using yfinance to bypass Finnhub's paywall.
        Returns a clean Datetime-indexed pandas DataFrame.
        """

        try:
            ticker = yf.Ticker(symbol)

            # Use a date range in New York time. yfinance's `end` is exclusive, so tomorrow's
            # date is needed to include today's bar (dropped below if today isn't finished).
            now = datetime.now(MARKET_TZ)
            end_date = now.date() + timedelta(days=1)
            start_date = now.date() - timedelta(days=days_back)

            df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            
            if df.empty:
                return pd.DataFrame()
                
            # yfinance capitalizes column names. Lowercase them to match our analyzer's expectations.
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            df = df[["open", "high", "low", "close", "volume"]]

            # Yahoo sometimes returns a row with no prices, e.g. a dividend dated on a day the
            # stock hasn't traded yet. A bar without a close can't be analyzed, so drop it, and
            # fill any missing high/low/open from the close so comparisons never hit a blank.
            df = df.dropna(subset=["close"])
            for column in ("open", "high", "low"):
                df[column] = df[column].fillna(df["close"])
            df["volume"] = df["volume"].fillna(0)
            # Slice off timezone info
            df.index = df.index.tz_localize(None)

            return self.drop_unfinished_bar(df.sort_index(), now)
            
        except Exception as e:
            print(f"[WARN] Failed to fetch historical data for {symbol} via yfinance: {e}")
            return pd.DataFrame()

    @staticmethod
    def drop_unfinished_bar(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """
        Today's daily bar keeps changing until the 4 PM close. Dropping it before then means
        "last close" always refers to a finished session:
          - 8:00 AM run  -> last close = yesterday (the pre-market gap is measured from it)
          - 2:00 PM run  -> last close = yesterday (the move "so far today" is measured from it)
          - 6:00 PM run  -> last close = today's 4 PM close (the after-hours move is measured from it)
        """
        if not df.empty and df.index[-1].date() == now.date() and now.time() < REGULAR_CLOSE:
            return df.iloc[:-1]
        return df

    def get_recent_news(self, symbol: str, days_back: int = 3) -> list:
        """
        Pulls recent company-specific headlines and links to serve as catalysts.
        """

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        data = self._get("/company-news", {
            "symbol": symbol,
            "from": start_date,
            "to": end_date
        })
        
        # Finnhub usually returns newest first, but we sort explicitly so the report never
        # leads with a stale headline. 'datetime' is a Unix timestamp (seconds).
        data = sorted(data, key=lambda article: article.get("datetime", 0), reverse=True)

        # Return the 10 most recent articles
        return [{
            "headline": article.get("headline"),
            "url": article.get("url"),
            "summary": article.get("summary"),
            "source": article.get("source"),
            "datetime": article.get("datetime"),
        } for article in data[:10]]

    def get_premarket_snapshot(self, symbol: str) -> dict:
        """
        Returns the latest price traded *today*, including pre-market trades.

        Why not Finnhub's /quote? Its free tier only tracks the regular 9:30-4:00 session,
        so at 8 AM its 'c' value is still yesterday's close. Yahoo's 1-minute bars with
        prepost=True include extended-hours trades (pre-market runs 4:00-9:30 AM ET).

        Returned keys:
          'price':            last traded price today (None if nothing traded yet today)
          'as_of':            timestamp of that trade, in New York time
          'session':          "pre-market", "regular", or "after-hours"
          'premarket_volume': shares traded today before 9:30 AM
          'traded_today':     False on weekends/holidays -> main.py uses this to skip the email
        """
        empty = {"price": None, "as_of": None, "session": None,
                 "premarket_volume": None, "traded_today": False}

        try:
            # period="5d" instead of "1d": right around 4 AM, Yahoo's "1d" window can still
            # point at yesterday. Pulling a few days and filtering to today's date is safer.
            bars = yf.Ticker(symbol).history(period="5d", interval="1m", prepost=True)
        except Exception as e:
            print(f"[WARN] Failed to fetch pre-market data for {symbol} via yfinance: {e}")
            return empty

        if bars.empty:
            return empty

        # Convert every bar's timestamp to New York time so "today" and "before 9:30" mean
        # the same thing whether this runs on your laptop or on GitHub's servers.
        index = bars.index
        index = index.tz_localize(MARKET_TZ) if index.tz is None else index.tz_convert(MARKET_TZ)
        bars = bars.set_axis(index)

        today = datetime.now(MARKET_TZ).date()
        todays_bars = bars[bars.index.date == today]
        if todays_bars.empty:
            return empty  # Weekend, market holiday, or it's before 4 AM

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

    def get_key_metrics(self, symbol: str) -> dict:
        """
        Pulls company fundamentals for each stock's section of the report (and its PDF page later).

        Finnhub's /stock/metric endpoint is on the free tier. It returns ~100 fields; we keep the
        handful a reader actually scans. Field names vary slightly between companies, so every
        lookup uses .get() and the formatter shows "—" for anything missing.
        """
        data = self._get("/stock/metric", {"symbol": symbol, "metric": "all"})
        metric = data.get("metric", {}) 
        return {
            # Finnhub reports market cap in millions of dollars
            "market_cap_m": metric.get("marketCapitalization"),
            "beta": metric.get("beta"),
            "dividend_yield": metric.get("currentDividendYieldTTM", metric.get("dividendYieldIndicatedAnnual")),
        }

    def get_yahoo_fundamentals(self, symbol: str) -> dict:
        """
        Forward earnings and short interest from Yahoo (no key needed; Finnhub's free tier has neither).

        'forward_eps' is the analysts' consensus EPS for the NEXT 12 MONTHS (see blend_ntm_eps);
        the analyzer divides the last close by it to get the forward P/E. Yahoo's own 'forwardPE'
        isn't used when we can avoid it: it divides by next *fiscal year's* EPS, which can be up to
        two years out (NVDA in October -> the year ending January 2028), so for a growing company it
        reads too low. Its 'forwardEps' is also sometimes stale (it disagreed with Yahoo's own
        estimates table for RDDT). It's kept only as a fallback when there are no estimates.

        Short interest is reported by exchanges twice a month, so it lags by up to ~2 weeks.
          'short_pct_float':  shares sold short as a % of the tradable float
          'short_ratio':      "days to cover": shares short / average daily volume
          'short_change_pct': change vs. the prior report, so a growing short position stands out
        Market cap and beta come along too, as fallbacks when Finnhub has none.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        shares_short, prior = info.get("sharesShort"), info.get("sharesShortPriorMonth")
        short_pct = info.get("shortPercentOfFloat")
        market_cap = info.get("marketCap")

        forward_eps = None
        try:
            estimates = ticker.earnings_estimate
            avg = estimates["avg"] if estimates is not None and "avg" in estimates else {}
            fiscal_year_end = info.get("nextFiscalYearEnd")  # end of the year Yahoo calls "0y"
            forward_eps = self.blend_ntm_eps(
                avg.get("0y"), avg.get("+1y"),
                datetime.fromtimestamp(fiscal_year_end, MARKET_TZ).date() if fiscal_year_end else None,
                datetime.now(MARKET_TZ).date())
        except Exception as e:
            print(f"    [WARN] {symbol} analyst estimates unavailable, using Yahoo's forward P/E: {e}")

        return {
            "forward_eps": forward_eps,
            "forward_pe": None if forward_eps is not None else info.get("forwardPE"),
            "shares_short": shares_short,
            "short_pct_float": short_pct * 100 if short_pct is not None else None,
            "short_ratio": info.get("shortRatio"),
            "short_change_pct": (shares_short / prior - 1) * 100 if shares_short and prior else None,
            "market_cap_m": market_cap / 1e6 if market_cap else None,
            "beta": info.get("beta"),
        }

    @staticmethod
    def blend_ntm_eps(eps_this_year, eps_next_year, fiscal_year_end, today):
        """
        Next-twelve-months EPS: the part of this fiscal year still ahead plus the matching slice
        of next year. With 3 months left in the year, NTM = 25% of this year + 75% of next year.

        Returns None when there's nothing usable, so the caller can fall back.
        """
        this_year = None if eps_this_year is None or pd.isna(eps_this_year) else float(eps_this_year)
        next_year = None if eps_next_year is None or pd.isna(eps_next_year) else float(eps_next_year)
        if this_year is None or next_year is None or fiscal_year_end is None:
            return next_year if next_year is not None else this_year
        weight = min(max((fiscal_year_end - today).days / 365, 0.0), 1.0)
        return weight * this_year + (1 - weight) * next_year

    def get_options_interest(self, symbol: str, days_ahead: int = 30) -> dict:
        """
        Total call and put open interest across option expirations in the next `days_ahead` days.

        Open interest = contracts still open (not yet closed or exercised). Near-dated contracts
        are where traders position for short-term moves, and longer-dated ones would mostly add
        noise from big hedges, so only the next month is counted. Yahoo updates open interest
        once a day, after the previous session settles.
        """
        ticker = yf.Ticker(symbol)
        cutoff = datetime.now(MARKET_TZ).date() + timedelta(days=days_ahead)
        expirations = [d for d in ticker.options if datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]
        if not expirations:
            return {}

        call_oi = put_oi = 0
        for expiry in expirations:
            chain = ticker.option_chain(expiry)
            call_oi += int(chain.calls["openInterest"].fillna(0).sum())
            put_oi += int(chain.puts["openInterest"].fillna(0).sum())
        return {
            "call_oi": call_oi,
            "put_oi": put_oi,
            "put_call_oi_ratio": put_oi / call_oi if call_oi else None,
            "options_window_days": days_ahead,
        }

    def get_company_profile(self, symbol: str) -> dict:
        """Company name and industry, so the report says "NVIDIA Corp · Semiconductors" instead of just "NVDA"."""
        data = self._get("/stock/profile2", {"symbol": symbol})
        return {
            "name": data.get("name"),
            "industry": data.get("finnhubIndustry"),
            "market_cap_m": data.get("marketCapitalization"),
        }

    def get_upcoming_earnings(self, symbols: list, days_ahead: int = 14) -> dict:
        """
        Finds earnings dates for the watchlist in the next `days_ahead` days.

        Design choice: ONE call for the whole market's calendar, then we filter to our tickers,
        instead of one call per ticker. That keeps us far under the 60 calls/minute limit as
        the watchlist grows.

        Returns {symbol: {"date": "2026-10-08", "hour": "amc", "eps_estimate": 1.23}}
          'hour' is Finnhub's code: "bmo" = before market open, "amc" = after market close.
        """
        today = datetime.now(MARKET_TZ).date()
        data = self._get("/calendar/earnings", {
            "from": today.strftime("%Y-%m-%d"),
            "to": (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
        })

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
        The biggest US gainers, losers and most-traded stocks from Yahoo's built-in screeners.

        Why: your watchlist only shows what you already follow. This section surfaces big moves
        elsewhere in the market. Yahoo's screeners use the last *regular* session, so at 8 AM
        they describe yesterday, which is how the report labels them.
        """
        movers = {}
        for key in ("day_gainers", "day_losers", "most_actives"):
            try:
                result = yf.screen(key, count=count)
                quotes = result.get("quotes", []) if isinstance(result, dict) else []
            except Exception as e:
                print(f"[WARN] Market movers screen '{key}' failed: {e}")
                quotes = []

            movers[key] = [{
                "symbol": q.get("symbol"),
                "name": q.get("shortName") or q.get("longName"),
                "price": q.get("regularMarketPrice"),
                "change_pct": q.get("regularMarketChangePercent"),
                "volume": q.get("regularMarketVolume"),
                "avg_volume": q.get("averageDailyVolume3Month"),
            } for q in quotes[:count]]
        return movers