import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

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

            # Use a date range 
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
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
            # Slice off timezone info
            df.index = df.index.tz_localize(None)
            
            return df.sort_index()
            
        except Exception as e:
            print(f"[WARN] Failed to fetch historical data for {symbol} via yfinance: {e}")
            return pd.DataFrame()

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
        
        # Return the 3 most recent articles
        return [{"headline": article.get("headline"), "url": article.get("url")} for article in data[:3]]