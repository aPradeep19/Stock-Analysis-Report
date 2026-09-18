# src/fetcher.py
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

class FinnhubFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        # Finnhub expects token authentication passed via the custom X-Finnhub-Token header
        self.headers = {"X-Finnhub-Token": self.api_key}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        if params is None:
            params = {}
        
        response = requests.get(f"{self.base_url}{endpoint}", headers=self.headers, params=params)
        
        # Finnhub free tier limit is 60 calls/minute. Handle gracefully if hit.
        if response.status_code == 429:
            print("Rate limit reached. Sleeping for 60 seconds...")
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
        Fetches daily OHLCV bars converted into a clean Datetime-indexed pandas DataFrame.
        
        Lookback Window Logic:
        We set the default to 90 calendar days (~62 trading sessions). A standard 50-day 
        Simple Moving Average requires a minimum of 50 valid data points. Setting days_back
        to 90 ensures the rolling window calculation will not return NaN values.
        """

        end_time = int(time.time())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        data = self._get("/stock/candle", {
            "symbol": symbol,
            "resolution": "D",
            "from": start_time,
            "to": end_time
        })
        
        # Check API status flag; 'no_data' or empty returns result in an empty DataFrame
        if data.get("s") != "ok":
            return pd.DataFrame()
        
        # Construct DataFrame and normalize Finnhub's abbreviated single-letter keys
        df = pd.DataFrame({
            "date": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })

        # Set datetime index for easy integration with the 'ta' library later
        return df.set_index("date").sort_index()

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
        
        # Return only the top 3 most recent articles to keep the Discord payload clean
        return [{"headline": article.get("headline"), "url": article.get("url")} for article in data[:3]]