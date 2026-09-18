# src/fetcher.py
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

class FinnhubFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
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
        """Fetches real-time/latest end-of-day quote data."""
        
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

    def get_historical_candles(self, symbol: str, days_back: int = 60) -> pd.DataFrame:
        """Fetches daily OHLCV data for technical analysis (MAs, RSI, Volume)."""

        end_time = int(time.time())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        data = self._get("/stock/candle", {
            "symbol": symbol,
            "resolution": "D",
            "from": start_time,
            "to": end_time
        })
        
        if data.get("s") != "ok":
            return pd.DataFrame()
            
        df = pd.DataFrame({
            "date": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })

        # Set datetime index for easy integration with the 'ta' library later
        return df.set_index("date")

    def get_recent_news(self, symbol: str, days_back: int = 3) -> list:
        """Fetches recent company news catalysts."""

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        data = self._get("/company-news", {
            "symbol": symbol,
            "from": start_date,
            "to": end_date
        })
        
        # Return only the top 3 most recent articles to keep the Discord payload clean
        return [{"headline": article.get("headline"), "url": article.get("url")} for article in data[:3]]