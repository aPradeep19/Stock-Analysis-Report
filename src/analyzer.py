import pandas as pd
from ta.momentum import RSIIndicator

class QuantitativeAnalyzer:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        
        # Load risk boundaries from config
        self.price_pct_limit = self.thresholds.get("price_change_pct", 5.0)
        self.vol_multiplier = self.thresholds.get("volume_surge_multiplier", 2.0)
        self.rsi_oversold = self.thresholds.get("rsi_oversold", 30)
        self.rsi_overbought = self.thresholds.get("rsi_overbought", 70)

    def _compute_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard quantitative indicators to the historical DataFrame.
        Requires at least 20 days of data for the SMA and 14 days for the RSI.
        """
        if df.empty or len(df) < 20:
            return df

        # 1. 20-Day Simple Moving Average of Volume
        df['vol_sma_20'] = df['volume'].rolling(window=20).mean()

        # 2. 14-Period Relative Strength Index (RSI)
        rsi_indicator = RSIIndicator(close=df['close'], window=14)
        df['rsi_14'] = rsi_indicator.rsi()

        return df

    def _analyze_sentiment(self, news: list) -> dict:
        """
        Evaluates the news data fetched from Finnhub. It scans the text of the 
        headlines and summaries for a hardcoded list of bullish words 
        (like "surge" or "upgrade") and bearish words (like "drop" or "downgrade").
        It adds or subtracts a point for each match to generate a net score, ultimately 
        returning a simple label: "Bullish", "Bearish", or "Neutral".
        """
        if not news:
            return {"score": 0, "label": "Neutral"}
            
        # Basic heuristic for demonstration: count bullish/bearish keywords
        bullish_words = ["surge", "jump", "beat", "upgrade", "record"]
        bearish_words = ["miss", "drop", "plunge", "downgrade", "lawsuit"]
        
        score = 0
        for article in news:
            text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
            if any(word in text for word in bullish_words):
                score += 1
            if any(word in text for word in bearish_words):
                score -= 1
                
        if score > 0:
            return {"score": score, "label": "Bullish"}
        elif score < 0:
            return {"score": score, "label": "Bearish"}
        return {"score": score, "label": "Neutral"}

    def analyze(self, pipeline_data: dict) -> dict:
        """
        Evaluates the entire watchlist against the configuration thresholds.
        Returns a dictionary containing all analysis. Trigger logic commented out
        for future updates.
        """
        flagged_tickers = {}

        for symbol, data in pipeline_data.items():
            quote = data.get("quote", {})
            raw_candles = data.get("candles", pd.DataFrame())
            news = data.get("news", [])

            # Skip if API failed to return usable data
            if raw_candles.empty or quote.get("change_pct") is None:
                continue

            # Compute technicals
            df = self._compute_technicals(raw_candles)
            latest_bar = df.iloc[-1] # The most recent closed trading session

            # Initialize tracking flags
            triggers = []

            # Daily Price Volatility
            daily_pct = quote.get("change_pct", 0)
            # if abs(daily_pct) >= self.price_pct_limit:
            if True:
                direction = "Up" if daily_pct > 0 else "Down"
                triggers.append(f"Price moved {daily_pct:+.2f}% ({direction})")

            # Volume Anomaly
            current_vol = latest_bar['volume']
            avg_vol = latest_bar['vol_sma_20']
            # if pd.notna(avg_vol) and avg_vol > 0:
            if True:
                surge_ratio = current_vol / avg_vol
                # if surge_ratio >= self.vol_multiplier:
                if True:
                    triggers.append(f"Volume surge: {surge_ratio:.1f}x the 20-day average")

            # RSI Extremes
            current_rsi = latest_bar['rsi_14']
            # if pd.notna(current_rsi):
            if True:
                if current_rsi <= self.rsi_oversold:
                    triggers.append(f"Oversold (RSI: {current_rsi:.1f})")
                elif current_rsi >= self.rsi_overbought:
                    triggers.append(f"Overbought (RSI: {current_rsi:.1f})")

            # Only flag the ticker if it breached at least one threshold
            if triggers:
                sentiment = self._analyze_sentiment(news)
                flagged_tickers[symbol] = {
                    "price": quote.get("current_price"),
                    "change_pct": daily_pct,
                    "triggers": triggers,
                    "sentiment": sentiment["label"],
                    "top_news": news[:2] # Pass only the top 2 articles to the formatter
                }

        return flagged_tickers