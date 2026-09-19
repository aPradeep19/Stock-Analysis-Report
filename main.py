import os
import sys
import yaml
from src.fetcher import FinnhubFetcher
from src.analyzer import QuantitativeAnalyzer


def load_config(filepath: str = "config.yaml") -> dict:
    """Reads configuration safely. Fails fast if the file is absent or malformed."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found at '{filepath}'")
    
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    # Load parameters
    config = load_config()
    watchlist = config.get("watchlist", [])
    thresholds = config.get("analysis_thresholds", {})

    # Extract credentials from environment 
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        print("[ERROR] Missing required environment variable 'FINNHUB_API_KEY'.")
        print("Set it locally with: export FINNHUB_API_KEY='your_key'")
        print("Or add it under GitHub Repository Settings -> Secrets and variables -> Actions.")
        sys.exit(1)

    # Initialize the ingestion client
    fetcher = FinnhubFetcher(api_key=api_key)
    analyzer= QuantitativeAnalyzer(thresholds)
    pipeline_data = {}

    print(f"[*] Beginning ingestion cycle for {len(watchlist)} tickers...\n")

    # Sequentially fetch data for each asset
    for symbol in watchlist:
        print(f"--> Fetching market data for: {symbol}")
        
        quote = fetcher.get_quote(symbol)
        candles = fetcher.get_historical_candles(symbol, days_back=300)
        news = fetcher.get_recent_news(symbol, days_back=3)

        pipeline_data[symbol] = {
            "quote": quote,
            "candles": candles,
            "news": news
        }
    
    analysis = analyzer.analyze(pipeline_data)



    # Output pipeline data validation summary
    print("\n======================= INGESTION VERIFICATION =======================")
    for symbol, data in pipeline_data.items():
        price = data["quote"].get("current_price")
        pct_change = data["quote"].get("change_pct")
        candle_count = len(data["candles"])
        news_count = len(data["news"])

        # Format price and change cleanly
        price_str = f"${price:.2f}" if price is not None else "N/A"
        pct_str = f"{pct_change:+.2f}%" if pct_change is not None else "N/A"

        print(f"\nTicker: {symbol:<6} | Last: {price_str:<10} | Change: {pct_str}")
        print(f"  └─ Daily Bars Retained: {candle_count} trading sessions")
        
        if news_count > 0:
            print(f"  └─ Latest Catalyst: \"{data['news'][0]['headline'][:80]}...\"")
        else:
            print("  └─ Latest Catalyst: None reported in the last 72 hours")

    print("\n======================= Analysis =======================")
    for symbol, data in analysis.items():
        print(f"\n     [{symbol}] - Current Price: ${data['price']:.2f}")
        for trigger in data['triggers']:
            print(f"      - {trigger}")
        print(f"      - Sentiment: {data['sentiment']}")


if __name__ == "__main__":
    main()