def load_config(filepath: str = "config.yaml") -> dict:
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    watchlist = config.get("watchlist", [])
    
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        sys.exit(1)

    fetcher = FinnhubFetcher(api_key=api_key)
    pipeline_data = {}

    for symbol in watchlist:
        quote = fetcher.get_quote(symbol)
        candles = fetcher.get_historical_candles(symbol, days_back=90)
        news = fetcher.get_recent_news(symbol, days_back=3)

        pipeline_data[symbol] = {
            "quote": quote,
            "candles": candles,
            "news": news
        }
    for symbol, data in pipeline_data.items():
            price = data["quote"].get("current_price")
            # ... formatting and printing ...