def load_config(filepath: str = "config.yaml") -> dict:
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    watchlist = config.get("watchlist", [])
    
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        sys.exit(1)