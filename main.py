"""
Pipeline entry point
--------------------
Runs the whole morning job in order:

  1. Fetch    (src/fetcher.py)    prices, pre-market trades, news, fundamentals, short interest, options, earnings, market
  2. Analyze  (src/analyzer.py)   indicators -> signals -> ranked alerts
  3. Format   (src/formatter.py, src/pdf_report.py)   email HTML + text + PDF newsletter
  4. Dispatch (src/dispatcher.py) Gmail SMTP to the recipient list

Usage:
  python main.py              fetch, analyze, build the report, and email it
  python main.py --dry-run    everything except sending; open the files in output/ to preview
  python main.py --force      send even if the market looks closed today (weekend/holiday)
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

from src.analyzer import QuantitativeAnalyzer
from src.dispatcher import EmailDispatcher, parse_recipients
from src.fetcher import MARKET_TZ, FinnhubFetcher
from src.formatter import ReportFormatter, fmt_pct

OUTPUT_DIR = "output"
MARKET_SYMBOLS = ["SPY", "QQQ"]   # benchmarks that get a pre-market gap
VIX_SYMBOL = "^VIX"               # volatility index (daily close only)


def load_config(filepath: str = "config.yaml") -> dict:
    """Reads configuration safely. Fails fast if the file is absent or malformed."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found at '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-market stock analysis newsletter")
    parser.add_argument("--dry-run", action="store_true", help="build the report but don't send it")
    parser.add_argument("--force", action="store_true", help="send even if the market looks closed")
    return parser.parse_args()


def safe_call(label: str, func, default):
    """
    Runs one API call and returns `default` if it fails.

    Why: the report has ~5 data sources per stock. If Finnhub's fundamentals endpoint hiccups
    for one ticker, that ticker's page should show "—" for market cap, not crash the whole run
    and leave everyone without a morning email.
    """
    try:
        return func()
    except Exception as e:
        print(f"    [WARN] {label} failed: {e}")
        return default


def fetch_symbol(fetcher: FinnhubFetcher, symbol: str, use_finnhub: bool, days_back: int,
                 options_days: int = 30) -> dict:
    data = {
        # yfinance (no key needed)
        "candles": safe_call(f"{symbol} price history", lambda: fetcher.get_historical_candles(symbol, days_back),
                             pd.DataFrame()),
        "premarket": safe_call(f"{symbol} pre-market", lambda: fetcher.get_premarket_snapshot(symbol), {}),
        "quote": {}, "news": [],
    }
    yahoo = safe_call(f"{symbol} Yahoo fundamentals", lambda: fetcher.get_yahoo_fundamentals(symbol), {})
    options = safe_call(f"{symbol} options", lambda: fetcher.get_options_interest(symbol, options_days), {})
    profile, metrics = {}, {}
    if use_finnhub:
        data["quote"] = safe_call(f"{symbol} quote", lambda: fetcher.get_quote(symbol), {})
        data["news"] = safe_call(f"{symbol} news", lambda: fetcher.get_recent_news(symbol, days_back=3), [])
        profile = safe_call(f"{symbol} profile", lambda: fetcher.get_company_profile(symbol), {})
        metrics = safe_call(f"{symbol} fundamentals", lambda: fetcher.get_key_metrics(symbol), {})
    # Later sources overwrite earlier ones, but only with real values. So Finnhub's numbers win
    # and Yahoo fills the gaps (e.g. on a dry run with no key). Forward P/E, short interest and
    # options only come from Yahoo.
    fundamentals = {}
    for source in (yahoo, options, profile, metrics):
        fundamentals.update({k: v for k, v in source.items() if v is not None})
    data["fundamentals"] = fundamentals
    return data


def main():
    args = parse_args()

    # 1. Load parameters
    config = load_config()
    watchlist = [s.upper() for s in config.get("watchlist", [])]
    thresholds = config.get("analysis_thresholds", {})
    report_cfg = config.get("report", {})
    movers_cfg = config.get("market_movers", {})

    if not watchlist:
        print("[ERROR] Watchlist is empty. Define tickers inside config.yaml.")
        sys.exit(1)

    # 2. Credentials from environment variables (never hardcoded, never committed)
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] Missing required environment variable 'FINNHUB_API_KEY'.")
        print("Set it locally with: export FINNHUB_API_KEY='your_key'")
        print("Or add it under GitHub Repository Settings -> Secrets and variables -> Actions.")
        sys.exit(1)
    if not api_key:
        print("[WARN] No FINNHUB_API_KEY: dry run continues with Yahoo data only (no news or earnings dates).")

    fetcher = FinnhubFetcher(api_key=api_key)
    analyzer = QuantitativeAnalyzer(thresholds)
    now = datetime.now(MARKET_TZ)

    # 3. Fetch every ticker. ~500 calendar days = ~340 sessions: enough for a 200-day average
    #    across the whole 6-month chart AND a full 52-week range.
    print(f"[*] Beginning ingestion cycle for {len(watchlist)} tickers...\n")
    pipeline_data = {}
    for symbol in watchlist:
        print(f"--> Fetching market data for: {symbol}")
        pipeline_data[symbol] = fetch_symbol(fetcher, symbol, bool(api_key), days_back=500,
                                             options_days=thresholds.get("options_window_days", 30))

    print("--> Fetching market backdrop (SPY, QQQ, VIX)")
    market_raw = {}
    for symbol in MARKET_SYMBOLS:
        market_raw[symbol] = {
            "candles": safe_call(f"{symbol} history", lambda s=symbol: fetcher.get_historical_candles(s, 60), pd.DataFrame()),
            "premarket": safe_call(f"{symbol} pre-market", lambda s=symbol: fetcher.get_premarket_snapshot(s), {}),
        }
    market_raw[VIX_SYMBOL] = {
        "candles": safe_call("VIX history", lambda: fetcher.get_historical_candles(VIX_SYMBOL, 30), pd.DataFrame()),
        "premarket": {},
    }

    earnings = {}
    if api_key:
        print("--> Checking the earnings calendar")
        earnings = safe_call("earnings calendar", lambda: fetcher.get_upcoming_earnings(
            watchlist, days_ahead=thresholds.get("earnings_lookahead_days", 7) + 7), {})

    movers = {}
    if movers_cfg.get("enabled", True):
        print("--> Scanning market movers")
        movers = safe_call("market movers", lambda: fetcher.get_market_movers(movers_cfg.get("count", 5)), {})

    # 4. Is the market open today? If nothing traded pre-market, it's a weekend or holiday.
    traded = [d["premarket"].get("traded_today") for d in list(pipeline_data.values()) + list(market_raw.values())]
    if not any(traded):
        print(f"\n[i] No trades found for {now:%A %b %d}: market closed (weekend/holiday) or before 4 AM ET.")
        if not (args.force or args.dry_run):
            print("    Skipping today's email. Use --force to send anyway.")
            return

    # 5. Analyze
    print("\n[*] Running quantitative analysis...")
    market = analyzer.analyze_market(market_raw)
    results = analyzer.analyze(pipeline_data, market, earnings)
    ranked = analyzer.rank_stocks(results, report_cfg.get("header_min_severity", 1.0))

    print(f"  └─ {len(ranked)} of {len(results)} stocks need attention:")
    for item in ranked:
        print(f"     [{item['symbol']}] {fmt_pct(item.get('gap_pct'))} vs. last close · score {item['score']}")
        for alert in item["alerts"]:
            print(f"        - {alert['text']}")
    for symbol, result in results.items():
        if "error" in result:
            print(f"     [{symbol}] skipped: {result['error']}")

    # 6. Format: email + PDF, saved to output/ so every run leaves a preview behind
    formatter = ReportFormatter(report_cfg, thresholds)
    ctx = formatter.build_context(results, ranked, market, movers, now)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, f"premarket_{now:%Y-%m-%d}")

    pdf_path = None
    if report_cfg.get("attach_pdf", True):
        try:
            from src.pdf_report import NewsletterPDF
            pdf_path = NewsletterPDF(formatter).build(ctx, pipeline_data, f"{stem}.pdf")
            print(f"\n[✓] PDF newsletter: {pdf_path}")
        except Exception as e:
            # A broken PDF should never block the email itself
            print(f"\n[WARN] PDF build failed, sending without it: {e}")
            pdf_path = None

    subject = formatter.subject(ctx)
    html_body = formatter.render_html(ctx, pdf_attached=pdf_path is not None)
    text_body = formatter.render_text(ctx, pdf_attached=pdf_path is not None)
    with open(f"{stem}.html", "w", encoding="utf-8") as f:
        f.write(html_body)
    with open(f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write(text_body)
    print(f"[✓] Email preview:  {stem}.html")
    print(f"    Subject: {subject}")

    if args.dry_run:
        print("\n[i] Dry run: nothing was sent. Open the files above to preview the report.")
        return

    # 7. Dispatch
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("EMAIL_PASSWORD")
    recipients = parse_recipients(os.environ.get("RECIPIENT_EMAILS", ""))
    if not all([sender, password, recipients]):
        print("\n[ERROR] Email settings missing. Set SENDER_EMAIL, EMAIL_PASSWORD and RECIPIENT_EMAILS,")
        print("        or run with --dry-run to build the report without sending it.")
        sys.exit(1)

    dispatcher = EmailDispatcher(sender, password, recipients, sender_name=report_cfg.get("title", "Pre-Market Brief"))
    sent = dispatcher.send(subject, html_body, text_body, attachments=[pdf_path] if pdf_path else [])
    if not sent:
        sys.exit(1)  # makes the GitHub Actions run fail, which emails you about it

    print("\n[✓] Pipeline execution finished.")


if __name__ == "__main__":
    main()
