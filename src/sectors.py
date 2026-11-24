"""
Automatic sector grouping: which sector fund each watchlist stock trades with.

Two signals, so new tickers need no manual setup:
  1. Price history. Each stock's daily returns are compared with each candidate fund's, after
     removing the overall market's move from both (otherwise every tech stock just matches the
     biggest tech fund). The best match wins if it's clear: a strong enough score and a clear
     lead over the runner-up.
  2. Yahoo's sector/industry label, when the price match is weak, tied, or there isn't enough
     history (new listings).
A manual override in config.yaml beats both.
"""

import pandas as pd

MARKET_PROXY = "SPY"

FUND_NAMES = {
    "SMH": "Semiconductor ETF",
    "IGV": "Software ETF",
    "XBI": "Biotech ETF",
    # Theme fund for AI-cloud providers (NBIS, CoreWeave, IREN...). Listed Aug 2026, so it only
    # wins price matches once a stock shares `min_history_days` of trading with it.
    "NCLD": "Neocloud ETF (Roundhill)",
    "XLK": "Tech sector ETF",
    "XLC": "Communication services ETF",
    "XLV": "Health care ETF",
    "XLF": "Financials ETF",
    "XLE": "Energy ETF",
    "XLI": "Industrials ETF",
    "XLY": "Consumer discretionary ETF",
    "XLP": "Consumer staples ETF",
    "XLU": "Utilities ETF",
    "XLB": "Materials ETF",
    "XLRE": "Real estate ETF",
}
CANDIDATE_FUNDS = list(FUND_NAMES)

# Yahoo industry names that deserve a narrower fund than their sector's.
INDUSTRY_FUNDS = {
    "Semiconductors": "SMH",
    "Semiconductor Equipment & Materials": "SMH",
    "Biotechnology": "XBI",
}
SECTOR_FUNDS = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
}


def fund_from_label(sector, industry):
    """Sector fund for Yahoo's labels, or None when there's no label."""
    if industry in INDUSTRY_FUNDS:
        return INDUSTRY_FUNDS[industry]
    if industry and industry.startswith("Software"):  # "Software - Application", "- Infrastructure"
        return "IGV"
    return SECTOR_FUNDS.get(sector)


def _market_removed(returns, market):
    """Daily returns with the market's share (beta x market return) taken out."""
    both = pd.concat([returns, market], axis=1, join="inner").dropna()
    if len(both) < 2 or both.iloc[:, 1].var() == 0:
        return pd.Series(dtype=float)
    beta = both.cov().iloc[0, 1] / both.iloc[:, 1].var()
    return both.iloc[:, 0] - beta * both.iloc[:, 1]


def match_scores(stock_closes, fund_closes, min_days=60):
    """
    {fund: score} for one stock, where score is the correlation of market-removed daily returns
    over each fund's overlapping history (up to the year fetched; less for young funds like
    NCLD). Funds sharing fewer than `min_days` days with the stock are left out.
    """
    if MARKET_PROXY not in fund_closes:
        return {}
    market = fund_closes[MARKET_PROXY].pct_change(fill_method=None)
    stock = _market_removed(stock_closes.pct_change(fill_method=None), market)
    if len(stock) < min_days:
        return {}
    scores = {}
    for fund in CANDIDATE_FUNDS:
        if fund not in fund_closes:
            continue
        fund_part = _market_removed(fund_closes[fund].pct_change(fill_method=None), market)
        both = pd.concat([stock, fund_part], axis=1, join="inner").dropna()
        if len(both) >= min_days:
            scores[fund] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
    return scores


def assign_sectors(stock_closes, fund_closes, labels, overrides=None, config=None):
    """
    {symbol: {"fund", "name", "method", "score"}} for every stock that can be placed.

    stock_closes: {symbol: Series of daily closes}; fund_closes: DataFrame of closes with the
    candidate funds and SPY as columns; labels: {symbol: (yahoo_sector, yahoo_industry)}.
    'method' says how the fund was chosen: "override", "price match" or "label".
    """
    config = config or {}
    overrides = {k.upper(): v.upper() for k, v in (overrides or {}).items()}
    min_score = float(config.get("min_match_score", 0.30))
    min_lead = float(config.get("min_match_lead", 0.03))
    min_days = int(config.get("min_history_days", 60))

    groups = {}
    for symbol, closes in stock_closes.items():
        if symbol in overrides:
            groups[symbol] = _group(overrides[symbol], "override", None)
            continue
        scores = match_scores(closes, fund_closes, min_days) if closes is not None else {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if (
            ranked
            and ranked[0][1] >= min_score
            and (len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= min_lead)
        ):
            groups[symbol] = _group(ranked[0][0], "price match", ranked[0][1])
            continue
        fund = fund_from_label(*labels.get(symbol, (None, None)))
        if fund:
            groups[symbol] = _group(fund, "label", None)
    return groups


def _group(fund, method, score):
    return {"fund": fund, "name": FUND_NAMES.get(fund, fund), "method": method, "score": score}


def sector_context(groups, fund_moves, stock_moves):
    """
    {symbol: {"fund", "name", "gap_pct", "last_session_pct", "stock", "peers"}} for the explainer.

    fund_moves comes from analyzer.analyze_market() on the funds. stock_moves is
    {symbol: {"gap_pct", "last_session_pct"}} for the watchlist; each entry's own moves go in
    "stock" (so the explainer can compare it with its fund), and "peers" lists the other
    watchlist stocks in the same group with their pre-market moves.
    """
    context = {}
    for symbol, group in groups.items():
        move = fund_moves.get(group["fund"])
        if not move:
            continue
        peers = [
            (other, (stock_moves.get(other) or {}).get("gap_pct"))
            for other, g in groups.items()
            if other != symbol and g["fund"] == group["fund"]
        ]
        context[symbol] = {
            "fund": group["fund"],
            "name": group["name"],
            "gap_pct": move.get("gap_pct"),
            "last_session_pct": move.get("last_session_pct"),
            "stock": stock_moves.get(symbol) or {},
            "peers": peers,
        }
    return context
