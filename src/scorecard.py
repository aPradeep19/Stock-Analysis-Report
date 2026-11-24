"""
Yesterday's scorecard: how the previous report's header alerts played out.

Every real send logs its header alerts as JSON lines. The workflow keeps that log on a separate
`history` branch, fetches it before each run and appends the new rows after a successful send.
Each run then scores the most recent logged report against the session that followed it, using
the daily candles main.py has already fetched (no extra API calls).

Only up/down alerts are scored, on three measures:
  gap held        the session closed on the flagged side of the close the alert was measured from
  follow-through  close vs. the price at alert time, signed so + means "further in the flagged
                  direction"
  vs. benchmark   follow-through minus the benchmark's own move over the same span
Earnings alerts get their realized move against the stock's typical daily move instead.

Nothing here may block the email: reading, scoring and logging failures all come back as a
status the report can show ("unavailable"), and main.py wraps every call.
"""

import json
import os
from datetime import date

import pandas as pd

HIT_RATE_MIN_DAYS = 20
DIRECTIONS = {"up": 1, "down": -1}
KIND_LABELS = {
    "move": "Price move",
    "volume": "Volume",
    "trend": "Moving average",
    "range": "52-week high/low",
    "momentum": "MACD",
}
BENCHMARK_NAMES = {"QQQ": "Nasdaq", "SPY": "S&P 500"}

FOOTNOTE = (
    "Held: the session closed on the flagged side of the prior close. Follow-through: close vs. "
    "the price when the alert went out, + meaning further in the flagged direction. vs. {bench}: "
    "follow-through minus the {bench}'s own move over the same span."
)


# ---------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------- #
def alert_rows(ranked, results, market, day, benchmark):
    """One dict per header alert in today's report, ready to log."""
    bench = market.get(benchmark) or {}
    rows = []
    for item in ranked:
        metrics = (results.get(item["symbol"]) or {}).get("metrics", {})
        for alert in item["alerts"]:
            row = {
                "date": day.isoformat(),
                "symbol": item["symbol"],
                "kind": alert.get("kind"),
                "direction": alert.get("direction"),
                "severity": alert.get("severity"),
                "text": alert.get("text"),
                "session": metrics.get("premarket_session") or "pre-market",
                "last_close": metrics.get("last_close"),
                "price": metrics.get("premarket_price"),
                "gap_pct": metrics.get("gap_pct"),
                "typical_move_pct": metrics.get("typical_move_pct"),
                "benchmark": benchmark if bench else None,
                "benchmark_last_close": bench.get("last_close"),
                "benchmark_price": bench.get("premarket_price"),
                "benchmark_gap_pct": bench.get("gap_pct"),
            }
            if alert.get("kind") == "event" and metrics.get("earnings_date"):
                row["earnings_date"] = metrics["earnings_date"]
                row["earnings_hour"] = metrics.get("earnings_hour")
            rows.append({k: _plain(v) for k, v in row.items()})
    return rows


def _plain(value):
    """JSON-safe: numpy/pandas scalars to Python, NaN to None."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if pd.isna(value) else round(value, 4)


def load_history(path):
    """
    (rows, status) where status is "ok", "missing" (no file yet: first run) or "error".
    Unreadable lines are skipped with a warning rather than failing the whole file.
    """
    if not path or not os.path.exists(path):
        return [], "missing"
    rows, bad = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(row, dict) and row.get("date") and row.get("symbol"):
                    rows.append(row)
                else:
                    bad += 1
    except OSError as e:
        print(f"    [WARN] Alert history unreadable: {e}")
        return [], "error"
    if bad:
        print(f"    [WARN] Skipped {bad} unreadable line(s) in the alert history.")
    return rows, "ok"


def write_pending(rows, history, path):
    """
    Writes today's rows for the workflow to append to the history branch. Returns the number
    written: 0 when there's nothing to log or the date is already logged (a second send that
    day), so each report date is logged once.
    """
    if not rows:
        return 0
    logged_dates = {row.get("date") for row in history}
    rows = [row for row in rows if row["date"] not in logged_dates]
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


# ---------------------------------------------------------------------- #
# Scoring
# ---------------------------------------------------------------------- #
def build_scorecard(history, status, candles, today):
    """
    Everything the renderers need, as a plain dict:
      status       "ok", "none" (nothing logged before today) or "unavailable" (history error)
      date         the report date being scored
      rows         one per directional alert from that report that could be scored
      skipped      directional alerts with no bar for their session (holiday, missed day...)
      summary      {"scored", "held", "avg_follow", "avg_vs_bench"}
      benchmark    benchmark symbol and display name
      earnings     realized move vs. typical for earnings that reported in the latest session
      hit_rates    per alert kind, once HIT_RATE_MIN_DAYS distinct report dates are logged
    `candles` is {symbol: daily DataFrame}, including the benchmark(s).
    """
    if status == "error":
        return {"status": "unavailable"}
    today_iso = today.isoformat()
    past = [row for row in history if str(row.get("date")) < today_iso]
    if not past:
        return {"status": "none"}

    latest = max(row["date"] for row in past)
    scored, skipped = [], 0
    for row in past:
        if row["date"] != latest or row.get("direction") not in DIRECTIONS:
            continue
        result = score_alert(row, candles)
        if result is None:
            skipped += 1
        else:
            scored.append(result)
    scored.sort(key=lambda r: -(r.get("severity") or 0))

    benchmark = next((r.get("benchmark") for r in past if r["date"] == latest and r.get("benchmark")), "QQQ")
    days_logged = len({row["date"] for row in past})
    return {
        "status": "ok",
        "date": latest,
        "rows": scored,
        "skipped": skipped,
        "summary": _summary(scored),
        "benchmark": {"symbol": benchmark, "name": BENCHMARK_NAMES.get(benchmark, benchmark)},
        "earnings": earnings_reactions(past, candles, benchmark),
        "hit_rates": hit_rates(past, candles) if days_logged >= HIT_RATE_MIN_DAYS else None,
        "days_logged": days_logged,
    }


def _session_bar(df, logged_date, session):
    """
    (bar, previous bar) for the session an alert is judged on, or None if it isn't there.
    A pre-market or intraday alert is judged on that day's session; an after-hours one on the
    next trading day's. Looked up by date, never guessed.
    """
    if df is None or df.empty:
        return None
    dates = [d.date() for d in df.index]
    day = date.fromisoformat(logged_date)
    if session == "after-hours":
        later = [i for i, d in enumerate(dates) if d > day]
        index = later[0] if later else None
    else:
        index = dates.index(day) if day in dates else None
    if index is None or index == 0:
        return None
    return df.iloc[index], df.iloc[index - 1]


def score_alert(row, candles):
    """Scores one up/down alert, or None when its session bar is missing."""
    found = _session_bar(candles.get(row["symbol"]), row["date"], row.get("session"))
    if found is None:
        return None
    bar, _ = found
    sign = DIRECTIONS[row["direction"]]
    close = float(bar["close"])
    base, price = row.get("last_close"), row.get("price")

    held = None if base is None else (close > base if sign > 0 else close < base)
    follow = None if not price else sign * (close / price - 1) * 100

    vs_bench = None
    bench_found = _session_bar(candles.get(row.get("benchmark")), row["date"], row.get("session"))
    bench_from = row.get("benchmark_price") or row.get("benchmark_last_close")
    if follow is not None and bench_found is not None and bench_from:
        bench_move = (float(bench_found[0]["close"]) / bench_from - 1) * 100
        vs_bench = follow - sign * bench_move

    return {
        "symbol": row["symbol"],
        "kind": row.get("kind"),
        "kind_label": KIND_LABELS.get(row.get("kind"), (row.get("kind") or "").title()),
        "direction": row["direction"],
        "severity": row.get("severity"),
        "text": row.get("text"),
        "price": price,
        "base": base,
        "close": close,
        "held": held,
        "follow_pct": follow,
        "vs_bench_pct": vs_bench,
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summary(scored):
    judged = [r for r in scored if r["held"] is not None]
    return {
        "scored": len(judged),
        "held": sum(1 for r in judged if r["held"]),
        "avg_follow": _mean(r["follow_pct"] for r in scored),
        "avg_vs_bench": _mean(r["vs_bench_pct"] for r in scored),
    }


def earnings_reactions(history, candles, benchmark):
    """
    Earnings whose reaction session is the most recent completed one: the report day itself for
    a release before the open, the next session for one after the close. Each stock's event is
    shown once, however many days it was flagged in advance.
    """
    reference = candles.get(benchmark)
    if reference is None or reference.empty:
        return []
    latest_session = reference.index[-1].date()
    seen, out = set(), []
    for row in sorted(history, key=lambda r: r["date"], reverse=True):
        event, symbol = row.get("earnings_date"), row["symbol"]
        if not event or (symbol, event) in seen:
            continue
        seen.add((symbol, event))
        session = "after-hours" if row.get("earnings_hour") == "amc" else "pre-market"
        found = _session_bar(candles.get(symbol), str(event), session)
        if found is None:
            continue
        bar, previous = found
        if bar.name.date() != latest_session:
            continue
        move = (float(bar["close"]) / float(previous["close"]) - 1) * 100
        typical = row.get("typical_move_pct")
        out.append(
            {
                "symbol": symbol,
                "event_date": str(event),
                "hour": row.get("earnings_hour"),
                "move_pct": move,
                "typical_pct": typical,
                "multiple": abs(move) / typical if typical else None,
            }
        )
    return out


def hit_rates(history, candles):
    """Per alert kind across every logged report: how often the gap held, and average results."""
    by_kind = {}
    for row in history:
        if row.get("direction") not in DIRECTIONS:
            continue
        result = score_alert(row, candles)
        if result is not None:
            by_kind.setdefault(result["kind_label"], []).append(result)
    table = []
    for label, results in by_kind.items():
        summary = _summary(results)
        table.append(
            {
                "kind": label,
                "scored": summary["scored"],
                "held_pct": summary["held"] / summary["scored"] * 100 if summary["scored"] else None,
                "avg_follow": summary["avg_follow"],
                "avg_vs_bench": summary["avg_vs_bench"],
            }
        )
    table.sort(key=lambda r: -r["scored"])
    return table


# ---------------------------------------------------------------------- #
# Wording shared by the email, text and PDF
# ---------------------------------------------------------------------- #
def summary_line(card):
    """The one-line headline for the section."""
    status = (card or {}).get("status")
    if status == "unavailable":
        return "Alert history unavailable today."
    if status != "ok":
        return "No prior alerts to score."
    day = date.fromisoformat(card["date"])
    when = f"{day:%a %b} {day.day}"
    s = card["summary"]
    if not s["scored"]:
        note = f" ({card['skipped']} couldn't be scored: no session data)" if card["skipped"] else ""
        return f"{when}: no directional alerts to score{note}."
    text = f"{when}: {s['held']} of {s['scored']} directional alert{'s' if s['scored'] != 1 else ''} held"
    if s["avg_follow"] is not None:
        text += f"; avg follow-through {s['avg_follow']:+.1f}%"
    if s["avg_vs_bench"] is not None:
        text += f" ({s['avg_vs_bench']:+.1f}% vs. {card['benchmark']['name']})"
    if card["skipped"]:
        text += f"; {card['skipped']} not scored (no session data)"
    return text + "."


def footnote(card):
    return FOOTNOTE.format(bench=((card or {}).get("benchmark") or {}).get("name", "Nasdaq"))


def earnings_line(item):
    day = date.fromisoformat(item["event_date"])
    timing = {"bmo": f"before {day:%a}'s open", "amc": f"after {day:%a}'s close"}.get(
        item.get("hour"), f"{day:%a}"
    )
    text = f"{item['symbol']} earnings ({timing}): moved {item['move_pct']:+.1f}%"
    if item.get("typical_pct"):
        text += f" vs. a typical {item['typical_pct']:.1f}% ({item['multiple']:.1f}x)"
    return text
