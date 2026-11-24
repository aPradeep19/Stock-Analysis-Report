"""
News sentiment with FinBERT (ProsusAI/finbert), a free model trained on financial news sentences.

Two steps:
  1. score_news() runs every article for every ticker through the model in one batched pass on
     CPU, and attaches positive / negative / neutral probabilities to each article.
  2. stock_sentiment() turns one ticker's scored articles into the report's label. Only
     articles that name the stock count (feeds include roundups that mention it in passing, and
     FinBERT scores the article's tone, not the stock's), and fresher articles weigh more.

The model is optional. transformers and torch are imported only when it runs; if they're missing,
the download fails or anything else goes wrong, one warning is logged, the articles stay
unscored, and the analyzer falls back to its keyword count. The report always builds.
"""

import re
import time

from src.timing import LAST_SESSION, OLDER, SINCE_CLOSE, UNKNOWN, close_times, timing

DEFAULT_MODEL = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")

DEFAULT_WEIGHTS = {SINCE_CLOSE: 1.0, LAST_SESSION: 0.6, OLDER: 0.3, UNKNOWN: 0.5}

# Endings stripped from company names before matching ("NVIDIA Corp" -> "NVIDIA").
LEGAL_SUFFIXES = re.compile(
    r"[\s,]+(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|nv|n\.v|sa|s\.a|ag|se|"
    r"holdings?|class [a-c]|cl [a-c]|common stock|ordinary shares?|shs|ads|adr)\.?$",
    re.IGNORECASE,
)
# First words too common to identify a company on their own ("General Electric" -> not "General").
GENERIC_WORDS = {
    "general", "american", "first", "united", "national", "international", "global", "world",
    "digital", "advanced", "applied", "american", "new", "north", "south", "east", "west", "great",
    "royal", "first", "capital", "group", "energy", "technology", "technologies", "systems", "health",
}  # fmt: skip


# ---------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------- #
def load_classifier(config):
    """A CPU text-classification pipeline for the configured model, or None (with one warning)."""
    model = config.get("model", DEFAULT_MODEL)
    started = time.perf_counter()
    try:
        from transformers import pipeline  # lazy: torch/transformers are optional installs
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # one "Loading weights" bar per run clutters the log
        classifier = pipeline("text-classification", model=model, tokenizer=model, device=-1, top_k=None)
    except Exception as e:
        print(f"    [WARN] FinBERT unavailable, using keyword sentiment: {type(e).__name__}: {str(e)[:160]}")
        return None
    print(f"    FinBERT loaded ({model}) in {time.perf_counter() - started:.1f}s")
    return classifier


# Some feeds fill the summary with a teaser for a different story ("Today's big story: Nvidia
# added $954 billion...") or prefix it with a heading. Tested on real watchlist news (Nov 2026):
# teasers flipped neutral headlines negative, so they're dropped and headings stripped.
DEFAULT_SKIP_SUMMARIES = ("today's big story", "today’s big story")
LEADING_HEADINGS = re.compile(r"^(?:key takeaways|summary|highlights)\s*:?\s*", re.IGNORECASE)


def _summary_lead(summary, sentences, skip_prefixes):
    summary = " ".join((summary or "").split())
    summary = LEADING_HEADINGS.sub("", summary)
    if not summary or summary.lower().startswith(tuple(p.lower() for p in skip_prefixes)):
        return ""
    return " ".join(re.split(r"(?<=[.!?])\s+", summary)[:sentences])


def article_text(article, mode="headline+summary", sentences=2, skip_prefixes=DEFAULT_SKIP_SUMMARIES):
    """What the model reads: the headline, plus the first sentence or two of the summary."""
    headline = " ".join((article.get("headline") or "").split())
    if mode == "headline":
        return headline
    lead = _summary_lead(article.get("summary"), sentences, skip_prefixes)
    # Many summaries just repeat the headline; don't feed it twice.
    if lead and lead.lower().rstrip(".") != headline.lower().rstrip("."):
        return (
            f"{headline}. {lead}"
            if headline and not headline.endswith((".", "!", "?"))
            else f"{headline} {lead}"
        )
    return headline


def _probabilities(output):
    """{"positive": p, "negative": p, "neutral": p} from one pipeline result (a list of label/score)."""
    if isinstance(output, dict):
        output = [output]
    probs = {label: 0.0 for label in LABELS}
    for item in output:
        label = str(item.get("label", "")).lower()
        if label in probs:
            probs[label] = float(item.get("score", 0.0))
    return probs


def score_news(pipeline_data, config, classifier=None):
    """
    Attaches article["sentiment"] = {"positive", "negative", "neutral", "label"} to every article
    in pipeline_data, scoring each distinct text once in one batched pass. Returns True if the
    articles were scored, False if the model was unavailable (articles left as they were).
    """
    if config.get("enabled", True) is False:
        return False
    articles = [a for data in pipeline_data.values() for a in data.get("news") or []]
    if not articles:
        return False
    mode = config.get("input", "headline+summary")
    sentences = int(config.get("summary_sentences", 2))
    skip = tuple(config.get("skip_summaries_starting_with") or DEFAULT_SKIP_SUMMARIES)
    texts = {id(a): article_text(a, mode, sentences, skip) for a in articles}
    unique = list(dict.fromkeys(t for t in texts.values() if t))
    if not unique:
        return False

    classifier = classifier or load_classifier(config)
    if classifier is None:
        return False
    started = time.perf_counter()
    try:
        outputs = classifier(
            unique, batch_size=int(config.get("batch_size", 16)), truncation=True, max_length=512, top_k=None
        )
        scores = {text: _probabilities(out) for text, out in zip(unique, outputs, strict=True)}
    except Exception as e:
        print(
            f"    [WARN] FinBERT scoring failed, using keyword sentiment: {type(e).__name__}: {str(e)[:160]}"
        )
        return False

    for article in articles:
        probs = scores.get(texts[id(article)])
        if probs:
            article["sentiment"] = {**probs, "label": max(LABELS, key=probs.get)}
    print(
        f"    FinBERT scored {len(unique)} distinct texts ({len(articles)} articles) "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return True


def compare_inputs(pipeline_data, classifier, sample=20, sentences=2):
    """
    Lines comparing headline-only and headline+summary scores on real articles, to choose the
    `input` setting. Only articles that have a summary are worth comparing.
    """
    picked = []
    for symbol, data in pipeline_data.items():
        for article in data.get("news") or []:
            if (article.get("summary") or "").strip() and len(picked) < sample:
                picked.append((symbol, article))
    if not picked:
        return ["No articles with summaries to compare."]
    short = [article_text(a, "headline") for _, a in picked]
    long = [article_text(a, "headline+summary", sentences) for _, a in picked]
    out_short = classifier(short, truncation=True, max_length=512, top_k=None)
    out_long = classifier(long, truncation=True, max_length=512, top_k=None)
    lines = []
    for (symbol, article), a, b, text in zip(picked, out_short, out_long, long, strict=True):
        pa, pb = _probabilities(a), _probabilities(b)
        lines.append(
            f"{symbol:5} headline {max(LABELS, key=pa.get):8} {pa['positive'] - pa['negative']:+.2f} | "
            f"+summary {max(LABELS, key=pb.get):8} {pb['positive'] - pb['negative']:+.2f} | {text[:150]}"
        )
    return lines


# ---------------------------------------------------------------------- #
# Relevance and the stock-level score
# ---------------------------------------------------------------------- #
def name_terms(symbol, name, aliases=None):
    """
    (ticker, [lowercase name phrases]) an article must contain to count for this stock: the
    company name without legal endings, its distinctive first word, and any configured aliases
    (e.g. Google for GOOGL).
    """
    phrases = []
    if name:
        cleaned = name.strip()
        while True:  # endings can stack: "Nebius Group N.V. Class A"
            shorter = LEGAL_SUFFIXES.sub("", cleaned).strip(" ,.")
            if shorter == cleaned:
                break
            cleaned = shorter
        if cleaned:
            phrases.append(cleaned.lower())
            first = cleaned.split()[0].strip(",.").lower()
            if len(first) >= 5 and first not in GENERIC_WORDS and first != cleaned.lower():
                phrases.append(first)
    phrases += [a.lower() for a in (aliases or []) if a]
    return symbol, list(dict.fromkeys(phrases))


def is_relevant(article, terms):
    """Whether the headline or summary names the stock (ticker in capitals, or a name phrase)."""
    symbol, phrases = terms
    text = f"{article.get('headline') or ''} {article.get('summary') or ''}"
    # Ticker as a whole word in capitals only, so NOW doesn't match "now" and MU not "mu".
    if symbol and re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", text):
        return True
    lowered = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", lowered) for p in phrases)


def label_for(score, relevant, config):
    """
    Bullish / Bearish / Mixed / Neutral for a weighted score over `relevant` article sentiments.
    Shared by the per-stock label and the sector / watchlist mood.
    """
    threshold = float(config.get("label_threshold", 0.25))
    confident = float(config.get("confident_probability", 0.70))
    if len(relevant) < int(config.get("min_relevant_articles", 2)):
        return "Neutral"
    has_pos = any(s["positive"] >= confident for s in relevant)
    has_neg = any(s["negative"] >= confident for s in relevant)
    if has_pos and has_neg and abs(score) < threshold:
        return "Mixed"
    if score >= threshold:
        return "Bullish"
    if score <= -threshold:
        return "Bearish"
    return "Neutral"


def stock_sentiment(news, symbol, name, candles, config):
    """
    The report's sentiment dict for one stock from FinBERT-scored articles, or None when none
    of its articles were scored (the caller then uses the keyword method).

    Per article: P(positive) - P(negative). Stock score: the average over relevant articles,
    weighted by freshness. Labels: Bullish / Bearish past the threshold, Mixed when confident
    positive and negative articles offset each other, otherwise Neutral; anything but Neutral
    needs `min_relevant_articles`. Also returns the strongest relevant story, if one stands out.
    """
    scored = [a for a in news or [] if a.get("sentiment")]
    if not scored:
        return None
    terms = name_terms(symbol, name, (config.get("aliases") or {}).get(symbol))
    weights = {**DEFAULT_WEIGHTS, **(config.get("weights") or {})}
    last_close, prev_close = close_times(candles)

    relevant, total_weight, weighted = [], 0.0, 0.0
    for article in scored:
        counted = is_relevant(article, terms)
        article["sentiment"]["counted"] = counted
        if not counted:
            continue
        weight = float(weights.get(timing(article.get("datetime"), last_close, prev_close), 0.5))
        weighted += weight * _net(article)
        total_weight += weight
        relevant.append(article)

    score = weighted / total_weight if total_weight else 0.0
    sentiments = [a["sentiment"] for a in relevant]
    counts = {label: sum(1 for s in sentiments if s["label"] == label) for label in LABELS}
    return {
        "score": round(score, 3),
        "label": label_for(score, sentiments, config),
        "positive": counts["positive"],
        "negative": counts["negative"],
        "neutral": counts["neutral"],
        "method": "finbert",
        "relevant": len(relevant),
        "total": len(scored),
        "strongest": strongest_story(relevant, config),
    }


def _net(article):
    s = article["sentiment"]
    return s["positive"] - s["negative"]


def strongest_story(relevant, config):
    """
    The relevant article with the clearest tone, so the label can be tied to an actual story.
    None when nothing is clearly positive or negative (|P(pos) - P(neg)| below the bar).
    """
    if not relevant:
        return None
    best = max(relevant, key=lambda a: (abs(_net(a)), a.get("datetime") or 0))
    net = _net(best)
    if abs(net) < float(config.get("strongest_min_net", 0.5)):
        return None
    return {
        "headline": best.get("headline"),
        "source": best.get("source"),
        "url": best.get("url"),
        "datetime": best.get("datetime"),
        "net": round(net, 3),
        "label": best["sentiment"]["label"],
    }


# ---------------------------------------------------------------------- #
# Derived signals (shown on the stock's page; sentiment doesn't raise alerts yet)
# ---------------------------------------------------------------------- #
# Which stories can bear on a move: news after the last close for a gap since that close, and
# news during the session for a move in that session (this morning's stories about today's
# pullback say nothing about why yesterday's session rallied).
MOVE_WINDOWS = {
    "overnight": ({SINCE_CLOSE}, "since the last close"),
    "last session": ({LAST_SESSION}, "during that session"),
}
# Stories newer than each window. If every article in the feed is newer, the feed (10 newest per
# stock) doesn't reach back that far, so "no stories in that window" would be a false claim.
NEWER_THAN = {"overnight": set(), "last session": {SINCE_CLOSE}}


def window_sentiment(news, candles, window, config):
    """
    FinBERT tone of the relevant stories published in one move's window (see MOVE_WINDOWS), as
    {"score", "label", "relevant", "method", "window_words"}; None if no article was scored.
    Uses the relevance flags stock_sentiment() has already set on each article.
    """
    if window not in MOVE_WINDOWS or not any(a.get("sentiment") for a in news or []):
        return None
    buckets, words = MOVE_WINDOWS[window]
    last_close, prev_close = close_times(candles)
    picked = [
        a
        for a in news or []
        if (a.get("sentiment") or {}).get("counted")
        and timing(a.get("datetime"), last_close, prev_close) in buckets
    ]
    score = sum(_net(a) for a in picked) / len(picked) if picked else 0.0
    times = [timing(a.get("datetime"), last_close, prev_close) for a in news or []]
    return {
        "score": round(score, 3),
        "label": label_for(score, [a["sentiment"] for a in picked], config),
        "relevant": len(picked),
        "method": "finbert",
        "window_words": words,
        "covered": any(t not in NEWER_THAN[window] for t in times),
    }


def _stories(n):
    return f"{n} {'story' if n == 1 else 'stories'}"


def divergence(move_pct, move_words, sentiment, config):
    """
    A line when a flagged price move and the news from the same window disagree, else None:
      - the stock rose on Bearish coverage, or fell on Bullish coverage
      - a move with no relevant news in its window, or only neutral / mixed coverage
    A move with no supporting news is often worth checking: flows, sector rotation, or news the
    feeds haven't caught yet. `sentiment` is window_sentiment() for the move's window.
    """
    if move_pct is None or not sentiment or sentiment.get("method") != "finbert":
        return None
    move = f"{'up' if move_pct > 0 else 'down'} {abs(move_pct):.1f}% {move_words}"
    when = f" {sentiment['window_words']}" if sentiment.get("window_words") else ""
    label, relevant = sentiment.get("label"), sentiment.get("relevant", 0)
    if (move_pct > 0 and label == "Bearish") or (move_pct < 0 and label == "Bullish"):
        return (
            f"News vs. price: {move} despite {label.lower()} coverage "
            f"(score {sentiment['score']:+.2f} over {_stories(relevant)} naming the stock{when})"
        )
    if relevant == 0:
        if not sentiment.get("covered", True):
            return None  # the feed doesn't reach back into this window: no claim either way
        return f"News vs. price: {move} with no stories naming the stock{when}"
    if label in ("Neutral", "Mixed") and abs(sentiment["score"]) < float(config.get("label_threshold", 0.25)):
        return (
            f"News vs. price: {move} without a clear news lead ({label.lower()} coverage, "
            f"score {sentiment['score']:+.2f} over {_stories(relevant)}{when})"
        )
    return None


def shift(today, history, config):
    """
    A line when today's news tone differs clearly from the stock's own recent average, else None.
    `history` is the stock's earlier daily scores, oldest first (FinBERT days only).
    """
    if not today or today.get("method") != "finbert" or today.get("relevant", 0) < 1:
        return None
    window = int(config.get("shift_window_days", 5))
    past = [h for h in history if h is not None][-window:]
    if len(past) < int(config.get("shift_min_days", 3)):
        return None
    average = sum(past) / len(past)
    change = today["score"] - average
    if abs(change) < float(config.get("shift_min_change", 0.4)):
        return None
    word = "more positive" if change > 0 else "more negative"
    return (
        f"News tone turned {word}: {today['score']:+.2f} today vs. a {average:+.2f} average over the "
        f"previous {len(past)} report days"
    )


def news_mood(results, groups, config):
    """
    Watchlist-wide and per-sector-group news tone from every relevant, FinBERT-scored story, as
    {"watchlist": mood, "groups": [mood, ...]}, or None without FinBERT. A story shared by
    several stocks counts once.
    """

    def mood(articles, name):
        seen, unique = set(), []
        for a in articles:
            key = (a.get("url") or "", (a.get("headline") or "").lower())
            if key not in seen:
                seen.add(key)
                unique.append(a)
        if not unique:
            return None
        score = sum(_net(a) for a in unique) / len(unique)
        sentiments = [a["sentiment"] for a in unique]
        return {
            "name": name,
            "score": round(score, 3),
            "label": label_for(score, sentiments, config),
            "stories": len(unique),
            "positive": sum(1 for s in sentiments if s["label"] == "positive"),
            "negative": sum(1 for s in sentiments if s["label"] == "negative"),
        }

    relevant_by_symbol = {}
    for symbol, result in results.items():
        if (result.get("sentiment_detail") or {}).get("method") != "finbert":
            continue
        relevant_by_symbol[symbol] = [
            a for a in result.get("all_news") or [] if (a.get("sentiment") or {}).get("counted")
        ]
    if not relevant_by_symbol:
        return None

    watchlist = mood([a for arts in relevant_by_symbol.values() for a in arts], "Watchlist")
    by_fund = {}
    for symbol, group in (groups or {}).items():
        if symbol in relevant_by_symbol:
            by_fund.setdefault((group["fund"], group["name"]), []).append(symbol)
    group_moods = []
    for (fund, name), symbols in sorted(by_fund.items()):
        if len(symbols) < 2:
            continue  # a "sector" of one stock is just that stock's own label
        m = mood([a for s in symbols for a in relevant_by_symbol[s]], name.replace(" ETF", ""))
        if m and m["stories"] >= int(config.get("min_relevant_articles", 2)):
            group_moods.append({**m, "fund": fund, "symbols": symbols})
    return {"watchlist": watchlist, "groups": group_moods}
