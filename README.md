# Stock-Analysis-Report

Every weekday before the market opens, this pipeline emails a **pre-market brief** to a list of
recipients. The top of the email calls out big or concerning moves in your watchlist, and each
stock then gets its own section. A PDF newsletter is attached: a cover page with the alerts and
a watchlist table, then one page per stock with a chart and key metrics.

It runs for free on GitHub Actions, using Finnhub's free tier, Yahoo Finance (via `yfinance`) and Gmail.

```
fetch (src/fetcher.py) → analyze (src/analyzer.py) → format (src/formatter.py + src/pdf_report.py) → send (src/dispatcher.py)
```

## What's in the brief

| Section | What it shows |
| --- | --- |
| Market backdrop | S&P 500 (SPY) and Nasdaq-100 (QQQ) pre-market moves, and the VIX |
| Big & concerning moves | Stocks that crossed a threshold, most urgent first, with the reason for each |
| Watchlist at a glance | Pre-market price, gap, last session, RSI and volume for every stock |
| Stock pages | Key metrics, alerts, signals and the latest headlines (the PDF adds a 6-month chart) |
| Outside your watchlist | Yesterday's biggest US gainers, losers and most-active stocks |

Alerts fire on:

- a large pre-market gap or last-session move, either past a fixed threshold or unusual for that particular stock
- a volume surge
- RSI extremes
- a close through the 50- or 200-day average
- a golden or death cross
- a 52-week high or low
- earnings within a week

Every threshold is in `config.yaml`.

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Get your credentials

| Variable | Where it comes from |
| --- | --- |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io) → Dashboard → API key (free) |
| `SENDER_EMAIL` | The Gmail address that sends the brief |
| `EMAIL_PASSWORD` | A Gmail **App Password**: Google Account → Security → 2-Step Verification (must be on) → App passwords → create one named "Stock Pipeline". Gmail shows 16 letters; spaces are optional. |
| `RECIPIENT_EMAILS` | Everyone who should get it, comma-separated: `a@x.com, b@y.com` |

Recipients are BCC'd, so nobody sees the other addresses.

### 3. Try it locally

```bash
export FINNHUB_API_KEY="..."
python main.py --dry-run          # builds output/premarket_<date>.html/.pdf/.txt without sending
```

Open the files in `output/` to preview. When the preview looks right, set the email variables
and run `python main.py` to send it for real.

| Flag | Effect |
| --- | --- |
| `--dry-run` | Build everything, send nothing. Works without a Finnhub key (Yahoo data only). |
| `--force` | Send even if the market looks closed today (weekends/holidays are skipped automatically). |

### 4. Automate it (GitHub Actions)

1. Push the repo to GitHub.
2. Go to **Settings → Secrets and variables → Actions → New repository secret** and add the four
   variables above as secrets.
3. Go to **Actions → Pre-Market Report → Run workflow** and tick *dry run* for a first test. The
   report is saved under the run's **Artifacts**.

After that it runs on its own at **7:00 AM Central (8:00 AM New York), Monday to Friday**, and
skips market holidays. Daylight saving time is handled automatically. GitHub can start scheduled
runs 5–15 minutes late, so the workflow starts at 6:45 AM Central and waits until 7:00 before it
fetches and sends; the email lands a minute or two after 7. If a run fails, GitHub emails you.

> GitHub pauses scheduled workflows in public repos after 60 days without a commit. If the
> emails stop, re-enable the workflow on the Actions tab or push any commit.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use synthetic data (`tests/fixtures.py`) and never touch the network. They also run on
every push (`.github/workflows/tests.yml`).

## Project layout

```
.github/workflows/premarket_report.yml   weekday schedule + manual "Run workflow" button
.github/workflows/tests.yml              runs the tests on every push
config.yaml                              watchlist, thresholds, report settings (no secrets)
main.py                                  orchestrates fetch → analyze → format → send
src/fetcher.py                           Finnhub (quotes, news, fundamentals, earnings) + yfinance (history, pre-market, movers)
src/analyzer.py                          indicators, alert rules, severity ranking
src/formatter.py                         email HTML + plain text + subject line
src/pdf_report.py                        PDF newsletter (cover page + one page per stock)
src/charts.py                            price/volume chart for each stock page
src/dispatcher.py                        Gmail SMTP sender (BCC list, PDF attachment)
tests/                                   unit tests + synthetic fixtures
```

Automated analysis for information only. This is not investment advice.
