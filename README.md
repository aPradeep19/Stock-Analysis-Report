# Stock-Analysis-Report

A pre-market brief for the stocks I follow, emailed every weekday at 7 AM Central.

## Today's report

Updated every weekday morning. This is the first page; click it to read the full report.

[![Today's pre-market brief](https://raw.githubusercontent.com/aPradeep19/Stock-Analysis-Report/reports/latest.png)](https://github.com/aPradeep19/Stock-Analysis-Report/blob/reports/latest.pdf)

## How it works

It pulls quotes, news and earnings dates from Finnhub, price history from Yahoo Finance, flags anything worth looking at (big gaps, volume spikes, RSI extremes, moving average crosses, 52-week highs/lows, upcoming earnings) and sends it out with a PDF attached. Runs for free on GitHub Actions.

The watchlist and alert thresholds live in `config.yaml`.

## Setup

```bash
pip install -r requirements.txt
```

You'll need four environment variables (or GitHub secrets):

- `FINNHUB_API_KEY`: free key from [finnhub.io](https://finnhub.io)
- `SENDER_EMAIL`: the Gmail account that sends the email
- `EMAIL_PASSWORD`: a Gmail [App Password](https://myaccount.google.com/apppasswords), not your normal password
- `RECIPIENT_EMAILS`: comma-separated list of who gets it (everyone is BCC'd)

## Running it

```bash
python main.py --dry-run   # build the report into output/ without sending
python main.py             # build and send
```

It skips weekends and holidays on its own. Use `--force` to send anyway.

## Scheduling

The workflow in `.github/workflows/premarket_report.yml` sends the email at 7 AM Central, Monday to Friday. Add the four variables above as repo secrets. You can also start it by hand from the Actions tab.

GitHub's own cron is often hours late, so the workflow is meant to be started by an outside scheduler at 7:00 AM:

1. Create a fine-grained personal access token for this repo with **Actions: Read and write**.
2. On a free scheduler such as [cron-job.org](https://cron-job.org), add a job for 7:00 AM `America/Chicago`, Monday to Friday:
   - `POST https://api.github.com/repos/<owner>/<repo>/actions/workflows/premarket_report.yml/dispatches`
   - Headers: `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`
   - Body: `{"ref":"main","inputs":{"automated":"true"}}`

GitHub's cron stays on as a backup. Automated runs skip themselves if today's report already went out or it's past 8:30 AM Central (the market is open, so the report would be stale). Runs started by hand always go through.

GitHub pauses scheduled workflows after 60 days with no commits, so if the backup stops, re-enable it there.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

Not investment advice.
