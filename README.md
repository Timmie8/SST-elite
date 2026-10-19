# SST Elite AI Trading Terminal — GitHub deployment package

This is the SST Elite Terminal MVP 0.1 prepared for deployment from GitHub to Render. It is a Python FastAPI application; **GitHub stores the source code, while Render runs the live web application**. GitHub Pages cannot run this Python backend.

## Included modules

1. AI Stock Scanner
2. AI Stock Analyzer
3. Trade Signals
4. Predictive Swing Scanner (only activates with a trained CatBoost model)
5. Market Intelligence

The dashboard uses Finnhub first and Yahoo Finance as a labelled fallback when enabled. Missing data is reported as unavailable; it is not replaced with fabricated sample prices.

## Deploy from GitHub to Render

1. Create a new **private** GitHub repository, for example `sst-elite-terminal`.
2. Extract this ZIP and upload the **contents** of the `SST_Elite_Terminal` folder to the repository root. The root should contain `app.py`, `requirements.txt`, `render.yaml`, `templates/` and `static/`.
3. On Render, choose **New + → Blueprint** and connect the GitHub repository. Render detects `render.yaml`.
4. When prompted, enter your Finnhub API key for `FINNHUB_API_KEY` in Render's environment settings. Do not put the real key in GitHub.
5. Deploy. Render installs the packages and starts FastAPI. Open the generated `https://...onrender.com` URL.
6. Check `https://YOUR-APP.onrender.com/api/health` to confirm the service is responding.

## Environment variables

| Variable | Required | Purpose |
|---|---:|---|
| `FINNHUB_API_KEY` | Recommended | Finnhub market-data access; configure as a Render secret/environment variable |
| `SST_CACHE_SECONDS` | No | Cache duration in seconds; default 180 |
| `SST_MAX_SCAN_TICKERS` | No | Maximum symbols per scan; default 25 |
| `SST_ALLOW_YFINANCE_FALLBACK` | No | Enables labelled Yahoo Finance fallback; default true |

A Finnhub API key can be obtained from https://finnhub.io/. Provider plan limits apply to quote and intraday candle endpoints.

## Run locally on Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your Finnhub key
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then visit http://127.0.0.1:8000.

## CatBoost model status

The CatBoost training source is included, but this package does not include a trained `.cbm` model or its feature-column file. Until both trained artifacts are installed, the Direction Score is explicitly a rule-based preview, not a machine-learning probability. The Predictive Scanner correctly reports that the model is not trained. Do not present raw, uncalibrated model outputs as win probabilities.

The `.gitignore` excludes `.cbm` and `.joblib` artifacts by default. If you later deploy a trained model, use private artifact storage or a secure deployment process rather than accidentally committing sensitive or large files to a public repository.

## Important production limitations

- Render's free service may sleep when idle and can take time to wake. It is suitable for testing, not a guaranteed always-on paid terminal.
- This MVP has no user login, membership billing, per-user rate limiting, persistent signal history, or push/email alerts. Do not expose it as a public paid multi-user service until those controls are implemented.
- Verify Finnhub and Yahoo Finance terms and redistribution rights before showing market data to customers. Data may be delayed or unavailable depending on provider plan.
- Trade levels are mechanical research calculations, not personalized investment advice or guaranteed outcomes.
- Before commercial deployment, add authentication, restricted access, rate limiting, monitoring, backups and a security review.

## Health endpoint

`GET /api/health` reports service status, whether the Finnhub key is configured, whether a CatBoost model is loaded, and the fallback/cache settings. Do not publish the actual API key.
