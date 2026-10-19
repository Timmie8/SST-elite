from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

load_dotenv()
BASE = Path(__file__).resolve().parent
API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
CACHE_SECONDS = int(os.getenv("SST_CACHE_SECONDS", "180"))
MAX_SCAN = int(os.getenv("SST_MAX_SCAN_TICKERS", "25"))
ALLOW_YF = os.getenv("SST_ALLOW_YFINANCE_FALLBACK", "true").lower() == "true"
MODEL_PATH = BASE / "sst_catboost_model.cbm"
FEATURE_PATH = BASE / "sst_feature_columns.joblib"

app = FastAPI(title="SST Elite AI Trading Terminal", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))
_CACHE: dict[str, tuple[float, Any]] = {}
_MODEL = None
_FEATURES = None


def cache_get(key: str):
    item = _CACHE.get(key)
    if item and time.time() - item[0] < CACHE_SECONDS:
        return item[1]
    return None


def cache_set(key: str, value: Any):
    _CACHE[key] = (time.time(), value)
    return value


def finnhub_get(path: str, params: dict[str, Any]) -> dict:
    if not API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is not configured")
    url = f"https://finnhub.io/api/v1/{path.lstrip('/')}"
    response = requests.get(url, params=params, headers={"X-Finnhub-Token": API_KEY}, timeout=18)
    if response.status_code == 429:
        raise RuntimeError("Finnhub rate limit reached (HTTP 429). Wait briefly and retry.")
    if response.status_code == 403:
        raise RuntimeError("Finnhub denied this endpoint for the current API plan (HTTP 403).")
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def normalise_yf(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns={c: str(c).title() for c in frame.columns})
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in frame.columns for c in needed):
        return pd.DataFrame(columns=needed)
    frame = frame[needed].copy().dropna()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch_candles(symbol: str, resolution: str = "D") -> tuple[pd.DataFrame, str]:
    symbol = symbol.upper().strip()
    cache_key = f"candles:{symbol}:{resolution}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached.copy(), "CACHE"

    periods = {"D": (420, "5y", "1d"), "60": (60, "3mo", "60m"), "15": (12, "10d", "15m")}
    days, yf_period, yf_interval = periods.get(resolution, (420, "5y", "1d"))
    finnhub_error = None
    if API_KEY:
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            payload = finnhub_get("stock/candle", {
                "symbol": symbol,
                "resolution": resolution,
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
            })
            if payload.get("s") == "ok" and payload.get("c"):
                frame = pd.DataFrame({
                    "Open": payload["o"], "High": payload["h"], "Low": payload["l"],
                    "Close": payload["c"], "Volume": payload["v"],
                }, index=pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert(None))
                frame = frame.dropna().sort_index()
                if not frame.empty:
                    cache_set(cache_key, frame)
                    return frame, "FINNHUB"
            finnhub_error = f"Finnhub returned no candles for {symbol} ({resolution})."
        except Exception as exc:
            finnhub_error = str(exc)
    else:
        finnhub_error = "Finnhub API key not configured."

    if ALLOW_YF:
        try:
            frame = normalise_yf(yf.download(symbol, period=yf_period, interval=yf_interval,
                                             auto_adjust=True, progress=False, threads=False))
            if not frame.empty:
                cache_set(cache_key, frame)
                return frame, "YAHOO FINANCE FALLBACK"
        except Exception as exc:
            finnhub_error = f"{finnhub_error} Yahoo fallback: {exc}"

    raise RuntimeError(f"Market data unavailable for {symbol} ({resolution}). {finnhub_error}")


def fetch_quote(symbol: str, daily: pd.DataFrame | None = None) -> tuple[dict, str]:
    cache_key = f"quote:{symbol}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached, "CACHE"
    if API_KEY:
        try:
            q = finnhub_get("quote", {"symbol": symbol})
            if float(q.get("c") or 0) > 0:
                result = {"price": float(q["c"]), "change_pct": float(q.get("dp") or 0),
                          "previous_close": float(q.get("pc") or 0), "timestamp": q.get("t")}
                cache_set(cache_key, result)
                return result, "FINNHUB"
        except Exception:
            pass
    if daily is None or daily.empty:
        daily, _ = fetch_candles(symbol, "D")
    last = daily.iloc[-1]
    previous = daily.iloc[-2] if len(daily) > 1 else last
    price = float(last["Close"])
    prev = float(previous["Close"])
    result = {"price": price, "change_pct": ((price / prev) - 1) * 100 if prev else 0,
              "previous_close": prev, "timestamp": int(pd.Timestamp(daily.index[-1]).timestamp())}
    cache_set(cache_key, result)
    return result, "CANDLE CLOSE FALLBACK"


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df.High.diff()
    down = -df.Low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = atr(df, period)
    pdi = 100 * plus.ewm(alpha=1/period, adjust=False).mean() / tr.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1/period, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x.Close
    x["EMA20"] = ema(c, 20); x["EMA50"] = ema(c, 50); x["EMA200"] = ema(c, 200)
    x["RSI14"] = rsi(c, 14); x["ATR14"] = atr(x, 14); x["ADX14"] = adx(x, 14)
    e12, e26 = ema(c, 12), ema(c, 26)
    x["MACD"] = e12-e26; x["MACD_SIGNAL"] = ema(x["MACD"], 9)
    x["ROC10"] = c.pct_change(10) * 100
    x["RVOL"] = x.Volume / x.Volume.rolling(20).mean().replace(0, np.nan)
    x["HIGH20"] = x.High.shift(1).rolling(20).max()
    x["LOW20"] = x.Low.shift(1).rolling(20).min()
    x["HIGH50"] = x.High.shift(1).rolling(50).max()
    return x


def score_direction(row: pd.Series, rel_strength: float | None) -> float:
    # Transparent fallback score only. It is NOT an ML probability.
    points = 0.0
    points += 12 if row.Close > row.EMA20 else 0
    points += 12 if row.EMA20 > row.EMA50 else 0
    points += 8 if row.EMA50 > row.EMA200 else 0
    points += 8 if row.EMA20 > row.EMA20_PREV else 0
    points += 10 if row.RSI14 > 50 else 0
    points += 8 if row.MACD > row.MACD_SIGNAL else 0
    points += 8 if row.ROC10 > 0 else 0
    points += 7 if row.ADX14 > 20 else 0
    points += 8 if row.RVOL > 1 else 0
    points += 5 if row.RVOL > 1.5 else 0
    points += 7 if rel_strength is not None and rel_strength > 0 else 0
    points += 7 if row.Close > row.HIGH20 else (3 if row.Close >= row.HIGH20 * 0.97 else 0)
    return float(np.clip(points, 0, 100))


def score_timing(daily: pd.DataFrame, hourly: pd.DataFrame, m15: pd.DataFrame) -> tuple[float, dict]:
    checks = {}
    score = 0.0
    d = daily.iloc[-1]
    h = build_indicators(hourly).iloc[-1] if len(hourly) >= 30 else None
    m = build_indicators(m15).iloc[-1] if len(m15) >= 30 else None
    checks["daily_trend"] = bool(d.Close > d.EMA20 and d.EMA20 > d.EMA50)
    if checks["daily_trend"]: score += 30
    checks["hourly_trend"] = bool(h is not None and h.Close > h.EMA20 and h.EMA20 > h.EMA50)
    if checks["hourly_trend"]: score += 25
    checks["intraday_momentum"] = bool(m is not None and m.RSI14 > 50 and m.MACD > m.MACD_SIGNAL)
    if checks["intraday_momentum"]: score += 20
    checks["volume_confirmation"] = bool(d.RVOL >= 1.2)
    if checks["volume_confirmation"]: score += 15
    checks["not_extended"] = bool(d.RSI14 <= 72 and d.Close <= d.EMA20 + 1.8*d.ATR14)
    if checks["not_extended"]: score += 10
    checks["intraday_data_available"] = h is not None and m is not None
    if not checks["intraday_data_available"]:
        score = min(score, 55)  # no full timing confirmation without intraday data
    return float(score), checks


def get_model():
    global _MODEL, _FEATURES
    if _MODEL is not None:
        return _MODEL, _FEATURES
    if not MODEL_PATH.exists():
        return None, None
    try:
        from catboost import CatBoostClassifier
        import joblib
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))
        features = joblib.load(FEATURE_PATH) if FEATURE_PATH.exists() else None
        _MODEL, _FEATURES = model, features
        return _MODEL, _FEATURES
    except Exception:
        return None, None


def model_direction_score(daily_raw: pd.DataFrame) -> tuple[float | None, str]:
    model, feature_columns = get_model()
    if model is None or not feature_columns:
        return None, "MODEL NOT TRAINED"
    try:
        # Keep feature definitions aligned with the existing SST CatBoost project.
        import sst_catboost_scanner as sst
        spy, _ = fetch_candles("SPY", "D")
        feat = sst.add_features(daily_raw, spy).replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
        if feat.empty:
            return None, "MODEL DATA INSUFFICIENT"
        row = feat.iloc[-1]
        x = pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)
        raw_score = float(model.predict_proba(x)[0, 1] * 100)
        return raw_score, "CATBOOST MODEL SCORE (UNCALIBRATED)"
    except Exception as exc:
        return None, f"MODEL UNAVAILABLE: {exc}"


def analyze_symbol(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    if not symbol or len(symbol) > 12:
        raise RuntimeError("Enter a valid ticker symbol.")
    daily, daily_source = fetch_candles(symbol, "D")
    if len(daily) < 210:
        raise RuntimeError(f"Insufficient daily history for {symbol}; at least 210 bars are required.")
    hourly, hourly_source = fetch_candles(symbol, "60")
    m15, m15_source = fetch_candles(symbol, "15")
    spy, spy_source = fetch_candles("SPY", "D")
    di = build_indicators(daily)
    di["EMA20_PREV"] = di.EMA20.shift(5)
    row = di.iloc[-1]
    spy_ret = float(spy.Close.pct_change(20).iloc[-1] * 100) if len(spy) > 21 else None
    stock_ret = float(daily.Close.pct_change(20).iloc[-1] * 100) if len(daily) > 21 else None
    rs = stock_ret - spy_ret if stock_ret is not None and spy_ret is not None else None
    rule_direction = score_direction(row, rs)
    model_score, model_status = model_direction_score(daily)
    direction = model_score if model_score is not None else rule_direction
    timing, timing_checks = score_timing(di, hourly, m15)
    quote, quote_source = fetch_quote(symbol, daily)
    price = quote["price"]
    atr_value = float(row.ATR14)
    support = float(daily.Low.tail(20).min())
    resistance = float(daily.High.tail(20).max())
    entry_low = price - 0.15 * atr_value
    entry_high = price + 0.15 * atr_value
    stop = min(price - 1.5 * atr_value, support - 0.15 * atr_value)
    risk = max(price - stop, 0.01)
    target1 = price + 1.5 * risk
    target2 = price + 2.5 * risk
    if direction >= 80 and timing >= 70:
        signal = "STRONG BUY"
    elif direction >= 65 and timing >= 60:
        signal = "BUY"
    elif direction >= 50:
        signal = "WATCH"
    else:
        signal = "AVOID"
    # Avoid presenting a setup as actionable if the data are incomplete.
    if not timing_checks["intraday_data_available"]:
        signal = "WATCH - INTRADAY DATA MISSING" if direction >= 60 else "DATA INCOMPLETE"
    sources = {"daily": daily_source, "hourly": hourly_source, "15m": m15_source,
               "spy": spy_source, "quote": quote_source}
    return {
        "ticker": symbol, "price": round(price, 4), "change_pct": round(quote["change_pct"], 2),
        "direction_score": round(direction, 1), "direction_score_type": model_status if model_score is not None else "RULE-BASED PREVIEW - CATBOOST NOT TRAINED",
        "rule_direction_score": round(rule_direction, 1), "timing_score": round(timing, 1),
        "signal": signal, "rsi": round(float(row.RSI14), 2), "ema20": round(float(row.EMA20), 4),
        "ema50": round(float(row.EMA50), 4), "ema200": round(float(row.EMA200), 4),
        "relative_volume": round(float(row.RVOL), 2) if pd.notna(row.RVOL) else None,
        "roc10": round(float(row.ROC10), 2), "adx": round(float(row.ADX14), 2),
        "atr": round(atr_value, 4), "relative_strength_vs_spy_20d": round(rs, 2) if rs is not None else None,
        "support": round(support, 4), "resistance": round(resistance, 4),
        "breakout20": bool(price > float(row.HIGH20)) if pd.notna(row.HIGH20) else False,
        "mtf": {"daily": "BULLISH" if timing_checks["daily_trend"] else "NOT CONFIRMED",
                "hourly": "BULLISH" if timing_checks["hourly_trend"] else "NOT CONFIRMED",
                "15m": "BULLISH" if timing_checks["intraday_momentum"] else "NOT CONFIRMED"},
        "timing_checks": timing_checks,
        "trade_setup": {"entry_low": round(entry_low, 2), "entry_high": round(entry_high, 2),
                        "stop_loss": round(stop, 2), "target1": round(target1, 2),
                        "target2": round(target2, 2), "risk_reward_t1": round((target1-price)/risk, 2),
                        "holding_period": "3–5 trading days", "status": "ILLUSTRATIVE SETUP - VERIFY BEFORE USE"},
        "data_sources": sources, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "model_status": model_status,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    model, _ = get_model()
    return {"status": "ok", "finnhub_key_configured": bool(API_KEY), "catboost_model_loaded": model is not None,
            "yfinance_fallback_enabled": ALLOW_YF, "cache_seconds": CACHE_SECONDS,
            "max_scan_tickers": MAX_SCAN, "server_time_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/api/analyze/{symbol}")
def analyze(symbol: str):
    try:
        return analyze_symbol(symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/scan")
def scan(symbols: str = Query(..., description="Comma-separated US stock tickers")):
    tickers = list(dict.fromkeys(x.strip().upper() for x in symbols.split(",") if x.strip()))
    if not tickers:
        raise HTTPException(status_code=400, detail="Enter at least one ticker.")
    if len(tickers) > MAX_SCAN:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_SCAN} tickers per scan. Split into smaller scans.")
    results, errors = [], []
    for symbol in tickers:
        try:
            results.append(analyze_symbol(symbol))
        except Exception as exc:
            errors.append({"ticker": symbol, "error": str(exc)})
    results.sort(key=lambda x: (x["direction_score"], x["timing_score"]), reverse=True)
    return {"results": results, "errors": errors, "scanned": len(results), "requested": len(tickers),
            "updated_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/api/market")
def market():
    out = {"updated_utc": datetime.now(timezone.utc).isoformat(), "source": "Finnhub primary / Yahoo fallback"}
    for ticker, key in [("SPY", "spy"), ("^VIX", "vix"), ("QQQ", "qqq"), ("IWM", "iwm"),
                        ("XLK", "technology"), ("XLF", "financials"), ("XLI", "industrials"),
                        ("XLE", "energy"), ("XLV", "healthcare"), ("XLY", "consumer_discretionary")]:
        try:
            frame, source = fetch_candles(ticker, "D")
            q, qsource = fetch_quote(ticker, frame)
            out[key] = {"price": round(q["price"], 2), "change_pct": round(q["change_pct"], 2),
                        "source": qsource if qsource != "CANDLE CLOSE FALLBACK" else source}
        except Exception as exc:
            out[key] = {"error": str(exc), "status": "DATA UNAVAILABLE"}
    out["market_bias"] = "BULLISH" if out.get("spy", {}).get("change_pct", 0) > 0 else "CAUTIOUS / MIXED"
    return out


@app.get("/api/predictive")
def predictive(symbols: str = "AAPL,MSFT,NVDA,AMD,PLTR,AVGO,TSLA,AMZN,META,CRDO"):
    model, _ = get_model()
    if model is None:
        return {"status": "MODEL NOT TRAINED", "message": "The existing CatBoost model has not yet been trained/installed. Scanner scores are rule-based previews, not ML probabilities.", "results": []}
    return scan(symbols)
