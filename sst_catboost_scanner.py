"""
SST Predictive CatBoost Stock Scanner
-------------------------------------
Purpose:
- Download US stock data with yfinance
- Build technical / momentum / volume features
- Train a CatBoost binary classifier
- Predict probability of a +5% move within the next 5 trading days
- Rank current stocks using CatBoost probability + SST technical score
- Save model and scanner results

Install:
    pip install yfinance pandas numpy scikit-learn catboost joblib

Run:
    python sst_catboost_scanner.py

Optional:
    python sst_catboost_scanner.py --tickers AAPL,NVMI,NVDA,CRDO,PLTR
    python sst_catboost_scanner.py --years 5 --top 30
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO",
    "TSLA", "AMD", "PLTR", "CRDO", "IONQ", "GLXY", "ORCL",
    "NVMI", "STX", "ONTO", "SHOP", "HUBS", "ACVA", "FUBO",
    "OOMA", "BB", "VRNS", "MNDY", "ATRC", "SONO", "NX",
    "PBF", "DAR", "COIN", "HOOD", "SNOW", "QQQ", "SPY"
]

MODEL_FILE = "sst_catboost_model.cbm"
FEATURE_FILE = "sst_feature_columns.joblib"
RESULT_FILE = "sst_scanner_results.csv"

HORIZON_DAYS = 5
TARGET_RETURN = 0.05


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_ticker(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # yfinance can return MultiIndex columns for a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return pd.DataFrame()

        df = df[required].copy()
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        return df

    except Exception as exc:
        print(f"[WARN] {ticker}: download failed: {exc}")
        return pd.DataFrame()


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    lowest = df["Low"].rolling(period).min()
    highest = df["High"].rolling(period).max()

    denominator = (highest - lowest).replace(0, np.nan)
    return 100 * (df["Close"] - lowest) / denominator


def macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0,
    )
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0,
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_value = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr_value.replace(0, np.nan)
    )

    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr_value.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(alpha=1 / period, adjust=False).mean()


def add_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create all model features using information available on the same day."""

    x = df.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    # Moving averages
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()
    x["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # Trend relationships
    x["Price_vs_EMA20"] = close / x["EMA20"] - 1
    x["Price_vs_EMA50"] = close / x["EMA50"] - 1
    x["Price_vs_EMA200"] = close / x["EMA200"] - 1

    x["EMA20_vs_EMA50"] = x["EMA20"] / x["EMA50"] - 1
    x["EMA50_vs_EMA200"] = x["EMA50"] / x["EMA200"] - 1

    x["EMA20_slope_5"] = x["EMA20"] / x["EMA20"].shift(5) - 1
    x["EMA50_slope_10"] = x["EMA50"] / x["EMA50"].shift(10) - 1

    # Momentum
    x["RSI14"] = rsi(close, 14)

    x["ROC5"] = close.pct_change(5)
    x["ROC10"] = close.pct_change(10)
    x["ROC20"] = close.pct_change(20)

    x["ATR14"] = atr(x, 14)
    x["ATR_pct"] = x["ATR14"] / close

    # MACD
    x["MACD"], x["MACD_signal"], x["MACD_hist"] = macd(close)

    # Stochastic
    x["Stoch14"] = stochastic(x, 14)

    # ADX
    x["ADX14"] = adx(x, 14)

    # Volume
    x["Volume_SMA20"] = volume.rolling(20).mean()
    x["Relative_Volume"] = volume / x["Volume_SMA20"].replace(0, np.nan)

    x["Volume_change_5"] = volume / volume.shift(5) - 1

    # Daily price action
    x["Daily_Return"] = close.pct_change()
    x["Range_pct"] = (high - low) / close
    x["Close_Position"] = (close - low) / (high - low).replace(0, np.nan)

    # Breakout features
    x["High20"] = high.shift(1).rolling(20).max()
    x["High50"] = high.shift(1).rolling(50).max()
    x["Low20"] = low.shift(1).rolling(20).min()

    x["Breakout20"] = (close > x["High20"]).astype(int)
    x["Breakout50"] = (close > x["High50"]).astype(int)

    x["Distance_High20"] = close / x["High20"] - 1
    x["Distance_Low20"] = close / x["Low20"] - 1

    # Relative strength versus SPY
    if spy_df is not None and not spy_df.empty:
        spy_close = spy_df["Close"].reindex(x.index).ffill()

        x["SPY_Return20"] = spy_close.pct_change(20)
        x["Stock_Return20"] = close.pct_change(20)
        x["Relative_Strength20"] = (
            x["Stock_Return20"] - x["SPY_Return20"]
        )

    else:
        x["SPY_Return20"] = 0.0
        x["Stock_Return20"] = close.pct_change(20)
        x["Relative_Strength20"] = 0.0

    return x


# ============================================================
# SST TECHNICAL SCORE
# ============================================================

def calculate_sst_score(row: pd.Series) -> float:
    """
    Rule-based SST score, 0-100.

    This is intentionally separate from CatBoost.
    CatBoost learns the historical probability; the SST score
    provides an interpretable technical confirmation layer.
    """

    score = 0.0

    # 1. Trend: 30 points
    if row["Close"] > row["EMA20"]:
        score += 7.5
    if row["EMA20"] > row["EMA50"]:
        score += 7.5
    if row["EMA50"] > row["EMA200"]:
        score += 7.5
    if row["EMA20_slope_5"] > 0:
        score += 7.5

    # 2. Momentum: 25 points
    if row["RSI14"] > 50:
        score += 8.0
    if row["MACD"] > row["MACD_signal"]:
        score += 8.0
    if row["ROC10"] > 0:
        score += 4.5
    if row["ADX14"] > 20:
        score += 4.5

    # 3. Volume: 15 points
    if row["Relative_Volume"] > 1.0:
        score += 5
    if row["Relative_Volume"] > 1.5:
        score += 5
    if row["Volume_change_5"] > 0:
        score += 5

    # 4. Breakout / relative strength: 20 points
    if row["Breakout20"] == 1:
        score += 8
    elif row["Distance_High20"] > -0.03:
        score += 4

    if row["Relative_Strength20"] > 0:
        score += 6
    if row["Close_Position"] > 0.65:
        score += 6

    # 5. RSI quality: 10 points
    if 50 <= row["RSI14"] <= 70:
        score += 5
    if 45 <= row["Stoch14"] <= 85:
        score += 5

    return float(np.clip(score, 0, 100))


# ============================================================
# DATASET CREATION
# ============================================================

def prepare_dataset(
    tickers: list[str],
    period: str = "5y",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Download all tickers, calculate features and create the target.

    Target:
        1 if future 5-day maximum close return >= +5%
        0 otherwise.

    The target is shifted backward and is never used as a feature.
    """

    print("\nDownloading SPY...")
    spy = download_ticker("SPY", period)

    all_rows = []
    raw_data = {}

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i:>3}/{len(tickers)}] {ticker}")

        raw = download_ticker(ticker, period)

        if raw.empty or len(raw) < 300:
            print(f"       skipped: insufficient data")
            continue

        features = add_features(raw, spy)

        # Future maximum close over the next HORIZON_DAYS.
        future_max = pd.concat(
            [
                features["Close"].shift(-i)
                for i in range(1, HORIZON_DAYS + 1)
            ],
            axis=1,
        ).max(axis=1)

        features["Future_Max_Return"] = (
            future_max / features["Close"] - 1
        )

        features["Target"] = (
            features["Future_Max_Return"] >= TARGET_RETURN
        ).astype(int)

        features["Ticker"] = ticker

        # Current data is retained separately for scanning.
        raw_data[ticker] = features.copy()

        all_rows.append(features)

    if not all_rows:
        raise RuntimeError("No usable ticker data was downloaded.")

    dataset = pd.concat(all_rows, axis=0)
    dataset = dataset.sort_index()

    return dataset, raw_data


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Price_vs_EMA20",
    "Price_vs_EMA50",
    "Price_vs_EMA200",
    "EMA20_vs_EMA50",
    "EMA50_vs_EMA200",
    "EMA20_slope_5",
    "EMA50_slope_10",
    "RSI14",
    "ROC5",
    "ROC10",
    "ROC20",
    "ATR_pct",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "Stoch14",
    "ADX14",
    "Relative_Volume",
    "Volume_change_5",
    "Daily_Return",
    "Range_pct",
    "Close_Position",
    "Distance_High20",
    "Distance_Low20",
    "Breakout20",
    "Breakout50",
    "SPY_Return20",
    "Stock_Return20",
    "Relative_Strength20",
]


# ============================================================
# TIME-SERIES SPLIT
# ============================================================

def time_split(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    """
    Chronological split:
        oldest 70%  -> train
        next 15%    -> validation
        newest 15%  -> test
    """

    dates = np.sort(dataset.index.unique())

    n = len(dates)

    train_end = int(n * train_fraction)
    validation_end = int(
        n * (train_fraction + validation_fraction)
    )

    # Purge the label horizon at each boundary. The target looks forward
    # HORIZON_DAYS sessions, so observations immediately before a split
    # must not carry labels whose outcome window crosses into the next split.
    purge = HORIZON_DAYS
    train_dates = dates[:max(0, train_end - purge)]
    validation_start = min(n, train_end + purge)
    validation_end_purged = max(validation_start, validation_end - purge)
    validation_dates = dates[validation_start:validation_end_purged]
    test_start = min(n, validation_end + purge)
    test_dates = dates[test_start:]

    train = dataset.loc[dataset.index.isin(train_dates)].copy()
    validation = dataset.loc[dataset.index.isin(validation_dates)].copy()
    test = dataset.loc[dataset.index.isin(test_dates)].copy()

    return train, validation, test


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(dataset: pd.DataFrame):
    print("\nPreparing model dataset...")

    model_data = dataset.dropna(
        subset=FEATURE_COLUMNS + ["Target"]
    ).copy()

    train, validation, test = time_split(model_data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["Target"]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["Target"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["Target"]

    print(f"Train rows      : {len(train):,}")
    print(f"Validation rows : {len(validation):,}")
    print(f"Test rows       : {len(test):,}")

    print(
        f"Positive target rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    model = CatBoostClassifier(
        iterations=1200,
        depth=7,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        l2_leaf_reg=6,
        random_strength=1.0,
        border_count=128,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
        auto_class_weights="Balanced",
    )

    print("\nTraining CatBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=(X_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    # Test probabilities.
    test_probability = model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= 0.50).astype(int)

    print("\n================ MODEL TEST ================")

    try:
        auc = roc_auc_score(y_test, test_probability)
        print(f"ROC-AUC   : {auc:.4f}")
    except ValueError:
        print("ROC-AUC   : unavailable")

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_test, test_prediction):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_test, test_prediction, zero_division=0):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_test, test_prediction, zero_division=0):.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_prediction))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_prediction,
            digits=4,
            zero_division=0,
        )
    )

    print("\nTop feature importance:")
    importance = pd.Series(
        model.get_feature_importance(),
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    print(importance.head(15).to_string())

    model.save_model(MODEL_FILE)
    joblib.dump(FEATURE_COLUMNS, FEATURE_FILE)

    print(f"\nModel saved: {MODEL_FILE}")

    return model


# ============================================================
# CURRENT SCANNER
# ============================================================

def scan_current(
    model: CatBoostClassifier,
    raw_data: dict[str, pd.DataFrame],
    top_n: int = 20,
) -> pd.DataFrame:

    results = []

    for ticker, df in raw_data.items():

        clean = df.dropna(subset=FEATURE_COLUMNS)

        if clean.empty:
            continue

        row = clean.iloc[-1]

        X = pd.DataFrame(
            [row[FEATURE_COLUMNS].values],
            columns=FEATURE_COLUMNS,
        )

        probability = float(
            model.predict_proba(X)[0, 1]
        )

        prediction = int(probability >= 0.50)

        sst_score = calculate_sst_score(row)

        # Combined score.
        # CatBoost = 65%
        # SST technical confirmation = 35%
        combined_score = (
            probability * 100 * 0.65
            + sst_score * 0.35
        )

        if combined_score >= 85 and probability >= 0.75:
            signal = "STRONG LONG"
        elif combined_score >= 75 and probability >= 0.65:
            signal = "LONG"
        elif combined_score >= 60 and probability >= 0.55:
            signal = "WATCH"
        elif combined_score >= 45:
            signal = "NEUTRAL"
        else:
            signal = "AVOID"

        results.append(
            {
                "Ticker": ticker,
                "Date": clean.index[-1].strftime("%Y-%m-%d"),
                "Close": float(row["Close"]),
                "CatBoost_Probability": probability * 100,
                "SST_Technical_Score": sst_score,
                "SST_AI_Score": combined_score,
                "Signal": signal,
                "RSI": float(row["RSI14"]),
                "Relative_Volume": float(row["Relative_Volume"]),
                "ROC10": float(row["ROC10"] * 100),
                "ADX": float(row["ADX14"]),
                "EMA20_vs_EMA50": float(
                    row["EMA20_vs_EMA50"] * 100
                ),
                "Relative_Strength20": float(
                    row["Relative_Strength20"] * 100
                ),
                "Breakout20": int(row["Breakout20"]),
            }
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    results_df = results_df.sort_values(
        "SST_AI_Score",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(RESULT_FILE, index=False)

    print("\n================ SST AI SCANNER ================")

    display_columns = [
        "Ticker",
        "Close",
        "CatBoost_Probability",
        "SST_Technical_Score",
        "SST_AI_Score",
        "Signal",
        "RSI",
        "Relative_Volume",
        "ROC10",
        "Relative_Strength20",
    ]

    print(
        results_df[display_columns]
        .head(top_n)
        .to_string(
            index=False,
            formatters={
                "Close": "{:.2f}".format,
                "CatBoost_Probability": "{:.1f}".format,
                "SST_Technical_Score": "{:.1f}".format,
                "SST_AI_Score": "{:.1f}".format,
                "RSI": "{:.1f}".format,
                "Relative_Volume": "{:.2f}".format,
                "ROC10": "{:.2f}".format,
                "Relative_Strength20": "{:.2f}".format,
            },
        )
    )

    print(f"\nResults saved: {RESULT_FILE}")

    return results_df


# ============================================================
# OPTIONAL CSV EXPORT OF TRAINING DATA
# ============================================================

def save_training_dataset(dataset: pd.DataFrame):
    path = "sst_training_dataset.csv"
    dataset.to_csv(path)
    print(f"Training dataset saved: {path}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SST Predictive CatBoost Stock Scanner"
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker list",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of historical data",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top stocks to display",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force model retraining",
    )

    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save complete training dataset",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    tickers = [
        t.strip().upper()
        for t in args.tickers.split(",")
        if t.strip()
    ]

    period = f"{args.years}y"

    print("==============================================")
    print("      SST PREDICTIVE CATBOOST SCANNER")
    print("==============================================")
    print(f"Tickers       : {len(tickers)}")
    print(f"History       : {period}")
    print(f"Target        : +{TARGET_RETURN * 100:.1f}%")
    print(f"Horizon       : {HORIZON_DAYS} trading days")
    print("==============================================")

    # Download / create features.
    dataset, raw_data = prepare_dataset(
        tickers,
        period=period,
    )

    if args.save_data:
        save_training_dataset(dataset)

    # Train a new model.
    model = train_model(dataset)

    # Current scan.
    results = scan_current(
        model,
        raw_data,
        top_n=args.top,
    )

    if results.empty:
        print("No current results.")
        return

    print("\n================ SUMMARY ================")

    strong = results[
        results["Signal"] == "STRONG LONG"
    ]

    long = results[
        results["Signal"] == "LONG"
    ]

    print(f"Strong Long : {len(strong)}")
    print(f"Long        : {len(long)}")
    print(f"Total       : {len(results)}")

    print("\nTop 10 SST candidates:")
    print(
        results[
            [
                "Ticker",
                "SST_AI_Score",
                "CatBoost_Probability",
                "SST_Technical_Score",
                "Signal",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
