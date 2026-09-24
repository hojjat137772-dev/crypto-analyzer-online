from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
import math

app = FastAPI(title="Crypto Analyzer Online API")

BINANCE = "https://api.binance.com/api/v3/klines"


def candles(symbol, interval, limit=250):
    r = requests.get(
        BINANCE,
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        },
        timeout=12
    )
    r.raise_for_status()

    rows = r.json()

    if not rows:
        raise ValueError("No market data")

    df = pd.DataFrame(
        rows,
        columns=[
            "t", "o", "h", "l", "c", "v",
            "x1", "q", "n", "tb", "tq", "x2"
        ]
    )

    for c in ["o", "h", "l", "c", "v"]:
        df[c] = pd.to_numeric(df[c])

    return df


def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, math.nan)

    return (
        100 - (100 / (1 + rs))
    ).fillna(50)


def atr(df, period=14):
    previous_close = df["c"].shift()

    tr = pd.concat(
        [
            df["h"] - df["l"],
            (df["h"] - previous_close).abs(),
            (df["l"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def macd(series):
    return ema(series, 12) - ema(series, 26)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Crypto Analyzer"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/analyze")
def analyze(
    symbol: str = "BTCUSDT",
    interval: str = "1h"
):

    try:
        df = candles(symbol, interval)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Market data error: {e}"
        )

    close = df["c"]

    price = float(close.iloc[-1])

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    rsi_value = float(rsi(close).iloc[-1])
    macd_value = float(macd(close).iloc[-1])
    atr_value = float(atr(df).iloc[-1])

    support = float(
        df["l"].tail(50).min()
    )

    resistance = float(
        df["h"].tail(50).max()
    )

    bullish = (
        price > ema20.iloc[-1]
        and ema20.iloc[-1] > ema50.iloc[-1]
    )

    bearish = (
        price < ema20.iloc[-1]
        and ema20.iloc[-1] < ema50.iloc[-1]
    )

    score = 50

    if bullish:
        score += 15

    if bearish:
        score -= 15

    if rsi_value > 55:
        score += 10

    elif rsi_value < 45:
        score -= 10

    if macd_value > 0:
        score += 10

    else:
        score -= 10

    score = max(
        0,
        min(100, score)
    )

    if (
        bullish
        and rsi_value >= 50
        and macd_value > 0
    ):
        signal = "LONG"

    elif (
        bearish
        and rsi_value <= 50
        and macd_value < 0
    ):
        signal = "SHORT"

    else:
        signal = "NO TRADE"

    if signal == "LONG":

        entry = price

        stop_loss = max(
            support,
            price - 1.5 * atr_value
        )

        risk = max(
            price - stop_loss,
            atr_value * 0.5
        )

        tp1 = price + 1.5 * risk
        tp2 = price + 2.5 * risk
        tp3 = price + 4 * risk

    elif signal == "SHORT":

        entry = price

        stop_loss = min(
            resistance,
            price + 1.5 * atr_value
        )

        risk = max(
            stop_loss - price,
            atr_value * 0.5
        )

        tp1 = price - 1.5 * risk
        tp2 = price - 2.5 * risk
        tp3 = price - 4 * risk

    else:

        entry = price
        stop_loss = price
        tp1 = price
        tp2 = price
        tp3 = price

    if bullish:
        trend = "صعودی"

    elif bearish:
        trend = "نزولی"

    else:
        trend = "خنثی"

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "signal": signal,
        "score": round(score, 1),

        "price": price,
        "entry": round(entry, 8),

        "stop_loss": round(
            stop_loss,
            8
        ),

        "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),
        "tp3": round(tp3, 8),

        "support": round(
            support,
            8
        ),

        "resistance": round(
            resistance,
            8
        ),

        "rsi": round(
            rsi_value,
            2
        ),

        "macd": round(
            macd_value,
            8
        ),

        "atr": round(
            atr_value,
            8
        ),

        "trend": trend,

        "note":
            "این خروجی تحلیل احتمالاتی بازار است و "
            "تضمین سود یا توصیه مالی نیست."
    }
