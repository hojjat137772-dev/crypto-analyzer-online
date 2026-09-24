from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import requests
import pandas as pd

app = FastAPI(title="Crypto Analyzer Online")


BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_candles(symbol="BTCUSDT", interval="1h", limit=200):
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }

    r = requests.get(BINANCE_URL, params=params, timeout=15)

    if r.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail="ارز یا تایم‌فریم معتبر نیست."
        )

    data = r.json()

    if not isinstance(data, list) or len(data) < 50:
        raise HTTPException(
            status_code=400,
            detail="داده کافی برای تحلیل وجود ندارد."
        )

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close",
        "volume", "close_time", "qav",
        "trades", "tb_base", "tb_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df


def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)

    return 100 - (100 / (1 + rs))


def analyze_market(symbol, interval):
    df = get_candles(symbol, interval)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    df["rsi"] = calculate_rsi(close)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    last = df.iloc[-1]

    price = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"])
    macd = float(last["macd"])
    macd_signal = float(last["signal"])
    atr = float(last["atr"])

    support = float(df["low"].tail(50).min())
    resistance = float(df["high"].tail(50).max())

    bullish = 0
    bearish = 0

    # Trend
    if price > ema20:
        bullish += 1
    else:
        bearish += 1

    if ema20 > ema50:
        bullish += 1
    else:
        bearish += 1

    if price > ema200:
        bullish += 1
    else:
        bearish += 1

    # RSI
    if 50 <= rsi <= 70:
        bullish += 1
    elif 30 <= rsi < 50:
        bearish += 1

    # MACD
    if macd > macd_signal:
        bullish += 1
    else:
        bearish += 1

    # Signal
    if bullish >= 4 and bullish > bearish:
        signal = "LONG"
        entry = price
        stop_loss = price - (atr * 1.5)
        tp1 = price + (atr * 2)
        tp2 = price + (atr * 3)
        tp3 = price + (atr * 4)

    elif bearish >= 4 and bearish > bullish:
        signal = "SHORT"
        entry = price
        stop_loss = price + (atr * 1.5)
        tp1 = price - (atr * 2)
        tp2 = price - (atr * 3)
        tp3 = price - (atr * 4)

    else:
        signal = "NO TRADE"
        entry = price
        stop_loss = None
        tp1 = None
        tp2 = None
        tp3 = None

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "price": round(price, 8),

        "signal": signal,

        "entry": round(entry, 8),

        "stop_loss": (
            round(stop_loss, 8)
            if stop_loss is not None else None
        ),

        "take_profit_1": (
            round(tp1, 8)
            if tp1 is not None else None
        ),

        "take_profit_2": (
            round(tp2, 8)
            if tp2 is not None else None
        ),

        "take_profit_3": (
            round(tp3, 8)
            if tp3 is not None else None
        ),

        "support": round(support, 8),
        "resistance": round(resistance, 8),

        "indicators": {
            "ema20": round(ema20, 8),
            "ema50": round(ema50, 8),
            "ema200": round(ema200, 8),
            "rsi": round(rsi, 2),
            "macd": round(macd, 8),
            "macd_signal": round(macd_signal, 8),
            "atr": round(atr, 8)
        },

        "bullish_score": bullish,
        "bearish_score": bearish,

        "note": "این خروجی تحلیل تکنیکال است و تضمین سود یا توصیه قطعی معامله نیست."
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <title>Crypto Analyzer</title>
        <style>
            body {
                font-family: Arial;
                background:#0b1220;
                color:white;
                padding:25px;
            }

            .box {
                max-width:700px;
                margin:auto;
                background:#151f33;
                padding:25px;
                border-radius:18px;
            }

            input, select, button {
                width:100%;
                padding:14px;
                margin:8px 0;
                border-radius:10px;
                border:0;
                box-sizing:border-box;
            }

            button {
                background:#16a34a;
                color:white;
                font-size:17px;
                font-weight:bold;
            }

            pre {
                white-space:pre-wrap;
                background:#08101d;
                padding:15px;
                border-radius:10px;
            }
        </style>
    </head>

    <body>

    <div class="box">

        <h1>📊 Crypto Analyzer Online</h1>

        <p>تحلیل آنلاین بازار ارز دیجیتال</p>

        <input id="symbol"
               value="BTCUSDT"
               placeholder="مثلاً BTCUSDT">

        <select id="interval">
            <option value="15m">15 دقیقه</option>
            <option value="1h" selected>1 ساعت</option>
            <option value="4h">4 ساعت</option>
            <option value="1d">1 روز</option>
        </select>

        <button onclick="analyze()">
            🔍 تحلیل بازار
        </button>

        <pre id="result">آماده تحلیل...</pre>

    </div>

    <script>

    async function analyze() {

        const symbol =
            document.getElementById("symbol").value;

        const interval =
            document.getElementById("interval").value;

        document.getElementById("result").textContent =
            "در حال دریافت اطلاعات بازار...";

        try {

            const response =
                await fetch(
                    `/analyze?symbol=${symbol}&interval=${interval}`
                );

            const data = await response.json();

            document.getElementById("result").textContent =
                JSON.stringify(data, null, 2);

        } catch(error) {

            document.getElementById("result").textContent =
                "خطا در دریافت اطلاعات بازار";

        }
    }

    </script>

    </body>
    </html>
    """


@app.get("/analyze")
def analyze(symbol: str = "BTCUSDT", interval: str = "1h"):
    return analyze_market(symbol, interval)


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "crypto-analyzer"
    }
