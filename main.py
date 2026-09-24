from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import math
import os
import time

app = FastAPI(
    title="Crypto Analyzer Online",
    version="3.0.0"
)

# =========================================================
# CoinGecko
# =========================================================

COINGECKO_URL = "https://api.coingecko.com/api/v3"

API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()

session = requests.Session()

session.headers.update({
    "User-Agent": "CryptoAnalyzerOnline/3.0",
    "Accept": "application/json"
})

if API_KEY:
    session.headers.update({
        "x-cg-demo-api-key": API_KEY
    })


# =========================================================
# Timeframes
# =========================================================

INTERVALS = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",

    "1 ساعت": "1h",
    "4 ساعت": "4h",
    "1 روز": "1d",
    "1 هفته": "1w"
}


# =========================================================
# Symbol helpers
# =========================================================

def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()

    s = (
        s.replace("/", "")
         .replace("-", "")
         .replace("_", "")
         .replace(" ", "")
    )

    return s


def coin_symbol(symbol: str) -> str:
    """
    BTCUSDT -> BTC
    ETHUSDT -> ETH
    BTCUSD  -> BTC
    """

    s = normalize_symbol(symbol)

    suffixes = [
        "USDT",
        "USDC",
        "USD",
        "BUSD",
        "EUR",
        "GBP",
        "BTC",
        "ETH"
    ]

    for suffix in suffixes:
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[:-len(suffix)]

    return s


def normalize_interval(interval: str) -> str:
    value = str(interval).strip().lower()

    return INTERVALS.get(value, value)


# =========================================================
# CoinGecko search
# =========================================================

_coin_cache = {}
_coin_cache_time = {}

CACHE_SECONDS = 300


def find_coin_id(symbol: str) -> str:

    clean_symbol = coin_symbol(symbol)

    now = time.time()

    cache_key = clean_symbol

    if (
        cache_key in _coin_cache
        and now - _coin_cache_time.get(cache_key, 0) < CACHE_SECONDS
    ):
        return _coin_cache[cache_key]

    try:
        r = session.get(
            f"{COINGECKO_URL}/search",
            params={
                "query": clean_symbol
            },
            timeout=15
        )
    except requests.RequestException:
        raise HTTPException(
            502,
            "ارتباط با منبع داده CoinGecko برقرار نشد."
        )

    if r.status_code != 200:
        raise HTTPException(
            502,
            f"CoinGecko خطا داد. کد: {r.status_code}"
        )

    data = r.json()

    coins = data.get("coins", [])

    if not coins:
        raise HTTPException(
            404,
            f"ارز {symbol} در CoinGecko پیدا نشد."
        )

    # اول دنبال symbol دقیق می‌گردیم
    exact = [
        c for c in coins
        if str(c.get("symbol", "")).upper() == clean_symbol.upper()
    ]

    if exact:
        # معمولاً اولین نتیجه محبوب‌ترین/مرتبط‌ترین است
        selected = exact[0]
    else:
        selected = coins[0]

    coin_id = selected.get("id")

    if not coin_id:
        raise HTTPException(
            404,
            "شناسه ارز پیدا نشد."
        )

    _coin_cache[cache_key] = coin_id
    _coin_cache_time[cache_key] = now

    return coin_id


# =========================================================
# Market data
# =========================================================

def get_market_prices(symbol="BTCUSDT", interval="1h"):

    interval = normalize_interval(interval)

    if interval not in {"1h", "4h", "1d", "1w"}:
        raise HTTPException(
            400,
            "تایم‌فریم معتبر نیست. از 1 ساعت، 4 ساعت، 1 روز یا 1 هفته استفاده کنید."
        )

    coin_id = find_coin_id(symbol)

    # تعداد روزهای مورد نیاز
    if interval == "1h":
        days = 30

    elif interval == "4h":
        days = 90

    elif interval == "1d":
        days = 365

    else:
        days = 1825

    try:
        r = session.get(
            f"{COINGECKO_URL}/coins/{coin_id}/market_chart",
            params={
                "vs_currency": "usd",
                "days": days,
                "precision": "full"
            },
            timeout=20
        )
    except requests.RequestException:
        raise HTTPException(
            502,
            "ارتباط با CoinGecko برقرار نشد."
        )

    if r.status_code != 200:

        try:
            error_data = r.json()
            message = error_data.get(
                "status",
                {}
            ).get(
                "error_message",
                "خطای دریافت داده از CoinGecko"
            )
        except Exception:
            message = "خطای دریافت داده از CoinGecko"

        raise HTTPException(
            502,
            message
        )

    data = r.json()

    prices = data.get("prices", [])

    if not prices:
        raise HTTPException(
            400,
            "داده قیمتی کافی برای این ارز وجود ندارد."
        )

    df = pd.DataFrame(
        prices,
        columns=["timestamp", "price"]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df.dropna()

    if len(df) < 60:
        raise HTTPException(
            400,
            "داده کافی برای تحلیل این ارز وجود ندارد."
        )

    df = df.set_index("timestamp")

    # -----------------------------------------------------
    # ساخت OHLC از نقاط قیمت
    # -----------------------------------------------------

    if interval == "1h":
        rule = "1h"

    elif interval == "4h":
        rule = "4h"

    elif interval == "1d":
        rule = "1D"

    else:
        rule = "7D"

    candles = df["price"].resample(rule).agg(
        ["first", "max", "min", "last"]
    )

    candles.columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    candles = candles.dropna()

    candles["volume"] = 0.0

    candles = candles.reset_index()

    if len(candles) < 60:
        raise HTTPException(
            400,
            "برای این تایم‌فریم کندل کافی وجود ندارد."
        )

    return candles, coin_id


# =========================================================
# Indicators
# =========================================================

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
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        float("nan")
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(50)


def atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def add_indicators(df):

    d = df.copy()

    d["ema20"] = ema(
        d["close"],
        20
    )

    d["ema50"] = ema(
        d["close"],
        50
    )

    d["ema100"] = ema(
        d["close"],
        100
    )

    d["ema200"] = ema(
        d["close"],
        200
    )

    d["rsi"] = rsi(
        d["close"],
        14
    )

    macd_fast = ema(
        d["close"],
        12
    )

    macd_slow = ema(
        d["close"],
        26
    )

    d["macd"] = (
        macd_fast - macd_slow
    )

    d["macd_signal"] = ema(
        d["macd"],
        9
    )

    d["macd_hist"] = (
        d["macd"] -
        d["macd_signal"]
    )

    d["atr"] = atr(
        d,
        14
    )

    return d.dropna().reset_index(
        drop=True
    )


# =========================================================
# Analysis
# =========================================================

def analyze(df):

    d = add_indicators(df)

    if len(d) < 20:
        raise HTTPException(
            400,
            "داده کافی برای تحلیل وجود ندارد."
        )

    x = d.iloc[-1]

    price = float(x["close"])
    current_atr = float(x["atr"])
    current_rsi = float(x["rsi"])

    score = 0
    reasons = []

    # -----------------------------------------------------
    # Trend
    # -----------------------------------------------------

    if price > x["ema20"]:
        score += 1
        reasons.append(
            "قیمت بالای EMA20 است."
        )
    else:
        score -= 1
        reasons.append(
            "قیمت زیر EMA20 است."
        )

    if x["ema20"] > x["ema50"]:
        score += 1
        reasons.append(
            "EMA20 بالای EMA50 است."
        )
    else:
        score -= 1
        reasons.append(
            "EMA20 زیر EMA50 است."
        )

    if x["ema50"] > x["ema200"]:
        score += 1
        reasons.append(
            "EMA50 بالای EMA200 است."
        )
    else:
        score -= 1
        reasons.append(
            "EMA50 زیر EMA200 است."
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if current_rsi >= 55:

        score += 1

        reasons.append(
            "RSI قدرت نسبی خریداران را نشان می‌دهد."
        )

    elif current_rsi <= 45:

        score -= 1

        reasons.append(
            "RSI قدرت نسبی فروشندگان را نشان می‌دهد."
        )

    else:

        reasons.append(
            "RSI در محدوده خنثی قرار دارد."
        )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if x["macd_hist"] > 0:

        score += 1

        reasons.append(
            "هیستوگرام MACD مثبت است."
        )

    else:

        score -= 1

        reasons.append(
            "هیستوگرام MACD منفی است."
        )

    # -----------------------------------------------------
    # Support / Resistance
    # -----------------------------------------------------

    recent = d.tail(
        min(100, len(d))
    )

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    # -----------------------------------------------------
    # Signal
    # -----------------------------------------------------

    if score >= 3:

        signal = "LONG"

        entry = price

        sl = min(
            support,
            price - 1.5 * current_atr
        )

        if sl >= entry:
            sl = entry - 1.5 * current_atr

        risk = max(
            entry - sl,
            current_atr * 0.5
        )

        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        tp3 = entry + 3.5 * risk

    elif score <= -3:

        signal = "SHORT"

        entry = price

        sl = max(
            resistance,
            price + 1.5 * current_atr
        )

        if sl <= entry:
            sl = entry + 1.5 * current_atr

        risk = max(
            sl - entry,
            current_atr * 0.5
        )

        tp1 = entry - 1.5 * risk
        tp2 = entry - 2.5 * risk
        tp3 = entry - 3.5 * risk

    else:

        signal = "NO TRADE"

        entry = price
        sl = None
        tp1 = None
        tp2 = None
        tp3 = None

    # -----------------------------------------------------
    # Safe rounding
    # -----------------------------------------------------

    def rnd(value):

        if value is None:
            return None

        try:

            value = float(value)

            if not math.isfinite(value):
                return None

            return round(value, 8)

        except Exception:

            return None

    return {

        "signal": signal,

        "price": rnd(price),

        "entry": rnd(entry),

        "stop_loss": rnd(sl),

        "take_profit_1": rnd(tp1),

        "take_profit_2": rnd(tp2),

        "take_profit_3": rnd(tp3),

        "rsi": round(
            current_rsi,
            2
        ),

        "ema20": rnd(
            x["ema20"]
        ),

        "ema50": rnd(
            x["ema50"]
        ),

        "ema200": rnd(
            x["ema200"]
        ),

        "macd": rnd(
            x["macd"]
        ),

        "macd_signal": rnd(
            x["macd_signal"]
        ),

        "atr": rnd(
            current_atr
        ),

        "support": rnd(
            support
        ),

        "resistance": rnd(
            resistance
        ),

        "score": score,

        "reasons": reasons
    }


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Crypto Analyzer Online",
        "version": "3.0.0",
        "data_source": "CoinGecko"
    }


# =========================================================
# Analyze API
# =========================================================

@app.get("/analyze")
def analyze_market(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h")
):

    normalized_symbol = normalize_symbol(
        symbol
    )

    normalized_interval = normalize_interval(
        interval
    )

    df, coin_id = get_market_prices(
        normalized_symbol,
        normalized_interval
    )

    result = analyze(df)

    result["symbol"] = normalized_symbol

    result["coin_id"] = coin_id

    result["interval"] = normalized_interval

    result["data_source"] = "CoinGecko"

    return result


# =========================================================
# Frontend
# =========================================================

HTML = """
<!doctype html>

<html lang="fa" dir="rtl">

<head>

<meta charset="utf-8">

<meta
name="viewport"
content="width=device-width,initial-scale=1"
>

<title>
Crypto Analyzer Online
</title>

<style>

*{
box-sizing:border-box;
}

body{

margin:0;

background:
radial-gradient(
circle at top,
#172554,
#07101f 55%,
#030712
);

color:white;

font-family:
Tahoma,
Arial,
sans-serif;

min-height:100vh;

}

.wrap{

max-width:800px;

margin:0 auto;

padding:25px 16px;

}

.card{

background:
rgba(17,29,50,.92);

border:
1px solid rgba(255,255,255,.08);

padding:28px;

border-radius:28px;

box-shadow:
0 20px 60px
rgba(0,0,0,.35);

}

h1{

direction:ltr;

text-align:left;

font-size:30px;

margin-top:0;

}

.sub{

font-size:19px;

color:#cbd5e1;

margin-bottom:28px;

}

label{

display:block;

margin-top:15px;

margin-bottom:6px;

color:#cbd5e1;

}

input,
select,
button{

width:100%;

padding:16px;

border:0;

border-radius:14px;

font-size:17px;

margin:6px 0;

}

input,
select{

background:#0b1528;

color:white;

border:
1px solid #24344f;

}

button{

background:#16a34a;

color:white;

font-weight:bold;

cursor:pointer;

transition:.2s;

}

button:hover{

background:#22c55e;

}

button:disabled{

opacity:.6;

cursor:wait;

}

.result{

margin-top:20px;

background:#050d1b;

padding:20px;

border-radius:18px;

line-height:2;

}

.grid{

display:grid;

grid-template-columns:
repeat(2,1fr);

gap:10px;

}

.box{

background:#17243a;

padding:12px;

border-radius:13px;

}

.signal{

font-size:28px;

font-weight:bold;

text-align:center;

padding:12px;

margin-bottom:15px;

}

.info{

color:#94a3b8;

font-size:13px;

margin-top:18px;

}

@media(max-width:600px){

.grid{

grid-template-columns:1fr 1fr;

}

h1{

font-size:24px;

}

}

</style>

</head>

<body>

<div class="wrap">

<div class="card">

<h1>
📊 Crypto Analyzer Online
</h1>

<div class="sub">
تحلیل آنلاین بازار ارز دیجیتال
</div>

<label>
نماد ارز
</label>

<input
id="symbol"
value="BTCUSDT"
placeholder="مثلاً BTCUSDT"
/>

<label>
تایم‌فریم
</label>

<select id="interval">

<option value="1h">
1 ساعت
</option>

<option value="4h">
4 ساعت
</option>

<option value="1d">
1 روز
</option>

<option value="1w">
1 هفته
</option>

</select>

<button
id="btn"
onclick="run()"
>
🔍 تحلیل بازار
</button>

<div id="result">

آماده تحلیل...

</div>

</div>

</div>

<script>

const f = v => {

if(v === null || v === undefined)
return "-";

return Number(v).toLocaleString(
"en-US",
{
maximumFractionDigits:8
}
);

};

async function run(){

const symbol =
document
.getElementById("symbol")
.value
.trim()
.toUpperCase();

const interval =
document
.getElementById("interval")
.value;

const button =
document.getElementById("btn");

const result =
document.getElementById("result");

button.disabled = true;

button.textContent =
"⏳ در حال دریافت داده...";

result.innerHTML =
"در حال دریافت اطلاعات بازار...";

try{

const url =
`/analyze?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}`;

const response =
await fetch(url);

let data;

try{

data = await response.json();

}catch(e){

throw new Error(
"پاسخ معتبر از سرور دریافت نشد."
);

}

if(!response.ok){

throw new Error(
data.detail ||
"خطای نامشخص"
);

}

result.innerHTML = `

<div class="signal">

${data.signal}

</div>

<div class="grid">

<div class="box">
قیمت
<br>
<b>${f(data.price)}</b>
</div>

<div class="box">
RSI
<br>
<b>${f(data.rsi)}</b>
</div>

<div class="box">
ورود
<br>
<b>${f(data.entry)}</b>
</div>

<div class="box">
حد ضرر
<br>
<b>${f(data.stop_loss)}</b>
</div>

<div class="box">
TP1
<br>
<b>${f(data.take_profit_1)}</b>
</div>

<div class="box">
TP2
<br>
<b>${f(data.take_profit_2)}</b>
</div>

<div class="box">
TP3
<br>
<b>${f(data.take_profit_3)}</b>
</div>

<div class="box">
امتیاز
<br>
<b>${data.score}</b>
</div>

<div class="box">
حمایت
<br>
<b>${f(data.support)}</b>
</div>

<div class="box">
مقاومت
<br>
<b>${f(data.resistance)}</b>
</div>

</div>

<br>

<div>

<b>
دلایل تحلیل:
</b>

<br>

${data.reasons.join("<br>")}

</div>

<div class="info">

منبع داده:
CoinGecko

<br>

این خروجی تحلیل تکنیکال است و
تضمین سود یا توصیه قطعی معامله نیست.

</div>

`;

}catch(error){

result.innerHTML =
"❌ خطا: " +
error.message;

}

finally{

button.disabled = false;

button.textContent =
"🔍 تحلیل بازار";

}

}

</script>

</body>

</html>
"""


# =========================================================
# Home
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML
