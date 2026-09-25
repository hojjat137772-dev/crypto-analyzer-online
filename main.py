from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import math

app = FastAPI(
    title="Crypto Analyzer Online",
    version="4.1.0"
)

KRAKEN_API = "https://api.kraken.com/0/public"

session = requests.Session()
session.headers.update({
    "User-Agent": "CryptoAnalyzerOnline/4.1"
})


# =========================================================
# TIMEFRAMES
# =========================================================

INTERVALS = {
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,

    "1 ساعت": 60,
    "4 ساعت": 240,
    "1 روز": 1440,
    "1 هفته": 10080
}


# =========================================================
# SYMBOL
# =========================================================

def normalize_symbol(symbol):
    return (
        str(symbol)
        .strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def normalize_interval(interval):
    value = str(interval).strip().lower()

    aliases = {
        "1 ساعت": "1h",
        "4 ساعت": "4h",
        "1 روز": "1d",
        "1 هفته": "1w"
    }

    return aliases.get(value, value)


def get_base_symbol(symbol):
    s = normalize_symbol(symbol)

    quotes = [
        "USDT",
        "USDC",
        "BUSD",
        "USD"
    ]

    for quote in quotes:
        if s.endswith(quote) and len(s) > len(quote):
            return s[:-len(quote)]

    return s


# =========================================================
# KRAKEN PAIR
# =========================================================

def get_pair_candidates(symbol):

    base = get_base_symbol(symbol)

    if base == "BTC":
        bases = ["XBT", "BTC"]
    else:
        bases = [base]

    candidates = []

    for b in bases:
        candidates.extend([
            b + "USDT",
            b + "USD",
            b + "USDC"
        ])

    return list(dict.fromkeys(candidates))


def get_asset_pairs():

    try:
        response = session.get(
            f"{KRAKEN_API}/AssetPairs",
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            return {}

        return data.get("result", {})

    except Exception:
        return {}


def find_kraken_pair(symbol):

    candidates = get_pair_candidates(symbol)

    # ابتدا مستقیماً جفت‌های رایج را امتحان می‌کنیم
    for pair in candidates:

        try:

            response = session.get(
                f"{KRAKEN_API}/OHLC",
                params={
                    "pair": pair,
                    "interval": 60
                },
                timeout=10
            )

            if response.status_code != 200:
                continue

            data = response.json()

            if not data.get("error"):

                result = data.get("result", {})

                if result:
                    return pair

        except Exception:
            continue

    # اگر نام مستقیم جواب نداد، AssetPairs را بررسی می‌کنیم
    pairs = get_asset_pairs()

    if not pairs:
        return None

    base = get_base_symbol(symbol)

    if base == "BTC":
        bases = {"BTC", "XBT"}
    else:
        bases = {base}

    for key, info in pairs.items():

        altname = str(
            info.get("altname", "")
        ).upper()

        wsname = str(
            info.get("wsname", "")
        ).upper()

        text = altname + " " + wsname

        for b in bases:

            if (
                b + "USD" in text
                or
                b + "USDT" in text
                or
                b + "/USD" in text
                or
                b + "/USDT" in text
            ):
                return key

    return None


# =========================================================
# OHLC DATA
# =========================================================

def get_ohlc(symbol, interval):

    interval = normalize_interval(interval)

    if interval not in INTERVALS:

        raise HTTPException(
            400,
            "تایم‌فریم معتبر نیست. "
            "از 1 ساعت، 4 ساعت، 1 روز یا 1 هفته استفاده کنید."
        )

    kraken_interval = INTERVALS[interval]

    pair = find_kraken_pair(symbol)

    if not pair:

        raise HTTPException(
            404,
            f"جفت معاملاتی {symbol} در Kraken پیدا نشد."
        )

    try:

        response = session.get(
            f"{KRAKEN_API}/OHLC",
            params={
                "pair": pair,
                "interval": kraken_interval
            },
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException:

        raise HTTPException(
            502,
            "ارتباط با Kraken برقرار نشد."
        )

    if data.get("error"):

        raise HTTPException(
            400,
            "Kraken خطا برگرداند: "
            + " | ".join(
                str(x)
                for x in data["error"]
            )
        )

    result = data.get("result", {})

    rows = None

    for key, value in result.items():

        if key != "last":
            rows = value
            break

    if not rows:

        raise HTTPException(
            400,
            "Kraken برای این ارز داده‌ای برنگرداند."
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count"
        ]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    if len(df) < 60:

        raise HTTPException(
            400,
            "داده کافی برای تحلیل این ارز وجود ندارد."
        )

    return df, pair


# =========================================================
# INDICATORS
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

    true_range = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"] -
                previous_close
            ).abs(),

            (
                df["low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
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

    d["ema200"] = ema(
        d["close"],
        200
    )

    d["rsi"] = rsi(
        d["close"],
        14
    )

    d["macd"] = (
        ema(d["close"], 12)
        -
        ema(d["close"], 26)
    )

    d["macd_signal"] = ema(
        d["macd"],
        9
    )

    d["macd_hist"] = (
        d["macd"]
        -
        d["macd_signal"]
    )

    d["atr"] = atr(
        d,
        14
    )
    # --------------------------
    # VOLUME ANALYSIS
    # --------------------------

    d["volume_ma20"] = d["volume"].rolling(20).mean()

    d["volume_ratio"] = (
        d["volume"] / d["volume_ma20"]
    )
    return (
        d
        .dropna()
        .reset_index(drop=True)
    )


# =========================================================
# ANALYSIS
# =========================================================

def analyze(df):

    d = add_indicators(df)

    if len(d) < 10:

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

# -------------------------
# SMART SCORING ENGINE
# -------------------------

# EMA20 / Price
if price > x["ema20"]:
    score += 1
    reasons.append("قیمت بالای EMA20 است.")
else:
    score -= 1
    reasons.append("قیمت زیر EMA20 است.")

# EMA20 / EMA50
if x["ema20"] > x["ema50"]:
    score += 1
    reasons.append("EMA20 بالای EMA50 است.")
else:
    score -= 1
    reasons.append("EMA20 زیر EMA50 است.")

# EMA50 / EMA200
if x["ema50"] > x["ema200"]:
    score += 1
    reasons.append("EMA50 بالای EMA200 است.")
else:
    score -= 1
    reasons.append("EMA50 زیر EMA200 است.")

# RSI
if current_rsi >= 60:
    score += 1
    reasons.append("RSI قدرت خریداران را تأیید می‌کند.")
elif current_rsi <= 40:
    score -= 1
    reasons.append("RSI قدرت فروشندگان را تأیید می‌کند.")
else:
    reasons.append("RSI در محدوده خنثی است.")

# MACD
if x["macd_hist"] > 0:
    score += 1
    reasons.append("MACD مثبت است.")
else:
    score -= 1
    reasons.append("MACD منفی است.")

# Volume confirmation
volume_ratio = float(x.get("volume_ratio", 1))

if volume_ratio >= 1.20:
    if score > 0:
        score += 1
        reasons.append("حجم معاملات ورود قدرت را تأیید می‌کند.")
    elif score < 0:
        score -= 1
        reasons.append("حجم معاملات فشار فروش را تأیید می‌کند.")
else:
    reasons.append("حجم معاملات تأیید قدرتمندی نمی‌دهد.")

# محدود کردن امتیاز
score = max(-6, min(6, score))
    
# -----------------------
# SMART SIGNAL ENGINE
# -----------------------

signal = "NO TRADE"
entry = price

stop_loss = None
tp1 = None
tp2 = None
tp3 = None
risk = None

# -----------------------
# LONG
# -----------------------

if score >= 3:

    signal = "LONG"
    entry = price

    # حد ضرر ترکیبی:
    # حمایت + ATR
    atr_stop = price - (1.5 * current_atr)

    if support is not None:
        stop_loss = min(support, atr_stop)
    else:
        stop_loss = atr_stop

    # جلوگیری از حد ضرر نامعتبر
    if stop_loss >= entry:
        stop_loss = entry - (1.5 * current_atr)

    risk = max(
        entry - stop_loss,
        current_atr * 0.5
    )

    # اهداف اولیه
    raw_tp1 = entry + (1.5 * risk)
    raw_tp2 = entry + (2.5 * risk)
    raw_tp3 = entry + (3.5 * risk)

    # استفاده از مقاومت در هدف اول، اگر معتبر باشد
    if resistance is not None and resistance > entry:
        tp1 = resistance
    else:
        tp1 = raw_tp1

    # جلوگیری از نزدیک بودن TP2 و TP3
    tp2 = max(raw_tp2, tp1 + risk)
    tp3 = max(raw_tp3, tp2 + risk)

    reasons.append(
        "مجموع شرایط تکنیکال برای LONG تأیید شده است."
    )

# -----------------------
# SHORT
# -----------------------

elif score <= -3:

    signal = "SHORT"
    entry = price

    # حد ضرر ترکیبی:
    # مقاومت + ATR
    atr_stop = price + (1.5 * current_atr)

    if resistance is not None:
        stop_loss = max(resistance, atr_stop)
    else:
        stop_loss = atr_stop

    # جلوگیری از حد ضرر نامعتبر
    if stop_loss <= entry:
        stop_loss = entry + (1.5 * current_atr)

    risk = max(
        stop_loss - entry,
        current_atr * 0.5
    )

    # اهداف اولیه
    raw_tp1 = entry - (1.5 * risk)
    raw_tp2 = entry - (2.5 * risk)
    raw_tp3 = entry - (3.5 * risk)

    # استفاده از حمایت در هدف اول، اگر معتبر باشد
    if support is not None and support < entry:
        tp1 = support
    else:
        tp1 = raw_tp1

    # جلوگیری از نزدیک بودن TP2 و TP3
    tp2 = min(raw_tp2, tp1 - risk)
    tp3 = min(raw_tp3, tp2 - risk)

    reasons.append(
        "مجموع شرایط تکنیکال برای SHORT تأیید شده است."
    )

# -----------------------
# NO TRADE
# -----------------------

else:

    signal = "NO TRADE"
    entry = price

    stop_loss = None
    tp1 = None
    tp2 = None
    tp3 = None
    risk = None

    reasons.append(
        "شرایط کافی برای ورود به معامله وجود ندارد."
    )

# ----------------
# CONFIDENCE ENGINE
# ----------------

# تبدیل امتیاز فعلی به بازه 0 تا 100
confidence = round(
    max(
        0,
        min(
            100,
            ((score + 6) / 12) * 100
        )
    )
)

# تقویت Confidence بر اساس RSI
if signal == "LONG":
    if 50 <= current_rsi <= 65:
        confidence += 5
    elif current_rsi > 70:
        confidence -= 5

elif signal == "SHORT":
    if 35 <= current_rsi <= 50:
        confidence += 5
    elif current_rsi < 30:
        confidence -= 5

# تقویت بر اساس حجم
if volume_ratio >= 1.20:
    confidence += 5

# محدود کردن به 0 تا 100
confidence = max(
    0,
    min(100, confidence)
)

confidence = round(confidence)
# ------------------------
# RISK / REWARD
# ------------------------

rr1 = None
rr2 = None
rr3 = None

if signal == "LONG" and risk is not None and risk > 0:

    rr1 = round(
        (tp1 - entry) / risk,
        2
    )

    rr2 = round(
        (tp2 - entry) / risk,
        2
    )

    rr3 = round(
        (tp3 - entry) / risk,
        2
    )

elif signal == "SHORT" and risk is not None and risk > 0:

    rr1 = round(
        (entry - tp1) / risk,
        2
    )

    rr2 = round(
        (entry - tp2) / risk,
        2
    )

    rr3 = round(
        (entry - tp3) / risk,
        2
    )
    # -------------------------
    # Rounding
    # -------------------------

    def rnd(value):

        if value is None:
            return None

        try:

            value = float(value)

            if not math.isfinite(value):
                return None

            return round(
                value,
                8
            )

        except Exception:

            return None

return {
    "signal": signal,

    "price": rnd(price),
    "entry": rnd(entry),

    "stop_loss": rnd(stop_loss),

    "take_profit_1": rnd(tp1),
    "take_profit_2": rnd(tp2),
    "take_profit_3": rnd(tp3),

    "risk": rnd(risk),

    "confidence": confidence,
"trend_strength": trend_strength,
    "rr1": rr1,
    "rr2": rr2,
    "rr3": rr3,

    "rsi": round(current_rsi, 2),

    "ema20": rnd(x["ema20"]),
    "ema50": rnd(x["ema50"]),
    "ema200": rnd(x["ema200"]),

    "macd": rnd(x["macd"]),
    "macd_signal": rnd(x["macd_signal"]),

    "atr": rnd(current_atr),

    "support": rnd(support),
    "resistance": rnd(resistance),

    "score": score,

    "reasons": reasons
# ============================================================
# MULTI TIMEFRAME ANALYSIS
# ============================================================

def get_multi_timeframe_analysis(symbol):
    timeframes = {
        "1h": "1 ساعت",
        "4h": "4 ساعت",
        "1d": "1 روز"
    }

    result = {}

    for interval, title in timeframes.items():
        try:
            df, pair = get_ohlc(symbol, interval)
            data = analyze(df)

            signal = data.get("signal", "NO TRADE")
            score = data.get("score", 0)

            if score >= 3:
                trend = "صعودی"
            elif score <= -3:
                trend = "نزولی"
            else:
                trend = "خنثی"

            result[interval] = {
                "title": title,
                "signal": signal,
                "trend": trend,
                "score": score,
                "confidence": data.get("confidence", 0),
                "rsi": data.get("rsi")
            }

        except Exception as e:
            result[interval] = {
                "title": title,
                "signal": "NO TRADE",
                "trend": "نامشخص",
                "score": 0,
                "confidence": 0,
                "rsi": None,
                "error": str(e)
            }

    # --------------------------------------------------------
    # FINAL MULTI-TIMEFRAME DECISION
    # --------------------------------------------------------

    h1 = result.get("1h", {})
    h4 = result.get("4h", {})
    d1 = result.get("1d", {})

    signals = [
        h1.get("signal"),
        h4.get("signal"),
        d1.get("signal")
    ]

    if h1.get("signal") == "LONG" and h4.get("signal") == "LONG":
        final_trend = "LONG"

    elif h1.get("signal") == "SHORT" and h4.get("signal") == "SHORT":
        final_trend = "SHORT"

    else:
        final_trend = "NO TRADE"

    # --------------------------------------------------------
    # ALIGNMENT SCORE
    # --------------------------------------------------------

    alignment = 0

    if h1.get("signal") == h4.get("signal"):
        if h1.get("signal") in ["LONG", "SHORT"]:
            alignment += 40

    if h4.get("signal") == d1.get("signal"):
        if h4.get("signal") in ["LONG", "SHORT"]:
            alignment += 30

    if d1.get("signal") == h1.get("signal"):
        if d1.get("signal") in ["LONG", "SHORT"]:
            alignment += 30

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "1h": h1,
        "4h": h4,
        "1d": d1,
        "final_signal": final_trend,
        "alignment": alignment
    }
# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Crypto Analyzer Online",
        "version": "4.1.0",
        "data_source": "Kraken"
    }


# =========================================================
# ANALYZE API
# =========================================================

@app.get("/analyze")
def analyze_market(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h")
):

    symbol = normalize_symbol(
        symbol
    )

    interval = normalize_interval(
        interval
    )

    df, pair = get_ohlc(
        symbol,
        interval
    )

    result = analyze(df)

    result["symbol"] = symbol

    result["pair"] = pair

    result["interval"] = interval
result["mtf"] = get_mtf_analysis(symbol)
    result["data_source"] = "Kraken"
    # MULTI TIMEFRAME
    result["multi_timeframe"] = get_multi_timeframe_analysis(symbol)
    return result


# =========================================================
# FRONTEND
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

min-height:100vh;

background:
radial-gradient(
circle at top,
#172554,
#07101f 55%,
#020617
);

color:white;

font-family:
Tahoma,
Arial,
sans-serif;

}

.wrap{

max-width:800px;

margin:auto;

padding:25px 16px;

}

.card{

background:
rgba(17,29,50,.95);

border:
1px solid rgba(255,255,255,.08);

padding:28px;

border-radius:28px;

box-shadow:
0 20px 60px rgba(0,0,0,.4);

}

h1{

direction:ltr;

text-align:left;

font-size:30px;

margin-top:0;

}

.sub{

font-size:20px;

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

margin:7px 0;

}

input,
select{

background:#081426;

color:white;

border:
1px solid #263751;

}

button{

background:#16a34a;

color:white;

font-weight:bold;

cursor:pointer;

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
1fr 1fr;

gap:10px;

}

.box{

background:#17243a;

padding:12px;

border-radius:13px;

}

.signal{

font-size:30px;

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

h1{
font-size:24px;
}

.grid{
grid-template-columns:1fr 1fr;
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
onclick="runAnalysis()"
>
🔍 تحلیل بازار
</button>

<div id="result">

آماده تحلیل...

</div>

</div>

</div>


<script>

function formatNumber(value){

if(
value === null ||
value === undefined
){

return "-";

}

return Number(value).toLocaleString(
"en-US",
{
maximumFractionDigits:8
}
);

}


async function runAnalysis(){

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
"⏳ در حال تحلیل...";

result.innerHTML =
"در حال دریافت داده‌های بازار...";


try{

const url =
"/analyze?symbol=" +
encodeURIComponent(symbol) +
"&interval=" +
encodeURIComponent(interval);

const response =
await fetch(url);

let data;

try{

data =
await response.json();

}
catch(error){

throw new Error(
"پاسخ معتبر از سرور دریافت نشد."
);

}


if(!response.ok){

throw new Error(
data.detail ||
"خطای نامشخص در سرور"
);

}


result.innerHTML = `

<div class="signal">

${data.signal}

</div>

<div class="signal">
${data.signal}
</div>

<div style="
margin:18px 0;
padding:18px;
border-radius:22px;
background:rgba(20,35,60,.75);
border:1px solid rgba(255,255,255,.08);
">

<h2 style="margin-top:0;text-align:center;">
📊 تحلیل چندتایم‌فریمی
</h2>

<div class="grid">

<div class="box">
1 ساعت
<br><br>
<b>${data.mtf?.["1h"]?.signal || "-"}</b>
<br>
${data.mtf?.["1h"]?.confidence ?? "-"}%
</div>

<div class="box">
4 ساعت
<br><br>
<b>${data.mtf?.["4h"]?.signal || "-"}</b>
<br>
${data.mtf?.["4h"]?.confidence ?? "-"}%
</div>

<div class="box">
1 روز
<br><br>
<b>${data.mtf?.["1d"]?.signal || "-"}</b>
<br>
${data.mtf?.["1d"]?.confidence ?? "-"}%
</div>

</div>

</div>

<div class="grid">
<div class="grid">

<div class="box">

قیمت

<br>

<b>
${formatNumber(data.price)}
</b>

</div>


<div class="box">

RSI

<br>

<b>
${formatNumber(data.rsi)}
</b>

</div>

<div class="box">

اعتماد تحلیل

<br>

<b>
${formatNumber(data.confidence)}%
</b>

</div>

<div class="box">

قدرت روند

<br>

<b>
${formatNumber(data.trend_strength)}
</b>

</div>

<div class="box">

R/R - TP1

<br>

<b>
${formatNumber(data.rr1)}
</b>

</div>

<div class="box">

R/R - TP2

<br>

<b>
${formatNumber(data.rr2)}
</b>

</div>

<div class="box">

R/R - TP3

<br>

<b>
${formatNumber(data.rr3)}
</b>

</div>
<div class="box">

ورود

<br>

<b>
${formatNumber(data.entry)}
</b>

</div>


<div class="box">

حد ضرر

<br>

<b>
${formatNumber(data.stop_loss)}
</b>

</div>


<div class="box">

TP1

<br>

<b>
${formatNumber(data.take_profit_1)}
</b>

</div>


<div class="box">

TP2

<br>

<b>
${formatNumber(data.take_profit_2)}
</b>

</div>


<div class="box">

TP3

<br>

<b>
${formatNumber(data.take_profit_3)}
</b>

</div>


<div class="box">

امتیاز

<br>

<b>
${data.score}
</b>

</div>

<div class="box">

اعتماد سیگنال

<br>

<b>
${formatNumber(data.confidence)}%
</b>

</div>

<div class="box">
<div class="box">

R/R - TP1

<br>

<b>
${formatNumber(data.rr1)}
</b>

</div>

<div class="box">

R/R - TP2

<br>

<b>
${formatNumber(data.rr2)}
</b>

</div>

<div class="box">

R/R - TP3

<br>

<b>
${formatNumber(data.rr3)}
</b>

</div>
<div class="box">

حمایت

<br>

<b>
${formatNumber(data.support)}
</b>

</div>


<div class="box">

مقاومت

<br>

<b>
${formatNumber(data.resistance)}
</b>

</div>

</div>


<br>


<b>
دلایل تحلیل:
</b>

<br>

${data.reasons.join("<br>")}


<div class="info">

منبع داده:
Kraken

<br>

جفت:
${data.pair}

<br><br>

این خروجی تحلیل تکنیکال است و
تضمین سود یا توصیه قطعی معامله نیست.

</div>

`;

}

catch(error){

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
# HOME
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML
