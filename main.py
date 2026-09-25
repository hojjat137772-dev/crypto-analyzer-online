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
    # EMA20
    # -------------------------

    if price > x["ema20"]:
        score += 1
        reasons.append("قیمت بالای EMA20 است.")
    else:
        score -= 1
        reasons.append("قیمت زیر EMA20 است.")

    # -------------------------
    # EMA20 / EMA50
    # -------------------------

    if x["ema20"] > x["ema50"]:
        score += 1
        reasons.append("EMA20 بالای EMA50 است.")
    else:
        score -= 1
        reasons.append("EMA20 زیر EMA50 است.")

    # -------------------------
    # EMA50 / EMA200
    # -------------------------

    if x["ema50"] > x["ema200"]:
        score += 1
        reasons.append("EMA50 بالای EMA200 است.")
    else:
        score -= 1
        reasons.append("EMA50 زیر EMA200 است.")

    # -------------------------
    # RSI
    # -------------------------

    if current_rsi >= 55:
        score += 1
        reasons.append("RSI نشان‌دهنده قدرت خریداران است.")

    elif current_rsi <= 45:
        score -= 1
        reasons.append("RSI نشان‌دهنده قدرت فروشندگان است.")

    else:
        reasons.append("RSI در محدوده خنثی است.")

    # -------------------------
    # MACD
    # -------------------------

    if x["macd_hist"] > 0:
        score += 1
        reasons.append("MACD مثبت است.")
    else:
        score -= 1
        reasons.append("MACD منفی است.")

# -------------------------
# Signal
# -------------------------

if score >= 3:

    signal = "LONG"
    entry = price

    stop_loss = min(
        support,
        price - 1.5 * current_atr
    )

    if stop_loss >= entry:
        stop_loss = entry - (1.5 * current_atr)

    risk = max(
        entry - stop_loss,
        current_atr * 0.5
    )

    tp1 = entry + 1.5 * risk
    tp2 = entry + 2.5 * risk
    tp3 = entry + 3.5 * risk

    reasons.append(
        "مجموع شرایط برای ورود LONG مناسب است."
    )


elif score <= -3:

    signal = "SHORT"
    entry = price

    stop_loss = max(
        resistance,
        price + 1.5 * current_atr
    )

    if stop_loss <= entry:
        stop_loss = entry + (1.5 * current_atr)

    risk = max(
        stop_loss - entry,
        current_atr * 0.5
    )

    tp1 = entry - 1.5 * risk
    tp2 = entry - 2.5 * risk
    tp3 = entry - 3.5 * risk

    reasons.append(
        "مجموع شرایط برای ورود SHORT مناسب است."
    )


else:

    signal = "NO TRADE"
    entry = price

    stop_loss = None
    tp1 = None
    tp2 = None
    tp3 = None

    risk = None

    reasons.append(
        "شرایط فعلی برای ورود قدرتمند کافی نیست."
    )
    # -------------------------
    # Confidence 0 - 100
    # -------------------------

    confidence = round(
        ((score + 5) / 10) * 100
    )

    confidence = max(
        0,
        min(100, confidence)
    )

    # -------------------------
    # Risk / Reward
    # -------------------------

    rr1 = None
    rr2 = None
    rr3 = None

    if signal == "LONG" and risk and risk > 0:

        rr1 = round((tp1 - entry) / risk, 2)
        rr2 = round((tp2 - entry) / risk, 2)
        rr3 = round((tp3 - entry) / risk, 2)

    elif signal == "SHORT" and risk and risk > 0:

        rr1 = round((entry - tp1) / risk, 2)
        rr2 = round((entry - tp2) / risk, 2)
        rr3 = round((entry - tp3) / risk, 2)

    
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

    result["data_source"] = "Kraken"

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
