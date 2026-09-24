from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import math

app = FastAPI(title="Crypto Analyzer Online", version="4.0.0")

KRAKEN = "https://api.kraken.com/0/public"

session = requests.Session()
session.headers.update({
    "User-Agent": "CryptoAnalyzerOnline/4.0"
})

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


def normalize_symbol(symbol):
    return (
        str(symbol).strip().upper()
        .replace("/", "")
        .replace("-", "")
        .replace(" ", "")
    )


def base_symbol(symbol):
    s = normalize_symbol(symbol)

    for q in ["USDT", "USDC", "USD", "BUSD"]:
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)]

    return s


def get_pairs():
    try:
        r = session.get(
            f"{KRAKEN}/AssetPairs",
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        return data.get("result", {})
    except Exception:
        raise HTTPException(
            502,
            "دریافت فهرست ارزهای Kraken ناموفق بود."
        )


def find_pair(symbol):
    base = base_symbol(symbol)

    if base == "BTC":
        bases = {"XBT", "BTC"}
    else:
        bases = {base}

    pairs = get_pairs()

    candidates = []

    for key, p in pairs.items():

        altname = str(p.get("altname", "")).upper()
        wsname = str(p.get("wsname", "")).upper()

        pair_text = altname + wsname

        if any(
            (b + "/USD") in pair_text or
            (b + "/USDT") in pair_text or
            (b + "USD") in pair_text or
            (b + "USDT") in pair_text
            for b in bases
        ):
            candidates.append(key)

    if not candidates:
        raise HTTPException(
            404,
            f"جفت معاملاتی {symbol} در Kraken پیدا نشد."
        )

    # ترجیح USDT
    for c in candidates:
        if "USDT" in c.upper():
            return c

    return candidates[0]


def get_ohlc(symbol, interval):

    if interval not in INTERVALS:
        raise HTTPException(
            400,
            "تایم‌فریم معتبر نیست."
        )

    kraken_interval = INTERVALS[interval]

    pair = find_pair(symbol)

    try:
        r = session.get(
            f"{KRAKEN}/OHLC",
            params={
                "pair": pair,
                "interval": kraken_interval
            },
            timeout=20
        )

        r.raise_for_status()

        data = r.json()

    except Exception:
        raise HTTPException(
            502,
            "ارتباط با منبع داده بازار برقرار نشد."
        )

    if data.get("error"):
        raise HTTPException(
            400,
            "Kraken داده‌ای برای این ارز برنگرداند."
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
            "داده کافی برای تحلیل وجود ندارد."
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

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    df = df.dropna().reset_index(drop=True)

    if len(df) < 60:
        raise HTTPException(
            400,
            "داده کافی برای تحلیل این ارز وجود ندارد."
        )

    return df, pair


def ema(s, n):
    return s.ewm(
        span=n,
        adjust=False
    ).mean()


def rsi(s, n=14):

    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n
    ).mean()

    al = loss.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n
    ).mean()

    rs = ag / al.replace(
        0,
        float("nan")
    )

    return (
        100 - 100 / (1 + rs)
    ).fillna(50)


def atr(df, n=14):

    pc = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n
    ).mean()


def indicators(df):

    d = df.copy()

    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)

    d["rsi"] = rsi(d["close"])

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

    d["atr"] = atr(d)

    return d.dropna().reset_index(
        drop=True
    )


def analyze(df):

    d = indicators(df)

    x = d.iloc[-1]

    price = float(x["close"])
    a = float(x["atr"])
    rv = float(x["rsi"])

    score = 0
    reasons = []

    if price > x["ema20"]:
        score += 1
        reasons.append("قیمت بالای EMA20 است.")
    else:
        score -= 1
        reasons.append("قیمت زیر EMA20 است.")

    if x["ema20"] > x["ema50"]:
        score += 1
        reasons.append("EMA20 بالای EMA50 است.")
    else:
        score -= 1
        reasons.append("EMA20 زیر EMA50 است.")

    if x["ema50"] > x["ema200"]:
        score += 1
        reasons.append("EMA50 بالای EMA200 است.")
    else:
        score -= 1
        reasons.append("EMA50 زیر EMA200 است.")

    if rv >= 55:
        score += 1
        reasons.append("RSI متمایل به قدرت خریداران است.")
    elif rv <= 45:
        score -= 1
        reasons.append("RSI متمایل به قدرت فروشندگان است.")
    else:
        reasons.append("RSI در محدوده خنثی است.")

    if x["macd_hist"] > 0:
        score += 1
        reasons.append("MACD مثبت است.")
    else:
        score -= 1
        reasons.append("MACD منفی است.")

    recent = d.tail(100)

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    if score >= 3:

        signal = "LONG"

        entry = price

        sl = min(
            support,
            price - 1.5 * a
        )

        if sl >= entry:
            sl = entry - 1.5 * a

        risk = max(
            entry - sl,
            a * 0.5
        )

        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        tp3 = entry + 3.5 * risk

    elif score <= -3:

        signal = "SHORT"

        entry = price

        sl = max(
            resistance,
            price + 1.5 * a
        )

        if sl <= entry:
            sl = entry + 1.5 * a

        risk = max(
            sl - entry,
            a * 0.5
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

    def rnd(v):

        if v is None:
            return None

        try:
            v = float(v)

            if not math.isfinite(v):
                return None

            return round(v, 8)

        except:
            return None

    return {
        "signal": signal,
        "price": rnd(price),
        "entry": rnd(entry),
        "stop_loss": rnd(sl),
        "take_profit_1": rnd(tp1),
        "take_profit_2": rnd(tp2),
        "take_profit_3": rnd(tp3),
        "rsi": round(rv, 2),
        "ema20": rnd(x["ema20"]),
        "ema50": rnd(x["ema50"]),
        "ema200": rnd(x["ema200"]),
        "macd": rnd(x["macd"]),
        "macd_signal": rnd(x["macd_signal"]),
        "atr": rnd(a),
        "support": rnd(support),
        "resistance": rnd(resistance),
        "score": score,
        "reasons": reasons
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "4.0.0",
        "data_source": "Kraken"
    }


@app.get("/analyze")
def analyze_market(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h")
):

    symbol = normalize_symbol(symbol)

    interval = normalize_interval(interval)

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


HTML = """
<!doctype html>
<html lang="fa" dir="rtl">

<head>

<meta charset="utf-8">

<meta
name="viewport"
content="width=device-width,initial-scale=1"
>

<title>Crypto Analyzer Online</title>

<style>

*{
box-sizing:border-box
}

body{
margin:0;
background:
linear-gradient(
180deg,
#0b1730,
#020617
);
color:white;
font-family:Tahoma,Arial;
min-height:100vh;
}

.wrap{
max-width:800px;
margin:auto;
padding:25px 16px;
}

.card{
background:#111d32;
padding:28px;
border-radius:28px;
box-shadow:0 20px 60px #0008;
}

h1{
direction:ltr;
text-align:left;
}

.sub{
font-size:20px;
color:#cbd5e1;
margin-bottom:25px;
}

label{
display:block;
margin-top:15px;
margin-bottom:5px;
}

input,select,button{
width:100%;
padding:16px;
border:0;
border-radius:14px;
font-size:17px;
margin:7px 0;
}

input,select{
background:#081426;
color:white;
border:1px solid #263751;
}

button{
background:#16a34a;
color:white;
font-weight:bold;
cursor:pointer;
}

button:disabled{
opacity:.6;
}

.result{
margin-top:20px;
background:#050d1b;
padding:18px;
border-radius:18px;
line-height:2;
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:10px;
}

.box{
background:#17243a;
padding:12px;
border-radius:12px;
}

.signal{
font-size:28px;
text-align:center;
font-weight:bold;
margin-bottom:15px;
}

.small{
color:#94a3b8;
font-size:13px;
margin-top:15px;
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

<label>نماد ارز</label>

<input
id="symbol"
value="BTCUSDT"
placeholder="مثلاً BTCUSDT"
/>

<label>تایم‌فریم</label>

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

const f = v =>
v == null
?
"-"
:
Number(v).toLocaleString(
"en-US",
{
maximumFractionDigits:8
}
);

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

const btn =
document.getElementById("btn");

const out =
document.getElementById("result");

btn.disabled=true;

btn.textContent=
"⏳ در حال تحلیل...";

out.textContent=
"دریافت داده‌های بازار...";

try{

const response =
await fetch(
`/analyze?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}`
);

const data =
await response.json();

if(!response.ok){

throw new Error(
data.detail ||
"خطا در دریافت اطلاعات"
);

}

out.innerHTML=`

<div class="signal">
${data.signal}
</div>

<div class="grid">

<div class="box">
قیمت<br>
<b>${f(data.price)}</b>
</div>

<div class="box">
RSI<br>
<b>${f(data.rsi)}</b>
</div>

<div class="box">
ورود<br>
<b>${f(data.entry)}</b>
</div>

<div class="box">
حد ضرر<br>
<b>${f(data.stop_loss)}</b>
</div>

<div class="box">
TP1<br>
<b>${f(data.take_profit_1)}</b>
</div>

<div class="box">
TP2<br>
<b>${f(data.take_profit_2)}</b>
</div>

<div class="box">
TP3<br>
<b>${f(data.take_profit_3)}</b>
</div>

<div class="box">
امتیاز<br>
<b>${data.score}</b>
</div>

<div class="box">
حمایت<br>
<b>${f(data.support)}</b>
</div>

<div class="box">
مقاومت<br>
<b>${f(data.resistance)}</b>
</div>

</div>

<br>

<b>دلایل تحلیل:</b>

<br>

${data.reasons.join("<br>")}

<div class="small">
منبع داده: Kraken<br>
این تحلیل تضمین سود یا توصیه قطعی معامله نیست.
</div>

`;

}catch(e){

out.innerHTML=
"❌ خطا: " +
e.message;

}

btn.disabled=false;

btn.textContent=
"🔍 تحلیل بازار";

}

</script>

</body>

</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML
