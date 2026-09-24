from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import math

app = FastAPI(title="Crypto Analyzer Online", version="2.0.0")

BINANCE_URL = "https://api.binance.us/api/v3/klines"

INTERVALS = {
    "1m":"1m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h",
    "1d":"1d","3d":"3d","1w":"1w",
    "1 دقیقه":"1m","5 دقیقه":"5m","15 دقیقه":"15m",
    "30 دقیقه":"30m","1 ساعت":"1h","2 ساعت":"2h",
    "4 ساعت":"4h","6 ساعت":"6h","12 ساعت":"12h",
    "1 روز":"1d","3 روز":"3d","1 هفته":"1w"
}

session = requests.Session()
session.headers.update({"User-Agent": "CryptoAnalyzerOnline/2.0"})


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "").replace("-", "").replace(" ", "")


def normalize_interval(interval: str) -> str:
    return INTERVALS.get(
        str(interval).strip().lower(),
        str(interval).strip().lower()
    )


def get_candles(symbol="BTCUSDT", interval="1h", limit=250):
    symbol = normalize_symbol(symbol)
    interval = normalize_interval(interval)

    if interval not in set(INTERVALS.values()):
        raise HTTPException(400, "تایم‌فریم معتبر نیست.")

    try:
        r = session.get(
            BINANCE_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": min(int(limit), 1000)
            },
            timeout=15
        )
    except requests.RequestException:
        raise HTTPException(502, "ارتباط با Binance برقرار نشد.")

    if r.status_code != 200:
        try:
            msg = r.json().get("msg", "ارز یا درخواست معتبر نیست.")
        except Exception:
            msg = "ارز یا درخواست معتبر نیست."
        raise HTTPException(400, msg)

    data = r.json()

    if not isinstance(data, list) or len(data) < 60:
        raise HTTPException(
            400,
            "داده کافی برای تحلیل این ارز وجود ندارد."
        )

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(data, columns=cols)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    return df.dropna().reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n
    ).mean()

    al = loss.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n
    ).mean()

    rs = ag / al.replace(0, float("nan"))

    return (
        100 - 100 / (1 + rs)
    ).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n
    ).mean()


def indicators(df):
    d = df.copy()

    for n in [20, 50, 100, 200]:
        d[f"ema{n}"] = ema(d["close"], n)

    d["rsi"] = rsi(d["close"])

    d["macd"] = (
        ema(d["close"], 12)
        - ema(d["close"], 26)
    )

    d["macd_signal"] = ema(d["macd"], 9)

    d["macd_hist"] = (
        d["macd"] - d["macd_signal"]
    )

    d["atr"] = atr(d)

    return d.dropna().reset_index(drop=True)


def analyze(df):
    d = indicators(df)
    x = d.iloc[-1]

    price = float(x.close)
    a = float(x.atr)
    rv = float(x.rsi)

    score = 0
    reasons = []

    checks = [
        (
            price > x.ema20,
            "قیمت بالای EMA20 است",
            "قیمت زیر EMA20 است"
        ),
        (
            x.ema20 > x.ema50,
            "EMA20 بالای EMA50 است",
            "EMA20 زیر EMA50 است"
        ),
        (
            x.ema50 > x.ema200,
            "EMA50 بالای EMA200 است",
            "EMA50 زیر EMA200 است"
        ),
        (
            rv >= 55,
            "RSI متمایل به قدرت خریداران است",
            None
        ),
        (
            rv <= 45,
            "RSI متمایل به قدرت فروشندگان است",
            None
        ),
        (
            x.macd_hist > 0,
            "MACD مثبت است",
            "MACD منفی است"
        )
    ]

    for i, (condition, pos, neg) in enumerate(checks):

        if i == 3:
            if condition:
                score += 1
                reasons.append(pos)
            continue

        if i == 4:
            if condition:
                score -= 1
                reasons.append(pos)
            continue

        if condition:
            score += 1
            reasons.append(pos)

        elif neg:
            score -= 1
            reasons.append(neg)

    support = float(
        d.tail(50).low.min()
    )

    resistance = float(
        d.tail(50).high.max()
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

        if not math.isfinite(float(v)):
            return None

        return round(float(v), 8)

    return {
        "signal": signal,
        "price": rnd(price),
        "entry": rnd(entry),
        "stop_loss": rnd(sl),
        "take_profit_1": rnd(tp1),
        "take_profit_2": rnd(tp2),
        "take_profit_3": rnd(tp3),
        "rsi": round(rv, 2),
        "ema20": rnd(x.ema20),
        "ema50": rnd(x.ema50),
        "ema200": rnd(x.ema200),
        "macd": rnd(x.macd),
        "macd_signal": rnd(x.macd_signal),
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
        "service": "Crypto Analyzer Online",
        "version": "2.0.0"
    }


@app.get("/analyze")
def analyze_market(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h")
):

    df = get_candles(
        symbol,
        interval
    )

    result = analyze(df)

    result["symbol"] = normalize_symbol(symbol)

    result["interval"] = normalize_interval(
        interval
    )

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

<title>
Crypto Analyzer Online
</title>

<style>

body{
margin:0;
background:#07101f;
color:white;
font-family:Tahoma,Arial
}

.wrap{
max-width:720px;
margin:30px auto;
padding:18px
}

.card{
background:#111d32;
padding:28px;
border-radius:26px
}

h1{
direction:ltr;
text-align:left
}

.sub{
font-size:20px;
color:#cbd5e1;
margin-bottom:25px
}

input,select,button{
width:100%;
box-sizing:border-box;
padding:16px;
border:0;
border-radius:14px;
font-size:17px;
margin:8px 0
}

button{
background:#16a34a;
color:white;
font-weight:bold;
cursor:pointer
}

.result{
margin-top:18px;
background:#050d1b;
padding:18px;
border-radius:16px;
line-height:2
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:8px
}

.box{
background:#17243a;
padding:10px;
border-radius:12px
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
>

<label>
تایم‌فریم
</label>

<select id="interval">

<option value="1m">
1 دقیقه
</option>

<option value="5m">
5 دقیقه
</option>

<option value="15m">
15 دقیقه
</option>

<option value="30m">
30 دقیقه
</option>

<option value="1h" selected>
1 ساعت
</option>

<option value="2h">
2 ساعت
</option>

<option value="4h">
4 ساعت
</option>

<option value="6h">
6 ساعت
</option>

<option value="12h">
12 ساعت
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
? "-"
: Number(v).toLocaleString(
"en-US",
{
maximumFractionDigits:8
}
);

function run(){

const s =
document
.getElementById("symbol")
.value
.trim()
.toUpperCase();

const i =
document
.getElementById("interval")
.value;

const b =
document.getElementById("btn");

const o =
document.getElementById("result");

b.disabled = true;

b.textContent =
"⏳ در حال تحلیل...";

o.textContent =
"دریافت داده‌های بازار...";

fetch(
`/analyze?symbol=${encodeURIComponent(s)}&interval=${encodeURIComponent(i)}`
)

.then(async r => {

let d = await r.json();

if(!r.ok)
throw Error(
d.detail || "خطا"
);

return d;

})

.then(d => {

o.innerHTML = `

<h2 style="text-align:center">
${d.signal}
</h2>

<div class="grid">

<div class="box">
قیمت
<br>
${f(d.price)}
</div>

<div class="box">
RSI
<br>
${f(d.rsi)}
</div>

<div class="box">
ورود
<br>
${f(d.entry)}
</div>

<div class="box">
حد ضرر
<br>
${f(d.stop_loss)}
</div>

<div class="box">
TP1
<br>
${f(d.take_profit_1)}
</div>

<div class="box">
TP2
<br>
${f(d.take_profit_2)}
</div>

<div class="box">
TP3
<br>
${f(d.take_profit_3)}
</div>

<div class="box">
امتیاز
<br>
${d.score}
</div>

<div class="box">
حمایت
<br>
${f(d.support)}
</div>

<div class="box">
مقاومت
<br>
${f(d.resistance)}
</div>

</div>

<p>

<b>
دلایل:
</b>

<br>

${d.reasons.join("<br>")}

</p>

<small>
این خروجی تحلیل تکنیکال است و تضمین سود
یا توصیه قطعی معامله نیست.
</small>

`;

})

.catch(e => {

o.textContent =
"❌ خطا: " + e.message;

})

.finally(() => {

b.disabled = false;

b.textContent =
"🔍 تحلیل بازار";

});

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
