from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import os
import math
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests
import pandas as pd

# ============================================================
# تحلیل‌گر بورس ایران - نسخه جدید
# منبع داده: Tindex API
# سفارش‌گذاری: فقط از طریق لینک EasyTrader
# ============================================================

app = FastAPI(
    title="تحلیل‌گر بورس ایران",
    version="4.0.0"
)

TINDEX_BASE = "https://tindex.app/api/public"
TINDEX_TOKEN = os.getenv("TINDEX_API_TOKEN", "").strip()
EASYTRADER_URL = "https://easytrader.emofid.com"

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
})


def fa_to_en(value):
    if value is None:
        return ""
    return str(value).translate(str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789"
    ))


def num(value):
    try:
        value = fa_to_en(value)
        return float(value)
    except Exception:
        return None


def round_num(value, digits=2):
    try:
        if value is None or not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def tindex_get(path, params=None, timeout=15):
    if not TINDEX_TOKEN:
        raise HTTPException(
            500,
            "متغیر محیطی TINDEX_API_TOKEN در Render تنظیم نشده است."
        )

    url = f"{TINDEX_BASE}{path}"

    try:
        r = session.get(
            url,
            params=params or {},
            timeout=(5, timeout),
            allow_redirects=True
        )
    except requests.RequestException as e:
        raise HTTPException(
            502,
            f"ارتباط با Tindex برقرار نشد: {e}"
        )

    if r.status_code == 401:
        raise HTTPException(502, "توکن Tindex نامعتبر یا فاقد دسترسی API است.")
    if r.status_code == 403:
        raise HTTPException(502, "دسترسی API Tindex برای این درخواست مجاز نیست.")
    if r.status_code == 429:
        retry = r.headers.get("Retry-After", "")
        raise HTTPException(
            429,
            f"سقف درخواست Tindex پر شده است. "
            f"لطفاً کمی بعد دوباره تلاش کنید. {retry}"
        )

    try:
        payload = r.json()
    except ValueError:
        raise HTTPException(502, "پاسخ نامعتبر از Tindex دریافت شد.")

    if r.status_code >= 400:
        raise HTTPException(
            502,
            payload.get("message", "خطا از سرویس Tindex")
            if isinstance(payload, dict) else "خطا از سرویس Tindex"
        )

    if isinstance(payload, dict) and payload.get("success") is False:
        raise HTTPException(
            502,
            payload.get("message", "Tindex درخواست را رد کرد.")
        )

    return payload


def _rows_from_stock_response(payload):
    """
    Tindex stock screener پاسخ را به شکل data/rows یا data/list برمی‌گرداند.
    این تابع چند شکل متداول را پشتیبانی می‌کند.
    """
    data = payload.get("data") if isinstance(payload, dict) else None

    if isinstance(data, dict):
        rows = data.get("rows")
        if isinstance(rows, list):
            return rows

        items = data.get("items")
        if isinstance(items, list):
            return items

    if isinstance(data, list):
        return data

    if isinstance(payload, dict):
        for key in ("rows", "items", "stocks"):
            if isinstance(payload.get(key), list):
                return payload[key]

    return []


def search_stock(symbol):
    symbol = str(symbol).strip()
    if not symbol:
        raise HTTPException(400, "نماد را وارد کنید.")

    # Tindex تمام نمادهای بورس تهران را زیر stock-energy قرار داده است.
    payload = tindex_get(
        "/stocks/by-category/stock-energy",
        params={
            "search": symbol,
            "page": 1,
            "per_page": 50,
            "lang": "fa",
        },
        timeout=20,
    )

    rows = _rows_from_stock_response(payload)

    if not rows:
        raise HTTPException(
            404,
            f"نماد «{symbol}» در Tindex پیدا نشد."
        )

    def text(x, *keys):
        for key in keys:
            value = x.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    normalized = []

    for x in rows:
        ticker = text(x, "ticker", "symbol", "code", "name_fa")
        name = text(x, "name", "company", "company_name", "title")
        slug = text(x, "slug", "ticker_slug")

        normalized.append({
            "slug": slug,
            "ticker": ticker,
            "name": name,
            "company": name,
            "price": num(x.get("price") or x.get("last_price")),
            "change": num(x.get("change") or x.get("change_percent")),
            "volume": num(x.get("volume")),
            "value": num(x.get("turnover") or x.get("trade_value")),
            "raw": x,
        })

    wanted = symbol.strip()

    exact = [
        x for x in normalized
        if x["ticker"].strip() == wanted
    ]

    if exact:
        return exact[0], normalized

    exact_slug_name = [
        x for x in normalized
        if wanted in (x["name"] or "")
    ]

    if exact_slug_name:
        return exact_slug_name[0], normalized

    # اگر API فیلتر جستجو را نادیده گرفته باشد، فقط وقتی یک نتیجه داریم
    # همان نتیجه را قبول می‌کنیم.
    if len(normalized) == 1:
        return normalized[0], normalized

    raise HTTPException(
        404,
        f"نماد «{symbol}» دقیقاً در نتایج Tindex پیدا نشد."
    )


def decode_tindex_days(t_values):
    """
    طبق مستندات Tindex:
    عنصر اول t مطلق است و از عنصر دوم به بعد مقدارها delta هستند.
    """
    if not t_values:
        return []

    out = []
    current = None

    for i, raw in enumerate(t_values):
        try:
            value = int(raw)
        except Exception:
            out.append(None)
            continue

        if i == 0:
            current = value
        else:
            current = current + value

        out.append(current)

    return out


def history_from_tindex(slug):
    if not slug:
        raise HTTPException(404, "شناسه Tindex نماد پیدا نشد.")

    # سه ماه حدود 90 روز داده می‌دهد و برای تحلیل فعلی کافی است.
    payload = tindex_get(
        f"/indicators/{quote(slug, safe='')}/candles",
        params={
            "range": "3m",
            "interval": "daily",
            "lang": "fa",
        },
        timeout=20,
    )

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise HTTPException(502, "ساختار داده شمعی Tindex نامعتبر است.")

    t = decode_tindex_days(data.get("t", []))
    opens = data.get("o", [])
    highs = data.get("h", [])
    lows = data.get("l", [])
    closes = data.get("c", [])

    n = min(len(t), len(opens), len(highs), len(lows), len(closes))

    if n < 60:
        raise HTTPException(
            400,
            f"داده کافی برای تحلیل وجود ندارد؛ فقط {n} روز داده دریافت شد."
        )

    records = []

    for i in range(n):
        if t[i] is None:
            continue

        try:
            date_value = (
                datetime(1970, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=int(t[i]))
            ).date().isoformat()
        except Exception:
            continue

        o = num(opens[i])
        h = num(highs[i])
        l = num(lows[i])
        c = num(closes[i])

        if c is None:
            continue

        records.append({
            "date": date_value,
            "open": o if o is not None else c,
            "high": h if h is not None else c,
            "low": l if l is not None else c,
            "close": c,
            "last": c,
            "volume": 0,
            "value": 0,
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise HTTPException(400, "تاریخچه قابل استفاده دریافت نشد.")

    for col in ["open", "high", "low", "close", "last", "volume", "value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])

    df = (
        df.dropna(subset=["close"])
          .drop_duplicates(subset=["date"], keep="last")
          .sort_values("date")
          .reset_index(drop=True)
    )

    if len(df) < 60:
        raise HTTPException(
            400,
            f"برای تحلیل تکنیکال حداقل 60 روز داده لازم است؛ "
            f"دریافتی: {len(df)} روز."
        )

    return df


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


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

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def build_indicators(df):
    d = df.copy()

    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)

    d["rsi"] = rsi(d["close"])

    d["ema12"] = ema(d["close"], 12)
    d["ema26"] = ema(d["close"], 26)
    d["macd"] = d["ema12"] - d["ema26"]
    d["macd_signal"] = ema(d["macd"], 9)
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    d["atr"] = atr(d)

    return d.dropna(
        subset=[
            "ema20",
            "ema50",
            "ema200",
            "rsi",
            "macd",
            "macd_signal",
            "atr",
        ]
    ).reset_index(drop=True)


def analyze_stock(df):
    d = build_indicators(df)

    if len(d) < 20:
        raise HTTPException(400, "داده کافی برای تحلیل تکنیکال وجود ندارد.")

    x = d.iloc[-1]

    price = float(x["close"])
    atr_value = float(x["atr"])
    rsi_value = float(x["rsi"])

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

    # اگر کمتر از 200 روز داده داشته باشیم، EMA200 از خود داده موجود ساخته می‌شود.
    if x["ema50"] > x["ema200"]:
        score += 1
        reasons.append("EMA50 بالای EMA200 است.")
    else:
        score -= 1
        reasons.append("EMA50 زیر EMA200 است.")

    if 55 <= rsi_value < 70:
        score += 1
        reasons.append("RSI در محدوده قدرت خریداران است.")
    elif rsi_value >= 75:
        score -= 1
        reasons.append("RSI بالا و مستعد اصلاح است.")
    elif rsi_value < 30:
        score += 1
        reasons.append("RSI در محدوده اشباع فروش است.")
    elif rsi_value < 45:
        score -= 1
        reasons.append("RSI ضعیف است.")

    if x["macd_hist"] > 0:
        score += 1
        reasons.append("MACD مثبت است.")
    else:
        score -= 1
        reasons.append("MACD منفی است.")

    support = float(d.tail(40)["low"].min())
    resistance = float(d.tail(40)["high"].max())

    recent_support = float(d.tail(20)["low"].min())
    recent_resistance = float(d.tail(20)["high"].max())

    # این بخش «پیشنهاد سرمایه‌گذاری قطعی» نیست؛
    # صرفاً خروجی الگوریتمی برای کمک به بررسی معامله است.
    if score >= 4:
        signal = "BUY SETUP"
        entry = price

        stop = min(
            recent_support * 0.985,
            price - 1.5 * atr_value
        )

        if stop >= entry:
            stop = entry - 1.5 * atr_value

        risk = max(entry - stop, atr_value * 0.5)

        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        tp3 = entry + 3.5 * risk

    elif score <= -3:
        signal = "EXIT / WEAK SETUP"
        entry = price

        stop = max(
            recent_resistance * 1.015,
            price + 1.5 * atr_value
        )

        if stop <= entry:
            stop = entry + 1.5 * atr_value

        risk = max(stop - entry, atr_value * 0.5)

        tp1 = entry - 1.5 * risk
        tp2 = entry - 2.5 * risk
        tp3 = entry - 3.5 * risk

    else:
        signal = "WAIT"
        entry = price
        stop = None
        tp1 = None
        tp2 = None
        tp3 = None
        risk = None

    rr = None
    if risk:
        rr = abs(tp2 - entry) / risk

    return {
        "signal": signal,
        "price": round_num(price),
        "entry": round_num(entry),
        "stop_loss": round_num(stop),
        "take_profit_1": round_num(tp1),
        "take_profit_2": round_num(tp2),
        "take_profit_3": round_num(tp3),
        "risk_reward_tp2": round_num(rr),
        "score": int(score),
        "rsi": round_num(rsi_value),
        "ema20": round_num(x["ema20"]),
        "ema50": round_num(x["ema50"]),
        "ema200": round_num(x["ema200"]),
        "macd": round_num(x["macd"], 4),
        "macd_signal": round_num(x["macd_signal"], 4),
        "atr": round_num(atr_value),
        "support": round_num(support),
        "resistance": round_num(resistance),
        "recent_support": round_num(recent_support),
        "recent_resistance": round_num(recent_resistance),
        "reasons": reasons,
        "data_days": len(df),
    }


def score_at_row(d, i):
    x = d.iloc[i]
    score = 0

    score += 1 if x["close"] > x["ema20"] else -1
    score += 1 if x["ema20"] > x["ema50"] else -1
    score += 1 if x["ema50"] > x["ema200"] else -1

    if x["rsi"] >= 55:
        score += 1
    elif x["rsi"] <= 45:
        score -= 1

    score += 1 if x["macd_hist"] > 0 else -1

    return score


def backtest(df, lookback=60):
    d = build_indicators(df)

    if len(d) < 40:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "message": "داده کافی برای بک‌تست وجود ندارد."
        }

    start = max(0, len(d) - lookback)

    wins = 0
    losses = 0
    trades = 0

    for i in range(start, len(d) - 5):
        row = d.iloc[i]

        if score_at_row(d, i) < 4:
            continue

        entry = float(row["close"])
        a = float(row["atr"])

        stop = entry - 1.5 * a
        target = entry + 3.0 * a

        trades += 1
        outcome = None

        for j in range(i + 1, min(i + 6, len(d))):
            future = d.iloc[j]

            if float(future["low"]) <= stop:
                outcome = "loss"
                break

            if float(future["high"]) >= target:
                outcome = "win"
                break

        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1

    decided = wins + losses
    win_rate = (wins / decided * 100) if decided else None

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round_num(win_rate),
        "lookback": lookback,
        "message": (
            "بک‌تست الگوریتمی است و کارمزد، صف خرید/فروش، "
            "لغزش قیمت و محدودیت نقدشوندگی را کامل شبیه‌سازی نمی‌کند."
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0.0",
        "data_source": "Tindex API",
        "token_configured": bool(TINDEX_TOKEN),
        "broker_link": EASYTRADER_URL,
    }


@app.get("/symbols")
def symbols(q: str = Query("استیل")):
    item, results = search_stock(q)

    return {
        "query": q,
        "selected": item,
        "results": results[:20],
    }


@app.get("/analyze")
def analyze_market(symbol: str = Query("استیل")):
    item, _ = search_stock(symbol)

    slug = item.get("slug")
    if not slug:
        raise HTTPException(
            502,
            "Tindex برای این نماد slug برنگرداند."
        )

    df = history_from_tindex(slug)
    result = analyze_stock(df)

    return {
        **result,
        "symbol": item.get("ticker") or symbol,
        "company": item.get("company") or item.get("name") or "",
        "tindex_slug": slug,
        "data_source": "Tindex",
        "easytrader_url": EASYTRADER_URL,
    }


@app.get("/backtest")
def run_backtest(
    symbol: str = Query("استیل"),
    lookback: int = Query(60, ge=30, le=90),
):
    item, _ = search_stock(symbol)

    slug = item.get("slug")
    if not slug:
        raise HTTPException(502, "Tindex slug برای نماد پیدا نشد.")

    df = history_from_tindex(slug)
    result = backtest(df, lookback)

    return {
        "symbol": item.get("ticker") or symbol,
        "tindex_slug": slug,
        **result,
    }


HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحلیل‌گر بورس ایران</title>
<style>
body{
 margin:0;background:#07101f;color:#fff;
 font-family:Tahoma,Arial,sans-serif
}
.wrap{max-width:760px;margin:18px auto;padding:10px}
.card{
 background:#111d32;padding:20px;border-radius:22px;
 box-shadow:0 8px 30px rgba(0,0,0,.25)
}
h1{margin-top:0}
.sub{color:#aebbd0;line-height:1.9}
input,button{
 width:100%;box-sizing:border-box;padding:14px;
 border-radius:13px;font-size:17px;margin:6px 0
}
input{
 background:#091426;color:#fff;border:1px solid #30405a
}
button{
 border:0;background:#16a34a;color:#fff;font-weight:bold
}
.secondary{background:#334155}
.result{
 margin-top:16px;background:#050d1b;padding:15px;
 border-radius:16px;line-height:2
}
.grid{
 display:grid;grid-template-columns:1fr 1fr;gap:9px
}
.box{
 background:#17243a;padding:10px;border-radius:12px
}
.good{color:#39d98a}
.bad{color:#ff7187}
.warn{color:#ffd166}
.small{font-size:12px;color:#91a0b7}
a{display:block;text-decoration:none}
@media(max-width:600px){
 .card{padding:15px}
 .grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h1>📈 تحلیل‌گر بورس ایران</h1>
<div class="sub">
داده Tindex + تحلیل تکنیکال + محدوده ورود + حد ضرر + حد سود + بک‌تست
</div>

<input id="symbol" value="استیل"
 placeholder="مثلاً استیل، فولاد، شستا">

<button onclick="analyze()">🔍 تحلیل نماد</button>
<button class="secondary" onclick="backtest()">🧪 بک‌تست</button>

<a href="https://easytrader.emofid.com" target="_blank">
<button class="secondary">💼 باز کردن ایزی‌تریدر</button>
</a>

<div id="result" class="result">
آماده تحلیل...
</div>
</div>
</div>

<script>
const f = v => {
 if(v === null || v === undefined) return "—";
 const n = Number(v);
 if(Number.isNaN(n)) return "—";
 return n.toLocaleString("fa-IR",{maximumFractionDigits:2});
};

function showError(o,e){
 o.innerHTML = "❌ خطا: " + e.message;
}

function analyze(){
 const s=document.getElementById("symbol").value.trim();
 const o=document.getElementById("result");

 if(!s){
   o.innerHTML="❌ نماد را وارد کنید.";
   return;
 }

 o.innerHTML="⏳ در حال دریافت داده Tindex و تحلیل...";

 fetch("/analyze?symbol="+encodeURIComponent(s))
 .then(async r=>{
   const d=await r.json();
   if(!r.ok) throw new Error(d.detail || "خطا");
   return d;
 })
 .then(d=>{
   const cls =
     d.signal === "BUY SETUP" ? "good" :
     d.signal === "EXIT / WEAK SETUP" ? "bad" : "warn";

   o.innerHTML = `
   <h2 class="${cls}">${d.signal}</h2>
   <b>${d.symbol}</b> — ${d.company || ""}

   <div class="grid">
    <div class="box">قیمت<br>${f(d.price)}</div>
    <div class="box">امتیاز<br>${f(d.score)}</div>
    <div class="box">RSI<br>${f(d.rsi)}</div>
    <div class="box">EMA20<br>${f(d.ema20)}</div>
    <div class="box">EMA50<br>${f(d.ema50)}</div>
    <div class="box">EMA200<br>${f(d.ema200)}</div>
    <div class="box">ورود<br>${f(d.entry)}</div>
    <div class="box">حد ضرر<br>${f(d.stop_loss)}</div>
    <div class="box">TP1<br>${f(d.take_profit_1)}</div>
    <div class="box">TP2<br>${f(d.take_profit_2)}</div>
    <div class="box">TP3<br>${f(d.take_profit_3)}</div>
    <div class="box">R/R<br>${f(d.risk_reward_tp2)}</div>
    <div class="box">حمایت<br>${f(d.support)}</div>
    <div class="box">مقاومت<br>${f(d.resistance)}</div>
   </div>

   <p><b>دلایل:</b><br>${d.reasons.join("<br>")}</p>

   <p class="small">
   ${d.data_days} روز داده برای تحلیل استفاده شد.
   منبع: Tindex.
   خروجی الگوریتمی است و تضمین سود نیست.
   </p>`;
 })
 .catch(e=>showError(o,e));
}

function backtest(){
 const s=document.getElementById("symbol").value.trim();
 const o=document.getElementById("result");

 if(!s){
   o.innerHTML="❌ نماد را وارد کنید.";
   return;
 }

 o.innerHTML="⏳ در حال اجرای بک‌تست...";

 fetch("/backtest?symbol="+encodeURIComponent(s)+"&lookback=60")
 .then(async r=>{
   const d=await r.json();
   if(!r.ok) throw new Error(d.detail || "خطا");
   return d;
 })
 .then(d=>{
   o.innerHTML=`
   <h2>🧪 بک‌تست ${d.symbol}</h2>
   <div class="grid">
    <div class="box">معاملات<br>${f(d.trades)}</div>
    <div class="box">برد<br>${f(d.wins)}</div>
    <div class="box">باخت<br>${f(d.losses)}</div>
    <div class="box">Win Rate<br>${f(d.win_rate)}٪</div>
   </div>
   <p class="small">${d.message}</p>`;
 })
 .catch(e=>showError(o,e));
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
