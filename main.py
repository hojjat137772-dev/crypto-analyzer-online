from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import requests
import pandas as pd
import math
from datetime import datetime

app = FastAPI(
    title="Iran Stock Analyzer - EasyTrader",
    version="3.0.0"
)

# ============================================================
# منبع داده بازار ایران: TSETMC
# سفارش‌گذاری: از طریق EasyTrader به صورت دستی/تأییدشده
# ============================================================

TSETMC_CDN = "https://cdn.tsetmc.com/api"
EASYTRADER_URL = "https://easytrader.emofid.com"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Android) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*"
})


def clean_symbol(symbol: str) -> str:
    return str(symbol).strip().replace("/", "").replace("-", "").upper()


def fa_to_en(text):
    if text is None:
        return ""
    return str(text).translate(str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789"
    ))


def get_json(url, timeout=20):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"ارتباط با منبع داده بورس برقرار نشد: {e}"
        )
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="پاسخ نامعتبر از منبع داده بورس دریافت شد."
        )


def search_instrument(symbol: str):
    symbol = str(symbol).strip()
    if not symbol:
        raise HTTPException(400, "نماد را وارد کنید.")

    data = get_json(
        f"{TSETMC_CDN}/Instrument/GetInstrumentSearch/{requests.utils.quote(symbol)}"
    )
    items = data.get("instrumentSearch", [])

    if not items:
        raise HTTPException(404, f"نماد «{symbol}» پیدا نشد.")

    # اولویت با تطبیق دقیق نماد
    exact = [
        x for x in items
        if str(x.get("lVal18AFC", "")).strip() == symbol
    ]
    return (exact[0] if exact else items[0]), items


def get_history(ins_code: str):
    data = get_json(
        f"{TSETMC_CDN}/ClosingPrice/GetClosingPriceDailyList/{ins_code}/0"
    )
    rows = data.get("closingPriceDaily", [])

    if not rows:
        raise HTTPException(404, "تاریخچه قیمت این نماد در دسترس نیست.")

    records = []
    for row in rows:
        # endpointهای TSETMC در نسخه‌های مختلف نام‌های نزدیک به هم دارند
        date_raw = (
            row.get("dEven")
            or row.get("deven")
            or row.get("date")
            or row.get("DEven")
        )

        close = row.get("pClosing")
        last = row.get("pDrCotVal")
        high = row.get("priceMax")
        low = row.get("priceMin")
        open_price = row.get("priceFirst")
        volume = row.get("qTotTran5J")
        value = row.get("qTotCap")

        if close is None:
            continue

        records.append({
            "date": str(date_raw) if date_raw is not None else "",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "last": last if last is not None else close,
            "volume": volume,
            "value": value
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise HTTPException(404, "داده کافی برای تحلیل پیدا نشد.")

    for c in ["open", "high", "low", "close", "last", "volume", "value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # اگر high/low/open در بعضی روزها نبود، از close استفاده می‌کنیم
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])
    df["open"] = df["open"].fillna(df["close"])

    df = df.dropna(subset=["close"]).copy()

    # قدیمی -> جدید
    df = df.iloc[::-1].reset_index(drop=True)

    # حذف تاریخ‌های تکراری
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if len(df) < 60:
        raise HTTPException(400, "برای تحلیل تکنیکال حداقل 60 روز داده لازم است.")

    return df


def get_live_info(isin: str):
    url = (
        f"https://webgw.tse.ir/InstrumentProvider/api/v1/"
        f"Instrument/LiveInstrumentByIdQuery/fa"
    )
    try:
        r = session.get(url, params={"InstrumentId": isin}, timeout=15)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}


def get_instrument_info(ins_code: str):
    try:
        data = get_json(
            f"{TSETMC_CDN}/Instrument/GetInstrumentInfo/{ins_code}",
            timeout=15
        )
        return data.get("instrumentInfo", data)
    except Exception:
        return {}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / n, adjust=False, min_periods=n
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / n, adjust=False, min_periods=n
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / n, adjust=False, min_periods=n
    ).mean()


def indicators(df):
    d = df.copy()

    for n in [20, 50, 100, 200]:
        d[f"ema{n}"] = ema(d["close"], n)

    d["rsi"] = rsi(d["close"])
    d["ema12"] = ema(d["close"], 12)
    d["ema26"] = ema(d["close"], 26)
    d["macd"] = d["ema12"] - d["ema26"]
    d["macd_signal"] = ema(d["macd"], 9)
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["atr"] = atr(d)

    # میانگین حجم
    d["vol20"] = d["volume"].rolling(20).mean()

    return d.dropna(subset=[
        "ema20", "ema50", "ema200", "rsi", "macd", "macd_signal", "atr"
    ]).reset_index(drop=True)


def round_num(v, digits=2):
    try:
        if v is None or not math.isfinite(float(v)):
            return None
        return round(float(v), digits)
    except Exception:
        return None


def analyze_stock(df):
    d = indicators(df)
    if len(d) < 30:
        raise HTTPException(400, "داده کافی برای تحلیل تکنیکال وجود ندارد.")

    x = d.iloc[-1]
    price = float(x["close"])
    atr_value = float(x["atr"])
    rv = float(x["rsi"])

    score = 0
    reasons = []

    # روند
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

    # RSI
    if 55 <= rv < 70:
        score += 1
        reasons.append("RSI در محدوده قدرت خریداران است.")
    elif rv > 75:
        score -= 1
        reasons.append("RSI بسیار بالا و مستعد اصلاح است.")
    elif rv < 30:
        score += 1
        reasons.append("RSI در محدوده اشباع فروش است.")
    elif rv < 45:
        score -= 1
        reasons.append("RSI ضعیف است.")

    # MACD
    if x["macd_hist"] > 0:
        score += 1
        reasons.append("هیستوگرام MACD مثبت است.")
    else:
        score -= 1
        reasons.append("هیستوگرام MACD منفی است.")

    # حجم
    if pd.notna(x["vol20"]) and x["volume"] > x["vol20"] * 1.2:
        if x["close"] >= x["open"]:
            score += 1
            reasons.append("حجم معاملات بالاتر از میانگین و کندل مثبت است.")
        else:
            score -= 1
            reasons.append("حجم بالا همراه با فشار فروش دیده می‌شود.")

    support = float(d.tail(60)["low"].min())
    resistance = float(d.tail(60)["high"].max())

    # سطوح نزدیک‌تر برای ورود/خروج
    recent_low = float(d.tail(20)["low"].min())
    recent_high = float(d.tail(20)["high"].max())

    if score >= 4:
        signal = "BUY / LONG BIAS"
        entry = price

        # حد ضرر ترکیبی: زیر حمایت اخیر یا 1.5 ATR
        sl = min(recent_low * 0.985, price - 1.5 * atr_value)
        if sl >= entry:
            sl = entry - 1.5 * atr_value

        risk = max(entry - sl, atr_value * 0.5)
        tp1 = entry + 1.5 * risk
        tp2 = entry + 2.5 * risk
        tp3 = entry + 3.5 * risk

    elif score <= -3:
        signal = "SELL / EXIT BIAS"
        entry = price

        sl = max(recent_high * 1.015, price + 1.5 * atr_value)
        if sl <= entry:
            sl = entry + 1.5 * atr_value

        risk = max(sl - entry, atr_value * 0.5)
        tp1 = entry - 1.5 * risk
        tp2 = entry - 2.5 * risk
        tp3 = entry - 3.5 * risk

    else:
        signal = "WAIT"
        entry = price
        sl = tp1 = tp2 = tp3 = None
        risk = None

    rr = None
    if risk and risk > 0:
        rr = abs(tp2 - entry) / risk

    return {
        "signal": signal,
        "price": round_num(price),
        "entry": round_num(entry),
        "stop_loss": round_num(sl),
        "take_profit_1": round_num(tp1),
        "take_profit_2": round_num(tp2),
        "take_profit_3": round_num(tp3),
        "risk_reward_tp2": round_num(rr, 2),
        "rsi": round_num(rv, 2),
        "ema20": round_num(x["ema20"]),
        "ema50": round_num(x["ema50"]),
        "ema200": round_num(x["ema200"]),
        "macd": round_num(x["macd"], 4),
        "macd_signal": round_num(x["macd_signal"], 4),
        "atr": round_num(atr_value),
        "support": round_num(support),
        "resistance": round_num(resistance),
        "recent_support": round_num(recent_low),
        "recent_resistance": round_num(recent_high),
        "score": int(score),
        "reasons": reasons
    }


def backtest(df, lookback=100):
    d = indicators(df).copy()
    if len(d) < lookback + 10:
        return {
            "trades": 0,
            "win_rate": None,
            "message": "داده کافی برای بک‌تست وجود ندارد."
        }

    # بک‌تست ساده و غیرتضمینی:
    # ورود وقتی امتیاز >=4، حدضرر 1.5 ATR و هدف 2R.
    wins = 0
    losses = 0
    trades = 0

    start = max(210, len(d) - lookback)

    for i in range(start, len(d) - 5):
        row = d.iloc[i]
        price = float(row["close"])
        a = float(row["atr"])

        score = 0
        score += 1 if price > row["ema20"] else -1
        score += 1 if row["ema20"] > row["ema50"] else -1
        score += 1 if row["ema50"] > row["ema200"] else -1
        score += 1 if row["rsi"] >= 55 else (-1 if row["rsi"] <= 45 else 0)
        score += 1 if row["macd_hist"] > 0 else -1

        if score < 4:
            continue

        entry = price
        sl = entry - 1.5 * a
        tp = entry + 3.0 * a

        trades += 1

        future = d.iloc[i + 1:i + 6]
        result = None

        for _, f in future.iterrows():
            if f["low"] <= sl:
                result = "loss"
                break
            if f["high"] >= tp:
                result = "win"
                break

        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1

    decided = wins + losses
    win_rate = (wins / decided * 100) if decided else None

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round_num(win_rate, 2),
        "lookback": lookback,
        "message": "بک‌تست ساده است و هزینه معاملات، صف، دامنه نوسان و لغزش قیمت را کامل مدل نمی‌کند."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Iran Stock Analyzer / EasyTrader",
        "version": "3.0.0",
        "data_source": "TSETMC",
        "broker": "Mofid EasyTrader"
    }


@app.get("/symbols")
def symbols(q: str = Query("استیل")):
    _, items = search_instrument(q)
    return {
        "query": q,
        "results": items[:20]
    }


@app.get("/analyze")
def analyze_market(symbol: str = Query("استیل")):
    item, _ = search_instrument(symbol)

    ins_code = item.get("insCode")
    isin = item.get("cIsin")
    ticker = item.get("lVal18AFC", symbol)
    company = item.get("lVal30", "")

    if not ins_code:
        raise HTTPException(404, "شناسه نماد پیدا نشد.")

    df = get_history(ins_code)
    result = analyze_stock(df)

    fundamental = get_instrument_info(ins_code)

    return {
        **result,
        "symbol": ticker,
        "company": company,
        "isin": isin,
        "ins_code": ins_code,
        "fundamental_raw": fundamental,
        "easytrader_url": EASYTRADER_URL
    }


@app.get("/backtest")
def run_backtest(
    symbol: str = Query("استیل"),
    lookback: int = Query(100, ge=30, le=1000)
):
    item, _ = search_instrument(symbol)
    ins_code = item.get("insCode")

    if not ins_code:
        raise HTTPException(404, "شناسه نماد پیدا نشد.")

    df = get_history(ins_code)
    result = backtest(df, lookback)

    return {
        "symbol": item.get("lVal18AFC", symbol),
        **result
    }


HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحلیل‌گر بورس ایران | ایزی‌تریدر</title>
<style>
body{margin:0;background:#07101f;color:#fff;font-family:Tahoma,Arial,sans-serif}
.wrap{max-width:760px;margin:20px auto;padding:14px}
.card{background:#111d32;padding:22px;border-radius:24px}
h1{margin-top:0}
.sub{color:#b9c4d6;margin-bottom:20px}
input,button{width:100%;box-sizing:border-box;padding:15px;border:0;border-radius:13px;font-size:17px;margin:7px 0}
input{background:#0b1424;color:#fff;border:1px solid #30405a}
button{background:#16a34a;color:#fff;font-weight:bold}
.secondary{background:#334155}
.result{margin-top:18px;background:#050d1b;padding:16px;border-radius:16px;line-height:2}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.box{background:#17243a;padding:11px;border-radius:12px}
.good{color:#39d98a}.bad{color:#ff7187}.warn{color:#ffd166}
a{display:block;text-align:center;color:#fff;text-decoration:none}
.small{font-size:12px;color:#91a0b7}
@media(max-width:600px){.wrap{padding:8px}.card{padding:16px}}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h1>📈 تحلیل‌گر بورس ایران</h1>
<div class="sub">اتصال داده بازار ایران + تحلیل تکنیکال + ورود/حدضرر/حدسود + بک‌تست</div>

<label>نماد</label>
<input id="symbol" value="استیل" placeholder="مثلاً استیل، فولاد، شستا">

<button onclick="analyze()">🔍 تحلیل نماد</button>
<button class="secondary" onclick="backtest()">🧪 بک‌تست</button>
<a href="https://easytrader.emofid.com" target="_blank">
<button class="secondary">💼 باز کردن ایزی‌تریدر</button>
</a>

<div id="result" class="result">آماده تحلیل...</div>
</div>
</div>

<script>
const f=v=>v==null?"—":Number(v).toLocaleString("fa-IR",{maximumFractionDigits:2});

function analyze(){
 const s=document.getElementById("symbol").value.trim();
 const o=document.getElementById("result");
 o.innerHTML="⏳ در حال دریافت اطلاعات نماد و تحلیل...";
 fetch(`/analyze?symbol=${encodeURIComponent(s)}`)
 .then(async r=>{let d=await r.json();if(!r.ok)throw Error(d.detail||"خطا");return d})
 .then(d=>{
   let cls=d.signal.includes("BUY")?"good":(d.signal.includes("SELL")?"bad":"warn");
   o.innerHTML=`
   <h2 class="${cls}">${d.signal}</h2>
   <b>${d.symbol}</b> — ${d.company||""}
   <div class="grid">
    <div class="box">قیمت<br>${f(d.price)}</div>
    <div class="box">امتیاز<br>${f(d.score)}</div>
    <div class="box">RSI<br>${f(d.rsi)}</div>
    <div class="box">ورود<br>${f(d.entry)}</div>
    <div class="box">حد ضرر<br>${f(d.stop_loss)}</div>
    <div class="box">TP1<br>${f(d.take_profit_1)}</div>
    <div class="box">TP2<br>${f(d.take_profit_2)}</div>
    <div class="box">TP3<br>${f(d.take_profit_3)}</div>
    <div class="box">حمایت<br>${f(d.support)}</div>
    <div class="box">مقاومت<br>${f(d.resistance)}</div>
    <div class="box">EMA20<br>${f(d.ema20)}</div>
    <div class="box">EMA50<br>${f(d.ema50)}</div>
   </div>
   <p><b>دلایل:</b><br>${d.reasons.join("<br>")}</p>
   <p class="small">منبع داده: TSETMC — اجرای سفارش در این نسخه به‌صورت مستقیم خودکار نمی‌شود؛ دکمه ایزی‌تریدر برای ورود به سامانه مفید است.</p>`;
 })
 .catch(e=>o.innerHTML="❌ خطا: "+e.message);
}

function backtest(){
 const s=document.getElementById("symbol").value.trim();
 const o=document.getElementById("result");
 o.innerHTML="⏳ در حال اجرای بک‌تست...";
 fetch(`/backtest?symbol=${encodeURIComponent(s)}&lookback=100`)
 .then(async r=>{let d=await r.json();if(!r.ok)throw Error(d.detail||"خطا");return d})
 .then(d=>{
   o.innerHTML=`
   <h2>🧪 نتیجه بک‌تست ${d.symbol}</h2>
   <div class="grid">
    <div class="box">تعداد معاملات<br>${f(d.trades)}</div>
    <div class="box">برد<br>${f(d.wins)}</div>
    <div class="box">باخت<br>${f(d.losses)}</div>
    <div class="box">Win Rate<br>${f(d.win_rate)}٪</div>
   </div>
   <p class="small">${d.message}</p>`;
 })
 .catch(e=>o.innerHTML="❌ خطا در بک‌تست: "+e.message);
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
