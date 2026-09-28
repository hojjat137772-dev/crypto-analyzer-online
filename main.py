import os
import re
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION = "5.2.0"
TINDEX_BASE = "https://tindex.app"
TINDEX_TOKEN = os.getenv("TINDEX_API_TOKEN", "").strip()
EASYTRADER_URL = "https://easytrader.emofid.com"

app = FastAPI(title="تحلیل‌گر بورس ایران", version=APP_VERSION)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})

# حافظه موقت برای کم کردن تعداد درخواست‌ها به Tindex
CACHE = {}
CACHE_TTL = 300  # 5 دقیقه


def fa_to_en_digits(value):
    if value is None:
        return ""
    s = str(value)
    table = str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789",
    )
    return s.translate(table)


def clean_number(value):
    s = fa_to_en_digits(value)
    s = s.replace(",", "").replace("٬", "").replace(" ", "").strip()
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in {"-", "."}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_header(text):
    s = fa_to_en_digits(text).strip().lower()
    s = s.replace(" ", "").replace("_", "")
    return s


def fetch_history_page(symbol, page=1):
    encoded = quote(symbol.strip(), safe="")
    url = f"{TINDEX_BASE}/en/stocks/{encoded}/history/"
    if page > 1:
        url += f"?page={page}"

    try:
        r = SESSION.get(url, timeout=20)
    except requests.RequestException as exc:
        raise RuntimeError(f"ارتباط با Tindex برقرار نشد: {exc}") from exc

    if r.status_code == 404:
        raise RuntimeError(f"نماد «{symbol}» در Tindex پیدا نشد.")
    if r.status_code != 200:
        raise RuntimeError(f"Tindex با کد {r.status_code} پاسخ داد.")

    soup = BeautifulSoup(r.text, "html.parser")

    # جدول تاریخچه را با نام ستون‌ها پیدا می‌کنیم.
    wanted = {"date", "open", "high", "low", "close"}
    target = None

    for table in soup.find_all("table"):
        headers = []
        first_row = table.find("tr")
        if first_row:
            headers = [normalize_header(x.get_text(" ", strip=True))
                       for x in first_row.find_all(["th", "td"])]
        if wanted.issubset(set(headers)):
            target = table
            break

    if target is None:
        # گاهی ساختار صفحه تغییر می‌کند؛ متن خطای قابل فهم برمی‌گردانیم.
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if "stock" not in title.lower() and "history" not in title.lower():
            raise RuntimeError(
                f"صفحه تاریخچه نماد «{symbol}» قابل خواندن نیست؛ "
                "ممکن است ساختار Tindex تغییر کرده باشد."
            )
        return []

    rows = []
    tr_list = target.find_all("tr")
    for tr in tr_list[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 5:
            continue

        # ستون‌ها بر اساس هدر جدول، نه جای ثابت، خوانده می‌شوند.
        header_cells = [
            normalize_header(x.get_text(" ", strip=True))
            for x in tr_list[0].find_all(["th", "td"])
        ]
        row_map = dict(zip(header_cells, cells))

        date_text = row_map.get("date", "")
        op = clean_number(row_map.get("open"))
        hi = clean_number(row_map.get("high"))
        lo = clean_number(row_map.get("low"))
        cl = clean_number(row_map.get("close"))

        if not date_text or pd.isna(cl):
            continue

        try:
            dt = pd.to_datetime(date_text, dayfirst=False, errors="coerce")
        except Exception:
            dt = pd.NaT

        if pd.isna(dt):
            continue

        rows.append({
            "date": dt,
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
        })

    return rows


def get_history(symbol, pages=25):
    key = symbol.strip()
    now = time.time()

    cached = CACHE.get(key)
    if cached and now - cached["time"] < CACHE_TTL:
        return cached["df"].copy()

    all_rows = []
    seen_dates = set()

    for page in range(1, pages + 1):
        rows = fetch_history_page(key, page)
        if not rows:
            break

        new_count = 0
        for row in rows:
            d = row["date"]
            if d not in seen_dates:
                seen_dates.add(d)
                all_rows.append(row)
                new_count += 1

        if new_count == 0:
            break

    if len(all_rows) < 20:
        raise RuntimeError(
            f"برای «{symbol}» داده تاریخی کافی دریافت نشد. "
            f"تعداد رکورد دریافت‌شده: {len(all_rows)}"
        )

    df = pd.DataFrame(all_rows)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    CACHE[key] = {"time": now, "df": df.copy()}
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def support_resistance(df, window=30):
    recent = df.tail(min(window, len(df)))
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance


def build_signal(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    x["atr"] = atr(x, 14)
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x["close"])

    last = x.iloc[-1]
    price = float(last["close"])
    atr_v = float(last["atr"]) if pd.notna(last["atr"]) else price * 0.03

    support, resistance = support_resistance(x, 30)

    score = 0
    reasons = []

    if price > last["ema20"]:
        score += 1
        reasons.append("قیمت بالاتر از EMA20 است")
    else:
        score -= 1
        reasons.append("قیمت پایین‌تر از EMA20 است")

    if price > last["ema50"]:
        score += 1
        reasons.append("قیمت بالاتر از EMA50 است")
    else:
        score -= 1
        reasons.append("قیمت پایین‌تر از EMA50 است")

    if price > last["ema200"]:
        score += 1
        reasons.append("قیمت بالاتر از EMA200 است")
    else:
        score -= 1
        reasons.append("قیمت پایین‌تر از EMA200 است")

    if last["rsi"] >= 55:
        score += 1
        reasons.append("RSI متمایل به قدرت خرید است")
    elif last["rsi"] <= 45:
        score -= 1
        reasons.append("RSI متمایل به ضعف است")

    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD بالاتر از خط سیگنال است")
    else:
        score -= 1
        reasons.append("MACD پایین‌تر از خط سیگنال است")

    if score >= 3:
        signal = "خرید / بررسی ورود"
        side = "LONG"
        entry = price
        stop = max(support, price - 1.8 * atr_v)
        if stop >= entry:
            stop = price - 1.8 * atr_v
        risk = max(entry - stop, atr_v * 0.5)
        tp1 = entry + 1.0 * risk
        tp2 = entry + 2.0 * risk
        tp3 = entry + 3.0 * risk
    elif score <= -3:
        signal = "فروش / اجتناب از ورود"
        side = "SHORT"
        entry = price
        stop = min(resistance, price + 1.8 * atr_v)
        if stop <= entry:
            stop = price + 1.8 * atr_v
        risk = max(stop - entry, atr_v * 0.5)
        tp1 = entry - 1.0 * risk
        tp2 = entry - 2.0 * risk
        tp3 = entry - 3.0 * risk
    else:
        signal = "خنثی / صبر"
        side = "WAIT"
        entry = price
        stop = price - 1.5 * atr_v
        tp1 = price + 1.0 * atr_v
        tp2 = price + 2.0 * atr_v
        tp3 = price + 3.0 * atr_v

    return {
        "signal": signal,
        "side": side,
        "score": int(score),
        "price": round(price, 2),
        "entry": round(entry, 2),
        "stop_loss": round(stop, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "take_profit_3": round(tp3, 2),
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "rsi": round(float(last["rsi"]), 2),
        "ema20": round(float(last["ema20"]), 2),
        "ema50": round(float(last["ema50"]), 2),
        "ema200": round(float(last["ema200"]), 2),
        "macd": round(float(last["macd"]), 4),
        "macd_signal": round(float(last["macd_signal"]), 4),
        "atr": round(float(atr_v), 2),
        "reasons": reasons,
        "data_points": int(len(x)),
        "last_date": str(pd.Timestamp(last["date"]).date()),
        "data_source": "Tindex public stock history (source shown by Tindex: Tsetmc)",
        "warning": "این خروجی سیگنال تحلیلی است و تضمین سود یا توصیه سرمایه‌گذاری نیست.",
    }


def run_backtest(df, threshold=2):
    """Adaptive backtest that works with Tindex's public 2-page history.
    With long history it uses EMA20/50/200; with the public ~40-row window
    it switches to EMA10/20 so the button produces a meaningful test instead
    of always returning zero due to the EMA200 warm-up requirement.
    """
    x = df.copy().sort_values("date").reset_index(drop=True)
    x["ema10"] = x["close"].ewm(span=10, adjust=False).mean()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["ema200"] = x["close"].ewm(span=200, adjust=False).mean()
    x["rsi"] = rsi(x["close"], 14)
    x["macd"], x["macd_signal"], _ = macd(x["close"])

    # Public Tindex history currently exposes about 40 rows without sign-in.
    # Use the longer EMA model when enough history exists; otherwise use the
    # shorter model so the backtest can actually run on the available window.
    short_mode = len(x) < 206
    start_index = 20 if short_mode else 200
    if len(x) <= start_index + 5:
        return {
            "trades": 0,
            "win_rate_pct": 0,
            "avg_return_pct": 0,
            "total_return_pct": 0,
            "note": f"داده کافی برای بک‌تست وجود ندارد؛ {len(x)} رکورد دریافت شد."
        }

    signals = []
    for i in range(start_index, len(x) - 5):
        row = x.iloc[i]
        score = 0

        if short_mode:
            score += 1 if row["close"] > row["ema10"] else -1
            score += 1 if row["ema10"] > row["ema20"] else -1
        else:
            score += 1 if row["close"] > row["ema20"] else -1
            score += 1 if row["close"] > row["ema50"] else -1
            score += 1 if row["close"] > row["ema200"] else -1

        score += 1 if row["rsi"] >= 55 else (-1 if row["rsi"] <= 45 else 0)
        score += 1 if row["macd"] > row["macd_signal"] else -1

        entry = float(x.iloc[i + 1]["open"])
        exit_price = float(x.iloc[i + 5]["close"])

        if score >= threshold:
            ret = (exit_price / entry - 1) * 100
            direction = "LONG"
        elif score <= -threshold:
            ret = (entry / exit_price - 1) * 100
            direction = "SHORT"
        else:
            continue

        signals.append({
            "date": str(pd.Timestamp(x.iloc[i + 1]["date"]).date()),
            "direction": direction,
            "return_pct": ret,
        })

    if not signals:
        return {
            "trades": 0,
            "win_rate_pct": 0,
            "avg_return_pct": 0,
            "total_return_pct": 0,
            "note": "در پنجره تاریخی موجود سیگنال کافی با این شروط ایجاد نشد."
        }

    bt = pd.DataFrame(signals)
    wins = (bt["return_pct"] > 0).sum()
    avg = bt["return_pct"].mean()
    total = ((1 + bt["return_pct"] / 100).prod() - 1) * 100
    mode_note = (
        "به‌دلیل محدودیت تاریخچه عمومی Tindex، این بک‌تست با EMA10/20 انجام شد."
        if short_mode else
        "این بک‌تست با EMA20/50/200 انجام شد."
    )

    return {
        "trades": int(len(bt)),
        "win_rate_pct": round(float(wins / len(bt) * 100), 2),
        "avg_return_pct": round(float(avg), 2),
        "total_return_pct": round(float(total), 2),
        "note": mode_note + " افق هر معامله 5 روز است؛ کارمزد، صف خرید/فروش و لغزش قیمت لحاظ نشده است."
    }


def money(v):
    if v is None or pd.isna(v):
        return "-"
    return f"{float(v):,.0f}"


HTML = r"""
<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحلیل‌گر بورس ایران</title>
<style>
body{font-family:Tahoma,Arial,sans-serif;background:#f4f6f8;margin:0;color:#18212b}
.wrap{max-width:900px;margin:auto;padding:18px}
.card{background:#fff;border-radius:18px;padding:18px;margin:12px 0;box-shadow:0 5px 20px #00000010}
h1{margin:0 0 8px;font-size:24px}
input,button{width:100%;box-sizing:border-box;padding:13px;border-radius:12px;border:1px solid #d5dbe0;font-size:16px}
button{background:#111827;color:white;border:0;margin-top:10px;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.item{background:#f8fafc;border-radius:12px;padding:12px}
.label{font-size:12px;color:#68717b}.value{font-size:18px;font-weight:bold;margin-top:4px}
.good{color:#087f5b}.bad{color:#c92a2a}.neutral{color:#b26a00}
.small{font-size:12px;color:#6b7280;line-height:1.8}
table{width:100%;border-collapse:collapse}td{padding:9px;border-bottom:1px solid #eee}
a{color:#2563eb}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h1>📊 تحلیل‌گر بورس ایران — نسخه 5.2</h1>
<div class="small">منبع داده قیمت: صفحه عمومی تاریخچه سهام Tindex. برای هر تحلیل چند صفحه از تاریخچه دریافت می‌شود و در سرور ۵ دقیقه کش می‌شود.</div>
<input id="symbol" value="استیل" placeholder="نماد، مثال: استیل">
<button onclick="analyze()">تحلیل نماد</button>
<button onclick="backtest()">بک‌تست</button>
</div>
<div id="out"></div>
<div class="card">
<a href="https://easytrader.emofid.com" target="_blank">ورود به ایزی‌تریدر مفید ↗</a>
</div>
</div>
<script>
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function box(label,value){return `<div class="item"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`}
async function analyze(){
 const s=document.getElementById('symbol').value.trim();
 if(!s)return;
 document.getElementById('out').innerHTML='<div class="card">⏳ در حال دریافت تاریخچه و محاسبه تحلیل...</div>';
 try{
  const r=await fetch('/analyze?symbol='+encodeURIComponent(s));
  const j=await r.json();
  if(!r.ok) throw new Error(j.detail||'خطا');
  const cls=j.side==='LONG'?'good':j.side==='SHORT'?'bad':'neutral';
  document.getElementById('out').innerHTML=`
  <div class="card">
   <h2 class="${cls}">${esc(j.signal)}</h2>
   <div class="grid">
    ${box('قیمت آخر',j.price)}
    ${box('امتیاز',j.score)}
    ${box('ورود',j.entry)}
    ${box('حد ضرر',j.stop_loss)}
    ${box('حد سود ۱',j.take_profit_1)}
    ${box('حد سود ۲',j.take_profit_2)}
    ${box('حد سود ۳',j.take_profit_3)}
    ${box('حمایت',j.support)}
    ${box('مقاومت',j.resistance)}
    ${box('RSI',j.rsi)}
    ${box('EMA20',j.ema20)}
    ${box('EMA50',j.ema50)}
    ${box('EMA200',j.ema200)}
   </div>
  </div>
  <div class="card"><h3>دلایل تحلیل</h3><ul>${j.reasons.map(x=>'<li>'+esc(x)+'</li>').join('')}</ul></div>
  <div class="card small">آخرین تاریخ داده: ${esc(j.last_date)} — تعداد رکورد: ${esc(j.data_points)}<br>${esc(j.data_source)}<br>${esc(j.warning)}</div>`;
 }catch(e){document.getElementById('out').innerHTML='<div class="card bad">خطا: '+esc(e.message)+'</div>'}
}
async function backtest(){
 const s=document.getElementById('symbol').value.trim();
 if(!s)return;
 document.getElementById('out').innerHTML='<div class="card">⏳ در حال اجرای بک‌تست...</div>';
 try{
  const r=await fetch('/backtest?symbol='+encodeURIComponent(s));
  const j=await r.json();
  if(!r.ok) throw new Error(j.detail||'خطا');
  document.getElementById('out').innerHTML=`
  <div class="card"><h2>نتیجه بک‌تست ${esc(s)}</h2>
  <div class="grid">
   ${box('تعداد معاملات',j.trades)}
   ${box('درصد برد',j.win_rate_pct+'%')}
   ${box('میانگین بازده',j.avg_return_pct+'%')}
   ${box('بازده مرکب',j.total_return_pct+'%')}
  </div>
  <p class="small">${esc(j.note)}</p></div>`;
 }catch(e){document.getElementById('out').innerHTML='<div class="card bad">خطا: '+esc(e.message)+'</div>'}
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "data_source": "Tindex public stock history",
        "token_configured": bool(TINDEX_TOKEN),
        "broker_link": EASYTRADER_URL,
    }


@app.get("/analyze")
def analyze(symbol: str = Query(..., min_length=1, max_length=50)):
    try:
        df = get_history(symbol, pages=25)
        return build_signal(df)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/backtest")
def backtest(symbol: str = Query(..., min_length=1, max_length=50)):
    try:
        # بک‌تست برای EMA200 حداقل به بیش از 205 روز داده نیاز دارد.
        # 20 صفحه تقریباً 400 روز معاملاتی در اختیار موتور می‌گذارد.
        df = get_history(symbol, pages=2)
        return run_backtest(df)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/symbols")
def symbols():
    return {
        "note": "جستجوی نماد از API حذف‌شده Tindex استفاده نمی‌کند؛ نماد مستقیماً از صفحه عمومی Tindex خوانده می‌شود.",
        "example": ["استیل", "شستا", "فملی", "خودرو"],
    }
