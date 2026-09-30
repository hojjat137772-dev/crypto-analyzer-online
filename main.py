import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

APP_VERSION = "1.0.0"
KRAKEN_BASE = "https://api.kraken.com/0/public"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFAULT_QUOTE = "USD"

app = FastAPI(title="تحلیل‌گر بازار رمز ارز", version=APP_VERSION)
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CryptoAnalyzerOnline/1.0",
    "Accept": "application/json",
})

CACHE = {}
CACHE_TTL = 300
ASSET_CACHE = {"time": 0, "pairs": {}}
MARKET_CACHE = {"time": 0, "rows": []}

INTERVALS = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def api_get(url, params=None, timeout=20):
    try:
        r = SESSION.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(502, f"خطا در دریافت داده بازار: {exc}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise HTTPException(502, "منبع داده خطا برگرداند: " + ", ".join(data["error"]))
    return data


def norm_symbol(value: str) -> str:
    s = str(value).strip().upper().replace("/", "")
    aliases = {"BTC": "BTCUSD", "XBTUSD": "BTCUSD", "XBT": "BTCUSD", "ETH": "ETHUSD"}
    return aliases.get(s, s)


def get_kraken_pairs():
    now = time.time()
    if ASSET_CACHE["pairs"] and now - ASSET_CACHE["time"] < 3600:
        return ASSET_CACHE["pairs"]
    data = api_get(f"{KRAKEN_BASE}/AssetPairs")
    pairs = {}
    for key, item in data.get("result", {}).items():
        alt = str(item.get("altname", key)).upper()
        ws = str(item.get("wsname", "")).upper().replace("/", "")
        base = str(item.get("base", "")).upper()
        quote = str(item.get("quote", "")).upper()
        # Prefer USD pairs. Kraken uses XBT internally for BTC.
        if quote not in {"ZUSD", "USD", "USDT"} and not ws.endswith("/USD"):
            continue
        for candidate in {key.upper(), alt, ws, ws.replace("XBT", "BTC"), alt.replace("XBT", "BTC")}:
            if candidate:
                pairs[candidate] = key
        if base in {"XXBT", "XBT", "BTC"}:
            pairs["BTCUSD"] = key
    ASSET_CACHE.update({"time": now, "pairs": pairs})
    return pairs


def resolve_kraken_pair(symbol: str) -> str:
    wanted = norm_symbol(symbol)
    pairs = get_kraken_pairs()
    if wanted in pairs:
        return pairs[wanted]
    # Try a base asset + USD form.
    if wanted.endswith("USD"):
        base = wanted[:-3]
        for candidate in (base + "USD", "X" + base + "ZUSD"):
            if candidate in pairs:
                return pairs[candidate]
    raise HTTPException(404, f"برای {symbol} جفت USD قابل تحلیل در Kraken پیدا نشد.")


def get_ohlc(symbol: str, interval: str = "1h") -> tuple[pd.DataFrame, str]:
    interval = interval.lower()
    if interval not in INTERVALS:
        raise HTTPException(400, "تایم‌فریم باید یکی از 15m، 1h، 4h یا 1d باشد.")
    pair = resolve_kraken_pair(symbol)
    key = f"ohlc:{pair}:{INTERVALS[interval]}"
    now = time.time()
    cached = CACHE.get(key)
    if cached and now - cached["time"] < CACHE_TTL:
        return cached["df"].copy(), pair

    data = api_get(f"{KRAKEN_BASE}/OHLC", {"pair": pair, "interval": INTERVALS[interval]})
    result = data.get("result", {})
    rows = result.get(pair) or next((v for k, v in result.items() if k != "last" and isinstance(v, list)), [])
    if not rows:
        raise HTTPException(502, f"داده OHLC برای {symbol} دریافت نشد.")

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        raise HTTPException(400, f"داده تاریخی کافی برای {symbol} وجود ندارد.")
    CACHE[key] = {"time": now, "df": df.copy()}
    return df, pair


def get_markets(limit=250):
    now = time.time()
    if MARKET_CACHE["rows"] and now - MARKET_CACHE["time"] < 600 and len(MARKET_CACHE["rows"]) >= limit:
        return MARKET_CACHE["rows"][:limit]
    rows = []
    pages = max(1, int(np.ceil(limit / 250)))
    for page in range(1, pages + 1):
        data = api_get(
            f"{COINGECKO_BASE}/coins/markets",
            {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(250, limit),
                "page": page,
                "sparkline": "false",
            },
        )
        rows.extend(data)
        if len(data) < 250:
            break
    MARKET_CACHE.update({"time": now, "rows": rows})
    return rows[:limit]


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(s):
    fast, slow = ema(s, 12), ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def support_resistance(df, window=30):
    recent = df.tail(min(window, len(df)))
    return float(recent["low"].min()), float(recent["high"].max())


def technical_frame(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"])
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x["close"])
    x["atr"] = atr(x)
    x["volume_ma20"] = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / x["volume_ma20"].replace(0, np.nan)
    return x.dropna(subset=["ema20", "ema50", "ema200", "rsi", "macd", "macd_signal", "atr"]).reset_index(drop=True)


def build_signal(df):
    x = technical_frame(df)
    if len(x) < 30:
        raise HTTPException(400, "داده کافی برای تحلیل وجود ندارد.")
    last = x.iloc[-1]
    price = float(last["close"])
    atr_v = max(float(last["atr"]), price * 0.001)
    rsi_v = float(last["rsi"])
    support, resistance = support_resistance(x, 30)

    trend = 0.0
    reasons = []
    if price > last["ema20"]:
        trend += 9; reasons.append("قیمت بالای EMA20 است")
    else:
        reasons.append("قیمت زیر EMA20 است")
    if last["ema20"] > last["ema50"]:
        trend += 9; reasons.append("EMA20 بالای EMA50 است")
    else:
        reasons.append("EMA20 زیر EMA50 است")
    if last["ema50"] > last["ema200"]:
        trend += 9; reasons.append("EMA50 بالای EMA200 است")
    else:
        reasons.append("EMA50 زیر EMA200 است")
    if price > last["ema200"]:
        trend += 8; reasons.append("قیمت بالای EMA200 است")
    else:
        reasons.append("قیمت زیر EMA200 است")
    bearish = sum([price < last["ema20"], last["ema20"] < last["ema50"], last["ema50"] < last["ema200"], price < last["ema200"]])
    if bearish >= 3:
        trend = max(0, 35 - trend)

    if 55 <= rsi_v <= 68:
        momentum, momentum_reason = 25, "RSI در محدوده قدرت خرید سالم"
    elif 50 <= rsi_v < 55:
        momentum, momentum_reason = 19, "RSI کمی بالاتر از خنثی"
    elif 68 < rsi_v <= 72:
        momentum, momentum_reason = 20, "RSI بالا و نزدیک اشباع خرید"
    elif rsi_v > 72:
        momentum, momentum_reason = 13, "RSI در ناحیه اشباع خرید"
    elif 45 <= rsi_v < 50:
        momentum, momentum_reason = 11, "RSI کمی ضعیف‌تر از خنثی"
    else:
        momentum, momentum_reason = 5, "مومنتوم ضعیف"

    macd_v, macd_sig, hist = map(float, [last["macd"], last["macd_signal"], last["macd_hist"]])
    if macd_v > macd_sig and hist > 0:
        macd_score, macd_reason = 15, "MACD و هیستوگرام صعودی"
    elif macd_v > macd_sig:
        macd_score, macd_reason = 11, "MACD بالاتر از خط سیگنال"
    elif hist > 0:
        macd_score, macd_reason = 8, "هیستوگرام MACD مثبت است"
    else:
        macd_score, macd_reason = 3, "MACD تأیید صعودی ندارد"

    span = max(resistance - support, 1e-12)
    position = (price - support) / span
    if price > resistance:
        sr_score, sr_reason = 15, "قیمت بالاتر از مقاومت اخیر"
    elif position <= .30:
        sr_score, sr_reason = 14, "قیمت نزدیک حمایت است"
    elif position <= .55:
        sr_score, sr_reason = 12, "قیمت در ناحیه متعادل حمایت/مقاومت"
    elif position <= .75:
        sr_score, sr_reason = 8, "قیمت در نیمه بالایی محدوده است"
    else:
        sr_score, sr_reason = 4, "قیمت به مقاومت نزدیک است"

    proposed_stop_long = max(support, price - 1.8 * atr_v)
    if proposed_stop_long >= price:
        proposed_stop_long = price - 1.8 * atr_v
    risk_long = max(price - proposed_stop_long, atr_v * .5)
    rr_res = (resistance - price) / risk_long if risk_long > 0 else 0
    rr_score = 10 if rr_res >= 2.5 else 8 if rr_res >= 1.8 else 5 if rr_res >= 1.2 else 2 if rr_res >= .7 else 0

    bullish = min(100, max(0, trend + momentum + macd_score + sr_score + rr_score))
    bearish_score = min(100, max(0, (35-trend)+(25-momentum)+(15-macd_score)+(15-sr_score)+(10-rr_score)))
    score = int(round(max(0, min(100, 50 + (bullish - bearish_score) / 2))))

    if score >= 75:
        signal, side = "خرید قوی / بررسی ورود", "LONG"
    elif score >= 60:
        signal, side = "خرید مشروط / بررسی ورود", "LONG"
    elif score <= 25:
        signal, side = "فروش / بررسی نزول", "SHORT"
    elif score <= 40:
        signal, side = "ضعیف / عدم ورود", "SHORT"
    else:
        signal, side = "خنثی / صبر", "WAIT"

    entry = price
    if side == "LONG":
        stop = proposed_stop_long
        risk = max(entry - stop, atr_v * .5)
        tp1, tp2, tp3 = entry + risk, entry + 2*risk, entry + 3*risk
    elif side == "SHORT":
        stop = min(resistance, price + 1.8*atr_v)
        if stop <= entry:
            stop = price + 1.8*atr_v
        risk = max(stop-entry, atr_v*.5)
        tp1, tp2, tp3 = entry-risk, entry-2*risk, entry-3*risk
    else:
        stop = price - 1.5*atr_v
        risk = max(abs(price-stop), atr_v*.5)
        tp1, tp2, tp3 = price + atr_v, price + 2*atr_v, price + 3*atr_v

    rr_tp2 = abs(tp2-entry) / max(abs(entry-stop), 1e-12)
    vol_ratio = float(last["volume_ratio"]) if pd.notna(last["volume_ratio"]) else 1.0
    reasons += [momentum_reason, macd_reason, sr_reason]
    if vol_ratio >= 1.2:
        reasons.append("حجم معاملات بالاتر از میانگین ۲۰ کندل است")

    return {
        "signal": signal, "side": side, "score": score,
        "score_label": "صعودی" if score >= 60 else ("نزولی" if score <= 40 else "خنثی"),
        "score_breakdown": {"trend": round(trend,1), "momentum_rsi": round(momentum,1), "macd": round(macd_score,1), "support_resistance": round(sr_score,1), "risk_reward": round(rr_score,1)},
        "price": round(price,8), "entry": round(entry,8), "stop_loss": round(stop,8),
        "take_profit_1": round(tp1,8), "take_profit_2": round(tp2,8), "take_profit_3": round(tp3,8),
        "risk_reward_tp2": round(rr_tp2,2), "support": round(support,8), "resistance": round(resistance,8),
        "rsi": round(rsi_v,2), "ema20": round(float(last["ema20"]),8), "ema50": round(float(last["ema50"]),8),
        "ema200": round(float(last["ema200"]),8), "macd": round(macd_v,8), "macd_signal": round(macd_sig,8),
        "atr": round(atr_v,8), "volume_ratio": round(vol_ratio,2), "reasons": reasons,
        "data_points": len(x), "last_date": str(pd.Timestamp(last["date"]).date()), "data_source": "Kraken OHLC",
        "warning": "این خروجی الگوریتمی است و تضمین سود یا توصیه سرمایه‌گذاری نیست.",
    }


def run_backtest(df, fee_pct=0.10):
    x = technical_frame(df)
    trades = []
    start = 30
    for i in range(start, len(x)-5):
        row = x.iloc[i]
        score = 0
        score += 1 if row["close"] > row["ema20"] else -1
        score += 1 if row["ema20"] > row["ema50"] else -1
        score += 1 if row["ema50"] > row["ema200"] else -1
        score += 1 if row["rsi"] >= 55 else (-1 if row["rsi"] <= 45 else 0)
        score += 1 if row["macd"] > row["macd_signal"] else -1
        entry = float(x.iloc[i+1]["open"])
        exit_price = float(x.iloc[i+5]["close"])
        fee = 2 * fee_pct
        if score >= 3:
            ret = (exit_price/entry - 1)*100 - fee
            direction = "LONG"
        elif score <= -3:
            ret = (entry/exit_price - 1)*100 - fee
            direction = "SHORT"
        else:
            continue
        trades.append({"date": str(pd.Timestamp(x.iloc[i+1]["date"]).date()), "direction": direction, "return_pct": round(ret,3)})
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"win_rate_pct":0,"avg_return_pct":0,"total_return_pct":0,"max_drawdown_pct":0,"note":"در داده موجود سیگنال کافی ایجاد نشد."}
    bt = pd.DataFrame(trades)
    wins = int((bt["return_pct"] > 0).sum())
    equity = (1 + bt["return_pct"]/100).cumprod()
    peak = equity.cummax()
    dd = (equity/peak - 1)*100
    return {"trades":len(bt),"wins":wins,"losses":len(bt)-wins,"win_rate_pct":round(wins/len(bt)*100,2),"avg_return_pct":round(float(bt["return_pct"].mean()),3),"total_return_pct":round(float((equity.iloc[-1]-1)*100),2),"max_drawdown_pct":round(float(dd.min()),2),"note":f"بک‌تست با ۵ کندل خروج و کارمزد فرضی {fee_pct:.2f}% برای هر طرف اجرا شد." ,"trade_log":trades[-50:]}


def market_rows():
    markets = get_markets(250)
    return [{"id":m.get("id"),"symbol":str(m.get("symbol","")).upper(),"name":m.get("name"),"market_cap":m.get("market_cap"),"price":m.get("current_price"),"change_24h":m.get("price_change_percentage_24h"),"volume_24h":m.get("total_volume")} for m in markets]


HTML = r'''<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحلیل‌گر بازار رمز ارز</title>
<style>
body{margin:0;background:#07101f;color:#fff;font-family:Tahoma,Arial,sans-serif}.wrap{max-width:760px;margin:auto;padding:10px}.card{background:#111d32;padding:18px;border-radius:22px;margin:10px 0}.sub{color:#aebbd0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.box{background:#18263d;padding:10px;border-radius:12px;text-align:center}input,select,button{width:100%;box-sizing:border-box;padding:14px;border-radius:12px;margin:5px 0;font-size:16px}input,select{background:#091426;color:#fff;border:1px solid #334155}button{border:0;background:#16a34a;color:#fff;font-weight:bold}.secondary{background:#334155}.good{color:#4ade80}.bad{color:#fb7185}.small{font-size:12px;color:#93a4bc}li{margin:7px 0}.scanrow{display:flex;justify-content:space-between;padding:9px;border-bottom:1px solid #26364f}.pill{padding:3px 7px;border-radius:8px;background:#26364f}@media(max-width:600px){.grid{grid-template-columns:1fr 1fr}}
</style></head><body><div class="wrap"><div class="card">
<h1>₿ تحلیل‌گر بازار رمز ارز</h1><div class="sub">نسخه کریپتو بر پایه معماری اپ بورس ایران: تحلیل تکنیکال، امتیاز ۰ تا ۱۰۰، ورود، حدضرر، ۳ حدسود، بک‌تست و اسکن بازار</div>
<input id="symbol" value="BTCUSD" placeholder="مثلاً BTCUSD یا ETHUSD"><select id="interval"><option>15m</option><option selected>1h</option><option>4h</option><option>1d</option></select>
<button onclick="analyze()">🔍 تحلیل رمز ارز</button><button class="secondary" onclick="backtest()">🧪 بک‌تست</button><button class="secondary" onclick="scan()">📊 اسکن ۲۰ ارز برتر و نمایش ۳ مورد اول</button>
<div id="out" class="card">آماده تحلیل...</div></div></div>
<script>
const f=v=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:8});
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
function box(a,b){return `<div class="box">${esc(a)}<br><b>${esc(f(b))}</b></div>`}
async function analyze(){const s=document.getElementById('symbol').value.trim(),i=document.getElementById('interval').value;document.getElementById('out').innerHTML='⏳ در حال تحلیل...';try{const r=await fetch(`/analyze?symbol=${encodeURIComponent(s)}&interval=${i}`);const j=await r.json();if(!r.ok)throw Error(j.detail||'خطا');document.getElementById('out').innerHTML=`<h2>${esc(j.signal)}</h2><div class="grid">${box('قیمت',j.price)}${box('امتیاز',j.score)}${box('ورود',j.entry)}${box('حد ضرر',j.stop_loss)}${box('TP1',j.take_profit_1)}${box('TP2',j.take_profit_2)}${box('TP3',j.take_profit_3)}${box('R/R تا TP2',j.risk_reward_tp2)}${box('حمایت',j.support)}${box('مقاومت',j.resistance)}${box('RSI',j.rsi)}${box('EMA20',j.ema20)}${box('EMA50',j.ema50)}${box('EMA200',j.ema200)}${box('حجم/میانگین',j.volume_ratio)}</div><div class="card"><h3>اجزای امتیاز</h3><div class="grid">${box('روند از 35',j.score_breakdown.trend)}${box('RSI از 25',j.score_breakdown.momentum_rsi)}${box('MACD از 15',j.score_breakdown.macd)}${box('حمایت/مقاومت از 15',j.score_breakdown.support_resistance)}${box('ریسک/بازده از 10',j.score_breakdown.risk_reward)}</div></div><div class="card"><h3>دلایل</h3><ul>${j.reasons.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div><div class="small">${esc(j.data_source)} — ${esc(j.last_date)} — ${j.data_points} کندل</div>`}catch(e){document.getElementById('out').innerHTML=`<span class="bad">خطا: ${esc(e.message)}</span>`}}
async function backtest(){const s=document.getElementById('symbol').value.trim(),i=document.getElementById('interval').value;document.getElementById('out').innerHTML='⏳ در حال اجرای بک‌تست...';try{const r=await fetch(`/backtest?symbol=${encodeURIComponent(s)}&interval=${i}`);const j=await r.json();if(!r.ok)throw Error(j.detail||'خطا');document.getElementById('out').innerHTML=`<h2>🧪 نتیجه بک‌تست</h2><div class="grid">${box('معاملات',j.trades)}${box('برد',j.wins)}${box('باخت',j.losses)}${box('Win Rate %',j.win_rate_pct)}${box('میانگین بازده %',j.avg_return_pct)}${box('بازده کل %',j.total_return_pct)}${box('Max Drawdown %',j.max_drawdown_pct)}</div><p class="small">${esc(j.note)}</p>`}catch(e){document.getElementById('out').innerHTML=`<span class="bad">خطا: ${esc(e.message)}</span>`}}
async function scan(){document.getElementById('out').innerHTML='⏳ در حال اسکن...';try{const r=await fetch('/scan?limit=20&top=3&interval='+document.getElementById('interval').value);const j=await r.json();if(!r.ok)throw Error(j.detail||'خطا');document.getElementById('out').innerHTML=`<h2>📊 نتیجه اسکن</h2>${j.results.map((x,k)=>`<div class="scanrow"><span>#${k+1} ${esc(x.symbol)} — ${esc(x.name)}</span><span class="pill">${esc(x.score)}</span></div>`).join('')}<p class="small">اسکن فقط دارایی‌هایی را بررسی می‌کند که جفت قابل دریافت در Kraken داشته باشند.</p>`}catch(e){document.getElementById('out').innerHTML=`<span class="bad">خطا: ${esc(e.message)}</span>`}}
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/health")
def health():
    return {"status":"ok","version":APP_VERSION,"data_source":"Kraken OHLC + CoinGecko market metadata"}


@app.get("/analyze")
def analyze(symbol: str = Query("BTCUSD", min_length=2, max_length=30), interval: str = Query("1h")):
    df, pair = get_ohlc(symbol, interval)
    result = build_signal(df)
    result.update({"symbol": norm_symbol(symbol), "pair": pair, "interval": interval})
    # Optional market metadata; analysis remains usable if CoinGecko is unavailable.
    try:
        needle = norm_symbol(symbol).replace("USD", "").replace("USDT", "").lower().replace("BTC", "btc")
        for m in get_markets(100):
            if str(m.get("symbol","")).lower() in {needle, "xbt" if needle == "btc" else needle}:
                result["market_cap_usd"] = m.get("market_cap")
                result["change_24h_pct"] = m.get("price_change_percentage_24h")
                result["volume_24h_usd"] = m.get("total_volume")
                result["name"] = m.get("name")
                break
    except Exception:
        pass
    return result


@app.get("/backtest")
def backtest(symbol: str = Query("BTCUSD", min_length=2, max_length=30), interval: str = Query("1h"), fee_pct: float = Query(0.10, ge=0, le=2)):
    df, pair = get_ohlc(symbol, interval)
    result = run_backtest(df, fee_pct)
    result.update({"symbol": norm_symbol(symbol), "pair": pair, "interval": interval, "data_source":"Kraken OHLC"})
    return result


@app.get("/symbols")
def symbols(limit: int = Query(100, ge=1, le=250)):
    return {"data_source":"CoinGecko", "symbols": market_rows()[:limit]}


@app.get("/scan")
def scan(limit: int = Query(20, ge=1, le=50), top: int = Query(3, ge=1, le=10), interval: str = Query("1h")):
    markets = get_markets(limit)
    results = []
    for m in markets:
        sym = str(m.get("symbol","")).upper()
        if not sym:
            continue
        try:
            df, pair = get_ohlc(sym + "USD", interval)
            sig = build_signal(df)
            results.append({"symbol":sym,"name":m.get("name"),"score":sig["score"],"signal":sig["signal"],"pair":pair})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"interval":interval,"scanned":len(results),"results":results[:top]}
