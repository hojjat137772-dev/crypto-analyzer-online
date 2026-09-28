import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION = "5.6.2"
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


SYMBOLS_CACHE = {"time": 0, "rows": []}
SYMBOLS_CACHE_TTL = 900  # 15 minutes


def fetch_stock_list_page(page=1):
    url = f"{TINDEX_BASE}/en/stocks/"
    params = {"page": page, "sort": "market_cap"}
    try:
        r = SESSION.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        raise RuntimeError(f"دریافت فهرست نمادها از Tindex ناموفق بود: {exc}") from exc
    if r.status_code != 200:
        raise RuntimeError(f"Tindex برای فهرست نمادها با کد {r.status_code} پاسخ داد.")
    return BeautifulSoup(r.text, "html.parser")


def parse_stock_list(soup):
    """Extract stock symbols robustly from Tindex screener HTML.

    Tindex may change the table markup. The stock detail links are more
    stable than the visible table columns, so use them first and fall back
    to table rows.
    """
    from urllib.parse import unquote

    rows = []
    seen = set()

    # Primary method: collect stock-detail links from the screener page.
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        m = re.search(r"/(?:en/)?stocks/([^/?#]+)/?(?:\?|#|$)", href)
        if not m:
            continue
        slug = unquote(m.group(1)).strip()
        if not slug or slug.lower() in {"history", "monthly", "performance", "fundamentals", "valuation"}:
            continue
        # Avoid option/contract links and keep only plausible Persian/Latin symbols.
        if len(slug) > 30 or "/" in slug:
            continue
        if slug not in seen:
            seen.add(slug)
            rows.append({"symbol": slug})

    if rows:
        return rows

    # Fallback: parse the first table whose header contains Symbol.
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue
        header = [normalize_header(x.get_text(" ", strip=True))
                  for x in trs[0].find_all(["th", "td"])]
        if not any(h == "symbol" or h.startswith("symbol") for h in header):
            continue
        for tr in trs[1:]:
            links = tr.find_all("a", href=True)
            symbol = None
            for a in links:
                href = a.get("href", "")
                m = re.search(r"/(?:en/)?stocks/([^/?#]+)/?(?:\?|#|$)", href)
                if m:
                    symbol = unquote(m.group(1)).strip()
                    break
            if not symbol:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if cells:
                    symbol = cells[0].split()[0].strip()
            if symbol and symbol not in seen:
                seen.add(symbol)
                rows.append({"symbol": symbol})
        if rows:
            break
    return rows

def get_all_symbols():
    """Load the full Tindex stock screener list for the dropdown.

    The API can be rate-limited to 1 request/minute on the free plan, so the
    dropdown deliberately uses the public HTML screener pages instead of
    making dozens of API calls. The result is cached for 6 hours.
    """
    now = time.time()
    if SYMBOLS_CACHE["rows"] and now - SYMBOLS_CACHE["time"] < 21600:
        return list(SYMBOLS_CACHE["rows"])

    def parse_screener_page(soup):
        # Tindex currently exposes the stock table with direct /stocks/<slug>/ links.
        # Parse links first so minor table markup changes do not empty the list.
        from urllib.parse import unquote
        rows = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            m = re.search(r"/(?:en/)?stocks/([^/?#]+)/?(?:\?|#|$)", href)
            if not m:
                continue
            sym = unquote(m.group(1)).strip()
            if not sym or len(sym) > 40 or sym.lower() in {"history", "monthly", "performance", "fundamentals", "valuation"}:
                continue
            if sym in seen:
                continue
            text_name = a.get_text(" ", strip=True) or sym
            rows.append({"symbol": sym, "name": text_name})
            seen.add(sym)
        return rows

    # First page gives us the total number of symbols and the first 20 rows.
    try:
        soup = fetch_stock_list_page(1)
        first = parse_screener_page(soup)
        page_text = soup.get_text(" ", strip=True)
        m_total = re.search(r"([\d,]+)\s+symbols", page_text, flags=re.I)
        total = int(m_total.group(1).replace(",", "")) if m_total else 1400
        last_page = min(80, max(1, (total + 19) // 20))
    except Exception as exc:
        # Keep a usable dropdown even when Tindex is temporarily unavailable.
        fallback = "فملی فولاد خودرو خساپا شستا وبملت وتجارت وبصادر شپنا شبندر شبریز شتران پارسان کگل کچاد ومعادن وغدیر شپترو نوری تاپیکو فارس حکشتی اخابر همراه مبین ذوب جم آریا کرمان کرمانشاه بوعلی وپارس وپاسار وپست وکار وسپهر وبهمن وملت ونیرو وپترو فخوز فخاس فروس فزر فصبا فسبز فاسمین فمراد فجر کگهر کگاز کدما کساپا کشرق کطبس کماسه کمنگنز کپرور کاذر دسبحان دالبر دزهراوی دکوثر تیپیکو برکت والبر وپخش وملل وسینا".split()
        rows = [{"symbol": x} for x in fallback]
        SYMBOLS_CACHE["rows"] = rows
        SYMBOLS_CACHE["time"] = now
        return list(rows)

    all_rows = []
    seen = set()
    for row in first:
        if row["symbol"] not in seen:
            seen.add(row["symbol"])
            all_rows.append(row)

    # Fetch remaining public HTML pages politely and sequentially.
    # This avoids hammering Tindex and triggering HTTP 429.
    if last_page > 1:
        for page in range(2, last_page + 1):
            page_rows = []
            for attempt in range(4):
                try:
                    page_rows = parse_screener_page(fetch_stock_list_page(page))
                    break
                except RuntimeError as exc:
                    if " 429" not in str(exc) and "429" not in str(exc):
                        break
                    time.sleep(3 * (attempt + 1))
                except Exception:
                    break
            for row in page_rows:
                sym = row.get("symbol", "")
                if sym and sym not in seen:
                    seen.add(sym)
                    all_rows.append(row)
            # Small pause between pages keeps the public page requests gentle.
            time.sleep(0.25)

    all_rows.sort(key=lambda x: x.get("symbol", ""))
    if len(all_rows) < 100 and first:
        # Never replace a real partial Tindex result with a hard-coded list.
        all_rows = first

    SYMBOLS_CACHE["rows"] = all_rows
    SYMBOLS_CACHE["time"] = now
    return list(all_rows)

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
    """Build a 0-100 signal score from independent technical factors.

    Score components:
      - Trend: 35 points (EMA20/50/200 alignment and price location)
      - Momentum: 25 points (RSI, with overbought/oversold penalty)
      - MACD: 15 points
      - Support/resistance location: 15 points
      - Risk/reward quality: 10 points

    The score is directional: a high score means bullish conditions,
    while a low score means bearish conditions. The UI also exposes the
    component breakdown so the number is not a black box.
    """
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

    # ---------- 1) Trend: 0..35 ----------
    trend_score = 0.0
    trend_reasons = []
    if price > float(last["ema20"]):
        trend_score += 8
        trend_reasons.append("قیمت بالاتر از EMA20")
    if float(last["ema20"]) > float(last["ema50"]):
        trend_score += 9
        trend_reasons.append("EMA20 بالاتر از EMA50")
    if float(last["ema50"]) > float(last["ema200"]):
        trend_score += 9
        trend_reasons.append("EMA50 بالاتر از EMA200")
    if price > float(last["ema200"]):
        trend_score += 9
        trend_reasons.append("قیمت بالاتر از EMA200")
    # If several trend conditions are bearish, preserve direction in the score.
    bearish_trend = 0
    bearish_trend += 1 if price < float(last["ema20"]) else 0
    bearish_trend += 1 if float(last["ema20"]) < float(last["ema50"]) else 0
    bearish_trend += 1 if float(last["ema50"]) < float(last["ema200"]) else 0
    bearish_trend += 1 if price < float(last["ema200"]) else 0
    if bearish_trend >= 3:
        trend_score = max(0.0, 35.0 - trend_score)
        trend_reasons.append("ساختار میانگین‌ها متمایل به نزول است")

    # ---------- 2) Momentum / RSI: 0..25 ----------
    rsi_v = float(last["rsi"])
    if 55 <= rsi_v <= 68:
        momentum_score = 25.0
        momentum_reason = "RSI در محدوده قدرت خرید سالم"
    elif 50 <= rsi_v < 55:
        momentum_score = 19.0
        momentum_reason = "RSI کمی بالاتر از خنثی"
    elif 68 < rsi_v <= 72:
        momentum_score = 20.0
        momentum_reason = "قدرت خرید بالا، با ریسک افزایش اشباع"
    elif rsi_v > 72:
        momentum_score = 13.0
        momentum_reason = "RSI بالا و احتمال اصلاح کوتاه‌مدت"
    elif 45 <= rsi_v < 50:
        momentum_score = 11.0
        momentum_reason = "RSI کمی ضعیف‌تر از خنثی"
    else:
        momentum_score = 5.0
        momentum_reason = "مومنتوم ضعیف"

    # ---------- 3) MACD: 0..15 ----------
    macd_v = float(last["macd"])
    macd_sig = float(last["macd_signal"])
    hist = float(last["macd_hist"])
    if macd_v > macd_sig and hist > 0:
        macd_score = 15.0
        macd_reason = "MACD و هیستوگرام صعودی"
    elif macd_v > macd_sig:
        macd_score = 11.0
        macd_reason = "MACD بالاتر از خط سیگنال"
    elif hist > 0:
        macd_score = 8.0
        macd_reason = "هیستوگرام MACD مثبت است"
    else:
        macd_score = 3.0
        macd_reason = "MACD تأیید صعودی ندارد"

    # ---------- 4) Location vs S/R: 0..15 ----------
    sr_score = 0.0
    sr_reason = ""
    span = max(resistance - support, 1e-9)
    position = (price - support) / span
    if price > resistance:
        sr_score = 15.0
        sr_reason = "قیمت بالاتر از مقاومت اخیر"
    elif position <= 0.30:
        sr_score = 14.0
        sr_reason = "قیمت نزدیک حمایت و دارای فضای رشد"
    elif position <= 0.55:
        sr_score = 12.0
        sr_reason = "فاصله مناسب از حمایت و مقاومت"
    elif position <= 0.75:
        sr_score = 8.0
        sr_reason = "قیمت در نیمه بالایی محدوده"
    else:
        sr_score = 4.0
        sr_reason = "قیمت به مقاومت نزدیک است"

    # ---------- 5) Risk/reward quality: 0..10 ----------
    proposed_stop = max(support, price - 1.8 * atr_v)
    if proposed_stop >= price:
        proposed_stop = price - 1.8 * atr_v
    risk = max(price - proposed_stop, atr_v * 0.5)
    rr_to_resistance = (resistance - price) / risk if risk > 0 else 0
    if rr_to_resistance >= 2.5:
        rr_score = 10.0
    elif rr_to_resistance >= 1.8:
        rr_score = 8.0
    elif rr_to_resistance >= 1.2:
        rr_score = 5.0
    elif rr_to_resistance >= 0.7:
        rr_score = 2.0
    else:
        rr_score = 0.0

    bullish_score = int(round(min(100, max(0,
        trend_score + momentum_score + macd_score + sr_score + rr_score))))

    # Mirror the bullish evidence to obtain a bearish score. This keeps the
    # displayed number intuitive: 0 = strongly bearish, 50 = mixed, 100 = bullish.
    bearish_score = int(round(min(100, max(0,
        (35 - trend_score) + (25 - momentum_score) +
        (15 - macd_score) + (15 - sr_score) + (10 - rr_score)))))

    # Convert the two-sided evidence into a single 0..100 score.
    score = int(round(50 + (bullish_score - bearish_score) / 2))
    score = max(0, min(100, score))

    if score >= 75:
        signal = "خرید قوی / بررسی ورود"
        side = "LONG"
    elif score >= 60:
        signal = "خرید مشروط / بررسی ورود"
        side = "LONG"
    elif score <= 25:
        signal = "فروش / اجتناب از ورود"
        side = "SHORT"
    elif score <= 40:
        signal = "ضعیف / اجتناب از ورود"
        side = "SHORT"
    else:
        signal = "خنثی / صبر"
        side = "WAIT"

    entry = price
    if side == "LONG":
        stop = proposed_stop
        risk = max(entry - stop, atr_v * 0.5)
        tp1 = entry + 1.0 * risk
        tp2 = entry + 2.0 * risk
        tp3 = entry + 3.0 * risk
    elif side == "SHORT":
        stop = min(resistance, price + 1.8 * atr_v)
        if stop <= entry:
            stop = price + 1.8 * atr_v
        risk = max(stop - entry, atr_v * 0.5)
        tp1 = entry - 1.0 * risk
        tp2 = entry - 2.0 * risk
        tp3 = entry - 3.0 * risk
    else:
        stop = price - 1.5 * atr_v
        risk = max(abs(price - stop), atr_v * 0.5)
        tp1 = price + 1.0 * atr_v
        tp2 = price + 2.0 * atr_v
        tp3 = price + 3.0 * atr_v

    # Risk/reward to TP2 for the displayed setup.
    rr_tp2 = abs(tp2 - entry) / max(abs(entry - stop), 1e-9)

    reasons = trend_reasons + [momentum_reason, macd_reason, sr_reason]
    if rr_score >= 8:
        reasons.append("نسبت ریسک/بازده تا مقاومت مناسب است")
    elif rr_score <= 2:
        reasons.append("فاصله تا مقاومت برای ورود مستقیم محدود است")

    return {
        "signal": signal,
        "side": side,
        "score": score,
        "score_label": "صعودی" if score >= 60 else ("نزولی" if score <= 40 else "خنثی"),
        "score_breakdown": {
            "trend": round(float(trend_score), 1),
            "momentum_rsi": round(float(momentum_score), 1),
            "macd": round(float(macd_score), 1),
            "support_resistance": round(float(sr_score), 1),
            "risk_reward": round(float(rr_score), 1),
        },
        "price": round(price, 2),
        "entry": round(entry, 2),
        "stop_loss": round(stop, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "take_profit_3": round(tp3, 2),
        "risk_reward_tp2": round(float(rr_tp2), 2),
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "rsi": round(rsi_v, 2),
        "ema20": round(float(last["ema20"]), 2),
        "ema50": round(float(last["ema50"]), 2),
        "ema200": round(float(last["ema200"]), 2),
        "macd": round(macd_v, 4),
        "macd_signal": round(macd_sig, 4),
        "atr": round(float(atr_v), 2),
        "reasons": reasons,
        "data_points": int(len(x)),
        "last_date": str(pd.Timestamp(last["date"]).date()),
        "data_source": "Tindex public stock history (source shown by Tindex: Tsetmc)",
        "warning": "امتیاز 0 تا 100 یک مدل تحلیلی است و تضمین سود یا توصیه سرمایه‌گذاری نیست.",
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


# وضعیت اسکن بازار در پس‌زمینه
SCAN_STATUS = {
    "running": False,
    "started_at": 0,
    "finished_at": 0,
    "processed": 0,
    "total": 0,
    "matches": 0,
    "results": [],
    "errors": 0,
    "message": "هنوز اسکن اجرا نشده است.",
}
SCAN_LOCK = threading.Lock()


def scan_one_symbol(symbol, min_score=60):
    try:
        # برای اسکن سریع بازار، از اولین صفحه تاریخچه استفاده می‌کنیم.
        # بعد از پیدا شدن نمادهای واجد شرایط، جزئیات کامل در جدول نمایش داده می‌شود.
        df = get_history(symbol, pages=2)
        signal = build_signal(df)
        if signal["side"] == "LONG" and signal["score"] >= min_score:
            return {
                "symbol": symbol,
                "score": signal["score"],
                "signal": signal["signal"],
                "price": signal["price"],
                "entry": signal["entry"],
                "stop_loss": signal["stop_loss"],
                "tp1": signal["take_profit_1"],
                "tp2": signal["take_profit_2"],
                "rr": signal["risk_reward_tp2"],
                "rsi": signal["rsi"],
            }
    except Exception:
        return None
    return None


def run_market_scan(min_score=60, workers=8):
    global SCAN_STATUS
    try:
        symbols = [x["symbol"] for x in get_all_symbols()]
    except Exception as exc:
        with SCAN_LOCK:
            SCAN_STATUS.update({
                "running": False,
                "finished_at": time.time(),
                "message": f"دریافت فهرست نمادها ناموفق بود: {exc}",
            })
        return

    with SCAN_LOCK:
        SCAN_STATUS.update({
            "running": True,
            "started_at": time.time(),
            "finished_at": 0,
            "processed": 0,
            "total": len(symbols),
            "matches": 0,
            "results": [],
            "errors": 0,
            "message": f"اسکن {len(symbols)} نماد بازار آغاز شد...",
        })

    results = []
    errors = 0
    # تعداد همزمان محدود نگه داشته شده تا فشار روی Tindex زیاد نشود.
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as executor:
        futures = {executor.submit(scan_one_symbol, sym, min_score): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                item = future.result()
                if item:
                    results.append(item)
            except Exception:
                errors += 1
            with SCAN_LOCK:
                SCAN_STATUS["processed"] += 1
                SCAN_STATUS["matches"] = len(results)
                SCAN_STATUS["errors"] = errors
                SCAN_STATUS["results"] = sorted(results, key=lambda x: (-x["score"], x["symbol"]))
                SCAN_STATUS["message"] = (
                    f"در حال اسکن: {SCAN_STATUS['processed']} از {SCAN_STATUS['total']} نماد — "
                    f"{len(results)} نماد واجد شرایط پیدا شد."
                )

    with SCAN_LOCK:
        SCAN_STATUS["running"] = False
        SCAN_STATUS["finished_at"] = time.time()
        SCAN_STATUS["results"] = sorted(results, key=lambda x: (-x["score"], x["symbol"]))
        SCAN_STATUS["matches"] = len(results)
        SCAN_STATUS["message"] = (
            f"اسکن کامل شد: از {len(symbols)} نماد، {len(results)} نماد شرایط ورود را داشتند."
        )


def start_market_scan(min_score=60):
    with SCAN_LOCK:
        if SCAN_STATUS["running"]:
            return False
    thread = threading.Thread(target=run_market_scan, args=(min_score,), daemon=True)
    thread.start()
    return True


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
body{font-family:Tahoma,Arial,sans-serif;background:linear-gradient(135deg,#eef4ff,#f8fafc 45%,#eefaf6);margin:0;color:#18212b;min-height:100vh}
.wrap{max-width:980px;margin:auto;padding:18px}
.card{background:rgba(255,255,255,.92);backdrop-filter:blur(10px);border:1px solid #ffffff;border-radius:22px;padding:20px;margin:14px 0;box-shadow:0 12px 35px #1f3b5d18}
h1{margin:0 0 8px;font-size:26px;letter-spacing:-.4px}
.hero{display:flex;justify-content:space-between;gap:14px;align-items:center;margin-bottom:16px}.badge{background:#e8f7f1;color:#087f5b;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:bold}
label{display:block;font-size:13px;font-weight:bold;margin:12px 0 7px;color:#475569}
input,select,button{width:100%;box-sizing:border-box;padding:13px 14px;border-radius:14px;border:1px solid #d8e0e8;font-size:16px;background:#fff;outline:none}input:focus,select:focus{border-color:#6b8cff;box-shadow:0 0 0 3px #6b8cff18}
.searchRow{display:grid;grid-template-columns:1fr 1.4fr;gap:10px}.searchBox{position:relative}.searchIcon{position:absolute;right:12px;top:12px;font-size:18px;color:#94a3b8}.searchBox input{padding-right:40px}
button{background:linear-gradient(135deg,#172554,#1e40af);color:white;border:0;margin-top:10px;cursor:pointer;font-weight:bold;box-shadow:0 7px 18px #1e40af25;transition:.15s}button:hover{transform:translateY(-1px)}button.secondary{background:#334155}.buttonGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.item{background:#f8fafc;border:1px solid #edf1f5;border-radius:15px;padding:13px}.label{font-size:12px;color:#68717b}.value{font-size:18px;font-weight:bold;margin-top:4px}
.good{color:#087f5b}.bad{color:#c92a2a}.neutral{color:#b26a00}.small{font-size:12px;color:#6b7280;line-height:1.8}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #eee;text-align:right}th{background:#f8fafc}a{color:#2563eb}
@media(max-width:650px){.wrap{padding:10px}.searchRow,.buttonGrid{grid-template-columns:1fr}.hero{align-items:flex-start}.card{padding:15px}}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<div class="hero"><div><h1>📊 تحلیل‌گر بورس ایران</h1><div class="small">تحلیل تکنیکال، سیگنال هوشمند، بک‌تست و اسکن بازار</div></div><div class="badge">نسخه 5.6.2</div></div>
<div class="small">منبع داده قیمت: صفحه عمومی تاریخچه سهام Tindex. برای هر تحلیل چند صفحه از تاریخچه دریافت می‌شود و در سرور ۵ دقیقه کش می‌شود.</div>
<label>انتخاب نماد</label>
<div class="searchRow">
 <div class="searchBox"><span class="searchIcon">⌕</span><input id="symbolSearch" placeholder="جستجوی نماد..." oninput="filterSymbols()"></div>
 <select id="symbol"><option value="">⏳ در حال دریافت فهرست نمادها...</option></select>
</div>
<div id="symbolCount" class="small">⏳ در حال دریافت فهرست نمادهای بازار...</div>
<div class="buttonGrid">
<button onclick="analyze()">🔍 تحلیل نماد</button>
<button class="secondary" onclick="backtest()">🧪 بک‌تست</button>
<button onclick="scanMarket()">🔎 اسکن کل بازار</button>
</div>
</div>
<div id="out"></div>
<div id="scanOut"></div>
<div class="card">
<a href="https://easytrader.emofid.com" target="_blank">ورود به ایزی‌تریدر مفید ↗</a>
</div>
</div>
<script>
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function box(label,value){return `<div class="item"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`}
let allSymbols=[];
async function loadSymbols(){
 try{
  const r=await fetch('/symbols');
  const j=await r.json();
  if(!r.ok) throw new Error(j.detail||'خطا در دریافت نمادها');
  allSymbols=j.symbols||[];
  renderSymbols(allSymbols);
  const list=document.getElementById('symbol');
  if(j.symbols.includes('استیل')) list.value='استیل';
  else if(j.symbols.length) list.value=j.symbols[0];
  document.getElementById('symbolCount').textContent=`✅ ${Number(j.count||0).toLocaleString('fa-IR')} نماد در فهرست بازار موجود است؛ می‌توانی از لیست انتخاب کنی یا جستجو کنی.`;
 }catch(e){
  document.getElementById('symbolCount').textContent='⚠️ دریافت فهرست نمادها ناموفق بود؛ صفحه را دوباره باز کن.';
 }
}


function renderSymbols(items){
 const list=document.getElementById('symbol');
 list.innerHTML=items.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
 if(items.includes('استیل')) list.value='استیل';
 else if(items.length) list.value=items[0];
}
function filterSymbols(){
 const q=document.getElementById('symbolSearch').value.trim().toLowerCase();
 const filtered=!q?allSymbols:allSymbols.filter(x=>String(x).toLowerCase().includes(q));
 renderSymbols(filtered);
 document.getElementById('symbolCount').textContent=`🔎 ${filtered.length.toLocaleString('fa-IR')} نماد مطابق جستجو`;
}

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
    ${box('امتیاز سیگنال (0-100)',j.score)}
    ${box('وضعیت امتیاز',j.score_label)}
    ${box('ریسک به ریوارد تا TP2',j.risk_reward_tp2)}
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
  <div class="card"><h3>اجزای امتیاز 0 تا 100</h3><div class="grid">
   ${box('روند (از 35)',j.score_breakdown.trend)}
   ${box('مومنتوم RSI (از 25)',j.score_breakdown.momentum_rsi)}
   ${box('MACD (از 15)',j.score_breakdown.macd)}
   ${box('حمایت/مقاومت (از 15)',j.score_breakdown.support_resistance)}
   ${box('ریسک/بازده (از 10)',j.score_breakdown.risk_reward)}
  </div></div>
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
async function scanMarket(){
 document.getElementById('scanOut').innerHTML='<div class="card">⏳ اسکن کل بازار در پس‌زمینه شروع می‌شود...</div>';
 try{
  const r=await fetch('/scan/start?min_score=60', {method:'POST'});
  const j=await r.json();
  if(!r.ok) throw new Error(j.detail||'خطا در شروع اسکن');
  pollScan();
 }catch(e){document.getElementById('scanOut').innerHTML='<div class="card bad">خطا: '+esc(e.message)+'</div>'}
}
async function pollScan(){
 try{
  const r=await fetch('/scan/status');
  const j=await r.json();
  const pct=j.total?Math.round(j.processed/j.total*100):0;
  let html=`<div class="card"><h2>🔎 اسکن کل بازار</h2>
   <div class="grid">
    ${box('وضعیت',j.running?'در حال اسکن':'پایان یافته')}
    ${box('پیشرفت',pct+'%')}
    ${box('نمادهای بررسی‌شده',j.processed+' / '+j.total)}
    ${box('نمادهای واجد شرایط',j.matches)}
   </div><p class="small">${esc(j.message)}</p>`;
  if(!j.running && j.results && j.results.length){
   html+=`<h3>نمادهای دارای شرایط ورود</h3><div style="overflow:auto"><table><thead><tr><th>نماد</th><th>امتیاز</th><th>قیمت</th><th>ورود</th><th>حدضرر</th><th>TP1</th><th>R/R</th><th>RSI</th></tr></thead><tbody>`;
   for(const x of j.results){
    html+=`<tr><td><b>${esc(x.symbol)}</b></td><td class="good"><b>${esc(x.score)}</b></td><td>${esc(x.price)}</td><td>${esc(x.entry)}</td><td>${esc(x.stop_loss)}</td><td>${esc(x.tp1)}</td><td>${esc(x.rr)}</td><td>${esc(x.rsi)}</td></tr>`;
   }
   html+='</tbody></table></div><p class="small">فقط نمادهای LONG با امتیاز حداقل 60 نمایش داده شده‌اند. برای تصمیم‌گیری نهایی، تحلیل کامل هر نماد را جداگانه اجرا کنید.</p>';
  } else if(!j.running){
   html+='<p class="small">در این اسکن نمادی با شرط فعلی پیدا نشد.</p>';
  }
  html+='</div>';
  document.getElementById('scanOut').innerHTML=html;
  if(j.running) setTimeout(pollScan,2500);
 }catch(e){document.getElementById('scanOut').innerHTML='<div class="card bad">خطا در دریافت وضعیت اسکن: '+esc(e.message)+'</div>'}
}

loadSymbols();
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


@app.post("/scan/start")
def scan_start(min_score: int = Query(60, ge=50, le=90)):
    with SCAN_LOCK:
        if SCAN_STATUS["running"]:
            return {"started": False, "message": "اسکن دیگری در حال اجراست.", "status": SCAN_STATUS}
    started = start_market_scan(min_score)
    return {
        "started": started,
        "min_score": min_score,
        "message": "اسکن کل بازار در پس‌زمینه شروع شد." if started else "اسکن شروع نشد.",
    }


@app.get("/scan/status")
def scan_status():
    with SCAN_LOCK:
        return dict(SCAN_STATUS)


@app.get("/symbols")
def symbols():
    try:
        rows = get_all_symbols()
        return {
            "count": len(rows),
            "symbols": [x["symbol"] for x in rows],
            "data_source": "Tindex public Tehran Stock Exchange screener",
        }
    except Exception as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
