import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

BASE = "https://cdn.tsetmc.com/api"
SYMBOL_FILE = Path("symbols.txt")
OUT = Path("data/market_data.json")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

S = requests.Session()
S.headers.update({
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
})


def get_json(url, timeout=25):
    r = S.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def search(symbol):
    url = f"{BASE}/Instrument/GetInstrumentSearch/{quote(symbol, safe='')}"
    data = get_json(url)

    items = data.get("instrumentSearch", []) if isinstance(data, dict) else []

    exact = [
        x for x in items
        if str(x.get("lVal18AFC", "")).strip() == symbol.strip()
    ]

    return exact[0] if exact else (items[0] if items else None)


def history(ins_code):
    url = f"{BASE}/ClosingPrice/GetClosingPriceDailyList/{ins_code}/0"
    data = get_json(url, timeout=35)

    return (
        data.get("closingPriceDaily", [])
        if isinstance(data, dict)
        else []
    )


def normalize(rows):
    out = []

    for r in rows:
        close = r.get("pClosing")

        if close is None:
            continue

        out.append({
            "date": r.get("dEven"),
            "open": r.get("priceFirst", close),
            "high": r.get("priceMax", close),
            "low": r.get("priceMin", close),
            "close": close,
            "last": r.get("pDrCotVal", close),
            "volume": r.get("qTotTran5J", 0),
            "value": r.get("qTotCap", 0),
        })

    return out


def main():

    symbols = []

    if SYMBOL_FILE.exists():
        symbols = [
            x.strip()
            for x in SYMBOL_FILE.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.strip().startswith("#")
        ]

    if not symbols:
        symbols = ["استیل"]

    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "TSETMC CDN via GitHub Actions",
        "symbols": {},
        "errors": {},
    }

    for symbol in symbols:

        try:
            item = search(symbol)

            if not item:
                raise RuntimeError("نماد در TSETMC پیدا نشد")

            ins_code = item.get("insCode")

            rows = history(ins_code)

            if len(rows) < 60:
                raise RuntimeError(
                    f"تاریخچه کافی نیست: {len(rows)} رکورد"
                )

            result["symbols"][symbol] = {
                "ticker": item.get("lVal18AFC", symbol),
                "company": item.get("lVal30", ""),
                "isin": item.get("cIsin", ""),
                "ins_code": ins_code,
                "history": normalize(rows),
            }

            print(f"OK {symbol}: {len(rows)} rows")

        except Exception as exc:

            result["errors"][symbol] = str(exc)

            print(f"ERROR {symbol}: {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    OUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":")
        ),
        encoding="utf-8"
    )

    if not result["symbols"]:
        raise SystemExit(
            "No symbol data was collected"
        )


if __name__ == "__main__":
    main()
