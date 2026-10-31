"""Загрузка дневных значений рыночных индексов с 2020 года.

Источники:
  * Yahoo Finance (неофициальный chart API, ключ не нужен) — фондовые индексы,
    волатильность, доллар, золото, доходность трежерис;
  * alternative.me — Crypto Fear & Greed Index (0 = крайний страх, 100 = жадность).

Каждый индекс сохраняется в data/raw/indices/<NAME>.csv. Дата — торговый день
по часовому поясу биржи (а не по UTC): иначе вечерние сессии фьючерсов и
индекса доллара попадают на соседний день. Текущий (ещё не закрытый) торговый
день отбрасывается. Файлы каждый раз перезаписываются целиком — это быстро.

Запуск:
    uv run python -m src.data.indices_download
    uv run python -m src.data.indices_download --only SPX VIX
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.binance_coins import START_DATE

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "indices"

# имя столбца -> (тикер Yahoo, описание)
YAHOO_INDICES: dict[str, tuple[str, str]] = {
    "SPX": ("^GSPC", "S&P 500"),
    "NASDAQ": ("^IXIC", "NASDAQ Composite"),
    "DJI": ("^DJI", "Dow Jones Industrial Average"),
    "RUT": ("^RUT", "Russell 2000"),
    "VIX": ("^VIX", "CBOE Volatility Index"),
    "DXY": ("DX-Y.NYB", "US Dollar Index (ICE)"),
    "GOLD": ("GC=F", "Gold futures, ближайший контракт, USD за унцию"),
    "UST10Y": ("^TNX", "Доходность 10-летних US Treasuries, % годовых"),
}
FNG_NAME = "FNG"

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FNG_URL = "https://api.alternative.me/fng/"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # без него Yahoo отвечает 429
REQUEST_PAUSE = 1.0
MAX_RETRIES = 5

log = logging.getLogger("indices")
session = requests.Session()
session.headers.update(HEADERS)


def _get_json(url: str, params: dict) -> dict:
    """GET с повторными попытками при сетевых ошибках, 429 и 5xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("%s: %s, повтор через %ss (%d/%d)", url, exc, wait, attempt, MAX_RETRIES)
            time.sleep(wait)
    raise RuntimeError(f"не удалось получить {url} за {MAX_RETRIES} попыток")


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    start = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
    data = _get_json(YAHOO_URL.format(symbol=symbol), {
        "period1": start, "period2": int(time.time()), "interval": "1d",
        "events": "history",
    })["chart"]
    if data["error"]:
        raise RuntimeError(f"{symbol}: {data['error']}")
    res = data["result"][0]
    tz = res["meta"]["exchangeTimezoneName"]
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        **{col: quote[col] for col in ("open", "high", "low", "close", "volume")},
    })
    # дата торгового дня — в часовом поясе биржи
    df["date"] = df["ts"].dt.tz_convert(tz).dt.date
    today_local = pd.Timestamp.now(tz=tz).date()
    df = df[df["date"] < today_local].dropna(subset=["close"])
    # если Yahoo отдал несколько баров на одну дату, берём последний
    df = df.drop_duplicates("date", keep="last")
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_fng() -> pd.DataFrame:
    data = _get_json(FNG_URL, {"limit": 0, "format": "json"})
    if data.get("metadata", {}).get("error"):
        raise RuntimeError(f"FNG: {data['metadata']['error']}")
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.date
    df["value"] = df["value"].astype(int)
    df = df.rename(columns={"value_classification": "classification"})
    df = df[df["date"] >= pd.Timestamp(START_DATE).date()]
    return df[["date", "value", "classification"]].sort_values("date").drop_duplicates("date", keep="last")


def save(name: str, df: pd.DataFrame) -> None:
    path = RAW_DIR / f"{name}.csv"
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    log.info("%-6s %d строк (%s — %s)", name, len(df), df["date"].min(), df["date"].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help=f"имена из {list(YAHOO_INDICES) + [FNG_NAME]}")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = args.only or [*YAHOO_INDICES, FNG_NAME]

    for name in names:
        try:
            if name == FNG_NAME:
                save(name, fetch_fng())
            else:
                save(name, fetch_yahoo(YAHOO_INDICES[name][0]))
        except Exception:
            log.exception("%s: ошибка загрузки", name)
        time.sleep(REQUEST_PAUSE)


if __name__ == "__main__":
    main()
