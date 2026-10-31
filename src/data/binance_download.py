"""Загрузка дневных OHLCV-свечей с Binance (спот, пары к USDT).

Источники:
  * rest    — публичный эндпоинт /api/v3/klines (ключ не нужен, до 1000 свечей
              за запрос, поэтому пагинация по времени);
  * archive — ZIP-архивы data.binance.vision (месячные + дневные за текущий месяц).

Каждая пара сохраняется в data/raw/binance/<SYMBOL>.csv. Если файл уже есть,
догружаются только недостающие дни, так что при сбое можно просто перезапустить.

Формат: date (UTC, начало дневной свечи), open, high, low, close, volume
(в монетах), quote_volume (в USDT). Незакрытая свеча текущего дня отбрасывается.

Запуск:
    uv run python -m src.data.binance_download                 # все монеты, REST
    uv run python -m src.data.binance_download --source archive
    uv run python -m src.data.binance_download --coins BTC ETH --full
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.data.binance_coins import COINS, START_DATE

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "binance"

# api.binance.com может быть недоступен из некоторых регионов;
# data-api.binance.vision — официальное зеркало только для публичных данных.
REST_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
ARCHIVE_BASE = "https://data.binance.vision/data/spot"

KLINES_LIMIT = 1000
REQUEST_PAUSE = 0.25   # сек; вес klines = 2 при лимите 6000/мин, запас большой
MAX_RETRIES = 5
DAY_MS = 86_400_000

COLUMNS = ["date", "open", "high", "low", "close", "volume", "quote_volume"]
KLINE_FIELDS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]

log = logging.getLogger("binance")
session = requests.Session()


class SymbolNotFound(Exception):
    pass


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET с повторными попытками: сетевые ошибки, 5xx и 429/418 (rate limit)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("%s: %s, повтор через %ss (%d/%d)", url, exc, wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            continue
        if resp.status_code in (429, 418):
            wait = int(resp.headers.get("Retry-After", 60))
            log.warning("rate limit (%s), пауза %ss", resp.status_code, wait)
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = 2 ** attempt
            log.warning("%s: HTTP %s, повтор через %ss", url, resp.status_code, wait)
            time.sleep(wait)
            continue
        return resp
    raise RuntimeError(f"не удалось получить {url} за {MAX_RETRIES} попыток")


def _to_frame(rows: list[list], now_ms: int) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_FIELDS)
    for col in ("open_time", "close_time"):
        df[col] = df[col].astype("int64")
        # с 2025 года спотовые архивы отдают время в микросекундах
        us = df[col] > 10**14
        df.loc[us, col] = df.loc[us, col] // 1000
    df = df[df["close_time"] < now_ms]  # отбрасываем незакрытую свечу
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df[COLUMNS]


def fetch_rest(symbol: str, start_ms: int, now_ms: int) -> pd.DataFrame:
    """Пагинация по /api/v3/klines: сдвигаем startTime на последнюю свечу + 1 день."""
    rows: list[list] = []
    last_error = None
    for base in REST_BASES:
        try:
            cursor = start_ms
            rows = []
            while cursor < now_ms:
                resp = _get(f"{base}/api/v3/klines", {
                    "symbol": symbol, "interval": "1d",
                    "startTime": cursor, "limit": KLINES_LIMIT,
                })
                if resp.status_code == 400 and "Invalid symbol" in resp.text:
                    raise SymbolNotFound(symbol)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                cursor = int(batch[-1][0]) + DAY_MS
                time.sleep(REQUEST_PAUSE)
                if len(batch) < KLINES_LIMIT:
                    break
            return _to_frame(rows, now_ms)
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            log.warning("%s недоступен (%s), пробую следующий адрес", base, exc)
    raise RuntimeError(f"REST недоступен для {symbol}: {last_error}")


def _read_zip(url: str) -> list[list] | None:
    resp = _get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    lines = [ln.split(",") for ln in raw.splitlines() if ln and ln[0].isdigit()]
    return lines


def fetch_archive(symbol: str, start_ms: int, now_ms: int) -> pd.DataFrame:
    """Месячные архивы за завершённые месяцы + дневные за текущий месяц."""
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    now = pd.Timestamp(now_ms, unit="ms", tz="UTC")
    rows: list[list] = []
    found_any = False
    month = start.tz_localize(None).to_period("M")
    current = now.tz_localize(None).to_period("M")
    while month < current:
        name = f"{symbol}-1d-{month.year}-{month.month:02d}"
        part = _read_zip(f"{ARCHIVE_BASE}/monthly/klines/{symbol}/1d/{name}.zip")
        if part is not None:
            rows.extend(part)
            found_any = True
        month += 1
        time.sleep(REQUEST_PAUSE)
    day = max(current.start_time.tz_localize("UTC"), start.normalize())
    while day < now.normalize():
        name = f"{symbol}-1d-{day:%Y-%m-%d}"
        part = _read_zip(f"{ARCHIVE_BASE}/daily/klines/{symbol}/1d/{name}.zip")
        if part is not None:
            rows.extend(part)
            found_any = True
        day += pd.Timedelta(days=1)
        time.sleep(REQUEST_PAUSE)
    if not found_any:
        raise SymbolNotFound(symbol)
    df = _to_frame(rows, now_ms)
    return df[df["date"] >= start]


def download_symbol(symbol: str, source: str, full: bool) -> None:
    path = RAW_DIR / f"{symbol}.csv"
    now_ms = int(time.time() * 1000)
    start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)

    old = None
    if path.exists() and not full:
        old = pd.read_csv(path, parse_dates=["date"])
        if not old.empty:
            start_ms = int(old["date"].max().timestamp() * 1000) + DAY_MS

    fetch = fetch_rest if source == "rest" else fetch_archive
    try:
        new = fetch(symbol, start_ms, now_ms)
    except SymbolNotFound:
        log.error("%s: пара не найдена на Binance", symbol)
        return

    df = new if old is None else pd.concat([old, new])
    df = (df.drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True))
    if df.empty:
        log.warning("%s: нет данных", symbol)
        return
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    tmp.replace(path)  # атомарная запись: при сбое старый файл не портится
    log.info("%s: %d свечей (%s — %s), новых %d", symbol, len(df),
             df["date"].min().date(), df["date"].max().date(), len(new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["rest", "archive"], default="rest")
    parser.add_argument("--coins", nargs="*", help="тикеры из binance_coins.COINS (по умолчанию все)")
    parser.add_argument("--full", action="store_true", help="перекачать с нуля, игнорируя готовые файлы")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    coins = args.coins or list(COINS)
    for coin in coins:
        for symbol in COINS[coin]:
            try:
                download_symbol(symbol, args.source, args.full)
            except Exception:
                # одна сломанная пара не должна останавливать всю выгрузку
                log.exception("%s: ошибка загрузки", symbol)


if __name__ == "__main__":
    main()
