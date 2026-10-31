"""Загрузка ставок финансирования (funding rate) бессрочных фьючерсов Binance.

Funding — периодический платёж между лонгами и шортами на USDⓈ-M perpetual.
Положительная ставка: лонги платят шортам, рынок перегрет в сторону покупок.
Отрицательная — перекос в шорты. Это один из стандартных индикаторов настроения.

Эндпоинт /fapi/v1/fundingRate публичный (ключ не нужен), до 1000 записей за
запрос. Обычно платежи раз в 8 часов (00:00, 08:00, 16:00 UTC), но у части
монет интервал бывает 4 или 1 час — поэтому дневное значение считается как
сумма всех ставок, начисленных в течение UTC-дня.

Пишет в data/raw/binance_funding/:
  <SYMBOL>.csv         сырые ставки: funding_time (UTC, ISO 8601), funding_rate
  funding_daily.csv    date, coin, funding_rate (сумма за день, доля, не %)

Запуск:
    uv run python -m src.data.funding_download
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.data.binance_coins import COINS, START_DATE
from src.data.binance_download import _get

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "binance_funding"
FAPI_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
LIMIT = 1000
REQUEST_PAUSE = 0.7  # лимит эндпоинта — 500 запросов за 5 минут

log = logging.getLogger("funding")


def fetch_symbol(symbol: str) -> pd.DataFrame:
    cursor = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    rows: list[dict] = []
    while cursor < now_ms:
        resp = _get(FAPI_URL, {"symbol": symbol, "startTime": cursor, "limit": LIMIT})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(REQUEST_PAUSE)
        if len(batch) < LIMIT:
            break
    df = pd.DataFrame(rows, columns=["fundingTime", "fundingRate"])
    df["funding_time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["funding_time", "funding_rate"]].drop_duplicates("funding_time")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").normalize()

    daily = []
    for coin, symbols in COINS.items():
        parts = []
        for symbol in symbols:
            try:
                df = fetch_symbol(symbol)
            except Exception:
                log.exception("%s: ошибка загрузки", symbol)
                continue
            if df.empty:
                log.warning("%s: нет фьючерса или ставок", symbol)
                continue
            df.to_csv(RAW_DIR / f"{symbol}.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
            log.info("%-10s %5d ставок (%s — %s)", symbol, len(df),
                     df["funding_time"].min().date(), df["funding_time"].max().date())
            parts.append(df)
        if not parts:
            continue
        # при переименовании старый контракт может торговаться параллельно с новым
        # (MATICUSDT жив до сих пор) и с другим интервалом выплат — поэтому
        # старую пару обрезаем строго до первой ставки новой, а не смешиваем
        for i in range(len(parts) - 1):
            parts[i] = parts[i][parts[i]["funding_time"] < parts[i + 1]["funding_time"].min()]
        df = pd.concat(parts)
        df["date"] = df["funding_time"].dt.floor("D")
        day = df.groupby("date")["funding_rate"].sum().reset_index()
        day = day[day["date"] < today]  # текущий день ещё не закончился
        # день запуска фьючерса обычно неполный: сумма ставок за него занижена
        if day["date"].iloc[0] > pd.Timestamp(START_DATE, tz="UTC"):
            day = day.iloc[1:]
        day.insert(1, "coin", coin)
        daily.append(day)

    out = pd.concat(daily, ignore_index=True)
    out.to_csv(RAW_DIR / "funding_daily.csv", index=False, date_format="%Y-%m-%d")
    log.info("funding_daily.csv: %d строк, %d монет", len(out), out["coin"].nunique())


if __name__ == "__main__":
    main()
