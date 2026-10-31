"""Проверка сырых свечей Binance и сборка сводных таблиц.

Читает data/raw/binance/<SYMBOL>.csv, склеивает пары переименованных монет
(см. binance_coins.COINS), проверяет качество и пишет в data/processed/:

  prices.parquet          даты × монеты, цена закрытия в USDT
  volumes.parquet         даты × монеты, объём торгов в USDT (quote_volume)
  returns.parquet         даты × монеты, лог-доходности ln(close_t / close_{t-1})
  binance_ohlcv.parquet   длинная таблица: date, coin, symbol, open..quote_volume
  binance_quality.csv     сводка проверок по каждой монете

Индекс — date, UTC, начало дневной свечи. Столбцы — тикеры (BTC, ETH, ...).
До начала торгов монеты и в днях без данных стоит NaN — пропуски не
заполняются, это решение остаётся за этапом очистки.

Запуск:
    uv run python -m src.data.binance_build
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.binance_coins import COINS, START_DATE

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "binance"
OUT_DIR = ROOT / "data" / "processed"

log = logging.getLogger("binance")


def load_coin(coin: str) -> pd.DataFrame | None:
    """Склеивает историю всех пар монеты; на общих датах приоритет у более поздней пары."""
    parts = []
    for symbol in COINS[coin]:
        path = RAW_DIR / f"{symbol}.csv"
        if not path.exists():
            log.warning("%s: нет файла %s, пропускаю пару", coin, path.name)
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["symbol"] = symbol
        parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df.insert(1, "coin", coin)
    return df


def check_coin(coin: str, df: pd.DataFrame, today: pd.Timestamp) -> dict:
    """Проверки из плана: полнота дат, дубли, согласованность OHLC, цены > 0."""
    first, last = df["date"].min(), df["date"].max()
    expected = (last - first).days + 1
    full_range = pd.date_range(first, last, freq="D", tz="UTC")
    missing = full_range.difference(pd.DatetimeIndex(df["date"]))

    ohlc_max = df[["open", "close"]].max(axis=1)
    ohlc_min = df[["open", "close"]].min(axis=1)
    bad_high = df["high"] < ohlc_max
    bad_low = df["low"] > ohlc_min
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)

    report = {
        "coin": coin,
        "symbols": "+".join(df["symbol"].unique()),
        "first_date": first.date(),
        "last_date": last.date(),
        "late_start": first > pd.Timestamp(START_DATE, tz="UTC"),
        "rows": len(df),
        "expected_rows": expected,
        "missing_days": len(missing),
        "missing_dates": ";".join(str(d.date()) for d in missing[:20]),
        "duplicates": int(df["date"].duplicated().sum()),
        "bad_high": int(bad_high.sum()),
        "bad_low": int(bad_low.sum()),
        "nonpositive_price": int(nonpos.sum()),
        "zero_volume_days": int((df["volume"] == 0).sum()),
        "unclosed_candle": bool(last >= today),
    }
    report["ok"] = not (report["duplicates"] or report["bad_high"] or report["bad_low"]
                        or report["nonpositive_price"] or report["unclosed_candle"])
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").normalize()

    frames, reports = [], []
    for coin in COINS:
        df = load_coin(coin)
        if df is None:
            log.error("%s: нет сырых данных, сначала запустите binance_download", coin)
            continue
        rep = check_coin(coin, df, today)
        reports.append(rep)
        level = logging.INFO if rep["ok"] and not rep["missing_days"] else logging.WARNING
        log.log(level, "%-5s %s — %s, строк %d/%d, пропусков %d%s", coin, rep["first_date"],
                rep["last_date"], rep["rows"], rep["expected_rows"], rep["missing_days"],
                "" if rep["ok"] else "  ОШИБКИ ПРОВЕРКИ")
        frames.append(df)

    quality = pd.DataFrame(reports)
    quality.to_csv(OUT_DIR / "binance_quality.csv", index=False)

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv.to_parquet(OUT_DIR / "binance_ohlcv.parquet", index=False)

    # общий календарь: все дни от START_DATE до последней закрытой свечи
    calendar = pd.date_range(pd.Timestamp(START_DATE, tz="UTC"), ohlcv["date"].max(), freq="D", name="date")
    order = [c for c in COINS if c in set(ohlcv["coin"])]

    def wide(col: str) -> pd.DataFrame:
        return ohlcv.pivot(index="date", columns="coin", values=col).reindex(calendar)[order]

    prices = wide("close")
    volumes = wide("quote_volume")
    # без fill_method: доходность через пропуск остаётся NaN, а не размазывается
    returns = np.log(prices / prices.shift(1))

    for name, table in [("prices", prices), ("volumes", volumes), ("returns", returns)]:
        table.columns.name = None
        table.to_parquet(OUT_DIR / f"{name}.parquet")
    log.info("сохранено в %s: %d дней × %d монет", OUT_DIR, len(prices), prices.shape[1])

    if not quality["ok"].all():
        log.error("проверки не пройдены: %s", ", ".join(quality.loc[~quality["ok"], "coin"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
