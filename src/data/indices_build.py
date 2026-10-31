"""Проверка сырых данных по индексам и сборка сводных таблиц.

Читает data/raw/indices/<NAME>.csv и пишет в data/processed/:

  indices.parquet          даты × индексы, значение на закрытие торгового дня
  indices_returns.parquet  даты × ценовые индексы, лог-доходности между
                           соседними торговыми днями
  indices_quality.csv      сводка проверок по каждому индексу

Календарь тот же, что у prices.parquet (Binance): все дни подряд, UTC.
Традиционные рынки не торгуют в выходные и праздники — там NaN, пропуски
не заполняются. Доходность понедельника считается от закрытия пятницы.

Запуск (после binance_build, чтобы взять календарь крипты):
    uv run python -m src.data.indices_build
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.binance_coins import START_DATE
from src.data.indices_download import FNG_NAME, YAHOO_INDICES

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "indices"
OUT_DIR = ROOT / "data" / "processed"

# для этих рядов доходность не имеет смысла: это уровни/проценты, а не цены
NON_PRICE = {"VIX", "UST10Y", FNG_NAME}
MAX_GAP_DAYS = 5  # больше — уже не праздник, а дыра в данных

log = logging.getLogger("indices")


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    if name == FNG_NAME:
        df["close"] = df["value"].astype(float)
    return df


def check(name: str, df: pd.DataFrame) -> dict:
    trading = name != FNG_NAME  # FNG публикуется каждый день, включая выходные
    gaps = df["date"].diff().dt.days
    report = {
        "index": name,
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "rows": len(df),
        "duplicates": int(df["date"].duplicated().sum()),
        "nonpositive": int((df["close"] <= 0).sum()) if name != FNG_NAME else 0,
        "weekend_rows": int((df["date"].dt.dayofweek >= 5).sum()) if trading else 0,
        "max_gap_days": int(gaps.max()),
        "long_gaps": ";".join(str(d.date()) for d in df.loc[gaps > (MAX_GAP_DAYS if trading else 1), "date"]),
    }
    if trading:
        # у индексов Yahoo изредка встречаются рассогласованные OHLC — это не
        # критично для close, поэтому только предупреждение
        report["bad_ohlc"] = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
                                  (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    report["ok"] = not (report["duplicates"] or report["nonpositive"] or report["weekend_rows"]
                        or report["max_gap_days"] > MAX_GAP_DAYS)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    names = [*YAHOO_INDICES, FNG_NAME]

    series, reports = {}, []
    for name in names:
        if not (RAW_DIR / f"{name}.csv").exists():
            log.error("%s: нет сырых данных, сначала запустите indices_download", name)
            continue
        df = load(name)
        rep = check(name, df)
        reports.append(rep)
        log.log(logging.INFO if rep["ok"] else logging.WARNING,
                "%-6s %s — %s, строк %d, макс. разрыв %d дн., кривых OHLC %s%s",
                name, rep["first_date"], rep["last_date"], rep["rows"], rep["max_gap_days"],
                rep.get("bad_ohlc", "-"), "" if rep["ok"] else "  ОШИБКИ ПРОВЕРКИ")
        series[name] = df.set_index("date")["close"]

    quality = pd.DataFrame(reports)
    quality.to_csv(OUT_DIR / "indices_quality.csv", index=False)

    prices_path = OUT_DIR / "prices.parquet"
    if prices_path.exists():
        calendar = pd.read_parquet(prices_path).index
    else:
        log.warning("нет prices.parquet, календарь строится до вчерашнего дня UTC")
        end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        calendar = pd.date_range(pd.Timestamp(START_DATE, tz="UTC"), end, freq="D", name="date")

    levels = pd.DataFrame(series)
    # доходности считаем по торговым дням каждого индекса, потом кладём на календарь
    returns = pd.DataFrame({
        name: np.log(s / s.shift(1)) for name, s in series.items() if name not in NON_PRICE
    })

    levels.reindex(calendar).to_parquet(OUT_DIR / "indices.parquet")
    returns.reindex(calendar).to_parquet(OUT_DIR / "indices_returns.parquet")
    log.info("сохранено в %s: %d дней × %d индексов", OUT_DIR, len(calendar), levels.shape[1])

    if not quality["ok"].all():
        log.error("проверки не пройдены: %s", ", ".join(quality.loc[~quality["ok"], "index"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
