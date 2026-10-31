"""Заполнение пробелов в итоговых таблицах простым методом — forward fill.

Пропуск заполняется последним известным значением. Это не заглядывает в
будущее (в отличие от интерполяции, которая берёт значение следующего дня),
поэтому заполненные таблицы безопасны для бэктеста.

Правила:
  * заполняются только пробелы ВНУТРИ истории ряда: до первого значения
    (монета ещё не торговалась, окно индикатора не набралось) остаётся NaN;
  * не больше MAX_FILL_DAYS дней подряд — более длинная дыра остаётся NaN и
    попадает в предупреждение, её нужно разбирать руками;
  * объём в заполненные дни = 0 (торгов не было);
  * фандинг в индикаторах не переносится: если фьючерс в этот день торговался,
    берётся настоящая ставка;
  * доходности пересчитываются из заполненных цен: в заполненные дни
    доходность 0, а всё движение за пропуск приходится на первый день после
    него (для фондовых индексов: суббота и воскресенье 0, понедельник —
    движение от пятницы).

Исходные таблицы в data/processed/ не меняются. Заполненные версии с теми же
именами пишутся в data/processed/filled/, плюс filled_cells.csv — список всех
заполненных ячеек (table, date, column), чтобы их можно было исключить.

Запуск (последним шагом, после indicators):
    uv run python -m src.data.fill_gaps
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.indicators import load_funding
from src.data.indices_build import NON_PRICE

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "filled"

MAX_FILL_DAYS = 5  # покрывает выходные + праздник; длиннее — уже дыра в данных
FUNDING_COLS = ["funding_rate", "funding_premium", "funding_premium_7"]

log = logging.getLogger("fill")


def ffill_internal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward fill внутри [первое, последнее] значение каждого столбца.

    Возвращает заполненную таблицу и булеву маску заполненных ячеек.
    Серии длиннее MAX_FILL_DAYS не заполняются вовсе (а не частично).
    """
    filled = df.copy()
    for col in df.columns:
        s = df[col]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            continue
        inner = s.loc[first:last]
        gap = inner.isna()
        run_id = (~gap).cumsum()
        run_len = gap.groupby(run_id).transform("sum")
        fillable = gap & (run_len <= MAX_FILL_DAYS)
        too_long = gap & ~fillable
        if too_long.any():
            log.warning("%s: %d дн. в пропусках длиннее %d дней не заполнены (первый %s)",
                        col, int(too_long.sum()), MAX_FILL_DAYS, too_long.idxmax().date())
        filled.loc[fillable[fillable].index, col] = inner.ffill().loc[fillable]
    mask = df.isna() & filled.notna()
    return filled, mask


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def mask_to_long(name: str, mask: pd.DataFrame) -> pd.DataFrame:
    s = mask.stack()
    s = s[s]
    return pd.DataFrame({
        "table": name,
        "date": s.index.get_level_values(0),
        "column": s.index.get_level_values(1),
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = []

    def save(name: str, df: pd.DataFrame, mask: pd.DataFrame | None) -> None:
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=not isinstance(df.index, pd.RangeIndex))
        n = 0
        if mask is not None:
            cells.append(mask_to_long(name, mask))
            n = int(mask.values.sum())
        log.info("%-18s заполнено ячеек: %d", name, n)

    # --- Binance: цены, объёмы, доходности ---
    prices, pmask = ffill_internal(pd.read_parquet(PROCESSED / "prices.parquet"))
    volumes = pd.read_parquet(PROCESSED / "volumes.parquet")
    volumes = volumes.mask(pmask, 0.0)
    save("prices", prices, pmask)
    save("volumes", volumes, pmask)
    save("returns", log_returns(prices), None)  # пересчёт, а не заполнение

    # --- рыночные индексы ---
    indices, imask = ffill_internal(pd.read_parquet(PROCESSED / "indices.parquet"))
    save("indices", indices, imask)
    price_cols = [c for c in indices.columns if c not in NON_PRICE]
    save("indices_returns", log_returns(indices[price_cols]), None)

    # --- общерыночные индикаторы ---
    market, mmask = ffill_internal(pd.read_parquet(PROCESSED / "market_indicators.parquet"))
    save("market_indicators", market, mmask)

    # --- индикаторы по монетам: длинная таблица -> по каждой монете на полный календарь ---
    ind = pd.read_parquet(PROCESSED / "indicators.parquet")
    funding = load_funding() if set(FUNDING_COLS) <= set(ind.columns) else None
    calendar = prices.index
    parts, masks = [], []
    for coin, g in ind.groupby("coin", sort=False):
        wide = g.drop(columns="coin").set_index("date").reindex(calendar)
        wide = wide.loc[g["date"].min():g["date"].max()]
        # пропуск — это отсутствующая строка, поэтому считаем по строке целиком:
        # если строки не было, заполняем все индикаторы; NaN разогрева окна не трогаем
        missing = pd.Series(~wide.index.isin(g["date"]), index=wide.index)
        run_len = missing.groupby((~missing).cumsum()).transform("sum")
        if (missing & (run_len > MAX_FILL_DAYS)).any():
            log.warning("indicators %s: пропуски длиннее %d дней не заполнены", coin, MAX_FILL_DAYS)
        missing_row = (missing & (run_len <= MAX_FILL_DAYS)).to_numpy()
        filled = wide.ffill()
        filled.loc[~missing_row] = wide.loc[~missing_row]
        m = pd.DataFrame(False, index=wide.index, columns=wide.columns)
        m.loc[missing_row] = filled.loc[missing_row].notna()
        # фандинг не переносим, а берём настоящий: фьючерс мог торговаться,
        # когда спот стоял (POL 2024-09-11 и 09-12)
        if funding is not None:
            real = (funding[funding["coin"] == coin].set_index("date")
                    .reindex(wide.index)[FUNDING_COLS])
            use = missing_row[:, None] & real.notna().to_numpy()
            filled[FUNDING_COLS] = filled[FUNDING_COLS].mask(use, real)
            m[FUNDING_COLS] = m[FUNDING_COLS] & ~use
        filled.insert(0, "coin", coin)
        parts.append(filled.reset_index())
        masks.append(m.add_prefix(f"{coin}:"))
    ind_filled = pd.concat(parts, ignore_index=True).sort_values(["date", "coin"]).reset_index(drop=True)
    ind_mask = pd.concat(masks, axis=1).fillna(False).astype(bool)
    save("indicators", ind_filled, ind_mask)

    report = pd.concat(cells, ignore_index=True)
    report["date"] = report["date"].dt.strftime("%Y-%m-%d")
    report.to_csv(OUT_DIR / "filled_cells.csv", index=False)
    log.info("filled_cells.csv: %d заполненных ячеек", len(report))


if __name__ == "__main__":
    main()
