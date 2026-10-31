"""Технические индикаторы по монетам и общерыночные индикаторы настроения.

Считается по уже собранным таблицам (сначала binance_build, indices_build и
funding_download). Пишет в data/processed/:

  indicators.parquet         длинная таблица date, coin, <индикаторы>
  market_indicators.parquet  даты × общерыночные индикаторы

Индикатор за дату D использует данные только по закрытие дня D включительно.
Чтобы торговать по нему, сигнал нужно сдвигать: решение на D+1 по данным D.

Индикаторы по монете считаются по её собственным свечам подряд: дни без
торгов (у POL 2024-09-11 и 09-12) просто пропускаются, а не заполняются.
Первые N дней после начала истории (окно индикатора) — NaN.

Запуск:
    uv run python -m src.data.indicators
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FUNDING_DIR = ROOT / "data" / "raw" / "binance_funding"

DAYS_PER_YEAR = 365  # крипта торгуется каждый день

# Процентная часть фандинга Binance: F = P + clamp(I − P, ±0.05%), где I —
# фиксированная оценка разницы ставок займа USDT и монеты, 0.03% в день
# (0.01% за 8 ч, 0.005% за 4 ч). Это «цена денег» для арбитража спот/перп,
# а не настроение рынка. У BNB процентная часть нулевая (проверено по данным:
# модальная ставка BNBUSDT во все годы — ровно 0).
FUNDING_INTEREST_DAILY = 0.0003
FUNDING_INTEREST_OVERRIDES = {"BNB": 0.0}

log = logging.getLogger("indicators")


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Сглаживание Уайлдера (как в классических RSI и ATR)."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder(-delta.clip(upper=0), n)
    return 100 - 100 / (1 + gain / loss)


def coin_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df — свечи одной монеты, отсортированные по дате."""
    c, h, l = df["close"], df["high"], df["low"]
    out = pd.DataFrame({"date": df["date"], "coin": df["coin"]})
    logret = np.log(c / c.shift(1))

    out["rsi_14"] = rsi(c, 14)

    # MACD 12/26/9 в процентах от цены, чтобы монеты были сравнимы
    macd = c.ewm(span=12, adjust=False, min_periods=26).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_pct"] = 100 * macd / c
    out["macd_signal_pct"] = 100 * signal / c
    out["macd_hist_pct"] = out["macd_pct"] - out["macd_signal_pct"]

    # полосы Боллинджера 20/2: %B (0 — нижняя полоса, 1 — верхняя) и ширина
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    out["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    out["bb_width"] = 4 * sd / mid

    # стохастик 14/3
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    # ATR 14 в процентах от цены
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_pct"] = 100 * wilder(tr, 14) / c

    # MFI 14 — «RSI с учётом объёма»
    tp = (h + l + c) / 3
    flow = tp * df["volume"]
    up = flow.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    down = flow.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    out["mfi_14"] = 100 - 100 / (1 + up / down)

    # тренд и моментум
    out["dist_sma50"] = c / c.rolling(50).mean() - 1
    out["dist_sma200"] = c / c.rolling(200).mean() - 1
    out["mom_7"] = logret.rolling(7).sum()
    out["mom_30"] = logret.rolling(30).sum()
    out["mom_90"] = logret.rolling(90).sum()

    # реализованная волатильность за 30 дней, годовая
    out["vol_30"] = logret.rolling(30).std() * np.sqrt(DAYS_PER_YEAR)

    # аномальность объёма: z-оценка log(объём в USDT) относительно 30 прошлых дней
    lv = np.log(df["quote_volume"])
    out["volume_z_30"] = (lv - lv.shift(1).rolling(30).mean()) / lv.shift(1).rolling(30).std()

    return out


def load_funding() -> pd.DataFrame | None:
    path = FUNDING_DIR / "funding_daily.csv"
    if not path.exists():
        log.warning("нет %s — индикаторы по фандингу пропущены (запустите funding_download)", path)
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ohlcv = pd.read_parquet(PROCESSED / "binance_ohlcv.parquet").sort_values(["coin", "date"])
    calendar = pd.read_parquet(PROCESSED / "prices.parquet").index
    coins = list(pd.read_parquet(PROCESSED / "prices.parquet").columns)

    ind = pd.concat([coin_indicators(g) for _, g in ohlcv.groupby("coin", sort=False)],
                    ignore_index=True)

    funding = load_funding()
    if funding is not None:
        ind = ind.merge(funding[["date", "coin", "funding_rate"]], on=["date", "coin"], how="left")
        base = ind["coin"].map(FUNDING_INTEREST_OVERRIDES).fillna(FUNDING_INTEREST_DAILY)
        # премия сверх стоимости денег: > 0 — перекос в лонги, < 0 — в шорты
        ind["funding_premium"] = ind["funding_rate"] - base
        ind["funding_premium_7"] = (ind.groupby("coin")["funding_premium"]
                                       .transform(lambda s: s.rolling(7, min_periods=7).mean()))

    ind = ind.sort_values(["date", "coin"]).reset_index(drop=True)
    ind.to_parquet(PROCESSED / "indicators.parquet", index=False)
    log.info("indicators.parquet: %d строк, %d индикаторов", len(ind), ind.shape[1] - 2)

    # --- общерыночные индикаторы ---
    def wide(col: str) -> pd.DataFrame:
        return ind.pivot(index="date", columns="coin", values=col).reindex(calendar)[coins]

    market = pd.DataFrame(index=calendar)
    indices = pd.read_parquet(PROCESSED / "indices.parquet")
    market["fng"] = indices["FNG"]

    rsi_w = wide("rsi_14")
    market["rsi_mean"] = rsi_w.mean(axis=1)
    market["share_overbought"] = (rsi_w > 70).sum(axis=1) / rsi_w.notna().sum(axis=1)
    market["share_oversold"] = (rsi_w < 30).sum(axis=1) / rsi_w.notna().sum(axis=1)

    for n in (50, 200):
        d = wide(f"dist_sma{n}")
        market[f"breadth_sma{n}"] = (d > 0).sum(axis=1) / d.notna().sum(axis=1)

    # «сезон альткоинов»: доля альтов, обогнавших BTC за 90 дней
    # (по мотивам Altcoin Season Index от blockchaincenter.net)
    mom90 = wide("mom_90")
    alts = mom90.drop(columns="BTC")
    market["altseason_90"] = alts.gt(mom90["BTC"], axis=0).sum(axis=1) / alts.notna().sum(axis=1)

    market["btc_rsi_14"] = rsi_w["BTC"]
    market["btc_vol_30"] = wide("vol_30")["BTC"]
    market["vol_30_mean"] = wide("vol_30").mean(axis=1)

    # средняя попарная корреляция доходностей за 30 дней — «всё падает вместе»
    returns = pd.read_parquet(PROCESSED / "returns.parquet")
    pairs = returns.rolling(30, min_periods=30).corr().stack().dropna()
    a, b = pairs.index.get_level_values(1), pairs.index.get_level_values(2)
    market["corr_30_mean"] = pairs[a < b].groupby(level=0).mean().reindex(calendar)

    if funding is not None:
        fp = wide("funding_premium")
        market["funding_premium_mean"] = fp.mean(axis=1)
        market["btc_funding_premium"] = fp["BTC"]

    market.to_parquet(PROCESSED / "market_indicators.parquet")
    log.info("market_indicators.parquet: %d дней × %d индикаторов", *market.shape)


if __name__ == "__main__":
    main()
