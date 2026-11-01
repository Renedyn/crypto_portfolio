"""Сборка CoinGecko Parquet и сверка с уже выгруженными закрытиями Binance.

    uv run python -m src.data.coingecko_build --threshold-pct 2

Дата D в итоговой таблице соответствует точке CoinGecko D+1 00:00 UTC,
то есть концу свечи Binance с датой открытия D. Пропуски не заполняются.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.data.binance_coins import COINS
from src.data.coingecko_coins import COINGECKO_IDS, validate_mapping
from src.data.coingecko_download import PARAMS, RAW_DIR, ROOT, snapshot_day, utc_now, write_json

OUT_DIR = ROOT / "data" / "processed"
FIELDS = {"prices": "coingecko_prices", "market_caps": "market_caps", "total_volumes": "coingecko_volumes"}
DAY = pd.Timedelta(days=1)
log = logging.getLogger("coingecko")


def calendar_for(records: dict[str, dict]) -> pd.DatetimeIndex:
    if set(records) != set(COINS):
        raise ValueError("Нужны сырые ответы для всех монет проекта")
    days = set()
    for coin, record in records.items():
        if record["params"] != PARAMS or record["endpoint"] != f"/coins/{COINGECKO_IDS[coin]}/market_chart":
            raise ValueError(f"{coin}: параметры/ID кэша не соответствуют конфигурации")
        fetched = datetime.fromisoformat(record["fetched_at"])
        if fetched.tzinfo is None:
            raise ValueError(f"{coin}: fetched_at должен содержать UTC offset")
        days.add(snapshot_day(fetched))
    if len(days) != 1:
        raise ValueError("В кэше разные дневные окна; повторите coingecko_download для всех монет")
    end = pd.Timestamp(days.pop(), tz="UTC") - DAY
    return pd.date_range(end=end, periods=365, freq="D", name="date")


def daily_series(record: dict, field: str, calendar: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    points = record["data"].get(field)
    if not isinstance(points, list) or not points or any(not isinstance(p, list) or len(p) != 2 for p in points):
        raise ValueError(f"{field}: ожидается непустой список [timestamp_ms, value]")
    stamps = pd.to_datetime([point[0] for point in points], unit="ms", utc=True, errors="raise")
    if stamps.isna().any() or stamps.has_duplicates:
        raise ValueError(f"{field}: пустые или повторяющиеся метки времени")
    values = pd.to_numeric(pd.Series([point[1] for point in points]), errors="coerce").to_numpy(dtype=float)
    midnight = stamps == stamps.normalize()
    # Не округляем свежую внутридневную точку к полуночи.
    labels = stamps - DAY
    in_window = (labels >= calendar[0]) & (labels <= calendar[-1])
    selected = midnight & in_window
    daily_values = values[selected]
    invalid = ~np.isfinite(daily_values) | (daily_values < 0 if field == "total_volumes" else daily_values <= 0)
    daily_values[invalid] = np.nan
    series = pd.Series(daily_values, index=labels[selected], dtype=float).sort_index().reindex(calendar)
    if not selected.any():
        raise ValueError(f"{field}: нет дневных точек в ожидаемом окне")
    quality = {
        "metric": field,
        "raw_points": len(points),
        "daily_points_in_window": int(selected.sum()),
        "discarded_intraday_points": int((~midnight).sum()),
        "discarded_outside_window_points": int((midnight & ~in_window).sum()),
        "invalid_values": int(invalid.sum()),
        "zero_values": int((values[selected] == 0).sum()),
        "missing_days": int(series.isna().sum()),
        "missing_dates": ";".join(d.strftime("%Y-%m-%d") for d in series.index[series.isna()]),
        "first_valid_date": str(series.first_valid_index().date()) if series.notna().any() else "",
        "last_valid_date": str(series.last_valid_index().date()) if series.notna().any() else "",
        "ok": not (invalid.any() or series.isna().any()),
    }
    return series, quality


def build_frames(records: dict[str, dict]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    calendar = calendar_for(records)
    tables = {field: pd.DataFrame(index=calendar) for field in FIELDS}
    checks = []
    for coin in COINS:
        for field in FIELDS:
            series, quality = daily_series(records[coin], field, calendar)
            tables[field][coin] = series
            checks.append({"coin": coin, **quality})
    return tables, pd.DataFrame(checks)


def compare_prices(cg: pd.DataFrame, binance: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Порог расхождения должен быть положительным конечным числом")
    for name, frame in (("CoinGecko", cg), ("Binance", binance)):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.index.has_duplicates:
            raise ValueError(f"{name}: нужен уникальный индекс DatetimeIndex с timezone")
        if frame.columns.has_duplicates or set(COINS) - set(frame.columns):
            raise ValueError(f"{name}: отсутствуют монеты проекта или повторяются столбцы")
        utc_index = frame.index.tz_convert("UTC")
        if not (utc_index == utc_index.normalize()).all():
            raise ValueError(f"{name}: даты должны быть началом дня UTC")
    common = cg.index.intersection(binance.index).tz_convert("UTC").sort_values()
    if common.empty:
        raise ValueError("Нет общего периода CoinGecko и Binance")
    parts = []
    for coin in COINS:
        left = cg.loc[common, coin].to_numpy(dtype=float)
        right = binance.loc[common, coin].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
        diff = np.full(len(common), np.nan)
        diff[valid] = (left[valid] / right[valid] - 1) * 100
        parts.append(pd.DataFrame({
            "date": common,
            "coin": coin,
            "coingecko_timestamp_utc": common + DAY,
            "coingecko_price_usd": left,
            "binance_close_usdt": right,
            "relative_diff_pct": diff,
            "abs_diff_pct": np.abs(diff),
            "comparable": valid,
            "flag_large_difference": valid & (np.abs(diff) > threshold),
        }))
    details = pd.concat(parts, ignore_index=True)
    summaries = []
    for coin, group in details.groupby("coin", sort=False):
        diffs = group.loc[group["comparable"], "abs_diff_pct"]
        summaries.append({
            "coin": coin,
            "overlap_days": len(group),
            "compared_days": len(diffs),
            "missing_or_invalid_days": int((~group["comparable"]).sum()),
            "median_abs_diff_pct": diffs.median(),
            "p95_abs_diff_pct": diffs.quantile(0.95),
            "max_abs_diff_pct": diffs.max(),
            "flagged_days": int(group["flag_large_difference"].sum()),
            "threshold_pct": threshold,
        })
    return details, pd.DataFrame(summaries)


def write_frame(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=index)
        else:
            frame.to_csv(temporary, index=index)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-pct", type=float, default=2.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = {coin: json.loads((RAW_DIR / f"{coin}.json").read_text()) for coin in COINS}
    catalog = json.loads((RAW_DIR / "coins_list.json").read_text())
    mapping = pd.DataFrame(validate_mapping(catalog["data"]))
    mapping["catalog_fetched_at"] = catalog["fetched_at"]
    tables, quality = build_frames(records)
    prices = tables["prices"]
    binance_path = OUT_DIR / "prices.parquet"
    binance = pd.read_parquet(binance_path)
    details, summary = compare_prices(prices, binance, args.threshold_pct)
    flagged = details[details["flag_large_difference"]].sort_values("abs_diff_pct", ascending=False)

    for field, name in FIELDS.items():
        write_frame(tables[field], OUT_DIR / f"{name}.parquet")
    write_frame(mapping, OUT_DIR / "coingecko_ids.csv", index=False)
    write_frame(quality, OUT_DIR / "coingecko_quality.csv", index=False)
    write_frame(details, OUT_DIR / "coingecko_binance_comparison.parquet", index=False)
    write_frame(summary, OUT_DIR / "coingecko_binance_summary.csv", index=False)
    write_frame(flagged, OUT_DIR / "coingecko_binance_flags.csv", index=False)

    # JSON-манифест пишется последним: параметры, периоды и хэши для воспроизводимости.
    source_paths = [RAW_DIR / "coins_list.json", *(RAW_DIR / f"{coin}.json" for coin in COINS), binance_path]
    report = {
        "built_at": utc_now().isoformat(),
        "params": PARAMS,
        "coins": list(COINS),
        "daily_rows": len(prices),
        "date_start": str(prices.index.min().date()),
        "date_end": str(prices.index.max().date()),
        "timestamp_rule": "date D = CoinGecko D+1 00:00 UTC = end of Binance candle D",
        "availability_rule": "CoinGecko midnight point is used only after 00:10 UTC; end-of-day values must be lagged for trading",
        "fetched_at": {coin: record["fetched_at"] for coin, record in records.items()},
        "comparison_start": str(details["date"].min().date()),
        "comparison_end": str(details["date"].max().date()),
        "compared_pairs": int(details["comparable"].sum()),
        "missing_comparison_pairs": int((~details["comparable"]).sum()),
        "threshold_pct": args.threshold_pct,
        "flagged_pairs": len(flagged),
        "flagged_dates": int(flagged["date"].nunique()),
        "quality_issues": int((~quality["ok"]).sum()),
        "median_abs_diff_pct": float(details["abs_diff_pct"].median()) if details["comparable"].any() else None,
        "max_abs_diff_pct": float(details["abs_diff_pct"].max()) if details["comparable"].any() else None,
        "summary": json.loads(summary.to_json(orient="records")),
        "worst_differences": json.loads(details.nlargest(10, "abs_diff_pct").to_json(orient="records", date_format="iso")),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "limitations": [
            "Demo provides only a rolling 365-day window; no historical market cap is fabricated before it.",
            "CoinGecko prices are USD aggregates; Binance prices are single-exchange USDT closes, without USD/USDT adjustment.",
            "Large differences are investigation flags, not proof that a source is wrong; no flagged observations are removed.",
            "Missing and invalid values remain NaN; no filling or interpolation is used.",
            "CoinGecko volumes are aggregated rolling 24-hour USD volumes, not Binance quote_volume.",
            "POL uses polygon-ecosystem-token; the requested window follows the MATIC/POL migration.",
        ],
    }
    write_json(OUT_DIR / "coingecko_report.json", report)
    log.info("Сохранено %d дней × %d монет, %s — %s", len(prices), len(prices.columns), report["date_start"], report["date_end"])
    log.info("Binance: %d сравнений, %d флагов > %.2f%%, %d неполных/некорректных рядов", report["compared_pairs"], len(flagged), args.threshold_pct, report["quality_issues"])


if __name__ == "__main__":
    main()
