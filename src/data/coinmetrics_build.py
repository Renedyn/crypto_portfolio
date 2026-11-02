"""Офлайн-сборка Coin Metrics, аудит пропусков/MATIC→POL и сверка с CoinGecko.

    uv run python -m src.data.coinmetrics_build

1d PriceUSD/CapMrktEstUSD за день D уже соответствуют закрытию D.
В отличие от сырых CoinGecko timestamps, даты Coin Metrics НЕ сдвигаются.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from src.data.binance_coins import COINS, START_DATE
from src.data.coingecko_build import write_frame
from src.data.coingecko_download import ROOT, utc_now, write_json
from src.data.coinmetrics_coins import ASSETS, METRICS, POL_MIGRATION_URL, POL_SWITCH_DATE, all_assets, validate_catalog
from src.data.coinmetrics_download import RAW_DIR, TIMESERIES, data_rows

OUT_DIR = ROOT / "data" / "processed"
DAY = pd.Timedelta(days=1)
log = logging.getLogger("coinmetrics")


def calendar_for(records: dict) -> pd.DatetimeIndex:
    if set(records) != set(all_assets()):
        raise ValueError("Нужны все 21 исходных актива, включая MATIC и POL")
    ends = set()
    for asset, record in records.items():
        p = record["params"]
        if (record["source"] != "coinmetrics_community" or record["endpoint"] != TIMESERIES
                or p.get("assets") != asset or p.get("metrics") != ",".join(METRICS)
                or p.get("frequency") != "1d" or p.get("start_time") != START_DATE):
            raise ValueError(f"Неподходящие параметры кэша {asset}")
        end = pd.Timestamp(p["end_time"], tz="UTC")
        fetched = pd.Timestamp(record["fetched_at"])
        if fetched.tzinfo is None or not pd.Timestamp(START_DATE, tz="UTC") <= end < fetched.tz_convert("UTC").normalize():
            raise ValueError(f"{asset}: окно должно завершаться закрытым днём UTC")
        ends.add(end)
    if len(ends) != 1:
        raise ValueError("Разные конечные даты в кэше: повторите coinmetrics_download для всех монет")
    return pd.date_range(pd.Timestamp(START_DATE, tz="UTC"), ends.pop(), freq="D", name="date")


def parse_asset(record: dict, asset: str, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = data_rows(record)
    if not rows or any(row.get("asset") != asset for row in rows):
        raise ValueError(f"{asset}: пустые данные или чужой актив")
    frame = pd.DataFrame(rows)
    if "time" not in frame or any(metric not in frame for metric in METRICS):
        raise ValueError(f"{asset}: нет временных меток или одной из требуемых метрик")
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="raise"), name="date")
    if stamps.isna().any() or stamps.has_duplicates or not (stamps == stamps.normalize()).all():
        raise ValueError(f"{asset}: дубли, пустые или недневные timestamps")
    if ((stamps < calendar[0]) | (stamps > calendar[-1])).any():
        raise ValueError(f"{asset}: строки за пределами запрошенного окна")
    frame.index = stamps
    parsed = pd.DataFrame(index=stamps)
    invalid = pd.DataFrame(index=stamps)
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").astype(float)
        bad = (~np.isfinite(values) | (values <= 0)) & frame[metric].notna()
        parsed[metric] = values.mask(~np.isfinite(values) | (values <= 0))
        invalid[metric] = bad
    return parsed.sort_index().reindex(calendar), invalid.reindex(calendar, fill_value=False)


def select_sources(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    sources = pd.DataFrame(index=calendar)
    for coin in COINS:
        if coin == "POL":
            sources[coin] = np.where(calendar < pd.Timestamp(POL_SWITCH_DATE, tz="UTC"), "matic", "pol")
        else:
            sources[coin] = ASSETS[coin][0]
    return sources


def quality_for(series: pd.Series, coin: str, metric: str, sources: pd.Series, invalid: pd.Series) -> tuple[dict, list[dict]]:
    valid = series.notna()
    first, last = series.first_valid_index(), series.last_valid_index()
    kinds = pd.Series("present", index=series.index)
    kinds.loc[~valid] = "internal"
    if first is None:
        kinds[:] = "all_missing"
    else:
        kinds.loc[series.index < first] = "leading"
        kinds.loc[series.index > last] = "trailing"
    # Разбиваем интервалы и на границе источников; не прячем null за сменой тикера.
    groups = (kinds.ne(kinds.shift()) | sources.ne(sources.shift())).cumsum()
    gaps = []
    for _, indices in kinds.groupby(groups).groups.items():
        kind = kinds.loc[indices[0]]
        if kind != "present":
            gaps.append({"coin": coin, "metric": metric, "source_asset": sources.loc[indices[0]],
                         "kind": kind, "start_date": indices[0].strftime("%Y-%m-%d"),
                         "end_date": indices[-1].strftime("%Y-%m-%d"), "days": len(indices)})
    return {
        "coin": coin, "metric": metric, "requested_days": len(series),
        "valid_days": int(valid.sum()), "missing_days": int((~valid).sum()),
        "leading_missing_days": int((kinds == "leading").sum()),
        "internal_missing_days": int((kinds == "internal").sum()),
        "trailing_missing_days": int((kinds == "trailing").sum()),
        "all_missing": first is None, "invalid_values": int(invalid.sum()),
        "first_valid_date": first.strftime("%Y-%m-%d") if first is not None else "",
        "last_valid_date": last.strftime("%Y-%m-%d") if last is not None else "",
    }, gaps


def build_frames(records: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    calendar = calendar_for(records)
    parsed, invalid = {}, {}
    for asset in all_assets():
        parsed[asset], invalid[asset] = parse_asset(records[asset], asset, calendar)
    sources = select_sources(calendar)
    tables = {metric: pd.DataFrame(index=calendar) for metric in METRICS}
    checks, gaps = [], []
    for coin in COINS:
        for metric in METRICS:
            series = pd.Series(np.nan, index=calendar, dtype=float)
            bad = pd.Series(False, index=calendar)
            for asset in ASSETS[coin]:
                chosen = sources[coin] == asset
                series.loc[chosen] = parsed[asset].loc[chosen, metric]
                bad.loc[chosen] = invalid[asset].loc[chosen, metric]
            tables[metric][coin] = series
            check, intervals = quality_for(series, coin, metric, sources[coin], bad)
            switches = sources[coin].ne(sources[coin].shift()) & sources[coin].shift().notna()
            check["source_switches"] = int(switches.sum())
            check["requires_migration_review"] = coin == "POL" and metric == "CapMrktEstUSD" and bool(switches.any())
            checks.append(check)
            gaps.extend(intervals)
    gap_columns = ["coin", "metric", "source_asset", "kind", "start_date", "end_date", "days"]
    return tables, pd.DataFrame(checks), pd.DataFrame(gaps, columns=gap_columns), sources, parsed


def compare(tables: dict, references: dict, price_threshold: float, cap_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if any(not np.isfinite(v) or v <= 0 for v in (price_threshold, cap_threshold)):
        raise ValueError("Пороги должны быть положительными конечными числами")
    parts, summaries = [], []
    for metric, cm in tables.items():
        cg = references[metric]
        for name, frame in (("Coin Metrics", cm), ("CoinGecko", cg)):
            if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None
                    or frame.index.has_duplicates or frame.columns.has_duplicates or set(COINS) - set(frame.columns)):
                raise ValueError(f"{name}: неверные монеты или индекс")
            utc = frame.index.tz_convert("UTC")
            if utc.isna().any() or not (utc == utc.normalize()).all():
                raise ValueError(f"{name}: индекс должен быть дневным UTC")
        common = cm.index.intersection(cg.index).tz_convert("UTC").sort_values()
        if common.empty:
            raise ValueError(f"{metric}: нет общего периода с CoinGecko")
        threshold = price_threshold if metric == "PriceUSD" else cap_threshold
        for coin in COINS:
            left, right = cm.loc[common, coin].to_numpy(float), cg.loc[common, coin].to_numpy(float)
            valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
            diff = np.full(len(common), np.nan)
            diff[valid] = (left[valid] / right[valid] - 1) * 100
            flags = valid & (np.abs(diff) > threshold)
            parts.append(pd.DataFrame({"date": common, "coin": coin, "metric": metric,
                                       "coingecko_timestamp_utc": common + DAY,
                                       "coinmetrics_usd": left, "coingecko_usd": right,
                                       "relative_diff_pct": diff, "abs_diff_pct": np.abs(diff),
                                       "comparable": valid, "flag_large_difference": flags}))
            abs_diff = pd.Series(np.abs(diff[valid]))
            summaries.append({"coin": coin, "metric": metric, "overlap_days": len(common),
                              "compared_days": int(valid.sum()), "missing_or_invalid_days": int((~valid).sum()),
                              "median_abs_diff_pct": abs_diff.median(), "p95_abs_diff_pct": abs_diff.quantile(.95),
                              "max_abs_diff_pct": abs_diff.max(), "flagged_days": int(flags.sum()), "threshold_pct": threshold})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summaries)


def alignment_diagnostics(cm: pd.DataFrame, cg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset in (-1, 0, 1):
        shifted = cm.copy()
        shifted.index = shifted.index + offset * DAY
        dates = shifted.index.intersection(cg.index)
        differences = ((shifted.loc[dates] / cg.loc[dates] - 1) * 100).abs()
        for coin in COINS:
            values = differences[coin].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"coin": coin, "extra_date_shift_days": offset, "compared_days": len(values),
                         "median_abs_diff_pct": values.median()})
    return pd.DataFrame(rows)


def polygon_transition(parsed: dict, tables: dict, sources: pd.DataFrame) -> pd.DataFrame:
    window = pd.date_range("2024-08-28", "2024-09-20", tz="UTC", name="date").intersection(sources.index)
    audit = pd.DataFrame(index=window)
    audit["chosen_asset"] = sources.loc[window, "POL"]
    for metric in METRICS:
        for asset in ("matic", "pol"):
            audit[f"{asset}_{metric}"] = parsed[asset].loc[window, metric]
        audit[f"chosen_{metric}"] = tables[metric].loc[window, "POL"]
        audit[f"pol_vs_matic_{metric}_pct"] = (audit[f"pol_{metric}"] / audit[f"matic_{metric}"] - 1) * 100
        audit[f"chosen_{metric}_change_pct"] = tables[metric]["POL"].pct_change(fill_method=None).reindex(window) * 100
    implied_supply = tables["CapMrktEstUSD"]["POL"] / tables["PriceUSD"]["POL"]
    audit["chosen_implied_circulating_supply"] = implied_supply.reindex(window)
    audit["implied_supply_change_pct"] = implied_supply.pct_change(fill_method=None).reindex(window) * 100
    audit["source_switch"] = (sources["POL"].ne(sources["POL"].shift()) & sources["POL"].shift().notna()).reindex(window)
    audit["flag_supply_change_gt_10pct"] = audit["implied_supply_change_pct"].abs() > 10
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-threshold-pct", type=float, default=2)
    parser.add_argument("--cap-threshold-pct", type=float, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = {asset: json.loads((RAW_DIR / f"{asset}.json").read_text()) for asset in all_assets()}
    catalog = json.loads((RAW_DIR / "catalog.json").read_text())
    coverage = pd.DataFrame(validate_catalog(data_rows(catalog)))
    asset_names = {row["asset"]: row["full_name"] for row in data_rows(json.loads((RAW_DIR / "assets.json").read_text()))}
    tables, quality, gaps, sources, parsed = build_frames(records)
    reference_paths = {"PriceUSD": OUT_DIR / "coingecko_prices.parquet", "CapMrktEstUSD": OUT_DIR / "market_caps.parquet"}
    references = {metric: pd.read_parquet(path) for metric, path in reference_paths.items()}
    details, summary = compare(tables, references, args.price_threshold_pct, args.cap_threshold_pct)
    flags = details[details["flag_large_difference"]].sort_values("abs_diff_pct", ascending=False)
    transition = polygon_transition(parsed, tables, sources)
    alignment = alignment_diagnostics(tables["PriceUSD"], references["PriceUSD"])
    mappings = []
    for coin in COINS:
        for asset in ASSETS[coin]:
            used = sources.index[sources[coin] == asset]
            mappings.append({"coin": coin, "coinmetrics_asset": asset, "name": asset_names[asset],
                             "selected_from": used.min().strftime("%Y-%m-%d") if len(used) else "",
                             "selected_through": used.max().strftime("%Y-%m-%d") if len(used) else "",
                             "migration_source": POL_MIGRATION_URL if coin == "POL" else ""})
    for metric, name in METRICS.items():
        write_frame(tables[metric], OUT_DIR / f"coinmetrics_{name}.parquet")
    for frame, filename in ((quality, "coinmetrics_quality.csv"), (gaps, "coinmetrics_gaps.csv"),
                            (coverage, "coinmetrics_coverage.csv"), (pd.DataFrame(mappings), "coinmetrics_ids.csv"),
                            (details, "coinmetrics_coingecko_comparison.parquet"), (summary, "coinmetrics_coingecko_summary.csv"),
                            (flags, "coinmetrics_coingecko_flags.csv"), (alignment, "coinmetrics_alignment.csv")):
        write_frame(frame, OUT_DIR / filename, index=False)
    write_frame(sources, OUT_DIR / "coinmetrics_sources.parquet")
    write_frame(transition, OUT_DIR / "coinmetrics_polygon_transition.csv")
    stats = {}
    for metric, table in tables.items():
        checks = quality[quality["metric"] == metric]
        compared = details[(details["metric"] == metric) & details["comparable"]]
        complete = table.index[table.notna().all(axis=1)]
        stats[metric] = {
            "missing_cells": int(table.isna().sum().sum()), "complete_days": len(complete),
            "first_complete_day": complete[0].strftime("%Y-%m-%d") if len(complete) else None,
            "leading_missing_cells": int(checks["leading_missing_days"].sum()),
            "internal_missing_cells": int(checks["internal_missing_days"].sum()),
            "trailing_missing_cells": int(checks["trailing_missing_days"].sum()),
            "compared_pairs": len(compared), "flagged_pairs": int(compared["flag_large_difference"].sum()),
            "median_abs_diff_pct": float(compared["abs_diff_pct"].median()) if len(compared) else None,
            "max_abs_diff_pct": float(compared["abs_diff_pct"].max()) if len(compared) else None,
        }
    inputs = [RAW_DIR / "catalog.json", RAW_DIR / "assets.json", *(RAW_DIR / f"{asset}.json" for asset in all_assets()), *reference_paths.values()]
    report = {
        "built_at": utc_now().isoformat(), "source": "Coin Metrics Community API",
        "license": "CC BY-NC 4.0; attribution required, non-commercial use",
        "metrics": METRICS, "date_start": str(sources.index.min().date()), "date_end": str(sources.index.max().date()),
        "rows": len(sources), "coins": list(COINS), "source_assets": all_assets(),
        "comparison_start": str(details["date"].min().date()), "comparison_end": str(details["date"].max().date()),
        "thresholds_pct": {"PriceUSD": args.price_threshold_pct, "CapMrktEstUSD": args.cap_threshold_pct},
        "statistics": stats,
        "timestamp_rule": "Coin Metrics 1d time D labels end-of-day D; compare to processed CoinGecko D (raw D+1 00:00 UTC). No shift of Coin Metrics.",
        "polygon_rule": {"switch_date": POL_SWITCH_DATE, "source": POL_MIGRATION_URL,
                         "rule": "matic before migration day; pol from migration day. No fallback, no sum of capitalizations, no rebasing.",
                         "binance_pol_listing_date": "2024-09-13"},
        "polygon_transition": json.loads(transition.reset_index().to_json(orient="records", date_format="iso")),
        "summary": json.loads(summary.to_json(orient="records")),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "limitations": [
            "The 2020 calendar does not imply each asset or metric existed from January 1; leading, internal and trailing gaps remain NaN.",
            "CapMrktEstUSD uses reported circulating supply, not total supply CapMrktCurUSD or fully diluted valuation.",
            "Coin Metrics may source circulating supply via CoinGecko; the capitalization comparison is not fully independent.",
            "The 1:1 MATIC/POL conversion does not guarantee equal reported circulating capitalization; overlapping caps must not be added.",
            "The stitched POL capitalization has migration-related supply/level breaks; it is not a continuous economic capitalization index. Inspect coinmetrics_polygon_transition.csv before using it for portfolio weights.",
            "Current revised history is not a point-in-time publication archive. End-of-day values cannot be used to trade earlier that day.",
            "Difference flags identify observations for review, not a proven error in either provider; no observations are filled, scaled or removed automatically.",
        ],
    }
    write_json(OUT_DIR / "coinmetrics_report.json", report)
    log.info("Сохранено %d дней × %d монет, %s — %s", len(sources), len(COINS), report["date_start"], report["date_end"])
    for metric, stats_row in stats.items():
        log.info("%s: %d пропусков (%d внутренних), %d сравнений с CG, %d флагов", metric,
                 stats_row["missing_cells"], stats_row["internal_missing_cells"], stats_row["compared_pairs"], stats_row["flagged_pairs"])


if __name__ == "__main__":
    main()
