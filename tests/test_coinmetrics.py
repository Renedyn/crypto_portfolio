"""Проверки календаря, миграции, пропусков и полной пагинации Coin Metrics."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.binance_coins import COINS, START_DATE
from src.data.coinmetrics_build import build_frames, calendar_for, compare, parse_asset, polygon_transition, quality_for
from src.data.coinmetrics_coins import METRICS, all_assets
from src.data.coinmetrics_download import BASE_URL, TIMESERIES, Client, cached, data_rows


def record(asset, rows, end="2024-09-05"):
    return {"source": "coinmetrics_community", "endpoint": TIMESERIES,
            "fetched_at": "2024-09-06T12:00:00+00:00",
            "params": {"assets": asset, "metrics": ",".join(METRICS), "frequency": "1d",
                       "start_time": START_DATE, "end_time": end},
            "pages": [{"data": rows}]}


def row(asset, day, price="1", cap="100"):
    return {"asset": asset, "time": day + "T00:00:00.000000000Z", "PriceUSD": price, "CapMrktEstUSD": cap}


class BuildTests(unittest.TestCase):
    def test_coinmetrics_day_is_not_shifted(self):
        calendar = pd.date_range("2021-01-01", periods=3, tz="UTC", name="date")
        r = record("btc", [row("btc", "2021-01-01", "29380.69"), row("btc", "2021-01-03", "33277.84")])
        parsed, _ = parse_asset(r, "btc", calendar)
        self.assertEqual(parsed.loc["2021-01-01", "PriceUSD"], 29380.69)
        self.assertTrue(np.isnan(parsed.loc["2021-01-02", "PriceUSD"]))

    def test_migration_uses_fixed_date_no_sum_or_fallback(self):
        records = {asset: record(asset, [row(asset, d) for d in ("2024-09-03", "2024-09-04", "2024-09-05")]) for asset in all_assets()}
        records["matic"] = record("matic", [row("matic", d, "2", "200") for d in ("2024-09-03", "2024-09-04", "2024-09-05")])
        records["pol"] = record("pol", [row("pol", "2024-09-03", "3", "300"),
                                         row("pol", "2024-09-04", "3", None), row("pol", "2024-09-05", "3", "300")])
        tables, _, gaps, sources, _ = build_frames(records)
        self.assertEqual(tables["PriceUSD"].loc["2024-09-03", "POL"], 2)
        self.assertEqual(tables["PriceUSD"].loc["2024-09-04", "POL"], 3)
        self.assertTrue(np.isnan(tables["CapMrktEstUSD"].loc["2024-09-04", "POL"]))
        self.assertEqual(tables["CapMrktEstUSD"].loc["2024-09-05", "POL"], 300)
        self.assertEqual(sources.loc["2024-09-04", "POL"], "pol")
        gap = gaps[(gaps.coin == "POL") & (gaps.metric == "CapMrktEstUSD") & (gaps.kind == "internal")]
        self.assertEqual(gap.iloc[0]["start_date"], "2024-09-04")

    def test_gap_classification_distinguishes_missing_edges(self):
        dates = pd.date_range("2020-01-01", periods=6, tz="UTC")
        values = pd.Series([np.nan, 1, np.nan, 1, np.nan, np.nan], index=dates)
        quality, gaps = quality_for(values, "BTC", "PriceUSD", pd.Series("btc", index=dates), pd.Series(False, index=dates))
        self.assertEqual(quality["leading_missing_days"], 1)
        self.assertEqual(quality["internal_missing_days"], 1)
        self.assertEqual(quality["trailing_missing_days"], 2)
        self.assertEqual(sum(gap["days"] for gap in gaps), 4)

    def test_migration_capitalization_break_is_visible_in_audit(self):
        dates = pd.date_range("2024-09-03", periods=3, tz="UTC", name="date")
        tables = {"PriceUSD": pd.DataFrame({"POL": [2., 3., 3.]}, index=dates),
                  "CapMrktEstUSD": pd.DataFrame({"POL": [200., 6., 9.]}, index=dates)}
        parsed = {"matic": pd.DataFrame({"PriceUSD": 2., "CapMrktEstUSD": 200.}, index=dates),
                  "pol": pd.DataFrame({"PriceUSD": 3., "CapMrktEstUSD": [5., 6., 9.]}, index=dates)}
        sources = pd.DataFrame({"POL": ["matic", "pol", "pol"]}, index=dates)
        audit = polygon_transition(parsed, tables, sources)
        self.assertAlmostEqual(audit.loc["2024-09-04", "chosen_CapMrktEstUSD_change_pct"], -97.)
        self.assertTrue(audit.loc["2024-09-04", "source_switch"])
        self.assertTrue(audit.loc["2024-09-04", "flag_supply_change_gt_10pct"])
        self.assertEqual(audit.loc["2024-09-04", "chosen_CapMrktEstUSD"], 6.)

    def test_null_and_invalid_cap_are_not_filled_or_mixed_with_total_supply(self):
        dates = pd.date_range("2020-01-01", periods=3, tz="UTC")
        rows = [row("btc", "2020-01-01", cap=None), row("btc", "2020-01-02", cap="0"), row("btc", "2020-01-03", cap="-1")]
        for item in rows:
            item["CapMrktCurUSD"] = "999"
        parsed, invalid = parse_asset(record("btc", rows), "btc", dates)
        self.assertTrue(parsed["CapMrktEstUSD"].isna().all())
        self.assertEqual(invalid["CapMrktEstUSD"].sum(), 2)

    def test_duplicate_or_foreign_asset_rejected(self):
        dates = pd.date_range("2020-01-01", periods=1, tz="UTC")
        with self.assertRaisesRegex(ValueError, "дубли"):
            parse_asset(record("btc", [row("btc", "2020-01-01")] * 2), "btc", dates)
        with self.assertRaisesRegex(ValueError, "чужой"):
            parse_asset(record("btc", [row("eth", "2020-01-01")]), "btc", dates)

    def test_mixed_end_dates_and_unclosed_days_rejected(self):
        records = {asset: record(asset, [row(asset, "2024-09-04")]) for asset in all_assets()}
        records["btc"]["params"]["end_time"] = "2024-09-04"
        with self.assertRaisesRegex(ValueError, "Разные конечные"):
            calendar_for(records)
        records["btc"]["params"]["end_time"] = "2024-09-06"
        with self.assertRaisesRegex(ValueError, "закрытым"):
            calendar_for(records)

    def test_comparison_matches_processed_cg_days_with_separate_thresholds(self):
        dates = pd.date_range("2026-01-01", periods=3, tz="UTC", name="date")
        cm = {metric: pd.DataFrame({coin: [100., 104., np.nan] for coin in COINS}, index=dates) for metric in METRICS}
        cg = {metric: pd.DataFrame(100., columns=list(COINS), index=dates) for metric in METRICS}
        details, summary = compare(cm, cg, 2, 5)
        self.assertEqual(len(details), 120)
        self.assertEqual(summary[summary.metric == "PriceUSD"].flagged_days.sum(), 20)
        self.assertEqual(summary[summary.metric == "CapMrktEstUSD"].flagged_days.sum(), 0)
        self.assertEqual(summary.missing_or_invalid_days.sum(), 40)
        self.assertEqual(details.iloc[0].coingecko_timestamp_utc, dates[0] + pd.Timedelta(days=1))
        self.assertEqual(details.iloc[0].relative_diff_pct, 0)


class DownloadTests(unittest.TestCase):
    def test_all_pages_are_followed_and_preserved(self):
        client = Client()
        next_url = BASE_URL + TIMESERIES + "?next_page_token=second"
        pages = [{"data": [row("btc", "2020-01-01")], "next_page_url": next_url},
                 {"data": [row("btc", "2020-01-02")]}]
        with patch.object(client, "get_page", side_effect=pages) as get:
            result = client.download(TIMESERIES, {"assets": "btc"})
        self.assertEqual(len(data_rows(result)), 2)
        self.assertEqual(result["pages"], pages)
        get.assert_any_call(next_url, None)
        client.session.close()

    def test_incomplete_or_cyclic_pagination_rejected(self):
        with self.assertRaisesRegex(ValueError, "Неполная"):
            data_rows({"pages": [{"data": [], "next_page_url": "pending"}]})
        client = Client()
        with patch.object(client, "get_page", return_value={"data": [], "next_page_url": BASE_URL + TIMESERIES}):
            with self.assertRaisesRegex(ValueError, "Циклическая"):
                client.download(TIMESERIES, {})
        client.session.close()

    def test_redirect_host_and_server_error_handling(self):
        with patch("src.data.coinmetrics_download.requests.Session") as session_type, \
             patch("src.data.coinmetrics_download.time.sleep") as sleep:
            client = Client()
            with self.assertRaisesRegex(ValueError, "официальном"):
                client.get_page("https://example.org/v4/timeseries/asset-metrics")
            session = session_type.return_value
            session.get.assert_not_called()
            failed = MagicMock(status_code=503, headers={"Retry-After": "9"})
            success = MagicMock(status_code=200)
            success.json.return_value = {"data": []}
            session.get.return_value.__enter__.side_effect = [failed, success]
            self.assertEqual(client.get_page(BASE_URL + TIMESERIES), {"data": []})
            sleep.assert_any_call(9.)
            self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    def test_changed_end_date_does_not_reuse_cache(self):
        r = record("btc", [row("btc", "2024-09-04")])
        with tempfile.TemporaryDirectory() as directory, patch("src.data.coinmetrics_download.utc_now", return_value=pd.Timestamp("2024-09-06T13:00Z").to_pydatetime()):
            path = Path(directory) / "btc.json"
            path.write_text(json.dumps(r))
            self.assertIsNotNone(cached(path, TIMESERIES, r["params"]))
            self.assertIsNone(cached(path, TIMESERIES, {**r["params"], "end_time": "2024-09-04"}))


if __name__ == "__main__":
    unittest.main()
