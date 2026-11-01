"""Проверки временного выравнивания, пропусков и сетевых ошибок; без API-кредитов."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.binance_coins import COINS
from src.data.coingecko_build import calendar_for, compare_prices, daily_series
from src.data.coingecko_coins import COINGECKO_IDS, validate_mapping
from src.data.coingecko_download import Client, PARAMS, cached, load_api_key


def point(stamp, value):
    return [int(pd.Timestamp(stamp).timestamp() * 1000), value]


class DailyDataTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.date_range("2026-01-01", periods=3, tz="UTC", name="date")

    def test_midnight_is_previous_close_intraday_is_not_rounded(self):
        record = {"data": {"prices": [point("2026-01-02T00:00Z", 100),
                                     point("2026-01-04T00:00Z", 120),
                                     point("2026-01-04T15:30Z", 999)]}}
        series, quality = daily_series(record, "prices", self.calendar)
        self.assertEqual(series.iloc[0], 100)
        self.assertTrue(np.isnan(series.iloc[1]))
        self.assertEqual(series.iloc[2], 120)
        self.assertEqual(quality["missing_days"], 1)
        self.assertEqual(quality["discarded_intraday_points"], 1)

    def test_invalid_capitalization_becomes_missing_not_zero(self):
        record = {"data": {"market_caps": [point("2026-01-02T00:00Z", 0),
                                          point("2026-01-03T00:00Z", None),
                                          point("2026-01-04T00:00Z", -1)]}}
        series, quality = daily_series(record, "market_caps", self.calendar)
        self.assertTrue(series.isna().all())
        self.assertEqual(quality["invalid_values"], 3)

    def test_duplicate_timestamp_fails_instead_of_silently_choosing(self):
        record = {"data": {"prices": [point("2026-01-02T00:00Z", 100), point("2026-01-02T00:00Z", 200)]}}
        with self.assertRaisesRegex(ValueError, "повторяющиеся"):
            daily_series(record, "prices", self.calendar)

    def test_comparison_uses_aligned_dates_and_keeps_missing_pairs(self):
        cg = pd.DataFrame({coin: [100., 110., np.nan] for coin in COINS}, index=self.calendar)
        binance = pd.DataFrame({coin: [100., 100., 120.] for coin in COINS}, index=self.calendar)
        details, summary = compare_prices(cg, binance, 2)
        btc = details[details["coin"] == "BTC"]
        self.assertAlmostEqual(btc.iloc[0]["relative_diff_pct"], 0)
        self.assertAlmostEqual(btc.iloc[1]["relative_diff_pct"], 10)
        self.assertTrue(btc.iloc[1]["flag_large_difference"])
        self.assertFalse(btc.iloc[2]["comparable"])
        self.assertTrue(np.isnan(btc.iloc[2]["relative_diff_pct"]))
        self.assertTrue((summary["compared_days"] == 2).all())
        self.assertEqual(btc.iloc[0]["coingecko_timestamp_utc"], pd.Timestamp("2026-01-02", tz="UTC"))

    def test_naive_binance_dates_are_rejected(self):
        cg = pd.DataFrame(100., index=self.calendar, columns=list(COINS))
        binance = cg.copy()
        binance.index = binance.index.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone"):
            compare_prices(cg, binance, 2)

    def test_publication_delay_and_mixed_cache_days(self):
        records = {coin: {"params": PARAMS, "endpoint": f"/coins/{coin_id}/market_chart",
                          "fetched_at": "2026-11-01T00:05:00+00:00"}
                   for coin, coin_id in COINGECKO_IDS.items()}
        calendar = calendar_for(records)
        self.assertEqual(len(calendar), 365)
        self.assertEqual(calendar[-1], pd.Timestamp("2026-10-30", tz="UTC"))
        records["BTC"]["fetched_at"] = "2026-11-01T00:15:00+00:00"
        with self.assertRaisesRegex(ValueError, "разные дневные окна"):
            calendar_for(records)


class DownloadTests(unittest.TestCase):
    def test_key_can_be_read_without_dotenv_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('# comment\nCOINGECKO_DEMO_API_KEY="test-key"\n')
            with patch.dict("os.environ", {"COINGECKO_DEMO_API_KEY": ""}):
                self.assertEqual(load_api_key(path), "test-key")

    def test_cache_expires_at_publication_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BTC.json"
            path.write_text(json.dumps({"endpoint": "/test", "params": {}, "data": [1],
                                        "fetched_at": "2026-11-01T00:05:00+00:00"}))
            before = pd.Timestamp("2026-11-01T00:09:00Z").to_pydatetime()
            after = pd.Timestamp("2026-11-01T00:11:00Z").to_pydatetime()
            self.assertIsNotNone(cached(path, "/test", {}, before))
            self.assertIsNone(cached(path, "/test", {}, after))

    def test_rate_limit_retry_and_authentication_fail_fast(self):
        with patch("src.data.coingecko_download.load_api_key", return_value="test-key"), \
             patch("src.data.coingecko_download.requests.Session") as session_type, \
             patch("src.data.coingecko_download.time.sleep") as sleep:
            session = session_type.return_value
            limited = MagicMock(status_code=429, headers={"Retry-After": "7"})
            success = MagicMock(status_code=200)
            success.json.return_value = {"prices": [[1, 2]]}
            session.get.return_value.__enter__.side_effect = [limited, success]
            record = Client().get("/test", {})
            self.assertEqual(session.get.call_count, 2)
            sleep.assert_any_call(7.0)
            self.assertNotIn("test-key", json.dumps(record))
            self.assertFalse(session.get.call_args.kwargs["allow_redirects"])
            session.get.reset_mock()
            session.get.return_value.__enter__.side_effect = [MagicMock(status_code=401)]
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                Client().get("/test", {})
            self.assertEqual(session.get.call_count, 1)

    def test_mapping_uses_explicit_id_even_with_same_symbol(self):
        catalog = [{"id": coin_id, "symbol": coin.lower(), "name": coin}
                   for coin, coin_id in COINGECKO_IDS.items()]
        catalog.insert(0, {"id": "unrelated-bitcoin", "symbol": "btc", "name": "Wrong asset"})
        rows = validate_mapping(catalog)
        self.assertEqual(rows[0]["coingecko_id"], "bitcoin")
        catalog = [entry for entry in catalog if entry["id"] != "bitcoin"]
        with self.assertRaisesRegex(ValueError, "BTC"):
            validate_mapping(catalog)


if __name__ == "__main__":
    unittest.main()
