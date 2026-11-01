"""Coin Metrics Community, без ключа: PriceUSD и CapMrktEstUSD с 2020 года.

    uv run python -m src.data.coinmetrics_download
    uv run python -m src.data.coinmetrics_download --coins POL --force
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import logging
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests

from src.data.binance_coins import START_DATE
from src.data.coingecko_download import ROOT, retry_delay, utc_now, write_json
from src.data.coinmetrics_coins import ASSETS, METRICS, all_assets, validate_catalog

BASE_URL = "https://community-api.coinmetrics.io/v4"
RAW_DIR = ROOT / "data" / "raw" / "coinmetrics"
TIMESERIES = "/timeseries/asset-metrics"
CATALOG = "/catalog-v2/asset-metrics"
log = logging.getLogger("coinmetrics")


def data_rows(record: dict) -> list[dict]:
    pages = record.get("pages")
    if not isinstance(pages, list) or not pages or pages[-1].get("next_page_url"):
        raise ValueError("Неполная пагинация Coin Metrics")
    rows = []
    for page in pages:
        if not isinstance(page.get("data"), list):
            raise ValueError("Coin Metrics: ожидается массив data в каждой странице")
        rows.extend(page["data"])
    return rows


class Client:
    def __init__(self, pause: float = 6.1):
        self.pause = pause
        self.last_request = float("-inf")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "crypto-portfolio/0.1"})

    def get_page(self, url: str, params: dict | None = None) -> dict:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "community-api.coinmetrics.io" or not parts.path.startswith("/v4/"):
            raise ValueError("Пагинация должна оставаться на официальном Community API")
        for attempt in range(1, 6):
            time.sleep(max(0.0, self.pause - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with self.session.get(url, params=params, timeout=(10, 60), allow_redirects=False) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        log.warning("HTTP %d, попытка %d/5", response.status_code, attempt)
                        if attempt < 5:
                            time.sleep(retry_delay(response.headers.get("Retry-After"), attempt))
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f"Coin Metrics: HTTP {response.status_code} для {parts.path}")
                    page = response.json()
                    if not isinstance(page, dict) or not isinstance(page.get("data"), list):
                        raise ValueError("Ответ не содержит массива data")
                    return page
            except (requests.RequestException, ValueError) as exc:
                log.warning("%s, попытка %d/5", type(exc).__name__, attempt)
                if attempt < 5:
                    time.sleep(retry_delay(None, attempt))
        raise RuntimeError("Coin Metrics: исчерпаны пять попыток; сохранённый кэш не изменён")

    def download(self, endpoint: str, params: dict) -> dict:
        pages, seen = [], set()
        url = BASE_URL + endpoint
        query = params
        started = utc_now().isoformat()
        while url:
            if url in seen:
                raise ValueError("Циклическая пагинация Coin Metrics")
            seen.add(url)
            page = self.get_page(url, query)
            pages.append(page)
            url, query = page.get("next_page_url"), None
        return {"source": "coinmetrics_community", "endpoint": endpoint, "params": params,
                "started_at": started, "fetched_at": utc_now().isoformat(), "pages": pages}


def cached(path: Path, endpoint: str, params: dict) -> dict | None:
    try:
        record = json.loads(path.read_text())
        fetched = datetime.fromisoformat(record["fetched_at"])
        now = utc_now()
        if fetched.tzinfo is None or fetched > now or fetched.date() != now.date():
            return None
        if record["endpoint"] == endpoint and record["params"] == params and data_rows(record):
            return record
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", nargs="+", choices=list(ASSETS), default=list(ASSETS))
    parser.add_argument("--end", type=date.fromisoformat, default=utc_now().date() - timedelta(days=1), help="последняя дата закрытия UTC, YYYY-MM-DD")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pause", type=float, default=6.1)
    args = parser.parse_args()
    if not math.isfinite(args.pause) or args.pause < 6.1:
        parser.error("--pause должен быть конечным числом не меньше 6.1 секунды")
    if not date.fromisoformat(START_DATE) <= args.end < utc_now().date():
        parser.error("--end должен быть завершённым UTC днём начиная с 2020-01-01")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = Client(args.pause)
    try:
        reference_params = {"assets": ",".join(all_assets()), "page_size": 10000}
        reference_path = RAW_DIR / "assets.json"
        reference = None if args.force else cached(reference_path, "/reference-data/assets", reference_params)
        if reference is None:
            reference = client.download("/reference-data/assets", reference_params)
            names = {row["asset"]: row.get("full_name") for row in data_rows(reference)}
            if any(not names.get(asset) for asset in all_assets()):
                raise ValueError("Неполный каталог названий Coin Metrics")
            write_json(reference_path, reference)
        catalog_params = {"assets": ",".join(all_assets()), "metrics": ",".join(METRICS), "page_size": 10000}
        path = RAW_DIR / "catalog.json"
        catalog = None if args.force else cached(path, CATALOG, catalog_params)
        if catalog is None:
            catalog = client.download(CATALOG, catalog_params)
            validate_catalog(data_rows(catalog))
            write_json(path, catalog)
        validate_catalog(data_rows(catalog))
        log.info("Community: доступны %d активов и обе метрики", len(all_assets()))
        assets = list(dict.fromkeys(asset for coin in args.coins for asset in ASSETS[coin]))
        for asset in assets:
            params = {"assets": asset, "metrics": ",".join(METRICS), "frequency": "1d",
                      "start_time": START_DATE, "end_time": args.end.isoformat(), "page_size": 10000}
            path = RAW_DIR / f"{asset}.json"
            record = None if args.force else cached(path, TIMESERIES, params)
            if record is not None:
                log.info("%s: свежий кэш", asset)
                continue
            record = client.download(TIMESERIES, params)
            rows = data_rows(record)
            if not rows or any(row.get("asset") != asset or not row.get("time") for row in rows):
                raise ValueError(f"{asset}: пустой или неверный ответ; прежний кэш сохранён")
            write_json(path, record)
            log.info("%s: %d строк, %d страниц", asset, len(rows), len(record["pages"]))
    finally:
        client.session.close()


if __name__ == "__main__":
    main()
