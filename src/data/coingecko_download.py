"""CoinGecko Demo: каталог ID и год дневных цен, капитализации, объёмов.

    uv run python -m src.data.coingecko_download
    uv run python -m src.data.coingecko_download --coins BTC ETH --force

Сырые ответы сохранены без изменений внутри JSON-конверта с метаданными.
Повтор в тот же день использует кэш; --force обновляет его принудительно.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import os
from pathlib import Path
import tempfile
import time

import requests

from src.data.coingecko_coins import COINGECKO_IDS, validate_mapping

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "coingecko"
BASE_URL = "https://api.coingecko.com/api/v3"
PARAMS = {"vs_currency": "usd", "days": 365, "interval": "daily", "precision": "full"}
log = logging.getLogger("coingecko")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_day(timestamp: datetime):
    """До 00:10 UTC последняя дневная точка ещё может быть не опубликована."""
    return (timestamp.astimezone(timezone.utc) - timedelta(minutes=10)).date()


def load_api_key(path: Path = ROOT / ".env") -> str:
    key = os.environ.get("COINGECKO_DEMO_API_KEY", "").strip()
    if not key and path.exists():
        for line in path.read_text().splitlines():
            name, sep, value = line.removeprefix("export ").partition("=")
            if sep and name.strip() == "COINGECKO_DEMO_API_KEY":
                value = value.strip()
                key = value[1:-1] if value[:1] in ("'", '"') and value[-1:] == value[:1] else value.split(" #", 1)[0].strip()
                break
    if not key or "\n" in key or "\r" in key:
        raise ValueError("Задайте COINGECKO_DEMO_API_KEY в окружении или .env")
    return key


def write_json(path: Path, value: object) -> None:
    """Атомарная запись: сетевой сбой не портит предыдущую выгрузку."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - utc_now()).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60.0, 2.0 ** attempt)


class Client:
    def __init__(self, pause: float = 2.5):
        self.pause = pause
        self.last_request = float("-inf")
        self.session = requests.Session()
        self.session.headers.update({
            "x-cg-demo-api-key": load_api_key(),
            "Accept": "application/json",
            "User-Agent": "crypto-portfolio/0.1",
        })

    def get(self, endpoint: str, params: dict) -> dict:
        for attempt in range(1, 6):
            time.sleep(max(0.0, self.pause - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                # Не пересылаем заголовок с ключом на возможный redirect.
                with self.session.get(BASE_URL + endpoint, params=params, timeout=(10, 60), allow_redirects=False) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        delay = retry_delay(response.headers.get("Retry-After"), attempt)
                        log.warning("%s: HTTP %d, попытка %d/5", endpoint, response.status_code, attempt)
                        if attempt < 5:
                            time.sleep(delay)
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f"CoinGecko {endpoint}: HTTP {response.status_code}; проверьте ключ, тариф и параметры")
                    data = response.json()
            except (requests.RequestException, ValueError) as exc:
                # Не логируем объект запроса/ответа или заголовки авторизации.
                log.warning("%s: %s, попытка %d/5", endpoint, type(exc).__name__, attempt)
                if attempt < 5:
                    time.sleep(retry_delay(None, attempt))
                continue
            return {"source": "coingecko", "endpoint": endpoint, "params": params,
                    "fetched_at": utc_now().isoformat(), "data": data}
        raise RuntimeError(f"CoinGecko {endpoint}: исчерпаны 5 попыток; сохранённый кэш не изменён")


def cached(path: Path, endpoint: str, params: dict, now: datetime, *, catalog: bool = False) -> dict | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        fetched = datetime.fromisoformat(record["fetched_at"])
        if fetched.tzinfo is None or fetched > now:
            return None
        fresh = now - fetched < timedelta(days=7) if catalog else snapshot_day(fetched) == snapshot_day(now)
        if fresh and record["endpoint"] == endpoint and record["params"] == params and record.get("data"):
            return record
    except (ValueError, KeyError, TypeError):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", nargs="+", choices=list(COINGECKO_IDS), default=list(COINGECKO_IDS))
    parser.add_argument("--force", action="store_true", help="обновить даже свежий кэш")
    parser.add_argument("--pause", type=float, default=2.5, help="минимальная пауза между запросами, секунды")
    args = parser.parse_args()
    if not args.pause >= 2.0:
        parser.error("--pause должен быть не меньше 2 секунд")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = Client(args.pause)
    try:
        path = RAW_DIR / "coins_list.json"
        catalog = None if args.force else cached(path, "/coins/list", {}, utc_now(), catalog=True)
        if catalog is None:
            catalog = client.get("/coins/list", {})
            validate_mapping(catalog["data"])
            write_json(path, catalog)
        rows = validate_mapping(catalog["data"])
        log.info("Проверены %d явных соответствий ID по /coins/list", len(rows))
        for coin in dict.fromkeys(args.coins):
            endpoint = f"/coins/{COINGECKO_IDS[coin]}/market_chart"
            path = RAW_DIR / f"{coin}.json"
            record = None if args.force else cached(path, endpoint, PARAMS, utc_now())
            if record is not None:
                log.info("%s: свежий кэш", coin)
                continue
            record = client.get(endpoint, PARAMS)
            if not isinstance(record["data"], dict) or not all(record["data"].get(field) for field in ("prices", "market_caps", "total_volumes")):
                raise ValueError(f"{coin}: неполный ответ market_chart; прежний кэш сохранён")
            write_json(path, record)
            log.info("%s (%s): сохранено %d точек цены", coin, COINGECKO_IDS[coin], len(record["data"]["prices"]))
    finally:
        client.session.close()


if __name__ == "__main__":
    main()
