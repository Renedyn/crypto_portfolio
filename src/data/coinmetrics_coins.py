"""Активы Coin Metrics и явное правило объединения MATIC/POL."""

from src.data.binance_coins import COINS

ASSETS = {
    "BTC": ["btc"], "ETH": ["eth"], "BNB": ["bnb"], "XRP": ["xrp"],
    "ADA": ["ada"], "SOL": ["sol"], "DOGE": ["doge"], "TRX": ["trx"],
    "DOT": ["dot"], "LTC": ["ltc"], "LINK": ["link"], "AVAX": ["avax"],
    "POL": ["matic", "pol"], "XLM": ["xlm"], "ATOM": ["atom"],
    "UNI": ["uni"], "ETC": ["etc"], "BCH": ["bch"], "FIL": ["fil"], "NEAR": ["near"],
}
METRICS = {"PriceUSD": "prices", "CapMrktEstUSD": "market_caps"}
# День миграции Polygon PoS; это не дата возобновления торгов Binance (13 сентября).
# Метки 1d Coin Metrics обозначают закрытие самого дня D.
POL_SWITCH_DATE = "2024-09-04"
POL_MIGRATION_URL = "https://polygon.technology/blog/matic-to-pol-migration-is-now-live-everything-you-need-to-know"


def all_assets() -> list[str]:
    if set(ASSETS) != set(COINS):
        raise ValueError("Список монет проекта изменился: обновите Coin Metrics ASSETS")
    return [asset for coin in COINS for asset in ASSETS[coin]]


def validate_catalog(rows: list[dict]) -> list[dict]:
    by_asset = {row["asset"]: row for row in rows}
    coverage = []
    for asset in all_assets():
        entry = by_asset.get(asset, {})
        for metric in METRICS:
            item = next((m for m in entry.get("metrics", []) if m["metric"] == metric), {})
            frequency = next((f for f in item.get("frequencies", []) if f["frequency"] == "1d" and f.get("community")), None)
            if frequency is None:
                raise ValueError(f"В Community нет дневной метрики {asset}/{metric}")
            coverage.append({"asset": asset, "metric": metric, **frequency})
    return coverage
