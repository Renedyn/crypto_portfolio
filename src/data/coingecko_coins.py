"""Явное соответствие активов проекта CoinGecko ID, без поиска по первому тикеру."""

from src.data.binance_coins import COINS

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "TRX": "tron",
    "DOT": "polkadot",
    "LTC": "litecoin",
    "LINK": "chainlink",
    "AVAX": "avalanche-2",
    # Нативный POL после миграции, не legacy MATIC (matic-network).
    "POL": "polygon-ecosystem-token",
    "XLM": "stellar",
    "ATOM": "cosmos",
    "UNI": "uniswap",
    "ETC": "ethereum-classic",
    "BCH": "bitcoin-cash",
    "FIL": "filecoin",
    "NEAR": "near",
}


def validate_mapping(coins_list: list[dict]) -> list[dict]:
    """Проверить заранее выбранные ID по каталогу; совпадения символа недостаточно."""
    if set(COINGECKO_IDS) != set(COINS):
        raise ValueError("Список Binance изменился: обновите COINGECKO_IDS вручную")
    by_id = {entry["id"]: entry for entry in coins_list}
    rows = []
    for coin, coin_id in COINGECKO_IDS.items():
        entry = by_id.get(coin_id)
        if entry is None or entry["symbol"].upper() != coin:
            raise ValueError(f"Перепроверьте соответствие {coin} -> {coin_id}")
        rows.append({
            "coin": coin,
            "binance_pairs": "+".join(COINS[coin]),
            "coingecko_id": coin_id,
            "coingecko_symbol": entry["symbol"],
            "coingecko_name": entry["name"],
            "source_url": f"https://www.coingecko.com/en/coins/{coin_id}",
            "notes": "POL after MATIC migration; no legacy MATIC history requested" if coin == "POL" else "",
        })
    return rows
