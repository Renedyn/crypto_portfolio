"""Список монет проекта и соответствующих пар Binance.

Ключ — тикер монеты, под которым она будет в столбцах итоговых таблиц.
Значение — список спотовых пар к USDT в хронологическом порядке. Если монету
переименовали (новый тикер на бирже), старая пара идёт первой: её история
склеивается с новой, а на пересекающихся датах приоритет у более поздней пары.

Стейблкоины не включаем: для портфеля они — аналог кэша.
"""

COINS: dict[str, list[str]] = {
    "BTC": ["BTCUSDT"],
    "ETH": ["ETHUSDT"],
    "BNB": ["BNBUSDT"],
    "XRP": ["XRPUSDT"],
    "ADA": ["ADAUSDT"],
    "SOL": ["SOLUSDT"],    # торги на Binance с 2020-08-11
    "DOGE": ["DOGEUSDT"],
    "TRX": ["TRXUSDT"],
    "DOT": ["DOTUSDT"],    # с 2020-08-18
    "LTC": ["LTCUSDT"],
    "LINK": ["LINKUSDT"],
    "AVAX": ["AVAXUSDT"],  # с 2020-09-22
    # Polygon: MATIC -> POL (миграция токена 1:1, сентябрь 2024)
    "POL": ["MATICUSDT", "POLUSDT"],
    "XLM": ["XLMUSDT"],
    "ATOM": ["ATOMUSDT"],
    "UNI": ["UNIUSDT"],    # с 2020-09-17
    "ETC": ["ETCUSDT"],
    "BCH": ["BCHUSDT"],    # BCHABC -> BCH в 2019, до начала нашего периода
    "FIL": ["FILUSDT"],    # с 2020-10-15
    "NEAR": ["NEARUSDT"],  # с 2020-10-14
}

START_DATE = "2020-01-01"
