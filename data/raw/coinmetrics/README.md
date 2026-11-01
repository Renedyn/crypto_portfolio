# Coin Metrics Community

Источник: **Coin Metrics**, [Community API v4](https://docs.coinmetrics.io/api/v4/). Данные предоставлены по [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) для некоммерческого использования с указанием источника. [Официальный архив и условия](https://github.com/coinmetrics/data).

Снимок получен 2026-10-31 UTC. В `assets.json` сохранены названия активов, в `catalog.json` — доступность дневных метрик. Файлы `<asset>.json` содержат 21 исходный актив: 19 обычных монет и отдельные `matic`, `pol`, из которых строится один столбец проекта `POL`.

Параметры: `metrics=PriceUSD,CapMrktEstUSD`, `frequency=1d`, `start_time=2020-01-01`, `end_time=2026-10-30`. У каждого актива собственная дата начала доступности; это не обещание наличия значений за весь календарь.

JSON-конверт содержит источник, endpoint, параметры, время начала/окончания загрузки и массив `pages`. Каждая страница API сохранена без изменения, включая строки с `null`; числовые строки сохраняют исходную точность. Все `next_page_url` были пройдены, последняя страница не должна содержать продолжение. Ключ не требуется.

`PriceUSD` — цена закрытия указанного дня D в USD. `CapMrktEstUSD` — капитализация по сообщаемому обращающемуся предложению в USD, а не `CapMrktCurUSD`. Timestamp `D 00:00 UTC` здесь является меткой дня, в отличие от исходных точек CoinGecko; при обработке Coin Metrics сдвига на сутки нет.

```bash
uv run python -m src.data.coinmetrics_download --end 2026-10-30
uv run python -m src.data.coinmetrics_build
```

`--force` обновляет снимок, `--coins POL` выгружает оба актива Polygon. Запросы выполняются с паузой 6,1 секунды и повторными попытками; повтор в тот же день использует кэш. Обработка сохранённого снимка работает без сети.

В объединённом POL источник меняется с MATIC на POL 2024-09-04, по дате миграции Polygon PoS. Капитализация в это время имеет структурные разрывы из-за раздельного учёта предложения. Они сохранены, а не сглажены: см. [аудит перехода](../../processed/coinmetrics_polygon_transition.csv). Многолетние результаты — [цены](../../processed/coinmetrics_prices.parquet) и [капитализация](../../processed/coinmetrics_market_caps.parquet); [пропуски](../../processed/coinmetrics_gaps.csv), [манифест с хэшами](../../processed/coinmetrics_report.json). Подробные результаты сверки и ограничения описаны в [README проекта](../../../README.md).
