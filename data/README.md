`raw_prices.parquet` is the frozen market-data snapshot used by the notebook and CLI runner.

`large_universe_raw_prices.parquet` is the cached snapshot used by the larger-universe notebook extension.

The workflow reads from these cache files by default and only downloads from `yfinance` when a file is missing or `refresh_data: true` is set in the config.
