from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
]


LARGE_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
    "XLF",
    "XLK",
    "XLE",
    "XLI",
    "XLP",
    "XLY",
    "XLV",
    "XLU",
    "XLB",
    "XLC",
    "TIP",
    "EMB",
    "BND",
    "AGG",
    "SHY",
    "MBB",
    "RWX",
    "IAU",
]


@dataclass
class DataBundle:
    prices: pd.DataFrame
    simple_returns: pd.DataFrame
    log_returns: pd.DataFrame


def download_price_data(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices for a liquid ETF universe."""

    import yfinance as yf

    tickers = list(tickers)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No price data was returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(-1):
            prices = raw.xs("Close", axis=1, level=-1)
        elif "Adj Close" in raw.columns.get_level_values(-1):
            prices = raw.xs("Adj Close", axis=1, level=-1)
        else:
            raise ValueError("Unable to locate a Close or Adj Close field in downloaded data.")
    else:
        prices = raw.rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def load_or_download_price_data(
    tickers: Iterable[str],
    start: str,
    end: str | None,
    raw_data_path: str | Path,
    auto_adjust: bool = True,
    progress: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a frozen price snapshot when available, otherwise download and cache it."""

    tickers = list(tickers)
    raw_path = Path(raw_data_path)

    if raw_path.exists() and not refresh:
        prices = pd.read_parquet(raw_path)
        prices.index = pd.to_datetime(prices.index)
    else:
        prices = download_price_data(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=progress,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(raw_path)

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Cached price snapshot is missing requested tickers: {missing}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    filtered = prices.loc[prices.index >= start_ts, tickers]
    if end_ts is not None:
        filtered = filtered.loc[filtered.index <= end_ts]
    return filtered.sort_index()


def clean_price_panel(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
    forward_fill_limit: int = 3,
) -> pd.DataFrame:
    """Apply conservative cleaning and drop assets with too much missing data."""

    cleaned = prices.copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    missingness = cleaned.isna().mean()
    keep_assets = missingness[missingness <= max_missing_frac].index.tolist()
    if not keep_assets:
        raise ValueError("All assets were dropped by the missingness filter.")

    cleaned = cleaned[keep_assets]
    cleaned = cleaned.ffill(limit=forward_fill_limit)
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.loc[:, cleaned.notna().any()]
    return cleaned


def compute_returns(prices: pd.DataFrame) -> DataBundle:
    """Compute simple and log returns from cleaned prices."""

    simple_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    log_returns = np.log(prices).diff().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    common_index = simple_returns.index.intersection(log_returns.index)
    return DataBundle(
        prices=prices.loc[common_index.min() : common_index.max()],
        simple_returns=simple_returns.loc[common_index],
        log_returns=log_returns.loc[common_index],
    )


def annualized_volatility(simple_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    return simple_returns.std().sort_values(ascending=False) * np.sqrt(periods_per_year)


def largest_absolute_moves(simple_returns: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    stacked = simple_returns.abs().stack().sort_values(ascending=False).head(top_n)
    result = stacked.reset_index()
    result.columns = ["date", "asset", "absolute_move"]
    return result


def split_adjusted_jump_flags(prices: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    jumps = prices.pct_change().abs()
    flagged = jumps.where(jumps >= threshold).stack().reset_index()
    flagged.columns = ["date", "asset", "absolute_price_jump"]
    return flagged.sort_values("absolute_price_jump", ascending=False)


def build_data_quality_report(
    prices: pd.DataFrame,
    simple_returns: pd.DataFrame,
    top_n_moves: int = 10,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Assemble tables used in the data-quality section of the notebook."""

    report: dict[str, pd.DataFrame | pd.Series] = {}
    report["missingness"] = prices.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    report["date_coverage"] = pd.DataFrame(
        {
            "start_date": prices.apply(lambda col: col.first_valid_index()),
            "end_date": prices.apply(lambda col: col.last_valid_index()),
            "observations": prices.notna().sum(),
        }
    ).sort_index()
    report["annualized_volatility"] = annualized_volatility(simple_returns).rename("annualized_volatility")
    report["largest_absolute_moves"] = largest_absolute_moves(simple_returns, top_n=top_n_moves)
    report["jump_flags"] = split_adjusted_jump_flags(prices)
    report["correlation"] = simple_returns.corr()
    return report
