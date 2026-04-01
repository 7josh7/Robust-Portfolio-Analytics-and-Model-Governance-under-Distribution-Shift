from __future__ import annotations

import numpy as np
import pandas as pd


def inject_additive_gaussian_noise(returns: pd.DataFrame, noise_scale: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    additive_noise = rng.normal(0.0, returns.std().mean() * noise_scale, size=returns.shape)
    noisy = returns + additive_noise
    return pd.DataFrame(noisy, index=returns.index, columns=returns.columns)


def inject_volatility_scaled_noise(returns: pd.DataFrame, noise_scale: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    asset_scale = returns.std().replace(0.0, returns.std().mean()).to_numpy()
    additive_noise = rng.normal(0.0, noise_scale, size=returns.shape) * asset_scale
    noisy = returns + additive_noise
    return pd.DataFrame(noisy, index=returns.index, columns=returns.columns)


def inject_outlier_shocks(
    returns: pd.DataFrame,
    shock_scale: float,
    shock_probability: float = 0.01,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shock_mask = rng.uniform(size=returns.shape) < shock_probability
    shock_values = rng.standard_t(df=4, size=returns.shape) * returns.std().mean() * shock_scale
    shocked = returns + shock_mask * shock_values
    return pd.DataFrame(shocked, index=returns.index, columns=returns.columns)


def inject_block_missingness(
    returns: pd.DataFrame,
    missing_fraction: float,
    block_size: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    missing = returns.copy()
    n_rows = len(missing.index)
    n_assets = len(missing.columns)
    n_blocks = max(1, int(missing_fraction * n_rows * n_assets / max(block_size, 1)))

    for _ in range(n_blocks):
        asset = missing.columns[rng.integers(0, n_assets)]
        start = int(rng.integers(0, max(n_rows - block_size + 1, 1)))
        end = min(start + block_size, n_rows)
        missing.iloc[start:end, missing.columns.get_loc(asset)] = np.nan

    return missing


def inject_stale_price_returns(
    returns: pd.DataFrame,
    stale_assets: list[str] | None = None,
    stale_probability: float = 0.03,
    max_stale_days: int = 3,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stale = returns.copy()
    candidate_assets = stale_assets or [asset for asset in stale.columns if asset in {"LQD", "HYG", "IEF", "TLT", "VNQ"}]
    if not candidate_assets:
        candidate_assets = stale.columns.tolist()

    for asset in candidate_assets:
        asset_idx = stale.columns.get_loc(asset)
        row = 0
        while row < len(stale):
            if rng.uniform() < stale_probability:
                lag = int(rng.integers(1, max_stale_days + 1))
                end = min(row + lag, len(stale) - 1)
                carry = stale.iloc[row : end + 1, asset_idx].sum()
                stale.iloc[row:end, asset_idx] = 0.0
                stale.iloc[end, asset_idx] = carry
                row = end + 1
            else:
                row += 1
    return stale


def apply_missing_data_method(returns: pd.DataFrame, method: str) -> pd.DataFrame:
    method = method.lower()
    if method == "zero_fill":
        return returns.fillna(0.0)
    if method == "ffill_then_zero":
        return returns.ffill(limit=3).fillna(0.0)
    if method == "drop_sparse_assets":
        keep = returns.columns[returns.isna().mean() <= 0.10]
        return returns.loc[:, keep].fillna(0.0)
    raise ValueError(f"Unknown missing-data method: {method}")
