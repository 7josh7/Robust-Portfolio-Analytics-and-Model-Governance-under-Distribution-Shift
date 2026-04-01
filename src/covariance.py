from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import ensure_psd, equal_weight


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    values = returns.fillna(0.0).to_numpy()
    covariance = np.cov(values, rowvar=False, ddof=1)
    return ensure_psd(covariance)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    values = returns.fillna(0.0).to_numpy()
    covariance = LedoitWolf().fit(values).covariance_
    return ensure_psd(covariance)


def oas_covariance(returns: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import OAS

    values = returns.fillna(0.0).to_numpy()
    covariance = OAS().fit(values).covariance_
    return ensure_psd(covariance)


def ewma_covariance(returns: pd.DataFrame, span: int = 63) -> np.ndarray:
    values = returns.fillna(0.0).to_numpy()
    if len(values) == 0:
        raise ValueError("Cannot estimate EWMA covariance from an empty return matrix.")

    decay = 2.0 / (span + 1.0)
    raw_weights = np.array([(1.0 - decay) ** (len(values) - i - 1) for i in range(len(values))], dtype=float)
    weights = raw_weights / raw_weights.sum()
    mean = np.average(values, axis=0, weights=weights)
    demeaned = values - mean
    covariance = (demeaned * weights[:, None]).T @ demeaned
    return ensure_psd(covariance)


def factor_covariance(
    returns: pd.DataFrame,
    factor_returns: pd.Series | None = None,
) -> np.ndarray:
    """
    Simple one-factor covariance estimate using a market-factor proxy.

    If no external factor is provided, the equal-weight portfolio return is used
    as a market proxy. This keeps the implementation lightweight while still
    exposing a factor-style covariance branch for comparison.
    """

    asset_returns = returns.fillna(0.0)
    if factor_returns is None:
        factor_returns = asset_returns @ equal_weight(asset_returns.columns)
    factor = factor_returns.reindex(asset_returns.index).fillna(0.0).to_numpy()

    if np.var(factor) <= 1e-12:
        return sample_covariance(asset_returns)

    data = asset_returns.to_numpy()
    factor_var = float(np.var(factor, ddof=1))
    betas = np.array(
        [
            np.cov(data[:, column_idx], factor, ddof=1)[0, 1] / factor_var
            for column_idx in range(data.shape[1])
        ],
        dtype=float,
    )
    residuals = data - np.outer(factor, betas)
    residual_var = np.var(residuals, axis=0, ddof=1)
    covariance = np.outer(betas, betas) * factor_var + np.diag(residual_var)
    return ensure_psd(covariance)


def estimate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    ewma_span: int = 63,
    factor_returns: pd.Series | None = None,
) -> np.ndarray:
    method = method.lower()
    if method == "sample":
        return sample_covariance(returns)
    if method == "ledoit_wolf":
        return ledoit_wolf_covariance(returns)
    if method == "oas":
        return oas_covariance(returns)
    if method == "ewma":
        return ewma_covariance(returns, span=ewma_span)
    if method == "factor":
        return factor_covariance(returns, factor_returns=factor_returns)
    raise ValueError(f"Unknown covariance estimation method: {method}")
