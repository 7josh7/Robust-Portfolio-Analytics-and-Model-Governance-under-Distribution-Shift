from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


def ensure_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    eigenvalues = np.clip(eigenvalues, floor, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "sample",
    ewma_span: int = 63,
) -> np.ndarray:
    if returns.empty:
        raise ValueError("Cannot estimate covariance from an empty return matrix.")

    values = returns.fillna(0.0).to_numpy()

    if method == "sample":
        covariance = np.cov(values, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        covariance = LedoitWolf().fit(values).covariance_
    elif method == "ewma":
        weights = np.exp(np.linspace(-1.0, 0.0, len(values)))
        weights = weights / weights.sum()
        mean = np.average(values, axis=0, weights=weights)
        demeaned = values - mean
        covariance = (demeaned * weights[:, None]).T @ demeaned
    else:
        raise ValueError(f"Unknown covariance estimation method: {method}")

    return ensure_psd(covariance)


def estimate_expected_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean().fillna(0.0)


def equal_weight(columns: pd.Index) -> pd.Series:
    n_assets = len(columns)
    weights = np.repeat(1.0 / n_assets, n_assets)
    return pd.Series(weights, index=columns, name="weight")


def inverse_volatility_weight(returns: pd.DataFrame, floor: float = 1e-6) -> pd.Series:
    vol = returns.std().clip(lower=floor)
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    return weights.rename("weight")


def solve_min_variance(
    covariance: np.ndarray,
    assets: pd.Index,
    mu: pd.Series | None = None,
    target_return: float | None = None,
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    solver: str = "SCS",
) -> dict[str, Any]:
    """Solve a long-only min-variance or target-return portfolio."""

    n_assets = len(assets)
    lower_bound, upper_bound = bounds
    w = cp.Variable(n_assets)
    objective = cp.quad_form(w, ensure_psd(covariance))

    if previous_weights is not None and turnover_penalty > 0.0:
        objective += turnover_penalty * cp.norm1(w - previous_weights.reindex(assets).fillna(0.0).to_numpy())

    constraints = [cp.sum(w) == 1.0, w >= lower_bound, w <= upper_bound]
    if mu is not None and target_return is not None:
        constraints.append(mu.reindex(assets).fillna(0.0).to_numpy() @ w >= target_return)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver, warm_start=True)

    if w.value is None:
        weights = equal_weight(assets)
    else:
        weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
        weights = weights.clip(lower=0.0)
        weights = weights / weights.sum()

    forecast_vol = float(np.sqrt(weights.to_numpy() @ ensure_psd(covariance) @ weights.to_numpy()) * np.sqrt(252))
    return {
        "weights": weights,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "forecast_vol": forecast_vol,
        "expected_return": float(mu.reindex(assets).fillna(0.0) @ weights) if mu is not None else np.nan,
    }


def solve_mean_variance(
    covariance: np.ndarray,
    assets: pd.Index,
    mu: pd.Series,
    risk_aversion: float = 10.0,
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    solver: str = "SCS",
) -> dict[str, Any]:
    """Solve a long-only sample mean-variance portfolio."""

    n_assets = len(assets)
    lower_bound, upper_bound = bounds
    w = cp.Variable(n_assets)

    utility = mu.reindex(assets).fillna(0.0).to_numpy() @ w - risk_aversion * cp.quad_form(w, ensure_psd(covariance))
    if previous_weights is not None and turnover_penalty > 0.0:
        utility -= turnover_penalty * cp.norm1(w - previous_weights.reindex(assets).fillna(0.0).to_numpy())

    constraints = [cp.sum(w) == 1.0, w >= lower_bound, w <= upper_bound]
    problem = cp.Problem(cp.Maximize(utility), constraints)
    problem.solve(solver=solver, warm_start=True)

    if w.value is None:
        weights = equal_weight(assets)
    else:
        weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
        weights = weights.clip(lower=0.0)
        weights = weights / weights.sum()

    forecast_vol = float(np.sqrt(weights.to_numpy() @ ensure_psd(covariance) @ weights.to_numpy()) * np.sqrt(252))
    return {
        "weights": weights,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "forecast_vol": forecast_vol,
        "expected_return": float(mu.reindex(assets).fillna(0.0) @ weights),
    }


def fit_equal_weight(train_returns: pd.DataFrame, **_: Any) -> dict[str, Any]:
    weights = equal_weight(train_returns.columns)
    return {
        "weights": weights,
        "status": "closed_form",
        "objective_value": np.nan,
        "forecast_vol": float((train_returns @ weights).std() * np.sqrt(252)),
        "expected_return": float(train_returns.mean() @ weights),
    }


def fit_inverse_volatility(train_returns: pd.DataFrame, **_: Any) -> dict[str, Any]:
    weights = inverse_volatility_weight(train_returns)
    return {
        "weights": weights,
        "status": "closed_form",
        "objective_value": np.nan,
        "forecast_vol": float((train_returns @ weights).std() * np.sqrt(252)),
        "expected_return": float(train_returns.mean() @ weights),
    }


def fit_sample_min_variance(
    train_returns: pd.DataFrame,
    target_return: float | None = None,
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    solver: str = "SCS",
) -> dict[str, Any]:
    mu = estimate_expected_returns(train_returns)
    cov = estimate_covariance(train_returns, method="sample")
    return solve_min_variance(
        covariance=cov,
        assets=train_returns.columns,
        mu=mu,
        target_return=target_return,
        bounds=bounds,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        solver=solver,
    )


def fit_shrinkage_min_variance(
    train_returns: pd.DataFrame,
    target_return: float | None = None,
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    solver: str = "SCS",
) -> dict[str, Any]:
    mu = estimate_expected_returns(train_returns)
    cov = estimate_covariance(train_returns, method="ledoit_wolf")
    return solve_min_variance(
        covariance=cov,
        assets=train_returns.columns,
        mu=mu,
        target_return=target_return,
        bounds=bounds,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        solver=solver,
    )


def fit_sample_mean_variance(
    train_returns: pd.DataFrame,
    bounds: tuple[float, float] = (0.0, 0.35),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    risk_aversion: float = 6.0,
    solver: str = "SCS",
) -> dict[str, Any]:
    mu = estimate_expected_returns(train_returns)
    cov = estimate_covariance(train_returns, method="sample")
    return solve_mean_variance(
        covariance=cov,
        assets=train_returns.columns,
        mu=mu,
        risk_aversion=risk_aversion,
        bounds=bounds,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        solver=solver,
    )
