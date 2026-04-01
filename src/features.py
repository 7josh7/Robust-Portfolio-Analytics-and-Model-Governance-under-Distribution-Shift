from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_EXPOSURE_MAP = {
    "SPY": {"equity": 1.0, "us_equity": 1.0, "international_equity": 0.0, "duration": 0.0, "credit": 0.0, "real_assets": 0.0},
    "QQQ": {"equity": 1.0, "us_equity": 1.0, "international_equity": 0.0, "duration": 0.0, "credit": 0.0, "real_assets": 0.0},
    "IWM": {"equity": 1.0, "us_equity": 1.0, "international_equity": 0.0, "duration": 0.0, "credit": 0.0, "real_assets": 0.0},
    "EFA": {"equity": 1.0, "us_equity": 0.0, "international_equity": 1.0, "duration": 0.0, "credit": 0.0, "real_assets": 0.0},
    "EEM": {"equity": 1.0, "us_equity": 0.0, "international_equity": 1.0, "duration": 0.0, "credit": 0.0, "real_assets": 0.0},
    "TLT": {"equity": 0.0, "us_equity": 0.0, "international_equity": 0.0, "duration": 1.0, "credit": 0.0, "real_assets": 0.0},
    "IEF": {"equity": 0.0, "us_equity": 0.0, "international_equity": 0.0, "duration": 1.0, "credit": 0.0, "real_assets": 0.0},
    "LQD": {"equity": 0.0, "us_equity": 0.0, "international_equity": 0.0, "duration": 0.5, "credit": 0.5, "real_assets": 0.0},
    "HYG": {"equity": 0.0, "us_equity": 0.0, "international_equity": 0.0, "duration": 0.1, "credit": 0.9, "real_assets": 0.0},
    "GLD": {"equity": 0.0, "us_equity": 0.0, "international_equity": 0.0, "duration": 0.0, "credit": 0.0, "real_assets": 1.0},
    "DBC": {"equity": 0.0, "us_equity": 0.0, "international_equity": 0.0, "duration": 0.0, "credit": 0.0, "real_assets": 1.0},
    "VNQ": {"equity": 0.5, "us_equity": 0.5, "international_equity": 0.0, "duration": 0.0, "credit": 0.0, "real_assets": 0.5},
}


def herfindahl_index(weights: pd.Series | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    return float(np.square(values).sum())


def top_k_weight_share(weights: pd.Series | np.ndarray, k: int = 3) -> float:
    values = np.sort(np.asarray(weights, dtype=float))
    if values.size == 0:
        return np.nan
    return float(values[-k:].sum())


def effective_rank(covariance: pd.DataFrame | np.ndarray, floor: float = 1e-12) -> float:
    matrix = np.asarray(covariance, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, floor, None)
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def average_pairwise_correlation(covariance: pd.DataFrame | np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=float)
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(vol, vol)
    np.fill_diagonal(corr, np.nan)
    return float(np.nanmean(corr))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def rolling_drawdown(returns: pd.Series, window: int = 63) -> pd.Series:
    return returns.rolling(window).apply(max_drawdown, raw=False)


def build_bucket_exposure_history(
    weights_history: pd.DataFrame,
    exposure_map: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    exposure_map = exposure_map or DEFAULT_EXPOSURE_MAP
    exposure_frame = pd.DataFrame(exposure_map).T.fillna(0.0)
    aligned = exposure_frame.reindex(weights_history.columns).fillna(0.0)
    bucket_history = weights_history.fillna(0.0) @ aligned
    bucket_history.index = weights_history.index
    return bucket_history


def build_instability_feature_frame(
    simple_returns: pd.DataFrame,
    weights_history: pd.DataFrame,
    rebalance_results: pd.DataFrame,
    lookback: int = 63,
    include_optional_features: bool = False,
) -> pd.DataFrame:
    """Create trailing diagnostics for the governance monitoring model."""

    features: list[dict[str, float]] = []
    for date, weights in weights_history.iterrows():
        history = simple_returns.loc[:date].tail(lookback)
        if history.empty:
            continue

        aligned_weights = weights.reindex(history.columns).fillna(0.0)
        cov = history.cov().fillna(0.0)
        portfolio_returns = history.fillna(0.0) @ aligned_weights
        turnover_column = "proposed_turnover" if "proposed_turnover" in rebalance_results.columns else "turnover"
        recent_turnover = rebalance_results.loc[:date, turnover_column].tail(3).mean()
        recent_risk_gap = (
            rebalance_results.loc[:date, "realized_vol"] - rebalance_results.loc[:date, "forecast_vol"]
        ).tail(3).abs().mean()

        features.append(
            {
                "date": date,
                "trailing_vol": portfolio_returns.std() * np.sqrt(252),
                "average_pairwise_correlation": average_pairwise_correlation(cov),
                "effective_rank": effective_rank(cov),
                "recent_turnover": recent_turnover,
                "herfindahl": herfindahl_index(aligned_weights),
                "top_3_weight_share": top_k_weight_share(aligned_weights, k=3),
                "recent_drawdown": max_drawdown(portfolio_returns),
                "recent_drawdown_abs": abs(max_drawdown(portfolio_returns)),
                "forecast_realized_risk_gap": recent_risk_gap,
            }
        )
        if include_optional_features:
            features[-1]["cross_sectional_dispersion"] = history.std(axis=1).mean() * np.sqrt(252)
            features[-1]["missingness"] = history.isna().mean().mean()
            features[-1]["stale_price_proxy"] = (history.fillna(0.0).abs() < 1e-12).mean().mean()

    feature_frame = pd.DataFrame(features).set_index("date").sort_index()
    return feature_frame


def build_instability_targets(
    rebalance_results: pd.DataFrame,
    turnover_quantile: float = 0.90,
    risk_gap_quantile: float = 0.90,
    drawdown_quantile: float = 0.10,
    drawdown_threshold: float | None = None,
) -> pd.DataFrame:
    """Label future instability events for monitoring-model training."""

    turnover_column = "proposed_turnover" if "proposed_turnover" in rebalance_results.columns else "turnover"
    future_turnover = rebalance_results[turnover_column].shift(-1)
    future_risk_gap = (rebalance_results["realized_vol"] - rebalance_results["forecast_vol"]).abs().shift(-1)
    future_drawdown = rebalance_results["hold_period_drawdown"].shift(-1)

    def _rank_tail_indicator(series: pd.Series, quantile: float, upper_tail: bool) -> pd.Series:
        valid = series.dropna()
        if valid.empty:
            return pd.Series(index=series.index, dtype="float64")
        pct_rank = valid.rank(pct=True, method="average")
        indicator = (pct_rank >= quantile).astype(int) if upper_tail else (pct_rank <= quantile).astype(int)
        return indicator.reindex(series.index)

    turnover_target = _rank_tail_indicator(future_turnover, turnover_quantile, upper_tail=True)
    risk_gap_target = _rank_tail_indicator(future_risk_gap, risk_gap_quantile, upper_tail=True)
    if drawdown_threshold is None:
        drawdown_target = _rank_tail_indicator(future_drawdown, drawdown_quantile, upper_tail=False)
    else:
        drawdown_target = (future_drawdown <= drawdown_threshold).astype(float)
    combined_target = ((turnover_target == 1) | (risk_gap_target == 1) | (drawdown_target == 1)).astype(int)

    return pd.DataFrame(
        {
            "future_turnover": future_turnover,
            "future_risk_gap": future_risk_gap,
            "future_drawdown": future_drawdown,
            "turnover_instability_target": turnover_target,
            "risk_gap_instability_target": risk_gap_target,
            "drawdown_instability_target": drawdown_target,
            "combined_instability_target": combined_target,
        }
    )
