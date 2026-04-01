from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .covariance import estimate_covariance_matrix
from .regime import estimate_regime_conditioned_inputs, estimate_regime_probabilities
from .robust import _validation_score, solve_drmv_regularized_min_variance
from .targets import build_alpha_bar, build_nominal_target, build_regime_conditioned_target_params


def _selection_turnover(weights: pd.Series, previous_weights: pd.Series | None) -> float:
    if previous_weights is None:
        return 0.0
    aligned_previous = previous_weights.reindex(weights.index).fillna(0.0)
    return float((weights - aligned_previous).abs().sum())


def prepare_regime_conditioned_inputs(
    train_returns: pd.DataFrame,
    lookback: int = 252,
    n_regimes: int = 2,
    covariance_method: str = "ledoit_wolf",
    random_state: int = 7,
) -> dict[str, Any]:
    market_factor = train_returns.fillna(0.0).mean(axis=1).rename("market_factor")
    regime_probs = estimate_regime_probabilities(
        market_factor_series=market_factor,
        n_regimes=n_regimes,
        lookback=min(lookback, len(train_returns)),
        random_state=random_state,
    )
    return estimate_regime_conditioned_inputs(
        asset_returns=train_returns,
        factor_returns=market_factor,
        regime_probs=regime_probs,
        lookback=min(lookback, len(train_returns)),
        covariance_method=covariance_method,
    )


def tune_drmv_regularized_min_variance(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    delta_grid: list[float],
    alpha_bar_scale_grid: list[float],
    covariance_methods: list[str],
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    p_norm: int | float = 2,
    target_method: str = "benchmark_fraction",
    target_scale: float = 0.50,
    target_quantile: float = 0.40,
    fixed_target_return: float = 0.0002,
    alpha_bar_rule: str = "delta_adjusted",
    selection_turnover_penalty_weight: float = 1.0,
    selection_risk_gap_penalty_weight: float = 2.0,
    selection_constraint_penalty_weight: float = 1.0,
    selection_fallback_penalty_weight: float = 10.0,
    metric: str = "composite",
    benchmark_returns: pd.Series | None = None,
    mean_returns: pd.Series | None = None,
    covariance: np.ndarray | None = None,
    regime_conditioned: bool = False,
    stressed_target_scale: float = 0.85,
    stressed_delta_scale: float = 1.25,
    stressed_probability: float = 0.0,
    solver: str = "SCS",
    rebalance_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Jointly tune DRMV ambiguity size, robust target adjustment, and covariance engine.

    This is a practical selector rather than the paper's full data-driven
    inference procedure, but it preserves the paper's key structure: delta and
    alpha_bar are chosen together and alpha_bar sits below the nominal target.
    """

    nominal_target, target_source = build_nominal_target(
        train_returns=train_returns,
        method=target_method,
        scale=target_scale,
        quantile=target_quantile,
        benchmark_returns=benchmark_returns,
        fixed_target_return=fixed_target_return,
    )

    best_result: dict[str, Any] | None = None
    best_score = -np.inf
    diagnostics_rows: list[dict[str, Any]] = []

    covariance_candidates = [("provided", covariance)] if covariance is not None else [
        (method, estimate_covariance_matrix(train_returns, method=method))
        for method in covariance_methods
    ]
    mean_vector = mean_returns.reindex(train_returns.columns).fillna(0.0) if mean_returns is not None else train_returns.mean().fillna(0.0)

    for covariance_method, covariance_matrix in covariance_candidates:
        for delta in delta_grid:
            for alpha_scale in alpha_bar_scale_grid:
                if regime_conditioned:
                    target_params = build_regime_conditioned_target_params(
                        rho=nominal_target,
                        delta=delta,
                        stressed_probability=stressed_probability,
                        alpha_bar_rule=alpha_bar_rule,
                        alpha_bar_scale=alpha_scale,
                        stressed_target_scale=stressed_target_scale,
                        stressed_delta_scale=stressed_delta_scale,
                    )
                    working_delta = float(target_params["delta"])
                    alpha_bar = float(target_params["alpha_bar"])
                else:
                    alpha_bar, _ = build_alpha_bar(
                        rho=nominal_target,
                        delta=delta,
                        rule=alpha_bar_rule,
                        scale=alpha_scale,
                    )
                    working_delta = float(delta)

                result = solve_drmv_regularized_min_variance(
                    mean_returns=mean_vector,
                    covariance=covariance_matrix,
                    delta=working_delta,
                    alpha_bar=alpha_bar,
                    p_norm=p_norm,
                    long_only=True,
                    lower_bound=bounds[0],
                    upper_bound=bounds[1],
                    previous_weights=previous_weights,
                    turnover_penalty=turnover_penalty,
                    solver=solver,
                    rebalance_date=rebalance_date,
                )

                val_portfolio_returns = val_returns.fillna(0.0) @ result["weights"]
                base_metric = _validation_score(val_portfolio_returns, metric="sharpe" if metric == "composite" else metric)
                validation_turnover = _selection_turnover(result["weights"], previous_weights)
                realized_validation_vol = float(val_portfolio_returns.std() * np.sqrt(252)) if not val_portfolio_returns.empty else np.nan
                forecast_vol = float(result.get("forecast_vol", np.nan))
                risk_gap = abs(realized_validation_vol - forecast_vol) if pd.notna(realized_validation_vol) and pd.notna(forecast_vol) else np.nan
                binding_penalty = selection_constraint_penalty_weight * float(
                    np.nan_to_num(max(1e-6 - float(result.get("binding_margin", np.nan)), 0.0), nan=1.0)
                )
                fallback_penalty = selection_fallback_penalty_weight * float(result.get("fallback_used", False))
                score = (
                    base_metric
                    - selection_turnover_penalty_weight * float(np.nan_to_num(validation_turnover, nan=0.0))
                    - selection_risk_gap_penalty_weight * float(np.nan_to_num(risk_gap, nan=0.0))
                    - binding_penalty
                    - fallback_penalty
                )

                result.update(
                    {
                        "validation_score": float(score),
                        "nominal_target_return": float(nominal_target),
                        "target_return": float(alpha_bar),
                        "target_source": target_source,
                        "target_rule": alpha_bar_rule,
                        "covariance_method": covariance_method,
                        "regime_conditioned": bool(regime_conditioned),
                        "stressed_probability": float(stressed_probability),
                        "alpha_bar_scale": float(alpha_scale),
                    }
                )
                diagnostics_rows.append(
                    {
                        "covariance_method": covariance_method,
                        "delta": float(working_delta),
                        "base_delta": float(delta),
                        "alpha_bar": float(alpha_bar),
                        "alpha_bar_scale": float(alpha_scale),
                        "validation_score": float(score),
                        "base_metric": float(base_metric),
                        "validation_turnover": float(validation_turnover),
                        "validation_risk_gap": float(risk_gap),
                        "binding_margin": float(result.get("binding_margin", np.nan)),
                        "binding_penalty": float(binding_penalty),
                        "fallback_penalty": float(fallback_penalty),
                        "status": str(result.get("status", "unknown")),
                    }
                )

                if best_result is None or score > best_score + 1e-12:
                    best_result = result
                    best_score = score
                elif best_result is not None and abs(score - best_score) <= 1e-12:
                    if float(result.get("binding_margin", -np.inf)) > float(best_result.get("binding_margin", -np.inf)):
                        best_result = result

    if best_result is None:
        raise ValueError("DRMV tuning failed to produce any candidate solution.")

    best_result["parameter_diagnostics"] = pd.DataFrame(diagnostics_rows)
    return best_result
