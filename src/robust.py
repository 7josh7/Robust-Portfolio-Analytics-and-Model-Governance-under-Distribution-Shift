from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .baselines import (
    ensure_psd,
    equal_weight,
    estimate_covariance,
    estimate_expected_returns,
    fit_sample_min_variance,
    inverse_volatility_weight,
    solve_min_variance,
)
from .covariance import estimate_covariance_matrix
from .features import herfindahl_index
from .targets import build_nominal_target


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _WassersteinProxyDppProgram:
    problem: cp.Problem
    weights: cp.Variable
    slack: cp.Variable
    sqrt_covariance: Any
    mean_returns: cp.Parameter
    epsilon: cp.Parameter
    target_return: cp.Parameter
    previous_weights: cp.Parameter
    turnover_penalty: cp.Parameter
    slack_penalty: cp.Parameter


@dataclass(slots=True)
class _DrmvDppProgram:
    problem: cp.Problem
    weights: cp.Variable
    sqrt_covariance: Any
    mean_returns: cp.Parameter
    sqrt_delta: cp.Parameter
    alpha_bar: cp.Parameter
    previous_weights: cp.Parameter
    turnover_penalty: cp.Parameter


def _selection_turnover(weights: pd.Series, previous_weights: pd.Series | None) -> float:
    if previous_weights is None:
        return 0.0
    aligned_previous = previous_weights.reindex(weights.index).fillna(0.0)
    return float((weights - aligned_previous).abs().sum())


def compute_dynamic_target_return(
    train_returns: pd.DataFrame,
    mode: str = "equal_weight_fraction",
    scale: float = 0.50,
    quantile: float = 0.40,
    fixed_target_return: float = 0.0002,
) -> tuple[float, str]:
    """Compute a practical target-return proxy that avoids brittle fixed hurdles."""
    return build_nominal_target(
        train_returns=train_returns,
        method=mode,
        scale=scale,
        quantile=quantile,
        fixed_target_return=fixed_target_return,
    )


def _resolve_mean_and_covariance(
    train_returns: pd.DataFrame,
    covariance_method: str = "ledoit_wolf",
    mean_returns: pd.Series | None = None,
    covariance: np.ndarray | None = None,
) -> tuple[pd.Series, np.ndarray]:
    resolved_mean = mean_returns.reindex(train_returns.columns).fillna(0.0) if mean_returns is not None else estimate_expected_returns(train_returns)
    resolved_covariance = covariance if covariance is not None else estimate_covariance_matrix(train_returns, method=covariance_method)
    return resolved_mean, ensure_psd(resolved_covariance)


def _matrix_square_root(covariance: np.ndarray) -> np.ndarray:
    covariance_psd = ensure_psd(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_psd)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _build_wasserstein_proxy_dpp_program(
    covariance_factor: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> _WassersteinProxyDppProgram:
    n_assets = covariance_factor.shape[1]
    w = cp.Variable(n_assets)
    s = cp.Variable(nonneg=True)
    robust_radius = cp.Variable(nonneg=True)
    turnover_abs = cp.Variable(n_assets, nonneg=True)
    mean_returns = cp.Parameter(n_assets, value=np.zeros(n_assets))
    epsilon = cp.Parameter(nonneg=True, value=0.0)
    target_return = cp.Parameter(value=0.0)
    previous_weights = cp.Parameter(n_assets, value=np.zeros(n_assets))
    turnover_penalty = cp.Parameter(nonneg=True, value=0.0)
    slack_penalty = cp.Parameter(nonneg=True, value=1.0)

    objective = cp.sum_squares(covariance_factor @ w) + slack_penalty * s + turnover_penalty * cp.sum(turnover_abs)
    constraints = [
        cp.sum(w) == 1.0,
        w >= lower_bound,
        w <= upper_bound,
        cp.norm(w, 2) <= robust_radius,
        turnover_abs >= w - previous_weights,
        turnover_abs >= previous_weights - w,
        mean_returns @ w - epsilon * robust_radius + s >= target_return,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if not problem.is_dpp():
        raise ValueError("Wasserstein proxy DPP program is not DPP-compliant.")
    return _WassersteinProxyDppProgram(
        problem=problem,
        weights=w,
        slack=s,
        sqrt_covariance=covariance_factor,
        mean_returns=mean_returns,
        epsilon=epsilon,
        target_return=target_return,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
        slack_penalty=slack_penalty,
    )


def _build_drmv_dpp_program(
    covariance_factor: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    p_norm: int | float,
) -> _DrmvDppProgram:
    n_assets = covariance_factor.shape[1]
    w = cp.Variable(n_assets)
    robust_radius = cp.Variable(nonneg=True)
    turnover_abs = cp.Variable(n_assets, nonneg=True)
    mean_returns = cp.Parameter(n_assets, value=np.zeros(n_assets))
    sqrt_delta = cp.Parameter(nonneg=True, value=0.0)
    alpha_bar = cp.Parameter(value=0.0)
    previous_weights = cp.Parameter(n_assets, value=np.zeros(n_assets))
    turnover_penalty = cp.Parameter(nonneg=True, value=0.0)

    objective = cp.norm(covariance_factor @ w, 2) + sqrt_delta * robust_radius + turnover_penalty * cp.sum(turnover_abs)
    constraints = [
        cp.sum(w) == 1.0,
        w >= lower_bound,
        w <= upper_bound,
        cp.norm(w, p_norm) <= robust_radius,
        turnover_abs >= w - previous_weights,
        turnover_abs >= previous_weights - w,
        mean_returns @ w >= alpha_bar + sqrt_delta * robust_radius,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if not problem.is_dpp():
        raise ValueError("DRMV DPP program is not DPP-compliant.")
    return _DrmvDppProgram(
        problem=problem,
        weights=w,
        sqrt_covariance=covariance_factor,
        mean_returns=mean_returns,
        sqrt_delta=sqrt_delta,
        alpha_bar=alpha_bar,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
    )


def solve_wasserstein_proxy_min_var(
    train_returns: pd.DataFrame,
    epsilon: float,
    target_return: float,
    covariance_method: str = "ledoit_wolf",
    mean_returns: pd.Series | None = None,
    covariance: np.ndarray | None = None,
    covariance_factor: np.ndarray | None = None,
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    slack_penalty: float = 10.0,
    allow_slack: bool = True,
    solver: str = "ECOS",
    rebalance_date: pd.Timestamp | None = None,
    dpp_program: _WassersteinProxyDppProgram | None = None,
) -> dict[str, Any]:
    """
    Practical Wasserstein-inspired proxy min-variance allocation.

    The expected-return constraint is robustified with a dual-norm penalty and,
    by default, a non-negative slack variable to maintain feasibility:

        mu.T @ w - epsilon * ||w||_2 + s >= target_return,  s >= 0

    This is a tractable distributionally robust proxy, not a full general
    Wasserstein DRO reformulation. Setting ``allow_slack=False`` restores a
    hard feasibility check, which is useful for zero-radius validation against
    the empirical target-return minimum-variance baseline.
    """

    assets = train_returns.columns
    mu, covariance = _resolve_mean_and_covariance(
        train_returns=train_returns,
        covariance_method=covariance_method,
        mean_returns=mean_returns,
        covariance=covariance,
    )
    covariance_factor = covariance_factor if covariance_factor is not None else _matrix_square_root(covariance)
    lower_bound, upper_bound = bounds

    # At zero radius with a hard feasibility constraint, the proxy should
    # reduce to the corresponding empirical target-return min-variance problem.
    if abs(float(epsilon)) <= 1e-15 and not allow_slack:
        baseline = solve_min_variance(
            covariance=covariance,
            assets=assets,
            mu=mu,
            target_return=target_return,
            bounds=bounds,
            previous_weights=previous_weights,
            turnover_penalty=turnover_penalty,
            solver=solver,
        )
        baseline.update(
            {
                "worst_case_return": float(baseline.get("expected_return", np.nan)),
                "chosen_epsilon": float(epsilon),
                "concentration": herfindahl_index(baseline["weights"]),
                "slack_used": 0.0,
                "binding_margin": float(baseline.get("expected_return", np.nan) - float(target_return)),
                "constraint_margin": float(baseline.get("expected_return", np.nan) - float(target_return)),
                "slack_penalty": float(slack_penalty),
                "target_return": float(target_return),
                "fallback_used": False,
                "soft_feasibility_enabled": False,
            }
        )
        return baseline

    if dpp_program is not None and allow_slack:
        dpp_program.mean_returns.value = mu.reindex(assets).fillna(0.0).to_numpy()
        dpp_program.epsilon.value = max(float(epsilon), 0.0)
        dpp_program.target_return.value = float(target_return)
        dpp_program.previous_weights.value = (
            previous_weights.reindex(assets).fillna(0.0).to_numpy() if previous_weights is not None else np.zeros(len(assets))
        )
        dpp_program.turnover_penalty.value = max(float(turnover_penalty), 0.0)
        dpp_program.slack_penalty.value = max(float(slack_penalty), 0.0)
        problem = dpp_program.problem
        problem.solve(solver=solver, warm_start=True)
        w = dpp_program.weights
        s = dpp_program.slack
    else:
        w = cp.Variable(len(assets))
        s = cp.Variable(nonneg=True) if allow_slack else None

        objective = cp.quad_form(w, ensure_psd(covariance))
        if allow_slack:
            objective += float(slack_penalty) * s
        if previous_weights is not None and turnover_penalty > 0.0:
            objective += turnover_penalty * cp.norm1(w - previous_weights.reindex(assets).fillna(0.0).to_numpy())

        mu_vector = mu.reindex(assets).fillna(0.0).to_numpy()
        if abs(float(epsilon)) <= 1e-15:
            robust_return = mu_vector @ w
        else:
            robust_return = mu_vector @ w - float(epsilon) * cp.norm(w, 2)
        constraints = [
            cp.sum(w) == 1.0,
            w >= lower_bound,
            w <= upper_bound,
        ]
        if allow_slack:
            constraints.append(robust_return + s >= float(target_return))
        else:
            constraints.append(robust_return >= float(target_return))

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=solver, warm_start=True)

    if w.value is None:
        fallback = fit_sample_min_variance(
            train_returns=train_returns,
            target_return=None,
            bounds=bounds,
            previous_weights=previous_weights,
            turnover_penalty=turnover_penalty,
            solver=solver,
        )
        fallback.update(
            {
                "status": f"fallback::{problem.status}",
                "chosen_epsilon": epsilon,
                "worst_case_return": np.nan,
                "concentration": herfindahl_index(fallback["weights"]),
                "slack_used": np.nan if allow_slack else 0.0,
                "binding_margin": np.nan,
                "constraint_margin": np.nan,
                "slack_penalty": slack_penalty,
                "target_return": target_return,
                "fallback_used": True,
                "soft_feasibility_enabled": bool(allow_slack),
            }
        )
        LOGGER.warning(
            "Wasserstein proxy fallback triggered | date=%s | epsilon=%.6f | target=%.6f | status=%s",
            rebalance_date,
            epsilon,
            target_return,
            problem.status,
        )
        return fallback

    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()
    covariance_psd = ensure_psd(covariance)
    variance = float(weights.to_numpy() @ covariance_psd @ weights.to_numpy())
    slack_used = float(max(np.asarray(s.value).item(), 0.0)) if allow_slack and s is not None and s.value is not None else 0.0
    worst_case_return = float(mu @ weights - float(epsilon) * np.linalg.norm(weights.to_numpy(), ord=2))
    binding_margin = float(worst_case_return + slack_used - target_return)

    LOGGER.info(
        "Wasserstein proxy solve | date=%s | epsilon=%.6f | target=%.6f | slack=%.6f | status=%s",
        rebalance_date,
        epsilon,
        target_return,
        slack_used,
        problem.status,
    )

    return {
        "weights": weights,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "forecast_vol": float(np.sqrt(variance) * np.sqrt(252)),
        "expected_return": float(mu @ weights),
        "worst_case_return": worst_case_return,
        "chosen_epsilon": epsilon,
        "concentration": herfindahl_index(weights),
        "slack_used": slack_used,
        "binding_margin": binding_margin,
        "constraint_margin": binding_margin,
        "slack_penalty": float(slack_penalty),
        "target_return": float(target_return),
        "fallback_used": False,
        "soft_feasibility_enabled": bool(allow_slack),
    }


def _validation_score(returns: pd.Series, metric: str) -> float:
    if returns.empty:
        return -np.inf

    metric = metric.lower()
    mean_return = returns.mean()
    volatility = returns.std()
    if metric == "sharpe":
        return float(mean_return / volatility) if volatility > 0 else -np.inf
    if metric == "return":
        return float(mean_return)
    if metric == "drawdown":
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        return float(drawdown.min())
    raise ValueError(f"Unknown validation metric: {metric}")


def solve_drmv_regularized_min_variance(
    mean_returns: pd.Series,
    covariance: np.ndarray,
    delta: float,
    alpha_bar: float,
    p_norm: int | float = 2,
    long_only: bool = True,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    allow_slack: bool = False,
    slack_penalty: float = 10.0,
    solver: str = "ECOS",
    rebalance_date: pd.Timestamp | None = None,
    covariance_factor: np.ndarray | None = None,
    dpp_program: _DrmvDppProgram | None = None,
    paper_mode: str = "practical_tuned_drmv",
) -> dict[str, Any]:
    """
    Paper-aligned DR mean-variance branch with regularized volatility.

    Objective:
        min  ||L w||_2 + sqrt(delta) * ||w||_p

    subject to:
        1.T w = 1
        mu.T w >= alpha_bar + sqrt(delta) * ||w||_p

    The optional slack switch is an operational softening. It defaults to
    `False` so the direct solver stays close to the paper-style formulation.
    """

    assets = mean_returns.index
    covariance_psd = ensure_psd(covariance)
    covariance_factor = covariance_factor if covariance_factor is not None else _matrix_square_root(covariance_psd)
    sqrt_delta = float(np.sqrt(max(delta, 0.0)))

    if dpp_program is not None and not allow_slack and long_only:
        dpp_program.mean_returns.value = mean_returns.reindex(assets).fillna(0.0).to_numpy()
        dpp_program.sqrt_delta.value = sqrt_delta
        dpp_program.alpha_bar.value = float(alpha_bar)
        dpp_program.previous_weights.value = (
            previous_weights.reindex(assets).fillna(0.0).to_numpy() if previous_weights is not None else np.zeros(len(assets))
        )
        dpp_program.turnover_penalty.value = max(float(turnover_penalty), 0.0)
        problem = dpp_program.problem
        problem.solve(solver=solver, warm_start=True)
        w = dpp_program.weights
        s = None
    else:
        w = cp.Variable(len(assets))
        s = cp.Variable(nonneg=True) if allow_slack else None
        robust_norm = cp.norm(w, p_norm)
        objective = cp.norm(covariance_factor @ w, 2) + sqrt_delta * robust_norm
        if previous_weights is not None and turnover_penalty > 0.0:
            objective += turnover_penalty * cp.norm1(w - previous_weights.reindex(assets).fillna(0.0).to_numpy())
        if allow_slack and s is not None:
            objective += float(slack_penalty) * s

        mu_vector = mean_returns.reindex(assets).fillna(0.0).to_numpy()
        constraints = [cp.sum(w) == 1.0]
        if long_only:
            constraints.extend([w >= lower_bound, w <= upper_bound])
        if allow_slack and s is not None:
            constraints.append(mu_vector @ w + s >= float(alpha_bar) + sqrt_delta * robust_norm)
        else:
            constraints.append(mu_vector @ w >= float(alpha_bar) + sqrt_delta * robust_norm)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=solver, warm_start=True)

    if w.value is None:
        fallback = solve_min_variance(
            covariance=covariance_psd,
            assets=assets,
            mu=mean_returns,
            target_return=None,
            bounds=(lower_bound, upper_bound),
            previous_weights=previous_weights,
            turnover_penalty=turnover_penalty,
            solver=solver,
        )
        fallback.update(
            {
                "status": f"fallback::{problem.status}",
                "chosen_delta": float(delta),
                "alpha_bar": float(alpha_bar),
                "worst_case_return": np.nan,
                "concentration": herfindahl_index(fallback["weights"]),
                "slack_used": np.nan if allow_slack else 0.0,
                "binding_margin": np.nan,
                "constraint_margin": np.nan,
                "robust_penalty": np.nan,
                "fallback_used": True,
                "soft_feasibility_enabled": bool(allow_slack),
                "p_norm": float(p_norm),
                "paper_mode": str(paper_mode),
            }
        )
        LOGGER.info(
            "DRMV fallback triggered | date=%s | delta=%.6f | alpha_bar=%.6f | status=%s",
            rebalance_date,
            float(delta),
            float(alpha_bar),
            problem.status,
        )
        return fallback

    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    if long_only:
        weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()
    norm_penalty = sqrt_delta * float(np.linalg.norm(weights.to_numpy(), ord=p_norm))
    realized_vol = float(np.sqrt(weights.to_numpy() @ covariance_psd @ weights.to_numpy()))
    expected_return = float(mean_returns @ weights)
    slack_used = float(max(np.asarray(s.value).item(), 0.0)) if allow_slack and s is not None and s.value is not None else 0.0
    binding_margin = float(expected_return - float(alpha_bar) - norm_penalty + slack_used)

    LOGGER.info(
        "DRMV solve | date=%s | delta=%.6f | alpha_bar=%.6f | margin=%.6f | status=%s",
        rebalance_date,
        float(delta),
        float(alpha_bar),
        binding_margin,
        problem.status,
    )

    return {
        "weights": weights,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "forecast_vol": float(realized_vol * np.sqrt(252)),
        "expected_return": expected_return,
        "worst_case_return": float(expected_return - norm_penalty),
        "chosen_delta": float(delta),
        "alpha_bar": float(alpha_bar),
        "concentration": herfindahl_index(weights),
        "slack_used": slack_used,
        "binding_margin": binding_margin,
        "constraint_margin": binding_margin,
        "robust_penalty": norm_penalty,
        "fallback_used": False,
        "soft_feasibility_enabled": bool(allow_slack),
        "p_norm": float(p_norm),
        "paper_mode": str(paper_mode),
    }


def _composite_validation_score(
    returns: pd.Series,
    forecast_vol: float,
    slack_used: float,
    turnover: float,
    epsilon_change: float,
    metric: str,
    selection_slack_penalty_weight: float,
    selection_turnover_penalty_weight: float,
    selection_risk_gap_penalty_weight: float,
    selection_epsilon_change_penalty_weight: float,
) -> tuple[float, dict[str, float]]:
    base_metric = _validation_score(returns, metric="sharpe" if metric == "composite" else metric)
    realized_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else np.nan
    risk_gap = abs(realized_vol - forecast_vol) if pd.notna(realized_vol) and pd.notna(forecast_vol) else np.nan

    penalties = {
        "base_metric": base_metric,
        "validation_turnover": turnover,
        "validation_risk_gap": risk_gap,
        "validation_slack_used": slack_used,
        "validation_epsilon_change": epsilon_change,
        "slack_penalty_component": selection_slack_penalty_weight * float(np.nan_to_num(slack_used, nan=0.0)),
        "turnover_penalty_component": selection_turnover_penalty_weight * float(np.nan_to_num(turnover, nan=0.0)),
        "risk_gap_penalty_component": selection_risk_gap_penalty_weight * float(np.nan_to_num(risk_gap, nan=0.0)),
        "epsilon_penalty_component": selection_epsilon_change_penalty_weight * float(np.nan_to_num(epsilon_change, nan=0.0)),
    }

    if metric == "composite":
        score = (
            base_metric
            - penalties["slack_penalty_component"]
            - penalties["turnover_penalty_component"]
            - penalties["risk_gap_penalty_component"]
            - penalties["epsilon_penalty_component"]
        )
    else:
        score = base_metric - penalties["epsilon_penalty_component"]

    penalties["selection_score"] = score
    penalties["realized_validation_vol"] = realized_vol
    return float(score), penalties


def tune_wasserstein_proxy_radius(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    epsilon_grid: list[float],
    covariance_method: str = "ledoit_wolf",
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    slack_penalty: float = 10.0,
    metric: str = "composite",
    target_return: float | None = None,
    target_return_mode: str = "equal_weight_fraction",
    target_return_scale: float = 0.50,
    target_return_quantile: float = 0.40,
    fixed_target_return: float = 0.0002,
    mean_returns: pd.Series | None = None,
    covariance: np.ndarray | None = None,
    previous_epsilon: float | None = None,
    selection_slack_penalty_weight: float = 5.0,
    selection_turnover_penalty_weight: float = 1.0,
    selection_risk_gap_penalty_weight: float = 2.0,
    selection_epsilon_change_penalty_weight: float = 5.0,
    solver: str = "ECOS",
    rebalance_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Grid-search the proxy robustness radius on a rolling validation window."""

    if target_return is None:
        resolved_target, target_source = compute_dynamic_target_return(
            train_returns=train_returns,
            mode=target_return_mode,
            scale=target_return_scale,
            quantile=target_return_quantile,
            fixed_target_return=fixed_target_return,
        )
    else:
        resolved_target = float(target_return)
        target_source = "explicit_target_return"

    resolved_mean, resolved_covariance = _resolve_mean_and_covariance(
        train_returns=train_returns,
        covariance_method=covariance_method,
        mean_returns=mean_returns,
        covariance=covariance,
    )
    covariance_factor = _matrix_square_root(resolved_covariance)
    dpp_program = _build_wasserstein_proxy_dpp_program(
        covariance_factor=covariance_factor,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
    )

    best_result: dict[str, Any] | None = None
    best_score = -np.inf
    diagnostics_rows: list[dict[str, Any]] = []
    val_returns_filled = val_returns.fillna(0.0)

    for epsilon in epsilon_grid:
        result = solve_wasserstein_proxy_min_var(
            train_returns=train_returns,
            epsilon=epsilon,
            target_return=resolved_target,
            covariance_method=covariance_method,
            mean_returns=resolved_mean,
            covariance=resolved_covariance,
            covariance_factor=covariance_factor,
            bounds=bounds,
            previous_weights=previous_weights,
            turnover_penalty=turnover_penalty,
            slack_penalty=slack_penalty,
            solver=solver,
            rebalance_date=rebalance_date,
            dpp_program=dpp_program,
        )
        val_portfolio_returns = val_returns_filled @ result["weights"]
        validation_turnover = _selection_turnover(result["weights"], previous_weights)
        epsilon_change = abs(float(epsilon) - float(previous_epsilon)) if previous_epsilon is not None else 0.0
        score, score_components = _composite_validation_score(
            returns=val_portfolio_returns,
            forecast_vol=float(result.get("forecast_vol", np.nan)),
            slack_used=float(result.get("slack_used", np.nan)),
            turnover=validation_turnover,
            epsilon_change=epsilon_change,
            metric=metric,
            selection_slack_penalty_weight=selection_slack_penalty_weight,
            selection_turnover_penalty_weight=selection_turnover_penalty_weight,
            selection_risk_gap_penalty_weight=selection_risk_gap_penalty_weight,
            selection_epsilon_change_penalty_weight=selection_epsilon_change_penalty_weight,
        )
        result["validation_score"] = score
        result["target_source"] = target_source
        result.update(score_components)

        diagnostics_rows.append(
            {
                "epsilon": float(epsilon),
                "validation_score": float(score),
                "base_metric": float(score_components["base_metric"]),
                "slack_used": float(result.get("slack_used", np.nan)),
                "validation_turnover": float(score_components["validation_turnover"]),
                "validation_risk_gap": float(score_components["validation_risk_gap"]),
                "validation_epsilon_change": float(score_components["validation_epsilon_change"]),
                "forecast_vol": float(result.get("forecast_vol", np.nan)),
                "realized_validation_vol": float(score_components["realized_validation_vol"]),
                "expected_return": float(result.get("expected_return", np.nan)),
                "status": result.get("status", "unknown"),
                "fallback_used": bool(result.get("fallback_used", False)),
            }
        )

        if best_result is None or score > best_score + 1e-12:
            best_result = result
            best_score = score
        elif best_result is not None and abs(score - best_score) <= 1e-12:
            if float(result.get("slack_used", np.inf)) < float(best_result.get("slack_used", np.inf)):
                best_result = result
            elif float(result.get("validation_epsilon_change", np.inf)) < float(best_result.get("validation_epsilon_change", np.inf)):
                best_result = result

    if best_result is None:
        raise ValueError("Radius tuning failed to produce a candidate solution.")

    LOGGER.info(
        "Selected Wasserstein proxy radius | date=%s | epsilon=%.6f | score=%.6f | slack=%.6f | eps_change=%.6f",
        rebalance_date,
        float(best_result.get("chosen_epsilon", np.nan)),
        float(best_result.get("validation_score", np.nan)),
        float(best_result.get("slack_used", np.nan)),
        float(best_result.get("validation_epsilon_change", np.nan)),
    )
    best_result["radius_diagnostics"] = pd.DataFrame(diagnostics_rows)
    best_result["target_return"] = resolved_target
    best_result["target_source"] = target_source
    return best_result


def solve_log_return_growth_proxy(
    log_returns: pd.DataFrame,
    epsilon: float,
    bounds: tuple[float, float] = (0.0, 0.25),
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    growth_risk_aversion: float = 2.0,
    solver: str = "SCS",
) -> dict[str, Any]:
    """
    Optional appendix model inspired by the Wasserstein-Kelly paper.

    This appendix proxy is intentionally built on **asset log-return samples**
    rather than simple returns. It optimizes a tractable worst-case log-return
    proxy of the form

        mean(log_r).T @ w - epsilon * ||w||_2 - gamma * w.T @ Sigma_log @ w

    with optional turnover regularization. This is still **not** the paper's
    exact Wasserstein-Kelly convex reformulation; it is a smaller, interview-
    friendly proxy that keeps the uncertainty model in log-return space.
    """

    assets = log_returns.columns
    lower_bound, upper_bound = bounds
    mu_log = log_returns.mean().fillna(0.0)
    cov_log = estimate_covariance(log_returns, method="sample")

    w = cp.Variable(len(assets))
    objective = mu_log.reindex(assets).to_numpy() @ w
    objective -= float(epsilon) * cp.norm(w, 2)
    objective -= float(growth_risk_aversion) * cp.quad_form(w, ensure_psd(cov_log))

    if previous_weights is not None and turnover_penalty > 0.0:
        objective -= turnover_penalty * cp.norm1(w - previous_weights.reindex(assets).fillna(0.0).to_numpy())

    constraints = [
        cp.sum(w) == 1.0,
        w >= lower_bound,
        w <= upper_bound,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver, warm_start=True)

    if w.value is None:
        weights = equal_weight(assets)
    else:
        weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
        weights = weights.clip(lower=0.0)
        weights = weights / weights.sum()

    expected_log_return = float(mu_log @ weights)
    worst_case_log_return = float(expected_log_return - float(epsilon) * np.linalg.norm(weights.to_numpy(), ord=2))
    log_growth_variance = float(weights.to_numpy() @ ensure_psd(cov_log) @ weights.to_numpy())

    return {
        "weights": weights,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "expected_log_return": expected_log_return,
        "worst_case_log_return": worst_case_log_return,
        "log_growth_vol": float(np.sqrt(log_growth_variance) * np.sqrt(252)),
        "growth_risk_aversion": float(growth_risk_aversion),
        "chosen_epsilon": float(epsilon),
    }


def solve_wasserstein_kelly_exact_p2(
    log_returns: pd.DataFrame,
    epsilon: float,
    bounds: tuple[float, float] = (0.0, 0.25),
    solver: str = "SCS",
    lambda_floor: float = 1e-8,
) -> dict[str, Any]:
    """
    Closer-to-paper Wasserstein-Kelly appendix solver for p = 2.

    This follows the convex reformulation in the Li paper for the Euclidean
    Wasserstein metric:

        max_{w, v^(j), lambda >= 0}
            (1/N) sum_j [
                r_j.T v^(j)
                - 1/4 * lambda * ||v^(j) / lambda||_2^2
                + sum_i v_i^(j) log(w_i / v_i^(j))
            ]
            - lambda * epsilon^2

    under a long-only simplex constraint on ``w``.
    """

    if log_returns.empty:
        raise ValueError("log_returns must contain at least one observation.")

    assets = log_returns.columns
    samples = log_returns.fillna(0.0).to_numpy(dtype=float)
    n_samples, n_assets = samples.shape
    lower_bound, upper_bound = bounds
    weight_floor = max(float(lower_bound), 1e-8)

    w = cp.Variable(n_assets)
    v = cp.Variable((n_samples, n_assets), nonneg=True)
    lambda_var = cp.Variable(nonneg=True)

    objective_terms = []
    for sample_idx in range(n_samples):
        sample_vector = samples[sample_idx]
        entropy_term = -cp.sum(cp.rel_entr(v[sample_idx, :], w))
        transport_penalty = 0.25 * cp.quad_over_lin(v[sample_idx, :], lambda_var)
        objective_terms.append(sample_vector @ v[sample_idx, :] - transport_penalty + entropy_term)

    objective = (1.0 / n_samples) * cp.sum(cp.hstack(objective_terms)) - lambda_var * float(epsilon) ** 2
    constraints = [
        cp.sum(w) == 1.0,
        w >= weight_floor,
        w <= upper_bound,
        lambda_var >= float(lambda_floor),
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver, warm_start=True)

    if w.value is None:
        fallback = solve_log_return_growth_proxy(
            log_returns=log_returns,
            epsilon=epsilon,
            bounds=bounds,
            solver=solver,
        )
        fallback.update(
            {
                "status": f"fallback::{problem.status}",
                "lambda_value": np.nan,
                "effective_n": float(1.0 / np.square(fallback["weights"]).sum()),
                "sample_average_log_wealth": np.nan,
                "model_form": "wasserstein_kelly_exact_p2_fallback",
            }
        )
        return fallback

    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()

    gross_asset_returns = np.exp(samples)
    portfolio_growth = gross_asset_returns @ weights.to_numpy(dtype=float)
    sample_log_wealth = np.log(np.clip(portfolio_growth, 1e-12, None))

    return {
        "weights": weights,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "chosen_epsilon": float(epsilon),
        "lambda_value": float(lambda_var.value) if lambda_var.value is not None else np.nan,
        "sample_average_log_wealth": float(sample_log_wealth.mean()),
        "worst_case_log_wealth_proxy": float(problem.value) if problem.value is not None else np.nan,
        "expected_log_return": float(log_returns.mean().fillna(0.0) @ weights),
        "effective_n": float(1.0 / np.square(weights).sum()),
        "model_form": "wasserstein_kelly_exact_p2",
    }


# Backward-compatible aliases for older notebook references.
solve_wasserstein_robust_min_variance = solve_wasserstein_proxy_min_var
tune_wasserstein_radius = tune_wasserstein_proxy_radius
solve_log_growth_allocation = solve_log_return_growth_proxy
