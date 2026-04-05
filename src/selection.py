from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .baselines import ensure_psd, inverse_volatility_weight
from .covariance import estimate_covariance_matrix
from .features import herfindahl_index, max_drawdown
from .regime import estimate_regime_conditioned_inputs_hmm, estimate_regime_conditioned_inputs_mixture
from .robust import (
    _build_drmv_dpp_program,
    _matrix_square_root,
    _validation_score,
    solve_drmv_regularized_min_variance,
)
from .targets import (
    build_alpha_bar,
    build_alpha_bar_paper_reference,
    build_delta_grid_paper_reference,
    build_nominal_target,
    build_regime_conditioned_target_params,
)


def _selection_turnover(weights: pd.Series, previous_weights: pd.Series | None) -> float:
    if previous_weights is None:
        return 0.0
    aligned_previous = previous_weights.reindex(weights.index).fillna(0.0)
    return float((weights - aligned_previous).abs().sum())


def prepare_regime_conditioned_inputs(
    train_returns: pd.DataFrame,
    lookback: int = 252,
    n_regimes: int = 2,
    regime_engine: str = "hmm",
    covariance_method: str = "state_aware",
    calm_covariance_method: str = "ledoit_wolf",
    stressed_covariance_method: str = "ewma",
    probability_temperature: float = 2.0,
    stressed_probability_threshold: float = 0.65,
    switching_variance: bool = True,
    current_probability_mode: str = "filtered",
    estimation_probability_mode: str = "smoothed",
    random_state: int = 7,
) -> dict[str, Any]:
    market_factor = train_returns.fillna(0.0).mean(axis=1).rename("market_factor")
    resolved_lookback = min(lookback, len(train_returns))
    if regime_engine.lower() == "mixture":
        return estimate_regime_conditioned_inputs_mixture(
            asset_returns=train_returns,
            factor_returns=market_factor,
            lookback=resolved_lookback,
            covariance_method=covariance_method,
            calm_covariance_method=calm_covariance_method,
            stressed_covariance_method=stressed_covariance_method,
            probability_temperature=probability_temperature,
            stressed_probability_threshold=stressed_probability_threshold,
        )
    if regime_engine.lower() == "hmm":
        return estimate_regime_conditioned_inputs_hmm(
            asset_returns=train_returns,
            factor_returns=market_factor,
            lookback=resolved_lookback,
            covariance_method=covariance_method,
            calm_covariance_method=calm_covariance_method,
            stressed_covariance_method=stressed_covariance_method,
            probability_temperature=probability_temperature,
            stressed_probability_threshold=stressed_probability_threshold,
            switching_variance=switching_variance,
            current_probability_mode=current_probability_mode,
            estimation_probability_mode=estimation_probability_mode,
        )
    raise ValueError(f"Unknown regime_engine: {regime_engine}")


def build_regime_search_overrides(
    delta_grid: list[float],
    turnover_penalty: float,
    stress_activation: float,
    stressed_delta_grid_multiplier: float = 3.0,
    stressed_turnover_multiplier: float = 4.0,
) -> dict[str, float | list[float]]:
    activation = float(np.clip(stress_activation, 0.0, 1.0))
    if activation <= 0.0:
        return {
            "delta_grid": sorted({float(delta) for delta in delta_grid}),
            "turnover_penalty": float(turnover_penalty),
        }

    multiplier = 1.0 + (float(stressed_delta_grid_multiplier) - 1.0) * activation
    expanded_grid = sorted({float(delta) for delta in delta_grid} | {float(delta) * multiplier for delta in delta_grid})
    adjusted_turnover_penalty = float(turnover_penalty) * (1.0 + (float(stressed_turnover_multiplier) - 1.0) * activation)
    return {
        "delta_grid": expanded_grid,
        "turnover_penalty": adjusted_turnover_penalty,
    }


def _resolve_objective_penalty_weights(
    objective_mode: str,
    turnover_penalty_weight: float,
    risk_gap_penalty_weight: float,
    constraint_penalty_weight: float,
    fallback_penalty_weight: float,
    sensitivity_penalty_weight: float,
    corruption_penalty_weight: float,
    stress_penalty_weight: float,
    concentration_penalty_weight: float,
    drawdown_penalty_weight: float,
) -> dict[str, float]:
    weights = {
        "turnover": float(turnover_penalty_weight),
        "risk_gap": float(risk_gap_penalty_weight),
        "constraint": float(constraint_penalty_weight),
        "fallback": float(fallback_penalty_weight),
        "sensitivity": float(sensitivity_penalty_weight),
        "corruption": float(corruption_penalty_weight),
        "stress": float(stress_penalty_weight),
        "concentration": float(concentration_penalty_weight),
        "drawdown": float(drawdown_penalty_weight),
    }
    mode = objective_mode.lower()
    if mode == "production":
        return weights
    if mode == "paper_alignment":
        weights["sensitivity"] = max(weights["sensitivity"], 2.0)
        weights["corruption"] = max(weights["corruption"], 1.5)
        weights["stress"] = max(weights["stress"], 4.0)
        weights["concentration"] = max(weights["concentration"], 0.5)
        weights["drawdown"] = max(weights["drawdown"], 2.0)
        return weights
    if mode == "appendix_kelly":
        weights["turnover"] = max(weights["turnover"], 0.25)
        weights["concentration"] = max(weights["concentration"], 1.0)
        weights["drawdown"] = max(weights["drawdown"], 2.5)
        weights["sensitivity"] = max(weights["sensitivity"], 1.0)
        return weights
    raise ValueError(f"Unknown objective_mode: {objective_mode}")


def _generate_corrupted_validation_returns(
    val_returns: pd.DataFrame,
    noise_scale: float,
    random_state: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    vol = val_returns.std().replace(0.0, np.nan).fillna(0.0)
    noise = pd.DataFrame(
        rng.normal(size=val_returns.shape),
        index=val_returns.index,
        columns=val_returns.columns,
    ).mul(vol.to_numpy(), axis=1)
    return val_returns.fillna(0.0) + float(noise_scale) * noise


def _candidate_sensitivity_penalty(
    candidate_weights: pd.Series,
    mean_returns: pd.Series,
    covariance_matrix: np.ndarray,
    covariance_factor: np.ndarray,
    train_returns: pd.DataFrame,
    delta: float,
    alpha_bar: float,
    p_norm: int | float,
    bounds: tuple[float, float],
    mean_perturbation_scale: float,
    covariance_perturbation_scale: float,
    solver: str,
    dpp_program: object | None = None,
) -> float:
    mean_shock = train_returns.std().fillna(0.0) / max(np.sqrt(len(train_returns)), 1.0)
    perturbed_mean = mean_returns.reindex(train_returns.columns).fillna(0.0) - float(mean_perturbation_scale) * mean_shock
    perturbed_covariance = ensure_psd(
        covariance_matrix + float(covariance_perturbation_scale) * np.diag(np.diag(covariance_matrix))
    )
    perturbed_covariance_factor = _matrix_square_root(perturbed_covariance)

    mean_result = solve_drmv_regularized_min_variance(
        mean_returns=perturbed_mean,
        covariance=covariance_matrix,
        covariance_factor=covariance_factor,
        delta=delta,
        alpha_bar=alpha_bar,
        p_norm=p_norm,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        solver=solver,
        dpp_program=dpp_program,
    )
    covariance_result = solve_drmv_regularized_min_variance(
        mean_returns=mean_returns,
        covariance=perturbed_covariance,
        covariance_factor=perturbed_covariance_factor,
        delta=delta,
        alpha_bar=alpha_bar,
        p_norm=p_norm,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        solver=solver,
        dpp_program=None,
    )
    mean_shift = float((candidate_weights - mean_result["weights"].reindex(candidate_weights.index).fillna(0.0)).abs().sum())
    covariance_shift = float((candidate_weights - covariance_result["weights"].reindex(candidate_weights.index).fillna(0.0)).abs().sum())
    return 0.5 * (mean_shift + covariance_shift)


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
    selection_sensitivity_penalty_weight: float = 0.0,
    selection_corruption_penalty_weight: float = 0.0,
    selection_stress_penalty_weight: float = 0.0,
    selection_concentration_penalty_weight: float = 0.0,
    selection_drawdown_penalty_weight: float = 0.0,
    mean_perturbation_scale: float = 0.25,
    covariance_perturbation_scale: float = 0.20,
    corruption_noise_scale: float = 0.15,
    stress_quantile: float = 0.35,
    selection_sensitivity_top_k: int = 12,
    metric: str = "composite",
    benchmark_returns: pd.Series | None = None,
    mean_returns: pd.Series | None = None,
    covariance: np.ndarray | None = None,
    regime_conditioned: bool = False,
    calibration_mode: str = "practical",
    objective_mode: str = "production",
    stressed_target_scale: float = 0.85,
    stressed_delta_scale: float = 1.25,
    stressed_probability: float = 0.0,
    solver: str = "ECOS",
    rebalance_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Jointly tune DRMV ambiguity size, robust target adjustment, and covariance engine.

    Two calibration modes are available:

    - ``practical`` keeps the current tuned workflow for production-style runs.
    - ``paper_reference`` uses a more explicit Blanchet-Chen-Zhou-inspired
      alpha-bar adjustment and sample-size-aware ambiguity grid.
    """

    nominal_target, target_source = build_nominal_target(
        train_returns=train_returns,
        method=target_method,
        scale=target_scale,
        quantile=target_quantile,
        benchmark_returns=benchmark_returns,
        fixed_target_return=fixed_target_return,
    )
    effective_penalties = _resolve_objective_penalty_weights(
        objective_mode=objective_mode,
        turnover_penalty_weight=selection_turnover_penalty_weight,
        risk_gap_penalty_weight=selection_risk_gap_penalty_weight,
        constraint_penalty_weight=selection_constraint_penalty_weight,
        fallback_penalty_weight=selection_fallback_penalty_weight,
        sensitivity_penalty_weight=selection_sensitivity_penalty_weight,
        corruption_penalty_weight=selection_corruption_penalty_weight,
        stress_penalty_weight=selection_stress_penalty_weight,
        concentration_penalty_weight=selection_concentration_penalty_weight,
        drawdown_penalty_weight=selection_drawdown_penalty_weight,
    )
    working_delta_grid = (
        build_delta_grid_paper_reference(sample_size=len(train_returns), base_grid=delta_grid)
        if calibration_mode.lower() == "paper_reference"
        else sorted(float(delta) for delta in delta_grid)
    )
    proxy_weights = (
        previous_weights.reindex(train_returns.columns).fillna(0.0)
        if previous_weights is not None
        else inverse_volatility_weight(train_returns).reindex(train_returns.columns).fillna(0.0)
    )
    proxy_norm = float(np.linalg.norm(proxy_weights.to_numpy(dtype=float), ord=p_norm))

    best_result: dict[str, Any] | None = None
    best_score = -np.inf
    diagnostics_rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    val_returns_filled = val_returns.fillna(0.0)
    corrupted_val_returns = _generate_corrupted_validation_returns(val_returns, noise_scale=corruption_noise_scale).fillna(0.0)
    market_proxy = val_returns_filled.mean(axis=1)
    stress_cutoff = market_proxy.quantile(stress_quantile) if not market_proxy.empty else np.nan
    stressed_mask = market_proxy <= stress_cutoff if pd.notna(stress_cutoff) else pd.Series(False, index=market_proxy.index)

    if covariance is not None:
        covariance_psd = ensure_psd(covariance)
        covariance_candidates = [("provided", covariance_psd, _matrix_square_root(covariance_psd))]
    else:
        covariance_candidates = []
        for method in covariance_methods:
            covariance_matrix = estimate_covariance_matrix(train_returns, method=method)
            covariance_candidates.append((method, covariance_matrix, _matrix_square_root(covariance_matrix)))
    mean_vector = mean_returns.reindex(train_returns.columns).fillna(0.0) if mean_returns is not None else train_returns.mean().fillna(0.0)

    for covariance_method, covariance_matrix, covariance_factor in covariance_candidates:
        dpp_program = _build_drmv_dpp_program(
            covariance_factor=covariance_factor,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            p_norm=p_norm,
        )
        for delta in working_delta_grid:
            for alpha_scale in alpha_bar_scale_grid:
                if calibration_mode.lower() == "paper_reference":
                    alpha_bar, alpha_source = build_alpha_bar_paper_reference(
                        rho=nominal_target,
                        delta=delta,
                        phi_norm_proxy=proxy_norm,
                        c=alpha_scale,
                    )
                    working_delta = float(delta)
                elif regime_conditioned:
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
                    alpha_source = str(target_params["alpha_bar_source"])
                else:
                    alpha_bar, alpha_source = build_alpha_bar(
                        rho=nominal_target,
                        delta=delta,
                        rule=alpha_bar_rule,
                        scale=alpha_scale,
                    )
                    working_delta = float(delta)

                result = solve_drmv_regularized_min_variance(
                    mean_returns=mean_vector,
                    covariance=covariance_matrix,
                    covariance_factor=covariance_factor,
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
                    dpp_program=dpp_program,
                    paper_mode="paper_reference_drmv" if calibration_mode.lower() == "paper_reference" else "practical_tuned_drmv",
                )

                val_portfolio_returns = val_returns_filled @ result["weights"]
                base_metric = _validation_score(val_portfolio_returns, metric="sharpe" if metric == "composite" else metric)
                validation_turnover = _selection_turnover(result["weights"], previous_weights)
                realized_validation_vol = float(val_portfolio_returns.std() * np.sqrt(252)) if not val_portfolio_returns.empty else np.nan
                forecast_vol = float(result.get("forecast_vol", np.nan))
                risk_gap = abs(realized_validation_vol - forecast_vol) if pd.notna(realized_validation_vol) and pd.notna(forecast_vol) else np.nan
                corrupted_returns = corrupted_val_returns @ result["weights"]
                corrupted_metric = _validation_score(corrupted_returns, metric="sharpe")
                corruption_penalty = effective_penalties["corruption"] * float(max(base_metric - corrupted_metric, 0.0))
                stressed_returns = val_portfolio_returns.loc[stressed_mask] if stressed_mask.any() else pd.Series(dtype=float)
                if stressed_returns.empty:
                    stress_penalty = 0.0
                    stressed_cvar = np.nan
                else:
                    stressed_cutoff = stressed_returns.quantile(0.25)
                    stressed_tail = stressed_returns.loc[stressed_returns <= stressed_cutoff]
                    stressed_cvar = float(stressed_tail.mean()) if not stressed_tail.empty else float(stressed_cutoff)
                    stress_penalty = effective_penalties["stress"] * abs(min(stressed_cvar, 0.0))
                validation_drawdown = abs(min(max_drawdown(val_portfolio_returns), 0.0)) if not val_portfolio_returns.empty else np.nan
                drawdown_penalty = effective_penalties["drawdown"] * float(np.nan_to_num(validation_drawdown, nan=0.0))
                concentration_penalty = effective_penalties["concentration"] * herfindahl_index(result["weights"])
                binding_penalty = effective_penalties["constraint"] * float(
                    np.nan_to_num(max(1e-6 - float(result.get("binding_margin", np.nan)), 0.0), nan=1.0)
                )
                fallback_penalty = effective_penalties["fallback"] * float(result.get("fallback_used", False))
                provisional_score = (
                    base_metric
                    - effective_penalties["turnover"] * float(np.nan_to_num(validation_turnover, nan=0.0))
                    - effective_penalties["risk_gap"] * float(np.nan_to_num(risk_gap, nan=0.0))
                    - corruption_penalty
                    - stress_penalty
                    - drawdown_penalty
                    - concentration_penalty
                    - binding_penalty
                    - fallback_penalty
                )

                result.update(
                    {
                        "validation_score": float(provisional_score),
                        "nominal_target_return": float(nominal_target),
                        "target_return": float(alpha_bar),
                        "target_source": target_source,
                        "target_rule": alpha_source,
                        "covariance_method": covariance_method,
                        "regime_conditioned": bool(regime_conditioned),
                        "stressed_probability": float(stressed_probability),
                        "alpha_bar_scale": float(alpha_scale),
                        "calibration_mode": calibration_mode,
                        "objective_mode": objective_mode,
                        "validation_sensitivity_penalty": np.nan,
                        "validation_corruption_penalty": float(corruption_penalty),
                        "validation_stress_penalty": float(stress_penalty),
                        "validation_concentration_penalty": float(concentration_penalty),
                        "validation_drawdown_penalty": float(drawdown_penalty),
                    }
                )
                candidate_records.append(
                    {
                        "result": result,
                        "mean_vector": mean_vector,
                        "covariance_matrix": covariance_matrix,
                        "covariance_factor": covariance_factor,
                        "dpp_program": dpp_program,
                        "working_delta": float(working_delta),
                        "alpha_bar": float(alpha_bar),
                        "provisional_score": float(provisional_score),
                        "diagnostics_row": {
                            "covariance_method": covariance_method,
                            "delta": float(working_delta),
                            "base_delta": float(delta),
                            "alpha_bar": float(alpha_bar),
                            "alpha_bar_scale": float(alpha_scale),
                            "provisional_validation_score": float(provisional_score),
                            "validation_score": np.nan,
                            "base_metric": float(base_metric),
                            "validation_turnover": float(validation_turnover),
                            "validation_risk_gap": float(risk_gap),
                            "validation_sensitivity_penalty": np.nan,
                            "validation_corruption_penalty": float(corruption_penalty),
                            "validation_stress_penalty": float(stress_penalty),
                            "validation_concentration_penalty": float(concentration_penalty),
                            "validation_drawdown_penalty": float(drawdown_penalty),
                            "stressed_cvar": float(stressed_cvar) if pd.notna(stressed_cvar) else np.nan,
                            "binding_margin": float(result.get("binding_margin", np.nan)),
                            "binding_penalty": float(binding_penalty),
                            "fallback_penalty": float(fallback_penalty),
                            "calibration_mode": calibration_mode,
                            "objective_mode": objective_mode,
                            "status": str(result.get("status", "unknown")),
                            "sensitivity_evaluated": False,
                        },
                    }
                )

    if not candidate_records:
        raise ValueError("DRMV tuning failed to produce any candidate solution.")

    ranked_candidates = sorted(candidate_records, key=lambda record: record["provisional_score"], reverse=True)
    if effective_penalties["sensitivity"] > 0.0:
        top_k = min(max(int(selection_sensitivity_top_k), 1), len(ranked_candidates))
    else:
        top_k = len(ranked_candidates)

    for candidate_idx, record in enumerate(ranked_candidates):
        sensitivity_penalty = 0.0
        sensitivity_evaluated = effective_penalties["sensitivity"] <= 0.0 or candidate_idx < top_k
        if sensitivity_evaluated and effective_penalties["sensitivity"] > 0.0:
            sensitivity_penalty = (
                effective_penalties["sensitivity"]
                * _candidate_sensitivity_penalty(
                    candidate_weights=record["result"]["weights"],
                    mean_returns=record["mean_vector"],
                    covariance_matrix=record["covariance_matrix"],
                    covariance_factor=record["covariance_factor"],
                    train_returns=train_returns,
                    delta=record["working_delta"],
                    alpha_bar=record["alpha_bar"],
                    p_norm=p_norm,
                    bounds=bounds,
                    mean_perturbation_scale=mean_perturbation_scale,
                    covariance_perturbation_scale=covariance_perturbation_scale,
                    solver=solver,
                    dpp_program=record["dpp_program"],
                )
            )
        final_score = float(record["provisional_score"] - sensitivity_penalty) if sensitivity_evaluated else np.nan

        record["result"]["validation_score"] = final_score if sensitivity_evaluated else float(record["provisional_score"])
        record["result"]["validation_sensitivity_penalty"] = float(sensitivity_penalty) if sensitivity_evaluated else np.nan
        record["diagnostics_row"]["validation_score"] = final_score
        record["diagnostics_row"]["validation_sensitivity_penalty"] = (
            float(sensitivity_penalty) if sensitivity_evaluated else np.nan
        )
        record["diagnostics_row"]["sensitivity_evaluated"] = bool(sensitivity_evaluated)
        diagnostics_rows.append(record["diagnostics_row"])

        if not sensitivity_evaluated:
            continue
        if best_result is None or final_score > best_score + 1e-12:
            best_result = record["result"]
            best_score = final_score
        elif best_result is not None and abs(final_score - best_score) <= 1e-12:
            if float(record["result"].get("binding_margin", -np.inf)) > float(best_result.get("binding_margin", -np.inf)):
                best_result = record["result"]

    if best_result is None:
        raise ValueError("DRMV tuning did not evaluate any candidate in the sensitivity stage.")

    best_result["parameter_diagnostics"] = pd.DataFrame(diagnostics_rows)
    return best_result
