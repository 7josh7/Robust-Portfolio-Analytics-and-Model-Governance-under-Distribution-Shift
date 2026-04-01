from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines, robust
from .features import herfindahl_index


def weights_sum_to_one(weights: pd.Series, tolerance: float = 1e-6) -> bool:
    return bool(abs(weights.sum() - 1.0) <= tolerance)


def weights_are_long_only(weights: pd.Series, tolerance: float = 1e-8) -> bool:
    return bool((weights >= -tolerance).all())


def covariance_is_psd(covariance: pd.DataFrame | np.ndarray, tolerance: float = 1e-8) -> bool:
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=float))
    return bool(eigenvalues.min() >= -tolerance)


def indices_are_aligned(*objects: pd.DataFrame | pd.Series) -> bool:
    if not objects:
        return True
    first_index = objects[0].index
    return all(first_index.equals(obj.index) for obj in objects[1:])


def build_input_check_table(weights_history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy, weights in weights_history.items():
        if weights.empty:
            continue
        rows.append(
            {
                "strategy": strategy,
                "no_missing_weights": bool(weights.notna().all().all()),
                "weights_sum_to_one": bool(weights.apply(weights_sum_to_one, axis=1).all()),
                "long_only": bool(weights.apply(weights_are_long_only, axis=1).all()),
                "max_concentration": float(weights.apply(herfindahl_index, axis=1).max()),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def build_numerical_check_table(
    rebalance_results: pd.DataFrame,
    covariance_snapshots: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    covariance_snapshots = covariance_snapshots or {}
    for strategy, frame in rebalance_results.groupby("strategy"):
        rows.append(
            {
                "strategy": strategy,
                "optimizer_status_ok": bool(~frame["status"].astype(str).str.contains("infeasible", case=False).any()),
                "has_objective_values": bool(frame["objective_value"].notna().any()),
                "has_hyperparameter_fallback_logic": bool(frame["chosen_epsilon"].notna().any()),
                "uses_soft_feasibility_slack": bool(frame["slack_used"].notna().any()) if "slack_used" in frame else False,
                "fallback_rate": float(frame["fallback_used"].fillna(False).mean()) if "fallback_used" in frame else 0.0,
                "covariance_psd": covariance_is_psd(covariance_snapshots[strategy])
                if strategy in covariance_snapshots
                else True,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def diagnose_zero_radius_proxy_alignment(
    train_returns: pd.DataFrame,
    target_return: float,
    bounds: tuple[float, float] = (0.0, 1.0),
    covariance_method: str = "sample",
    previous_weights: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    slack_penalty: float = 1_000.0,
    solver: str = "SCS",
) -> pd.DataFrame:
    """
    Decompose the zero-radius proxy into components so we can isolate where
    any mismatch versus the empirical target-return min-variance baseline
    comes from.
    """

    empirical = baselines.fit_sample_min_variance(
        train_returns=train_returns,
        target_return=target_return,
        bounds=bounds,
        solver=solver,
    )
    variants = {
        "empirical_target_min_var": empirical,
        "proxy_hard_constraint_no_turnover": robust.solve_wasserstein_proxy_min_var(
            train_returns=train_returns,
            epsilon=0.0,
            target_return=target_return,
            covariance_method=covariance_method,
            bounds=bounds,
            previous_weights=None,
            turnover_penalty=0.0,
            slack_penalty=slack_penalty,
            allow_slack=False,
            solver=solver,
        ),
        "proxy_soft_slack_only": robust.solve_wasserstein_proxy_min_var(
            train_returns=train_returns,
            epsilon=0.0,
            target_return=target_return,
            covariance_method=covariance_method,
            bounds=bounds,
            previous_weights=None,
            turnover_penalty=0.0,
            slack_penalty=slack_penalty,
            allow_slack=True,
            solver=solver,
        ),
        "proxy_soft_slack_plus_turnover": robust.solve_wasserstein_proxy_min_var(
            train_returns=train_returns,
            epsilon=0.0,
            target_return=target_return,
            covariance_method=covariance_method,
            bounds=bounds,
            previous_weights=previous_weights,
            turnover_penalty=turnover_penalty,
            slack_penalty=slack_penalty,
            allow_slack=True,
            solver=solver,
        ),
    }

    empirical_weights = empirical["weights"]
    rows: list[dict[str, float | str | bool]] = []
    for label, result in variants.items():
        weights = result["weights"].reindex(empirical_weights.index).fillna(0.0)
        rows.append(
            {
                "variant": label,
                "weight_l1_vs_empirical": float((weights - empirical_weights).abs().sum()),
                "max_abs_weight_diff_vs_empirical": float((weights - empirical_weights).abs().max()),
                "expected_return": float(result.get("expected_return", np.nan)),
                "target_return": float(result.get("target_return", target_return)),
                "slack_used": float(result.get("slack_used", np.nan)),
                "binding_margin": float(result.get("binding_margin", np.nan)),
                "turnover_penalty": float(turnover_penalty if "turnover" in label else 0.0),
                "soft_feasibility_enabled": bool(result.get("soft_feasibility_enabled", False)),
                "fallback_used": bool(result.get("fallback_used", False)),
                "objective_value": float(result.get("objective_value", np.nan)),
                "status": str(result.get("status", "unknown")),
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def run_regression_tests(
    equal_weight_weights: pd.Series,
    empirical_weights: pd.Series | None = None,
    robust_zero_radius_weights: pd.Series | None = None,
    robust_large_radius_weights: pd.Series | None = None,
    noise_summary: pd.DataFrame | None = None,
    tolerance: float = 1e-3,
) -> pd.DataFrame:
    """Notebook-friendly smoke tests for model validation."""

    tests: list[dict] = []
    tests.append(
        {
            "test": "equal_weight_sums_to_one",
            "passed": weights_sum_to_one(equal_weight_weights, tolerance=tolerance),
            "detail": float(equal_weight_weights.sum()),
        }
    )

    if empirical_weights is not None and robust_zero_radius_weights is not None:
        l1_distance = float((empirical_weights - robust_zero_radius_weights).abs().sum())
        tests.append(
            {
                "test": "zero_radius_matches_empirical_reasonably",
                "passed": l1_distance <= max(0.10, 10 * tolerance),
                "detail": l1_distance,
            }
        )

    if robust_zero_radius_weights is not None and robust_large_radius_weights is not None:
        zero_concentration = herfindahl_index(robust_zero_radius_weights)
        large_concentration = herfindahl_index(robust_large_radius_weights)
        tests.append(
            {
                "test": "larger_radius_does_not_increase_concentration_materially",
                "passed": large_concentration <= zero_concentration + 5 * tolerance,
                "detail": large_concentration - zero_concentration,
            }
        )

    if noise_summary is not None and {"strategy", "noise_level", "sharpe_ratio"} <= set(noise_summary.columns):
        pivot = noise_summary.pivot(index="noise_level", columns="strategy", values="sharpe_ratio")
        if {"sample_min_var", "wasserstein_proxy_min_var"} <= set(pivot.columns):
            naive_drop = float(pivot["sample_min_var"].iloc[0] - pivot["sample_min_var"].iloc[-1])
            robust_drop = float(pivot["wasserstein_proxy_min_var"].iloc[0] - pivot["wasserstein_proxy_min_var"].iloc[-1])
            tests.append(
                {
                    "test": "robust_model_degrades_less_under_noise",
                    "passed": robust_drop <= naive_drop + 1e-6,
                    "detail": robust_drop - naive_drop,
                }
            )

    return pd.DataFrame(tests)
