from __future__ import annotations

import numpy as np
import pandas as pd

from src import baselines, robust, selection, validation


def _toy_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = rng.normal(loc=[0.0005, 0.0004, 0.0003], scale=[0.01, 0.011, 0.009], size=(252, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_weights_and_psd_checks() -> None:
    train_returns = _toy_returns()
    covariance = baselines.estimate_covariance(train_returns, method="sample")
    weights = baselines.equal_weight(train_returns.columns)

    assert validation.weights_sum_to_one(weights)
    assert validation.covariance_is_psd(covariance)


def test_soft_slack_proxy_returns_weights_when_target_is_too_high() -> None:
    train_returns = _toy_returns()
    result = robust.solve_wasserstein_proxy_min_var(
        train_returns=train_returns,
        epsilon=0.01,
        target_return=0.05,
        covariance_method="sample",
        bounds=(0.0, 1.0),
        slack_penalty=10.0,
    )

    assert result["weights"].notna().all()
    assert np.isclose(result["weights"].sum(), 1.0)
    assert float(result["slack_used"]) > 0.0


def test_zero_radius_proxy_is_close_to_sample_target_min_var() -> None:
    train_returns = _toy_returns()
    target_return = 0.0001
    empirical = baselines.fit_sample_min_variance(
        train_returns=train_returns,
        target_return=target_return,
        bounds=(0.0, 1.0),
    )
    robust_zero = robust.solve_wasserstein_proxy_min_var(
        train_returns=train_returns,
        epsilon=0.0,
        target_return=target_return,
        covariance_method="sample",
        bounds=(0.0, 1.0),
        slack_penalty=1_000.0,
        allow_slack=False,
    )

    l1_distance = float((empirical["weights"] - robust_zero["weights"]).abs().sum())
    assert l1_distance < 0.10


def test_radius_tuning_returns_composite_diagnostics() -> None:
    train_returns = _toy_returns()
    val_returns = train_returns.tail(63)

    result = robust.tune_wasserstein_proxy_radius(
        train_returns=train_returns.iloc[:-63],
        val_returns=val_returns,
        epsilon_grid=[0.0, 0.001, 0.01],
        covariance_method="sample",
        bounds=(0.0, 1.0),
        metric="composite",
        previous_epsilon=0.001,
        selection_slack_penalty_weight=5.0,
        selection_turnover_penalty_weight=1.0,
        selection_risk_gap_penalty_weight=2.0,
        selection_epsilon_change_penalty_weight=5.0,
    )

    assert "radius_diagnostics" in result
    assert "validation_epsilon_change" in result
    assert "validation_risk_gap" in result
    assert {"epsilon", "validation_turnover", "validation_risk_gap", "validation_epsilon_change"} <= set(
        result["radius_diagnostics"].columns
    )


def test_log_return_growth_proxy_uses_log_return_inputs() -> None:
    train_returns = _toy_returns()
    log_returns = np.log1p(train_returns)

    result = robust.solve_log_return_growth_proxy(
        log_returns=log_returns,
        epsilon=0.01,
        bounds=(0.0, 1.0),
        growth_risk_aversion=1.0,
    )

    assert result["weights"].notna().all()
    assert np.isclose(result["weights"].sum(), 1.0)
    assert "expected_log_return" in result
    assert "worst_case_log_return" in result


def test_drmv_regularized_min_variance_returns_weights_and_margin() -> None:
    train_returns = _toy_returns()
    mean_returns = train_returns.mean()
    covariance = baselines.estimate_covariance(train_returns, method="sample")

    result = robust.solve_drmv_regularized_min_variance(
        mean_returns=mean_returns,
        covariance=covariance,
        delta=0.001,
        alpha_bar=-0.05,
        p_norm=2,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    assert result["weights"].notna().all()
    assert np.isclose(result["weights"].sum(), 1.0)
    assert "binding_margin" in result
    assert pd.notna(result["chosen_delta"])


def test_drmv_selector_returns_parameter_diagnostics() -> None:
    train_returns = _toy_returns().iloc[:-63]
    val_returns = _toy_returns().iloc[-63:]

    result = selection.tune_drmv_regularized_min_variance(
        train_returns=train_returns,
        val_returns=val_returns,
        delta_grid=[0.0, 0.001],
        alpha_bar_scale_grid=[0.75, 1.0],
        covariance_methods=["sample", "ledoit_wolf"],
        bounds=(0.0, 1.0),
        target_method="benchmark_fraction",
        target_scale=0.50,
        alpha_bar_rule="delta_adjusted",
    )

    assert "parameter_diagnostics" in result
    assert {"delta", "alpha_bar", "covariance_method", "validation_score"} <= set(result["parameter_diagnostics"].columns)
