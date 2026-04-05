from __future__ import annotations

import numpy as np
import pandas as pd

from src import regime


def test_regime_classifier_and_conditional_summary() -> None:
    dates = pd.date_range("2024-01-01", periods=16, freq="W")
    feature_frame = pd.DataFrame(
        {
            "trailing_vol": np.linspace(0.08, 0.30, 16),
            "average_pairwise_correlation": np.concatenate([np.linspace(0.20, 0.45, 8), np.linspace(0.55, 0.85, 8)]),
            "effective_rank": np.linspace(6.0, 2.5, 16),
            "recent_turnover": np.linspace(0.05, 0.35, 16),
            "herfindahl": np.linspace(0.08, 0.18, 16),
            "recent_drawdown_abs": np.concatenate([np.linspace(0.01, 0.04, 8), np.linspace(0.06, 0.14, 8)]),
            "forecast_realized_risk_gap": np.linspace(0.01, 0.08, 16),
            "cross_sectional_dispersion": np.linspace(0.10, 0.35, 16),
        },
        index=dates,
    )
    target = regime.build_regime_labels(feature_frame)

    result = regime.train_regime_classifier(
        feature_frame=feature_frame[regime.DEFAULT_REGIME_FEATURE_COLUMNS],
        target=target,
        model_type="random_forest",
        test_fraction=0.25,
        random_state=7,
    )

    assert result["status"] in {"trained", "dummy_single_class_train"}
    assert "accuracy" in result["metrics"]
    assert not result["predictions"].empty

    rebalance_results = pd.DataFrame(
        {
            "strategy": "wasserstein_proxy_min_var",
            "chosen_epsilon": np.linspace(0.0005, 0.005, 16),
            "slack_used": np.linspace(0.0, 0.02, 16),
            "turnover": np.linspace(0.05, 0.30, 16),
            "execution_eta": np.linspace(0.20, 1.00, 16),
            "realized_vol": np.linspace(0.10, 0.24, 16),
            "forecast_vol": np.linspace(0.09, 0.21, 16),
            "hold_period_return": np.linspace(0.002, 0.010, 16),
        },
        index=dates,
    )

    summary = regime.summarize_regime_conditionals(
        rebalance_results=rebalance_results,
        regime_predictions=result["predictions"],
        strategy="wasserstein_proxy_min_var",
        regime_column="predicted_regime",
    )

    assert not summary.empty
    assert "avg_chosen_epsilon" in summary.columns


def test_regime_conditioned_inputs_return_mean_and_covariance() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="W")
    asset_returns = pd.DataFrame(
        {
            "A": np.linspace(-0.01, 0.02, 40),
            "B": np.linspace(-0.005, 0.015, 40),
            "C": np.linspace(0.0, 0.01, 40),
        },
        index=dates,
    )
    market_factor = asset_returns.mean(axis=1)

    regime_probs = regime.estimate_regime_probabilities(market_factor, n_regimes=2, lookback=40, random_state=7)
    inputs = regime.estimate_regime_conditioned_inputs(
        asset_returns=asset_returns,
        factor_returns=market_factor,
        regime_probs=regime_probs,
        lookback=40,
        covariance_method="sample",
    )

    assert set(inputs["mean_returns"].index) == set(asset_returns.columns)
    assert inputs["covariance"].shape == (3, 3)
    assert 0.0 <= float(inputs["stressed_probability"]) <= 1.0


def test_hmm_regime_conditioned_inputs_return_probabilities_and_covariance() -> None:
    dates = pd.date_range("2021-01-01", periods=120, freq="W")
    calm_block = np.random.default_rng(7).normal(0.002, 0.01, size=(60, 3))
    stressed_block = np.random.default_rng(8).normal(-0.003, 0.025, size=(60, 3))
    asset_returns = pd.DataFrame(
        np.vstack([calm_block, stressed_block]),
        index=dates,
        columns=["A", "B", "C"],
    )
    market_factor = asset_returns.mean(axis=1)

    hmm_fit = regime.fit_two_state_hmm(market_factor, lookback=120)
    filtered = regime.infer_filtered_regime_probs(hmm_fit)
    smoothed = regime.infer_smoothed_regime_probs(hmm_fit)
    inputs = regime.estimate_regime_conditioned_inputs_hmm(
        asset_returns=asset_returns,
        factor_returns=market_factor,
        lookback=120,
        covariance_method="state_aware",
        calm_covariance_method="ledoit_wolf",
        stressed_covariance_method="ewma",
    )

    filtered_state_columns = [column for column in filtered.columns if column.startswith("regime_") and column.endswith("_prob")]
    smoothed_state_columns = [column for column in smoothed.columns if column.startswith("regime_") and column.endswith("_prob")]
    assert len(filtered_state_columns) == 2
    assert len(smoothed_state_columns) == 2
    assert set(inputs["mean_returns"].index) == set(asset_returns.columns)
    assert inputs["covariance"].shape == (3, 3)
    assert inputs["regime_model_version"] in {"hmm_markov_regression", "hmm_fallback_to_mixture"}
