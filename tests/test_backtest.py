from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest


def _equal_weight_strategy(train_returns, val_returns, previous_weights, config):
    weights = pd.Series(1.0 / train_returns.shape[1], index=train_returns.columns, name="weight")
    return {"weights": weights, "forecast_vol": 0.10}


def test_summarize_backtest_window_uses_window_specific_rebalances() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    daily_returns = pd.DataFrame(
        {
            "strategy_a": [0.01, 0.00, -0.01, 0.01, 0.00, 0.01],
            "strategy_b": [0.00, 0.01, 0.01, -0.01, 0.02, 0.00],
        },
        index=dates,
    )
    gross_daily_returns = daily_returns + 0.001
    weights_history = {
        "strategy_a": pd.DataFrame({"asset_1": [0.6, 0.7, 0.8], "asset_2": [0.4, 0.3, 0.2]}, index=dates[[0, 2, 4]]),
        "strategy_b": pd.DataFrame({"asset_1": [0.5, 0.5, 0.4], "asset_2": [0.5, 0.5, 0.6]}, index=dates[[0, 2, 4]]),
    }
    rebalance_results = pd.DataFrame(
        {
            "strategy": ["strategy_a", "strategy_a", "strategy_a", "strategy_b", "strategy_b", "strategy_b"],
            "turnover": [0.10, 0.20, 0.90, 0.30, 0.40, 0.50],
            "concentration": [0.52, 0.58, 0.68, 0.50, 0.50, 0.52],
            "top_3_weight_share": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "realized_vol": [0.12, 0.13, 0.14, 0.11, 0.10, 0.13],
            "forecast_vol": [0.11, 0.12, 0.12, 0.09, 0.09, 0.10],
            "slack_used": [0.00, 0.00, 0.10, 0.00, 0.01, 0.02],
            "chosen_epsilon": [0.001, 0.001, 0.010, 0.001, 0.005, 0.005],
            "objective_value": np.arange(6, dtype=float),
            "status": ["optimal"] * 6,
        },
        index=dates[[0, 2, 4, 0, 2, 4]],
    ).sort_index()

    window_summary = backtest.summarize_backtest_window(
        daily_returns_window=daily_returns,
        gross_daily_returns_window=gross_daily_returns,
        weights_history=weights_history,
        rebalance_results=rebalance_results,
        start_date="2024-01-02",
        end_date="2024-01-05",
    )

    expected_turnover = 0.55
    actual_turnover = float(window_summary.loc["strategy_a", "average_turnover"])
    assert np.isclose(actual_turnover, expected_turnover)


def test_apply_execution_controls_skips_small_trade() -> None:
    previous = pd.Series({"asset_1": 0.60, "asset_2": 0.40})
    proposed = pd.Series({"asset_1": 0.62, "asset_2": 0.38})

    execution = backtest.apply_execution_controls(
        proposed_weights=proposed,
        previous_weights=previous,
        no_trade_band_l1=0.05,
        full_rebalance_band_l1=0.20,
    )

    assert bool(execution["trade_skipped"])
    assert float(execution["execution_eta"]) == 0.0
    pd.testing.assert_series_equal(execution["weights"], previous.rename("weight"))


def test_apply_execution_controls_partially_blends_medium_trade() -> None:
    previous = pd.Series({"asset_1": 0.60, "asset_2": 0.40})
    proposed = pd.Series({"asset_1": 0.70, "asset_2": 0.30})

    execution = backtest.apply_execution_controls(
        proposed_weights=proposed,
        previous_weights=previous,
        no_trade_band_l1=0.05,
        full_rebalance_band_l1=0.25,
    )

    assert not bool(execution["trade_skipped"])
    assert 0.0 < float(execution["execution_eta"]) < 1.0
    assert float(execution["executed_turnover"]) < float(execution["proposed_turnover"])


def test_run_sensitivity_scenarios_parallel_matches_serial() -> None:
    dates = pd.date_range("2023-01-01", periods=220, freq="B")
    rng = np.random.default_rng(11)
    simple_returns = pd.DataFrame(rng.normal(0.0002, 0.01, size=(len(dates), 3)), index=dates, columns=["A", "B", "C"])
    config = {
        "train_window": 120,
        "val_window": 21,
        "scenario_parallel_jobs": 2,
        "scenario_parallel_backend": "loky",
    }
    perturbations = {
        "shift": lambda frame: frame + 0.001,
        "scale": lambda frame: frame * 1.05,
    }
    rebalance_dates = [dates[-20], dates[-10]]

    serial = backtest.run_sensitivity_scenarios(
        simple_returns=simple_returns,
        strategies={"toy": _equal_weight_strategy},
        config=config,
        rebalance_dates=rebalance_dates,
        perturbations=perturbations,
        n_jobs=1,
    ).sort_values(["rebalance_date", "strategy", "perturbation"]).reset_index(drop=True)
    parallel = backtest.run_sensitivity_scenarios(
        simple_returns=simple_returns,
        strategies={"toy": _equal_weight_strategy},
        config=config,
        rebalance_dates=rebalance_dates,
        perturbations=perturbations,
        n_jobs=2,
    ).sort_values(["rebalance_date", "strategy", "perturbation"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(serial, parallel)


def test_run_corruption_stress_parallel_matches_serial() -> None:
    dates = pd.date_range("2023-01-01", periods=220, freq="B")
    rng = np.random.default_rng(19)
    simple_returns = pd.DataFrame(rng.normal(0.0001, 0.01, size=(len(dates), 3)), index=dates, columns=["A", "B", "C"])
    config = {
        "train_window": 120,
        "val_window": 21,
        "rebalance_freq": 21,
        "transaction_cost_bps": 0.0,
        "bounds": (0.0, 1.0),
        "scenario_parallel_jobs": 2,
        "scenario_parallel_backend": "loky",
    }
    corruption_scenarios = {
        "clean": simple_returns,
        "shifted": simple_returns + 0.0005,
    }

    serial = backtest.run_corruption_stress(
        simple_returns=simple_returns,
        strategies={"toy": _equal_weight_strategy},
        corruption_scenarios=corruption_scenarios,
        config=config,
        n_jobs=1,
    ).sort_values(["corruption", "strategy"]).reset_index(drop=True)
    parallel = backtest.run_corruption_stress(
        simple_returns=simple_returns,
        strategies={"toy": _equal_weight_strategy},
        corruption_scenarios=corruption_scenarios,
        config=config,
        n_jobs=2,
    ).sort_values(["corruption", "strategy"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(serial, parallel)


def test_corruption_degradation_summary_adds_relative_columns() -> None:
    summary = pd.DataFrame(
        {
            "corruption": ["clean", "shock", "clean", "shock"],
            "strategy": ["a", "a", "b", "b"],
            "sharpe_ratio": [1.0, 0.8, 0.9, 0.6],
            "average_turnover": [0.10, 0.15, 0.12, 0.20],
            "forecast_realized_risk_gap": [0.02, 0.03, 0.04, 0.06],
            "annualized_return_cost_drag": [0.01, 0.02, 0.015, 0.025],
        }
    )

    degraded = backtest.summarize_corruption_degradation(summary)

    assert {"sharpe_drop_vs_clean", "turnover_increase_vs_clean", "risk_gap_increase_vs_clean"} <= set(degraded.columns)
    shock_a = degraded[(degraded["corruption"] == "shock") & (degraded["strategy"] == "a")].iloc[0]
    assert np.isclose(float(shock_a["sharpe_drop_vs_clean"]), 0.2)


def test_summary_keeps_chosen_epsilon_and_delta_columns_separate() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    daily_returns = pd.DataFrame(
        {
            "proxy": [0.001, 0.0, 0.002, -0.001],
            "drmv": [0.0005, -0.0002, 0.001, 0.0003],
        },
        index=dates,
    )
    weights_history = {
        "proxy": pd.DataFrame({"asset_1": [0.6, 0.7], "asset_2": [0.4, 0.3]}, index=dates[[0, 2]]),
        "drmv": pd.DataFrame({"asset_1": [0.5, 0.45], "asset_2": [0.5, 0.55]}, index=dates[[0, 2]]),
    }
    rebalance_results = pd.DataFrame(
        {
            "strategy": ["proxy", "proxy", "drmv", "drmv"],
            "turnover": [0.1, 0.2, 0.15, 0.1],
            "concentration": [0.52, 0.58, 0.50, 0.51],
            "top_3_weight_share": [1.0, 1.0, 1.0, 1.0],
            "realized_vol": [0.11, 0.12, 0.09, 0.10],
            "forecast_vol": [0.10, 0.11, 0.08, 0.09],
            "slack_used": [0.0, 0.0, 0.0, 0.0],
            "chosen_epsilon": [0.001, 0.002, np.nan, np.nan],
            "chosen_delta": [np.nan, np.nan, 0.01, 0.02],
            "objective_value": [1.0, 1.1, 0.8, 0.9],
            "status": ["optimal"] * 4,
        },
        index=dates[[0, 2, 0, 2]],
    ).sort_index()

    summary = backtest.summarize_backtest(
        daily_returns=daily_returns,
        weights_history=weights_history,
        rebalance_results=rebalance_results,
    )

    assert np.isclose(float(summary.loc["proxy", "average_chosen_epsilon"]), 0.0015)
    assert pd.isna(summary.loc["proxy", "average_chosen_delta"])
    assert pd.isna(summary.loc["drmv", "average_chosen_epsilon"])
    assert np.isclose(float(summary.loc["drmv", "average_chosen_delta"]), 0.015)
