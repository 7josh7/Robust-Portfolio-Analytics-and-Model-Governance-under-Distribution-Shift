from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest


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
