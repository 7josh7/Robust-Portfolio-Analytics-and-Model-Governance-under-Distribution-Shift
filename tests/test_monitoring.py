from __future__ import annotations

import numpy as np
import pandas as pd

from src import features, monitoring


def test_monitoring_handles_single_class_train_or_test_split() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    features = pd.DataFrame(
        {
            "trailing_vol": np.linspace(0.10, 0.20, 10),
            "average_pairwise_correlation": np.linspace(0.30, 0.60, 10),
            "effective_rank": np.linspace(2.0, 5.0, 10),
            "recent_turnover": np.linspace(0.05, 0.25, 10),
            "herfindahl": np.linspace(0.10, 0.16, 10),
            "top_3_weight_share": np.linspace(0.40, 0.60, 10),
            "recent_drawdown": np.linspace(-0.02, -0.08, 10),
            "recent_drawdown_abs": np.linspace(0.02, 0.08, 10),
            "forecast_realized_risk_gap": np.linspace(0.01, 0.06, 10),
        },
        index=dates,
    )
    target = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], index=dates)

    result = monitoring.train_instability_detector(
        feature_frame=features,
        target=target,
        model_type="logistic",
        test_fraction=0.30,
    )

    assert "status" in result
    assert "metrics" in result
    assert "predictions" in result
    assert not result["predictions"].empty


def test_instability_targets_use_rank_based_tails_under_ties() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="W")
    rebalance_results = pd.DataFrame(
        {
            "turnover": [0.0, 0.0, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05],
            "realized_vol": np.linspace(0.10, 0.22, 12),
            "forecast_vol": np.linspace(0.09, 0.18, 12),
            "hold_period_drawdown": [-0.01, -0.02, -0.01, -0.03, -0.02, -0.04, -0.05, -0.04, -0.06, -0.07, -0.08, -0.09],
        },
        index=dates,
    )

    targets = features.build_instability_targets(
        rebalance_results=rebalance_results,
        turnover_quantile=0.80,
        risk_gap_quantile=0.80,
        drawdown_quantile=0.20,
    )

    positive_rates = targets[
        ["turnover_instability_target", "risk_gap_instability_target", "drawdown_instability_target"]
    ].mean()

    assert positive_rates["turnover_instability_target"] < 0.60
    assert positive_rates["risk_gap_instability_target"] < 0.60
    assert positive_rates["drawdown_instability_target"] < 0.60
