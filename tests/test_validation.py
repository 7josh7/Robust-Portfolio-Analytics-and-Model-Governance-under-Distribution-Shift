from __future__ import annotations

import pandas as pd

from src import baselines, validation


def _toy_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.0010, 0.0005, -0.0004, 0.0008, 0.0002, 0.0007],
            "B": [0.0007, 0.0004, -0.0002, 0.0006, 0.0001, 0.0005],
            "C": [0.0005, 0.0003, -0.0001, 0.0004, 0.0002, 0.0004],
        }
    )


def test_zero_radius_alignment_diagnostic_decomposes_variants() -> None:
    train_returns = _toy_returns()
    diagnostic = validation.diagnose_zero_radius_proxy_alignment(
        train_returns=train_returns,
        target_return=0.0002,
        bounds=(0.0, 1.0),
        covariance_method="sample",
        previous_weights=baselines.equal_weight(train_returns.columns),
        turnover_penalty=0.25,
        slack_penalty=100.0,
    )

    expected_variants = {
        "empirical_target_min_var",
        "proxy_hard_constraint_no_turnover",
        "proxy_soft_slack_only",
        "proxy_soft_slack_plus_turnover",
    }
    assert set(diagnostic.index) == expected_variants
    assert float(diagnostic.loc["empirical_target_min_var", "weight_l1_vs_empirical"]) == 0.0
    assert not bool(diagnostic.loc["proxy_hard_constraint_no_turnover", "soft_feasibility_enabled"])
    assert bool(diagnostic.loc["proxy_soft_slack_only", "soft_feasibility_enabled"])
