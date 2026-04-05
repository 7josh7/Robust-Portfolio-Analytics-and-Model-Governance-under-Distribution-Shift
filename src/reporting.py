from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest, features


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return value


def save_weights_history(weights_history: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for strategy, frame in weights_history.items():
        frame.to_csv(output_path / f"weights_{strategy}.csv")


def build_diagnostics_payload(
    config: dict | object,
    summary: pd.DataFrame,
    rebalance_results: pd.DataFrame,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"config": _json_safe(config), "summary": _json_safe(summary)}
    payload["strategy_status_counts"] = _json_safe(
        rebalance_results.groupby("strategy")["status"].value_counts().rename("count").reset_index()
    )
    payload["strategy_diagnostics"] = {}

    for strategy, frame in rebalance_results.groupby("strategy"):
        latest_row = frame.sort_index().iloc[-1].to_dict()
        payload["strategy_diagnostics"][strategy] = {
            "latest": _json_safe(latest_row),
            "averages": _json_safe(
                frame[
                    [
                        column
                        for column in [
                            "turnover",
                            "proposed_turnover",
                            "slack_used",
                            "chosen_epsilon",
                            "chosen_delta",
                            "epsilon_change",
                            "execution_eta",
                            "realized_vol",
                            "forecast_vol",
                        ]
                        if column in frame.columns
                    ]
                ].mean()
            ),
        }
    return payload


def save_diagnostics_json(payload: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def save_backtest_figures(
    daily_returns: pd.DataFrame,
    rebalance_results: pd.DataFrame,
    weights_history: dict[str, pd.DataFrame],
    output_dir: str | Path,
    focus_strategies: list[str] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if focus_strategies is None:
        focus_strategies = [
            strategy
            for strategy in [
                "sample_min_var",
                "sample_mean_variance",
                "wasserstein_proxy_min_var",
                "drmv_regularized_min_var",
                "drmv_regime_covariance_min_var_hmm",
            ]
            if strategy in daily_returns.columns
        ]
        if not focus_strategies:
            focus_strategies = daily_returns.columns.tolist()[:5]

    wealth = (1.0 + daily_returns.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    wealth[focus_strategies].plot(ax=ax)
    ax.set_title("Net Wealth Paths")
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    fig.savefig(output_path / "wealth_paths.png", dpi=150)
    plt.close(fig)

    perf = backtest.build_rolling_performance_diagnostics(daily_returns[focus_strategies], window=126)
    reb = backtest.build_rolling_rebalance_diagnostics(rebalance_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    perf["rolling_sharpe"][focus_strategies].plot(ax=axes[0, 0])
    axes[0, 0].set_title("Rolling 6M Sharpe")
    perf["rolling_drawdown"][focus_strategies].plot(ax=axes[0, 1])
    axes[0, 1].set_title("Rolling 6M Drawdown")
    if "rolling_turnover" in reb:
        reb["rolling_turnover"][focus_strategies].plot(ax=axes[1, 0])
        axes[1, 0].set_title("Rolling Rebalance Turnover")
    if "rolling_risk_gap" in reb:
        reb["rolling_risk_gap"][focus_strategies].plot(ax=axes[1, 1])
        axes[1, 1].set_title("Rolling Rebalance Risk Gap")
    fig.tight_layout()
    fig.savefig(output_path / "rolling_diagnostics.png", dpi=150)
    plt.close(fig)

    proxy_name = "wasserstein_proxy_min_var"
    if proxy_name in rebalance_results["strategy"].unique():
        proxy_frame = rebalance_results[rebalance_results["strategy"] == proxy_name].sort_index()
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        proxy_frame["chosen_epsilon"].plot(ax=axes[0], color="#1f77b4")
        axes[0].set_title("Chosen Epsilon")
        proxy_frame["slack_used"].plot(ax=axes[1], color="#d62728")
        axes[1].set_title("Slack Usage")
        proxy_frame["execution_eta"].plot(ax=axes[2], color="#2ca02c")
        axes[2].set_title("Execution Eta")
        fig.tight_layout()
        fig.savefig(output_path / "wasserstein_proxy_controls.png", dpi=150)
        plt.close(fig)

        if proxy_name in weights_history:
            exposure_history = features.build_bucket_exposure_history(weights_history[proxy_name])
            fig, ax = plt.subplots(figsize=(12, 6))
            exposure_history.plot.area(ax=ax, alpha=0.85)
            ax.set_title("Wasserstein Proxy Bucket Exposures")
            ax.set_ylabel("Weight")
            fig.tight_layout()
            fig.savefig(output_path / "wasserstein_proxy_exposures.png", dpi=150)
            plt.close(fig)
