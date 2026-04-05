from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import herfindahl_index, max_drawdown, top_k_weight_share


StrategyFunction = Callable[[pd.DataFrame, pd.DataFrame, pd.Series | None, dict], dict]


def _config_to_dict(config: dict | object) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("Config must be a dict or dataclass instance.")


def _resolve_parallel_jobs(requested_n_jobs: int | None, task_count: int) -> int:
    if task_count <= 1 or requested_n_jobs is None:
        return 1
    if int(requested_n_jobs) == -1:
        return min(os.cpu_count() or 1, task_count)
    return min(max(int(requested_n_jobs), 1), task_count)


def compute_turnover(new_weights: pd.Series, old_weights: pd.Series | None) -> float:
    if old_weights is None:
        return float(new_weights.abs().sum())
    aligned_old = old_weights.reindex(new_weights.index).fillna(0.0)
    return float((new_weights - aligned_old).abs().sum())


def apply_execution_controls(
    proposed_weights: pd.Series,
    previous_weights: pd.Series | None,
    no_trade_band_l1: float = 0.0,
    full_rebalance_band_l1: float | None = None,
) -> dict[str, object]:
    """Apply no-trade bands and partial execution blending to proposed weights."""

    if previous_weights is None:
        return {
            "weights": proposed_weights,
            "proposed_turnover": float(proposed_weights.abs().sum()),
            "executed_turnover": float(proposed_weights.abs().sum()),
            "execution_eta": 1.0,
            "trade_skipped": False,
        }

    aligned_previous = previous_weights.reindex(proposed_weights.index).fillna(0.0)
    proposed_turnover = float((proposed_weights - aligned_previous).abs().sum())
    if proposed_turnover <= no_trade_band_l1:
        return {
            "weights": aligned_previous.rename("weight"),
            "proposed_turnover": proposed_turnover,
            "executed_turnover": 0.0,
            "execution_eta": 0.0,
            "trade_skipped": True,
        }

    full_band = full_rebalance_band_l1 if full_rebalance_band_l1 is not None else no_trade_band_l1
    if full_band <= no_trade_band_l1:
        eta = 1.0
    else:
        eta = float(np.clip((proposed_turnover - no_trade_band_l1) / (full_band - no_trade_band_l1), 0.0, 1.0))

    executed_weights = aligned_previous + eta * (proposed_weights - aligned_previous)
    executed_weights = executed_weights.clip(lower=0.0)
    executed_weights = executed_weights / executed_weights.sum()
    executed_turnover = float((executed_weights - aligned_previous).abs().sum())
    return {
        "weights": executed_weights.rename("weight"),
        "proposed_turnover": proposed_turnover,
        "executed_turnover": executed_turnover,
        "execution_eta": eta,
        "trade_skipped": False,
    }


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns.fillna(0.0)).prod()
    return float(wealth ** (periods_per_year / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year)) if not returns.empty else np.nan


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    volatility = annualized_volatility(returns, periods_per_year=periods_per_year)
    if pd.isna(volatility) or volatility == 0:
        return np.nan
    return annualized_return(returns, periods_per_year=periods_per_year) / volatility


def downside_deviation(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)) if not downside.empty else 0.0


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    if returns.empty:
        return np.nan
    cutoff = returns.quantile(alpha)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if not tail.empty else cutoff


def run_rolling_backtest(
    simple_returns: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    config: dict | object,
) -> dict[str, object]:
    """Run a rolling train/validation/test portfolio backtest."""

    cfg = _config_to_dict(config)
    returns = simple_returns.sort_index().dropna(how="all")
    train_window = cfg["train_window"]
    val_window = cfg["val_window"]
    rebalance_freq = cfg["rebalance_freq"]
    transaction_cost_bps = cfg.get("transaction_cost_bps", 0.0)
    start_idx = train_window + val_window

    if len(returns) <= start_idx:
        raise ValueError("Not enough observations for the requested train and validation windows.")

    previous_weights = {name: None for name in strategies}
    previous_epsilons = {name: None for name in strategies}
    rebalance_rows: list[dict] = []
    daily_return_store: dict[str, list[pd.Series]] = {name: [] for name in strategies}
    gross_daily_return_store: dict[str, list[pd.Series]] = {name: [] for name in strategies}
    weight_store: dict[str, list[pd.Series]] = {name: [] for name in strategies}
    proposed_weight_store: dict[str, list[pd.Series]] = {name: [] for name in strategies}

    for rebalance_idx in range(start_idx, len(returns), rebalance_freq):
        rebalance_date = returns.index[rebalance_idx]
        train_slice = returns.iloc[rebalance_idx - val_window - train_window : rebalance_idx - val_window]
        val_slice = returns.iloc[rebalance_idx - val_window : rebalance_idx]
        hold_slice = returns.iloc[rebalance_idx : min(rebalance_idx + rebalance_freq, len(returns))]
        hold_slice_filled = hold_slice.fillna(0.0)
        strategy_config = dict(cfg)
        strategy_config["rebalance_date"] = rebalance_date

        for name, strategy in strategies.items():
            strategy_config["previous_epsilon"] = previous_epsilons[name]
            result = strategy(train_slice, val_slice, previous_weights[name], strategy_config)
            proposed_weights = result["weights"].reindex(returns.columns).fillna(0.0)
            execution = apply_execution_controls(
                proposed_weights=proposed_weights,
                previous_weights=previous_weights[name],
                no_trade_band_l1=cfg.get("no_trade_band_l1", 0.0),
                full_rebalance_band_l1=cfg.get("full_rebalance_band_l1"),
            )
            weights = execution["weights"]
            turnover = float(execution["executed_turnover"])
            transaction_cost = turnover * transaction_cost_bps / 10_000.0

            gross_realized_returns = (hold_slice_filled @ weights).rename(name)
            realized_returns = gross_realized_returns.copy()
            if not gross_realized_returns.empty:
                realized_returns.iloc[0] -= transaction_cost

            forecast_vol = result.get("forecast_vol", np.nan)
            realized_vol = annualized_volatility(realized_returns)
            gross_realized_vol = annualized_volatility(gross_realized_returns)
            hold_drawdown = max_drawdown(realized_returns)

            rebalance_rows.append(
                {
                    "date": rebalance_date,
                    "strategy": name,
                    "objective_value": result.get("objective_value", np.nan),
                    "forecast_vol": forecast_vol,
                    "realized_vol": realized_vol,
                    "expected_return": result.get("expected_return", np.nan),
                    "worst_case_return": result.get("worst_case_return", np.nan),
                    "chosen_epsilon": result.get("chosen_epsilon", np.nan),
                    "chosen_delta": result.get("chosen_delta", np.nan),
                    "validation_score": result.get("validation_score", np.nan),
                    "validation_sensitivity_penalty": result.get("validation_sensitivity_penalty", np.nan),
                    "validation_corruption_penalty": result.get("validation_corruption_penalty", np.nan),
                    "validation_stress_penalty": result.get("validation_stress_penalty", np.nan),
                    "target_return": result.get("target_return", np.nan),
                    "nominal_target_return": result.get("nominal_target_return", np.nan),
                    "alpha_bar": result.get("alpha_bar", np.nan),
                    "target_source": result.get("target_source", ""),
                    "target_rule": result.get("target_rule", ""),
                    "covariance_method": result.get("covariance_method", ""),
                    "paper_mode": result.get("paper_mode", ""),
                    "calibration_mode": result.get("calibration_mode", ""),
                    "objective_mode": result.get("objective_mode", ""),
                    "regime_conditioned": result.get("regime_conditioned", False),
                    "stressed_probability": result.get("stressed_probability", np.nan),
                    "regime_model_version": result.get("regime_model_version", ""),
                    "regime_model_status": result.get("regime_model_status", ""),
                    "probability_mode": result.get("probability_mode", ""),
                    "slack_used": result.get("slack_used", np.nan),
                    "binding_margin": result.get("binding_margin", np.nan),
                    "robust_penalty": result.get("robust_penalty", np.nan),
                    "slack_penalty": result.get("slack_penalty", np.nan),
                    "fallback_used": result.get("fallback_used", False),
                    "execution_eta": execution["execution_eta"],
                    "trade_skipped": execution["trade_skipped"],
                    "proposed_turnover": execution["proposed_turnover"],
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "hold_period_return": float((1.0 + realized_returns).prod() - 1.0) if not realized_returns.empty else np.nan,
                    "gross_hold_period_return": float((1.0 + gross_realized_returns).prod() - 1.0)
                    if not gross_realized_returns.empty
                    else np.nan,
                    "hold_period_drawdown": hold_drawdown,
                    "gross_realized_vol": gross_realized_vol,
                    "concentration": herfindahl_index(weights),
                    "top_3_weight_share": top_k_weight_share(weights, k=3),
                    "proposed_concentration": herfindahl_index(proposed_weights),
                    "epsilon_change": (
                        abs(float(result.get("chosen_epsilon", np.nan)) - float(previous_epsilons[name]))
                        if previous_epsilons[name] is not None and pd.notna(result.get("chosen_epsilon", np.nan))
                        else 0.0
                    ),
                    "status": result.get("status", "unknown"),
                }
            )
            weight_store[name].append(weights.rename(rebalance_date))
            proposed_weight_store[name].append(proposed_weights.rename(rebalance_date))
            daily_return_store[name].append(realized_returns)
            gross_daily_return_store[name].append(gross_realized_returns)
            previous_weights[name] = weights
            if pd.notna(result.get("chosen_epsilon", np.nan)):
                previous_epsilons[name] = float(result["chosen_epsilon"])

    rebalance_results = pd.DataFrame(rebalance_rows).set_index("date").sort_index()
    daily_returns = pd.concat(
        {
            name: pd.concat(series_list).sort_index()
            for name, series_list in daily_return_store.items()
            if series_list
        },
        axis=1,
    )
    daily_returns.columns = daily_returns.columns.get_level_values(0)
    weights_history = {
        name: pd.DataFrame(weight_rows).sort_index()
        for name, weight_rows in weight_store.items()
        if weight_rows
    }
    proposed_weights_history = {
        name: pd.DataFrame(weight_rows).sort_index()
        for name, weight_rows in proposed_weight_store.items()
        if weight_rows
    }
    gross_daily_returns = pd.concat(
        {
            name: pd.concat(series_list).sort_index()
            for name, series_list in gross_daily_return_store.items()
            if series_list
        },
        axis=1,
    )
    gross_daily_returns.columns = gross_daily_returns.columns.get_level_values(0)

    return {
        "rebalance_results": rebalance_results,
        "daily_returns": daily_returns,
        "gross_daily_returns": gross_daily_returns,
        "weights_history": weights_history,
        "proposed_weights_history": proposed_weights_history,
    }


def summarize_backtest(
    daily_returns: pd.DataFrame,
    weights_history: dict[str, pd.DataFrame],
    rebalance_results: pd.DataFrame,
    gross_daily_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate return and governance metrics by strategy."""

    summary_rows: list[dict] = []
    for strategy in daily_returns.columns:
        strategy_returns = daily_returns[strategy].dropna()
        strategy_gross_returns = (
            gross_daily_returns[strategy].dropna()
            if gross_daily_returns is not None and strategy in gross_daily_returns.columns
            else pd.Series(dtype=float)
        )
        strategy_weights = weights_history.get(strategy, pd.DataFrame())
        strategy_rebalances = rebalance_results[rebalance_results["strategy"] == strategy]

        avg_concentration = strategy_rebalances["concentration"].mean()
        avg_top3_share = strategy_rebalances["top_3_weight_share"].mean()
        avg_turnover = strategy_rebalances["turnover"].mean()
        avg_proposed_turnover = strategy_rebalances["proposed_turnover"].mean() if "proposed_turnover" in strategy_rebalances else np.nan
        risk_gap = (strategy_rebalances["realized_vol"] - strategy_rebalances["forecast_vol"]).abs().mean()
        avg_slack = strategy_rebalances["slack_used"].mean() if "slack_used" in strategy_rebalances else np.nan
        avg_constraint_margin = strategy_rebalances["binding_margin"].mean() if "binding_margin" in strategy_rebalances else np.nan
        binding_frequency = (
            float((strategy_rebalances["binding_margin"].fillna(np.inf) <= 1e-6).mean())
            if "binding_margin" in strategy_rebalances and not strategy_rebalances.empty
            else np.nan
        )
        positive_slack_fraction = (
            float((strategy_rebalances["slack_used"].fillna(0.0) > 1e-10).mean())
            if "slack_used" in strategy_rebalances and not strategy_rebalances.empty
            else np.nan
        )
        avg_epsilon = strategy_rebalances["chosen_epsilon"].mean() if "chosen_epsilon" in strategy_rebalances else np.nan
        avg_delta = strategy_rebalances["chosen_delta"].mean() if "chosen_delta" in strategy_rebalances else np.nan
        avg_epsilon_change = strategy_rebalances["epsilon_change"].mean() if "epsilon_change" in strategy_rebalances else np.nan
        avg_execution_eta = strategy_rebalances["execution_eta"].mean() if "execution_eta" in strategy_rebalances else np.nan
        trade_skip_fraction = (
            float(strategy_rebalances["trade_skipped"].fillna(False).mean()) if "trade_skipped" in strategy_rebalances else np.nan
        )
        average_weight_change = np.nan
        if len(strategy_weights) > 1:
            average_weight_change = strategy_weights.diff().abs().sum(axis=1).dropna().mean()

        drawdown = max_drawdown(strategy_returns)
        ann_return = annualized_return(strategy_returns)
        gross_ann_return = annualized_return(strategy_gross_returns) if not strategy_gross_returns.empty else np.nan
        gross_ann_vol = annualized_volatility(strategy_gross_returns) if not strategy_gross_returns.empty else np.nan
        gross_sharpe = sharpe_ratio(strategy_gross_returns) if not strategy_gross_returns.empty else np.nan
        summary_rows.append(
            {
                "strategy": strategy,
                "annualized_return": ann_return,
                "annualized_volatility": annualized_volatility(strategy_returns),
                "sharpe_ratio": sharpe_ratio(strategy_returns),
                "gross_annualized_return": gross_ann_return,
                "gross_annualized_volatility": gross_ann_vol,
                "gross_sharpe_ratio": gross_sharpe,
                "annualized_return_cost_drag": gross_ann_return - ann_return if pd.notna(gross_ann_return) else np.nan,
                "max_drawdown": drawdown,
                "calmar_ratio": ann_return / abs(drawdown) if pd.notna(drawdown) and drawdown != 0 else np.nan,
                "downside_deviation": downside_deviation(strategy_returns),
                "cvar_5pct": cvar(strategy_returns, alpha=0.05),
                "average_turnover": avg_turnover,
                "average_proposed_turnover": avg_proposed_turnover,
                "average_concentration": avg_concentration,
                "average_top_3_weight_share": avg_top3_share,
                "average_weight_change_l1": average_weight_change,
                "forecast_realized_risk_gap": risk_gap,
                "average_slack_used": avg_slack,
                "average_constraint_margin": avg_constraint_margin,
                "constraint_binding_frequency": binding_frequency,
                "positive_slack_fraction": positive_slack_fraction,
                "average_chosen_epsilon": avg_epsilon,
                "average_chosen_delta": avg_delta,
                "average_epsilon_change": avg_epsilon_change,
                "average_execution_eta": avg_execution_eta,
                "trade_skip_fraction": trade_skip_fraction,
            }
        )

    return pd.DataFrame(summary_rows).set_index("strategy").sort_values("sharpe_ratio", ascending=False)


def filter_backtest_window(
    daily_returns: pd.DataFrame,
    weights_history: dict[str, pd.DataFrame],
    rebalance_results: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    gross_daily_returns: pd.DataFrame | None = None,
) -> dict[str, object]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    filtered_rebalances = rebalance_results.loc[(rebalance_results.index >= start_ts) & (rebalance_results.index <= end_ts)].copy()
    filtered_weights = {
        name: frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
        for name, frame in weights_history.items()
    }
    filtered_daily_returns = daily_returns.loc[(daily_returns.index >= start_ts) & (daily_returns.index <= end_ts)].copy()
    filtered_gross_returns = None
    if gross_daily_returns is not None:
        filtered_gross_returns = gross_daily_returns.loc[
            (gross_daily_returns.index >= start_ts) & (gross_daily_returns.index <= end_ts)
        ].copy()

    return {
        "daily_returns": filtered_daily_returns,
        "gross_daily_returns": filtered_gross_returns,
        "weights_history": filtered_weights,
        "rebalance_results": filtered_rebalances,
    }


def summarize_backtest_window(
    daily_returns_window: pd.DataFrame,
    weights_history: dict[str, pd.DataFrame],
    rebalance_results: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    gross_daily_returns_window: pd.DataFrame | None = None,
) -> pd.DataFrame:
    window_artifacts = filter_backtest_window(
        daily_returns=daily_returns_window,
        gross_daily_returns=gross_daily_returns_window,
        weights_history=weights_history,
        rebalance_results=rebalance_results,
        start_date=start_date,
        end_date=end_date,
    )
    return summarize_backtest(
        daily_returns=window_artifacts["daily_returns"],
        gross_daily_returns=window_artifacts["gross_daily_returns"],
        weights_history=window_artifacts["weights_history"],
        rebalance_results=window_artifacts["rebalance_results"],
    )


def build_rolling_performance_diagnostics(
    daily_returns: pd.DataFrame,
    window: int = 126,
    periods_per_year: int = 252,
) -> dict[str, pd.DataFrame]:
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    rolling_drawdown = daily_returns.rolling(window).apply(lambda x: max_drawdown(pd.Series(x)), raw=False)
    return {
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_drawdown,
    }


def build_rolling_rebalance_diagnostics(
    rebalance_results: pd.DataFrame,
    window: int = 6,
) -> dict[str, pd.DataFrame]:
    diagnostics: dict[str, pd.DataFrame] = {}
    fields = [
        "turnover",
        "proposed_turnover",
        "slack_used",
        "binding_margin",
        "epsilon_change",
        "chosen_epsilon",
        "chosen_delta",
        "execution_eta",
        "forecast_vol",
        "realized_vol",
    ]
    rebalance_results = rebalance_results.copy()
    rebalance_results["rebalance_date"] = rebalance_results.index
    rebalance_results["risk_gap"] = (rebalance_results["realized_vol"] - rebalance_results["forecast_vol"]).abs()
    fields.append("risk_gap")

    for field in fields:
        if field not in rebalance_results.columns:
            continue
        pivot = rebalance_results.pivot_table(index="rebalance_date", columns="strategy", values=field, aggfunc="last")
        diagnostics[field] = pivot.sort_index()
        diagnostics[f"rolling_{field}"] = pivot.sort_index().rolling(window).mean()
    return diagnostics


def _run_single_sensitivity_rebalance(
    rebalance_date: pd.Timestamp,
    simple_returns: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    cfg: dict,
    perturbations: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> list[dict]:
    rows: list[dict] = []
    rebalance_loc = simple_returns.index.get_indexer([pd.Timestamp(rebalance_date)], method="nearest")[0]
    train_end = rebalance_loc - cfg["val_window"]
    train_start = train_end - cfg["train_window"]
    if train_start < 0:
        return rows

    train_sample = simple_returns.iloc[train_start:train_end]
    val_sample = simple_returns.iloc[train_end:rebalance_loc]
    strategy_config = dict(cfg)
    strategy_config["rebalance_date"] = pd.Timestamp(rebalance_date)
    strategy_config["previous_epsilon"] = None

    base_results = {
        name: strategy(train_sample, val_sample, None, strategy_config)
        for name, strategy in strategies.items()
    }

    for perturbation_name, perturbation in perturbations.items():
        perturbed_train = perturbation(train_sample)
        perturbed_results = {
            name: strategy(perturbed_train, val_sample, None, strategy_config)
            for name, strategy in strategies.items()
        }
        for strategy_name, base_result in base_results.items():
            base_weights = base_result["weights"]
            perturbed_weights = perturbed_results[strategy_name]["weights"].reindex(base_weights.index).fillna(0.0)
            rows.append(
                {
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "strategy": strategy_name,
                    "perturbation": perturbation_name,
                    "weight_l1_change": float((base_weights - perturbed_weights).abs().sum()),
                    "concentration_change": herfindahl_index(perturbed_weights) - herfindahl_index(base_weights),
                    "top_3_share_change": top_k_weight_share(perturbed_weights, k=3) - top_k_weight_share(base_weights, k=3),
                    "forecast_vol_change": perturbed_results[strategy_name]["forecast_vol"] - base_result["forecast_vol"],
                }
            )
    return rows


def run_sensitivity_scenarios(
    simple_returns: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    config: dict | object,
    rebalance_dates: list[pd.Timestamp],
    perturbations: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    n_jobs: int | None = None,
) -> pd.DataFrame:
    cfg = _config_to_dict(config)
    requested_jobs = cfg.get("scenario_parallel_jobs", 1) if n_jobs is None else n_jobs
    parallel_jobs = _resolve_parallel_jobs(requested_jobs, len(rebalance_dates))
    if parallel_jobs == 1:
        rows = [
            row
            for rebalance_date in rebalance_dates
            for row in _run_single_sensitivity_rebalance(
                rebalance_date=rebalance_date,
                simple_returns=simple_returns,
                strategies=strategies,
                cfg=cfg,
                perturbations=perturbations,
            )
        ]
    else:
        task_rows = Parallel(
            n_jobs=parallel_jobs,
            backend=cfg.get("scenario_parallel_backend", "loky"),
            max_nbytes=None,
        )(
            delayed(_run_single_sensitivity_rebalance)(
                rebalance_date=rebalance_date,
                simple_returns=simple_returns,
                strategies=strategies,
                cfg=cfg,
                perturbations=perturbations,
            )
            for rebalance_date in rebalance_dates
        )
        rows = [row for batch in task_rows for row in batch]
    return pd.DataFrame(rows)


def _run_single_corruption_scenario(
    scenario_name: str,
    scenario_returns: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    cfg: dict,
) -> pd.DataFrame:
    artifacts = run_rolling_backtest(simple_returns=scenario_returns, strategies=strategies, config=cfg)
    summary = summarize_backtest(
        daily_returns=artifacts["daily_returns"],
        gross_daily_returns=artifacts["gross_daily_returns"],
        weights_history=artifacts["weights_history"],
        rebalance_results=artifacts["rebalance_results"],
    ).reset_index()
    summary["corruption"] = scenario_name
    return summary


def run_corruption_stress(
    simple_returns: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    corruption_scenarios: dict[str, pd.DataFrame],
    config: dict | object,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    cfg = _config_to_dict(config)
    scenario_items = list(corruption_scenarios.items())
    requested_jobs = cfg.get("scenario_parallel_jobs", 1) if n_jobs is None else n_jobs
    parallel_jobs = _resolve_parallel_jobs(requested_jobs, len(scenario_items))
    if parallel_jobs == 1:
        rows = [
            _run_single_corruption_scenario(
                scenario_name=scenario_name,
                scenario_returns=scenario_returns,
                strategies=strategies,
                cfg=cfg,
            )
            for scenario_name, scenario_returns in scenario_items
        ]
    else:
        rows = Parallel(
            n_jobs=parallel_jobs,
            backend=cfg.get("scenario_parallel_backend", "loky"),
            max_nbytes=None,
        )(
            delayed(_run_single_corruption_scenario)(
                scenario_name=scenario_name,
                scenario_returns=scenario_returns,
                strategies=strategies,
                cfg=cfg,
            )
            for scenario_name, scenario_returns in scenario_items
        )
    return pd.concat(rows, ignore_index=True)


def summarize_sensitivity_results(sensitivity_results: pd.DataFrame) -> pd.DataFrame:
    if sensitivity_results.empty:
        return pd.DataFrame()
    return (
        sensitivity_results.groupby(["strategy", "perturbation"], observed=True)
        .agg(
            avg_weight_l1_change=("weight_l1_change", "mean"),
            max_weight_l1_change=("weight_l1_change", "max"),
            avg_concentration_change=("concentration_change", "mean"),
            avg_top_3_share_change=("top_3_share_change", "mean"),
            avg_forecast_vol_change=("forecast_vol_change", "mean"),
        )
        .reset_index()
        .sort_values(["perturbation", "avg_weight_l1_change", "strategy"], ascending=[True, False, True])
    )


def summarize_corruption_degradation(
    corruption_summary: pd.DataFrame,
    clean_label: str = "clean",
) -> pd.DataFrame:
    if corruption_summary.empty:
        return pd.DataFrame()
    if clean_label not in set(corruption_summary["corruption"]):
        return corruption_summary.copy()

    clean_reference = corruption_summary[corruption_summary["corruption"] == clean_label].set_index("strategy")
    enriched = corruption_summary.copy()
    enriched["sharpe_drop_vs_clean"] = enriched.apply(
        lambda row: clean_reference.loc[row["strategy"], "sharpe_ratio"] - row["sharpe_ratio"],
        axis=1,
    )
    enriched["turnover_increase_vs_clean"] = enriched.apply(
        lambda row: row["average_turnover"] - clean_reference.loc[row["strategy"], "average_turnover"],
        axis=1,
    )
    enriched["risk_gap_increase_vs_clean"] = enriched.apply(
        lambda row: row["forecast_realized_risk_gap"] - clean_reference.loc[row["strategy"], "forecast_realized_risk_gap"],
        axis=1,
    )
    enriched["gross_to_net_drag_change"] = enriched.apply(
        lambda row: row["annualized_return_cost_drag"] - clean_reference.loc[row["strategy"], "annualized_return_cost_drag"],
        axis=1,
    )
    return enriched


def summarize_by_group(
    daily_returns: pd.DataFrame,
    group_series: pd.Series,
) -> pd.DataFrame:
    """Summarize performance conditional on an external grouping variable."""

    rows: list[dict] = []
    for strategy in daily_returns.columns:
        joined = pd.concat([daily_returns[strategy], group_series.rename("group")], axis=1, join="inner").dropna()
        for group_name, frame in joined.groupby("group"):
            returns = frame[strategy]
            rows.append(
                {
                    "strategy": strategy,
                    "group": group_name,
                    "annualized_return": annualized_return(returns),
                    "annualized_volatility": annualized_volatility(returns),
                    "sharpe_ratio": sharpe_ratio(returns),
                    "max_drawdown": max_drawdown(returns),
                }
            )
    return pd.DataFrame(rows)
