from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "Robust Portfolio Analytics and Model Governance under Distribution Shift.ipynb"
CELL_COUNTER = count()


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"cell-{next(CELL_COUNTER):04d}",
        "metadata": {},
        "source": dedent(source).splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"cell-{next(CELL_COUNTER):04d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """# Robust Portfolio Analytics and Model Governance under Distribution Shift

This notebook studies portfolio construction under estimation fragility, but it does so with a deliberately honest scope. The main engine is a **Wasserstein-inspired robust proxy allocator**, not a full distributionally robust optimization library. The project emphasizes reproducibility, soft-feasibility diagnostics, corruption-aware stress testing, and governance-style monitoring rather than claiming unconditional outperformance.
"""
    ),
    markdown_cell(
        """## Section 0 - Implemented vs not fully implemented

**Implemented**

- Rolling out-of-sample backtesting with explicit train, validation, and test windows
- Frozen market-data workflow with a cached raw price snapshot
- Baselines, a fragile sample mean-variance benchmark, a Wasserstein-proxy min-variance allocator, and a paper-aligned DR mean-variance branch
- Soft feasibility via a slack variable, composite radius selection, epsilon smoothing, and execution controls
- Historical stress windows, model stress scenarios, and corruption-aware noisy-data experiments
- Prototype monitoring workflow with separate instability targets and calibration metrics
- A targeted heuristic regime-tagging extension used only for conditional governance diagnostics
- Static versus regime-conditioned DRMV inputs using a lightweight two-state market-factor regime engine

**Not fully implemented**

- Exact Wasserstein DRO dual reformulations from the theory papers
- Exact noisy-channel ambiguity-set construction from the TV-noise paper
- The exact Wasserstein-Kelly log-return ambiguity construction and convex reformulation
- Production-scale data engineering, broker connectivity, or enterprise orchestration
"""
    ),
    markdown_cell(
        """## Section 0.5 - Paper alignment and scope

The theoretical papers motivate this project, but the implementation stays intentionally narrower than the original theory:

- The main proxy allocator is a **Wasserstein-inspired proxy**, not an exact Esfahani-Kuhn reformulation.
- The new DR mean-variance branch is closer in spirit to Blanchet-Chen-Zhou, but it is still a practical implementation rather than the paper's full inference stack.
- The noisy-data section is a **corruption-aware stress-testing layer**, not Farokhi's convolution-based noisy-observation ambiguity model.
- The Kelly-style appendix below is a **log-return-space proxy** inspired by the paper, not the paper's exact convex reformulation.
"""
    ),
    markdown_cell(
        """## Section 1 - Imports and reproducibility controls

The project now uses a typed config object loaded from `configs/base.yaml`. The data horizon is frozen with a fixed end date and the workflow reads from `data/raw_prices.parquet` by default, only downloading if the cache is missing or explicitly refreshed.
"""
    ),
    code_cell(
        """# Optional bootstrap for a fresh environment.
# %pip install -r requirements.txt
"""
    ),
    code_cell(
        """from pathlib import Path
import logging
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import backtest, baselines, corruption, data, features, monitoring, regime, robust, selection, validation
from src.config import BacktestConfig

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

CONFIG = BacktestConfig.from_yaml(PROJECT_ROOT / "configs" / "base.yaml")
SEED = CONFIG.seed
np.random.seed(SEED)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

CONFIG
"""
    ),
    markdown_cell(
        """## Section 2 - Data pipeline and frozen snapshot

This notebook uses a liquid ETF universe because it is easy to explain, broad enough to expose cross-asset fragility, and compact enough to keep the notebook readable. The important reproducibility fix is that the raw price panel is cached on disk and replayed from the same file on subsequent runs.
"""
    ),
    code_cell(
        """UNIVERSE = data.DEFAULT_UNIVERSE[: CONFIG.target_universe_size]
RAW_DATA_PATH = PROJECT_ROOT / CONFIG.raw_data_path

prices_raw = data.load_or_download_price_data(
    tickers=UNIVERSE,
    start=CONFIG.start_date,
    end=CONFIG.end_date,
    raw_data_path=RAW_DATA_PATH,
    auto_adjust=True,
    progress=False,
    refresh=CONFIG.refresh_data,
)

prices = data.clean_price_panel(prices_raw, max_missing_frac=0.05, forward_fill_limit=3)
bundle = data.compute_returns(prices)

prices = bundle.prices
simple_returns = bundle.simple_returns
log_returns = bundle.log_returns

quality_report = data.build_data_quality_report(prices, simple_returns)
display(quality_report["date_coverage"])
display(quality_report["missingness"].to_frame())
display(quality_report["annualized_volatility"].to_frame())
"""
    ),
    markdown_cell(
        """## Section 3 - Modeling choice and research claim

The claim in this notebook is intentionally narrower than "robust optimization outperformed." A more defensible objective is:

> A Wasserstein-inspired proxy allocator and a paper-aligned DR mean-variance branch give us disciplined ways to measure robustness tradeoffs under noisy or stressed inputs, but the current specification still treats robustness as something to **measure and compare**, not something to assume.

That is why the notebook measures:

- allocation stability under perturbations
- gross versus net performance
- concentration and turnover
- forecast versus realized risk gaps
- positive-slack frequency and chosen-epsilon diagnostics
"""
    ),
    markdown_cell(
        """## Section 4 - Strategy set

The strategy set now has two robust branches:

- `wasserstein_proxy_min_var`: a practical proxy baseline
- `drmv_regularized_min_var`: a closer-to-paper DR mean-variance branch

There is also a `drmv_regime_conditioned_min_var` variant that feeds the DRMV branch with regime-conditioned expected returns and covariance estimates from a lightweight two-state market-factor model.
"""
    ),
    code_cell(
        """def equal_weight_strategy(train_returns, val_returns, previous_weights, config):
    return baselines.fit_equal_weight(train_returns)


def inverse_vol_strategy(train_returns, val_returns, previous_weights, config):
    return baselines.fit_inverse_volatility(train_returns)


def sample_min_var_strategy(train_returns, val_returns, previous_weights, config):
    return baselines.fit_sample_min_variance(
        train_returns=train_returns,
        target_return=None,
        bounds=tuple(config["bounds"]),
        previous_weights=previous_weights,
        turnover_penalty=config["turnover_penalty"],
    )


def shrinkage_min_var_strategy(train_returns, val_returns, previous_weights, config):
    return baselines.fit_shrinkage_min_variance(
        train_returns=train_returns,
        target_return=None,
        bounds=tuple(config["bounds"]),
        previous_weights=previous_weights,
        turnover_penalty=config["turnover_penalty"],
    )


def sample_mean_variance_strategy(train_returns, val_returns, previous_weights, config):
    return baselines.fit_sample_mean_variance(
        train_returns=train_returns,
        bounds=(tuple(config["bounds"])[0], 0.35),
        previous_weights=previous_weights,
        turnover_penalty=config["turnover_penalty"],
        risk_aversion=4.0,
    )


def wasserstein_proxy_strategy(train_returns, val_returns, previous_weights, config):
    return robust.tune_wasserstein_proxy_radius(
        train_returns=train_returns,
        val_returns=val_returns,
        epsilon_grid=config["wasserstein_proxy_radius_grid"],
        covariance_method=config["covariance_method"],
        bounds=tuple(config["bounds"]),
        previous_weights=previous_weights,
        turnover_penalty=config["turnover_penalty"],
        slack_penalty=config["slack_penalty"],
        metric=config["robust_validation_metric"],
        target_return_mode=config["target_return_mode"],
        target_return_scale=config["target_return_scale"],
        target_return_quantile=config["target_return_quantile"],
        fixed_target_return=config["fixed_target_return"],
        previous_epsilon=config.get("previous_epsilon"),
        selection_slack_penalty_weight=config["selection_slack_penalty_weight"],
        selection_turnover_penalty_weight=config["selection_turnover_penalty_weight"],
        selection_risk_gap_penalty_weight=config["selection_risk_gap_penalty_weight"],
        selection_epsilon_change_penalty_weight=config["selection_epsilon_change_penalty_weight"],
        rebalance_date=config.get("rebalance_date"),
    )


def drmv_regularized_strategy(train_returns, val_returns, previous_weights, config):
    return selection.tune_drmv_regularized_min_variance(
        train_returns=train_returns,
        val_returns=val_returns,
        delta_grid=config["drmv_delta_grid"],
        alpha_bar_scale_grid=config["drmv_alpha_bar_scale_grid"],
        covariance_methods=config["drmv_covariance_methods"],
        bounds=tuple(config["bounds"]),
        previous_weights=previous_weights,
        turnover_penalty=config["turnover_penalty"],
        p_norm=config["drmv_p_norm"],
        target_method=config["drmv_target_method"],
        target_scale=config["drmv_target_scale"],
        target_quantile=config["target_return_quantile"],
        fixed_target_return=config["fixed_target_return"],
        alpha_bar_rule=config["drmv_alpha_bar_rule"],
        selection_turnover_penalty_weight=config["selection_turnover_penalty_weight"],
        selection_risk_gap_penalty_weight=config["selection_risk_gap_penalty_weight"],
        selection_constraint_penalty_weight=config["selection_constraint_penalty_weight"],
        selection_fallback_penalty_weight=config["selection_fallback_penalty_weight"],
        selection_sensitivity_penalty_weight=config["selection_sensitivity_penalty_weight"],
        selection_corruption_penalty_weight=config["selection_corruption_penalty_weight"],
        selection_stress_penalty_weight=config["selection_stress_penalty_weight"],
        mean_perturbation_scale=config["selection_mean_perturbation_scale"],
        covariance_perturbation_scale=config["selection_covariance_perturbation_scale"],
        corruption_noise_scale=config["selection_corruption_noise_scale"],
        stress_quantile=config["selection_stress_quantile"],
        selection_sensitivity_top_k=config["selection_sensitivity_top_k"],
        metric=config["robust_validation_metric"],
        rebalance_date=config.get("rebalance_date"),
    )


def _build_regime_conditioned_drmv_result(train_returns, val_returns, previous_weights, config, input_mode="both"):
    regime_inputs = selection.prepare_regime_conditioned_inputs(
        train_returns=train_returns,
        lookback=config["regime_lookback"],
        n_regimes=config["regime_states"],
        covariance_method=config["regime_covariance_method"],
        calm_covariance_method=config["regime_calm_covariance_method"],
        stressed_covariance_method=config["regime_stressed_covariance_method"],
        probability_temperature=config["regime_probability_temperature"],
        stressed_probability_threshold=config["regime_probability_threshold"],
        random_state=config["seed"],
    )
    overrides = selection.build_regime_search_overrides(
        delta_grid=config["drmv_delta_grid"],
        turnover_penalty=config["turnover_penalty"],
        stress_activation=regime_inputs["stress_activation"],
        stressed_delta_grid_multiplier=config["regime_stressed_delta_grid_multiplier"],
        stressed_turnover_multiplier=config["regime_stressed_turnover_multiplier"],
    )
    mean_vector = regime_inputs["mean_returns"] if input_mode in {"mean", "both"} else None
    covariance_matrix = regime_inputs["covariance"] if input_mode in {"covariance", "both"} else None
    return selection.tune_drmv_regularized_min_variance(
        train_returns=train_returns,
        val_returns=val_returns,
        delta_grid=overrides["delta_grid"],
        alpha_bar_scale_grid=config["drmv_alpha_bar_scale_grid"],
        covariance_methods=config["drmv_covariance_methods"],
        bounds=tuple(config["bounds"]),
        previous_weights=previous_weights,
        turnover_penalty=overrides["turnover_penalty"],
        p_norm=config["drmv_p_norm"],
        target_method=config["drmv_target_method"],
        target_scale=config["drmv_target_scale"],
        target_quantile=config["target_return_quantile"],
        fixed_target_return=config["fixed_target_return"],
        alpha_bar_rule=config["drmv_alpha_bar_rule"],
        selection_turnover_penalty_weight=config["selection_turnover_penalty_weight"],
        selection_risk_gap_penalty_weight=config["selection_risk_gap_penalty_weight"],
        selection_constraint_penalty_weight=config["selection_constraint_penalty_weight"],
        selection_fallback_penalty_weight=config["selection_fallback_penalty_weight"],
        selection_sensitivity_penalty_weight=config["selection_sensitivity_penalty_weight"],
        selection_corruption_penalty_weight=config["selection_corruption_penalty_weight"],
        selection_stress_penalty_weight=config["selection_stress_penalty_weight"],
        mean_perturbation_scale=config["selection_mean_perturbation_scale"],
        covariance_perturbation_scale=config["selection_covariance_perturbation_scale"],
        corruption_noise_scale=config["selection_corruption_noise_scale"],
        stress_quantile=config["selection_stress_quantile"],
        selection_sensitivity_top_k=config["selection_sensitivity_top_k"],
        metric=config["robust_validation_metric"],
        mean_returns=mean_vector,
        covariance=covariance_matrix,
        regime_conditioned=True,
        stressed_probability=regime_inputs["stress_activation"],
        stressed_target_scale=config["regime_stressed_target_scale"],
        stressed_delta_scale=config["regime_stressed_delta_scale"],
        rebalance_date=config.get("rebalance_date"),
    )


def drmv_regime_conditioned_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="both",
    )


def drmv_regime_mean_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="mean",
    )


def drmv_regime_covariance_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="covariance",
    )


STRATEGIES = {
    "equal_weight": equal_weight_strategy,
    "inverse_vol": inverse_vol_strategy,
    "sample_min_var": sample_min_var_strategy,
    "shrinkage_min_var": shrinkage_min_var_strategy,
    "sample_mean_variance": sample_mean_variance_strategy,
    "wasserstein_proxy_min_var": wasserstein_proxy_strategy,
    "drmv_regularized_min_var": drmv_regularized_strategy,
    "drmv_regime_conditioned_min_var": drmv_regime_conditioned_strategy,
}
"""
    ),
    code_cell(
        """artifacts = backtest.run_rolling_backtest(simple_returns=simple_returns, strategies=STRATEGIES, config=CONFIG)
rebalance_results = artifacts["rebalance_results"]
daily_returns = artifacts["daily_returns"]
gross_daily_returns = artifacts["gross_daily_returns"]
weights_history = artifacts["weights_history"]

summary_table = backtest.summarize_backtest(
    daily_returns=daily_returns,
    gross_daily_returns=gross_daily_returns,
    weights_history=weights_history,
    rebalance_results=rebalance_results,
)
summary_table
"""
    ),
    code_cell(
        """fig, ax = plt.subplots(figsize=(12, 6))
((1 + daily_returns.fillna(0.0)).cumprod()).plot(ax=ax)
ax.set_title("Net Wealth Paths")
ax.set_ylabel("Growth of $1")
plt.show()

proxy_diagnostics = rebalance_results[rebalance_results["strategy"] == "wasserstein_proxy_min_var"][
    ["chosen_epsilon", "target_return", "slack_used", "binding_margin", "turnover", "forecast_vol", "realized_vol"]
]
display(proxy_diagnostics.tail(12))
display(proxy_diagnostics[["chosen_epsilon", "slack_used"]].describe())

proxy_diagnostics = proxy_diagnostics.join(
    pd.qcut(
        proxy_diagnostics["realized_vol"].rank(method="first"),
        q=3,
        labels=["low_vol", "mid_vol", "high_vol"],
    ).rename("realized_vol_regime")
)
display(proxy_diagnostics.groupby("realized_vol_regime")[["chosen_epsilon", "slack_used"]].mean())

drmv_diagnostics = rebalance_results[rebalance_results["strategy"].isin(["drmv_regularized_min_var", "drmv_regime_conditioned_min_var"])][
    ["strategy", "chosen_delta", "alpha_bar", "nominal_target_return", "binding_margin", "covariance_method", "stressed_probability", "fallback_used"]
]
display(drmv_diagnostics.tail(12))
"""
    ),
    markdown_cell(
        """## Section 5 - Split-aware evaluation and rolling diagnostics

Headline metrics can hide instability. The tables below separate the validation period from early and late test segments, then add rolling diagnostics so we can see whether stability claims persist through time rather than only in averages.
"""
    ),
    code_cell(
        """test_start = simple_returns.index[CONFIG.train_window + CONFIG.val_window]
test_midpoint = daily_returns.loc[test_start:].index[len(daily_returns.loc[test_start:]) // 2]

split_windows = {
    "early_test": (test_start, test_midpoint),
    "late_test": (test_midpoint, daily_returns.index.max()),
    "full_out_of_sample": (daily_returns.index.min(), daily_returns.index.max()),
}

split_rows = []
for label, (start, end) in split_windows.items():
    split_summary = backtest.summarize_backtest_window(
        daily_returns_window=daily_returns,
        gross_daily_returns_window=gross_daily_returns,
        weights_history=weights_history,
        rebalance_results=rebalance_results,
        start_date=start,
        end_date=end,
    ).reset_index()
    split_summary["window"] = label
    split_rows.append(split_summary)

split_summary_table = pd.concat(split_rows, ignore_index=True)
display(split_summary_table[["window", "strategy", "annualized_return", "gross_annualized_return", "sharpe_ratio", "gross_sharpe_ratio", "average_turnover", "trade_skip_fraction", "average_execution_eta", "constraint_binding_frequency", "positive_slack_fraction"]])

validation_score_summary = (
    rebalance_results.groupby("strategy")[[
        "validation_score",
        "chosen_epsilon",
        "chosen_delta",
        "slack_used",
        "execution_eta",
        "epsilon_change",
        "binding_margin",
        "validation_sensitivity_penalty",
        "validation_corruption_penalty",
        "validation_stress_penalty",
        "stressed_probability",
    ]]
    .mean()
    .rename(columns={"validation_score": "avg_validation_score"})
)
display(validation_score_summary)

drmv_summary = summary_table.loc[
    [strategy for strategy in summary_table.index if strategy in ["drmv_regularized_min_var", "drmv_regime_conditioned_min_var", "wasserstein_proxy_min_var"]],
    [
        "annualized_return",
        "sharpe_ratio",
        "average_turnover",
        "forecast_realized_risk_gap",
        "average_constraint_margin",
        "constraint_binding_frequency",
        "average_chosen_delta",
    ],
]
display(drmv_summary)

rolling_perf = backtest.build_rolling_performance_diagnostics(daily_returns, window=126)
rolling_rebalance = backtest.build_rolling_rebalance_diagnostics(rebalance_results, window=6)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
rolling_perf["rolling_sharpe"][["sample_min_var", "wasserstein_proxy_min_var", "drmv_regularized_min_var", "drmv_regime_conditioned_min_var"]].plot(ax=axes[0, 0])
axes[0, 0].set_title("Rolling 6M Sharpe")
axes[0, 0].set_ylabel("Sharpe")

rolling_perf["rolling_drawdown"][["sample_min_var", "wasserstein_proxy_min_var", "drmv_regularized_min_var", "drmv_regime_conditioned_min_var"]].plot(ax=axes[0, 1])
axes[0, 1].set_title("Rolling 6M Drawdown")
axes[0, 1].set_ylabel("Drawdown")

rolling_rebalance["rolling_turnover"][["sample_min_var", "wasserstein_proxy_min_var", "drmv_regularized_min_var", "drmv_regime_conditioned_min_var"]].plot(ax=axes[1, 0])
axes[1, 0].set_title("Rolling Rebalance Turnover")
axes[1, 0].set_ylabel("Turnover")

rolling_rebalance["rolling_risk_gap"][["sample_min_var", "wasserstein_proxy_min_var", "drmv_regularized_min_var", "drmv_regime_conditioned_min_var"]].plot(ax=axes[1, 1])
axes[1, 1].set_title("Rolling Rebalance Risk Gap")
axes[1, 1].set_ylabel("|realized - forecast vol|")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
proxy_only = rebalance_results[rebalance_results["strategy"] == "wasserstein_proxy_min_var"].sort_index()
proxy_only["chosen_epsilon"].plot(ax=axes[0], color="#1f77b4")
axes[0].set_title("Proxy Chosen Epsilon")
axes[0].set_ylabel("epsilon")

proxy_only["slack_used"].plot(ax=axes[1], color="#d62728")
axes[1].set_title("Proxy Slack Usage")
axes[1].set_ylabel("slack")

proxy_only["execution_eta"].plot(ax=axes[2], color="#2ca02c")
axes[2].set_title("Proxy Execution Eta")
axes[2].set_ylabel("eta")
plt.tight_layout()
plt.show()
"""
    ),
    code_cell(
        """proxy_bucket_exposure = features.build_bucket_exposure_history(weights_history["wasserstein_proxy_min_var"])
sample_bucket_exposure = features.build_bucket_exposure_history(weights_history["sample_min_var"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
proxy_bucket_exposure.plot.area(ax=axes[0], alpha=0.85)
axes[0].set_title("Wasserstein Proxy Bucket Exposures")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Weight")

sample_bucket_exposure.plot.area(ax=axes[1], alpha=0.85)
axes[1].set_title("Sample Min-Var Bucket Exposures")
axes[1].set_xlabel("Date")
plt.show()
"""
    ),
    markdown_cell(
        """## Section 5.5 - Static versus regime-conditioned DRMV inputs

The regime-conditioned branch needs to earn its place empirically. This comparison isolates three ways of feeding state-aware inputs into DRMV on a recent, targeted rebalance sample so the attribution is easier to read:

- regime-conditioned mean only
- regime-conditioned covariance only
- both together

That helps distinguish whether the current gain is coming from state-aware covariance, state-aware mean, or their combination.
"""
    ),
    code_cell(
        """drmv_input_strategies = {
    "drmv_regularized_min_var": drmv_regularized_strategy,
    "drmv_regime_mean_min_var": drmv_regime_mean_strategy,
    "drmv_regime_covariance_min_var": drmv_regime_covariance_strategy,
    "drmv_regime_conditioned_min_var": drmv_regime_conditioned_strategy,
}

regime_input_config = CONFIG.to_dict()
regime_input_config["rebalance_freq"] = CONFIG.rebalance_freq * 2
regime_input_returns = simple_returns.loc["2018-01-01":].copy()

drmv_input_artifacts = backtest.run_rolling_backtest(
    simple_returns=regime_input_returns,
    strategies=drmv_input_strategies,
    config=regime_input_config,
)

drmv_input_summary = backtest.summarize_backtest(
    daily_returns=drmv_input_artifacts["daily_returns"],
    gross_daily_returns=drmv_input_artifacts["gross_daily_returns"],
    weights_history=drmv_input_artifacts["weights_history"],
    rebalance_results=drmv_input_artifacts["rebalance_results"],
)

drmv_input_rebalances = drmv_input_artifacts["rebalance_results"]
stress_split = pd.qcut(
    drmv_input_rebalances["stressed_probability"].fillna(0.0).rank(method="first"),
    q=3,
    labels=["low_stress", "mid_stress", "high_stress"],
)
stress_conditionals = (
    drmv_input_rebalances.assign(stress_bucket=stress_split)
    .groupby(["strategy", "stress_bucket"])[["turnover", "binding_margin", "realized_vol", "forecast_vol"]]
    .mean()
)
drmv_tradeoff = drmv_input_summary[
    [
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "forecast_realized_risk_gap",
        "constraint_binding_frequency",
        "average_chosen_delta",
    ]
]

display(drmv_tradeoff)
display(stress_conditionals)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=False)
drmv_tradeoff["sharpe_ratio"].sort_values().plot(kind="barh", ax=axes[0], color="#1f77b4")
axes[0].set_title("DRMV Input Comparison: Sharpe")

drmv_tradeoff["average_turnover"].sort_values().plot(kind="barh", ax=axes[1], color="#ff7f0e")
axes[1].set_title("DRMV Input Comparison: Turnover")

drmv_tradeoff["forecast_realized_risk_gap"].sort_values().plot(kind="barh", ax=axes[2], color="#2ca02c")
axes[2].set_title("DRMV Input Comparison: Risk Gap")
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Section 6 - Sensitivity analysis

A robustness story is stronger when it measures weight sensitivity directly. The next block samples several rebalance dates and perturbs the training data in multiple ways, so the comparison is not hanging on one convenient snapshot.
"""
    ),
    code_cell(
        """candidate_dates = rebalance_results.index.unique().sort_values()
sampled_dates = list(candidate_dates[np.linspace(0, len(candidate_dates) - 1, CONFIG.sensitivity_rebalance_samples, dtype=int)])

perturbation_functions = {
    "gaussian_noise": lambda frame: corruption.inject_additive_gaussian_noise(frame, noise_scale=0.35, seed=SEED),
    "outlier_shock": lambda frame: corruption.inject_outlier_shocks(frame, shock_scale=4.0, shock_probability=0.015, seed=SEED),
    "block_missingness": lambda frame: corruption.apply_missing_data_method(
        corruption.inject_block_missingness(frame, missing_fraction=0.08, block_size=7, seed=SEED),
        method="ffill_then_zero",
    ),
    "vol_scaled_noise": lambda frame: corruption.inject_volatility_scaled_noise(frame, noise_scale=0.30, seed=SEED),
}

sensitivity_table = backtest.run_sensitivity_scenarios(
    simple_returns=simple_returns,
    strategies={
        "sample_min_var": sample_min_var_strategy,
        "shrinkage_min_var": shrinkage_min_var_strategy,
        "sample_mean_variance": sample_mean_variance_strategy,
        "wasserstein_proxy_min_var": wasserstein_proxy_strategy,
        "drmv_regularized_min_var": drmv_regularized_strategy,
        "drmv_regime_conditioned_min_var": drmv_regime_conditioned_strategy,
    },
    config=CONFIG,
    rebalance_dates=sampled_dates,
    perturbations=perturbation_functions,
)

sensitivity_summary = (
    sensitivity_table.groupby("strategy")[["weight_l1_change", "concentration_change", "top_3_share_change", "forecast_vol_change"]]
    .agg(["mean", "max"])
)
paper_aligned_sensitivity = (
    sensitivity_table[sensitivity_table["strategy"].isin(["sample_mean_variance", "drmv_regularized_min_var", "drmv_regime_conditioned_min_var"])]
    .groupby("strategy")[["weight_l1_change", "forecast_vol_change", "concentration_change"]]
    .mean()
    .sort_values("weight_l1_change")
)
clear_result = pd.DataFrame(
    {
        "question": ["DRMV more stable than sample mean-variance under perturbations?"],
        "sample_mean_variance_avg_l1": [paper_aligned_sensitivity.loc["sample_mean_variance", "weight_l1_change"]],
        "drmv_regularized_avg_l1": [paper_aligned_sensitivity.loc["drmv_regularized_min_var", "weight_l1_change"]],
        "drmv_regime_avg_l1": [paper_aligned_sensitivity.loc["drmv_regime_conditioned_min_var", "weight_l1_change"]],
        "drmv_beats_sample_mv": [
            bool(paper_aligned_sensitivity.loc["drmv_regularized_min_var", "weight_l1_change"] < paper_aligned_sensitivity.loc["sample_mean_variance", "weight_l1_change"])
        ],
    }
)
display(sensitivity_table.sort_values(["rebalance_date", "perturbation", "weight_l1_change"], ascending=[True, True, False]).head(20))
display(sensitivity_summary)
display(paper_aligned_sensitivity)
display(clear_result)
"""
    ),
    markdown_cell(
        """## Section 7 - Corruption-aware noisy-data experiment

The noisy-data section now uses multiple corruption mechanisms rather than one generic Gaussian shock. It also compares different missing-data handling policies and reports degradation relative to the clean baseline. This is still a **stress-testing extension inspired by the noisy-data paper**, not an implementation of the paper's convolution-based ambiguity model.
"""
    ),
    code_cell(
        """corruption_scenarios = {
    "clean": simple_returns,
    "gaussian_zero_fill": corruption.apply_missing_data_method(
        corruption.inject_additive_gaussian_noise(simple_returns, noise_scale=0.30, seed=SEED), "zero_fill"
    ),
    "block_missing_ffill": corruption.apply_missing_data_method(
        corruption.inject_block_missingness(simple_returns, missing_fraction=0.06, block_size=8, seed=SEED), "ffill_then_zero"
    ),
    "outlier_zero_fill": corruption.apply_missing_data_method(
        corruption.inject_outlier_shocks(simple_returns, shock_scale=4.0, shock_probability=0.015, seed=SEED), "zero_fill"
    ),
    "stale_price_ffill": corruption.apply_missing_data_method(
        corruption.inject_stale_price_returns(simple_returns, stale_probability=0.04, max_stale_days=3, seed=SEED), "ffill_then_zero"
    ),
    "vol_scaled_zero_fill": corruption.apply_missing_data_method(
        corruption.inject_volatility_scaled_noise(simple_returns, noise_scale=0.35, seed=SEED), "zero_fill"
    ),
    "block_missing_drop_sparse": corruption.apply_missing_data_method(
        corruption.inject_block_missingness(simple_returns, missing_fraction=0.06, block_size=8, seed=SEED), "drop_sparse_assets"
    ),
}

noise_summary = backtest.run_corruption_stress(
    simple_returns=simple_returns,
    strategies={
        "sample_min_var": sample_min_var_strategy,
        "sample_mean_variance": sample_mean_variance_strategy,
        "wasserstein_proxy_min_var": wasserstein_proxy_strategy,
        "drmv_regularized_min_var": drmv_regularized_strategy,
        "drmv_regime_conditioned_min_var": drmv_regime_conditioned_strategy,
    },
    corruption_scenarios=corruption_scenarios,
    config=CONFIG,
)

clean_reference = noise_summary[noise_summary["corruption"] == "clean"].set_index("strategy")
noise_summary["sharpe_drop_vs_clean"] = noise_summary.apply(lambda row: clean_reference.loc[row["strategy"], "sharpe_ratio"] - row["sharpe_ratio"], axis=1)
noise_summary["turnover_increase_vs_clean"] = noise_summary.apply(lambda row: row["average_turnover"] - clean_reference.loc[row["strategy"], "average_turnover"], axis=1)
noise_summary["risk_gap_increase_vs_clean"] = noise_summary.apply(lambda row: row["forecast_realized_risk_gap"] - clean_reference.loc[row["strategy"], "forecast_realized_risk_gap"], axis=1)
noise_summary["gross_to_net_drag_change"] = noise_summary.apply(lambda row: row["annualized_return_cost_drag"] - clean_reference.loc[row["strategy"], "annualized_return_cost_drag"], axis=1)

display(noise_summary[["corruption", "strategy", "sharpe_ratio", "sharpe_drop_vs_clean", "average_turnover", "turnover_increase_vs_clean", "forecast_realized_risk_gap", "risk_gap_increase_vs_clean"]])
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
for strategy, frame in noise_summary.groupby("strategy"):
    if strategy == "clean":
        continue
    frame = frame.sort_values("corruption")
    axes[0].bar(frame["corruption"] + "\\n" + strategy, frame["sharpe_drop_vs_clean"], alpha=0.8)
    axes[1].bar(frame["corruption"] + "\\n" + strategy, frame["turnover_increase_vs_clean"], alpha=0.8)
    axes[2].bar(frame["corruption"] + "\\n" + strategy, frame["risk_gap_increase_vs_clean"], alpha=0.8)

axes[0].set_title("Sharpe Drop vs Clean")
axes[1].set_title("Turnover Increase vs Clean")
axes[2].set_title("Risk-Gap Increase vs Clean")
for ax in axes:
    ax.tick_params(axis="x", rotation=90)
plt.show()
"""
    ),
    markdown_cell(
        """## Section 8 - Monitoring as a governance prototype

This is intentionally not an alpha model. The point is to create a monitoring workflow that ranks the probability of future instability events. Targets are separated into turnover, risk-gap, and drawdown instability so the notebook does not collapse heterogeneous behaviors into one noisy label. The current results should still be read as **governance scaffolding** rather than strong predictive evidence.
"""
    ),
    code_cell(
        """proxy_rebalances = rebalance_results[rebalance_results["strategy"] == "wasserstein_proxy_min_var"].copy()
proxy_weights = weights_history["wasserstein_proxy_min_var"].copy()

feature_frame = features.build_instability_feature_frame(
    simple_returns=simple_returns,
    weights_history=proxy_weights,
    rebalance_results=proxy_rebalances,
    lookback=CONFIG.monitoring_lookback,
    include_optional_features=True,
)
target_frame = features.build_instability_targets(proxy_rebalances)

monitoring_feature_columns = [
    "trailing_vol",
    "average_pairwise_correlation",
    "effective_rank",
    "recent_turnover",
    "herfindahl",
    "top_3_weight_share",
    "recent_drawdown_abs",
    "forecast_realized_risk_gap",
]

monitoring_targets = {
    "turnover": "turnover_instability_target",
    "risk_gap": "risk_gap_instability_target",
    "drawdown": "drawdown_instability_target",
}

monitoring_results = {}
monitoring_metric_rows = []
for label, target_col in monitoring_targets.items():
    dataset = feature_frame[monitoring_feature_columns].join(target_frame[target_col], how="inner").dropna()
    if dataset[target_col].nunique() < 2:
        continue
    result = monitoring.train_instability_detector(
        feature_frame=dataset.drop(columns=target_col),
        target=dataset[target_col],
        model_type="logistic",
    )
    monitoring_results[label] = result
    row = {"target": label}
    row.update(result["metrics"])
    monitoring_metric_rows.append(row)

monitoring_metrics = pd.DataFrame(monitoring_metric_rows)
target_positive_rates = target_frame[
    ["turnover_instability_target", "risk_gap_instability_target", "drawdown_instability_target", "combined_instability_target"]
].mean().rename("positive_rate").to_frame()
display(target_positive_rates)
display(monitoring_metrics)

if "risk_gap" in monitoring_results:
    display(monitoring_results["risk_gap"]["feature_importance"])
    display(monitoring_results["risk_gap"]["calibration_table"])
"""
    ),
    code_cell(
        """latest_features = feature_frame.iloc[-1]
latest_monitoring_features = feature_frame[monitoring_feature_columns].iloc[-1]
if "risk_gap" in monitoring_results:
    latest_probability = float(
        monitoring_results["risk_gap"]["model"].predict_proba(latest_monitoring_features.to_frame().T)[:, 1][0]
    )
else:
    latest_probability = np.nan

monitoring_report = monitoring.build_monitoring_report(latest_features, latest_probability)
monitoring_report
"""
    ),
    markdown_cell(
        """## Section 8.5 - Heuristic regime tagging extension

This is the one targeted ML extension in the notebook. It does **not** forecast returns or produce weights directly. Instead, it labels heuristic market states from trailing diagnostics, trains a simple classifier on those same feature families, and then asks whether proxy diagnostics like slack, turnover, and epsilon selection look different across predicted regimes. High classification accuracy here should be interpreted as **recovering a hand-crafted tagging rule**, not as evidence of a powerful predictive model.
"""
    ),
    code_cell(
        """regime_labels = regime.build_regime_labels(
    feature_frame,
    vol_quantile=CONFIG.regime_label_quantile,
    corr_quantile=CONFIG.regime_label_quantile,
    drawdown_quantile=CONFIG.regime_label_quantile,
    dispersion_quantile=CONFIG.regime_label_quantile,
)

regime_feature_frame = feature_frame[regime.DEFAULT_REGIME_FEATURE_COLUMNS].dropna()
regime_result = regime.train_regime_classifier(
    feature_frame=regime_feature_frame,
    target=regime_labels,
    model_type=CONFIG.regime_model_type,
    test_fraction=CONFIG.regime_test_fraction,
    random_state=CONFIG.seed,
)
regime_scope = pd.DataFrame(
    {
        "task_scope": [regime_result["task_scope"]],
        "interpretation": ["heuristic label recovery, not forward prediction"],
    }
)
regime_metrics = pd.DataFrame([regime_result["metrics"]])
regime_conditional_summary = regime.summarize_regime_conditionals(
    rebalance_results=rebalance_results,
    regime_predictions=regime_result["predictions"],
    strategy="wasserstein_proxy_min_var",
    regime_column="predicted_regime",
)

display(regime_scope)
display(regime_metrics)
display(regime_result["confusion_matrix"])
display(regime_result["feature_importance"])
display(regime_conditional_summary)
"""
    ),
    markdown_cell(
        """## Section 9 - Stress testing

The historical stress summary now uses a **window-aware summarizer**, so it no longer mixes window-specific performance with full-sample turnover and governance metrics. This is a correctness fix, not just a presentation tweak.
"""
    ),
    code_cell(
        """stress_windows = {
    "covid_selloff": ("2020-02-19", "2020-03-23"),
    "rates_shock_2022": ("2022-01-03", "2022-10-14"),
    "growth_rally_2023": ("2023-01-03", "2023-07-31"),
}

stress_rows = []
for window_name, (start, end) in stress_windows.items():
    window_summary = backtest.summarize_backtest_window(
        daily_returns_window=daily_returns,
        gross_daily_returns_window=gross_daily_returns,
        weights_history=weights_history,
        rebalance_results=rebalance_results,
        start_date=start,
        end_date=end,
    ).reset_index()
    window_summary["stress_window"] = window_name
    stress_rows.append(window_summary)

stress_summary = pd.concat(stress_rows, ignore_index=True)
display(stress_summary[["stress_window", "strategy", "annualized_return", "sharpe_ratio", "average_turnover", "forecast_realized_risk_gap", "positive_slack_fraction"]])
"""
    ),
    code_cell(
        """model_stress_config = CONFIG.to_dict()
model_stress_config["transaction_cost_bps"] = 15.0
model_stress_config["slack_penalty"] = CONFIG.slack_penalty * 1.25

model_stress_artifacts = backtest.run_rolling_backtest(
    simple_returns=simple_returns,
    strategies={
        "sample_min_var": sample_min_var_strategy,
        "wasserstein_proxy_min_var": wasserstein_proxy_strategy,
        "drmv_regularized_min_var": drmv_regularized_strategy,
    },
    config=model_stress_config,
)
model_stress_summary = backtest.summarize_backtest(
    daily_returns=model_stress_artifacts["daily_returns"],
    gross_daily_returns=model_stress_artifacts["gross_daily_returns"],
    weights_history=model_stress_artifacts["weights_history"],
    rebalance_results=model_stress_artifacts["rebalance_results"],
)
model_stress_summary
"""
    ),
    markdown_cell(
        """## Section 10 - Log-return growth appendix

This optional appendix revisits the Kelly paper in a deliberately modest way. The model below is built on **asset log-return samples**, not simple returns, and applies the robustness penalty in that same log-return space. That makes it better aligned with the paper's modeling lesson, while still remaining a tractable proxy rather than the paper's exact convex Wasserstein-Kelly reformulation.
"""
    ),
    code_cell(
        """appendix_epsilons = [0.0, 0.001, 0.005, 0.01]
appendix_rows = []
appendix_weights = {}

for epsilon in appendix_epsilons:
    result = robust.solve_log_return_growth_proxy(
        log_returns=log_returns,
        epsilon=epsilon,
        bounds=CONFIG.bounds,
        growth_risk_aversion=2.0,
    )
    appendix_rows.append(
        {
            "epsilon": epsilon,
            "expected_log_return": result["expected_log_return"],
            "worst_case_log_return": result["worst_case_log_return"],
            "log_growth_vol": result["log_growth_vol"],
            "effective_n": float(1.0 / np.square(result["weights"]).sum()),
            "status": result["status"],
        }
    )
    appendix_weights[f"eps_{epsilon:.3f}"] = result["weights"]

appendix_summary = pd.DataFrame(appendix_rows)
appendix_weight_frame = pd.DataFrame(appendix_weights).fillna(0.0)

display(appendix_summary)
display(appendix_weight_frame)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
appendix_summary.plot(x="epsilon", y=["expected_log_return", "worst_case_log_return"], marker="o", ax=axes[0])
axes[0].set_title("Log-Return Proxy vs Epsilon")
axes[0].set_ylabel("Mean log-return proxy")

appendix_summary.plot(x="epsilon", y="effective_n", marker="o", ax=axes[1], color="#2ca02c")
axes[1].set_title("Diversification vs Ambiguity Radius")
axes[1].set_ylabel("Effective number of holdings")

appendix_weight_frame.plot(kind="bar", ax=axes[2])
axes[2].set_title("Log-Return Proxy Weights")
axes[2].set_ylabel("Weight")
axes[2].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    markdown_cell(
        """## Section 11 - Larger-universe extension

The base notebook uses 12 cross-asset ETFs because they keep the logic readable. This section adds a larger 30-instrument liquid ETF panel as a scale check, not as a new flagship claim. The point is to show that the workflow and diagnostics still run on a broader cross section with manageable compute.
"""
    ),
    code_cell(
        """import time

large_universe_summary = pd.DataFrame()
large_universe_runtime_seconds = np.nan

try:
    large_universe = data.LARGE_UNIVERSE[: CONFIG.large_universe_size]
    large_prices_raw = data.load_or_download_price_data(
        tickers=large_universe,
        start=CONFIG.start_date,
        end=CONFIG.end_date,
        raw_data_path=PROJECT_ROOT / CONFIG.large_universe_raw_data_path,
        auto_adjust=True,
        progress=False,
        refresh=CONFIG.refresh_data,
    )
    large_prices = data.clean_price_panel(large_prices_raw, max_missing_frac=0.08, forward_fill_limit=3)
    large_bundle = data.compute_returns(large_prices)

    large_strategy_set = {
        "sample_min_var": sample_min_var_strategy,
        "shrinkage_min_var": shrinkage_min_var_strategy,
        "wasserstein_proxy_min_var": wasserstein_proxy_strategy,
        "drmv_regularized_min_var": drmv_regularized_strategy,
    }

    start_clock = time.perf_counter()
    large_artifacts = backtest.run_rolling_backtest(
        simple_returns=large_bundle.simple_returns,
        strategies=large_strategy_set,
        config=CONFIG,
    )
    large_universe_runtime_seconds = time.perf_counter() - start_clock
    large_universe_summary = backtest.summarize_backtest(
        daily_returns=large_artifacts["daily_returns"],
        gross_daily_returns=large_artifacts["gross_daily_returns"],
        weights_history=large_artifacts["weights_history"],
        rebalance_results=large_artifacts["rebalance_results"],
    )
    large_universe_summary["runtime_seconds"] = large_universe_runtime_seconds
    large_universe_summary["n_assets"] = large_bundle.simple_returns.shape[1]
except Exception as exc:
    large_universe_summary = pd.DataFrame({"status": [str(exc)]}, index=["large_universe_run"])

display(large_universe_summary)
"""
    ),
    markdown_cell(
        """## Section 12 - Validation and test harness

The project now includes both notebook-friendly checks and a real `pytest` suite. The validation block below decomposes the zero-radius comparison into a hard-constraint apples-to-apples case plus the soft-slack and turnover-enhanced variants used in the main workflow, so any mismatch is explained rather than hidden. It also keeps the direct noisy-input regression check visible: if the proxy still fails that test, the notebook should be read as a measurement and governance framework rather than proof of superior robustness.
"""
    ),
    code_cell(
        """input_checks = validation.build_input_check_table(weights_history)
numerical_checks = validation.build_numerical_check_table(rebalance_results)

proxy_rebalance_dates = weights_history["wasserstein_proxy_min_var"].index
diagnostic_date = proxy_rebalance_dates[-1]
diagnostic_loc = simple_returns.index.get_indexer([diagnostic_date], method="nearest")[0]
diagnostic_train_end = diagnostic_loc - CONFIG.val_window
diagnostic_train_start = diagnostic_train_end - CONFIG.train_window
diagnostic_train = simple_returns.iloc[diagnostic_train_start:diagnostic_train_end].copy()

diagnostic_previous_weights = None
if len(proxy_rebalance_dates) > 1:
    diagnostic_previous_weights = weights_history["wasserstein_proxy_min_var"].loc[:diagnostic_date].iloc[-2]

target_return, target_source = robust.compute_dynamic_target_return(
    diagnostic_train,
    mode=CONFIG.target_return_mode,
    scale=CONFIG.target_return_scale,
    quantile=CONFIG.target_return_quantile,
    fixed_target_return=CONFIG.fixed_target_return,
)

empirical_snapshot = baselines.fit_sample_min_variance(
    train_returns=diagnostic_train,
    target_return=target_return,
    bounds=CONFIG.bounds,
)
zero_radius_diagnostic = validation.diagnose_zero_radius_proxy_alignment(
    train_returns=diagnostic_train,
    target_return=target_return,
    bounds=CONFIG.bounds,
    covariance_method="sample",
    previous_weights=diagnostic_previous_weights,
    turnover_penalty=CONFIG.turnover_penalty,
    slack_penalty=CONFIG.slack_penalty,
)
proxy_zero_hard = robust.solve_wasserstein_proxy_min_var(
    train_returns=diagnostic_train,
    epsilon=0.0,
    target_return=target_return,
    covariance_method="sample",
    bounds=CONFIG.bounds,
    slack_penalty=CONFIG.slack_penalty,
    allow_slack=False,
)
proxy_large = robust.solve_wasserstein_proxy_min_var(
    train_returns=diagnostic_train,
    epsilon=CONFIG.wasserstein_proxy_radius_grid[-1],
    target_return=target_return,
    bounds=CONFIG.bounds,
    slack_penalty=CONFIG.slack_penalty,
)

regression_tests = validation.run_regression_tests(
    equal_weight_weights=baselines.equal_weight(diagnostic_train.columns),
    empirical_weights=empirical_snapshot["weights"],
    robust_zero_radius_weights=proxy_zero_hard["weights"],
    robust_large_radius_weights=proxy_large["weights"],
    noise_summary=noise_summary.assign(
        noise_level=noise_summary["corruption"].map(
            {
                "clean": 0.0,
                "gaussian_zero_fill": 1.0,
                "block_missing_ffill": 2.0,
                "outlier_zero_fill": 3.0,
                "stale_price_ffill": 4.0,
                "vol_scaled_zero_fill": 5.0,
            }
        )
    ),
)

display(input_checks)
display(numerical_checks)
display(zero_radius_diagnostic)
display(regression_tests)

noise_claim_row = regression_tests.loc[regression_tests["test"] == "robust_model_degrades_less_under_noise"]
if not noise_claim_row.empty and not bool(noise_claim_row["passed"].iloc[0]):
    print(
        "Current validation note: the automated noisy-input regression check is still negative, "
        "so the notebook does not claim that the proxy is empirically more robust under corruption in this specification."
    )
"""
    ),
    markdown_cell(
        """## Section 13 - Conclusions

The revised project supports a stronger and more honest story than the original draft:

1. The allocator is implemented as a **tractable Wasserstein proxy**, not oversold as a full DRO system.
2. Soft feasibility makes the optimization workflow diagnosable: target tension shows up as **slack**, not silent failure.
3. Historical stress summaries are now methodologically correct because governance metrics are filtered to the same window as returns.
4. The new DRMV branch is closer to the regularized DR mean-variance literature and gives us a cleaner ambiguity parameter (`delta`) plus a conservative robust target (`alpha_bar`) to track in validation.
5. Regime-conditioned DRMV inputs create a practical bridge from market-state inference to portfolio construction, and the notebook now compares regime-conditioned mean, covariance, and both together rather than treating the regime branch as a black box.
6. Corruption-aware experiments and direct sensitivity analysis support a stability-measurement narrative better than raw Sharpe alone, with the clearest paper-aligned result coming from perturbation stability versus fragile sample mean-variance, but the automated noisy-input regression check should still be treated as an empirical test rather than a foregone conclusion.
7. The optional Kelly-style appendix moves the uncertainty model into **log-return space**, but it is still a tractable proxy rather than the paper's exact Wasserstein-Kelly reformulation.
8. The monitoring layer is best understood as a **prototype governance workflow** with calibration and alert diagnostics, not a predictive alpha claim.
9. The heuristic regime-tagging extension is intentionally narrow: it helps condition diagnostics on market state, but its high accuracy mainly reflects recovery of a hand-crafted labeling rule, not a standalone predictive edge.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (robust-portfolio-analytics)",
            "language": "python",
            "name": "robust-portfolio-analytics",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

