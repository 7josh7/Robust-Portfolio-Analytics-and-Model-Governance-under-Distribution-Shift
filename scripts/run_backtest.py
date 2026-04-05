from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import backtest, baselines, data, features, regime, reporting, robust, selection  # noqa: E402
from src.config import BacktestConfig  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=".*You specified your problem should be solved by ECOS.*",
    category=FutureWarning,
)


def equal_weight_strategy(train_returns, val_returns, previous_weights, config):
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
        selection_concentration_penalty_weight=config["selection_concentration_penalty_weight"],
        selection_drawdown_penalty_weight=config["selection_drawdown_penalty_weight"],
        mean_perturbation_scale=config["selection_mean_perturbation_scale"],
        covariance_perturbation_scale=config["selection_covariance_perturbation_scale"],
        corruption_noise_scale=config["selection_corruption_noise_scale"],
        stress_quantile=config["selection_stress_quantile"],
        selection_sensitivity_top_k=config["selection_sensitivity_top_k"],
        metric=config["robust_validation_metric"],
        calibration_mode=config["drmv_calibration_mode"],
        objective_mode=config["drmv_objective_mode"],
        rebalance_date=config.get("rebalance_date"),
    )


def drmv_paper_reference_strategy(train_returns, val_returns, previous_weights, config):
    return selection.tune_drmv_regularized_min_variance(
        train_returns=train_returns,
        val_returns=val_returns,
        delta_grid=config["drmv_delta_grid"],
        alpha_bar_scale_grid=config["drmv_paper_reference_alpha_bar_scale_grid"],
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
        selection_concentration_penalty_weight=config["selection_concentration_penalty_weight"],
        selection_drawdown_penalty_weight=config["selection_drawdown_penalty_weight"],
        mean_perturbation_scale=config["selection_mean_perturbation_scale"],
        covariance_perturbation_scale=config["selection_covariance_perturbation_scale"],
        corruption_noise_scale=config["selection_corruption_noise_scale"],
        stress_quantile=config["selection_stress_quantile"],
        selection_sensitivity_top_k=config["selection_sensitivity_top_k"],
        metric=config["robust_validation_metric"],
        calibration_mode="paper_reference",
        objective_mode="paper_alignment",
        rebalance_date=config.get("rebalance_date"),
    )


def _build_regime_conditioned_drmv_result(
    train_returns,
    val_returns,
    previous_weights,
    config,
    input_mode="both",
    regime_engine="mixture",
    calibration_mode="practical",
    objective_mode="production",
):
    regime_inputs = selection.prepare_regime_conditioned_inputs(
        train_returns=train_returns,
        lookback=config["regime_lookback"],
        n_regimes=config["regime_states"],
        regime_engine=regime_engine,
        covariance_method=config["regime_covariance_method"],
        calm_covariance_method=config["regime_calm_covariance_method"],
        stressed_covariance_method=config["regime_stressed_covariance_method"],
        probability_temperature=config["regime_probability_temperature"],
        stressed_probability_threshold=config["regime_probability_threshold"],
        switching_variance=config["regime_switching_variance"],
        current_probability_mode=config["regime_current_probability_mode"],
        estimation_probability_mode=config["regime_estimation_probability_mode"],
        random_state=config["seed"],
    )
    if calibration_mode == "practical":
        overrides = selection.build_regime_search_overrides(
            delta_grid=config["drmv_delta_grid"],
            turnover_penalty=config["turnover_penalty"],
            stress_activation=regime_inputs["stress_activation"],
            stressed_delta_grid_multiplier=config["regime_stressed_delta_grid_multiplier"],
            stressed_turnover_multiplier=config["regime_stressed_turnover_multiplier"],
        )
    else:
        overrides = {
            "delta_grid": config["drmv_delta_grid"],
            "turnover_penalty": config["turnover_penalty"],
        }
    mean_vector = regime_inputs["mean_returns"] if input_mode in {"mean", "both"} else None
    covariance_matrix = regime_inputs["covariance"] if input_mode in {"covariance", "both"} else None
    result = selection.tune_drmv_regularized_min_variance(
        train_returns=train_returns,
        val_returns=val_returns,
        delta_grid=overrides["delta_grid"],
        alpha_bar_scale_grid=(
            config["drmv_paper_reference_alpha_bar_scale_grid"]
            if calibration_mode == "paper_reference"
            else config["drmv_alpha_bar_scale_grid"]
        ),
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
        selection_concentration_penalty_weight=config["selection_concentration_penalty_weight"],
        selection_drawdown_penalty_weight=config["selection_drawdown_penalty_weight"],
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
        calibration_mode=calibration_mode,
        objective_mode=objective_mode,
        rebalance_date=config.get("rebalance_date"),
    )
    result.update(
        {
            "regime_model_version": regime_inputs.get("regime_model_version", regime_engine),
            "regime_model_status": regime_inputs.get("regime_model_status", "trained"),
            "probability_mode": regime_inputs.get("probability_mode", regime_engine),
        }
    )
    return result


def drmv_regime_conditioned_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="both",
        regime_engine="mixture",
        calibration_mode="practical",
        objective_mode="production",
    )


def drmv_regime_mean_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="mean",
        regime_engine="mixture",
        calibration_mode="practical",
        objective_mode="production",
    )


def drmv_regime_covariance_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="covariance",
        regime_engine="mixture",
        calibration_mode="practical",
        objective_mode="production",
    )


def drmv_regime_covariance_hmm_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="covariance",
        regime_engine="hmm",
        calibration_mode="practical",
        objective_mode="paper_alignment",
    )


def drmv_regime_conditioned_hmm_strategy(train_returns, val_returns, previous_weights, config):
    return _build_regime_conditioned_drmv_result(
        train_returns=train_returns,
        val_returns=val_returns,
        previous_weights=previous_weights,
        config=config,
        input_mode="both",
        regime_engine="hmm",
        calibration_mode="practical",
        objective_mode="paper_alignment",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robust portfolio analytics backtest.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    config = BacktestConfig.from_yaml(PROJECT_ROOT / args.config)
    universe = data.DEFAULT_UNIVERSE[: config.target_universe_size]

    prices_raw = data.load_or_download_price_data(
        tickers=universe,
        start=config.start_date,
        end=config.end_date,
        raw_data_path=PROJECT_ROOT / config.raw_data_path,
        auto_adjust=True,
        progress=False,
        refresh=config.refresh_data,
    )
    prices = data.clean_price_panel(prices_raw, max_missing_frac=0.05, forward_fill_limit=3)
    bundle = data.compute_returns(prices)

    strategies = {
        "equal_weight": equal_weight_strategy,
        "inverse_vol": inverse_vol_strategy,
        "sample_min_var": sample_min_var_strategy,
        "shrinkage_min_var": shrinkage_min_var_strategy,
        "sample_mean_variance": sample_mean_variance_strategy,
        "wasserstein_proxy_min_var": wasserstein_proxy_strategy,
        "drmv_regularized_min_var": drmv_regularized_strategy,
        "drmv_paper_reference_min_var": drmv_paper_reference_strategy,
        "drmv_regime_conditioned_min_var": drmv_regime_conditioned_strategy,
        "drmv_regime_covariance_min_var_mixture": drmv_regime_covariance_strategy,
        "drmv_regime_covariance_min_var_hmm": drmv_regime_covariance_hmm_strategy,
        "drmv_regime_conditioned_min_var_hmm": drmv_regime_conditioned_hmm_strategy,
    }
    artifacts = backtest.run_rolling_backtest(bundle.simple_returns, strategies, config)
    summary = backtest.summarize_backtest(
        daily_returns=artifacts["daily_returns"],
        gross_daily_returns=artifacts["gross_daily_returns"],
        weights_history=artifacts["weights_history"],
        rebalance_results=artifacts["rebalance_results"],
    )

    output_dir = PROJECT_ROOT / "outputs" / "cli_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv")
    artifacts["rebalance_results"].to_csv(output_dir / "rebalance_results.csv")
    artifacts["daily_returns"].to_csv(output_dir / "daily_returns_net.csv")
    artifacts["gross_daily_returns"].to_csv(output_dir / "daily_returns_gross.csv")
    reporting.save_weights_history(artifacts["weights_history"], output_dir)
    reporting.save_weights_history(artifacts["proposed_weights_history"], output_dir / "proposed_weights")
    reporting.save_diagnostics_json(
        reporting.build_diagnostics_payload(config, summary, artifacts["rebalance_results"]),
        output_dir / "diagnostics.json",
    )
    reporting.save_backtest_figures(
        daily_returns=artifacts["daily_returns"],
        rebalance_results=artifacts["rebalance_results"],
        weights_history=artifacts["weights_history"],
        output_dir=output_dir / "figures",
    )

    proxy_rebalances = artifacts["rebalance_results"][artifacts["rebalance_results"]["strategy"] == "wasserstein_proxy_min_var"].copy()
    proxy_weights = artifacts["weights_history"]["wasserstein_proxy_min_var"].copy()
    regime_features = features.build_instability_feature_frame(
        simple_returns=bundle.simple_returns,
        weights_history=proxy_weights,
        rebalance_results=proxy_rebalances,
        lookback=config.monitoring_lookback,
        include_optional_features=True,
    )
    regime_labels = regime.build_regime_labels(
        regime_features,
        vol_quantile=config.regime_label_quantile,
        corr_quantile=config.regime_label_quantile,
        drawdown_quantile=config.regime_label_quantile,
        dispersion_quantile=config.regime_label_quantile,
    )
    regime_result = regime.train_regime_classifier(
        feature_frame=regime_features[regime.DEFAULT_REGIME_FEATURE_COLUMNS].dropna(),
        target=regime_labels,
        model_type=config.regime_model_type,
        test_fraction=config.regime_test_fraction,
        random_state=config.seed,
    )
    regime_output_dir = output_dir / "regime"
    regime_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([regime_result["metrics"]]).to_csv(regime_output_dir / "metrics.csv", index=False)
    regime_result["confusion_matrix"].to_csv(regime_output_dir / "confusion_matrix.csv")
    regime_result["feature_importance"].to_csv(regime_output_dir / "feature_importance.csv", index=False)
    regime_result["predictions"].to_csv(regime_output_dir / "predictions.csv")
    regime.summarize_regime_conditionals(
        rebalance_results=artifacts["rebalance_results"],
        regime_predictions=regime_result["predictions"],
        strategy="wasserstein_proxy_min_var",
        regime_column="predicted_regime",
    ).to_csv(regime_output_dir / "conditional_summary.csv")

    print(summary.round(4))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
