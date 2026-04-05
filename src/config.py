from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BacktestConfig:
    seed: int = 7
    start_date: str = "2012-01-01"
    end_date: str = "2025-12-31"
    train_window: int = 252 * 3
    val_window: int = 252
    rebalance_freq: int = 21
    transaction_cost_bps: float = 5.0
    target_universe_size: int = 12
    fixed_target_return: float = 0.0002
    target_return_mode: str = "equal_weight_fraction"
    target_return_scale: float = 0.50
    target_return_quantile: float = 0.40
    bounds: tuple[float, float] = (0.0, 0.25)
    turnover_penalty: float = 0.001
    no_trade_band_l1: float = 0.05
    full_rebalance_band_l1: float = 0.20
    covariance_method: str = "ledoit_wolf"
    regime_lookback: int = 252
    regime_states: int = 2
    regime_engine: str = "hmm"
    regime_covariance_method: str = "state_aware"
    regime_calm_covariance_method: str = "ledoit_wolf"
    regime_stressed_covariance_method: str = "ewma"
    regime_probability_temperature: float = 3.0
    regime_probability_threshold: float = 0.70
    regime_current_probability_mode: str = "filtered"
    regime_estimation_probability_mode: str = "smoothed"
    regime_switching_variance: bool = True
    monitoring_lookback: int = 63
    scenario_parallel_jobs: int = -1
    scenario_parallel_backend: str = "loky"
    regime_model_type: str = "random_forest"
    regime_test_fraction: float = 0.30
    regime_label_quantile: float = 0.67
    slack_penalty: float = 10.0
    robust_validation_metric: str = "composite"
    selection_slack_penalty_weight: float = 5.0
    selection_turnover_penalty_weight: float = 1.0
    selection_risk_gap_penalty_weight: float = 2.0
    selection_epsilon_change_penalty_weight: float = 5.0
    selection_constraint_penalty_weight: float = 1.0
    selection_fallback_penalty_weight: float = 10.0
    selection_sensitivity_penalty_weight: float = 1.0
    selection_corruption_penalty_weight: float = 1.25
    selection_stress_penalty_weight: float = 3.0
    selection_concentration_penalty_weight: float = 0.25
    selection_drawdown_penalty_weight: float = 1.0
    selection_mean_perturbation_scale: float = 0.25
    selection_covariance_perturbation_scale: float = 0.20
    selection_corruption_noise_scale: float = 0.15
    selection_stress_quantile: float = 0.25
    selection_sensitivity_top_k: int = 12
    raw_data_path: str = "data/raw_prices.parquet"
    large_universe_raw_data_path: str = "data/large_universe_raw_prices.parquet"
    refresh_data: bool = False
    large_universe_size: int = 30
    sensitivity_rebalance_samples: int = 5
    drmv_target_method: str = "benchmark_fraction"
    drmv_target_scale: float = 0.50
    drmv_alpha_bar_rule: str = "delta_adjusted"
    drmv_p_norm: int = 2
    drmv_calibration_mode: str = "practical"
    drmv_objective_mode: str = "production"
    drmv_alpha_bar_scale_grid: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 1.0])
    drmv_paper_reference_alpha_bar_scale_grid: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    drmv_delta_grid: list[float] = field(
        default_factory=lambda: [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2]
    )
    drmv_covariance_methods: list[str] = field(
        default_factory=lambda: ["ledoit_wolf", "oas", "ewma"]
    )
    regime_stressed_target_scale: float = 0.40
    regime_stressed_delta_scale: float = 4.0
    regime_stressed_turnover_multiplier: float = 6.0
    regime_stressed_delta_grid_multiplier: float = 4.0
    wasserstein_proxy_radius_grid: list[float] = field(
        default_factory=lambda: [0.0, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def raw_data_path_obj(self) -> Path:
        return Path(self.raw_data_path)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BacktestConfig":
        normalized = dict(payload)
        if "bounds" in normalized:
            normalized["bounds"] = tuple(normalized["bounds"])
        if "wasserstein_radius_grid" in normalized and "wasserstein_proxy_radius_grid" not in normalized:
            normalized["wasserstein_proxy_radius_grid"] = normalized.pop("wasserstein_radius_grid")
        return cls(**normalized)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BacktestConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        return cls.from_dict(payload)
