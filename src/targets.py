from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import equal_weight, inverse_volatility_weight


def build_nominal_target(
    train_returns: pd.DataFrame,
    method: str = "benchmark_fraction",
    scale: float = 0.50,
    quantile: float = 0.40,
    benchmark_returns: pd.Series | None = None,
    fixed_target_return: float = 0.0002,
) -> tuple[float, str]:
    """Build a nominal target-return hurdle before ambiguity adjustment."""

    method = method.lower()
    if method == "fixed":
        return float(fixed_target_return), "fixed_target_return"

    if method == "benchmark_fraction":
        if benchmark_returns is None:
            benchmark_returns = train_returns.fillna(0.0) @ equal_weight(train_returns.columns)
        return float(scale * benchmark_returns.mean()), "benchmark_fraction"

    if method == "equal_weight_fraction":
        benchmark_returns = train_returns.fillna(0.0) @ equal_weight(train_returns.columns)
        return float(scale * benchmark_returns.mean()), "equal_weight_fraction"

    if method == "inverse_vol_fraction":
        benchmark_returns = train_returns.fillna(0.0) @ inverse_volatility_weight(train_returns)
        return float(scale * benchmark_returns.mean()), "inverse_vol_fraction"

    if method == "portfolio_quantile":
        benchmark_returns = train_returns.fillna(0.0) @ equal_weight(train_returns.columns)
        return float(benchmark_returns.quantile(quantile)), "portfolio_quantile"

    raise ValueError(f"Unknown nominal target method: {method}")


def build_alpha_bar(
    rho: float,
    delta: float,
    rule: str = "delta_adjusted",
    scale: float = 1.0,
    floor: float | None = None,
) -> tuple[float, str]:
    """
    Convert a nominal target rho into a conservative robust target alpha_bar.

    The default rule follows the paper-aligned intuition that alpha_bar should
    sit below the nominal target by an amount linked to ambiguity size.
    """

    rule = rule.lower()
    if rule == "delta_adjusted":
        alpha_bar = float(rho - scale * np.sqrt(max(delta, 0.0)))
    elif rule == "fraction_of_nominal":
        alpha_bar = float(scale * rho)
    elif rule == "none":
        alpha_bar = float(rho)
    else:
        raise ValueError(f"Unknown alpha_bar rule: {rule}")

    if floor is not None:
        alpha_bar = max(alpha_bar, float(floor))
    return alpha_bar, rule


def build_alpha_bar_paper_reference(
    rho: float,
    delta: float,
    phi_norm_proxy: float | None = None,
    c: float = 1.0,
    floor: float | None = None,
) -> tuple[float, str]:
    """
    Build a paper-reference robust return threshold.

    This keeps the Blanchet-Chen-Zhou intuition explicit: the robust target
    sits below the nominal hurdle by an amount linked to ambiguity size. The
    optional norm proxy allows the adjustment to reflect a benchmark portfolio
    scale rather than a pure constant.
    """

    norm_proxy = float(phi_norm_proxy) if phi_norm_proxy is not None else 1.0
    norm_proxy = max(norm_proxy, 1e-8)
    alpha_bar = float(rho - float(c) * np.sqrt(max(delta, 0.0)) * norm_proxy)
    if floor is not None:
        alpha_bar = max(alpha_bar, float(floor))
    return alpha_bar, "paper_reference"


def build_delta_grid_paper_reference(
    sample_size: int,
    base_grid: list[float] | None = None,
    scale: float = 1.0,
    multipliers: list[float] | None = None,
    minimum_delta: float = 1e-6,
    maximum_delta: float | None = None,
) -> list[float]:
    """
    Build a paper-reference ambiguity grid centered on the n^{-1/2} intuition.

    The paper does not prescribe a single fixed finite-sample grid, so this
    helper provides a compact research-friendly approximation: it mixes the
    project's practical grid with additional values whose order reflects the
    sample-size discussion in the DRMV paper.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    if multipliers is None:
        multipliers = [0.05, 0.10, 0.25, 0.50, 1.0]

    candidates = {0.0}
    if base_grid is not None:
        candidates.update(float(delta) for delta in base_grid if float(delta) >= 0.0)

    reference_radius = float(scale) / np.sqrt(float(sample_size))
    for multiplier in multipliers:
        candidate = max(float(minimum_delta), (float(multiplier) * reference_radius) ** 2)
        if maximum_delta is not None:
            candidate = min(candidate, float(maximum_delta))
        candidates.add(candidate)

    return sorted(candidates)


def build_regime_conditioned_target_params(
    rho: float,
    delta: float,
    stressed_probability: float,
    alpha_bar_rule: str = "delta_adjusted",
    alpha_bar_scale: float = 1.0,
    stressed_target_scale: float = 0.85,
    stressed_delta_scale: float = 1.25,
    floor: float | None = None,
) -> dict[str, float | str]:
    """
    Adjust target parameters using a simple stressed-state probability input.

    This is a lightweight operational bridge from regime inference to
    allocation inputs: higher stressed probability lowers the nominal hurdle
    and increases ambiguity size.
    """

    stress = float(np.clip(stressed_probability, 0.0, 1.0))
    rho_regime = float(rho * (1.0 - (1.0 - stressed_target_scale) * stress))
    delta_regime = float(delta * (1.0 + (stressed_delta_scale - 1.0) * stress))
    alpha_bar, alpha_source = build_alpha_bar(
        rho=rho_regime,
        delta=delta_regime,
        rule=alpha_bar_rule,
        scale=alpha_bar_scale,
        floor=floor,
    )
    return {
        "rho": rho_regime,
        "delta": delta_regime,
        "alpha_bar": float(alpha_bar),
        "alpha_bar_source": str(alpha_source),
        "stressed_probability": stress,
    }
