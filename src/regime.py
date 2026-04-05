from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .baselines import ensure_psd
from .covariance import estimate_covariance_matrix


DEFAULT_REGIME_FEATURE_COLUMNS = [
    "trailing_vol",
    "average_pairwise_correlation",
    "effective_rank",
    "recent_turnover",
    "herfindahl",
    "recent_drawdown_abs",
    "forecast_realized_risk_gap",
    "cross_sectional_dispersion",
]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / max(weights.sum(), 1e-12)
    return np.sum(values * normalized[:, None], axis=0)


def _weighted_covariance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / max(weights.sum(), 1e-12)
    mean = _weighted_mean(values, normalized)
    demeaned = values - mean
    covariance = (demeaned * normalized[:, None]).T @ demeaned
    return covariance


def _resolve_stressed_state(
    probability_frame: pd.DataFrame,
    factor_returns: pd.Series,
    probability_columns: list[str],
) -> tuple[int, int]:
    state_rows: list[tuple[int, float, float]] = []
    aligned_factor = factor_returns.reindex(probability_frame.index).fillna(0.0)
    factor_values = aligned_factor.to_numpy(dtype=float)
    for probability_column in probability_columns:
        state_index = int(probability_column.split("_")[1])
        weights = probability_frame[probability_column].to_numpy(dtype=float)
        weight_sum = max(weights.sum(), 1e-12)
        mean = float(np.sum(weights * factor_values) / weight_sum)
        variance = float(np.sum(weights * np.square(factor_values - mean)) / weight_sum)
        state_rows.append((state_index, mean, variance))

    stressed_state = min(state_rows, key=lambda item: (item[1], -item[2]))[0]
    calm_state = max(state_rows, key=lambda item: (item[1], -item[2]))[0]
    return stressed_state, calm_state


def _format_regime_probability_frame(
    probabilities: np.ndarray | pd.DataFrame,
    index: pd.Index,
    factor_returns: pd.Series,
) -> pd.DataFrame:
    if isinstance(probabilities, pd.DataFrame):
        probability_frame = probabilities.copy()
        probability_frame.index = index
    else:
        probability_frame = pd.DataFrame(
            np.asarray(probabilities, dtype=float),
            index=index,
            columns=[f"regime_{idx}_prob" for idx in range(np.asarray(probabilities).shape[1])],
        )

    probability_columns = [column for column in probability_frame.columns if str(column).endswith("_prob")]
    stressed_state, calm_state = _resolve_stressed_state(probability_frame, factor_returns, probability_columns)
    probability_frame["most_likely_regime"] = probability_frame[probability_columns].to_numpy().argmax(axis=1)
    probability_frame["stressed_state"] = stressed_state
    probability_frame["calm_state"] = calm_state
    probability_frame["stressed_probability"] = probability_frame[f"regime_{stressed_state}_prob"]
    probability_frame["calm_probability"] = probability_frame[f"regime_{calm_state}_prob"]
    return probability_frame


def estimate_mixture_regime_probabilities(
    market_factor_series: pd.Series,
    n_regimes: int = 2,
    lookback: int | None = None,
    random_state: int = 7,
) -> pd.DataFrame:
    """
    Estimate simple regime probabilities from a market-factor proxy.

    This is a lightweight two-state approximation using a Gaussian mixture on
    factor returns and their absolute magnitude. It is meant to feed
    regime-conditioned portfolio inputs, not to be treated as a macro regime
    oracle.
    """

    from sklearn.mixture import GaussianMixture

    factor = market_factor_series.dropna()
    if lookback is not None:
        factor = factor.tail(lookback)
    if factor.empty:
        raise ValueError("Market factor series is empty after lookback filtering.")

    feature_frame = pd.DataFrame(
        {
            "market_factor": factor,
            "abs_market_factor": factor.abs(),
        },
        index=factor.index,
    )
    if len(feature_frame) < max(10, n_regimes):
        probabilities = np.full((len(feature_frame), n_regimes), 1.0 / n_regimes)
    else:
        model = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")
            model.fit(feature_frame.to_numpy())
        probabilities = model.predict_proba(feature_frame.to_numpy())

    regime_probs = _format_regime_probability_frame(
        probabilities=probabilities,
        index=feature_frame.index,
        factor_returns=feature_frame["market_factor"],
    )
    regime_probs["regime_model_version"] = "mixture"
    return regime_probs


def estimate_regime_probabilities(
    market_factor_series: pd.Series,
    n_regimes: int = 2,
    lookback: int | None = None,
    random_state: int = 7,
) -> pd.DataFrame:
    """
    Backward-compatible alias for the lightweight regime-mixture estimator.

    The HMM workflow now lives in ``fit_two_state_hmm`` and
    ``estimate_regime_conditioned_inputs_hmm``.
    """

    return estimate_mixture_regime_probabilities(
        market_factor_series=market_factor_series,
        n_regimes=n_regimes,
        lookback=lookback,
        random_state=random_state,
    )


def fit_two_state_hmm(
    market_factor_series: pd.Series,
    lookback: int | None = None,
    switching_variance: bool = True,
) -> dict[str, object]:
    """
    Fit a two-state hidden Markov / Markov-switching model on a market factor.

    This uses ``statsmodels`` Markov regression as a tractable two-regime proxy
    for a Costa-Kwon-style latent market-state engine.
    """

    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    factor = market_factor_series.dropna()
    if lookback is not None:
        factor = factor.tail(lookback)
    if len(factor) < 30:
        raise ValueError("At least 30 observations are required to fit the two-state HMM.")

    model = MarkovRegression(
        factor.astype(float),
        k_regimes=2,
        trend="c",
        switching_variance=bool(switching_variance),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
        warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
        result = model.fit(disp=False)

    return {
        "model": model,
        "result": result,
        "series": factor,
        "lookback": len(factor),
        "regime_model_version": "hmm_markov_regression",
    }


def infer_filtered_regime_probs(
    hmm_fit: dict[str, object],
) -> pd.DataFrame:
    result = hmm_fit["result"]
    factor = hmm_fit["series"]
    probabilities = pd.DataFrame(result.filtered_marginal_probabilities, index=factor.index)
    probabilities.columns = [f"regime_{idx}_prob" for idx in range(probabilities.shape[1])]
    formatted = _format_regime_probability_frame(probabilities, factor.index, factor)
    formatted["probability_mode"] = "filtered"
    formatted["regime_model_version"] = hmm_fit.get("regime_model_version", "hmm_markov_regression")
    return formatted


def infer_smoothed_regime_probs(
    hmm_fit: dict[str, object],
) -> pd.DataFrame:
    result = hmm_fit["result"]
    factor = hmm_fit["series"]
    probabilities = pd.DataFrame(result.smoothed_marginal_probabilities, index=factor.index)
    probabilities.columns = [f"regime_{idx}_prob" for idx in range(probabilities.shape[1])]
    formatted = _format_regime_probability_frame(probabilities, factor.index, factor)
    formatted["probability_mode"] = "smoothed"
    formatted["regime_model_version"] = hmm_fit.get("regime_model_version", "hmm_markov_regression")
    return formatted


def estimate_regime_conditioned_mean(
    asset_returns: pd.DataFrame,
    regime_probs: pd.DataFrame,
) -> dict[str, pd.Series]:
    aligned_probs = regime_probs.reindex(asset_returns.index).dropna()
    aligned_returns = asset_returns.reindex(aligned_probs.index).fillna(0.0)
    values = aligned_returns.to_numpy(dtype=float)
    state_means: dict[str, pd.Series] = {}

    for probability_column in [column for column in aligned_probs.columns if str(column).endswith("_prob")]:
        state_weights = aligned_probs[probability_column].to_numpy(dtype=float)
        if state_weights.sum() <= 1e-12:
            state_means[probability_column] = aligned_returns.mean().rename(probability_column)
            continue
        state_means[probability_column] = pd.Series(
            _weighted_mean(values, state_weights),
            index=aligned_returns.columns,
            name=probability_column,
        )
    return state_means


def estimate_regime_conditioned_covariance(
    asset_returns: pd.DataFrame,
    regime_probs: pd.DataFrame,
    covariance_method: str = "ledoit_wolf",
    calm_covariance_method: str | None = None,
    stressed_covariance_method: str | None = None,
) -> dict[str, np.ndarray]:
    aligned_probs = regime_probs.reindex(asset_returns.index).dropna()
    aligned_returns = asset_returns.reindex(aligned_probs.index).fillna(0.0)
    state_covariances: dict[str, np.ndarray] = {}
    state_columns = [column for column in aligned_probs.columns if str(column).endswith("_prob")]
    stressed_state = int(aligned_probs["stressed_state"].iloc[-1]) if "stressed_state" in aligned_probs else 0
    values = aligned_returns.to_numpy(dtype=float)

    for probability_column in state_columns:
        state_weights = aligned_probs[probability_column].to_numpy(dtype=float)
        state_cov = _weighted_covariance(values, state_weights) if state_weights.sum() > 1e-12 else np.nan
        state_index = int(probability_column.split("_")[1])
        if covariance_method == "state_aware":
            selected_covariance_method = (
                stressed_covariance_method if state_index == stressed_state else calm_covariance_method
            ) or "ledoit_wolf"
        else:
            selected_covariance_method = covariance_method
        weighted_sample_covariance = (
            estimate_covariance_matrix(aligned_returns, method="sample")
            if isinstance(state_cov, float) or np.isnan(state_cov).any()
            else ensure_psd(state_cov)
        )
        if selected_covariance_method == "sample":
            state_covariances[probability_column] = weighted_sample_covariance
            continue

        subset_cutoff = float(np.quantile(state_weights, 0.60)) if len(state_weights) > 5 else 0.0
        state_subset = aligned_returns.loc[state_weights >= subset_cutoff]
        if len(state_subset) < max(20, aligned_returns.shape[1] * 3):
            state_subset = aligned_returns
        method_covariance = estimate_covariance_matrix(state_subset, method=selected_covariance_method)
        state_covariances[probability_column] = ensure_psd(0.50 * weighted_sample_covariance + 0.50 * method_covariance)
    return state_covariances


def _combine_regime_state_inputs(
    aligned_returns: pd.DataFrame,
    state_means: dict[str, pd.Series],
    state_covariances: dict[str, np.ndarray],
    latest_probabilities: pd.Series,
) -> tuple[pd.Series, np.ndarray]:
    probability_columns = [column for column in latest_probabilities.index if str(column).endswith("_prob")]
    mixture_mean = sum(float(latest_probabilities[column]) * state_means[column] for column in probability_columns)
    mixture_covariance = np.zeros((aligned_returns.shape[1], aligned_returns.shape[1]), dtype=float)
    mean_vector = mixture_mean.reindex(aligned_returns.columns).to_numpy(dtype=float)

    for column in probability_columns:
        state_mean_vector = state_means[column].reindex(aligned_returns.columns).to_numpy(dtype=float)
        state_covariance = state_covariances[column]
        mean_gap = state_mean_vector - mean_vector
        mixture_covariance += float(latest_probabilities[column]) * (state_covariance + np.outer(mean_gap, mean_gap))

    return mixture_mean.rename("regime_conditioned_mean"), ensure_psd(mixture_covariance)


def estimate_regime_conditioned_inputs_mixture(
    asset_returns: pd.DataFrame,
    factor_returns: pd.Series | None = None,
    regime_probs: pd.DataFrame | None = None,
    lookback: int = 252,
    covariance_method: str = "ledoit_wolf",
    calm_covariance_method: str | None = None,
    stressed_covariance_method: str | None = None,
    probability_temperature: float = 1.0,
    stressed_probability_threshold: float = 0.0,
) -> dict[str, object]:
    """
    Estimate current regime-conditioned mean returns and covariance.

    The recent return window is split probabilistically across the inferred
    regimes. State-specific means and covariances are estimated first, then the
    latest regime probabilities are used to form a current mixture input pair.
    """

    recent_returns = asset_returns.tail(lookback).fillna(0.0)
    if recent_returns.empty:
        raise ValueError("Asset returns are empty after lookback filtering.")

    if factor_returns is None:
        factor_returns = recent_returns.mean(axis=1).rename("market_factor")
    else:
        factor_returns = factor_returns.reindex(recent_returns.index).fillna(0.0)

    if regime_probs is None:
        regime_probs = estimate_mixture_regime_probabilities(
            factor_returns,
            n_regimes=2,
            lookback=len(recent_returns),
        )

    aligned_probs = regime_probs.reindex(recent_returns.index).dropna()
    aligned_returns = recent_returns.reindex(aligned_probs.index).fillna(0.0)
    probability_columns = [column for column in aligned_probs.columns if column.endswith("_prob")]
    if not probability_columns:
        raise ValueError("Regime probability frame does not contain probability columns.")

    state_means = estimate_regime_conditioned_mean(aligned_returns, aligned_probs)
    state_covariances = estimate_regime_conditioned_covariance(
        aligned_returns,
        aligned_probs,
        covariance_method=covariance_method,
        calm_covariance_method=calm_covariance_method,
        stressed_covariance_method=stressed_covariance_method,
    )

    latest_probabilities = aligned_probs[probability_columns].iloc[-1]
    if probability_temperature != 1.0:
        sharpened = np.power(np.clip(latest_probabilities.to_numpy(dtype=float), 1e-12, 1.0), float(probability_temperature))
        latest_probabilities = pd.Series(sharpened / sharpened.sum(), index=latest_probabilities.index)
    mixture_mean, mixture_covariance = _combine_regime_state_inputs(
        aligned_returns=aligned_returns,
        state_means=state_means,
        state_covariances=state_covariances,
        latest_probabilities=latest_probabilities,
    )

    latest_regime = int(aligned_probs["most_likely_regime"].iloc[-1])
    stressed_state = int(aligned_probs["stressed_state"].iloc[-1]) if "stressed_state" in aligned_probs else 0
    stressed_probability = float(aligned_probs[f"regime_{stressed_state}_prob"].iloc[-1])
    stress_activation = float(
        np.clip((stressed_probability - stressed_probability_threshold) / max(1.0 - stressed_probability_threshold, 1e-12), 0.0, 1.0)
    )

    return {
        "mean_returns": mixture_mean.rename("regime_conditioned_mean"),
        "covariance": mixture_covariance,
        "latest_regime": latest_regime,
        "stressed_probability": stressed_probability,
        "stress_activation": stress_activation,
        "regime_probabilities": aligned_probs,
        "state_means": state_means,
        "state_covariances": state_covariances,
        "regime_model_version": "mixture",
        "probability_mode": "mixture",
    }


def estimate_regime_conditioned_inputs_hmm(
    asset_returns: pd.DataFrame,
    factor_returns: pd.Series | None = None,
    lookback: int = 252,
    covariance_method: str = "ledoit_wolf",
    calm_covariance_method: str | None = None,
    stressed_covariance_method: str | None = None,
    probability_temperature: float = 1.0,
    stressed_probability_threshold: float = 0.0,
    switching_variance: bool = True,
    current_probability_mode: str = "filtered",
    estimation_probability_mode: str = "smoothed",
) -> dict[str, object]:
    """
    Estimate regime-conditioned inputs from a two-state HMM/Markov-switching fit.

    The state probabilities are inferred from a market factor, state-specific
    moments are estimated on the recent history, and the latest regime
    probabilities are used to mix those state inputs into current portfolio
    parameters.
    """

    recent_returns = asset_returns.tail(lookback).fillna(0.0)
    if recent_returns.empty:
        raise ValueError("Asset returns are empty after lookback filtering.")

    if factor_returns is None:
        factor_returns = recent_returns.mean(axis=1).rename("market_factor")
    else:
        factor_returns = factor_returns.reindex(recent_returns.index).fillna(0.0)

    try:
        hmm_fit = fit_two_state_hmm(
            market_factor_series=factor_returns,
            lookback=len(recent_returns),
            switching_variance=switching_variance,
        )
        filtered_probs = infer_filtered_regime_probs(hmm_fit)
        smoothed_probs = infer_smoothed_regime_probs(hmm_fit)
    except Exception as exc:
        mixture_result = estimate_regime_conditioned_inputs_mixture(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            lookback=lookback,
            covariance_method=covariance_method,
            calm_covariance_method=calm_covariance_method,
            stressed_covariance_method=stressed_covariance_method,
            probability_temperature=probability_temperature,
            stressed_probability_threshold=stressed_probability_threshold,
        )
        mixture_result["regime_model_version"] = "hmm_fallback_to_mixture"
        mixture_result["regime_model_status"] = f"fallback::{type(exc).__name__}"
        return mixture_result

    current_mode = current_probability_mode.lower()
    estimation_mode = estimation_probability_mode.lower()
    estimation_probs = smoothed_probs if estimation_mode == "smoothed" else filtered_probs
    current_probs = filtered_probs if current_mode == "filtered" else smoothed_probs

    aligned_estimation_probs = estimation_probs.reindex(recent_returns.index).dropna()
    aligned_returns = recent_returns.reindex(aligned_estimation_probs.index).fillna(0.0)
    probability_columns = [column for column in aligned_estimation_probs.columns if column.endswith("_prob")]
    state_means = estimate_regime_conditioned_mean(aligned_returns, aligned_estimation_probs)
    state_covariances = estimate_regime_conditioned_covariance(
        aligned_returns,
        aligned_estimation_probs,
        covariance_method=covariance_method,
        calm_covariance_method=calm_covariance_method,
        stressed_covariance_method=stressed_covariance_method,
    )

    latest_probabilities = current_probs.reindex(aligned_returns.index).dropna()[probability_columns].iloc[-1]
    if probability_temperature != 1.0:
        sharpened = np.power(np.clip(latest_probabilities.to_numpy(dtype=float), 1e-12, 1.0), float(probability_temperature))
        latest_probabilities = pd.Series(sharpened / sharpened.sum(), index=latest_probabilities.index)

    mixture_mean, mixture_covariance = _combine_regime_state_inputs(
        aligned_returns=aligned_returns,
        state_means=state_means,
        state_covariances=state_covariances,
        latest_probabilities=latest_probabilities,
    )

    latest_probs_frame = current_probs.reindex(aligned_returns.index).dropna()
    stressed_state = int(latest_probs_frame["stressed_state"].iloc[-1]) if "stressed_state" in latest_probs_frame else 0
    stressed_probability = float(latest_probs_frame[f"regime_{stressed_state}_prob"].iloc[-1])
    stress_activation = float(
        np.clip((stressed_probability - stressed_probability_threshold) / max(1.0 - stressed_probability_threshold, 1e-12), 0.0, 1.0)
    )

    return {
        "mean_returns": mixture_mean.rename("regime_conditioned_mean"),
        "covariance": mixture_covariance,
        "latest_regime": int(latest_probs_frame["most_likely_regime"].iloc[-1]),
        "stressed_probability": stressed_probability,
        "stress_activation": stress_activation,
        "regime_probabilities": latest_probs_frame,
        "filtered_regime_probabilities": filtered_probs,
        "smoothed_regime_probabilities": smoothed_probs,
        "state_means": state_means,
        "state_covariances": state_covariances,
        "regime_model_version": "hmm_markov_regression",
        "probability_mode": current_mode,
        "estimation_probability_mode": estimation_mode,
        "regime_model_status": "trained",
    }


def estimate_regime_conditioned_inputs(
    asset_returns: pd.DataFrame,
    factor_returns: pd.Series | None = None,
    regime_probs: pd.DataFrame | None = None,
    lookback: int = 252,
    covariance_method: str = "ledoit_wolf",
    calm_covariance_method: str | None = None,
    stressed_covariance_method: str | None = None,
    probability_temperature: float = 1.0,
    stressed_probability_threshold: float = 0.0,
) -> dict[str, object]:
    """
    Backward-compatible alias for the original lightweight regime-mixture path.
    """

    return estimate_regime_conditioned_inputs_mixture(
        asset_returns=asset_returns,
        factor_returns=factor_returns,
        regime_probs=regime_probs,
        lookback=lookback,
        covariance_method=covariance_method,
        calm_covariance_method=calm_covariance_method,
        stressed_covariance_method=stressed_covariance_method,
        probability_temperature=probability_temperature,
        stressed_probability_threshold=stressed_probability_threshold,
    )


def build_regime_labels(
    feature_frame: pd.DataFrame,
    vol_quantile: float = 0.67,
    corr_quantile: float = 0.67,
    drawdown_quantile: float = 0.67,
    dispersion_quantile: float = 0.67,
) -> pd.Series:
    """
    Build simple, heuristic market-state labels from trailing diagnostics.

    These labels are intentionally descriptive rather than canonical. They are
    designed for governance-style conditioning of portfolio diagnostics, not as
    a macroeconomic truth set.
    """

    required_columns = {"trailing_vol", "average_pairwise_correlation", "recent_drawdown_abs"}
    missing_columns = required_columns.difference(feature_frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required regime features: {sorted(missing_columns)}")

    dataset = feature_frame.copy()
    dispersion = dataset["cross_sectional_dispersion"] if "cross_sectional_dispersion" in dataset else dataset["trailing_vol"]

    vol_high = dataset["trailing_vol"] >= dataset["trailing_vol"].quantile(vol_quantile)
    corr_high = dataset["average_pairwise_correlation"] >= dataset["average_pairwise_correlation"].quantile(corr_quantile)
    drawdown_high = dataset["recent_drawdown_abs"] >= dataset["recent_drawdown_abs"].quantile(drawdown_quantile)
    dispersion_high = dispersion >= dispersion.quantile(dispersion_quantile)

    labels = pd.Series("calm", index=dataset.index, dtype="object")
    labels.loc[corr_high] = "crowded"
    labels.loc[(~corr_high) & drawdown_high] = "drawdown_pressure"
    labels.loc[(vol_high & corr_high) | (drawdown_high & dispersion_high)] = "stressed"

    categories = ["calm", "crowded", "drawdown_pressure", "stressed"]
    return pd.Series(pd.Categorical(labels, categories=categories, ordered=True), index=dataset.index, name="regime")


def train_regime_classifier(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    model_type: str = "random_forest",
    test_fraction: float = 0.30,
    random_state: int = 7,
) -> dict[str, object]:
    """
    Train a time-ordered classifier for heuristic regime tags.

    This is intentionally a label-recovery workflow over hand-crafted market-
    state labels derived from the same diagnostic feature family. It is useful
    as an operational tagging layer, but it should not be interpreted as a
    strong predictive ML result.
    """

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = pd.concat([feature_frame, target.rename("regime")], axis=1).dropna()
    if dataset.empty or dataset["regime"].nunique() < 2:
        raise ValueError("Regime dataset needs at least two classes after alignment and NaN removal.")

    split_idx = max(int(len(dataset) * (1.0 - test_fraction)), 1)
    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]
    if test.empty:
        raise ValueError("Test split is empty; reduce the test fraction or provide more data.")

    X_train = train.drop(columns="regime")
    y_train = train["regime"].astype(str)
    X_test = test.drop(columns="regime")
    y_test = test["regime"].astype(str)

    training_status = "trained"
    if y_train.nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
        training_status = "dummy_single_class_train"
    elif model_type == "logistic":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=random_state,
                        multi_class="auto",
                    ),
                ),
            ]
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), index=X_test.index, name="predicted_regime")

    probability_frame = pd.DataFrame(index=X_test.index)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = [str(label) for label in getattr(model, "classes_", [])]
        probability_frame = pd.DataFrame(
            probabilities,
            index=X_test.index,
            columns=[f"prob_{label}" for label in classes],
        )
        confidence = probability_frame.max(axis=1).rename("prediction_confidence")
    else:
        confidence = pd.Series(np.nan, index=X_test.index, name="prediction_confidence")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

    labels = sorted(pd.Index(y_test).union(predictions).unique())
    confusion = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    try:
        importance = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=20,
            random_state=random_state,
            scoring="accuracy",
        )
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    except ValueError:
        feature_importance = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": np.nan,
                "importance_std": np.nan,
            }
        )

    prediction_frame = pd.concat(
        [
            y_test.rename("actual_regime"),
            predictions,
            confidence,
            probability_frame,
        ],
        axis=1,
    )

    return {
        "model": model,
        "status": training_status,
        "task_scope": "heuristic_label_recovery",
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": feature_importance,
        "predictions": prediction_frame,
        "train_index": X_train.index,
        "test_index": X_test.index,
    }


def summarize_regime_conditionals(
    rebalance_results: pd.DataFrame,
    regime_predictions: pd.DataFrame | pd.Series,
    strategy: str = "wasserstein_proxy_min_var",
    regime_column: str = "predicted_regime",
) -> pd.DataFrame:
    """Summarize proxy diagnostics conditional on classified regimes."""

    proxy = rebalance_results[rebalance_results["strategy"] == strategy].copy()
    proxy["risk_gap"] = (proxy["realized_vol"] - proxy["forecast_vol"]).abs()

    if isinstance(regime_predictions, pd.Series):
        regime_frame = regime_predictions.rename(regime_column).to_frame()
    else:
        if regime_column not in regime_predictions.columns:
            raise ValueError(f"Missing regime column: {regime_column}")
        regime_frame = regime_predictions[[regime_column]].copy()

    joined = proxy.join(regime_frame, how="inner")
    if joined.empty:
        return pd.DataFrame()

    summary = (
        joined.groupby(regime_column, observed=True)
        .agg(
            n_rebalances=("strategy", "size"),
            avg_chosen_epsilon=("chosen_epsilon", "mean"),
            avg_slack_used=("slack_used", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_execution_eta=("execution_eta", "mean"),
            avg_risk_gap=("risk_gap", "mean"),
            avg_hold_period_return=("hold_period_return", "mean"),
        )
        .sort_index()
    )
    return summary
