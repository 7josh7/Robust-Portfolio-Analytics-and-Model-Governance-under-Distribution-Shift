from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


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
