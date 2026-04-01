from __future__ import annotations

import numpy as np
import pandas as pd


def _top_bucket_metrics(y_true: pd.Series, probabilities: pd.Series, quantile: float = 0.90) -> tuple[float, float]:
    cutoff = probabilities.quantile(quantile)
    alert_bucket = probabilities >= cutoff
    if alert_bucket.sum() == 0:
        return np.nan, np.nan

    precision = float(y_true.loc[alert_bucket].mean())
    total_events = float(y_true.sum())
    capture_rate = float(y_true.loc[alert_bucket].sum() / total_events) if total_events > 0 else np.nan
    return precision, capture_rate


def _calibration_table(y_true: pd.Series, probabilities: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    dataset = pd.DataFrame({"actual": y_true, "predicted": probabilities}).dropna()
    if dataset.empty:
        return pd.DataFrame(columns=["bin", "mean_predicted_probability", "event_rate", "count"])

    n_unique = max(1, min(n_bins, dataset["predicted"].nunique()))
    dataset = dataset.assign(bin=pd.qcut(dataset["predicted"], q=n_unique, duplicates="drop"))
    calibration = (
        dataset.groupby("bin", observed=True)
        .agg(
            mean_predicted_probability=("predicted", "mean"),
            event_rate=("actual", "mean"),
            count=("actual", "size"),
        )
        .reset_index()
    )
    calibration["bin"] = calibration["bin"].astype(str)
    return calibration


def train_instability_detector(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    model_type: str = "logistic",
    test_fraction: float = 0.30,
    random_state: int = 7,
) -> dict[str, object]:
    """Train a time-ordered instability detector for governance alerts."""

    from sklearn.calibration import calibration_curve
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = pd.concat([feature_frame, target.rename("target")], axis=1).dropna()
    if dataset.empty or dataset["target"].nunique() < 2:
        raise ValueError("Monitoring dataset needs at least two classes after alignment and NaN removal.")

    split_idx = max(int(len(dataset) * (1.0 - test_fraction)), 1)
    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]
    if test.empty:
        raise ValueError("Test split is empty; reduce the test fraction or provide more data.")

    X_train = train.drop(columns="target")
    y_train = train["target"]
    X_test = test.drop(columns="target")
    y_test = test["target"]

    training_status = "trained"
    if y_train.nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0]))
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
                    ),
                ),
            ]
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    probability_matrix = model.predict_proba(X_test)
    model_classes = list(getattr(model, "classes_", []))
    if 1 in model_classes:
        class_one_idx = model_classes.index(1)
        probability_values = probability_matrix[:, class_one_idx]
    else:
        probability_values = np.zeros(len(X_test), dtype=float)

    probabilities = pd.Series(probability_values, index=X_test.index, name="instability_probability")
    predictions = (probabilities >= 0.50).astype(int).rename("predicted_label")

    top_decile_precision, top_decile_capture_rate = _top_bucket_metrics(y_test, probabilities, quantile=0.90)
    try:
        roc_auc = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = float(average_precision_score(y_test, probabilities))
    except ValueError:
        pr_auc = np.nan

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "positive_rate": float(y_test.mean()),
        "top_decile_precision": top_decile_precision,
        "top_decile_capture_rate": top_decile_capture_rate,
    }

    confusion = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )

    try:
        importance = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=random_state)
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

    prediction_frame = pd.concat([y_test.rename("actual_label"), probabilities, predictions], axis=1)
    calibration = _calibration_table(y_test, probabilities, n_bins=5)
    try:
        prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=min(5, len(probabilities)), strategy="quantile")
        calibration_curve_df = pd.DataFrame({"mean_predicted_probability": prob_pred, "event_rate": prob_true})
    except ValueError:
        calibration_curve_df = pd.DataFrame(columns=["mean_predicted_probability", "event_rate"])

    return {
        "model": model,
        "status": training_status,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": feature_importance,
        "predictions": prediction_frame,
        "calibration_table": calibration,
        "calibration_curve": calibration_curve_df,
        "train_index": X_train.index,
        "test_index": X_test.index,
    }


def build_monitoring_report(
    latest_feature_row: pd.Series,
    instability_probability: float,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Translate diagnostics into a human-readable alert report."""

    thresholds = thresholds or {
        "instability_probability": 0.50,
        "recent_turnover": 0.50,
        "herfindahl": 0.18,
        "average_pairwise_correlation": 0.70,
        "forecast_realized_risk_gap": 0.05,
        "recent_drawdown_abs": 0.08,
    }

    checks = [
        ("turnover_alert", "recent_turnover", thresholds["recent_turnover"]),
        ("concentration_alert", "herfindahl", thresholds["herfindahl"]),
        ("covariance_instability_alert", "average_pairwise_correlation", thresholds["average_pairwise_correlation"]),
        ("risk_gap_alert", "forecast_realized_risk_gap", thresholds["forecast_realized_risk_gap"]),
        ("drawdown_alert", "recent_drawdown_abs", thresholds["recent_drawdown_abs"]),
    ]

    rows = []
    for check_name, feature_name, cutoff in checks:
        latest_value = float(latest_feature_row.get(feature_name, np.nan))
        rows.append({"check": check_name, "alert": latest_value >= cutoff, "latest_value": latest_value})

    rows.append(
        {
            "check": "instability_model_alert",
            "alert": instability_probability >= thresholds["instability_probability"],
            "latest_value": instability_probability,
        }
    )
    return pd.DataFrame(rows)
