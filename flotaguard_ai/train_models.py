"""Train the competition version of FlotaGuard AI.

Methodology changes versus the exploratory notebook:
- Aggregate raw 20-second-ish process rows to hourly process summaries to align the model with the hourly quality signal.
- Forecast the NEXT hour's silica concentrate, rather than estimating the already-observed current value.
- Use a strict chronological 80/20 holdout (no future timestamps in training).
- Use only current process summaries plus the last 6 known hourly silica results.
- Tune the alert probability threshold on a validation slice inside the training period only.
- Fit process-state clusters on training data and label them from observed training silica means.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "e85b593a-e552-4dbc-9018-1db12b6121e2.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
TARGET = "% Silica Concentrate"
QUALITY_THRESHOLD = 3.5  # Prototype threshold; plant-specific calibration is required before deployment.


def load_hourly():
    df = pd.read_csv(DATA, decimal=",").drop_duplicates().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable")
    process_features = [c for c in df.columns if c not in ["date", "% Iron Concentrate", TARGET]]
    hourly = (
        df.groupby("date", as_index=False)[process_features + [TARGET]]
          .mean()
          .sort_values("date")
          .reset_index(drop=True)
    )
    return hourly, process_features


def build_supervised(hourly, process_features):
    data = hourly.copy()
    data["target_next_hour"] = data[TARGET].shift(-1)
    model_features = list(process_features)
    for k in range(6):
        name = "silica_current" if k == 0 else f"silica_lag_{k}h"
        data[name] = data[TARGET].shift(k)
        model_features.append(name)
    data = data.dropna().reset_index(drop=True)
    return data, model_features


def train():
    hourly, process_features = load_hourly()
    data, model_features = build_supervised(hourly, process_features)

    split = int(len(data) * 0.80)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    X_train, y_train = train[model_features], train["target_next_hour"]
    X_test, y_test = test[model_features], test["target_next_hour"]

    reg = XGBRegressor(
        n_estimators=650, max_depth=5, learning_rate=0.03,
        subsample=0.90, colsample_bytree=0.90, reg_lambda=5,
        objective="reg:squarederror", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(X_test)

    # Alert model: tune threshold on a validation slice strictly inside training period.
    train_alert = (y_train > QUALITY_THRESHOLD).astype(int)
    val_split = int(len(train) * 0.80)
    sub_train = train.iloc[:val_split]
    val = train.iloc[val_split:]
    sub_y = (sub_train["target_next_hour"] > QUALITY_THRESHOLD).astype(int)
    val_y = (val["target_next_hour"] > QUALITY_THRESHOLD).astype(int)
    pos_weight = float((len(sub_y) - sub_y.sum()) / max(sub_y.sum(), 1))

    clf_for_threshold = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.90, colsample_bytree=0.90, reg_lambda=5,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1,
    )
    clf_for_threshold.fit(sub_train[model_features], sub_y)
    val_prob = clf_for_threshold.predict_proba(val[model_features])[:, 1]
    candidates = np.arange(0.30, 0.81, 0.02)
    probability_threshold = float(max(candidates, key=lambda t: f1_score(val_y, val_prob >= t)))

    full_pos_weight = float((len(train_alert) - train_alert.sum()) / max(train_alert.sum(), 1))
    clf = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.90, colsample_bytree=0.90, reg_lambda=5,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        scale_pos_weight=full_pos_weight, random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, train_alert)
    test_prob = clf.predict_proba(X_test)[:, 1]
    test_alert = (y_test > QUALITY_THRESHOLD).astype(int)
    test_pred_alert = (test_prob >= probability_threshold).astype(int)

    # Process-pattern model uses only process features (not the target/history).
    scaler = StandardScaler().fit(train[process_features])
    kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=20, batch_size=512)
    kmeans.fit(scaler.transform(train[process_features]))
    train_cluster = kmeans.predict(scaler.transform(train[process_features]))
    profile = (
        pd.DataFrame({"cluster": train_cluster, "silica": train[TARGET].to_numpy()})
        .groupby("cluster")["silica"].agg(["mean", "median", "count"]).reset_index()
    )
    ordered = profile.sort_values("mean")["cluster"].tolist()
    pattern_names = ["Lower-Silica Pattern", "Moderate-Silica Pattern", "Elevated-Silica Pattern", "Higher-Silica Pattern"]
    label_map = {int(c): label for c, label in zip(ordered, pattern_names)}
    profiles = {
        str(int(r.cluster)): {
            "label": label_map[int(r.cluster)],
            "mean_silica": round(float(r["mean"]), 3),
            "median_silica": round(float(r["median"]), 3),
            "training_hours": int(r["count"]),
        }
        for _, r in profile.iterrows()
    }

    metrics = {
        "task": "One-hour-ahead silica quality forecasting and alert detection",
        "validation": "Strict chronological 80/20 holdout by hourly timestamp",
        "train_period": [str(train["date"].iloc[0]), str(train["date"].iloc[-1])],
        "test_period": [str(test["date"].iloc[0]), str(test["date"].iloc[-1])],
        "train_hours": int(len(train)),
        "test_hours": int(len(test)),
        "regression": {
            "r2": round(float(r2_score(y_test, reg_pred)), 4),
            "mae_percentage_points": round(float(mean_absolute_error(y_test, reg_pred)), 4),
            "rmse_percentage_points": round(float(mean_squared_error(y_test, reg_pred) ** 0.5), 4),
        },
        "alert_classification": {
            "prototype_silica_threshold_pct": QUALITY_THRESHOLD,
            "probability_threshold": round(probability_threshold, 2),
            "accuracy": round(float(accuracy_score(test_alert, test_pred_alert)), 4),
            "precision_alert": round(float(precision_score(test_alert, test_pred_alert, zero_division=0)), 4),
            "recall_alert": round(float(recall_score(test_alert, test_pred_alert, zero_division=0)), 4),
            "f1_alert": round(float(f1_score(test_alert, test_pred_alert, zero_division=0)), 4),
            "test_alert_rate": round(float(test_alert.mean()), 4),
        },
        "important_limitations": [
            "This is a historical-data prototype, not a production control system.",
            "The 3.5% silica threshold is a configurable prototype setting, not claimed as a universal industry standard.",
            "A real deployment requires plant-specific recalibration, live sensor integration, monitoring for process drift, and operator validation.",
        ],
    }

    # Robust ranges from training period for UI and out-of-distribution warnings.
    input_ranges = {}
    for c in process_features:
        s = train[c]
        lo, hi, med = float(s.quantile(0.01)), float(s.quantile(0.99)), float(s.median())
        input_ranges[c] = {"min": lo, "max": hi, "median": med}
    silica_series = train[TARGET]
    silica_range = {
        "min": float(silica_series.quantile(0.01)),
        "max": float(silica_series.quantile(0.99)),
        "median": float(silica_series.median()),
    }

    # Demo scenarios come only from the chronological held-out test period.
    demo = test[["date", TARGET, "target_next_hour"] + model_features].copy()
    demo["predicted_next"] = reg_pred
    demo["alert_probability"] = test_prob
    demo["predicted_alert"] = test_pred_alert

    def choose(target_value, require_alert=None):
        subset = demo
        if require_alert is not None:
            subset = subset[subset["predicted_alert"] == int(require_alert)]
        i = (subset["target_next_hour"] - target_value).abs().idxmin()
        r = demo.loc[i]
        return {
            "timestamp": str(r["date"]),
            "historical_next_hour_silica": round(float(r["target_next_hour"]), 3),
            "model_prediction_on_holdout": round(float(r["predicted_next"]), 3),
            "model_alert_probability": round(float(r["alert_probability"]), 3),
            "inputs": {c: float(r[c]) for c in model_features},
        }

    # Pick a real UI Watch case using the same triage rule as app.py.
    watch_mask = (demo["predicted_alert"] == 0) & (
        (demo["predicted_next"] >= QUALITY_THRESHOLD - 0.35)
        | (demo["alert_probability"] >= probability_threshold * 0.65)
    )
    watch_pool = demo[watch_mask].copy()
    if len(watch_pool):
        watch_i = (
            (watch_pool["target_next_hour"] - 3.4).abs()
            + 0.5 * (watch_pool["predicted_next"] - 3.3).abs()
        ).idxmin()
        wr = demo.loc[watch_i]
        watch_scenario = {
            "timestamp": str(wr["date"]),
            "historical_next_hour_silica": round(float(wr["target_next_hour"]), 3),
            "model_prediction_on_holdout": round(float(wr["predicted_next"]), 3),
            "model_alert_probability": round(float(wr["alert_probability"]), 3),
            "inputs": {c: float(wr[c]) for c in model_features},
        }
    else:
        watch_scenario = choose(3.2, False)

    scenarios = {
        "Stable": choose(1.6, False),
        "Watch": watch_scenario,
        "High Risk": choose(4.6, True),
    }

    reg_importance = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(model_features, reg.feature_importances_)],
        key=lambda x: x["importance"], reverse=True,
    )[:10]
    clf_importance = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(model_features, clf.feature_importances_)],
        key=lambda x: x["importance"], reverse=True,
    )[:10]

    joblib.dump(reg, OUT / "regressor.pkl")
    joblib.dump(clf, OUT / "alert_classifier.pkl")
    joblib.dump(scaler, OUT / "process_scaler.pkl")
    joblib.dump(kmeans, OUT / "process_kmeans.pkl")
    (OUT / "process_features.json").write_text(json.dumps(process_features, indent=2), encoding="utf-8")
    (OUT / "model_features.json").write_text(json.dumps(model_features, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({
        "quality_threshold_silica_pct": QUALITY_THRESHOLD,
        "probability_threshold": probability_threshold,
        "forecast_horizon": "1 hour",
        "input_definition": "Current-hour mean process measurements + latest 6 hourly silica quality results",
    }, indent=2), encoding="utf-8")
    (OUT / "cluster_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    (OUT / "input_ranges.json").write_text(json.dumps(input_ranges, indent=2), encoding="utf-8")
    (OUT / "silica_range.json").write_text(json.dumps(silica_range, indent=2), encoding="utf-8")
    (OUT / "demo_scenarios.json").write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    (OUT / "feature_importance.json").write_text(json.dumps({"regression": reg_importance, "alert": clf_importance}, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(json.dumps(scenarios, indent=2))


if __name__ == "__main__":
    train()
