from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

app = Flask(__name__)

regressor = joblib.load(MODEL_DIR / "regressor.pkl")
alert_classifier = joblib.load(MODEL_DIR / "alert_classifier.pkl")
process_scaler = joblib.load(MODEL_DIR / "process_scaler.pkl")
process_kmeans = joblib.load(MODEL_DIR / "process_kmeans.pkl")


def load_json(name):
    return json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))

PROCESS_FEATURES = load_json("process_features.json")
MODEL_FEATURES = load_json("model_features.json")
METRICS = load_json("metrics.json")
CONFIG = load_json("config.json")
CLUSTER_PROFILES = load_json("cluster_profiles.json")
INPUT_RANGES = load_json("input_ranges.json")
SILICA_RANGE = load_json("silica_range.json")
SCENARIOS = load_json("demo_scenarios.json")
IMPORTANCE = load_json("feature_importance.json")

DISPLAY_NAMES = {
    "silica_current": "Latest silica result",
    "silica_lag_1h": "Silica 1 hour ago",
    "silica_lag_2h": "Silica 2 hours ago",
    "silica_lag_3h": "Silica 3 hours ago",
    "silica_lag_4h": "Silica 4 hours ago",
    "silica_lag_5h": "Silica 5 hours ago",
}

GROUPS = {
    "Feed & chemistry": [
        "% Iron Feed", "% Silica Feed", "Starch Flow", "Amina Flow",
        "Ore Pulp Flow", "Ore Pulp pH", "Ore Pulp Density",
    ],
    "Flotation air flow": [f"Flotation Column 0{i} Air Flow" for i in range(1, 8)],
    "Flotation levels": [f"Flotation Column 0{i} Level" for i in range(1, 8)],
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    feature_meta = {}
    for feature in PROCESS_FEATURES:
        r = INPUT_RANGES[feature]
        feature_meta[feature] = {
            "label": feature,
            "group": next(g for g, cols in GROUPS.items() if feature in cols),
            "min": r["min"], "max": r["max"], "default": r["median"],
        }
    for feature in MODEL_FEATURES:
        if feature.startswith("silica_"):
            feature_meta[feature] = {
                "label": DISPLAY_NAMES.get(feature, feature),
                "group": "Recent quality history",
                "min": SILICA_RANGE["min"], "max": SILICA_RANGE["max"],
                "default": SILICA_RANGE["median"],
            }
    return jsonify({
        "project": {
            "name": "FlotaGuard AI",
            "subtitle": "One-hour-ahead flotation quality early warning",
            "human_in_the_loop": True,
        },
        "model_features": MODEL_FEATURES,
        "process_features": PROCESS_FEATURES,
        "groups": GROUPS,
        "feature_meta": feature_meta,
        "config": CONFIG,
        "metrics": METRICS,
        "scenarios": SCENARIOS,
        "feature_importance": IMPORTANCE,
        "cluster_profiles": CLUSTER_PROFILES,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    values = payload.get("inputs")
    if not isinstance(values, dict):
        return jsonify({"error": "Expected JSON object: {inputs: {...}}"}), 400

    missing = [f for f in MODEL_FEATURES if f not in values]
    if missing:
        return jsonify({"error": "Missing required inputs", "missing": missing}), 400

    clean = {}
    invalid = []
    for f in MODEL_FEATURES:
        try:
            v = float(values[f])
            if not np.isfinite(v):
                raise ValueError
            clean[f] = v
        except (TypeError, ValueError):
            invalid.append(f)
    if invalid:
        return jsonify({"error": "Inputs must be finite numbers", "invalid": invalid}), 400

    x = pd.DataFrame([[clean[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    predicted_silica = float(regressor.predict(x)[0])
    alert_probability = float(alert_classifier.predict_proba(x)[0, 1])
    p_threshold = float(CONFIG["probability_threshold"])
    quality_threshold = float(CONFIG["quality_threshold_silica_pct"])
    alert_status = "ALERT" if alert_probability >= p_threshold else "SAFE"

    # Triage band is a UI aid, not a third trained class.
    if alert_status == "ALERT":
        triage = "High risk"
    elif predicted_silica >= quality_threshold - 0.35 or alert_probability >= p_threshold * 0.65:
        triage = "Watch"
    else:
        triage = "Stable"

    process_row = pd.DataFrame([[clean[f] for f in PROCESS_FEATURES]], columns=PROCESS_FEATURES)
    cluster_id = int(process_kmeans.predict(process_scaler.transform(process_row))[0])
    pattern = CLUSTER_PROFILES[str(cluster_id)]

    range_warnings = []
    for f in PROCESS_FEATURES:
        r = INPUT_RANGES[f]
        if clean[f] < r["min"] or clean[f] > r["max"]:
            range_warnings.append({
                "feature": f, "value": clean[f],
                "training_1pct": r["min"], "training_99pct": r["max"],
            })
    for f in ["silica_current", "silica_lag_1h", "silica_lag_2h", "silica_lag_3h", "silica_lag_4h", "silica_lag_5h"]:
        if clean[f] < SILICA_RANGE["min"] or clean[f] > SILICA_RANGE["max"]:
            range_warnings.append({
                "feature": f, "value": clean[f],
                "training_1pct": SILICA_RANGE["min"], "training_99pct": SILICA_RANGE["max"],
            })

    return jsonify({
        "predicted_next_hour_silica": round(predicted_silica, 3),
        "alert_probability": round(alert_probability, 4),
        "alert_status": alert_status,
        "triage": triage,
        "prototype_quality_threshold": quality_threshold,
        "forecast_above_quality_threshold": bool(predicted_silica > quality_threshold),
        "process_pattern": {
            "cluster_id": cluster_id,
            "label": pattern["label"],
            "training_mean_silica": pattern["mean_silica"],
        },
        "out_of_distribution_warnings": range_warnings,
        "note": "Decision-support prototype only; operator review is required.",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
