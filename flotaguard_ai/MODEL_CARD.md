# FlotaGuard AI — Model Card

## Intended use
Decision support for demonstrating early quality-risk forecasting in a flotation process. The system predicts the next hour's silica concentrate and flags elevated risk for an operator to review.

## Not intended for
Autonomous plant control, safety-critical actuation, or production decisions without plant-specific validation.

## Data and target
The supplied historical flotation dataset is aggregated to hourly process means. The forecast target is next-hour `% Silica Concentrate`. `% Iron Concentrate` is excluded from the model inputs to avoid using another final concentrate output as a shortcut.

## Inputs
- 21 current-hour process measurements
- latest six hourly silica quality results

## Validation
Strict chronological 80/20 holdout. Later timestamps are never used to train the final models.

See `models/metrics.json` for exact results.

## Alert definition
For the prototype, an alert target is defined as next-hour silica > 3.5%. This is a configurable demonstration threshold, not a universal industrial specification.

## Explainability
The UI displays global XGBoost feature importance. It is not a causal explanation and it is not a per-prediction SHAP explanation.

## Known limitations
- Historical data may not represent a new plant or future operating regime.
- Concept/process drift can degrade accuracy.
- The model uses hourly summaries, not direct streaming sensor ingestion.
- Alert probability is a model score and is not presented as a calibrated physical probability.
- Real deployment requires plant-specific validation, monitoring, and human oversight.
