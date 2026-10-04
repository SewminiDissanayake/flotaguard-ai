# Pitch-safe facts

## One-line problem
Quality deviations can be discovered after the process has already moved on; operators benefit from an earlier signal.

## One-line solution
FlotaGuard AI combines process measurements and recent quality history to forecast next-hour silica risk and present it as operator decision support.

## Why AI is intentional
- XGBoost regression: predicts next-hour silica as a continuous quality value.
- XGBoost classification: detects elevated probability of crossing the prototype quality threshold.
- K-Means: summarizes recurring process patterns without pretending the clusters are absolute safety states.

## Numbers you can defend
- Chronological test set: 819 future hourly windows.
- Regression R²: 0.5904.
- Regression MAE: 0.5444 percentage points.
- Alert precision: 0.7417.
- Alert recall: 0.7417.
- Alert F1: 0.7417.

## Smart answers to likely questions
**Why not claim the older R² 0.91?**  
The earlier exploratory notebook used a random split on time-correlated measurements. For the competition version, validation was deliberately made stricter with a future chronological holdout.

**Does it control the plant?**  
No. It is human-in-the-loop decision support. The operator remains responsible for process actions.

**Why 3.5% silica?**  
It is the prototype alert threshold used to demonstrate the workflow. A deployment must use the plant's validated quality specification and retrain/recalibrate the alert model.

**What makes this feasible?**  
The prototype has trained models, a Flask inference API, exact feature validation, held-out demo cases, and an operator dashboard. Production still requires live integration and plant validation.
