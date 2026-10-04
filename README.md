# FlotaGuard AI

### One-Hour-Ahead Flotation Quality Early-Warning System

FlotaGuard AI is an AI-powered decision-support prototype designed for mineral flotation processes.

It uses flotation process measurements and recent quality history to forecast next-hour silica quality and estimate the probability of a quality-risk condition, giving operators an earlier decision window before the next quality result arrives.

## Key Features

- One-hour-ahead silica quality forecasting
- Quality-risk probability estimation
- Operating-pattern identification using clustering
- Interactive process simulation scenarios
- Feature-importance visualization
- Model validation dashboard
- Human-in-the-loop decision support

## AI Architecture

FlotaGuard AI combines:

- **Regression** for next-hour silica forecasting
- **Classification** for quality-risk detection
- **K-Means clustering** for operating-pattern analysis

## Model Validation

The models were evaluated using a chronological future holdout rather than randomly mixing past and future observations.

- Forecast R²: **0.590**
- Alert F1 Score: **0.742**
- Forecast Horizon: **+1 hour**
- Model Inputs: **27 process features**

## Technology Stack

- Python
- Flask
- XGBoost
- Random Forest
- K-Means
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript

## Project Structure

```text
flotaguard-ai/
├── app.py
├── models/
├── static/
├── templates/
├── requirements.txt
├── train_models.py
├── MODEL_CARD.md
└── PITCH_FACTS.md
