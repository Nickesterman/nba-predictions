"""
train_model.py
==============
Trains/loads the NBA prediction model and generates weekly_predictions.csv.

- Designed to run locally or via GitHub Actions (weekly).
- Tries NBAPredictor2025().predict_weekly_games() with and without a week_start,
  so it works with either method signature.
- Ensures the CSV has the columns the dashboard expects.
"""

from datetime import datetime, timedelta
import pandas as pd
import joblib
import os

from nba_model_2025_26 import NBAPredictor2025

# Season anchor: Oct 22, 2025 (opening night)
SEASON_START = datetime(2025, 10, 22)

# Columns the dashboard relies on
REQUIRED_COLS = [
    "date", "time", "away", "home", "predicted_winner",
    "team1_win_prob", "confidence", "spread", "expected_value"
]

def get_aligned_week_start(today: datetime) -> datetime:
    """Return the Monday-equivalent start aligned to SEASON_START ‘week 0’.
    Here we define a 'week' as 7-day windows starting SEASON_START."""
    if today < SEASON_START:
        return SEASON_START
    delta_days = (today - SEASON_START).days
    week_index = delta_days // 7
    return SEASON_START + timedelta(days=7 * week_index)

def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and types are reasonable."""
    # Make a copy so we don't mutate input
    df = df.copy()

    # Add any missing columns with defaults
    defaults = {
        "time": "TBD",
        "away": "Away",
        "home": "Home",
        "predicted_winner": "TBD",
        "team1_win_prob": 0.5,
        "confidence": "MEDIUM",
        "spread": 0.0,
        "expected_value": 0.0
    }

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = defaults.get(col, None)

    # Parse/format date
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date  # store as date only
    except Exception:
        # If date parse fails, use week start as a fallback
        df["date"] = pd.to_datetime(SEASON_START).date()

    # Normalize types / casing
    df["confidence"] = df["confidence"].astype(str).str.upper()
    numeric_cols = ["team1_win_prob", "spread", "expected_value"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df[REQUIRED_COLS]

def main():
    print("📊 Training/Loading NBA Prediction Model...")
    predictor = NBAPredictor2025()
    predictor.train_model()
    print("✅ Model training complete.")

    # Determine the aligned week start for upcoming predictions
    today = datetime.now()
    week_start = get_aligned_week_start(today)

    print(f"🔮 Generating predictions for week starting {week_start.date()}...")

    # Be tolerant of different method signatures
    try:
        predictions = predictor.predict_weekly_games()
    except TypeError:
        predictions = predictor.predict_weekly_games(week_start)

    # Convert to DataFrame and normalize shape
    predictions_df = pd.DataFrame(predictions)
    predictions_df = normalize_frame(predictions_df)

    # Save ISO format for 'date'
    predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m-%d")
    predictions_df.to_csv("weekly_predictions.csv", index=False)
    print(f"✅ Saved {len(predictions_df)} predictions to weekly_predictions.csv")

    # Optionally save the model (handy for debugging or local scripts)
    joblib.dump(predictor, "nba_model.pkl")
    print("✅ Model saved as nba_model.pkl")

if __name__ == "__main__":
    main()
