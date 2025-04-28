import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import os

warnings.filterwarnings("ignore")

# Create the outputs directory if it doesn't exist
os.makedirs("models", exist_ok=True)


def train_risk_model(df):
    # STEP 1: Currency Conversion (Using estimated current rates)
    currency_rates = {"USD": 1, "GBP": 1.32, "TR": 0.026, "Unknown": 1}

    df["stake_usd"] = df.apply(
        lambda row: row["stake"] * currency_rates.get(row["currency"], 1), axis=1
    )
    df["revenue_usd"] = df.apply(
        lambda row: row["revenue"] * currency_rates.get(row["currency"], 1), axis=1
    )

    # STEP 2: Aggregate Customer Features
    agg_df = (
        df.groupby("customer_id")
        .agg(
            total_bets=("stake_usd", "count"),
            total_stake=("stake_usd", "sum"),
            avg_stake=("stake_usd", "mean"),
            win_rate=("outcome", "mean"),
            total_revenue=("revenue_usd", "sum"),
            avg_odd=("odd", "mean"),
            std_stake=("stake_usd", "std"),
            std_odd=("odd", "std"),
        )
        .reset_index()
    )

    agg_df["std_stake"].fillna(0, inplace=True)
    agg_df["std_odd"].fillna(0, inplace=True)

    # STEP 3: Define Risk Label
    conditions = (
        (agg_df["win_rate"] < 0.3)
        | (agg_df["std_stake"] > agg_df["avg_stake"])
        | (agg_df["total_revenue"] < 0)
    )
    agg_df["risky"] = np.where(conditions, 1, 0)

    # STEP 4: Prepare Data for Modeling
    features = [
        "total_bets",
        "total_stake",
        "avg_stake",
        "win_rate",
        "total_revenue",
        "avg_odd",
        "std_stake",
        "std_odd",
    ]
    X = agg_df[features]
    y = agg_df["risky"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # STEP 5: Hyperparameter Tuning with GridSearchCV (Regularisation-focused)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_

    print("Best hyperparameters selected:", grid_search.best_params_)

    # STEP 6: Save Test Data and Best Model
    test_output = X_test.copy()
    test_output["outcome"] = y_test
    test_output.to_csv("data/processed_data/task1_test_data.csv", index=False)

    joblib.dump(best_rf_model, "models/risk_classifier_rf.pkl")

    print("Best model and test data saved successfully.")


if __name__ == "__main__":
    merged_df = pd.read_csv("data/processed_data/cleaned_merged_data.csv")
    train_risk_model(merged_df)
