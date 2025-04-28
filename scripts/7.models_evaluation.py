import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Ensure output directory exists
os.makedirs("outputs/evaluation", exist_ok=True)


def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
    }

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_dict, index=["Classification_Model"])
    metrics_df.to_csv(
        "outputs/evaluation/classification_metrics.csv", float_format="%.4f"
    )

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Classification Model")
    plt.tight_layout()
    plt.savefig("outputs/evaluation/confusion_matrix.png", bbox_inches="tight")
    plt.close()


def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_dict, index=["Regression_Model"])
    metrics_df.to_csv("outputs/evaluation/regression_metrics.csv", float_format="%.4f")


def evaluate_models():
    # Evaluate Risk Classification Model (Task 1)
    try:
        print("Evaluating Risk Classification Model (Task 1)...")
        rf_risk_model = joblib.load("models/risk_classifier_rf.pkl")
        X_test_risk = pd.read_csv("data/processed_data/task1_test_data.csv")
        y_test_risk = X_test_risk.pop("outcome")
        evaluate_classification_model(rf_risk_model, X_test_risk, y_test_risk)
        print("Classification evaluation saved.")
    except FileNotFoundError:
        print("Task 1 test data not found. Skipping Task 1 evaluation.")

    # Evaluate Future Revenue Regression Model (Task 2)
    try:
        print("\nEvaluating Future Revenue Prediction Model (Task 2)...")
        rf_revenue_model = joblib.load("models/future_revenue_rf_model.pkl")
        X_test_revenue = pd.read_csv("data/processed_data/task2_test_data.csv")
        y_test_revenue = X_test_revenue.pop("outcome")
        evaluate_regression_model(rf_revenue_model, X_test_revenue, y_test_revenue)
        print("Regression evaluation saved.")
    except FileNotFoundError:
        print("Task 2 test data not found. Skipping Task 2 evaluation.")


if __name__ == "__main__":
    evaluate_models()
