import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# Ensure output directory exists
os.makedirs("outputs/visualizations", exist_ok=True)


def multivariate_analysis(df):
    # Identify column types
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_cols = [col for col in num_cols if col not in ["customer_id"]]
    cat_cols = [col for col in cat_cols if col not in ["timestamp"]]

    ### Correlation Heatmap (Numerical Features)
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="magma", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix – Numerical Features")
    plt.tight_layout()
    fig.savefig(
        "outputs/visualizations/multivariate_correlation_heatmap.png",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("data/processed_data/cleaned_merged_data.csv")
    multivariate_analysis(df)
