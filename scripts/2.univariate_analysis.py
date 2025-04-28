import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# Create the outputs directory if it doesn't exist
os.makedirs("outputs/visualizations", exist_ok=True)


def univariate_analysis(df):
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove ID and timestamp columns
    numerical_cols = [col for col in numerical_cols if col not in ["customer_id"]]
    categorical_cols = [col for col in categorical_cols if col not in ["timestamp"]]

    # Plot numerical variables
    if numerical_cols:
        num_plots = len(numerical_cols)
        num_rows = (num_plots + 1) // 2

        fig, axes = plt.subplots(num_rows, 2, figsize=(14, 5 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color="steelblue")
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.suptitle("Univariate Analysis - Numerical Variables", fontsize=16, y=1.02)

        # Save the numerical plot
        fig.savefig(
            "outputs/visualizations/univariate_numerical.png", bbox_inches="tight"
        )
        plt.close(fig)  # Close to avoid displaying during batch runs

    # Plot categorical variables
    if categorical_cols:
        cat_plots = len(categorical_cols)
        cat_rows = (cat_plots + 1) // 2

        fig2, axes2 = plt.subplots(cat_rows, 2, figsize=(14, 5 * cat_rows))
        axes2 = axes2.flatten()

        for i, col in enumerate(categorical_cols):
            sns.countplot(
                data=df,
                x=col,
                ax=axes2[i],
                palette="muted",
                order=df[col].value_counts().index,
            )
            axes2[i].set_title(f"Distribution of {col}")
            axes2[i].set_xlabel(col)
            axes2[i].set_ylabel("Count")
            axes2[i].tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes2)):
            fig2.delaxes(axes2[j])

        fig2.tight_layout()
        plt.suptitle("Univariate Analysis - Categorical Variables", fontsize=16, y=1.02)

        # Save the categorical plot
        fig2.savefig(
            "outputs/visualizations/univariate_categorical.png", bbox_inches="tight"
        )
        plt.close(fig2)


if __name__ == "__main__":
    df = pd.read_csv("data/processed_data/cleaned_merged_data.csv")
    univariate_analysis(df)
