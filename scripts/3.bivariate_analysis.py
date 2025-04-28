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


def bivariate_analysis(df):
    # Identify column types
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove unnecessary columns
    num_cols = [col for col in num_cols if col not in ["customer_id"]]
    cat_cols = [col for col in cat_cols if col not in ["timestamp"]]

    ### 1. Numerical vs Numerical
    num_pairs = [("stake", "revenue"), ("odd", "revenue"), ("stake", "odd")]
    nrows = (len(num_pairs) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    axes = axes.flatten()

    for i, (x, y) in enumerate(num_pairs):
        sns.scatterplot(data=df, x=x, y=y, ax=axes[i], alpha=0.3)
        sns.regplot(data=df, x=x, y=y, ax=axes[i], scatter=False, color="red")
        axes[i].set_title(f"{y} vs {x}")
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.suptitle("Bivariate Analysis – Numerical vs Numerical", fontsize=16, y=1.02)
    fig.savefig(
        "outputs/visualizations/bivariate_numerical_vs_numerical.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    ### 2. Numerical vs Categorical
    target = "revenue"
    cat_targets = ["gender", "country", "currency"]
    nrows2 = (len(cat_targets) + 1) // 2
    fig2, axes2 = plt.subplots(nrows2, 2, figsize=(14, 5 * nrows2))
    axes2 = axes2.flatten()

    for i, cat in enumerate(cat_targets):
        sns.boxplot(data=df, x=cat, y=target, ax=axes2[i])
        axes2[i].set_title(f"{target} by {cat}")
        axes2[i].set_xlabel(cat)
        axes2[i].set_ylabel(target)
        axes2[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes2)):
        fig2.delaxes(axes2[j])

    fig2.tight_layout()
    plt.suptitle(
        "Bivariate Analysis – Revenue by Categorical Features", fontsize=16, y=1.02
    )
    fig2.savefig(
        "outputs/visualizations/bivariate_numerical_vs_categorical.png",
        bbox_inches="tight",
    )
    plt.close(fig2)

    ### 3. Categorical vs Categorical
    cat_pairs = [("gender", "outcome"), ("country", "outcome")]
    nrows3 = (len(cat_pairs) + 1) // 2
    fig3, axes3 = plt.subplots(nrows3, 2, figsize=(14, 5 * nrows3))
    axes3 = axes3.flatten()

    for i, (cat1, cat2) in enumerate(cat_pairs):
        ct = pd.crosstab(df[cat1], df[cat2], normalize="index")
        sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", ax=axes3[i])
        axes3[i].set_title(f"{cat2} distribution by {cat1}")
        axes3[i].set_xlabel(cat2)
        axes3[i].set_ylabel(cat1)

    for j in range(i + 1, len(axes3)):
        fig3.delaxes(axes3[j])

    fig3.tight_layout()
    plt.suptitle(f"Bivariate Analysis – {cat1} by {cat2} Features", fontsize=16, y=1.02)
    fig3.savefig(
        "outputs/visualizations/bivariate_categorical_vs_categorical.png",
        bbox_inches="tight",
    )
    plt.close(fig3)


if __name__ == "__main__":
    df = pd.read_csv("data/processed_data/cleaned_merged_data.csv")
    bivariate_analysis(df)
