import pandas as pd
import os

# Create the outputs directory if it doesn't exist
os.makedirs("data/processed_data", exist_ok=True)


def clean_and_merge_data():
    # Loading the data
    bets_df = pd.read_csv("data/raw_data/CRM_handout_bets.csv")
    customers_df = pd.read_csv("data/raw_data/customer_details.csv")

    # Drop Unnamed index columns
    bets_df.drop(columns=["Unnamed: 0"], inplace=True)
    customers_df.drop(columns=["Unnamed: 0"], inplace=True)

    # Convert timestamp to datetime format
    bets_df["timestamp"] = pd.to_datetime(bets_df["timestamp"], errors="coerce")

    # Standardise gender values
    customers_df["gender"] = customers_df["gender"].replace(
        {
            "M": "Male",
            "F": "Female",
            "Unassigned": "Unknown",
            "U": "Unknown",
            "Null": "Unknown",
        }
    )

    # Standardise country names
    customers_df["country"] = customers_df["country"].replace(
        {
            "UK": "United Kingdom",
            "Britain": "United Kingdom",
            "United States of America": "United States",
        }
    )

    # Replace "Null" with "Unknown" in payment_method and currency columns
    customers_df["payment_method"] = customers_df["payment_method"].replace(
        {"Null": "Unknown"}
    )
    customers_df["currency"] = customers_df["currency"].replace({"Null": "Unknown"})

    # Rename customer_id in bets_df for clarity before merge
    bets_df = bets_df.rename(columns={"cust_id": "customer_id"})

    # Perform the merge
    merged_df = pd.merge(bets_df, customers_df, on="customer_id", how="left")

    # Save cleaned and merged data to CSV for reuse
    merged_df.to_csv("data/processed_data/cleaned_merged_data.csv", index=False)

    return merged_df


if __name__ == "__main__":
    clean_and_merge_data()
