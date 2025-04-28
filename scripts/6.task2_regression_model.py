import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings("ignore")

# Create the outputs directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def train_future_revenue_model(df):
   
   # Currency Conversion (Using estimated current rates)
    currency_rates = {
        'USD': 1,
        'GBP': 1.32,
        'TR': 0.026,
        'Unknown': 1,
    }

    df['stake_usd'] = df.apply(lambda row: row['stake'] * currency_rates.get(row['currency'], 1), axis=1)
    df['revenue_usd'] = df.apply(lambda row: row['revenue'] * currency_rates.get(row['currency'], 1), axis=1)
   
    # STEP 1: Data Preparation
    df = df.sort_values(by=['customer_id', 'timestamp'])
    df['bet_rank'] = df.groupby('customer_id').cumcount() + 1
    df['total_bets'] = df.groupby('customer_id')['bet_rank'].transform('max')

    # Split into historical (70%) and future (30%) based on bet rank
    df['set'] = np.where(df['bet_rank'] <= 0.7 * df['total_bets'], 'historical', 'future')

    # Filter out customers who have only historical or only future bets
    set_counts = df.groupby(['customer_id', 'set']).size().unstack(fill_value=0)
    valid_customers = set_counts[(set_counts['historical'] > 0) & (set_counts['future'] > 0)].index
    filtered_df = df[df['customer_id'].isin(valid_customers)]

    # Aggregate features from the historical period for each customer
    agg_df = filtered_df[filtered_df['set'] == 'historical'].groupby('customer_id').agg(
        total_bets=('stake_usd', 'count'),
        total_stake=('stake_usd', 'sum'),
        avg_stake=('stake_usd', 'mean'),
        win_rate=('outcome', 'mean'),
        total_revenue=('revenue_usd', 'sum'),
        avg_odd=('odd', 'mean'),
        std_stake=('stake_usd', 'std'),
        std_odd=('odd', 'std'),
    ).reset_index()

    # Fill missing std values (e.g., for single-bet users)
    agg_df['std_stake'].fillna(0, inplace=True)
    agg_df['std_odd'].fillna(0, inplace=True)

    # Create the target variable: total revenue in the future period
    future_revenue = filtered_df[filtered_df['set'] == 'future'].groupby('customer_id')['revenue_usd'].sum().reset_index()
    agg_df = pd.merge(agg_df, future_revenue, on='customer_id', how='left')
    agg_df.rename(columns={'revenue_usd': 'future_revenue'}, inplace=True)

    # STEP 2: Feature Engineering
    features = ['total_bets', 'total_stake', 'avg_stake', 'win_rate', 
                'total_revenue', 'avg_odd', 'std_stake', 'std_odd']

    X = agg_df[features]
    y = agg_df['future_revenue']

    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # STEP 3: Train Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # STEP 4: Save Model & Test Data
    joblib.dump(rf_model, 'models/future_revenue_rf_model.pkl')
    test_output = X_test.copy()
    test_output['outcome'] = y_test
    test_output.to_csv('data/processed_data/task2_test_data.csv', index=False)

    print("Model and test data saved successfully.")

if __name__ == "__main__":
    merged_df = pd.read_csv('data/processed_data/cleaned_merged_data.csv')
    train_future_revenue_model(merged_df)