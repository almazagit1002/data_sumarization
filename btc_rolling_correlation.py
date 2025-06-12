import pandas as pd
import os

# List of files to read
files = [
    "rolling_correlation_bonds_interest_combined_summary.csv",
    "rolling_correlation_commodities_combined_summary.csv",
    "rolling_correlation_financial_combined_summary.csv",
    "rolling_correlation_key_sectors_combined_summary.csv",
    "rolling_correlation_stocks_combined_summary.csv",
]

# Base path if these files are in a folder (edit if needed)
base_dir = "correlation_outputs"

# Store filtered DataFrames
btc_dataframes = []

for filename in files:
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    try:
        df = pd.read_csv(filepath)

        # Check for 'pair' column and filter rows containing 'BTC'
        if 'pair' in df.columns:
            btc_df = df[df['pair'].str.contains("BTC", case=False, na=False)]
            btc_dataframes.append(btc_df)
        else:
            print(f"'pair' column not found in {filename}, skipping.")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Combine and export
if btc_dataframes:
    combined_btc_df = pd.concat(btc_dataframes, ignore_index=True)
    output_path = os.path.join(base_dir, "rolling_correlation_BTC_summary_combined.csv")
    combined_btc_df.to_csv(output_path, index=False)
    print(f"Combined BTC correlations saved to: {output_path}")
else:
    print("No BTC data found in any file.")
