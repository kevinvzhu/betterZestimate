import pandas as pd

# Paths to your files
property_data_path = "data/processed_data.csv"
market_data_path = "data/market_data.csv"  # This should contain at least date plus unemployment_rate, mortgage_rate, zillow_index, case_shiller_index

# Load the property-level data
property_df = pd.read_csv(property_data_path, parse_dates=['date'])

# Load the market-level data
market_df = pd.read_csv(market_data_path, parse_dates=['date'])

# OPTIONAL: Ensure both dataframes are aligned by month
# Convert both to monthly frequency if needed
# property_df['year_month'] = property_df['date'].dt.to_period('M')
# market_df['year_month'] = market_df['date'].dt.to_period('M')

# For simplicity, assume daily or monthly matches exactly, so just merge on `date`
merged_df = pd.merge(property_df, market_df, on='date', how='left')

# Check columns exist
# The merged_df should now have columns like:
# unemployment_rate, mortgage_rate, zillow_index, case_shiller_index
required_cols = ['unemployment_rate', 'mortgage_rate', 'zillow_index', 'case_shiller_index']
missing_cols = [col for col in required_cols if col not in merged_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}. Please ensure market_data.csv contains them.")

# Handle missing values if any
merged_df = merged_df.ffill().bfill()

# Save the merged data
merged_output_path = "data/merged_data_with_macro.csv"
merged_df.to_csv(merged_output_path, index=False)
print(f"Merged data saved to {merged_output_path}")