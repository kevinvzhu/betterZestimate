import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import json
import numpy as np

def preprocess_data(file_path):
    # Load data
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    # Relevant columns
    selected_features = [
        'city', 'state', 'bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
        'zipcode', 'homeType', 'propertyTaxRate', 'daysOnZillow',
        'tourViewCount', 'taxAssessedValue', 'price', 'zestimate', 'country'
    ]

    # Extract only relevant columns
    df = df[selected_features]

    # Replace missing prices with zestimate if available
    df['price'] = df.apply(
        lambda row: row['zestimate'] if pd.isna(row['price']) and not pd.isna(row['zestimate']) else row['price'],
        axis=1
    )

    # Remove rows where price is still missing
    df = df.dropna(subset=['price'])

    # Fill missing values
    df = df.fillna({
        'city': 'unknown',
        'state': 'unknown',
        'country': 'unknown',
        'zipcode': 'unknown',
        'homeType': 'unknown',
        'propertyTaxRate': -1,
        'bedrooms': -1,  # Replace missing bedrooms with -1
        'bathrooms': -1,  # Replace missing bathrooms with -1
        'livingArea': df['livingArea'].median(),  # Replace with median
        'yearBuilt': df['yearBuilt'].median(),  # Replace with median
        'daysOnZillow': -1,  # Replace with -1
        'tourViewCount': -1,  # Replace with -1
        'taxAssessedValue': df['taxAssessedValue'].median(),  # Replace with median
    })

    # Fill missing categorical features with "unknown"
    categorical_features = ['city', 'state', 'zipcode', 'homeType', 'country']
    df[categorical_features] = df[categorical_features].fillna('unknown')

    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    available_categorical_features = [col for col in categorical_features if col in df.columns]
    encoded = encoder.fit_transform(df[available_categorical_features])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(available_categorical_features))

    # Concatenate the one-hot encoded columns with the DataFrame
    df = pd.concat([df.drop(columns=available_categorical_features), encoded_df], axis=1)

    # Normalize numeric features
    numeric_features = ['bedrooms', 'bathrooms', 'livingArea', 'yearBuilt',
                        'propertyTaxRate', 'daysOnZillow', 'tourViewCount', 'taxAssessedValue', 'price']
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Check for remaining NaNs and fill them with a default value
    if df.isnull().values.any():
        print("Data still contains NaN values after processing!")
        print(df[df.isnull().any(axis=1)])
        # Fill remaining NaNs with a default value, e.g., 0 for numeric and 'unknown' for categorical
        df = df.fillna({
            col: 0 if df[col].dtype in [np.float64, np.int64] else 'unknown'
            for col in df.columns
        })

    # Remove rows with any NaN values
    df = df.dropna()

    # Set price_min and price_max for use in the environment
    price_min = df['price'].min()
    price_max = df['price'].max()

    # Save processed data for debugging
    df.to_csv('/Users/KZJer/Documents/GitHub/betterZestimate/src/preprocessing/processed_debug.csv', index=False)
    print("Processed data saved to processed_debug.csv")

    return df, scaler, encoder, price_min, price_max