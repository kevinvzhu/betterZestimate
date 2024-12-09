import os
import json
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

def load_property_data(property_dir):
    """
    Load and parse JSON files containing property data.
    Extract key fields: zpid, datePostedString as 'date', address info,
    price, bedrooms, bathrooms, livingArea, propertyType, city, state, zipcode.
    """
    all_files = glob(os.path.join(property_dir, "*.json"))
    records = []
    
    for f in all_files:
        with open(f, 'r') as infile:
            data = json.load(infile)
            # 'data' is a list of property objects
            for prop in data:
                zpid = prop.get('zpid')
                # Date is stored in 'datePostedString'
                date_str = prop.get('datePostedString')  # e.g. "2024-12-08"
                date = None
                if date_str:
                    # Attempt to parse date; if fails, date stays None
                    try:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        # If parsing fails, you might log this or skip the record
                        pass

                address = prop.get('address', {})
                city = address.get('city')
                state = address.get('state')
                zipcode = address.get('zipcode')

                # Price value is in price.value
                price_info = prop.get('price', {})
                price = price_info.get('value')

                bedrooms = prop.get('bedrooms')
                bathrooms = prop.get('bathrooms')
                livingArea = prop.get('livingArea')
                propertyType = prop.get('propertyType')

                # Only append if we have a zpid and price, for example:
                if zpid is not None and price is not None:
                    records.append({
                        'zpid': zpid,
                        'date': date,
                        'city': city,
                        'state': state,
                        'zipcode': zipcode,
                        'price': price,
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'livingArea': livingArea,
                        'propertyType': propertyType
                    })
    return pd.DataFrame(records)


if __name__ == "__main__":
    property_dir = "data/property_data"  # Update if needed
    df = load_property_data(property_dir)

    # Drop rows with missing critical values (price, zpid, etc.)
    df = df.dropna(subset=['zpid', 'price'])

    # If date is important and might be missing, consider dropping if missing:
    # df = df.dropna(subset=['date'])

    # Convert to a monthly period if desired:
    # df['year_month'] = df['date'].dt.to_period('M')

    # Additional data cleaning / normalization can go here:
    # For example, log-transform price:
    # df['log_price'] = np.log1p(df['price'])

    # Save the cleaned data
    output_path = 'data/processed_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data preprocessing done. '{output_path}' created.")