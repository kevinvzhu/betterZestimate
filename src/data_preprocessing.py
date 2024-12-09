import os
import json
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

def load_property_data(property_dir):
    """
    Load and parse JSON files containing property data.
    Extract key fields and also historical price points from 'homeValueChartData'.
    """
    all_files = glob(os.path.join(property_dir, "*.json"))
    records = []

    for f in all_files:
        with open(f, 'r') as infile:
            data = json.load(infile)
            # data is a list of properties
            for prop in data:
                zpid = prop.get('zpid')
                if zpid is None:
                    continue

                # Try datePostedString
                date_str = prop.get('datePostedString')
                # If not available, try dateSoldString
                if date_str is None:
                    date_str = prop.get('dateSoldString')

                # Parse date if possible
                date = None
                if date_str:
                    try:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass

                address = prop.get('address', {})
                city = address.get('city')
                state = address.get('state')
                zipcode = address.get('zipcode')

                price_info = prop.get('price', {})
                current_price = price_info.get('value')  # current listing price
                bedrooms = prop.get('bedrooms')
                bathrooms = prop.get('bathrooms')
                livingArea = prop.get('livingArea')
                propertyType = prop.get('propertyType')

                # Extract historical data from homeValueChartData if available
                hv_data = prop.get('homeValueChartData', [])

                if hv_data:
                    # hv_data is a list of series (e.g. "This home", "Sale", etc.)
                    for series in hv_data:
                        series_points = series.get('points', [])
                        for pt in series_points:
                            # pt has 'x' (timestamp ms) and 'y' (price)
                            # Convert x to datetime
                            timestamp = pt.get('x')
                            hist_price = pt.get('y')
                            if timestamp is not None and hist_price is not None:
                                dt = datetime.fromtimestamp(timestamp/1000.0, tz=datetime.timezone.utc)
                                # Create a record for each historical price point
                                if hist_price is not None and zpid is not None:
                                    records.append({
                                        'zpid': zpid,
                                        'date': dt,
                                        'city': city,
                                        'state': state,
                                        'zipcode': zipcode,
                                        'price': hist_price,
                                        'bedrooms': bedrooms,
                                        'bathrooms': bathrooms,
                                        'livingArea': livingArea,
                                        'propertyType': propertyType
                                    })
                else:
                    # If no historical data, at least record the current data if price available
                    if current_price is not None:
                        # If we have at least a date from posted or sold, use it
                        # if date is None, we can skip or assign a default date
                        if date is None:
                            # If no date at all, skip
                            continue
                        records.append({
                            'zpid': zpid,
                            'date': date,
                            'city': city,
                            'state': state,
                            'zipcode': zipcode,
                            'price': current_price,
                            'bedrooms': bedrooms,
                            'bathrooms': bathrooms,
                            'livingArea': livingArea,
                            'propertyType': propertyType
                        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    property_dir = "data/property_data"
    df = load_property_data(property_dir)

    # Drop rows with missing critical values
    df = df.dropna(subset=['zpid', 'price', 'date'])

    # Convert date to monthly period if desired
    df['year_month'] = df['date'].dt.to_period('M')

    # Optional: log-transform price
    # df['log_price'] = np.log1p(df['price'])

    output_path = 'data/processed_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data preprocessing done. '{output_path}' created.")