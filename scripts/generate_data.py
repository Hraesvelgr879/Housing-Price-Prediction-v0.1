import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_housing_data(n_samples=1000):
    """Generate synthetic housing data with realistic features"""
    
    # Location categories (affects price significantly)
    locations = ['Downtown', 'Suburbs', 'Rural', 'Waterfront', 'Industrial']
    location_multipliers = {'Downtown': 1.5, 'Suburbs': 1.2, 'Rural': 0.8, 'Waterfront': 1.8, 'Industrial': 0.7}
    
    # Generate base features
    data = {
        'location': np.random.choice(locations, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'square_feet': np.random.normal(2000, 500, n_samples),
        'lot_size': np.random.normal(8000, 2000, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'crime_rate': np.random.exponential(2, n_samples),  # Lower is better
        'school_rating': np.random.normal(7, 1.5, n_samples),  # 1-10 scale
        'distance_to_city': np.random.exponential(10, n_samples),  # Miles
        'garage_spaces': np.random.randint(0, 4, n_samples),
        'has_pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'has_fireplace': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'property_tax': np.random.normal(5000, 1500, n_samples)
    }
    
    # Ensure realistic constraints
    data['square_feet'] = np.clip(data['square_feet'], 800, 5000)
    data['lot_size'] = np.clip(data['lot_size'], 3000, 20000)
    data['crime_rate'] = np.clip(data['crime_rate'], 0.1, 10)
    data['school_rating'] = np.clip(data['school_rating'], 1, 10)
    data['distance_to_city'] = np.clip(data['distance_to_city'], 0.5, 50)
    data['property_tax'] = np.clip(data['property_tax'], 1000, 15000)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate realistic prices based on features
    base_price = 100000
    
    # Price calculation with realistic relationships
    price = (
        base_price +
        df['square_feet'] * 80 +  # $80 per sq ft
        df['bedrooms'] * 15000 +   # $15k per bedroom
        df['bathrooms'] * 10000 +  # $10k per bathroom
        df['lot_size'] * 2 +       # $2 per sq ft of lot
        (50 - df['age']) * 1000 +  # Depreciation
        (10 - df['crime_rate']) * 8000 +  # Crime impact
        df['school_rating'] * 12000 +     # School quality
        (50 - df['distance_to_city']) * 2000 +  # Distance penalty
        df['garage_spaces'] * 8000 +      # Garage value
        df['has_pool'] * 25000 +          # Pool bonus
        df['has_fireplace'] * 15000 -     # Fireplace bonus
        df['property_tax'] * 2            # Tax burden
    )
    
    # Apply location multipliers
    for i, location in enumerate(df['location']):
        price.iloc[i] *= location_multipliers[location]
    
    # Add some noise to make it more realistic
    price += np.random.normal(0, 20000, n_samples)
    price = np.clip(price, 50000, 2000000)  # Reasonable price range
    
    df['price'] = price.round(-3)  # Round to nearest thousand
    
    return df

# Generate the dataset
print("Generating housing dataset...")
housing_data = generate_housing_data(1000)

# Save to CSV
housing_data.to_csv('housing_data.csv', index=False)

print(f"Dataset generated with {len(housing_data)} samples")
print("\nDataset Info:")
print(housing_data.info())
print("\nFirst few rows:")
print(housing_data.head())
print(f"\nPrice statistics:")
print(housing_data['price'].describe())
