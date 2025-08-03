import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def create_simple_housing_dataset(n_samples=1000):
    """
    Create a simple, clean housing dataset and save directly to CSV
    """
    print(f"🏠 Creating Simple Housing Dataset ({n_samples:,} samples)...")
    
    # Generate basic housing data
    data = {
        # Basic property features
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.25, 0.05]),
        'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.45, 0.35, 0.05]),
        'square_feet': np.random.normal(2000, 600, n_samples).astype(int),
        'lot_size': np.random.normal(8000, 2500, n_samples).astype(int),
        'age': np.random.exponential(20, n_samples).astype(int),
        
        # Location (simplified)
        'location': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples, p=[0.3, 0.5, 0.2]),
        
        # Neighborhood factors
        'crime_rate': np.random.exponential(2.5, n_samples),
        'school_rating': np.random.normal(7, 1.5, n_samples),
        'distance_to_city': np.random.exponential(12, n_samples),
        
        # Property features
        'garage_spaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.4, 0.1]),
        'has_pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'has_fireplace': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        
        # Financial
        'property_tax': np.random.normal(5000, 1500, n_samples)
    }
    
    # Clean up the data to realistic ranges
    data['square_feet'] = np.clip(data['square_feet'], 800, 5000)
    data['lot_size'] = np.clip(data['lot_size'], 3000, 20000)
    data['age'] = np.clip(data['age'], 0, 80)
    data['crime_rate'] = np.clip(data['crime_rate'], 0.1, 8.0).round(2)
    data['school_rating'] = np.clip(data['school_rating'], 1.0, 10.0).round(1)
    data['distance_to_city'] = np.clip(data['distance_to_city'], 0.5, 50.0).round(1)
    data['property_tax'] = np.clip(data['property_tax'], 1500, 12000).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("💰 Calculating house prices...")
    
    # Calculate realistic prices
    base_price = 100000
    
    # Price calculation with realistic relationships
    price = (
        base_price +
        df['square_feet'] * 75 +           # $75 per sq ft
        df['bedrooms'] * 15000 +           # $15k per bedroom
        df['bathrooms'] * 10000 +          # $10k per bathroom
        df['lot_size'] * 1.5 +             # $1.50 per sq ft of lot
        (80 - df['age']) * 800 +           # Depreciation
        (8 - df['crime_rate']) * 5000 +    # Crime impact
        df['school_rating'] * 8000 +       # School quality
        (50 - df['distance_to_city']) * 1500 +  # Distance penalty
        df['garage_spaces'] * 7000 +       # Garage value
        df['has_pool'] * 20000 +           # Pool bonus
        df['has_fireplace'] * 12000 -      # Fireplace bonus
        df['property_tax'] * 1.5           # Tax burden
    )
    
    # Apply location multipliers
    location_multipliers = {'Downtown': 1.4, 'Suburbs': 1.1, 'Rural': 0.8}
    for i, location in enumerate(df['location']):
        price.iloc[i] *= location_multipliers[location]
    
    # Add some realistic noise
    price += np.random.normal(0, 15000, n_samples)
    price = np.clip(price, 60000, 1500000)  # Reasonable price range
    
    df['price'] = price.round(-3).astype(int)  # Round to nearest thousand
    
    # Add a few derived features
    df['price_per_sqft'] = (df['price'] / df['square_feet']).round(2)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Reorder columns with price first
    column_order = [
        'price', 'location', 'bedrooms', 'bathrooms', 'square_feet', 'lot_size',
        'age', 'crime_rate', 'school_rating', 'distance_to_city',
        'garage_spaces', 'has_pool', 'has_fireplace', 'property_tax',
        'price_per_sqft', 'total_rooms'
    ]
    
    df = df[column_order]
    
    # Save to CSV
    filename = 'housing_data.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Dataset saved to '{filename}'")
    print(f"📊 Dataset summary:")
    print(f"   - Samples: {len(df):,}")
    print(f"   - Features: {len(df.columns)}")
    print(f"   - Price range: ${df['price'].min():,} - ${df['price'].max():,}")
    print(f"   - Average price: ${df['price'].mean():,.0f}")
    print(f"   - File size: {pd.read_csv(filename).memory_usage(deep=True).sum()/1024:.1f} KB")
    
    # Show sample data
    print(f"\n📋 Sample data:")
    print(df.head().to_string(index=False))
    
    return df

if __name__ == "__main__":
    dataset = create_simple_housing_dataset(1000)
    print(f"\n🚀 Dataset ready for machine learning!")
    print(f"Next step: Run 'python scripts/data_analysis.py' to analyze the data")
