import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_comprehensive_housing_dataset(n_samples=2000):
    """
    Generate a comprehensive and realistic housing dataset
    """
    print("🏠 Generating Comprehensive Housing Dataset...")
    print("=" * 50)
    
    # Define realistic location data with economic characteristics
    locations_data = {
        'Downtown': {
            'base_multiplier': 1.8,
            'crime_rate_range': (1.5, 4.0),
            'school_rating_range': (6.0, 9.0),
            'distance_to_city': (0.5, 3.0),
            'property_tax_multiplier': 1.3
        },
        'Suburbs': {
            'base_multiplier': 1.2,
            'crime_rate_range': (0.5, 2.5),
            'school_rating_range': (7.0, 9.5),
            'distance_to_city': (8.0, 25.0),
            'property_tax_multiplier': 1.1
        },
        'Waterfront': {
            'base_multiplier': 2.2,
            'crime_rate_range': (0.3, 1.8),
            'school_rating_range': (7.5, 9.8),
            'distance_to_city': (5.0, 30.0),
            'property_tax_multiplier': 1.5
        },
        'Rural': {
            'base_multiplier': 0.7,
            'crime_rate_range': (0.1, 1.2),
            'school_rating_range': (5.0, 7.5),
            'distance_to_city': (25.0, 60.0),
            'property_tax_multiplier': 0.8
        },
        'Industrial': {
            'base_multiplier': 0.6,
            'crime_rate_range': (2.0, 6.0),
            'school_rating_range': (4.0, 6.5),
            'distance_to_city': (3.0, 15.0),
            'property_tax_multiplier': 0.9
        },
        'Historic': {
            'base_multiplier': 1.4,
            'crime_rate_range': (1.0, 3.0),
            'school_rating_range': (6.5, 8.5),
            'distance_to_city': (2.0, 8.0),
            'property_tax_multiplier': 1.2
        }
    }
    
    # Generate base data
    data = {}
    
    # Location (affects many other features)
    locations = list(locations_data.keys())
    location_weights = [0.15, 0.35, 0.08, 0.20, 0.12, 0.10]  # Realistic distribution
    data['location'] = np.random.choice(locations, n_samples, p=location_weights)
    
    # Initialize other features
    data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    data['bathrooms'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.10, 0.40, 0.35, 0.12, 0.03])
    
    # Square footage (correlated with bedrooms)
    base_sqft = 800 + data['bedrooms'] * 300 + np.random.normal(0, 200, n_samples)
    data['square_feet'] = np.clip(base_sqft, 600, 6000).astype(int)
    
    # Lot size (varies by location)
    base_lot_size = np.random.normal(8000, 3000, n_samples)
    for i, location in enumerate(data['location']):
        if location == 'Downtown':
            base_lot_size[i] *= 0.6  # Smaller lots downtown
        elif location == 'Rural':
            base_lot_size[i] *= 2.0  # Larger lots in rural areas
        elif location == 'Waterfront':
            base_lot_size[i] *= 1.5  # Larger waterfront lots
    
    data['lot_size'] = np.clip(base_lot_size, 2000, 50000).astype(int)
    
    # House age
    data['age'] = np.random.exponential(15, n_samples).astype(int)
    data['age'] = np.clip(data['age'], 0, 100)
    
    # Year built (derived from age)
    current_year = datetime.now().year
    data['year_built'] = current_year - data['age']
    
    # Location-dependent features
    data['crime_rate'] = np.zeros(n_samples)
    data['school_rating'] = np.zeros(n_samples)
    data['distance_to_city'] = np.zeros(n_samples)
    data['property_tax'] = np.zeros(n_samples)
    
    for i, location in enumerate(data['location']):
        loc_info = locations_data[location]
        
        # Crime rate
        crime_min, crime_max = loc_info['crime_rate_range']
        data['crime_rate'][i] = np.random.uniform(crime_min, crime_max)
        
        # School rating
        school_min, school_max = loc_info['school_rating_range']
        data['school_rating'][i] = np.random.uniform(school_min, school_max)
        
        # Distance to city
        dist_min, dist_max = loc_info['distance_to_city']
        data['distance_to_city'][i] = np.random.uniform(dist_min, dist_max)
        
        # Property tax (base amount with location multiplier)
        base_tax = np.random.normal(4500, 1200)
        data['property_tax'][i] = base_tax * loc_info['property_tax_multiplier']
    
    # Clip values to realistic ranges
    data['crime_rate'] = np.clip(data['crime_rate'], 0.1, 10.0)
    data['school_rating'] = np.clip(data['school_rating'], 1.0, 10.0)
    data['distance_to_city'] = np.clip(data['distance_to_city'], 0.1, 80.0)
    data['property_tax'] = np.clip(data['property_tax'], 1000, 20000)
    
    # Garage spaces (correlated with house size and location)
    garage_prob = np.zeros((n_samples, 4))  # 0, 1, 2, 3+ spaces
    for i in range(n_samples):
        if data['location'][i] == 'Downtown':
            garage_prob[i] = [0.4, 0.3, 0.2, 0.1]  # Less parking downtown
        elif data['location'][i] == 'Rural':
            garage_prob[i] = [0.1, 0.2, 0.4, 0.3]  # More parking in rural
        else:
            garage_prob[i] = [0.2, 0.3, 0.4, 0.1]  # Standard distribution
    
    data['garage_spaces'] = np.array([np.random.choice([0, 1, 2, 3], p=garage_prob[i]) 
                                     for i in range(n_samples)])
    
    # Amenities (probability based on house value indicators)
    luxury_indicator = (data['square_feet'] > 2500) & (data['bedrooms'] >= 4)
    
    # Pool (more likely in warmer areas and luxury homes)
    pool_prob = np.where(luxury_indicator, 0.4, 0.15)
    pool_prob = np.where(np.array(data['location']) == 'Waterfront', pool_prob * 1.5, pool_prob)
    data['has_pool'] = np.random.binomial(1, pool_prob)
    
    # Fireplace (more common in older homes and luxury homes)
    fireplace_prob = np.where(luxury_indicator, 0.6, 0.3)
    fireplace_prob = np.where(data['age'] > 20, fireplace_prob * 1.3, fireplace_prob)
    data['has_fireplace'] = np.random.binomial(1, np.clip(fireplace_prob, 0, 1))
    
    # Air conditioning (more common in newer homes)
    ac_prob = np.where(data['age'] < 20, 0.9, 0.6)
    data['has_ac'] = np.random.binomial(1, ac_prob)
    
    # Basement (regional and age dependent)
    basement_prob = np.where(data['age'] > 30, 0.7, 0.4)
    basement_prob = np.where(np.array(data['location']) == 'Rural', basement_prob * 1.2, basement_prob)
    data['has_basement'] = np.random.binomial(1, np.clip(basement_prob, 0, 1))
    
    # Hardwood floors (more common in newer and luxury homes)
    hardwood_prob = np.where(luxury_indicator, 0.8, 0.4)
    hardwood_prob = np.where(data['age'] < 15, hardwood_prob * 1.3, hardwood_prob)
    data['has_hardwood_floors'] = np.random.binomial(1, np.clip(hardwood_prob, 0, 1))
    
    # Updated kitchen (more likely in newer homes or recently renovated)
    updated_kitchen_prob = np.where(data['age'] < 10, 0.8, 0.3)
    data['updated_kitchen'] = np.random.binomial(1, updated_kitchen_prob)
    
    # Updated bathroom (similar to kitchen)
    updated_bathroom_prob = np.where(data['age'] < 10, 0.7, 0.25)
    data['updated_bathroom'] = np.random.binomial(1, updated_bathroom_prob)
    
    # Walkability score (higher in urban areas)
    walkability_base = np.random.normal(50, 15, n_samples)
    for i, location in enumerate(data['location']):
        if location == 'Downtown':
            walkability_base[i] += 30
        elif location == 'Suburbs':
            walkability_base[i] += 10
        elif location == 'Rural':
            walkability_base[i] -= 20
        elif location == 'Industrial':
            walkability_base[i] -= 10
    
    data['walkability_score'] = np.clip(walkability_base, 0, 100).astype(int)
    
    # Nearby amenities count
    amenities_base = np.random.poisson(5, n_samples)
    for i, location in enumerate(data['location']):
        if location == 'Downtown':
            amenities_base[i] += 8
        elif location == 'Suburbs':
            amenities_base[i] += 3
        elif location == 'Rural':
            amenities_base[i] -= 3
    
    data['nearby_amenities'] = np.clip(amenities_base, 0, 20)
    
    # HOA fees (more common in suburbs and newer developments)
    hoa_prob = np.where(np.array(data['location']) == 'Suburbs', 0.6, 0.2)
    hoa_prob = np.where(data['age'] < 20, hoa_prob * 1.5, hoa_prob)
    has_hoa = np.random.binomial(1, np.clip(hoa_prob, 0, 1))
    
    hoa_fees = np.where(has_hoa, np.random.normal(150, 75), 0)
    data['hoa_monthly_fee'] = np.clip(hoa_fees, 0, 500)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Round numerical values appropriately
    df['crime_rate'] = df['crime_rate'].round(2)
    df['school_rating'] = df['school_rating'].round(1)
    df['distance_to_city'] = df['distance_to_city'].round(1)
    df['property_tax'] = df['property_tax'].round(0).astype(int)
    df['hoa_monthly_fee'] = df['hoa_monthly_fee'].round(0).astype(int)
    
    return df, locations_data

def calculate_realistic_prices(df, locations_data):
    """
    Calculate realistic house prices based on all features
    """
    print("\n💰 Calculating Realistic House Prices...")
    print("=" * 50)
    
    # Base price calculation
    base_price = 80000
    
    # Feature contributions to price
    price = np.full(len(df), base_price, dtype=float)
    
    # Square footage (major factor)
    price += df['square_feet'] * np.random.normal(85, 10, len(df))
    
    # Bedrooms and bathrooms
    price += df['bedrooms'] * np.random.normal(12000, 2000, len(df))
    price += df['bathrooms'] * np.random.normal(8000, 1500, len(df))
    
    # Lot size
    price += df['lot_size'] * np.random.normal(1.5, 0.3, len(df))
    
    # Age depreciation (non-linear)
    age_factor = np.where(df['age'] <= 5, 1.1,  # New homes premium
                 np.where(df['age'] <= 15, 1.0,   # Stable value
                 np.where(df['age'] <= 30, 0.95,  # Slight depreciation
                 np.where(df['age'] <= 50, 0.85,  # More depreciation
                          0.75))))              # Significant depreciation
    price *= age_factor
    
    # Location multipliers
    for i, location in enumerate(df['location']):
        loc_multiplier = locations_data[location]['base_multiplier']
        price[i] *= loc_multiplier
    
    # Crime rate impact (exponential decay)
    crime_impact = np.exp(-df['crime_rate'] / 3) * 0.3 + 0.7
    price *= crime_impact
    
    # School rating impact
    school_impact = (df['school_rating'] / 10) * 0.4 + 0.8
    price *= school_impact
    
    # Distance to city (closer is generally better, but not always)
    distance_impact = np.where(df['distance_to_city'] < 5, 1.1,  # Close to city premium
                      np.where(df['distance_to_city'] < 15, 1.0,  # Optimal distance
                      np.where(df['distance_to_city'] < 30, 0.95, # Slight penalty
                               0.85)))                            # Far penalty
    price *= distance_impact
    
    # Garage spaces
    price += df['garage_spaces'] * np.random.normal(8000, 1000, len(df))
    
    # Amenities
    price += df['has_pool'] * np.random.normal(25000, 5000, len(df))
    price += df['has_fireplace'] * np.random.normal(8000, 2000, len(df))
    price += df['has_ac'] * np.random.normal(5000, 1000, len(df))
    price += df['has_basement'] * np.random.normal(15000, 3000, len(df))
    price += df['has_hardwood_floors'] * np.random.normal(12000, 2000, len(df))
    price += df['updated_kitchen'] * np.random.normal(20000, 4000, len(df))
    price += df['updated_bathroom'] * np.random.normal(10000, 2000, len(df))
    
    # Walkability and amenities
    walkability_bonus = (df['walkability_score'] - 50) * 200
    price += walkability_bonus
    
    amenities_bonus = df['nearby_amenities'] * np.random.normal(2000, 500, len(df))
    price += amenities_bonus
    
    # HOA impact (can be positive or negative)
    hoa_impact = np.where(df['hoa_monthly_fee'] > 0, 
                         -df['hoa_monthly_fee'] * 50 + 10000,  # HOA adds value but monthly cost
                         0)
    price += hoa_impact
    
    # Property tax impact (higher taxes can indicate better services)
    tax_impact = (df['property_tax'] - 4500) * 2
    price += tax_impact
    
    # Add some realistic noise
    noise = np.random.normal(0, 15000, len(df))
    price += noise
    
    # Ensure realistic price ranges
    price = np.clip(price, 50000, 3000000)
    
    # Round to nearest thousand
    price = np.round(price / 1000) * 1000
    
    return price.astype(int)

def add_derived_features(df):
    """
    Add derived features that might be useful for ML models
    """
    print("\n🔧 Adding Derived Features...")
    print("=" * 50)
    
    # Price per square foot (will be calculated after price)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['sqft_per_room'] = (df['square_feet'] / df['total_rooms']).round(0)
    
    # Luxury score
    luxury_features = ['has_pool', 'has_fireplace', 'has_hardwood_floors', 
                      'updated_kitchen', 'updated_bathroom']
    df['luxury_score'] = df[luxury_features].sum(axis=1)
    
    # Age categories
    df['age_category'] = pd.cut(df['age'], 
                               bins=[0, 5, 15, 30, 50, 100], 
                               labels=['New', 'Recent', 'Mature', 'Old', 'Very Old'])
    
    # Size categories
    df['size_category'] = pd.cut(df['square_feet'], 
                                bins=[0, 1200, 2000, 3000, 6000], 
                                labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    # Crime level categories
    df['crime_level'] = pd.cut(df['crime_rate'], 
                              bins=[0, 1, 3, 6, 10], 
                              labels=['Very Low', 'Low', 'Medium', 'High'])
    
    # School quality categories
    df['school_quality'] = pd.cut(df['school_rating'], 
                                 bins=[0, 5, 7, 8.5, 10], 
                                 labels=['Poor', 'Average', 'Good', 'Excellent'])
    
    # Distance categories
    df['distance_category'] = pd.cut(df['distance_to_city'], 
                                    bins=[0, 5, 15, 30, 80], 
                                    labels=['Very Close', 'Close', 'Moderate', 'Far'])
    
    return df

def create_dataset_summary(df):
    """
    Create a comprehensive summary of the dataset
    """
    summary = {
        'dataset_info': {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'creation_date': datetime.now().isoformat(),
            'price_range': {
                'min': int(df['price'].min()),
                'max': int(df['price'].max()),
                'mean': int(df['price'].mean()),
                'median': int(df['price'].median()),
                'std': int(df['price'].std())
            }
        },
        'feature_summary': {},
        'location_analysis': {},
        'correlations': {}
    }
    
    # Feature summary
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            summary['feature_summary'][col] = {
                'type': 'numerical',
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        else:
            summary['feature_summary'][col] = {
                'type': 'categorical',
                'unique_values': df[col].nunique(),
                'categories': df[col].value_counts().to_dict()
            }
    
    # Location analysis
    location_stats = df.groupby('location')['price'].agg(['count', 'mean', 'median', 'std'])
    for location in location_stats.index:
        summary['location_analysis'][location] = {
            'count': int(location_stats.loc[location, 'count']),
            'avg_price': int(location_stats.loc[location, 'mean']),
            'median_price': int(location_stats.loc[location, 'median']),
            'price_std': int(location_stats.loc[location, 'std'])
        }
    
    # Top correlations with price
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['price'].sort_values(ascending=False)
    summary['correlations'] = {col: float(corr) for col, corr in correlations.items() 
                              if col != 'price' and abs(corr) > 0.1}
    
    return summary

def main():
    """
    Main function to generate the complete housing dataset
    """
    print("🏠 COMPREHENSIVE HOUSING DATASET GENERATOR")
    print("=" * 60)
    
    # Generate base dataset
    df, locations_data = generate_comprehensive_housing_dataset(2000)
    
    # Calculate realistic prices
    df['price'] = calculate_realistic_prices(df, locations_data)
    
    # Add derived features
    df = add_derived_features(df)
    
    # Add price per square foot (now that we have price)
    df['price_per_sqft'] = (df['price'] / df['square_feet']).round(2)
    
    # Reorder columns for better readability
    column_order = [
        'price', 'location', 'bedrooms', 'bathrooms', 'square_feet', 'lot_size',
        'age', 'year_built', 'crime_rate', 'school_rating', 'distance_to_city',
        'garage_spaces', 'has_pool', 'has_fireplace', 'has_ac', 'has_basement',
        'has_hardwood_floors', 'updated_kitchen', 'updated_bathroom',
        'walkability_score', 'nearby_amenities', 'property_tax', 'hoa_monthly_fee',
        'price_per_sqft', 'total_rooms', 'sqft_per_room', 'luxury_score',
        'age_category', 'size_category', 'crime_level', 'school_quality', 'distance_category'
    ]
    
    df = df[column_order]
    
    # Save dataset
    df.to_csv('comprehensive_housing_data.csv', index=False)
    
    # Create and save summary
    summary = create_dataset_summary(df)
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print(f"\n✅ Dataset Generated Successfully!")
    print("=" * 60)
    print(f"📊 Total Samples: {len(df):,}")
    print(f"📋 Total Features: {len(df.columns)}")
    print(f"💰 Price Range: ${df['price'].min():,} - ${df['price'].max():,}")
    print(f"📈 Average Price: ${df['price'].mean():,.0f}")
    print(f"📍 Locations: {', '.join(df['location'].unique())}")
    
    print(f"\n📁 Files Created:")
    print("   - comprehensive_housing_data.csv (main dataset)")
    print("   - dataset_summary.json (detailed summary)")
    
    print(f"\n🔍 Sample Data Preview:")
    print("-" * 60)
    print(df.head())
    
    print(f"\n📊 Dataset Statistics:")
    print("-" * 60)
    print(df.describe())
    
    return df, summary

if __name__ == "__main__":
    dataset, summary = main()
