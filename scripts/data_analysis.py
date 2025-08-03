import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv('housing_data.csv')

print("=== HOUSING PRICE PREDICTION - DATA ANALYSIS ===\n")

# Basic dataset information
print("1. DATASET OVERVIEW")
print("-" * 50)
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\nData types:")
print(df.dtypes)

# Statistical summary
print("\n2. STATISTICAL SUMMARY")
print("-" * 50)
print(df.describe())

# Price distribution
print("\n3. PRICE DISTRIBUTION ANALYSIS")
print("-" * 50)
print(f"Mean price: ${df['price'].mean():,.2f}")
print(f"Median price: ${df['price'].median():,.2f}")
print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
print(f"Standard deviation: ${df['price'].std():,.2f}")

# Check for outliers using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Correlation analysis
print("\n4. CORRELATION ANALYSIS")
print("-" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_with_price = df[numeric_cols].corr()['price'].sort_values(ascending=False)
print("Correlation with price:")
for col, corr in correlation_with_price.items():
    if col != 'price':
        print(f"{col:20}: {corr:6.3f}")

# Location analysis
print("\n5. LOCATION ANALYSIS")
print("-" * 50)
location_stats = df.groupby('location')['price'].agg(['count', 'mean', 'median', 'std']).round(2)
print(location_stats)

# Feature relationships
print("\n6. KEY FEATURE INSIGHTS")
print("-" * 50)

# Bedrooms vs Price
bedroom_stats = df.groupby('bedrooms')['price'].mean().sort_index()
print("Average price by bedrooms:")
for bedrooms, avg_price in bedroom_stats.items():
    print(f"{bedrooms} bedrooms: ${avg_price:,.0f}")

# Square footage impact
sqft_corr = df['square_feet'].corr(df['price'])
print(f"\nSquare footage correlation with price: {sqft_corr:.3f}")

# Crime rate impact
crime_corr = df['crime_rate'].corr(df['price'])
print(f"Crime rate correlation with price: {crime_corr:.3f}")

# School rating impact
school_corr = df['school_rating'].corr(df['price'])
print(f"School rating correlation with price: {school_corr:.3f}")

print("\n7. DATA QUALITY ASSESSMENT")
print("-" * 50)
print("Checking for data quality issues...")

# Check for unrealistic values
issues = []
if (df['bedrooms'] > 10).any():
    issues.append("Unrealistic bedroom counts detected")
if (df['square_feet'] < 200).any():
    issues.append("Unrealistically small houses detected")
if (df['age'] < 0).any():
    issues.append("Negative age values detected")

if issues:
    for issue in issues:
        print(f"⚠️  {issue}")
else:
    print("✅ No major data quality issues detected")

print("\n" + "="*60)
print("Data analysis complete! Ready for preprocessing and modeling.")
