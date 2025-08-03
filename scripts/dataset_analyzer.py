import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def analyze_housing_dataset():
    """
    Comprehensive analysis of the generated housing dataset
    """
    print("🔍 HOUSING DATASET ANALYSIS")
    print("=" * 50)
    
    # Load dataset
    try:
        df = pd.read_csv('comprehensive_housing_data.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Dataset not found. Please run create_housing_dataset.py first.")
        return
    
    # Basic information
    print(f"\n1. BASIC DATASET INFORMATION")
    print("-" * 40)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Price analysis
    print(f"\n2. PRICE ANALYSIS")
    print("-" * 40)
    price_stats = df['price'].describe()
    print(f"Price Statistics:")
    for stat, value in price_stats.items():
        print(f"  {stat}: ${value:,.0f}")
    
    # Price distribution by location
    print(f"\n3. PRICE BY LOCATION")
    print("-" * 40)
    location_analysis = df.groupby('location')['price'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    
    for location in location_analysis.index:
        stats = location_analysis.loc[location]
        print(f"\n{location}:")
        print(f"  Count: {stats['count']}")
        print(f"  Avg Price: ${stats['mean']:,.0f}")
        print(f"  Median: ${stats['median']:,.0f}")
        print(f"  Range: ${stats['min']:,.0f} - ${stats['max']:,.0f}")
    
    # Feature correlations
    print(f"\n4. FEATURE CORRELATIONS WITH PRICE")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['price'].sort_values(ascending=False)
    
    print("Top positive correlations:")
    positive_corrs = correlations[correlations > 0.1].head(10)
    for feature, corr in positive_corrs.items():
        if feature != 'price':
            print(f"  {feature:25}: {corr:6.3f}")
    
    print("\nTop negative correlations:")
    negative_corrs = correlations[correlations < -0.1].head(5)
    for feature, corr in negative_corrs.items():
        print(f"  {feature:25}: {corr:6.3f}")
    
    # Categorical feature analysis
    print(f"\n5. CATEGORICAL FEATURES ANALYSIS")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            for value, count in value_counts.head().items():
                pct = (count / len(df)) * 100
                print(f"  {value}: {count} ({pct:.1f}%)")
    
    # Feature ranges and distributions
    print(f"\n6. NUMERICAL FEATURE RANGES")
    print("-" * 40)
    key_features = ['square_feet', 'lot_size', 'age', 'crime_rate', 'school_rating', 
                   'distance_to_city', 'walkability_score', 'property_tax']
    
    for feature in key_features:
        if feature in df.columns:
            stats = df[feature].describe()
            print(f"\n{feature}:")
            print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
            print(f"  Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
            print(f"  Median: {stats['50%']:.1f}")
    
    # Amenities analysis
    print(f"\n7. AMENITIES ANALYSIS")
    print("-" * 40)
    amenity_cols = ['has_pool', 'has_fireplace', 'has_ac', 'has_basement', 
                   'has_hardwood_floors', 'updated_kitchen', 'updated_bathroom']
    
    for amenity in amenity_cols:
        if amenity in df.columns:
            count = df[amenity].sum()
            pct = (count / len(df)) * 100
            avg_price_with = df[df[amenity] == 1]['price'].mean()
            avg_price_without = df[df[amenity] == 0]['price'].mean()
            price_diff = avg_price_with - avg_price_without
            
            print(f"\n{amenity.replace('_', ' ').title()}:")
            print(f"  Homes with feature: {count} ({pct:.1f}%)")
            print(f"  Avg price with: ${avg_price_with:,.0f}")
            print(f"  Avg price without: ${avg_price_without:,.0f}")
            print(f"  Price difference: ${price_diff:,.0f}")
    
    # Data quality checks
    print(f"\n8. DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    quality_issues = []
    
    # Check for unrealistic values
    if (df['bedrooms'] > 8).any():
        quality_issues.append("Some houses have >8 bedrooms")
    
    if (df['bathrooms'] > 6).any():
        quality_issues.append("Some houses have >6 bathrooms")
    
    if (df['square_feet'] < 400).any():
        quality_issues.append("Some houses are <400 sq ft")
    
    if (df['age'] < 0).any():
        quality_issues.append("Some houses have negative age")
    
    if (df['price'] < 30000).any():
        quality_issues.append("Some houses are priced <$30,000")
    
    # Check for logical inconsistencies
    if (df['bathrooms'] > df['bedrooms'] + 2).any():
        quality_issues.append("Some houses have too many bathrooms relative to bedrooms")
    
    if (df['square_feet'] < df['bedrooms'] * 200).any():
        quality_issues.append("Some houses are too small for the number of bedrooms")
    
    if quality_issues:
        print("⚠️  Potential quality issues found:")
        for issue in quality_issues:
            print(f"   - {issue}")
    else:
        print("✅ No major data quality issues detected")
    
    # Summary statistics
    print(f"\n9. SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Most expensive location: {df.groupby('location')['price'].mean().idxmax()}")
    print(f"Least expensive location: {df.groupby('location')['price'].mean().idxmin()}")
    print(f"Average price per sq ft: ${df['price_per_sqft'].mean():.2f}")
    print(f"Most common bedroom count: {df['bedrooms'].mode().iloc[0]}")
    print(f"Most common bathroom count: {df['bathrooms'].mode().iloc[0]}")
    print(f"Average house age: {df['age'].mean():.1f} years")
    print(f"Homes with pools: {df['has_pool'].sum()} ({df['has_pool'].mean()*100:.1f}%)")
    print(f"Homes with fireplaces: {df['has_fireplace'].sum()} ({df['has_fireplace'].mean()*100:.1f}%)")
    
    return df

def create_dataset_visualizations(df):
    """
    Create comprehensive visualizations of the dataset
    """
    print(f"\n🎨 CREATING DATASET VISUALIZATIONS")
    print("-" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Price distribution
    plt.subplot(3, 4, 1)
    plt.hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Price Distribution', fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.ticklabel_format(style='plain', axis='x')
    
    # 2. Price by location
    plt.subplot(3, 4, 2)
    location_prices = df.groupby('location')['price'].mean().sort_values()
    bars = plt.barh(range(len(location_prices)), location_prices.values, color='lightcoral')
    plt.yticks(range(len(location_prices)), location_prices.index)
    plt.title('Average Price by Location', fontweight='bold')
    plt.xlabel('Average Price ($)')
    plt.ticklabel_format(style='plain', axis='x')
    
    # 3. Square feet vs Price
    plt.subplot(3, 4, 3)
    plt.scatter(df['square_feet'], df['price'], alpha=0.5, s=20)
    plt.xlabel('Square Feet')
    plt.ylabel('Price ($)')
    plt.title('Price vs Square Feet', fontweight='bold')
    plt.ticklabel_format(style='plain')
    
    # 4. Age vs Price
    plt.subplot(3, 4, 4)
    plt.scatter(df['age'], df['price'], alpha=0.5, s=20, color='green')
    plt.xlabel('Age (years)')
    plt.ylabel('Price ($)')
    plt.title('Price vs Age', fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 5. Bedrooms distribution
    plt.subplot(3, 4, 5)
    bedroom_counts = df['bedrooms'].value_counts().sort_index()
    plt.bar(bedroom_counts.index, bedroom_counts.values, color='orange', alpha=0.7)
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Count')
    plt.title('Bedroom Distribution', fontweight='bold')
    
    # 6. Crime rate vs Price
    plt.subplot(3, 4, 6)
    plt.scatter(df['crime_rate'], df['price'], alpha=0.5, s=20, color='red')
    plt.xlabel('Crime Rate')
    plt.ylabel('Price ($)')
    plt.title('Price vs Crime Rate', fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 7. School rating vs Price
    plt.subplot(3, 4, 7)
    plt.scatter(df['school_rating'], df['price'], alpha=0.5, s=20, color='purple')
    plt.xlabel('School Rating')
    plt.ylabel('Price ($)')
    plt.title('Price vs School Rating', fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 8. Amenities impact
    plt.subplot(3, 4, 8)
    amenities = ['has_pool', 'has_fireplace', 'has_ac', 'has_basement']
    price_diffs = []
    labels = []
    
    for amenity in amenities:
        if amenity in df.columns:
            with_amenity = df[df[amenity] == 1]['price'].mean()
            without_amenity = df[df[amenity] == 0]['price'].mean()
            price_diffs.append(with_amenity - without_amenity)
            labels.append(amenity.replace('has_', '').replace('_', ' ').title())
    
    bars = plt.bar(labels, price_diffs, color='gold', alpha=0.7)
    plt.title('Price Impact of Amenities', fontweight='bold')
    plt.ylabel('Price Difference ($)')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 9. Correlation heatmap (top features)
    plt.subplot(3, 4, 9)
    numeric_cols = ['price', 'square_feet', 'bedrooms', 'bathrooms', 'age', 
                   'crime_rate', 'school_rating', 'luxury_score']
    corr_matrix = df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlations', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 10. Price per sqft by location
    plt.subplot(3, 4, 10)
    df.boxplot(column='price_per_sqft', by='location', ax=plt.gca())
    plt.title('Price per Sq Ft by Location', fontweight='bold')
    plt.xlabel('Location')
    plt.ylabel('Price per Sq Ft ($)')
    plt.xticks(rotation=45)
    plt.suptitle('')  # Remove automatic title
    
    # 11. Luxury score distribution
    plt.subplot(3, 4, 11)
    luxury_counts = df['luxury_score'].value_counts().sort_index()
    plt.bar(luxury_counts.index, luxury_counts.values, color='magenta', alpha=0.7)
    plt.xlabel('Luxury Score')
    plt.ylabel('Count')
    plt.title('Luxury Score Distribution', fontweight='bold')
    
    # 12. Distance to city vs Price
    plt.subplot(3, 4, 12)
    plt.scatter(df['distance_to_city'], df['price'], alpha=0.5, s=20, color='brown')
    plt.xlabel('Distance to City (miles)')
    plt.ylabel('Price ($)')
    plt.title('Price vs Distance to City', fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('housing_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'housing_dataset_analysis.png'")

def main():
    """
    Main function to analyze the housing dataset
    """
    df = analyze_housing_dataset()
    if df is not None:
        create_dataset_visualizations(df)
        
        print(f"\n" + "="*60)
        print("DATASET ANALYSIS COMPLETE!")
        print("="*60)
        print("Files created:")
        print("- housing_dataset_analysis.png (comprehensive visualizations)")
        print("\nThe dataset is ready for machine learning!")

if __name__ == "__main__":
    main()
