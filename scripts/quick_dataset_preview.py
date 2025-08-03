import pandas as pd
import numpy as np
import os

def preview_dataset(filename="housing_data.csv"):
    """
    Quick preview of the generated housing dataset
    """
    print(f"🔍 DATASET PREVIEW: {filename}")
    print("=" * 50)
    
    try:
        # Load dataset
        df = pd.read_csv(filename)
        
        # Basic info
        print(f"📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 File Size: {os.path.getsize(filename)/1024:.1f} KB")
        print(f"🧮 Memory Usage: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
        
        # Column information
        print(f"\n📋 COLUMNS ({len(df.columns)}):")
        print("-" * 30)
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            print(f"{i:2d}. {col:25} ({dtype:8}) - {unique_vals:4} unique values")
        
        # Data types summary
        print(f"\n📈 DATA TYPES SUMMARY:")
        print("-" * 30)
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{str(dtype):15}: {count:2d} columns")
        
        # Missing values check
        print(f"\n🔍 MISSING VALUES:")
        print("-" * 30)
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values found")
        else:
            for col, count in missing[missing > 0].items():
                print(f"{col}: {count} missing ({count/len(df)*100:.1f}%)")
        
        # Price statistics
        print(f"\n💰 PRICE STATISTICS:")
        print("-" * 30)
        price_stats = df['price'].describe()
        print(f"Min:      ${price_stats['min']:>10,.0f}")
        print(f"25%:      ${price_stats['25%']:>10,.0f}")
        print(f"Median:   ${price_stats['50%']:>10,.0f}")
        print(f"Mean:     ${price_stats['mean']:>10,.0f}")
        print(f"75%:      ${price_stats['75%']:>10,.0f}")
        print(f"Max:      ${price_stats['max']:>10,.0f}")
        print(f"Std Dev:  ${price_stats['std']:>10,.0f}")
        
        # Location distribution
        print(f"\n📍 LOCATION DISTRIBUTION:")
        print("-" * 30)
        location_counts = df['location'].value_counts()
        for location, count in location_counts.items():
            pct = count / len(df) * 100
            avg_price = df[df['location'] == location]['price'].mean()
            print(f"{location:12}: {count:4d} ({pct:5.1f}%) - Avg: ${avg_price:,.0f}")
        
        # Sample data
        print(f"\n📋 SAMPLE DATA (First 5 rows):")
        print("-" * 30)
        # Show key columns for preview
        key_cols = ['price', 'location', 'bedrooms', 'bathrooms', 'square_feet', 'age', 'crime_rate', 'school_rating']
        available_cols = [col for col in key_cols if col in df.columns]
        print(df[available_cols].head().to_string(index=False))
        
        # Feature correlations with price
        print(f"\n📊 TOP CORRELATIONS WITH PRICE:")
        print("-" * 30)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['price'].sort_values(ascending=False)
        
        print("Positive correlations:")
        pos_corrs = correlations[correlations > 0.1].head(8)
        for feature, corr in pos_corrs.items():
            if feature != 'price':
                print(f"  {feature:25}: {corr:6.3f}")
        
        print("\nNegative correlations:")
        neg_corrs = correlations[correlations < -0.1].head(5)
        for feature, corr in neg_corrs.items():
            print(f"  {feature:25}: {corr:6.3f}")
        
        # Amenities summary
        amenity_cols = [col for col in df.columns if col.startswith('has_')]
        if amenity_cols:
            print(f"\n🏡 AMENITIES SUMMARY:")
            print("-" * 30)
            for amenity in amenity_cols:
                count = df[amenity].sum()
                pct = count / len(df) * 100
                print(f"{amenity.replace('has_', '').replace('_', ' ').title():20}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n✅ Dataset preview complete!")
        print(f"🚀 Ready for machine learning pipeline!")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found.")
        print("Please run 'python scripts/generate_housing_dataset.py' first.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def compare_datasets():
    """
    Compare multiple dataset files if they exist
    """
    dataset_files = [
        'housing_data_small.csv',
        'housing_data.csv', 
        'housing_data_large.csv'
    ]
    
    existing_files = [f for f in dataset_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No dataset files found.")
        return
    
    print(f"📊 DATASET COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    
    for filename in existing_files:
        try:
            df = pd.read_csv(filename)
            file_size_kb = os.path.getsize(filename) / 1024
            
            comparison_data.append({
                'File': filename,
                'Rows': f"{len(df):,}",
                'Columns': len(df.columns),
                'Size (KB)': f"{file_size_kb:.1f}",
                'Price Range': f"${df['price'].min():,} - ${df['price'].max():,}",
                'Avg Price': f"${df['price'].mean():,.0f}"
            })
        except Exception as e:
            print(f"❌ Error reading {filename}: {str(e)}")
    
    if comparison_data:
        # Create comparison table
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        print(f"\n💡 Recommendations:")
        print("- Use housing_data_small.csv for quick testing and development")
        print("- Use housing_data.csv for standard ML training")
        print("- Use housing_data_large.csv for advanced analysis and better model performance")

if __name__ == "__main__":
    # Preview the main dataset
    df = preview_dataset("housing_data.csv")
    
    # If main dataset doesn't exist, try others
    if df is None:
        for filename in ["housing_data_small.csv", "housing_data_large.csv"]:
            if os.path.exists(filename):
                print(f"\nTrying {filename}...")
                df = preview_dataset(filename)
                break
    
    # Compare all available datasets
    print(f"\n" + "="*60)
    compare_datasets()
