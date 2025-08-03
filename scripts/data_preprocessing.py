import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_housing_data():
    """Comprehensive data preprocessing for housing price prediction"""
    
    print("=== DATA PREPROCESSING ===\n")
    
    # Load data
    df = pd.read_csv('housing_data.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # 1. Handle missing values (if any)
    print("\n1. HANDLING MISSING VALUES")
    print("-" * 40)
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        # Fill missing values with appropriate strategies
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("✅ No missing values found")
    
    # 2. Feature Engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 40)
    
    # Create new features
    df['price_per_sqft'] = df['price'] / df['square_feet']
    df['rooms_total'] = df['bedrooms'] + df['bathrooms']
    df['age_category'] = pd.cut(df['age'], bins=[0, 5, 15, 30, 50], labels=['New', 'Recent', 'Mature', 'Old'])
    df['luxury_score'] = (df['has_pool'] + df['has_fireplace'] + (df['garage_spaces'] > 2).astype(int))
    df['sqft_per_room'] = df['square_feet'] / df['rooms_total']
    
    # Crime rate categories
    df['crime_level'] = pd.cut(df['crime_rate'], bins=[0, 1, 3, 6, 10], labels=['Low', 'Medium', 'High', 'Very High'])
    
    # School rating categories
    df['school_quality'] = pd.cut(df['school_rating'], bins=[0, 5, 7, 8.5, 10], labels=['Poor', 'Average', 'Good', 'Excellent'])
    
    print("✅ New features created:")
    new_features = ['price_per_sqft', 'rooms_total', 'age_category', 'luxury_score', 'sqft_per_room', 'crime_level', 'school_quality']
    for feature in new_features:
        print(f"   - {feature}")
    
    # 3. Encode categorical variables
    print("\n3. ENCODING CATEGORICAL VARIABLES")
    print("-" * 40)
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"✅ Dataset shape after encoding: {X_encoded.shape}")
    
    # 4. Handle outliers
    print("\n4. OUTLIER DETECTION AND HANDLING")
    print("-" * 40)
    
    # Detect outliers using IQR method for price
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (y < lower_bound) | (y > upper_bound)
    n_outliers = outlier_mask.sum()
    
    print(f"Outliers detected: {n_outliers} ({n_outliers/len(y)*100:.1f}%)")
    
    # Option to remove extreme outliers (top/bottom 1%)
    extreme_outliers = (y < y.quantile(0.01)) | (y > y.quantile(0.99))
    n_extreme = extreme_outliers.sum()
    
    if n_extreme > 0:
        print(f"Removing {n_extreme} extreme outliers...")
        X_encoded = X_encoded[~extreme_outliers]
        y = y[~extreme_outliers]
    
    # 5. Feature scaling
    print("\n5. FEATURE SCALING")
    print("-" * 40)
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✅ Features scaled using StandardScaler")
    
    # 6. Save preprocessed data and scaler
    print("\n6. SAVING PREPROCESSED DATA")
    print("-" * 40)
    
    # Save datasets
    X_train_scaled.to_csv('X_train.csv', index=False)
    X_test_scaled.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    # Save scaler and feature names
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save feature information
    feature_info = {
        'feature_names': X_train.columns.tolist(),
        'categorical_features': categorical_cols,
        'numerical_features': numerical_cols,
        'n_features': len(X_train.columns)
    }
    
    import json
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Preprocessed data saved:")
    print("   - X_train.csv, X_test.csv")
    print("   - y_train.csv, y_test.csv")
    print("   - scaler.pkl")
    print("   - feature_info.json")
    
    # 7. Final summary
    print("\n7. PREPROCESSING SUMMARY")
    print("-" * 40)
    print(f"Final feature count: {X_train_scaled.shape[1]}")
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Target variable range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    print(f"Target variable mean: ${y_train.mean():,.0f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_housing_data()
    print("\n" + "="*50)
    print("Preprocessing complete! Ready for model training.")
