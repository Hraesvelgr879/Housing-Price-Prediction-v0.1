import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def update_preprocessing_for_new_dataset():
    """
    Update the preprocessing pipeline to work with the comprehensive dataset
    """
    print("🔄 UPDATING ML PIPELINE FOR NEW DATASET")
    print("=" * 50)
    
    # Load the comprehensive dataset
    try:
        df = pd.read_csv('comprehensive_housing_data.csv')
        print(f"✅ Loaded comprehensive dataset: {df.shape}")
    except FileNotFoundError:
        print("❌ Comprehensive dataset not found. Please run create_housing_dataset.py first.")
        return
    
    print(f"\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Target variable: price")
    print(f"Price range: ${df['price'].min():,} - ${df['price'].max():,}")
    
    # Separate features and target
    target_col = 'price'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\n2. FEATURE PREPROCESSING")
    print("-" * 30)
    
    # Identify different types of features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Handle category features (convert to string first)
    category_features = []
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].astype(str)
            category_features.append(col)
    
    categorical_features.extend(category_features)
    
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Handle missing values (if any)
    print(f"\n3. HANDLING MISSING VALUES")
    print("-" * 30)
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count}")
        
        # Fill missing values
        for col in numeric_features:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_features:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        print("✅ No missing values found")
    
    # Encode categorical variables
    print(f"\n4. ENCODING CATEGORICAL VARIABLES")
    print("-" * 30)
    
    # One-hot encode categorical variables
    if categorical_features:
        print(f"One-hot encoding {len(categorical_features)} categorical features...")
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        print(f"Features after encoding: {X_encoded.shape[1]}")
    else:
        X_encoded = X.copy()
        print("No categorical features to encode")
    
    # Handle outliers in target variable
    print(f"\n5. OUTLIER HANDLING")
    print("-" * 30)
    
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (y < lower_bound) | (y > upper_bound)
    n_outliers = outlier_mask.sum()
    
    print(f"Outliers detected: {n_outliers} ({n_outliers/len(y)*100:.1f}%)")
    
    # Remove extreme outliers (top/bottom 1%)
    extreme_outliers = (y < y.quantile(0.01)) | (y > y.quantile(0.99))
    n_extreme = extreme_outliers.sum()
    
    if n_extreme > 0:
        print(f"Removing {n_extreme} extreme outliers...")
        X_encoded = X_encoded[~extreme_outliers]
        y = y[~extreme_outliers]
        print(f"Dataset shape after outlier removal: {X_encoded.shape}")
    
    # Split the data
    print(f"\n6. TRAIN-TEST SPLIT")
    print("-" * 30)
    
    # Create price bins for stratified split
    price_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, 
        stratify=price_bins
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training price range: ${y_train.min():,} - ${y_train.max():,}")
    print(f"Test price range: ${y_test.min():,} - ${y_test.max():,}")
    
    # Feature scaling
    print(f"\n7. FEATURE SCALING")
    print("-" * 30)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✅ Features scaled using StandardScaler")
    
    # Save processed data
    print(f"\n8. SAVING PROCESSED DATA")
    print("-" * 30)
    
    X_train_scaled.to_csv('X_train.csv', index=False)
    X_test_scaled.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    # Save the original unscaled data as well
    X_train.to_csv('X_train_unscaled.csv', index=False)
    X_test.to_csv('X_test_unscaled.csv', index=False)
    
    # Save scaler and feature information
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save feature information
    feature_info = {
        'original_features': feature_cols,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'encoded_features': X_train.columns.tolist(),
        'n_features_original': len(feature_cols),
        'n_features_encoded': len(X_train.columns),
        'target_variable': target_col,
        'preprocessing_steps': [
            'Missing value imputation',
            'One-hot encoding of categorical variables',
            'Outlier removal (extreme 1%)',
            'Stratified train-test split',
            'Standard scaling'
        ]
    }
    
    import json
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Processed data saved:")
    print("   - X_train.csv, X_test.csv (scaled features)")
    print("   - X_train_unscaled.csv, X_test_unscaled.csv (original features)")
    print("   - y_train.csv, y_test.csv (target values)")
    print("   - scaler.pkl (feature scaler)")
    print("   - feature_info.json (feature information)")
    
    # Final summary
    print(f"\n9. PREPROCESSING SUMMARY")
    print("-" * 30)
    print(f"Original features: {len(feature_cols)}")
    print(f"Final features: {X_train_scaled.shape[1]}")
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Target range: ${y_train.min():,} - ${y_train.max():,}")
    print(f"Average target: ${y_train.mean():,.0f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = update_preprocessing_for_new_dataset()
    print("\n" + "="*60)
    print("PREPROCESSING UPDATE COMPLETE!")
    print("You can now run the ML training pipeline with the new dataset.")
    print("="*60)
