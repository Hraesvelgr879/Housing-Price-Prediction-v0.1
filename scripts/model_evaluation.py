import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json

def evaluate_model_performance():
    """Comprehensive model evaluation with visualizations"""
    
    print("=== MODEL EVALUATION AND ANALYSIS ===\n")
    
    # Load data and models
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').squeeze()
    
    # Load best model and metadata
    best_model = joblib.load('best_model.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Evaluating best model: {metadata['best_model']}")
    print(f"Test R²: {metadata['best_test_r2']:.4f}")
    print(f"Test RMSE: ${metadata['best_test_rmse']:,.0f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate detailed metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    residuals = y_test - y_pred
    
    print("\n" + "="*50)
    print("DETAILED PERFORMANCE METRICS")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error: ${rmse:,.0f}")
    print(f"Mean Absolute Error: ${mae:,.0f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Residuals Mean: ${residuals.mean():,.0f}")
    print(f"Residuals Std: ${residuals.std():,.0f}")
    
    # Prediction accuracy analysis
    print("\n" + "="*50)
    print("PREDICTION ACCURACY ANALYSIS")
    print("="*50)
    
    # Calculate prediction accuracy within different ranges
    accuracy_ranges = [10000, 25000, 50000, 100000]
    for range_val in accuracy_ranges:
        within_range = np.abs(residuals) <= range_val
        accuracy = within_range.mean() * 100
        print(f"Predictions within ±${range_val:,}: {accuracy:.1f}%")
    
    # Price range analysis
    print("\n" + "="*50)
    print("PERFORMANCE BY PRICE RANGE")
    print("="*50)
    
    # Create price bins
    price_bins = pd.qcut(y_test, q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    for price_range in price_bins.cat.categories:
        mask = price_bins == price_range
        if mask.sum() > 0:
            range_r2 = r2_score(y_test[mask], y_pred[mask])
            range_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            price_min = y_test[mask].min()
            price_max = y_test[mask].max()
            
            print(f"{price_range} (${price_min:,.0f} - ${price_max:,.0f}):")
            print(f"  R²: {range_r2:.4f}, RMSE: ${range_rmse:,.0f}, MAE: ${range_mae:,.0f}")
    
    # Feature importance analysis (if available)
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "="*50)
        print("TOP 15 MOST IMPORTANT FEATURES")
        print("="*50)
        
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"{i+1:2d}. {row['feature']:30} {row['importance']:.4f}")
    
    # Error analysis
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    
    # Find worst predictions
    abs_errors = np.abs(residuals)
    worst_predictions_idx = abs_errors.nlargest(5).index
    
    print("5 Worst Predictions:")
    for i, idx in enumerate(worst_predictions_idx, 1):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        error = residuals.iloc[idx]
        error_pct = (error / actual) * 100
        
        print(f"{i}. Actual: ${actual:,.0f}, Predicted: ${predicted:,.0f}")
        print(f"   Error: ${error:,.0f} ({error_pct:+.1f}%)")
    
    # Best predictions
    print("\n5 Best Predictions:")
    best_predictions_idx = abs_errors.nsmallest(5).index
    
    for i, idx in enumerate(best_predictions_idx, 1):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        error = residuals.iloc[idx]
        error_pct = (error / actual) * 100
        
        print(f"{i}. Actual: ${actual:,.0f}, Predicted: ${predicted:,.0f}")
        print(f"   Error: ${error:,.0f} ({error_pct:+.1f}%)")
    
    # Model comparison (if available)
    try:
        with open('model_results.json', 'r') as f:
            all_results = json.load(f)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        
        comparison_data = []
        for model_name, metrics in all_results.items():
            comparison_data.append({
                'Model': model_name,
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Training Time (s)': metrics['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
    except FileNotFoundError:
        print("Model comparison data not found.")
    
    # Save evaluation results
    evaluation_results = {
        'model_name': metadata['best_model'],
        'test_r2': float(r2),
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_mape': float(mape),
        'residuals_mean': float(residuals.mean()),
        'residuals_std': float(residuals.std()),
        'evaluation_date': pd.Timestamp.now().isoformat()
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n✅ Evaluation results saved to 'evaluation_results.json'")
    
    return y_test, y_pred, residuals, evaluation_results

def create_prediction_example():
    """Create an example of how to use the model for new predictions"""
    
    print("\n" + "="*50)
    print("PREDICTION EXAMPLE")
    print("="*50)
    
    # Load model and scaler
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with open('feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    print("Example: Predicting price for a new house")
    print("-" * 40)
    
    # Create example house data (this would normally come from user input)
    example_house = {
        'bedrooms': 3,
        'bathrooms': 2,
        'square_feet': 2000,
        'lot_size': 8000,
        'age': 10,
        'crime_rate': 2.5,
        'school_rating': 8.0,
        'distance_to_city': 15,
        'garage_spaces': 2,
        'has_pool': 1,
        'has_fireplace': 1,
        'property_tax': 6000,
        'location': 'Suburbs'
    }
    
    print("House features:")
    for key, value in example_house.items():
        print(f"  {key}: {value}")
    
    # Note: In a real implementation, you would need to:
    # 1. Create a DataFrame with the example house data
    # 2. Apply the same preprocessing steps (feature engineering, encoding)
    # 3. Scale the features using the saved scaler
    # 4. Make the prediction
    
    print("\nTo make predictions with new data:")
    print("1. Load the saved model and scaler")
    print("2. Preprocess new data using the same steps")
    print("3. Scale features using the saved scaler")
    print("4. Use model.predict() to get price prediction")
    
    print(f"\nModel expects {len(feature_info['feature_names'])} features after preprocessing")

if __name__ == "__main__":
    y_test, y_pred, residuals, results = evaluate_model_performance()
    create_prediction_example()
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETE!")
    print("="*60)
