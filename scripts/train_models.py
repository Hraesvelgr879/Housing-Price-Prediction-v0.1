import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time

def train_and_evaluate_models():
    """Train multiple regression models and compare their performance"""
    
    print("=== HOUSING PRICE PREDICTION - MODEL TRAINING ===\n")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Store results
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        start_time = time.time()
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'training_time': training_time,
                'overfitting': train_r2 - test_r2
            }
            
            trained_models[name] = model
            
            # Print results
            print(f"✅ {name} completed in {training_time:.2f}s")
            print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"   Train RMSE: ${train_rmse:,.0f} | Test RMSE: ${test_rmse:,.0f}")
            print(f"   Test MAE: ${test_mae:,.0f}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {str(e)}")
            continue
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('test_r2', ascending=False)
    
    print("\nModel Performance Ranking (by Test R²):")
    print("-" * 50)
    for i, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"{i}. {model_name}")
        print(f"   Test R²: {row['test_r2']:.4f}")
        print(f"   Test RMSE: ${row['test_rmse']:,.0f}")
        print(f"   Test MAE: ${row['test_mae']:,.0f}")
        print(f"   Overfitting: {row['overfitting']:.4f}")
        print()
    
    # Find best model
    best_model_name = comparison_df.index[0]
    best_model = trained_models[best_model_name]
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {comparison_df.loc[best_model_name, 'test_r2']:.4f}")
    print(f"   Test RMSE: ${comparison_df.loc[best_model_name, 'test_rmse']:,.0f}")
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n📊 Feature Importance ({best_model_name}):")
        print("-" * 40)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:25} {row['importance']:.4f}")
    
    # Save models and results
    print("\n" + "="*60)
    print("SAVING MODELS AND RESULTS")
    print("="*60)
    
    # Save best model
    joblib.dump(best_model, 'best_model.pkl')
    print(f"✅ Best model ({best_model_name}) saved as 'best_model.pkl'")
    
    # Save all models
    joblib.dump(trained_models, 'all_models.pkl')
    print("✅ All models saved as 'all_models.pkl'")
    
    # Save results
    with open('model_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {k: float(v) for k, v in metrics.items()}
        json.dump(json_results, f, indent=2)
    
    print("✅ Results saved as 'model_results.json'")
    
    # Save model metadata
    metadata = {
        'best_model': best_model_name,
        'best_test_r2': float(comparison_df.loc[best_model_name, 'test_r2']),
        'best_test_rmse': float(comparison_df.loc[best_model_name, 'test_rmse']),
        'training_date': pd.Timestamp.now().isoformat(),
        'n_features': len(X_train.columns),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Metadata saved as 'model_metadata.json'")
    
    return trained_models, results, best_model_name

if __name__ == "__main__":
    models, results, best_model = train_and_evaluate_models()
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
