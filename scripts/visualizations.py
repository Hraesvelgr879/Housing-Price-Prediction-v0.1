import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import r2_score

def create_visualizations():
    """Create comprehensive visualizations for the housing price prediction project"""
    
    print("=== CREATING VISUALIZATIONS ===\n")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load data
    df = pd.read_csv('housing_data.csv')
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').squeeze()
    
    # Load model and make predictions
    best_model = joblib.load('best_model.pkl')
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    
    # Load metadata
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Price Distribution
    plt.subplot(4, 3, 1)
    plt.hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.ticklabel_format(style='plain', axis='x')
    
    # 2. Price by Location
    plt.subplot(4, 3, 2)
    location_prices = df.groupby('location')['price'].mean().sort_values(ascending=True)
    bars = plt.barh(range(len(location_prices)), location_prices.values, color='lightcoral')
    plt.yticks(range(len(location_prices)), location_prices.index)
    plt.title('Average Price by Location', fontsize=14, fontweight='bold')
    plt.xlabel('Average Price ($)')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 5000, bar.get_y() + bar.get_height()/2, 
                f'${width:,.0f}', ha='left', va='center')
    
    # 3. Correlation Heatmap
    plt.subplot(4, 3, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 4. Actual vs Predicted Prices
    plt.subplot(4, 3, 4)
    plt.scatter(y_test, y_pred, alpha=0.6, color='darkblue', s=30)
    
    # Perfect prediction line
    min_price = min(y_test.min(), y_pred.min())
    max_price = max(y_test.max(), y_pred.max())
    plt.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Actual vs Predicted Prices\nR² = {r2_score(y_test, y_pred):.4f}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.ticklabel_format(style='plain')
    
    # 5. Residuals Plot
    plt.subplot(4, 3, 5)
    plt.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    plt.ticklabel_format(style='plain')
    
    # 6. Residuals Distribution
    plt.subplot(4, 3, 6)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Residuals ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
    
    # 7. Feature Importance (if available)
    plt.subplot(4, 3, 7)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                       color='purple', alpha=0.7)
        plt.yticks(range(len(feature_importance)), 
                  [f.replace('_', ' ').title() for f in feature_importance['feature']])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
    
    # 8. Price vs Square Feet
    plt.subplot(4, 3, 8)
    plt.scatter(df['square_feet'], df['price'], alpha=0.6, color='brown', s=30)
    
    # Add trend line
    z = np.polyfit(df['square_feet'], df['price'], 1)
    p = np.poly1d(z)
    plt.plot(df['square_feet'], p(df['square_feet']), "r--", alpha=0.8, lw=2)
    
    plt.xlabel('Square Feet')
    plt.ylabel('Price ($)')
    plt.title('Price vs Square Feet', fontsize=14, fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 9. Price vs Age
    plt.subplot(4, 3, 9)
    plt.scatter(df['age'], df['price'], alpha=0.6, color='teal', s=30)
    
    # Add trend line
    z = np.polyfit(df['age'], df['price'], 1)
    p = np.poly1d(z)
    plt.plot(df['age'], p(df['age']), "r--", alpha=0.8, lw=2)
    
    plt.xlabel('Age (years)')
    plt.ylabel('Price ($)')
    plt.title('Price vs House Age', fontsize=14, fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 10. Model Performance Comparison
    plt.subplot(4, 3, 10)
    try:
        with open('model_results.json', 'r') as f:
            all_results = json.load(f)
        
        models = list(all_results.keys())
        r2_scores = [all_results[model]['test_r2'] for model in models]
        
        bars = plt.bar(range(len(models)), r2_scores, color='lightgreen', alpha=0.8)
        plt.xticks(range(len(models)), [m.replace(' ', '\n') for m in models], rotation=0)
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    except FileNotFoundError:
        plt.text(0.5, 0.5, 'Model comparison\ndata not available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    
    # 11. Price by Number of Bedrooms
    plt.subplot(4, 3, 11)
    bedroom_stats = df.groupby('bedrooms')['price'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(bedroom_stats['bedrooms'], bedroom_stats['mean'], 
                  yerr=bedroom_stats['std'], capsize=5, color='gold', alpha=0.8)
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Price ($)')
    plt.title('Average Price by Number of Bedrooms', fontsize=14, fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5000,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # 12. Crime Rate vs Price
    plt.subplot(4, 3, 12)
    plt.scatter(df['crime_rate'], df['price'], alpha=0.6, color='red', s=30)
    
    # Add trend line
    z = np.polyfit(df['crime_rate'], df['price'], 1)
    p = np.poly1d(z)
    plt.plot(df['crime_rate'], p(df['crime_rate']), "b--", alpha=0.8, lw=2)
    
    plt.xlabel('Crime Rate')
    plt.ylabel('Price ($)')
    plt.title('Price vs Crime Rate', fontsize=14, fontweight='bold')
    plt.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('housing_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate detailed model performance plot
    create_model_performance_plot()
    
    print("✅ Visualizations created and saved as 'housing_price_analysis.png'")

def create_model_performance_plot():
    """Create a detailed model performance visualization"""
    
    try:
        with open('model_results.json', 'r') as f:
            all_results = json.load(f)
        
        # Create performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(all_results.keys())
        
        # R² Scores
        r2_scores = [all_results[model]['test_r2'] for model in models]
        bars1 = ax1.bar(models, r2_scores, color='lightblue', alpha=0.8)
        ax1.set_title('Test R² Scores by Model', fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # RMSE
        rmse_scores = [all_results[model]['test_rmse'] for model in models]
        bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.8)
        ax2.set_title('Test RMSE by Model', fontweight='bold')
        ax2.set_ylabel('RMSE ($)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.ticklabel_format(style='plain', axis='y')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Training Time
        training_times = [all_results[model]['training_time'] for model in models]
        bars3 = ax3.bar(models, training_times, color='lightgreen', alpha=0.8)
        ax3.set_title('Training Time by Model', fontweight='bold')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        
        # Overfitting Analysis
        overfitting = [all_results[model]['overfitting'] for model in models]
        bars4 = ax4.bar(models, overfitting, color='orange', alpha=0.8)
        ax4.set_title('Overfitting Analysis (Train R² - Test R²)', fontweight='bold')
        ax4.set_ylabel('Overfitting Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Model performance comparison saved as 'model_performance_comparison.png'")
        
    except FileNotFoundError:
        print("❌ Model results not found. Run model training first.")

if __name__ == "__main__":
    create_visualizations()
    print("\n" + "="*50)
    print("VISUALIZATION CREATION COMPLETE!")
    print("="*50)
