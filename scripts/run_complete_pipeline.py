import subprocess
import sys
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print('='*60)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, f'scripts/{script_name}'], 
                              capture_output=True, text=True, check=True)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        
        execution_time = time.time() - start_time
        print(f"✅ {description} completed successfully in {execution_time:.2f} seconds")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Run the complete machine learning pipeline"""
    
    print("🏠 HOUSING PRICE PREDICTION - COMPLETE ML PIPELINE")
    print("="*60)
    print("This pipeline will:")
    print("1. Generate synthetic housing data")
    print("2. Perform exploratory data analysis")
    print("3. Preprocess the data")
    print("4. Train multiple ML models")
    print("5. Evaluate model performance")
    print("6. Create comprehensive visualizations")
    print("="*60)
    
    # Define the pipeline steps
    pipeline_steps = [
        ('generate_data.py', 'STEP 1: Generating Housing Dataset'),
        ('data_analysis.py', 'STEP 2: Exploratory Data Analysis'),
        ('data_preprocessing.py', 'STEP 3: Data Preprocessing'),
        ('train_models.py', 'STEP 4: Training ML Models'),
        ('model_evaluation.py', 'STEP 5: Model Evaluation'),
        ('visualizations.py', 'STEP 6: Creating Visualizations')
    ]
    
    # Track pipeline progress
    successful_steps = 0
    total_steps = len(pipeline_steps)
    pipeline_start_time = time.time()
    
    # Execute each step
    for i, (script, description) in enumerate(pipeline_steps, 1):
        print(f"\n📊 Progress: {i}/{total_steps} steps")
        
        success = run_script(script, description)
        
        if success:
            successful_steps += 1
        else:
            print(f"\n⚠️  Pipeline stopped at step {i} due to error.")
            print("Please check the error messages above and fix any issues.")
            break
        
        # Small delay between steps
        time.sleep(1)
    
    # Pipeline summary
    total_time = time.time() - pipeline_start_time
    
    print(f"\n{'='*60}")
    print("🎯 PIPELINE EXECUTION SUMMARY")
    print('='*60)
    print(f"Total steps completed: {successful_steps}/{total_steps}")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if successful_steps == total_steps:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        print("📊 Data files:")
        print("   - housing_data.csv (original dataset)")
        print("   - X_train.csv, X_test.csv (processed features)")
        print("   - y_train.csv, y_test.csv (target values)")
        print("\n🤖 Model files:")
        print("   - best_model.pkl (best performing model)")
        print("   - all_models.pkl (all trained models)")
        print("   - scaler.pkl (feature scaler)")
        print("\n📈 Results files:")
        print("   - model_results.json (all model performance metrics)")
        print("   - model_metadata.json (best model information)")
        print("   - evaluation_results.json (detailed evaluation)")
        print("   - feature_info.json (feature information)")
        print("\n🎨 Visualization files:")
        print("   - housing_price_analysis.png (comprehensive analysis)")
        print("   - model_performance_comparison.png (model comparison)")
        
        print(f"\n💡 Next steps:")
        print("1. Review the generated visualizations")
        print("2. Analyze model performance metrics")
        print("3. Use the best model for making predictions")
        print("4. Consider feature engineering improvements")
        print("5. Experiment with hyperparameter tuning")
        
    else:
        print("❌ PIPELINE INCOMPLETE")
        print("Some steps failed to execute. Please review the error messages.")
    
    print("="*60)

if __name__ == "__main__":
    main()
