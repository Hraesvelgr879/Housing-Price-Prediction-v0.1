import subprocess
import sys
import time
import os

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
    """Run the complete updated machine learning pipeline"""
    
    print("🏠 COMPREHENSIVE HOUSING PRICE PREDICTION PIPELINE")
    print("="*60)
    print("This updated pipeline will:")
    print("1. Generate comprehensive housing dataset with 30+ features")
    print("2. Analyze the dataset with detailed statistics")
    print("3. Update preprocessing for the new dataset")
    print("4. Train multiple ML models")
    print("5. Evaluate model performance")
    print("6. Create comprehensive visualizations")
    print("="*60)
    
    # Define the updated pipeline steps
    pipeline_steps = [
        ('create_housing_dataset.py', 'STEP 1: Creating Comprehensive Housing Dataset'),
        ('dataset_analyzer.py', 'STEP 2: Analyzing Dataset'),
        ('update_ml_pipeline.py', 'STEP 3: Updating Preprocessing Pipeline'),
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
    print("🎯 COMPREHENSIVE PIPELINE EXECUTION SUMMARY")
    print('='*60)
    print(f"Total steps completed: {successful_steps}/{total_steps}")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if successful_steps == total_steps:
        print("🎉 COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY!")
        
        # Check what files were created
        created_files = []
        expected_files = [
            'comprehensive_housing_data.csv',
            'dataset_summary.json',
            'housing_dataset_analysis.png',
            'X_train.csv', 'X_test.csv',
            'y_train.csv', 'y_test.csv',
            'best_model.pkl',
            'model_results.json',
            'housing_price_analysis.png'
        ]
        
        for file in expected_files:
            if os.path.exists(file):
                created_files.append(file)
        
        print(f"\n📁 Generated Files ({len(created_files)}):")
        print("📊 Dataset files:")
        dataset_files = [f for f in created_files if 'data' in f or f.endswith('.csv')]
        for file in dataset_files:
            print(f"   - {file}")
        
        print("\n🤖 Model files:")
        model_files = [f for f in created_files if 'model' in f or f.endswith('.pkl')]
        for file in model_files:
            print(f"   - {file}")
        
        print("\n📈 Analysis files:")
        analysis_files = [f for f in created_files if f.endswith('.json')]
        for file in analysis_files:
            print(f"   - {file}")
        
        print("\n🎨 Visualization files:")
        viz_files = [f for f in created_files if f.endswith('.png')]
        for file in viz_files:
            print(f"   - {file}")
        
        print(f"\n🎯 Dataset Highlights:")
        try:
            import pandas as pd
            df = pd.read_csv('comprehensive_housing_data.csv')
            print(f"   - {len(df):,} housing samples")
            print(f"   - {len(df.columns)} features")
            print(f"   - Price range: ${df['price'].min():,} - ${df['price'].max():,}")
            print(f"   - {len(df['location'].unique())} different locations")
            print(f"   - Features include: location, size, age, amenities, neighborhood data")
        except:
            print("   - Comprehensive dataset with realistic housing features")
        
        print(f"\n💡 Next Steps:")
        print("1. 📊 Review the dataset analysis visualizations")
        print("2. 🤖 Check model performance metrics")
        print("3. 🔍 Analyze feature importance")
        print("4. 🚀 Use the model for price predictions")
        print("5. 📈 Consider hyperparameter tuning")
        print("6. 🌐 Deploy the model or create a web interface")
        
        print(f"\n🔗 Quick Commands:")
        print("# View dataset info:")
        print("python -c \"import pandas as pd; df=pd.read_csv('comprehensive_housing_data.csv'); print(df.info()); print(df.describe())\"")
        print("\n# Make a prediction (example):")
        print("python -c \"import joblib; model=joblib.load('best_model.pkl'); print('Model loaded successfully!')\"")
        
    else:
        print("❌ PIPELINE INCOMPLETE")
        print("Some steps failed to execute. Please review the error messages.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all required packages are installed")
        print("2. Check that you have write permissions in the current directory")
        print("3. Ensure sufficient disk space for dataset and model files")
    
    print("="*60)

if __name__ == "__main__":
    main()
