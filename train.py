"""
Main training script for laptop price prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.data_loader import load_data, initial_data_cleaning, split_data
from src.data.transformers import FullPipeline1
from src.models.model_training import (get_base_models, evaluate_models_reg, 
                                       create_voting_regressor, save_model)
from src.visualization.plots import plot_regression_performance, create_model_comparison_plot
from src.utils.helpers import shapiro_test_normality


def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_PATH = "data/laptopData.csv"  # Update with actual path
    MODEL_SAVE_PATH = "models/laptop_price_voting_regressor.pkl"
    PIPELINE_SAVE_PATH = "models/preprocessing_pipeline.pkl"
    
    print("Starting Laptop Price Prediction Training Pipeline...")
    
    # Step 1: Load and clean data
    print("\n1. Loading and cleaning data...")
    df = load_data(DATA_PATH)
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    df_clean = initial_data_cleaning(df)
    
    # Step 2: Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    # Step 3: Determine scaling features using normality test
    print("\n3. Analyzing feature distributions...")
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # Fix data types first
    train_copy = train_data.copy()
    train_copy['Ram'] = train_copy['Ram'].str.replace('GB', '').astype('int32')
    train_copy['Weight'] = train_copy['Weight'].str.replace('kg', '').astype('float32')
    
    # Get numerical columns after basic feature engineering
    num_cols = ['Weight', 'Ram', 'Price']  # Basic numerical columns
    feats_std_scale, feats_min_max_scale = shapiro_test_normality(train_copy, num_cols)
    
    # Add engineered features to scaling lists
    feats_min_max_scale.extend(['ppi', 'HDD', 'SSD'])
    
    # Step 4: Create preprocessing pipeline
    print("\n4. Creating preprocessing pipeline...")
    full_pipeline = FullPipeline1(feats_min_max_scale, num_cols)
    
    # Transform data
    X_train_processed, y_train_processed = full_pipeline.fit_transform(X_train, y_train)
    X_test_processed, y_test_processed = full_pipeline.transform(X_test, y_test)
    
    print(f"Processed data shapes: X_train {X_train_processed.shape}, X_test {X_test_processed.shape}")
    
    # Step 5: Model selection and evaluation
    print("\n5. Evaluating multiple models...")
    models = get_base_models()
    results = evaluate_models_reg(models, X_train_processed, y_train_processed, 
                                  X_test_processed, y_test_processed, cv=5)
    
    print("\nModel Evaluation Results:")
    print(results)
    
    # Create comparison plot
    create_model_comparison_plot(results, 'Test Score')
    
    # Step 6: Create and train final ensemble model
    print("\n6. Training final voting regressor...")
    voting_regressor = create_voting_regressor(X_train_processed, y_train_processed)
    
    # Evaluate final model
    from src.models.model_training import evaluate_model
    train_mae, train_rmse, train_r2 = evaluate_model(voting_regressor, X_train_processed, y_train_processed)
    test_mae, test_rmse, test_r2 = evaluate_model(voting_regressor, X_test_processed, y_test_processed)
    
    print(f"\nFinal Voting Regressor Performance:")
    print(f"Train - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Test  - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    # Plot performance
    y_pred = voting_regressor.predict(X_test_processed)
    plot_regression_performance(y_test_processed, y_pred)
    
    # Step 7: Save model and pipeline
    print("\n7. Saving model and pipeline...")
    save_model(voting_regressor, MODEL_SAVE_PATH, full_pipeline, PIPELINE_SAVE_PATH)
    
    print("\nTraining pipeline completed successfully!")
    
    return voting_regressor, full_pipeline


if __name__ == "__main__":
    model, pipeline = main()