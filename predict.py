"""
Prediction script for laptop price prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
from src.models.model_training import load_model


def predict_laptop_price(laptop_specs, model_path="models/laptop_price_voting_regressor.pkl", 
                         pipeline_path="models/preprocessing_pipeline.pkl"):
    """
    Predict laptop price based on specifications
    
    Args:
        laptop_specs (dict): Dictionary containing laptop specifications
        model_path (str): Path to the trained model
        pipeline_path (str): Path to the preprocessing pipeline
        
    Returns:
        float: Predicted price
    """
    
    # Load model and pipeline
    try:
        model, pipeline = load_model(model_path, pipeline_path)
    except Exception as e:
        print(f"Error loading model or pipeline: {e}")
        return None
    
    # Create DataFrame from input specifications
    df = pd.DataFrame([laptop_specs])
    
    # Create dummy target for preprocessing
    dummy_target = pd.DataFrame({'Price': [0]})
    
    try:
        # Preprocess the data
        X_processed, _ = pipeline.transform(df, dummy_target)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def main():
    """Example usage of the prediction function"""
    
    # Example laptop specifications
    sample_laptops = [
        {
            'Company': 'Dell',
            'TypeName': 'Notebook',
            'Inches': 15.6,
            'ScreenResolution': '1920x1080',
            'Cpu': 'Intel Core i5 8250U 1.6GHz',
            'Ram': '8GB',
            'Memory': '256GB SSD',
            'Gpu': 'Intel UHD Graphics 620',
            'OpSys': 'Windows 10',
            'Weight': '2.1kg'
        },
        {
            'Company': 'Apple',
            'TypeName': 'Ultrabook',
            'Inches': 13.3,
            'ScreenResolution': '2560x1600 IPS Panel Retina Display',
            'Cpu': 'Intel Core i5 1.8GHz',
            'Ram': '8GB',
            'Memory': '128GB SSD',
            'Gpu': 'Intel HD Graphics 6000',
            'OpSys': 'Mac OS X',
            'Weight': '1.37kg'
        },
        {
            'Company': 'HP',
            'TypeName': 'Gaming',
            'Inches': 15.6,
            'ScreenResolution': '1920x1080 IPS Panel',
            'Cpu': 'Intel Core i7 7700HQ 2.8GHz',
            'Ram': '16GB',
            'Memory': '1TB HDD +  128GB SSD',
            'Gpu': 'Nvidia GeForce GTX 1050',
            'OpSys': 'Windows 10',
            'Weight': '2.85kg'
        }
    ]
    
    print("Laptop Price Predictions:")
    print("=" * 50)
    
    for i, laptop in enumerate(sample_laptops, 1):
        print(f"\nLaptop {i}:")
        print(f"Brand: {laptop['Company']}")
        print(f"Type: {laptop['TypeName']}")
        print(f"Screen: {laptop['Inches']}\" {laptop['ScreenResolution']}")
        print(f"CPU: {laptop['Cpu']}")
        print(f"RAM: {laptop['Ram']}")
        print(f"Storage: {laptop['Memory']}")
        print(f"GPU: {laptop['Gpu']}")
        print(f"OS: {laptop['OpSys']}")
        print(f"Weight: {laptop['Weight']}")
        
        predicted_price = predict_laptop_price(laptop)
        
        if predicted_price is not None:
            print(f"Predicted Price: ${predicted_price:.2f}")
        else:
            print("Prediction failed!")
        print("-" * 30)


if __name__ == "__main__":
    main()