from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
import sys
import os

# Add the parent directory to the path to import FullPipeline1 and all transformers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transformers import (
    FullPipeline1, 
    ColumnSelector, 
    Datatypefix, 
    Extractionfeature, 
    Transformation, 
    OneHotEncodeColumns, 
    LabelEncodeColumns,
    ScalingTransform, 
    DropColumnsTransformer
)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Try to load the trained model with error handling
try:
    with open('../models/laptop_price_voting_regressor1.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Try to load the fitted FullPipeline1 object with error handling
try:
    with open('../models/fitted_fullpipeline1.pkl', 'rb') as file:
        fitted_pipeline = pickle.load(file)
    print("✅ Pipeline loaded successfully")
    print(f"Pipeline type: {type(fitted_pipeline)}")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    fitted_pipeline = None

# Define categorical columns and their possible values (based on the FullPipeline1 requirements)
categorical_columns = {
    'Company': ['Acer', 'Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'Toshiba', 'MSI', 'Samsung', 'Other'],
    'TypeName': ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook'],
    'Cpu': ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD', 'Other'],
    'Gpu': ['Intel', 'Nvidia', 'AMD'],
    'OpSys': ['Windows', 'macOS', 'Linux', 'No OS', 'Other']
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and pipeline are loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
        
        if fitted_pipeline is None:
            return jsonify({'error': 'Pipeline not loaded. Check server logs.'}), 500

        # Get JSON data from request
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        print(f"📥 Received data: {data}")

        # Extract and format features according to FullPipeline data format
        # Handle HDD and SSD combination into Memory field
        try:
            hdd_value = int(float(str(data.get('HDD', '0'))))
            ssd_value = int(float(str(data.get('SSD', '256'))))
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid HDD/SSD values: {e}'}), 400
        
        print(f"💾 Storage values - HDD: {hdd_value}GB, SSD: {ssd_value}GB")

        # Format Memory field to match the dataset format
        memory_parts = []
        if ssd_value > 0:
            memory_parts.append(f"{ssd_value}GB SSD")
        if hdd_value > 0:
            if memory_parts:
                memory_parts.append(f" +  {hdd_value//1000}TB HDD" if hdd_value >= 1000 else f" +  {hdd_value}GB HDD")
            else:
                memory_parts.append(f"{hdd_value//1000}TB HDD" if hdd_value >= 1000 else f"{hdd_value}GB HDD")
        
        memory_str = "".join(memory_parts) if memory_parts else "256GB SSD"
        
        # Format Ram field to match dataset format (e.g., "8GB")
        ram_value = str(data.get('Ram', '8'))
        ram_str = f"{ram_value}GB"
        
        # Format Weight field to match dataset format (e.g., "2.0kg")
        weight_value = str(data.get('Weight', '2.0'))
        weight_str = f"{weight_value}kg"
        
        features = {
            'Company': str(data.get('Company', 'Other')),
            'TypeName': str(data.get('TypeName', 'Notebook')),
            'Inches': str(data.get('Inches', '15.6')),
            'ScreenResolution': str(data.get('ScreenResolution', '1920x1080')),
            'Cpu': str(data.get('Cpu', 'Intel Core i5')),
            'Ram': ram_str,
            'Memory': memory_str,
            'Gpu': str(data.get('Gpu', 'Intel')),
            'OpSys': str(data.get('OpSys', 'Windows')),
            'Weight': weight_str
        }

        print(f"🔧 Formatted features: {features}")

        # Create DataFrame for prediction
        input_df = pd.DataFrame([features])
        print(f"📊 Input DataFrame shape: {input_df.shape}")
        print(f"📊 Input DataFrame columns: {input_df.columns.tolist()}")
        
        # Create fake y (Price) data for the pipeline
        fake_y = pd.DataFrame({'Price': [1000.0]})  # Dummy price value

        # Transform the input using the FullPipeline
        print("🔄 Transforming input data...")
        processed_input, _ = fitted_pipeline.transform(input_df, fake_y)
        print(f"✅ Transformation successful. Shape: {processed_input.shape}")

        # Make prediction
        print("🔮 Making prediction...")
        prediction = model.predict(processed_input)
        prediction = prediction.reshape(-1, 1)
        predicted_price = fitted_pipeline.inverse_transform_y(prediction)  # Inverse transform if necessary
        pred_value = float(predicted_price[0][0])
        print(f"💰 Predicted price:  ₹{pred_value:.2f}")

        return jsonify({'price': round(pred_value, 2)})
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)