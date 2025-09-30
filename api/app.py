from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
with open('../models/laptop_price_voting_regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Define categorical columns and their possible values (based on the repository's dataset)
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
        # Get JSON data from request
        data = request.get_json()

        # Extract features from the request
        features = {
            'Company': data.get('Company', 'Other'),
            'TypeName': data.get('TypeName', 'Notebook'),
            'Ram': int(data.get('Ram', 8)),
            'Weight': float(data.get('Weight', 2.0)),
            'Touchscreen': int(data.get('Touchscreen', 0)),
            'Ips': int(data.get('Ips', 0)),
            'Ppi': float(data.get('Ppi', 141.21)),
            'Cpu': data.get('Cpu', 'Intel Core i5'),
            'HDD': int(data.get('HDD', 0)),
            'SSD': int(data.get('SSD', 256)),
            'Gpu': data.get('Gpu', 'Intel'),
            'OpSys': data.get('OpSys', 'Windows')
        }

        # Create DataFrame for prediction
        input_df = pd.DataFrame([features])

        # Perform one-hot encoding for categorical variables
        for col in categorical_columns:
            input_df[col] = input_df[col].apply(lambda x: x if x in categorical_columns[col] else 'Other')

        input_df = pd.get_dummies(input_df, columns=['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys'])

        # Ensure all columns from training are present
        with open('../models/preprocessing_pipeline.pkl', 'rb') as file:
            model_columns = pickle.load(file)
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # Make prediction
        prediction = model.predict(input_df)
        predicted_price = np.exp(prediction[0])  # Inverse of log transformation

        return jsonify({'price': round(predicted_price, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)