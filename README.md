# Laptop Price Predictor

A comprehensive machine learning project that predicts laptop prices based on various specifications including brand, type, screen size, processor, RAM, storage, graphics card, operating system, and weight. This project demonstrates end-to-end ML pipeline development from raw data processing to model deployment.

**Dataset**: [Uncleaned Laptop Price Dataset](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset) - A real-world dataset requiring extensive preprocessing and feature engineering.

**📓 Kaggle Notebook**: [View the complete analysis and development process](https://www.kaggle.com/code/abdocan/laptop-price-predictor-91-0-216-mae-0-292-rmse/notebook)

## Project Structure

```
laptop-price-predictor/
│
├── data/                          # Data directory
│   └── laptopData.csv            # Dataset (not included in repo)
│
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── data_loader.py        # Data loading and cleaning functions
│   │   └── transformers.py       # Custom preprocessing transformers
│   │
│   ├── models/                   # Model training and evaluation
│   │   └── model_training.py     # Model training functions
│   │
│   ├── visualization/            # Visualization functions
│   │   └── plots.py             # Plotting and visualization utilities
│   │
│   └── utils/                    # Utility functions
│       └── helpers.py           # Helper functions for analysis
│
├── models/                       # Saved models
│   ├── laptop_price_voting_regressor.pkl    # Trained model
│   └── preprocessing_pipeline.pkl           # Preprocessing pipeline
│
├── notebooks/                    # Jupyter notebooks
│   └── laptop-price-predictor-91-0-216-mae-0-292-rmse.ipynb  # Original development notebook
│
├── .github/                      # GitHub workflows
│   └── workflows/
│       └── ci.yml               # Continuous Integration
│
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## Features

The model uses the following features to predict laptop prices:

- **Company**: Laptop manufacturer (Dell, HP, Lenovo, etc.)
- **TypeName**: Type of laptop (Notebook, Gaming, Ultrabook, etc.)
- **Inches**: Screen size in inches
- **ScreenResolution**: Display resolution and features (Touchscreen, IPS)
- **Cpu**: Processor information
- **Ram**: RAM capacity in GB
- **Memory**: Storage information (HDD, SSD, Hybrid, Flash Storage)
- **Gpu**: Graphics card information
- **OpSys**: Operating system
- **Weight**: Laptop weight in kg

## Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Data Cleaning**: Removal of problematic entries and missing values
2. **Feature Engineering**:
   - Extracting touchscreen and IPS panel information from screen resolution
   - Creating PPI (Pixels Per Inch) feature
   - Categorizing processors (Intel Core i3/i5/i7, Other Intel, AMD)
   - Breaking down storage into HDD, SSD, Hybrid, and Flash Storage components
   - Categorizing operating systems (Windows, Mac, Others/Linux)
3. **Data Transformation**:
   - Log transformation for skewed features
   - Standard scaling for normally distributed features
   - Min-Max scaling for non-normal features
   - One-hot encoding for categorical variables

## Model Performance

The final ensemble model (Voting Regressor) combines:
- Random Forest Regressor
- CatBoost Regressor  
- XGBoost Regressor

**Performance Metrics:**
- **MAE (Mean Absolute Error)**: 0.216
- **RMSE (Root Mean Square Error)**: 0.292
- **R² Score**: 91.0%

## Dataset

The dataset used in this project is the **Uncleaned Laptop Price Dataset** from Kaggle, which contains laptop specifications and their corresponding prices. 

**Dataset Source**: [Uncleaned Laptop Price Dataset](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset)

### Dataset Features:
- **1,303 laptop entries** with 11 features
- Contains raw, uncleaned data requiring preprocessing
- Includes various laptop brands, specifications, and price ranges
- Price data in Indian Rupees (INR)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/laptop-price-predictor.git
cd laptop-price-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Visit the [dataset link](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset)
   - Download `laptopData.csv`
   - Place it in the `data/` directory

**💡 Tip**: Check out the [complete development process on Kaggle](https://www.kaggle.com/code/abdocan/laptop-price-predictor-91-0-216-mae-0-292-rmse/notebook) to see the step-by-step analysis and model building.

## Usage

### Training the Model

1. Ensure your dataset (`laptopData.csv`) is in the `data/` directory
2. Run the training script:
```bash
python train.py
```

The training process will:
- Load and clean the raw dataset
- Apply feature engineering and preprocessing
- Train multiple models and compare performance
- Create an ensemble voting regressor
- Save the best model and preprocessing pipeline

### Using the Trained Model

```python
import joblib
from src.data.transformers import FullPipeline1

# Load the trained model and preprocessing pipeline
model = joblib.load('models/laptop_price_voting_regressor.pkl')
pipeline = joblib.load('models/preprocessing_pipeline.pkl')

# Example prediction
import pandas as pd

# Create sample data
sample_data = pd.DataFrame({
    'Company': ['Dell'],
    'TypeName': ['Notebook'],
    'Inches': [15.6],
    'ScreenResolution': ['1920x1080'],
    'Cpu': ['Intel Core i5 8250U 1.6GHz'],
    'Ram': ['8GB'],
    'Memory': ['256GB SSD'],
    'Gpu': ['Intel UHD Graphics 620'],
    'OpSys': ['Windows 10'],
    'Weight': ['2.1kg']
})

# Preprocess and predict
X_processed, _ = pipeline.transform(sample_data, pd.DataFrame({'Price': [0]}))
predicted_price = model.predict(X_processed)
print(f"Predicted Price: ${predicted_price[0]:.2f}")
```

## Model Architecture

The project uses an ensemble approach with three base models:

1. **Random Forest Regressor**: Handles non-linear relationships and feature interactions
2. **CatBoost Regressor**: Excellent performance on categorical features
3. **XGBoost Regressor**: Gradient boosting for high predictive accuracy

The final prediction is made using a Voting Regressor that averages the predictions from all three models.

## Data Analysis Insights

Key insights discovered during exploratory data analysis:

- **Screen specifications**: Screen size (Inches) and PPI (Pixels Per Inch) are strongly correlated with price
- **Laptop categories**: Gaming laptops and workstations command premium prices compared to basic notebooks
- **Brand premium**: Apple laptops have significantly higher average prices across all categories
- **Hardware impact**: RAM capacity and storage type (SSD vs HDD) are major price drivers
- **Display features**: Touchscreen capabilities and IPS displays add substantial cost premiums
- **Performance components**: Dedicated graphics cards significantly increase laptop pricing
- **Weight factor**: Lighter ultrabooks typically cost more than heavier traditional laptops

### Data Quality Challenges Addressed:
- **Missing values**: Handled null entries and inconsistent data formats
- **Inconsistent units**: Standardized RAM (GB) and Weight (kg) representations  
- **Complex strings**: Parsed screen resolution, CPU, and storage specifications
- **Categorical variations**: Normalized operating system and brand naming conventions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [Ehtisham Sadiq](https://www.kaggle.com/ehtishamsadiq) for providing the Uncleaned Laptop Price Dataset on Kaggle
- **Original Data Source**: Laptop specifications and pricing data collected from various sources
- **Development**: Original analysis and model development documented on [Kaggle](https://www.kaggle.com/code/abdocan/laptop-price-predictor-91-0-216-mae-0-292-rmse/notebook)
- Scikit-learn, XGBoost, and CatBoost teams for excellent ML libraries
- The open-source community for various tools and libraries used

## Links

- **GitHub Repository**: [Link](https://github.com/SaadAbdElGhaffar/laptop-price-predictor)
- **Kaggle Notebook**: [Link](https://www.kaggle.com/code/abdocan/laptop-price-predictor-91-0-216-mae-0-292-rmse/notebook)
- **Dataset Source**: [Link](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset)

## Contact

Your Name - your.email@example.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.