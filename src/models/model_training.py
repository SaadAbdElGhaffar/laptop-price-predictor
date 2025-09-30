"""
Model training and evaluation functions
"""

import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor,
                              VotingRegressor, StackingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def baseline_reg(X_train, y_train, X_test, y_test, strategy='mean'):
    """
    Create baseline regression model using DummyRegressor
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        strategy: Strategy for dummy regressor
        
    Returns:
        dict: Baseline metrics
    """
    dummy_regressor = DummyRegressor(strategy=strategy)
    dummy_regressor.fit(X_train, y_train)
    dummy_predictions = dummy_regressor.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, dummy_predictions)
    mae = mean_absolute_error(y_test, dummy_predictions)
    r2 = r2_score(y_test, dummy_predictions)

    # Print the results
    print("Baseline MSE:", mse)
    print("Baseline MAE:", mae)
    print("Baseline R2 Score:", r2)
    
    return {'mse': mse, 'mae': mae, 'r2': r2}


def get_base_models():
    """
    Get list of base regression models for comparison
    
    Returns:
        list: List of instantiated models
    """
    models = [
        LinearRegression(),
        Ridge(random_state=ord("S")),
        Lasso(random_state=ord("S")),
        ElasticNet(random_state=ord("S")),
        DecisionTreeRegressor(random_state=ord("S")),
        RandomForestRegressor(random_state=ord("S")),
        GradientBoostingRegressor(random_state=ord("S")),
        XGBRegressor(random_state=ord("S")),
        ExtraTreesRegressor(random_state=ord("S")),
        BaggingRegressor(random_state=ord("S")),
        AdaBoostRegressor(random_state=ord("S")),
        CatBoostRegressor(random_state=ord("S"), verbose=False),
        SVR(),
        KNeighborsRegressor()
    ]
    return models


def evaluate_models_reg(models, X_train, y_train, X_test, y_test, cv=5):
    """
    Evaluate multiple regression models using cross-validation
    
    Args:
        models: List of models to evaluate
        X_train, y_train: Training data
        X_test, y_test: Test data
        cv: Number of cross-validation folds
        
    Returns:
        pd.DataFrame: Results dataframe with metrics for all models
    """
    results = []
    start_total = time.time()

    for model in models:
        start = time.time()

        # Cross-validation scores
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                return_train_score=True)
        
        # Mean scores from cross-validation (for training)
        mean_train_mae = -np.mean(scores['train_neg_mean_absolute_error'])
        mean_train_rmse = np.sqrt(-np.mean(scores['train_neg_mean_squared_error']))
        mean_train_r2 = np.mean(scores['train_r2'])
        
        # Test set evaluation
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        # Train set evaluation
        train_preds = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        
        # Store results
        results_dict = {
            'Algorithm': model.__class__.__name__,
            'Train Score': mean_train_r2,
            'Test Score': test_r2,
            'Train MAE': mean_train_mae,
            'Test MAE': test_mae,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train MSE': mean_train_rmse,
            'Test MSE': test_rmse
        }
        results.append(results_dict)

    total_time = time.time() - start_total
    results_df = pd.DataFrame(results)
    results_df.set_index('Algorithm', inplace=True)
    results_df['Total Time'] = total_time
    results_df = results_df.sort_values(by='Test Score', ascending=False)

    return results_df


def evaluate_model(model, X, y):
    """
    Evaluate a single model and return metrics
    
    Args:
        model: Trained model
        X, y: Data to evaluate on
        
    Returns:
        tuple: (mae, rmse, r2)
    """
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2


def hyperparameter_tuning_rf(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        RandomizedSearchCV: Fitted search object
    """
    rf = RandomForestRegressor(random_state=ord("S"))

    param_dist_rf = {
        'n_estimators': randint(100, 1000),  
        'max_depth': randint(3, 15),           
        'min_samples_split': randint(2, 20),   
        'min_samples_leaf': randint(1, 20),    
        'max_features': uniform(0.1, 0.9),     
        'bootstrap': [True, False]             
    }

    rf_random = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist_rf, 
        n_iter=100, 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train)
    return rf_random


def hyperparameter_tuning_catboost(X_train, y_train):
    """
    Perform hyperparameter tuning for CatBoost
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        RandomizedSearchCV: Fitted search object
    """
    catboost = CatBoostRegressor(random_state=ord("S"), verbose=0)

    param_dist_cb = {
        'iterations': randint(100, 1000),    
        'depth': randint(3, 10),             
        'learning_rate': uniform(0.01, 0.05), 
        'l2_leaf_reg': uniform(1, 5),         
        'border_count': randint(32, 100),     
    }

    cb_random = RandomizedSearchCV(
        estimator=catboost, 
        param_distributions=param_dist_cb, 
        n_iter=20,  
        cv=2,      
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )   

    cb_random.fit(X_train, y_train)
    return cb_random


def hyperparameter_tuning_xgb(X_train, y_train):
    """
    Perform hyperparameter tuning for XGBoost
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        RandomizedSearchCV: Fitted search object
    """
    xgb = XGBRegressor(random_state=ord("S"))

    param_dist_xgb = {
        'n_estimators': randint(100, 2000),  
        'learning_rate': uniform(0.01, 0.3), 
        'max_depth': randint(3, 15),          
        'subsample': uniform(0.5, 0.5),       
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 1),               
        'reg_alpha': uniform(0, 1),          
        'reg_lambda': uniform(0, 1)          
    }

    xgb_random = RandomizedSearchCV(
        estimator=xgb, 
        param_distributions=param_dist_xgb, 
        n_iter=100, 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    xgb_random.fit(X_train, y_train)
    return xgb_random


def create_voting_regressor(X_train, y_train):
    """
    Create and train a voting regressor with best performing models
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        VotingRegressor: Fitted voting regressor
    """
    models = [
        ('RandomForest', RandomForestRegressor(random_state=ord("S"))),
        ('CatBoost', CatBoostRegressor(random_state=ord("S"), verbose=False)),
        ('XGB', XGBRegressor(random_state=ord("S")))
    ]

    voting_regressor = VotingRegressor(estimators=models)
    voting_regressor.fit(X_train, y_train)
    
    return voting_regressor


def create_stacking_regressor(X_train, y_train):
    """
    Create and train a stacking regressor with best performing models
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        StackingRegressor: Fitted stacking regressor
    """
    base_models = [
        ('RandomForest', RandomForestRegressor(random_state=ord("S"))),
        ('CatBoost', CatBoostRegressor(random_state=ord("S"), verbose=False)),
        ('XGB', XGBRegressor(random_state=ord("S")))
    ]

    meta_model = LinearRegression()
    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stacking_regressor.fit(X_train, y_train)
    
    return stacking_regressor


def save_model(model, model_path, pipeline=None, pipeline_path=None):
    """
    Save trained model and preprocessing pipeline
    
    Args:
        model: Trained model to save
        model_path: Path to save the model
        pipeline: Preprocessing pipeline (optional)
        pipeline_path: Path to save the pipeline (optional)
    """
    import joblib
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the pipeline if provided
    if pipeline is not None and pipeline_path is not None:
        joblib.dump(pipeline, pipeline_path)
        print(f"Preprocessing pipeline saved to: {pipeline_path}")


def load_model(model_path, pipeline_path=None):
    """
    Load trained model and preprocessing pipeline
    
    Args:
        model_path: Path to the saved model
        pipeline_path: Path to the saved pipeline (optional)
        
    Returns:
        tuple: (model, pipeline) or just model if no pipeline path provided
    """
    import joblib
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    if pipeline_path is not None:
        pipeline = joblib.load(pipeline_path)
        print(f"Pipeline loaded from: {pipeline_path}")
        return model, pipeline
    
    return model