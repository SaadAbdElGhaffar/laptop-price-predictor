"""
Data loading and initial processing functions
"""

import pandas as pd
import numpy as np
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the laptop dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def initial_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning including removing problematic rows
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Remove rows with '?' values and problematic weight values
    indices = []
    for col in df_clean.columns:
        if '?' in df_clean[col].values:
            indices.extend(df_clean[df_clean[col] == '?'].index.tolist())
    
    if 'Weight' in df_clean.columns:
        weight_indices = df_clean[df_clean['Weight'] == '0.0002kg'].index.tolist()
        indices.extend(weight_indices)
    
    # Remove problematic rows
    if indices:
        df_clean = df_clean.drop(indices, axis=0)
        print(f"Removed {len(indices)} problematic rows")
    
    # Drop unnamed columns
    unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in col]
    if unnamed_cols:
        df_clean = df_clean.drop(columns=unnamed_cols)
        print(f"Dropped columns: {unnamed_cols}")
    
    # Drop null rows
    initial_shape = df_clean.shape[0]
    df_clean = df_clean.dropna()
    final_shape = df_clean.shape[0]
    
    if initial_shape != final_shape:
        print(f"Dropped {initial_shape - final_shape} rows with null values")
    
    print(f"Final cleaned data shape: {df_clean.shape}")
    return df_clean


def split_data(df: pd.DataFrame, target_column: str = 'Price', test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into training and testing sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(target_column, axis=1)
    y = df[[target_column]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    return info


def show_unique_values(df: pd.DataFrame, columns: list = None):
    """
    Display unique values for specified columns or all categorical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to analyze
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for column in columns:
        if column in df.columns:
            unique_values = df[column].unique()
            print(f"Unique values in '{column}':\n{unique_values}\n")


def calculate_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing data ratios for all columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with missing data ratios
    """
    missing_ratio = (df.isnull().sum() / len(df)) * 100
    missing_data = missing_ratio[missing_ratio > 0]
    
    if len(missing_data) > 0:
        missing_data_df = pd.DataFrame({'Missing Ratio %': missing_data}).sort_values('Missing Ratio %', ascending=False)
        return missing_data_df
    else:
        print("No missing values found in the dataset")
        return pd.DataFrame()


def unique_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate unique value counts and percentages for all columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with unique counts and percentages
    """
    num_unique = df.nunique().sort_values(ascending=False)
    pct_unique = (df.nunique().sort_values(ascending=False) / len(df) * 100).round(3)
    pct_unique = pct_unique.astype(str) + '%'
    
    unique = pd.DataFrame({
        'Unique Count': num_unique,
        'Percentage Unique': pct_unique
    })

    return unique