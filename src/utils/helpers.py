"""
Utility functions for data analysis and feature selection
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor


def drop_highly_correlated_features(df, target_col, num_cols, corr_threshold=0.7, exclude_cols=None):
    """
    Drop highly correlated features based on their correlation with target variable
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        num_cols (list): List of numerical columns
        corr_threshold (float): Correlation threshold
        exclude_cols (list): Columns to exclude from dropping
        
    Returns:
        pd.DataFrame: DataFrame with highly correlated features removed
    """
    if exclude_cols is None:
        exclude_cols = []

    df_copy = df.copy()
    # Correlation matrix for the numerical columns
    corr_matrix = df_copy[num_cols].corr()
    
    features_to_drop = []

    for row_idx in range(corr_matrix.values.shape[0]):
        for col_idx in range(row_idx + 1, corr_matrix.values.shape[0]):
            if np.abs(corr_matrix.values[row_idx, col_idx]) > corr_threshold:
                # Calculate correlation of each feature with the target variable
                var_row_corr_with_response = np.abs(np.corrcoef(df_copy[target_col], df_copy[num_cols[row_idx]])[0, 1])
                var_col_corr_with_response = np.abs(np.corrcoef(df_copy[target_col], df_copy[num_cols[col_idx]])[0, 1])
                
                if var_row_corr_with_response > var_col_corr_with_response:
                    if num_cols[col_idx] not in exclude_cols:
                        print(f"We will drop the column '{num_cols[col_idx]}' due to lower correlation with target '{target_col}'")
                        features_to_drop.append(num_cols[col_idx])
                else:
                    if num_cols[row_idx] not in exclude_cols:
                        print(f"We will drop the column '{num_cols[row_idx]}' due to lower correlation with target '{target_col}'")
                        features_to_drop.append(num_cols[row_idx])
    
    # Remove duplicates and drop features
    features_to_drop = list(set(features_to_drop))
    if features_to_drop:
        df_copy = df_copy.drop(columns=features_to_drop)
    
    return df_copy


def drop_high_vif_features(X_train, vif_threshold=20):
    """
    Drop features with high Variance Inflation Factor (VIF) to reduce multicollinearity
    
    Args:
        X_train (pd.DataFrame): Training data
        vif_threshold (float): VIF threshold above which features are dropped
        
    Returns:
        tuple: (cleaned_dataframe, list_of_dropped_features)
    """
    feats_high_vif_to_be_dropped = []
    cleaned_df_from_multi_coll = X_train.copy()
    num_cols_removed = 0
    
    # Loop through the features of the dataframe
    for origin_df_feat_index in range(len(X_train.columns)):
        num_cols_removed = X_train.shape[1] - cleaned_df_from_multi_coll.shape[1]
        cleaned_df_feat_index = origin_df_feat_index - num_cols_removed
        
        # Break if we've processed all remaining columns
        if cleaned_df_feat_index >= len(cleaned_df_from_multi_coll.columns):
            break
            
        # Calculate the VIF for the current feature
        try:
            VIF_FEAT = variance_inflation_factor(cleaned_df_from_multi_coll.values, cleaned_df_feat_index)
            
            # If VIF exceeds the threshold, drop the feature
            if VIF_FEAT > vif_threshold:
                feature_to_drop = cleaned_df_from_multi_coll.columns[cleaned_df_feat_index]
                feats_high_vif_to_be_dropped.append(feature_to_drop)
                cleaned_df_from_multi_coll = cleaned_df_from_multi_coll.drop(columns=[feature_to_drop])
                print(f"Dropped {feature_to_drop} due to high VIF: {VIF_FEAT:.2f}")
        except:
            # Skip if VIF calculation fails
            continue
    
    return cleaned_df_from_multi_coll, feats_high_vif_to_be_dropped


def shapiro_test_normality(df, num_cols, sample_size=500, alpha=0.001):
    """
    Test normality of numerical columns using Shapiro-Wilk test
    
    Args:
        df (pd.DataFrame): Input dataframe
        num_cols (list): List of numerical columns
        sample_size (int): Sample size for testing
        alpha (float): Significance level
        
    Returns:
        tuple: (features_for_std_scaling, features_for_minmax_scaling)
    """
    feats_std_scale = []
    feats_min_max_scale = []
    
    # Sample a subset of the data if dataset is large
    if len(df) > sample_size:
        sample_data = df.sample(replace=False, n=sample_size, random_state=42)
    else:
        sample_data = df

    for col in num_cols:
        if col in sample_data.columns:
            # Perform Shapiro-Wilk test
            stat, p = shapiro(sample_data[col])
            print(f'{col}: W_Statistic={stat:.3f}, p={p:.8f}')

            # Check for normality
            if p > alpha:
                print(f'{col} looks like Gaussian (fail to reject H0)')
                feats_std_scale.append(col)
            else:
                print(f'{col} does not look Gaussian (reject H0)')
                feats_min_max_scale.append(col)

    return feats_std_scale, feats_min_max_scale


def transform_skewed_columns(df, skew_threshold=1):
    """
    Apply log transformation to skewed columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        skew_threshold (float): Skewness threshold above which transformation is applied
        
    Returns:
        pd.DataFrame: Dataframe with transformed skewed columns
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    transformed_cols = []
    for col in numeric_cols:
        skewness = df_copy[col].skew()
        if abs(skewness) > skew_threshold:
            df_copy[col] = np.log1p(df_copy[col])
            transformed_cols.append(col)
            print(f"Applied log transformation to {col} (skewness: {skewness:.3f})")
    
    if not transformed_cols:
        print("No columns required transformation")
    
    return df_copy


def detect_outliers_iqr(df, columns, multiplier=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to check for outliers
        multiplier (float): IQR multiplier for outlier detection
        
    Returns:
        dict: Dictionary with outlier information for each column
    """
    outlier_info = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }
            
            print(f"{col}: {len(outliers)} outliers ({(len(outliers) / len(df)) * 100:.2f}%)")
    
    return outlier_info


def winsorize_outliers(df, columns, limits=(0.05, 0.05)):
    """
    Winsorize outliers by capping extreme values
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to winsorize
        limits (tuple): Lower and upper percentile limits
        
    Returns:
        pd.DataFrame: Dataframe with winsorized values
    """
    from scipy.stats import mstats
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            original_values = df_copy[col].copy()
            df_copy[col] = mstats.winsorize(df_copy[col], limits=limits)
            
            changed_count = (original_values != df_copy[col]).sum()
            print(f"Winsorized {changed_count} values in {col}")
    
    return df_copy


def get_feature_types(df):
    """
    Categorize features by data type
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary with categorized feature lists
    """
    feature_types = {
        'numerical': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'boolean': df.select_dtypes(include=['bool']).columns.tolist()
    }
    
    return feature_types


def memory_usage_optimization(df):
    """
    Optimize memory usage by downcasting numerical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Memory-optimized dataframe
    """
    df_copy = df.copy()
    
    # Optimize integers
    int_cols = df_copy.select_dtypes(include=['int']).columns
    for col in int_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
    
    # Optimize floats
    float_cols = df_copy.select_dtypes(include=['float']).columns
    for col in float_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
    
    # Convert object columns to category if they have few unique values
    obj_cols = df_copy.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df_copy[col].nunique() / len(df_copy) < 0.5:  # If less than 50% unique values
            df_copy[col] = df_copy[col].astype('category')
    
    print(f"Memory usage reduced from {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB to {df_copy.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_copy