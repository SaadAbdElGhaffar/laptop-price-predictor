"""
Visualization functions for data exploration and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


def num_plot_dist(df, num_features):
    """
    Plot distribution of numerical features using histogram and boxplot
    
    Args:
        df (pd.DataFrame): Input dataframe
        num_features (list): List of numerical feature names
    """
    fig, axes = plt.subplots(len(num_features), 2, figsize=(15, len(num_features) * 5))
    if len(num_features) == 1:
        axes = [axes] 
    for i, column in enumerate(num_features):
        sns.histplot(data=df, x=column, ax=axes[i][0], kde=True, palette="Blues")
        axes[i][0].set_title(f'Histogram with KDE for {column}')
        
        sns.boxplot(data=df, x=column, ax=axes[i][1], palette="Blues")
        axes[i][1].set_title(f'Box Plot for {column}')
    
    plt.tight_layout()
    plt.show()


def plot_dist_cat(df, cat_features):
    """
    Plot distribution of categorical features using count plots
    
    Args:
        df (pd.DataFrame): Input dataframe
        cat_features (list): List of categorical feature names
    """
    fig, axes = plt.subplots(len(cat_features), 1, figsize=(15, len(cat_features) * 5))
    if len(cat_features) == 1:
        axes = [axes]
    for i, column in enumerate(cat_features):
        order = df[column].value_counts().index  
        sns.countplot(data=df, x=column, order=order, ax=axes[i], palette="Blues_d", saturation=0.8)
        axes[i].set_title(f'Count Plot for {column}')
        axes[i].tick_params(axis='x', rotation=90) 

    plt.tight_layout()
    plt.show()


def label_plot_dist_cat(df, cat_features, target):
    """
    Plot categorical features vs target variable using bar plots
    
    Args:
        df (pd.DataFrame): Input dataframe
        cat_features (list): List of categorical feature names
        target (str): Target variable name
    """
    fig, axes = plt.subplots(len(cat_features), 1, figsize=(15, len(cat_features) * 5))
    if len(cat_features) == 1:
        axes = [axes]
    for i, column in enumerate(cat_features):
        order = df.groupby(column)[target].mean().sort_values(ascending=False).index
        sns.barplot(data=df, x=column, y=target, order=order, ax=axes[i], palette="Blues_d", saturation=0.8)
        axes[i].set_title(f'Bar Plot for {column} with Avg {target}')
        axes[i].tick_params(axis='x', rotation=90) 

    plt.tight_layout()
    plt.show()


def label_plot_dist_num(df, num_features, target):
    """
    Plot numerical features vs target variable using scatter plots
    
    Args:
        df (pd.DataFrame): Input dataframe
        num_features (list): List of numerical feature names
        target (str): Target variable name
    """
    fig, axes = plt.subplots(len(num_features), 1, figsize=(15, len(num_features) * 5))
    if len(num_features) == 1:
        axes = [axes]
    for i, column in enumerate(num_features):
        sns.scatterplot(data=df, x=column, y=target, ax=axes[i], palette="Blues")
        axes[i].set_title(f'Scatter Plot for {column} vs {target}')

    plt.tight_layout()
    plt.show()


def skewness_heatmap(df, num_features, center_value=0):
    """
    Create heatmap showing skewness of numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        num_features (list): List of numerical feature names
        center_value (float): Center value for the heatmap
    """
    # Calculate skewness and prepare DataFrame
    skewness_df = pd.DataFrame(df[num_features].skew().sort_values(), columns=["Skewness"])
    
    # Create the heatmap
    sns.heatmap(skewness_df, cmap='coolwarm', annot=True, cbar=True,
                center=center_value, vmin=-abs(skewness_df).max().max(), vmax=abs(skewness_df).max().max())
    
    plt.title("Skewness Heatmap")
    plt.xlabel("Skewness")
    plt.show()


def plot_correlation_heatmap(df, num_features):
    """
    Plot correlation heatmap for numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        num_features (list): List of numerical feature names
    """
    corr = df[num_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()


def plot_regression_performance(y_test, y_pred, color='m'):
    """
    Plot regression model performance with multiple diagnostic plots
    
    Args:
        y_test: True values
        y_pred: Predicted values
        color: Color for the plots
    """
    y_test_series = pd.Series(y_test.iloc[:, 0]).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)
    
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

    # Actual vs Predicted scatter plot
    axs[0].scatter(y_test_series, y_pred_series, color=color)
    axs[0].plot([y_test_series.min(), y_test_series.max()], [y_test_series.min(), y_test_series.max()], 'k--', lw=3, color='k')
    axs[0].set_xlabel('Actual Values')
    axs[0].set_ylabel('Predicted Values')
    axs[0].set_title('Scatter Plot with Regression Line')

    # Histogram of errors
    errors = y_test_series - y_pred_series
    axs[1].hist(errors, bins=50, color=color)
    axs[1].axvline(x=errors.median(), color='k', linestyle='--', lw=3)
    axs[1].set_xlabel('Errors')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of Errors')

    # Residual plot
    axs[2].scatter(y_pred_series, errors, color=color)
    axs[2].axhline(y=0, color='k', linestyle='-', lw=3)
    axs[2].set_xlabel('Predicted Values')
    axs[2].set_ylabel('Errors')
    axs[2].set_title('Residual Plot')

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature', palette='Blues_d')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature importances")


def plot_learning_curve(model, X, y, cv=5):
    """
    Plot learning curve to diagnose overfitting/underfitting
    
    Args:
        model: Model to evaluate
        X, y: Training data
        cv: Cross-validation folds
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_model_comparison_plot(results_df, metric='Test Score'):
    """
    Create comparison plot for multiple models
    
    Args:
        results_df: DataFrame with model results
        metric: Metric to compare
    """
    plt.figure(figsize=(12, 8))
    results_sorted = results_df.sort_values(metric, ascending=True)
    
    sns.barplot(data=results_sorted.reset_index(), x=metric, y='Algorithm', palette='Blues_d')
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel(metric)
    plt.tight_layout()
    plt.show()