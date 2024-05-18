# Setting the seed
import pandas as pd
from pycaret.regression import *
import random
import numpy as np
import sweetviz as sv
import pickle
from sklearn.tree import _tree
from functions_code import *
random.seed(666) 

# Functions Section (Holding function code for use during the script):

# Function to look at nans in data
def columns_with_nans(df):
    """
    Returns a list of column names in the DataFrame that contain NaN values.
    
    Parameters:
    - df: pandas DataFrame
    
    Returns:
    - List of column names containing NaN values
    """
    return df.columns[df.isnull().any()].tolist()

#Function to get unique column types
def unique_column_types(df):
    """
    Returns a dictionary with unique column types as keys and
    the number of columns with each type as values.
    
    Parameters:
    - df: pandas DataFrame
    
    Returns:
    - Dictionary of unique column types and their counts
    """
    column_types = df.dtypes
    unique_types = column_types.unique()
    type_counts = {type_: sum(column_types == type_) for type_ in unique_types}
    return type_counts

def cap_values(data, columns_to_cap, method="std_dev", num_std_devs=3):
    """
    Cap values in specific columns of the DataFrame that are greater than a specified number 
    of standard deviations away from the mean or 1.5 times the IQR to the range of mean ± X standard deviations, 
    where X is a small random deviation number.

    If data is normal, use the std_dev approach. If it is skewed then the IQR range approach is better.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - columns_to_cap: List of column names to be capped
    - method: Method for capping, either "std_dev" (standard deviation) or "iqr" (interquartile range) (default: "std_dev")
    - num_std_devs: Number of standard deviations away from the mean to cap the values (default: 3)
    
    Returns:
    - Capped DataFrame
    """
    capped_data = data.copy() 
    
    for column in columns_to_cap:
        # Calculate the mean and standard deviation or quartiles of the column
        if method == "std_dev":
            column_mean = np.mean(capped_data[column])
            column_std = np.std(capped_data[column])
            upper_bound = column_mean + num_std_devs * column_std
            lower_bound = column_mean - num_std_devs * column_std
        elif method == "iqr":
            q25, q75 = np.percentile(capped_data[column], [25, 75])
            iqr = q75 - q25
            upper_bound = q75 + 1.5 * iqr
            lower_bound = q25 - 1.5 * iqr
        else:
            raise ValueError("Invalid method. Choose 'std_dev' or 'iqr'.")
        
        # Generate a random small deviation for the column, can be used to give slightly differnet capping results
        # Can help with numpy floating point errors that can occur if values are causing underflow errors
        small_deviation = np.random.uniform(low=0.001, high=0.01)
        
        # Cap values greater than the upper bound to the upper bound - small deviation
        capped_data[column] = np.where(capped_data[column] > upper_bound,
                                       upper_bound - small_deviation,
                                       capped_data[column])
        
        # Cap values less than the lower bound to the lower bound + small deviation
        capped_data[column] = np.where(capped_data[column] < lower_bound,
                                       lower_bound + small_deviation,
                                       capped_data[column])
    
    return capped_data


def summary_for_column(data, column_name="CCF", method="std_dev", threshold=3):
    """
    Generates a summary for a column including the number of results
    more than a specified threshold away from the mean or quartiles.

    Parameters:
    - data: pandas DataFrame containing the data
    - column_name: Name of the column for which to generate the summary (default: "CCF")
    - method: Method for outlier detection, either "std_dev" (standard deviation) or "iqr" (interquartile range) (default: "std_dev")
    - threshold: Number of standard deviations away from the mean or interquartile range to consider as an outlier (default: 3)

    Returns:
    - Dictionary containing the summary information
    """
    column_data = data[column_name]
    
    # Calculate mean and standard deviation or quartiles
    if method == "std_dev":
        column_mean = np.mean(column_data)
        column_std = np.std(column_data)
        upper_bound = column_mean + threshold * column_std
        lower_bound = column_mean - threshold * column_std
    elif method == "iqr":
        q25, q75 = np.percentile(column_data, [25, 75])
        iqr = q75 - q25
        upper_bound = q75 + threshold * iqr
        lower_bound = q25 - threshold * iqr
    else:
        raise ValueError("Invalid method. Choose 'std_dev' or 'iqr'.")
    
    # Count the number of values more than the threshold away from the mean or quartiles
    num_outliers = np.sum((column_data > upper_bound) | (column_data < lower_bound))
    
    # Generate summary dictionary
    summary = {
        "Mean": np.round(column_mean, 2) if method == "std_dev" else None,
        "Standard Deviation": np.round(column_std, 2) if method == "std_dev" else None,
        "Quartile 25": np.round(q25, 2) if method == "iqr" else None,
        "Quartile 75": np.round(q75, 2) if method == "iqr" else None,
        "IQR": np.round(iqr, 2) if method == "iqr" else None,
        "Upper Bound (Mean + {} SD)".format(threshold): np.round(upper_bound, 2),
        "Lower Bound (Mean - {} SD)".format(threshold): np.round(lower_bound, 2),
        "Number of Outliers (>{})".format(threshold): num_outliers
    }
    
    return summary

def extract_top_features(model, data, X):
    """
    Extracts the top X features based on their importance scores from a trained model.

    Args:
    - model: A trained model from PyCaret.
    - data: The dataset used for training the model.
    - X: The number of top features to extract.

    Returns:
    - top_features_df: A DataFrame containing the top X features and their importance scores.
    """
    # Get feature importance
    feature_importance = model.feature_importances_

    # Create a dictionary to map feature names to their importance scores
    feature_importance_dict = {col: importance for col, importance in zip(data.columns, feature_importance)}

    # Sort the dictionary by importance scores in descending order
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract the top X features
    top_features = sorted_feature_importance[:X]

    # Create DataFrame from top_features list
    top_features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])

    return top_features_df


def get_split_values(model, data, feature_name):
    """
    Extracts the split values used for a given feature in each decision tree of a Random Forest model.

    Args:
    - model: The trained Random Forest model.
    - data: The dataset used for training the model.
    - feature_name: The name of the feature for which split values are extracted.

    Returns:
    - split_values: A list of lists containing the split values used for the specified feature in each tree.
    """
    feature_index = data.columns.get_loc(feature_name)
    split_values = []

    for estimator in model.estimators_:
        tree = estimator.tree_
        feature_split_values = []
        def traverse(node):
            nonlocal feature_split_values
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                return
            if tree.feature[node] == feature_index:
                feature_split_values.append(tree.threshold[node])
            traverse(tree.children_left[node])
            traverse(tree.children_right[node])
        traverse(0)
        split_values.append(feature_split_values)

    return split_values


def set_missing_to_na(dataframe, columns):
    """
    Set missing values to 'NA' in specified columns of a DataFrame.

    Parameters:
    dataframe (DataFrame): Input DataFrame.
    columns (list): List of column names to apply the operation to.

    Returns:
    DataFrame: DataFrame with missing values replaced by 'NA'.
    """

    # Make a copy of the DataFrame to avoid modifying the original
    df = dataframe.copy()

    # Iterate over the specified columns
    for col in columns:
        # Set missing values to 'NA' in the column
        df[col] = df[col].fillna('NA')

    return df
