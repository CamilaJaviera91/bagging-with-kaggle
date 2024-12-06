"""
1. Import necessary libraries
"""
import pandas as pd                                  # For creating and manipulating DataFrames and Series
from sklearn.ensemble import BaggingClassifier       # Combines base estimators to improve robustness, especially for high-variance models
from sklearn.tree import DecisionTreeClassifier      # Implements a decision tree classifier
from sklearn.model_selection import train_test_split # Splits the dataset into training and test sets
from sklearn.metrics import accuracy_score           # Calculates the accuracy of a model
from sklearn.impute import SimpleImputer             # Handles missing (NaN) values in the dataset
from sklearn.preprocessing import LabelEncoder       # Encodes categorical labels (non-numeric variables) into numeric values
from kaggle_connect import kaggle_connect            # Custom function to fetch the dataset using Kaggle API

"""
2. Load and explore the dataset
Load data using the kaggle_connect() function
"""
data = kaggle_connect()
dataf = pd.DataFrame(data)
print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
print("-" * 32)
print("\n")

# Print the columns to understand the structure of the data
print("\nSelect columns from the list:")
for column in dataf.columns:
    print(f"- Column '{column}': type {dataf[column].dtype}")
print("-" * 32)