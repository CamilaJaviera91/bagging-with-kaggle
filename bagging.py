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
