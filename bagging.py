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
from google_sheets_utils import csv_to_sheets        # Custom function to transform .csv into a spreadsheet
import curses                                        # Create text-based user interfaces (TUIs) in the terminal. 

"""
2. Load and explore the dataset
Load data using the kaggle_connect() function
"""

# Create a new function so we can access kaggle_connect file
def run_kaggle_download():
    rkd = curses.wrapper(kaggle_connect)
    return rkd

data = run_kaggle_download()
dataf = pd.DataFrame(data)
print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
print("-" * 32)
print("\n")

# Print the columns to understand the structure of the data
print("\nSelect columns from the list:")
for column in dataf.columns:
    print(f"- Column '{column}': type {dataf[column].dtype}")
print("-" * 32)

"""
3. Preprocess the data
Prepare the dataset to make it suitable for modeling.
"""

def select_columns():
    """
    Function to interactively select the columns to be used for analysis.
    """
    print("\nSelect the columns to be used")
    print("-" * 32)
    selected_columns = []
    
    while True:
        col = input("Enter column name (or type 'e' to exit): ")
        if col.lower() == "e":
            break
        else:
            selected_columns.append(col)
    
    return selected_columns

# Call the function to choose columns and create a subset of the data
selected_columns = select_columns()
subset_data = data[selected_columns]

# Create a new DataFrame with the selected columns
df = pd.DataFrame(subset_data)

# Identify unique values in each column to decide how to encode or process the data
print("\n")
print("-" * 32)
print("Unique values in columns:")
for column in df.columns:
    print(f"Column '{column}': {df[column].nunique()} unique values")

# Identify columns with missing values
print("\n")
def identify_null_columns():
    """
    Function to identify columns with missing values.
    """
    columns_with_null = df.columns[df.isnull().any()].to_list()
    print("-" * 32)
    print("Columns with missing values:")
    types_of_null_columns = df[columns_with_null].dtypes
    return types_of_null_columns

print(identify_null_columns())

# Impute missing values
for column in df:
    # Impute missing values with the mean for numeric columns
    if df[column].dtype in ['float64', 'int64']:
        imputer = SimpleImputer(strategy='mean')
        df[column] = imputer.fit_transform(df[column].values.reshape(-1, 1))
        print(identify_null_columns())
    else:
        print(f"Column '{column}' is not numeric (int or float).")

# Encode categorical variables with 2 or fewer unique values
print("\n")
print("-"*32)
for column in df.columns:
    if df[column].nunique() <= 2:
        #Showing the original values
        print(f"{column} : {df[column].unique()}")
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        
        #showing the encoded values
        print(f"{column} : {df[column].unique()}")

# User input for target column
target_column = input("\nEnter the column to be used as the target variable: ")        

# Split the data into features (X) and target (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

"""
4. Split the data into training and testing sets
Ensures fair evaluation of the model
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
5. Train the Bagging model
Build a Bagging model using decision trees as the base estimator.
"""

# Initialize the Bagging model
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

# Train the model
bagging_model.fit(X_train, y_train)

"""
6. Evaluate the model
Measure the accuracy of the model on the test set.
"""

# Make predictions on the test set
y_pred = bagging_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nThe accuracy of the Bagging model on the dataset is: {accuracy:.2f}")

"""
7. Save dataset as .csv file
"""

output_name = input("\nEnter the name to save the .csv file: ")
output_dataset = df.to_csv(f"./save/{output_name}.csv", index=False)

print(f"\nThe dataset has been saved as '{output_name}.csv'")

"""
8. Save data set into google spreadsheet
"""

csv_to_sheets()