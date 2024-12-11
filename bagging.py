
#Import necessary libraries

import pandas as pd                                                         
# For creating and manipulating DataFrames and Series
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
# Combines base estimators to improve robustness, especially for high-variance models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# Implements a decision tree classifier
from sklearn.model_selection import train_test_split
# Splits the dataset into training and test sets
from sklearn.metrics import accuracy_score
# Calculates the accuracy of a model
from sklearn.impute import SimpleImputer
# Handles missing (NaN) values in the dataset
from sklearn.preprocessing import LabelEncoder
# Encodes categorical labels (non-numeric variables) into numeric values
from kaggle_connect import kaggle_connect
# Custom function to fetch the dataset using Kaggle API
from google_sheets_utils import csv_to_sheets
# Custom function to transform .csv into a spreadsheet
import curses
# Create text-based user interfaces (TUIs) in the terminal. 
import os
# Provides a way to interact with the operating system. 

def run_kaggle_download():
    """
    Wrapper function to run Kaggle connect using curses.
    """
    return curses.wrapper(kaggle_connect)

def menu(stdscr):
    stdscr.clear()

    # Variables
    dataf = None
    df = None
    model = None
    y = None

    # Step 1: Download Dataset
    stdscr.addstr("Step 1: Download Kaggle Dataset\n")
    stdscr.addstr("Press Enter to start...\n")
    stdscr.refresh()
    stdscr.getstr()

    dataf = run_kaggle_download()
    if dataf is None or dataf.empty:
        stdscr.addstr("Failed to load data or dataset is empty. Exiting...\n")
        stdscr.refresh()
        stdscr.getstr()
        return

    stdscr.addstr(f"\nLoaded data with {dataf.shape[0]} rows and {dataf.shape[1]} columns.\n")
    stdscr.addstr("Press Enter to continue...\n")
    stdscr.refresh()
    stdscr.getstr()

    # Step 2: Preprocessing and Selecting Features
    stdscr.clear()
    stdscr.addstr("Step 2: Selecting Features and Preprocessing\n")
    stdscr.addstr("Columns available:\n")
    for idx, col in enumerate(dataf.columns):
        stdscr.addstr(f"{idx + 1}. {col}\n")
    stdscr.addstr("Enter column numbers to select as features (comma-separated, without spaces):\n")
    stdscr.refresh()

    selected_columns = stdscr.getstr().decode('utf-8').strip().split(',')
    selected_columns = [dataf.columns[int(idx) - 1] for idx in selected_columns if idx.isdigit()]
    if not selected_columns:
        stdscr.addstr("No columns selected. Exiting...\n")
        stdscr.refresh()
        stdscr.getstr()
        return

    df = dataf[selected_columns]
    stdscr.addstr(f"Selected columns: {', '.join(selected_columns)}\n")
    stdscr.addstr("Press Enter to preprocess data...\n")
    stdscr.refresh()
    stdscr.getstr()

    # Preprocess Data
    imputer = SimpleImputer(strategy="mean")
    label_encoders = {}
    for column in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    target_column = selected_columns[-1]
    X, y = df.drop(target_column, axis=1), df[target_column]

    # Step 3: Choose Model Type
    stdscr.clear()
    stdscr.addstr("Step 3: Choose Model Type\n")
    stdscr.addstr("1. Classification\n")
    stdscr.addstr("2. Regression\n")
    stdscr.refresh()

    model_choice = stdscr.getstr().decode('utf-8').strip()
    if model_choice == "1":  # Classification
        if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 20:
            stdscr.addstr("The target column contains continuous values.\n")
            stdscr.addstr("Would you like to:\n")
            stdscr.addstr("1. Switch to regression.\n")
            stdscr.addstr("2. Automatically convert the target column into categories.\n")
            stdscr.refresh()
            correction_choice = stdscr.getstr().decode('utf-8').strip()
            if correction_choice == "1":
                model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=10, random_state=42)
            elif correction_choice == "2":
                y = pd.cut(y, bins=3, labels=["Low", "Medium", "High"])
                model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
            else:
                stdscr.addstr("Invalid choice. Exiting...\n")
                stdscr.refresh()
                stdscr.getstr()
                return
        else:
            model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
    elif model_choice == "2":  # Regression
        model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=10, random_state=42)
    else:
        stdscr.addstr("Invalid choice. Exiting...\n")
        stdscr.refresh()
        stdscr.getstr()
        return

    # Step 4: Train Model
    stdscr.clear()
    stdscr.addstr("Step 4: Training Model\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)

    stdscr.addstr("Model trained successfully.\n")
    stdscr.addstr("Press Enter to evaluate the model...\n")
    stdscr.refresh()
    stdscr.getstr()

    # Step 5: Evaluate Model
    stdscr.clear()
    stdscr.addstr("Step 5: Evaluating Model\n")
    if model_choice == "1":
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        stdscr.addstr(f"The accuracy of the model is: {accuracy:.2f}\n")
    else:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        stdscr.addstr(f"The Mean Squared Error (MSE) of the model is: {mse:.2f}\n")
    stdscr.addstr("Press Enter to save dataset to CSV...\n")
    stdscr.refresh()
    stdscr.getstr()

    # Step 6: Save Dataset to CSV
    stdscr.clear()
    stdscr.addstr("Step 6: Saving Dataset to CSV\n")
    os.makedirs('./save', exist_ok=True)
    
    stdscr.addstr("Enter the filename (without extension): ")
    stdscr.refresh()
    file = stdscr.getstr().decode('utf-8').strip()  # Capturar el nombre del archivo ingresado por el usuario
    
    if not file:  # Usar un nombre predeterminado si el usuario no ingresa nada
        file = "output_dataset"

    df.to_csv(f'./save/{file}.csv', index=False)  # Guardar el archivo con el nombre proporcionado
    stdscr.addstr(f"Dataset saved as './save/{file}.csv'\n")
    stdscr.addstr("Press Enter to export to Google Sheets...\n")
    stdscr.refresh()
    stdscr.getstr()

    # Step 7: Export Dataset to Google Sheets
    stdscr.clear()
    stdscr.addstr("Step 7: Exporting Dataset to Google Sheets\n")
    csv_to_sheets()
    stdscr.addstr("Dataset exported to Google Sheets successfully.\n")
    stdscr.addstr("Press Enter to finish...\n")
    stdscr.refresh()
    stdscr.getstr()

    # End of the program
    stdscr.clear()
    stdscr.addstr("All steps completed successfully! Exiting...\n")
    stdscr.refresh()
    stdscr.getstr()


curses.wrapper(menu)