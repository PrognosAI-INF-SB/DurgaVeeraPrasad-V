import pandas as pd
import numpy as np
import os
import sys

# --- PATH CONFIGURATION ---

BASE_PATH = r'C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\data'

# --- File Names ---
RAW_FILES = {
    # (Input Raw File Name, Output Processed File Name)
    'train': ('train_FD001.txt', 'train_FD001.csv'),
    # Use .csv extension for the inputs, matching your uploaded files
    'test': ('test_FD001.txt', 'test_FD001.csv'),  
    'rul': ('RUL_FD001.txt', 'RUL_FD001.csv')        
}

# --- Common Column Configuration ---
INDEX_COLUMNS = ['unit_number', 'cycle']
OP_SETTING_COLUMNS = [f'op_setting_{i}' for i in range(1, 4)]
SENSOR_COLUMNS = [f's{i}' for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS

DROPPED_COLUMNS = ['op_setting_3', 's1', 's5', 's10', 's16', 's18', 's19']
FEATURE_COLUMNS_TO_KEEP = [col for col in OP_SETTING_COLUMNS + SENSOR_COLUMNS if col not in DROPPED_COLUMNS]
FINAL_COLUMNS = INDEX_COLUMNS + FEATURE_COLUMNS_TO_KEEP

def process_sensor_data(input_name, output_name):
    """Loads raw CMAPSS sensor data (train/test), cleans it, and saves as CSV."""
    input_file_path = os.path.join(BASE_PATH, input_name)
    output_file_path = os.path.join(BASE_PATH, output_name)
    
    print(f"\nProcessing sensor data file: {input_name}...")
    
    try:
        # 1. Load the file (space-separated is correct for all FD001 data files)
        df = pd.read_csv(input_file_path, sep='\s+', header=None)
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at {input_file_path}. Please place the file in the 'data' folder.")
        # We must exit if a file is missing, as subsequent steps will rely on it.
        sys.exit(1)
    
    # Clean up trailing columns if present
    if df.shape[1] > len(ALL_COLUMNS):
        df.drop(columns=df.columns[len(ALL_COLUMNS):], inplace=True)
    
    # Assign the 26 column names
    df.columns = ALL_COLUMNS
    
    # Select only the required columns (Index + Selected Features)
    df_processed = df[FINAL_COLUMNS].copy()
    df_processed.to_csv(output_file_path, index=False)
    
    print(f"-> Successfully saved {output_name} (Shape: {df_processed.shape})")

def process_rul_targets(input_name, output_name):
    """Loads the RUL target file, assigns a column name, and saves as CSV."""
    input_file_path = os.path.join(BASE_PATH, input_name)
    # FIX: Use the variable name 'output_name' from the function argument here:
    output_file_path = os.path.join(BASE_PATH, output_name)
    
    print(f"\nProcessing RUL target file: {input_name}...")
    
    try:
        # 1. Load the file (simple single column file)
        df = pd.read_csv(input_file_path, header=None)
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at {input_file_path}. Please place the file in the 'data' folder.")
        sys.exit(1)
        
    df.columns = ['RUL']
    df.to_csv(output_file_path, index=False)
    
    print(f"-> Successfully saved {output_name} (Shape: {df.shape})")

if __name__ == '__main__':
    if not os.path.isdir(BASE_PATH):
        print(f"FATAL ERROR: The BASE_PATH '{BASE_PATH}' does not exist.")
        sys.exit(1)
        
    print(f"Starting data conversion. Checking files in: {BASE_PATH}")
    
    # 1. Process Training Sensor Data
    process_sensor_data(RAW_FILES['train'][0], RAW_FILES['train'][1])

    # 2. Process Test Sensor Data
    process_sensor_data(RAW_FILES['test'][0], RAW_FILES['test'][1])

    # 3. Process Test RUL Targets
    process_rul_targets(RAW_FILES['rul'][0], RAW_FILES['rul'][1])
    
    print("\n--- Data Conversion Finished ---")
    print("You can now proceed to run the model training script: python model_trainer.py")