"""
Data Preparation & Feature Engineering for CMAPSS Dataset (FD001 CSV)

Objective:
    - Load, preprocess, and prepare CMAPSS dataset for model training
    - Create rolling window sequences
    - Compute RUL targets
    - Save processed data as .npz for easy model training

File Path:
    C:/Users/DELL/OneDrive/Desktop/Sample-Test/data/train_FD001.csv
"""

import pandas as pd
import numpy as np
import os

# -----------------------
# CONFIGURATION
# -----------------------
FILE_PATH = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone2 Durga Veera Prasad V\Milestone2 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001.csv"
WINDOW_SIZE = 30  # rolling window length
OUTPUT_FILE = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone2 Durga Veera Prasad V\Milestone2 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001_processed.npz"

# -----------------------
# STEP 1: Load CSV
# -----------------------
def load_data(file_path):
    """Load CMAPSS CSV file."""
    df = pd.read_csv(file_path)
    return df

# -----------------------
# STEP 2: Compute RUL
# -----------------------
def add_rul(df):
    """Compute Remaining Useful Life (RUL) per cycle."""
    max_cycle = df.groupby("unit_nr")["time_in_cycles"].max().reset_index()
    max_cycle.columns = ["unit_nr", "max_cycles"]
    df = df.merge(max_cycle, on="unit_nr")
    df["RUL"] = df["max_cycles"] - df["time_in_cycles"]
    df.drop("max_cycles", axis=1, inplace=True)
    return df

# -----------------------
# STEP 3: Normalize and Select Features
# -----------------------
def preprocess(df):
    """Normalize features and select relevant sensor columns."""
    sensor_cols = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + sensor_cols

    # Fill missing values for op_setting_3 (if any) with 0
    if "op_setting_3" in df.columns:
        df["op_setting_3"] = df["op_setting_3"].fillna(0)

    # Normalize features
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    return df, feature_cols

# -----------------------
# STEP 4: Generate Sequences
# -----------------------
def generate_sequences(df, seq_length, feature_cols):
    """Create rolling window sequences and labels."""
    X, y = [], []
    for unit in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit]
        unit_features = unit_df[feature_cols].values
        unit_labels = unit_df["RUL"].values

        for i in range(len(unit_df) - seq_length + 1):
            X.append(unit_features[i:i+seq_length])
            y.append(unit_labels[i + seq_length - 1])

    return np.array(X), np.array(y)

# -----------------------
# STEP 5: Data Verification
# -----------------------
def verify_data(X, y, df, feature_cols):
    """Check for missing values and preview data."""
    print("\n Data Verification:")
    print("Missing values per column:\n", df.isnull().sum())
    print(f"Feature columns used: {feature_cols}")
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print("Sample RUL values:", y[:10])

# -----------------------
# STEP 6: Save Processed Data as NPZ
# -----------------------
def save_data_npz(X, y, output_file):
    """Save X and y together in a single .npz file."""
    np.savez(output_file, X=X, y=y)
    print(f"\n Data saved as NPZ: {output_file}")

# -----------------------
# MAIN PIPELINE
# -----------------------
if __name__ == "__main__":
    # Load dataset
    df = load_data(FILE_PATH)

    # Compute RUL
    df = add_rul(df)

    # Preprocess & normalize features
    df, feature_cols = preprocess(df)

    # Generate rolling window sequences
    X, y = generate_sequences(df, WINDOW_SIZE, feature_cols)

    # Verify data
    verify_data(X, y, df, feature_cols)

    # Save as NPZ
    save_data_npz(X, y, OUTPUT_FILE)

    print("\n Data preparation completed")
