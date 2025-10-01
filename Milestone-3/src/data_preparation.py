"""
Data Preparation & Feature Engineering for CMAPSS Dataset (FD001)

Objective:
    - Load, preprocess, and prepare CMAPSS dataset for model training
    - Create rolling window sequences
    - Compute RUL targets
    - Save processed data as .npz for easy model training
"""

import pandas as pd
import numpy as np
import os

# -----------------------
# CONFIGURATION
# -----------------------
FILE_PATH = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001.txt"
WINDOW_SIZE = 30  # rolling window length
OUTPUT_FILE = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001_processed.npz"

# -----------------------
# STEP 1: Load CSV
# -----------------------
def load_data(file_path):
    """Load CMAPSS dataset with correct headers."""
    col_names = [
        "unit_nr", "time_in_cycles",
        "op_setting_1", "op_setting_2", "op_setting_3",
        "s1", "s2", "s3", "s4", "s5",
        "s6", "s7", "s8", "s9", "s10",
        "s11", "s12", "s13", "s14", "s15",
        "s16", "s17", "s18", "s19", "s20", "s21"
    ]

    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)

    # Debug: check loaded columns
    print(" Loaded columns:", df.columns.tolist())
    print(" First 5 rows:\n", df.head())
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
    """Normalize features and select relevant sensor columns with NaN/Inf-safe handling."""
    sensor_cols = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + sensor_cols

    # Replace Inf/-Inf with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Normalize features
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    # Ensure no NaNs/Infs remain
    df[feature_cols] = np.nan_to_num(df[feature_cols], nan=0.0, posinf=1.0, neginf=-1.0)

    return df, feature_cols

# -----------------------
# STEP 4: Generate Sequences
# -----------------------
def generate_sequences(df, seq_length, feature_cols):
    """Create rolling window sequences and labels."""
    x, y = [], []
    for unit in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit]
        unit_features = unit_df[feature_cols].values
        unit_labels = unit_df["RUL"].values

        for i in range(len(unit_df) - seq_length + 1):
            seq_x = unit_features[i:i+seq_length]
            seq_y = unit_labels[i + seq_length - 1]

            # Skip sequences with NaN values
            if np.isnan(seq_x).any() or np.isnan(seq_y):
                continue

            x.append(seq_x)
            y.append(seq_y)

    return np.array(x), np.array(y)

# -----------------------
# STEP 5: Data Verification
# -----------------------
def verify_data(x, y, df, feature_cols):
    """Check for missing values and preview data."""
    print("\n Data Verification:")
    print("Missing values per column:\n", df.isnull().sum())
    print(f"Feature columns used: {feature_cols}")
    print(f"x shape: {x.shape} | y shape: {y.shape}")
    print("Sample RUL values:", y[:10])

# -----------------------
# STEP 6: Save Processed Data as NPZ
# -----------------------
def save_data_npz(x, y, output_file):
    """Save x and y together in a single .npz file."""
    np.savez(output_file, X_train=x, y_train=y)
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
    x, y = generate_sequences(df, WINDOW_SIZE, feature_cols)

    # Verify data
    verify_data(x, y, df, feature_cols)

    # Save as NPZ
    save_data_npz(x, y, OUTPUT_FILE)

    print("\n Data preparation completed ")
