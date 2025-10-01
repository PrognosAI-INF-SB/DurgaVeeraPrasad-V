import pandas as pd

# File paths
txt_file = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone1 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001.txt"
csv_file = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone1 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001.csv"

# Column names (same as CMAPSS)
col_names = [
    "unit_nr", "time_in_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

# Load TXT
df = pd.read_csv(txt_file, sep=r"\s+", header=None, names=col_names)

# Save as CSV
df.to_csv(csv_file, index=False)

print("Conversion complete:", csv_file)
