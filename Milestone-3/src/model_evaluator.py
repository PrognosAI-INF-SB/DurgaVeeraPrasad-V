import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
import os

# -----------------------------
# Paths
# -----------------------------
model_path = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\models\model_training.h5"
train_data_path = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001_processed.npz"
rul_data_path = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\RUL_FD001.csv"
plots_dir = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\outputs\Plots"
pdf_dir = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\outputs\PDF file"

# Ensure output directories exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# -----------------------------
# Load Data
# -----------------------------
train_data = np.load(train_data_path)
X_train = train_data['X_train']
y_train = train_data['y_train']

rul_df = pd.read_csv(rul_data_path)
y_test = rul_df.values.flatten()  # actual RUL values

# -----------------------------
# Predict with NaN-safe handling
# -----------------------------
y_pred = model.predict(X_train, verbose=1).flatten()

# Replace any NaN predictions with 0
y_pred = np.nan_to_num(y_pred, nan=0.0)

# -----------------------------
# Evaluation: RMSE
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f"RMSE on train set: {rmse:.4f}")

# -----------------------------
# Plot Predicted vs Actual RUL
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(y_train, label='Actual RUL', marker='o')
plt.plot(y_pred, label='Predicted RUL', marker='x')
plt.title('Predicted vs Actual RUL')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.legend()
plot_path = os.path.join(plots_dir, 'predicted_vs_actual_rul.png')
plt.savefig(plot_path)
plt.close()

# -----------------------------
# Generate PDF Report
# -----------------------------
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "RUL Prediction Model Evaluation Report", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, f"RMSE on train set: {rmse:.4f}")
pdf.ln(5)
pdf.multi_cell(0, 8, "Plots comparing predicted vs actual RUL:")
pdf.image(plot_path, x=30, w=150)

pdf_output_path = os.path.join(pdf_dir, "RUL_Model_Evaluation_Report.pdf")
pdf.output(pdf_output_path)

print(f"PDF report saved at: {pdf_output_path}")
print(f"Plot saved at: {plot_path}")
