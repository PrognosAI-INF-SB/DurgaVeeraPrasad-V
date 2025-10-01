from fpdf import FPDF
import os

# -----------------------------
# Paths
# -----------------------------
plots_dir = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\outputs\Plots"
pdf_output_path = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Milestone3 Durga Veera Prasad V\Rul-Predictive-Maintenance\outputs\PDF file\Milestone3_Documentation.pdf"

# -----------------------------
# Create PDF
# -----------------------------
pdf = FPDF('P', 'mm', 'A4')
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.add_page()
pdf.set_font("Arial", 'B', 20)
pdf.cell(0, 20, "Milestone 3 Documentation", ln=True, align='C')
pdf.ln(10)
pdf.set_font("Arial", '', 14)
pdf.multi_cell(0, 8, "Project: PrognosAI:AI-Driven Predictive Maintenance System Using Time-Series Sensor Data\n\n"
                     "Objective:\n"
                     "- Evaluate trained RUL prediction model\n"
                     "- Generate plots and PDF report\n"
                     "- Document workflow and results\n")

# -----------------------------
# Section: Data Preparation
# -----------------------------
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "1. Data Preparation", ln=True)
pdf.ln(5)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, "The CMAPSS FD001 dataset was preprocessed to generate rolling window sequences.\n"
                     "- Window size: 30 cycles\n"
                     "- Sensor features normalized\n"
                     "- RUL computed as remaining cycles\n"
                     "- Missing values handled by fillna(0)\n"
                     "- Processed data saved in NPZ format for model input\n")

# -----------------------------
# Section: Model Evaluation
# -----------------------------
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "2. Model Evaluation", ln=True)
pdf.ln(5)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, "The trained Keras model was evaluated on the train set.\n"
                     "- Loss function: MSE\n"
                     "- Optimizer: Adam\n"
                     "- RMSE on train set calculated\n")

# Include plot if exists
plot_path = os.path.join(plots_dir, 'predicted_vs_actual_rul.png')
if os.path.exists(plot_path):
    pdf.ln(5)
    pdf.multi_cell(0, 8, "Predicted vs Actual RUL Plot:")
    pdf.image(plot_path, x=25, w=160)
    pdf.ln(5)

# -----------------------------
# Section: PDF Report
# -----------------------------
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "3. PDF Report Generation", ln=True)
pdf.ln(5)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, "A PDF report was generated automatically including:\n"
                     "- RMSE value\n"
                     "- Comparison plot of predicted vs actual RUL\n"
                     "- Summary of evaluation\n"
                     "- Saved at outputs/PDF file directory\n")

# -----------------------------
# Section: Directory Structure
# -----------------------------
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "4. Directory Structure", ln=True)
pdf.ln(5)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, 
    "Project Directory Structure:\n"
    "- data/: Contains raw and processed dataset files\n"
    "- models/: Saved Keras model files\n"
    "- outputs/Plots/: Generated plots\n"
    "- outputs/PDF file/: Generated PDF reports\n"
    "- src/: Python scripts for data preparation, model evaluation, and reporting\n"
)

# -----------------------------
# Save PDF
# -----------------------------
os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
pdf.output(pdf_output_path)
print(f" Documentation PDF saved at: {pdf_output_path}")
