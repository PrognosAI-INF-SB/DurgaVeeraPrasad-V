from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# --- Metadata lines to add at the top of the PDF ---
text_lines = [
    "Project Name: PrognosAI: AI-Driven Predictive Maintenance System Using Time-Series Sensor Data",
    "Dataset: train_FD001 – NASA Turbofan Jet Engine Data Set",
    "Prepared by: Durga Veera Prasad V",
    "Date: 30th September 2025",
    "",
]

# --- Documentation Content for Milestone 2 ---
doc_content = """
# Milestone 2 Documentation

## Objective
The goal of Milestone 2 is to prepare the processed dataset into sequences suitable for training deep learning models (e.g., LSTM/GRU) for Remaining Useful Life (RUL) prediction.

## Steps Completed
1. **Data Scaling** – Normalized sensor values using MinMaxScaler.
2. **Sequence Generation** – Created fixed-length sequences (length = 50 cycles) for time-series modeling.
3. **RUL Labeling** – Capped RUL at 125 cycles to avoid skewness from long tails.
4. **Data Saving** – Exported processed sequences into `.npz` format for efficient reuse.

## Outputs
- `X_train.npz` and `y_train.npz`: Training sequences and labels.
- Verified sequence distribution and alignment with engine cycles.

## Next Steps
- Build baseline deep learning models (LSTM/GRU).
- Define evaluation metrics (RMSE, MAE).
- Begin hyperparameter tuning and cross-validation.
"""

# --- Function to generate PDF ---
def generate_milestone2_pdf():
    # Set the output path to your folder
    output_path = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone2 Durga Veera Prasad V\Milestone2 Durga Veera Prasad V\Rul-Predictive-Maintenance\Milestone2_Documentation.pdf"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create PDF doc
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flowables = []

    # Add metadata text lines
    for line in text_lines:
        flowables.append(Paragraph(line, styles["Normal"]))
    flowables.append(Spacer(1, 12))  # Add space after metadata

    # Add main documentation text
    for para in doc_content.split("\n"):
        if para.strip() == "":
            flowables.append(Spacer(1, 12))  # Blank line = spacer
        elif para.strip().startswith("# "):  # Heading 1
            flowables.append(Paragraph(f"<b><font size=14>{para[2:]}</font></b>", styles["Normal"]))
            flowables.append(Spacer(1, 10))
        elif para.strip().startswith("## "):  # Heading 2
            flowables.append(Paragraph(f"<b><font size=12>{para[3:]}</font></b>", styles["Normal"]))
            flowables.append(Spacer(1, 8))
        else:  # Normal paragraph
            flowables.append(Paragraph(para, styles["Normal"]))
            flowables.append(Spacer(1, 6))

    # Build PDF
    doc.build(flowables)
    print(f" Milestone 2 documentation saved as: {output_path}")


# --- Run the function ---
if __name__ == "__main__":
    generate_milestone2_pdf()
