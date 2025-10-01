from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# PDF file path
pdf_file = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone1 Durga Veera Prasad V\Rul-Predictive-Maintenance\Milestone1_Final_Report.pdf"

# Create a canvas
c = canvas.Canvas(pdf_file, pagesize=A4)
width, height = A4

# Margins
x_margin = 2*cm
y_position = height - 2*cm

# Title
c.setFont("Helvetica-Bold", 16)
c.drawString(x_margin, y_position, "Milestone 1: Data Preparation & Feature Engineering")
y_position -= 1*cm

c.setFont("Helvetica", 11)

# Content lines
text_lines = [
    "Project Name: PrognosAI: AI-Driven Predictive Maintenance System Using Time-Series Sensor Data",
    "Dataset: train_FD001 – NASA Turbofan Jet Engine Data Set",
    "Prepared by: Durga Veera Prasad V",
    "Date: 30th September 2025",
    "",
    "1. Objective:",
    "- Load, preprocess, and prepare the CMAPSS dataset for model training.",
    "- Create rolling window sequences.",
    "- Compute RUL targets.",
    "- Save processed data for model training.",
    "",
    "2. Dataset Description:",
    "- Source: train_FD001.txt (converted to CSV: train_FD001.csv)",
    "- Columns: unit_nr, time_in_cycles, op_settings 1-3, sensors s1-s21, RUL",
    "- Selected 10 sensors + 3 operational settings for modeling.",
    "",
    "3. Methodology:",
    "Step 1: Load Data - Read CSV using pandas.",
    "Step 2: Compute RUL - RUL = max(cycle) - current cycle per engine.",
    "Step 3: Feature Selection & Normalization - Select features, fill missing values, standardize.",
    "Step 4: Generate Rolling Window Sequences - Window length 30, sequence labeled with last cycle RUL.",
    "Step 5: Data Verification - Check missing values, shapes, sample RULs.",
    "Step 6: Save Processed Data - Save X and y as .npz for model training.",
    "",
    "4. Results:",
    "- Number of sequences: 17,731",
    "- Sequence length: 30 cycles",
    "- Number of features per sequence: 13",
    "- Missing values in features used: 0",
    "- Sample RUL values: [162, 161, 160, 159, 158, 157, 156, 155, 154, 153]",
    "",
    "5. Conclusion:",
    "- Data preparation pipeline successfully completed.",
    "- X (rolling windows) and y (RUL labels) ready for model training.",
    "- Data integrity verified; pipeline is reproducible and robust.",
]

# Write lines to PDF
line_height = 14
for line in text_lines:
    c.drawString(x_margin, y_position, line)
    y_position -= line_height
    if y_position < 2*cm:  # Start a new page if space is low
        c.showPage()
        c.setFont("Helvetica", 11)
        y_position = height - 2*cm

# Save PDF
c.save()
print("Milestone 1 PDF report created at:", pdf_file)
