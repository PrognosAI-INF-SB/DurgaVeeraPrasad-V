import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
import glob
# Import UploadedFile type hint for robust checking
from streamlit.runtime.uploaded_file_manager import UploadedFile 

# ----------------------------- #
# CONFIGURATION
# ----------------------------- #

# NOTE: Update these paths to match your environment
CMAPSS_FOLDER = r"C:\Users\DELL\OneDrive\Desktop\Milestone-4\CMAPSS\converted_csv"
MODEL_PATH = r"C:\Users\DELL\OneDrive\Desktop\Milestone-4\optimized_lstm_final.keras"
SCALER_PATH = r"C:\Users\DELL\OneDrive\Desktop\Milestone-4\final_scalers.pkl"
SEQ_LEN_DEFAULT = 50
CYCLE_TO_DAYS = 1

# ----------------------------- #
# LOAD MODEL AND SCALERS
# ----------------------------- #

@st.cache_resource
def load_model_scalers():
    try:
        model = load_model(MODEL_PATH)
        scalers = joblib.load(SCALER_PATH)
        scaler_X = scalers["scaler_X"]
        scaler_y = scalers["scaler_y"]

        # FIX for "Scaler does not have feature_names_in_" warning 
        if not hasattr(scaler_X, "feature_names_in_"):
            # This list must match the 14 features used during training EXACTLY.
            inferred_features = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's15', 's17', 's20', 's21', 'op3'] 
            scaler_X.feature_names_in_ = np.array(inferred_features, dtype=object)

        st.success("Model and scalers loaded successfully")
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model or scalers: {e}")
        st.stop()

# ----------------------------- #
# FILE PARSING (FIXED for UploadedFile error)
# ----------------------------- #

def parse_cmapss_txt_or_csv(file_input):
    """
    Parses CMAPSS data from a file path (string) or a Streamlit UploadedFile object.
    
    Args:
        file_input: File path string or st.runtime.uploaded_file_manager.UploadedFile object.
    """
    file_name = "unknown"
    
    try:
        if isinstance(file_input, str):
            # Handles local file paths (like test files)
            file_name = file_input
            if file_name.lower().endswith(".csv"):
                df = pd.read_csv(file_name)
            else:
                # Assuming standard CMAPSS text format for non-CSV files
                df = pd.read_csv(file_name, sep=r"\s+", header=None)
        
        elif isinstance(file_input, UploadedFile):
            # Handles Streamlit uploaded file objects
            file_name = file_input.name
            
            # Use the file object itself with pandas, check extension via .name
            if file_name.lower().endswith((".txt", ".dat")):
                df = pd.read_csv(file_input, sep=r"\s+", header=None)
            else:
                df = pd.read_csv(file_input)
        
        else:
            st.error("Invalid file input type provided.")
            return None
            
        # Common data processing steps
        df = df.apply(pd.to_numeric, errors='coerce')

        if len(df.columns) >= 26:
            cols = ['Engine_ID', 'Cycle'] + [f'op_set{i+1}' for i in range(3)] + [f'sensor{i+1}' for i in range(len(df.columns) - 5)]
            df.columns = cols[:len(df.columns)]
            
        if 'Engine_ID' not in df.columns:
            df['Engine_ID'] = 1
        if 'Cycle' not in df.columns:
            df['Cycle'] = np.arange(1, len(df) + 1)

        return df
    except Exception as e:
        st.error(f"Error reading file {file_name}: {e}")
        return None

# ----------------------------- #
# SEQUENCE CREATION 
# ----------------------------- #

def create_inference_sequences(df, seq_len, feature_cols):
    X, engines, infos = [], [], []

    id_col = None
    for col in df.columns:
        if "engine" in col.lower() or "unit" in col.lower() or "id" in col.lower():
            id_col = col
            break

    if id_col is None:
        st.warning("No Engine ID column found. Assigning Engine_ID = 1 for all rows.")
        df["Engine_ID"] = 1
        id_col = "Engine_ID"

    if "Cycle" not in df.columns:
        df["Cycle"] = np.arange(1, len(df) + 1)

    # FIX: Use groupby keys to reliably get unique engine IDs
    try:
        engine_ids = df.groupby(id_col).groups.keys()
    except Exception as e:
        st.error(f"Error extracting unique Engine IDs: {e}")
        return np.array([]), [], pd.DataFrame()


    for eng in sorted(engine_ids):
        part = df[df[id_col] == eng].sort_values("Cycle")
        
        if len(part) < seq_len:
            # Pad the sequence if the data is shorter than seq_len
            pad = np.repeat(part[feature_cols].iloc[[0]].values, seq_len - len(part), axis=0)
            seq = np.vstack([pad, part[feature_cols].values])
        else:
            # Take the last seq_len cycles
            seq = part[feature_cols].values[-seq_len:]
            
        X.append(seq)
        engines.append(eng)
        infos.append(part.iloc[-1].to_dict())

    return np.array(X), engines, pd.DataFrame(infos)

# ----------------------------- #
# COLUMN RENAMING
# ----------------------------- #

def rename_columns_auto(df):
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if "sensor" in lower:
            num = ''.join([ch for ch in lower if ch.isdigit()])
            rename_map[col] = f"s{num}"
        elif "op_set" in lower or "setting" in lower:
            num = ''.join([ch for ch in lower if ch.isdigit()])
            rename_map[col] = f"op{num}"
    return df.rename(columns=rename_map)

# ----------------------------- #
# PREDICTION FUNCTION
# ----------------------------- #

def predict_rul(df, model, scaler_X, seq_len):
    df = df.copy()
    df = rename_columns_auto(df)

    # Determine expected features from the fitted scaler (guaranteed to exist by load_model_scalers)
    if hasattr(scaler_X, "feature_names_in_"):
        expected_features = list(scaler_X.feature_names_in_)
    else:
        st.error("Feature names missing. Cannot proceed with feature selection.")
        return pd.DataFrame()

    # Ensure all expected features are present (fill with 0 if missing)
    for f in expected_features:
        if f not in df.columns:
            df[f] = 0

    df_features = df[expected_features].copy()

    try:
        X_scaled = scaler_X.transform(df_features)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        return pd.DataFrame()

    if 'Engine_ID' not in df.columns:
        df['Engine_ID'] = 1
    if 'Cycle' not in df.columns:
        df['Cycle'] = np.arange(1, len(df) + 1)

    df_scaled = pd.concat([
        df[['Engine_ID', 'Cycle']].astype(int).reset_index(drop=True),
        pd.DataFrame(X_scaled, columns=expected_features)
    ], axis=1)

    X, engines, _ = create_inference_sequences(df_scaled, seq_len, expected_features)
    
    if X.size == 0:
        st.warning("No sequences created for prediction.")
        return pd.DataFrame()

    preds = model.predict(X).ravel()

    result = pd.DataFrame({
        "Engine_ID": engines,
        "Predicted_RUL": preds
    })
    return result

# ----------------------------- #
# STREAMLIT DASHBOARD
# ----------------------------- #

st.set_page_config(page_title="PrognosAI Dashboard", layout="wide")
st.title("PrognosAI Milestone 4 Dashboard")

model, scaler_X, scaler_y = load_model_scalers()

st.sidebar.header("Dashboard Settings")
seq_len = st.sidebar.number_input("Sequence length", 10, 200, SEQ_LEN_DEFAULT)

# New: Input mode: use test files, upload, or manual/random input
input_mode = st.sidebar.selectbox("Input Mode", ["Use Test Files", "Upload File", "Manual / Random Input"]) 

# Simplified input labels (No 'days')
warn_th = st.sidebar.number_input("Warning threshold", 1, 100, 50)
crit_th = st.sidebar.number_input("Critical threshold", 1, 100, 20)

# If user chooses to upload file
uploaded_file = None
if input_mode == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or TXT (CMAPSS format)")

# If user chooses manual/random input, dynamically create inputs based on expected features
manual_df = None
if input_mode == "Manual / Random Input":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manual / Random Input Configuration")

    # Get expected features from scaler
    if hasattr(scaler_X, "feature_names_in_"):
        expected_features = list(scaler_X.feature_names_in_)
    else:
        expected_features = [f"s{i+1}" for i in range(14)] # Fallback for display

    # Create a section in main area for manual input
    st.subheader("Manual or Random Input for a single engine")
    st.write("Enter values for each sensor. These values should be **standardized** (mean ~0, variance ~1) as the model expects normalized input.")

    # layout inputs in columns
    cols = st.columns(4)
    # Initialize session state for persistence
    if 'manual_values' not in st.session_state:
        st.session_state.manual_values = {feat: float(0.0) for feat in expected_features}

    for i, feat in enumerate(expected_features):
        col = cols[i % 4]
        # Use session state for the input widget value
        st.session_state.manual_values[feat] = col.number_input(
            f"{feat}", 
            value=st.session_state.manual_values[feat], 
            format="%.4f", 
            key=f"manual_input_{feat}"
        )

    if st.button("Randomize Sensors"):
        for feat in expected_features:
            st.session_state.manual_values[feat] = float(np.random.normal(loc=0.0, scale=1.0))
        # Rerun to update input values
        st.experimental_rerun()

    seq_len_manual = st.number_input("Sequence length for manual input", 10, 200, seq_len)
    if st.button("Predict RUL for manual input"):
        # Build a one-row dataframe from session state
        df_manual = pd.DataFrame([st.session_state.manual_values])
        df_manual["Engine_ID"] = 1
        df_manual["Cycle"] = 1

        # Call predict_rul which will pad the sequence if seq_len_manual > 1
        preds_manual = predict_rul(df_manual, model, scaler_X, seq_len_manual)
        if not preds_manual.empty:
            preds_manual['Alert'] = preds_manual['Predicted_RUL'].apply(
                lambda x: 'CRITICAL' if x <= crit_th else ('WARNING' if x <= warn_th else 'NORMAL')
            )
            st.success("Prediction complete")
            st.table(preds_manual)
        else:
            st.error("Prediction failed for manual input.")

# --- LOAD/SELECT TEST FILES (existing behavior) ---
if input_mode == "Use Test Files":
    test_files = glob.glob(os.path.join(CMAPSS_FOLDER, "test_*.csv"))
    if not test_files:
        st.warning(f"No test files found in folder: {CMAPSS_FOLDER}")
        test_files = glob.glob(os.path.join(os.path.dirname(CMAPSS_FOLDER), "test_*.csv"))
        if not test_files:
            st.warning(f"No test files found in fallback folder.")
            st.stop()

    selected_option = st.sidebar.selectbox(
        "Select Test File",
        ["All Files"] + [os.path.basename(f) for f in test_files]
    )
else:
    selected_option = None

all_preds = []

def process_and_display(file_input):
    """Processes a file (path or UploadedFile) and returns predictions."""
    if isinstance(file_input, str):
        st.write(f"Processing file: **{os.path.basename(file_input)}**")
    elif hasattr(file_input, 'name'):
        st.write(f"Processing uploaded file: **{file_input.name}**")
        
    df = parse_cmapss_txt_or_csv(file_input)
    
    if df is not None and not df.empty:
        current_seq_len = seq_len
        preds = predict_rul(df, model, scaler_X, current_seq_len)
        
        if not preds.empty:
            preds['Alert'] = preds['Predicted_RUL'].apply(
                lambda x: 'CRITICAL' if x <= crit_th else ('WARNING' if x <= warn_th else 'NORMAL')
            )
            max_cycles = df.groupby('Engine_ID')['Cycle'].max().to_dict()
            preds['Elapsed_Life'] = preds['Engine_ID'].map(max_cycles)
            
            preds['Remaining_Days'] = preds['Predicted_RUL'] * CYCLE_TO_DAYS
            preds['Elapsed_Days'] = preds['Elapsed_Life'] * CYCLE_TO_DAYS
            preds['Total_Days'] = preds['Elapsed_Days'] + preds['Remaining_Days']
            preds['Remaining_Years'] = preds['Remaining_Days'] / 365
            preds['Total_Years'] = preds['Total_Days'] / 365
            return preds
    return pd.DataFrame()

# If user uploaded a file, process it
if uploaded_file is not None:
    # Pass the UploadedFile object directly
    preds = process_and_display(uploaded_file)
    if not preds.empty:
        all_preds.append(preds)

# If using test files
if input_mode == "Use Test Files" and selected_option:
    if selected_option == "All Files":
        for file_path in test_files:
            preds = process_and_display(file_path)
            if not preds.empty:
                all_preds.append(preds)
    else:
        # Locate the selected file's full path
        file_path = os.path.join(CMAPSS_FOLDER, selected_option)
        if not os.path.exists(file_path):
            file_path = [f for f in test_files if os.path.basename(f) == selected_option]
            if file_path:
                file_path = file_path[0]
            else:
                st.error(f"Selected file {selected_option} not found.")
                file_path = None
        
        if file_path:
            preds = process_and_display(file_path)
            if not preds.empty:
                all_preds.append(preds)

# --- MAIN RESULTS DISPLAY ---
if all_preds:
    preds_all = pd.concat(all_preds, ignore_index=True)
    alert_colors = {'NORMAL': 'green', 'WARNING': 'orange', 'CRITICAL': 'red'}

    st.header("Results and Visualizations")
    
    st.subheader("Engine Life Summary Table")
    st.dataframe(preds_all[['Engine_ID', 'Elapsed_Days', 'Remaining_Days', 'Total_Days',
                             'Remaining_Years', 'Total_Years', 'Alert']])

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Alert Status Distribution (Critical: {crit_th} Days, Warning: {warn_th} Days)")
        pie = px.pie(preds_all, names='Alert', title="Alert Status", color='Alert',
                      color_discrete_map=alert_colors)
        st.plotly_chart(pie, use_container_width=True)

    with col2:
        st.subheader("Remaining Useful Life (Days)")
        fig_days = px.scatter(preds_all, x='Engine_ID', y='Predicted_RUL', color='Alert',
                              color_discrete_map=alert_colors, title="RUL Predictions (Days)")
        fig_days.add_hline(y=warn_th, line_dash="dot", line_color="orange", annotation_text=f"Warning Threshold ({warn_th} days)")
        fig_days.add_hline(y=crit_th, line_dash="dot", line_color="red", annotation_text=f"Critical Threshold ({crit_th} days)")
        fig_days.update_layout(showlegend=False)
        st.plotly_chart(fig_days, use_container_width=True)

    st.subheader("Engine Life Span Overview")
    fig_life = go.Figure()
    
    preds_all_sorted = preds_all.sort_values('Remaining_Days', ascending=False)
    
    fig_life.add_trace(go.Bar(
        y=preds_all_sorted['Engine_ID'].astype(str),
        x=preds_all_sorted['Elapsed_Days'],
        orientation='h',
        name='Elapsed Life (Days)',
        marker_color='lightgray'
    ))
    
    for alert, color in alert_colors.items():
        subset = preds_all_sorted[preds_all_sorted['Alert'] == alert]
        if not subset.empty:
            fig_life.add_trace(go.Bar(
                y=subset['Engine_ID'].astype(str),
                x=subset['Remaining_Days'],
                orientation='h',
                name=f'Remaining Life ({alert})',
                marker_color=color
            ))

    fig_life.update_layout(barmode='stack', xaxis_title='Days', yaxis_title='Engine ID', height=500)
    st.plotly_chart(fig_life, use_container_width=True)

    csv = preds_all.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "prognosai_results.csv", "text/csv")

# ----------------------------- #
# SUMMARY COMPARISON SECTION
# ----------------------------- #

st.header("Summary: Test File Comparison")

summary_data = []
detailed_results = {}
test_folder = CMAPSS_FOLDER 
test_files = sorted(glob.glob(os.path.join(test_folder, "test_FD*.csv")))

for file_path in test_files:
    file_name = os.path.basename(file_path)
    st.write(f"Processing summary for: **{file_name}**")
    try:
        df_test = parse_cmapss_txt_or_csv(file_path)
        preds = predict_rul(df_test, model, scaler_X, seq_len=SEQ_LEN_DEFAULT) 

        if not preds.empty:
            avg_rul = preds["Predicted_RUL"].mean()
            min_rul = preds["Predicted_RUL"].min()
            max_rul = preds["Predicted_RUL"].max()

            preds["Alert"] = preds["Predicted_RUL"].apply(
                lambda x: 'CRITICAL' if x <= crit_th else ('WARNING' if x <= warn_th else 'NORMAL')
            )

            summary_data.append({
                "File": file_name,
                "Engines": preds["Engine_ID"].nunique(),
                "Avg RUL": round(avg_rul, 2),
                "Min RUL": round(min_rul, 2),
                "Max RUL": round(max_rul, 2)
            })
            detailed_results[file_name] = preds
    except Exception as e:
        st.error(f"Error processing {file_name} for summary: {e}")

if summary_data:
    df_summary = pd.DataFrame(summary_data)
    
    st.subheader("Average RUL per Test File")
    st.dataframe(df_summary)

    fig = px.bar(
        df_summary,
        x="File",
        y="Avg RUL",
        text="Avg RUL",
        color="File",
        title="Average Predicted RUL Comparison",
        height=400
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed File Exploration")
    selected_file = st.selectbox("Select a test file to explore details", df_summary["File"])

    if selected_file in detailed_results:
        preds_sel = detailed_results[selected_file]
        st.markdown(f"**Detailed View: {selected_file}**")

        alert_colors = {'NORMAL': 'green', 'WARNING': 'orange', 'CRITICAL': 'red'}

        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            fig_detail = px.bar(
                preds_sel,
                x="Engine_ID",
                y="Predicted_RUL",
                color="Alert",
                color_discrete_map=alert_colors,
                title="RUL for Each Engine"
            )
            fig_detail.add_hline(y=warn_th, line_dash="dot", line_color="orange")
            fig_detail.add_hline(y=crit_th, line_dash="dot", line_color="red")
            st.plotly_chart(fig_detail, use_container_width=True)
            
        with col_sum2:
            pie_alerts = px.pie(
                preds_sel,
                names="Alert",
                title="Alert Status Distribution",
                color="Alert",
                color_discrete_map=alert_colors
            )
            st.plotly_chart(pie_alerts, use_container_width=True)
else:
    st.info("No valid predictions generated for summary comparison.")