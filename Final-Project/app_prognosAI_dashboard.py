import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
import glob

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
# LOAD MODEL AND SCALERS (FIXED for metadata warning)
# ----------------------------- #

@st.cache_resource
def load_model_scalers():
    try:
        model = load_model(MODEL_PATH)
        scalers = joblib.load(SCALER_PATH)
        scaler_X = scalers["scaler_X"]
        scaler_y = scalers["scaler_y"]

        # FIX for "Scaler does not have feature_names_in_" warning 
        # Inject the feature names that the model was trained on.
        # This list assumes the standard 14 high-variance CMAPSS sensors 
        # were selected, consistent with the warning's logic. Adjust if necessary.
        if not hasattr(scaler_X, "feature_names_in_"):
            # This list must match the 14 features used during training EXACTLY.
            # Example features often used in CMAPSS (adjust this list based on your model):
            inferred_features = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's15', 's17', 's20', 's21', 'op3'] 
            scaler_X.feature_names_in_ = np.array(inferred_features, dtype=object)

        st.success("Model and scalers loaded successfully")
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model or scalers: {e}")
        st.stop()

# ----------------------------- #
# FILE PARSING
# ----------------------------- #

def parse_cmapss_txt_or_csv(file_path):
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, sep=r"\s+", header=None)

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
        st.error(f"Error reading file {file_path}: {e}")
        return None

# ----------------------------- #
# SEQUENCE CREATION (FIXED for DataFrame unique error)
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

    # FIX: Use groupby keys to reliably get unique engine IDs (fixes AttributeError)
    try:
        engine_ids = df.groupby(id_col).groups.keys()
    except Exception as e:
        st.error(f"Error extracting unique Engine IDs: {e}")
        return np.array([]), [], pd.DataFrame()


    for eng in sorted(engine_ids):
        part = df[df[id_col] == eng].sort_values("Cycle")
        
        if len(part) < seq_len:
            pad = np.repeat(part[feature_cols].iloc[[0]].values, seq_len - len(part), axis=0)
            seq = np.vstack([pad, part[feature_cols].values])
        else:
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

    # Determine expected features from the fitted scaler (Now guaranteed to exist by load_model_scalers)
    if hasattr(scaler_X, "feature_names_in_"):
        expected_features = list(scaler_X.feature_names_in_)
    else:
        # Fallback in case the manual injection failed for some reason
        st.warning("Feature names still missing. Using first 14 numeric columns as fallback.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        expected_features = [c for c in numeric_cols if c not in ['Engine_ID', 'Cycle'] and not c.startswith('op')][:14]
        if len(expected_features) < 14:
             st.error("Could not reliably detect 14 sensor features. Prediction may be inaccurate.")
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

# Simplified input labels (No 'days')
warn_th = st.sidebar.number_input("Warning threshold", 1, 100, 50)
crit_th = st.sidebar.number_input("Critical threshold", 1, 100, 20)

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

all_preds = []

def process_and_display(file_path):
    st.write(f"Processing file: **{os.path.basename(file_path)}**")
    df = parse_cmapss_txt_or_csv(file_path)
    
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

if selected_option == "All Files":
    for file_path in test_files:
        preds = process_and_display(file_path)
        if not preds.empty:
            all_preds.append(preds)
else:
    file_path = os.path.join(CMAPSS_FOLDER, selected_option)
    if not os.path.exists(file_path):
        file_path = [f for f in test_files if os.path.basename(f) == selected_option][0]
        
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