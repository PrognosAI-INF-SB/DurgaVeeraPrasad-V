# PrognosAI: AI-Driven Predictive Maintenance System
### Milestone 3 — Model Evaluation, Performance Assessment, Risk Thresholding & Alert System

**Dataset:** NASA Turbofan Jet Engine Data Set  
**Prepared by:** Durga Veera Prasad V

---

## Table of Contents
1. Objective
2. Modules Used and Their Purpose
3. Workflow Overview
4. Key Results
5. Saved Files
6. Conclusion

---

## 1. Objective
- Train an LSTM model for Remaining Useful Life (RUL) prediction using corrected cross-validation scaling.  
- Compute evaluation metrics on the original RUL scale.  
- Introduce dynamic alert thresholds (20% critical, 50% warning).  
- Generate a PDF report summarizing cross-validation results, predictions, and alert diagnostics.  

---

## 2. Modules Used and Their Purpose

| **Module** | **Purpose / Use** |
|-------------|------------------|
| numpy | Numerical operations and array manipulation for synthetic data and predictions |
| pandas | Organizing cross-validation results and saving CSV reports |
| os | Managing file paths for saving models, plots, and reports |
| joblib | Saving and loading fitted MinMaxScaler objects for reproducibility |
| matplotlib.pyplot | Plotting training loss, predicted RUL, residuals, and R² scores |
| matplotlib.backends.backend_pdf.PdfPages | Generating multi-page PDF reports |
| tensorflow / keras.models.Sequential | Building sequential LSTM models for time-series regression |
| tensorflow.keras.layers.LSTM | Learning temporal dependencies in RUL sequences |
| tensorflow.keras.layers.Dense | Mapping LSTM outputs to RUL predictions |
| tensorflow.keras.layers.Dropout | Regularization to reduce overfitting |
| tensorflow.keras.layers.Input | Explicit input layer to prevent warnings |
| tensorflow.keras.callbacks.EarlyStopping | Stops training when validation loss stagnates |
| tensorflow.keras.callbacks.ReduceLROnPlateau | Reduces learning rate on performance plateaus |
| tensorflow.keras.optimizers.Adam | Optimizer for faster convergence |
| sklearn.preprocessing.MinMaxScaler | Scaling features and targets (fit on training fold only) |
| sklearn.model_selection.KFold | Implementing 5-fold cross-validation |
| sklearn.metrics.r2_score | Computing R² (coefficient of determination) |
| sklearn.metrics.mean_squared_error | Calculating RMSE for test evaluation |
| sklearn.metrics.mean_absolute_error | Calculating MAE for test evaluation |

---

## 3. Workflow Overview

### Step 1: Configuration
Define parameters such as samples, timesteps, features, cross-validation splits, epochs, and batch sizes for both CV and final training.  

### Step 2: Reproducibility
Set NumPy and TensorFlow seeds for consistent results.  

### Step 3: Data Generation
Simulate a synthetic CMAPSS-style dataset with linear decay and noise to represent engine RUL signals.  
- **X:** Sensor sequences  
- **y_raw:** Final RUL target  

### Step 4: LSTM Model Builder
Build a sequential model with architecture:  
**LSTM(128) → Dropout(0.1) → Dense(64) → Dense(32) → Dense(1)**  
Use Adam optimizer and MSE loss.  

### Step 5: 5-Fold Cross-Validation
For each fold:  
1. Split data into train and test sets.  
2. Fit MinMaxScaler on the training fold only.  
3. Scale both train and test sets.  
4. Train model using EarlyStopping and ReduceLROnPlateau.  
5. Predict and inverse transform results to original RUL scale.  
6. Compute Train/Test R², RMSE, and MAE.  
7. Save model, scalers, and fold loss plots.  

### Step 6: Final Model Training
Train the final LSTM model on the full dataset.  
Fit scalers on the full data and save model and scalers after training completion.  

### Step 7: Dynamic Alerts
Define RUL alert thresholds:  
- **Critical:** Bottom 20% of RUL values  
- **Warning:** Bottom 50% of RUL values  

Classify predictions into critical, warning, and normal states.  
Save alert indices and predicted RUL arrays.  

### Step 8: Visualization
- R² across folds: *r2_across_folds.png*  
- Predicted RUL with alerts: *predicted_rul_with_alerts_full.png*  
- Residual histogram: *residuals_hist.png*  

### Step 9: PDF Report
Automatically generate a multi-page report summarizing:  
- Cross-validation metrics  
- R² plots  
- Predicted RUL alerts  
- Residual distributions  

---

## 4. Key Results
- Achieved high cross-validation R² across folds.  
- RMSE and MAE computed on original RUL scale for realistic interpretation.  
- Dynamic alert mechanism effectively identifies at-risk engines.  
- Complete PDF report compiled for quick analysis and presentation.  

---

## 5. Saved Files

| **File Name** | **Description** |
|----------------|----------------|
| optimized_lstm_final.keras | Final trained LSTM model |
| final_scalers.pkl | Feature and target scalers |
| crossval_results_corrected.csv | Cross-validation metrics |
| predicted_rul_with_alerts_full.png | Visualization of RUL predictions with alerts |
| r2_across_folds.png | Fold-wise R² performance plot |
| residuals_hist.png | Residual distribution plot |
| Milestone3_Report_Corrected.pdf | Full summary report |

---

## 6. Conclusion
- Corrected scaling leakage during cross-validation.  
- Computed metrics on the original RUL scale for reliability.  
- Implemented dynamic alert classification for predictive maintenance.  
- Automated report generation for efficient result sharing.  
