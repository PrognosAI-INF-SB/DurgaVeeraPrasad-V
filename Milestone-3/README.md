Milestone 3: Model Evaluation & Performance Assessment

Project Name: PrognosAI: AI-Driven Predictive Maintenance System Using Time-Series Sensor Data
Dataset: NASA Turbofan Jet Engine Data Set
Prepared by: Durga Veera Prasad V

Objective:  
Evaluate the performance of the optimized LSTM model using 5-fold cross-validation and multiple metrics, and visualize training behavior and prediction accuracy.

---

Modules Used and Purpose:
numpy: Array handling and numerical computations.  
tensorflow / keras: Build and train the LSTM model with layers like `LSTM`, `Dense`, `Dropout`, and `BatchNormalization`.  
sklearn.preprocessing.MinMaxScaler: Scale features and target values.  
sklearn.model_selection.KFold: Perform 5-fold cross-validation.  
sklearn.metrics: Compute evaluation metrics (`r2_score`, `mean_squared_error`, `mean_absolute_error`).  
matplotlib.pyplot: Plot training/validation loss curves and R² scores.  
pandas: Store and summarize cross-validation results.  
joblib: Save scalers for future inference.

---





Steps Implemented:
1. Data Preparation:
   - Used CMAPSS-style synthetic sequences (`X`) and RUL targets (`y_raw`) for demonstration.
   - Scaled features and targets using `MinMaxScaler`.

2. Improved LSTM Model:
   - Constructed a deeper LSTM network with two LSTM layers, dropout, dense layers, and batch normalization for better generalization.
   - Compiled with `Adam` optimizer and MSE loss.

3. Cross-Validation:
   - Applied 5-fold cross-validation to evaluate model performance.
   - Tracked metrics per fold: Train/Test R², RMSE, MAE.
   - Plotted training and validation loss curves for each fold.

4. Evaluation & Visualization:
   - Calculated average metrics across folds.
   - Visualized R² scores per fold.
   - Saved cross-validation results as `crossval_results.csv`.

5. Final Model Training & Saving:
   - Retrained LSTM on full dataset.
   - Saved optimized model (`.keras`) and scalers (`.pkl`) for deployment or inference.

---




Deliverables:
- Optimized LSTM model with final weights.
- Scalers for feature and target normalization.
- Cross-validation metrics and plots.
- CSV file of fold-wise results.

Evaluation:
- Verified high R² scores (>95% target on training data).
- Checked convergence via loss curves.
- Confirmed generalization using cross-validation metrics.
