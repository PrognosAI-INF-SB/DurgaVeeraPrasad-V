"""
Milestone 2: Model Development & Training

Objective:
    - Train a time-series deep learning model (LSTM) to predict RUL
    - Includes handling NaNs and RUL scaling to prevent NaN loss
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -----------------------
# CONFIGURATION
# -----------------------
DATA_FILE = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone2 Durga Veera Prasad V\Milestone2 Durga Veera Prasad V\Rul-Predictive-Maintenance\data\train_FD001_processed.npz"
MODEL_FILE = r"C:\Users\DELL\OneDrive\Desktop\Final Project\Milestone2 Durga Veera Prasad V\Milestone2 Durga Veera Prasad V\Rul-Predictive-Maintenance\models\model_training.h5"
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0005

# -----------------------
# STEP 1: Load Preprocessed Data
# -----------------------
data = np.load(DATA_FILE)
X = data["X"]
y = data["y"]

print(f"Original X shape: {X.shape}, y shape: {y.shape}")
print("NaNs in X before handling:", np.isnan(X).sum())
print("NaNs in y before handling:", np.isnan(y).sum())

# -----------------------
# STEP 1.5: Handle NaNs in features
# -----------------------
# Fill NaNs with 0
X = np.nan_to_num(X, nan=0.0)
print("NaNs in X after filling:", np.isnan(X).sum())

# -----------------------
# STEP 2: Scale RUL to 0-1
# -----------------------
y_max = y.max()
y = y / y_max  # scale to 0-1

# -----------------------
# STEP 3: Train/Validation Split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, shuffle=True, random_state=42
)
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# -----------------------
# STEP 4: Define LSTM Model
# -----------------------
num_features = X.shape[2]

model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, num_features), return_sequences=True),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# -----------------------
# STEP 5: Callbacks
# -----------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True)
]

# -----------------------
# STEP 6: Train Model
# -----------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------
# STEP 7: Plot Training & Validation Loss
# -----------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()

# -----------------------
# STEP 8: Evaluate Predictions (Optional)
# -----------------------
y_pred = model.predict(X_val)
y_pred_rescaled = y_pred * y_max
y_val_rescaled = y_val * y_max

plt.figure(figsize=(8,5))
plt.plot(y_val_rescaled[:200], label='Actual RUL')
plt.plot(y_pred_rescaled[:200], label='Predicted RUL')
plt.title("Predicted vs Actual RUL (Validation Sample)")
plt.xlabel("Sample")
plt.ylabel("RUL")
plt.legend()
plt.grid()
plt.show()

print(f" Model training complete. Weights saved at: {MODEL_FILE}")
