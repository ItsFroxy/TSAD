import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# === Step 1: Load Healthy + Mixed Data ===
with open('healthy_data.json') as f:
    healthy_data = json.load(f)

with open('mixed_data.json') as f:
    mixed_data = json.load(f)

features = ['heart_rate','spo2','respiratory_rate','systolic_bp','diastolic_bp','temperature','hrv','steps','calories_burned','skin_temp','eda']
window_size = 60

# === Step 2: Prepare Data ===
def extract_windows(patients, label=None):
    windows, labels = [], []
    for person in patients:
        ts = pd.DataFrame(person['time_series'])[features].dropna().values
        for i in range(len(ts) - window_size + 1):
            win = ts[i:i+window_size]
            windows.append(win)
            labels.append(label)
    return np.array(windows), np.array(labels)

# Train only on normal
X_train, _ = extract_windows(healthy_data, label=0)

# Test on mixed (assume anomaly present if state_context is 'anomaly')
def is_anomaly(ts): return any(x.get("state_context") == "anomaly" for x in ts)
X_test, y_test = [], []
for person in mixed_data:
    ts_df = pd.DataFrame(person['time_series'])
    ts = ts_df[features].dropna().values
    for i in range(len(ts) - window_size + 1):
        win = ts[i:i+window_size]
        label = int(any(ts_df.iloc[i:i+window_size]['state_context'] == "anomaly"))
        X_test.append(win)
        y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)

# === Step 3: LSTM Autoencoder ===
input_dim = len(features)
inputs = Input(shape=(window_size, input_dim))
x = LSTM(64, activation="tanh")(inputs)
x = RepeatVector(window_size)(x)
x = LSTM(64, activation="tanh", return_sequences=True)(x)
outputs = TimeDistributed(Dense(input_dim))(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# === Step 4: Threshold from training ===
recons = model.predict(X_train)
train_mse = np.mean(np.square(X_train - recons), axis=(1,2))
threshold = np.mean(train_mse) + 3*np.std(train_mse)
print(f"Threshold (μ + 3σ): {threshold:.5f}")

# === Step 5: Evaluate on test set ===
X_pred = model.predict(X_test)
mse_test = np.mean(np.square(X_test - X_pred), axis=(1,2))
y_pred = (mse_test > threshold).astype(int)

# === Step 6: Metrics ===
print("\n=== Evaluation Report ===")
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}")
print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
