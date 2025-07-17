import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional
import matplotlib.pyplot as plt

# === 1. Load JSON dataset ===
with open("synthetic_wearable_dataset.json") as f:
    patients = json.load(f)

all_records = []
for patient in patients:
    pid = patient["patient_id"]
    reviewed = patient["reviewed"]
    for entry in patient["time_series"]:
        entry["patient_id"] = pid
        entry["reviewed"] = reviewed
        all_records.append(entry)

df = pd.DataFrame(all_records)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["patient_id", "timestamp"])

# === 2. Select features ===
features = ['heart_rate', 'spo2', 'temperature', 'steps', 'respiratory_rate',
            'systolic_bp', 'diastolic_bp', 'hrv', 'skin_temp', 'eda']
df = df[["patient_id", "timestamp", "reviewed"] + features].copy()

# === 3. Normalize features ===
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# === 4. Generate sliding windows ===
window_size = 60
X = []
y = []
for pid, group in df.groupby("patient_id"):
    data = group[features].values
    flags = group["reviewed"].values  # reviewed = True means no anomaly

    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        # If reviewed=False in any time point of the window, label as anomaly
        label = int((~flags[i:i + window_size]).any())
        X.append(window)
        y.append(label)

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)

# === 5. Train/Test Split ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 6. Build Bi-LSTM Autoencoder ===
timesteps = X_train.shape[1]
input_dim = X_train.shape[2]

inputs = Input(shape=(timesteps, input_dim))
encoded = Bidirectional(LSTM(64, activation='tanh'))(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(decoded)
outputs = TimeDistributed(Dense(input_dim))(decoded)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

# === 7. Train only on normal data ===
X_train_normal = X_train[y_train == 0]
model.fit(X_train_normal, X_train_normal, epochs=20, batch_size=64, validation_split=0.1, verbose=2)

# === 8. Compute reconstruction error threshold ===
X_train_pred = model.predict(X_train_normal)
mse = np.mean((X_train_pred - X_train_normal) ** 2, axis=(1, 2))
threshold = np.mean(mse) + 3 * np.std(mse)
print(f"🔍 Threshold (μ + 3σ): {threshold:.5f}")

# === 9. Evaluate on test set ===
X_test_pred = model.predict(X_test)
mse_test = np.mean((X_test_pred - X_test) ** 2, axis=(1, 2))
y_pred = (mse_test > threshold).astype(int)

# === 10. Print Metrics ===
print("\n=== Evaluation Report ===")
print(classification_report(y_test, y_pred, digits=4))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}")

# Optional: Plot distribution of reconstruction error
plt.hist(mse_test[y_test==0], bins=50, alpha=0.7, label="Normal")
plt.hist(mse_test[y_test==1], bins=50, alpha=0.7, label="Anomalies")
plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.show()
