import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === 1. Load JSON Dataset ===
with open('synthetic_realistic_unlabeled_dataset.json') as f:
    data_json = json.load(f)

# === 2. Flatten all patient time-series into one DataFrame ===
all_records = []
labels = []  # for anomaly marking
for patient in data_json:
    pid = patient['patient_id']
    ts = patient['time_series']
    for entry in ts:
        row = entry.copy()
        row['patient_id'] = pid
        row['timestamp'] = row.pop('timestamp')
        all_records.append(row)
        # Optional: mark anomaly if HR > 170 (as per injected logic)
        labels.append(1 if row['heart_rate'] >= 170 or row['hrv'] <= 10 else 0)

df = pd.DataFrame(all_records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['anomaly'] = labels

# === 3. Choose relevant numerical features ===
features = [
    'heart_rate', 'spo2', 'respiratory_rate', 'systolic_bp',
    'diastolic_bp', 'temperature', 'hrv', 'steps',
    'calories_burned', 'skin_temp', 'eda'
]
data = df[features + ['patient_id', 'timestamp', 'anomaly']].copy()

# === 4. Normalize features ===
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# === 5. Create sliding windows ===
window_size = 60
X, y = [], []
step = 1

for pid, group in data.groupby('patient_id'):
    group = group.sort_values('timestamp')
    values = group[features].values
    labels = group['anomaly'].values
    for i in range(0, len(values) - window_size + 1, step):
        X.append(values[i:i + window_size])
        y.append(int(labels[i:i + window_size].max()))  # mark window as anomalous if any point is

X = np.array(X)
y = np.array(y)
print(f"Shape of X: {X.shape} (windows, timesteps, features)")

# === 6. Train/Test Split ===
np.random.seed(42)
idx = np.arange(len(X))
np.random.shuffle(idx)
train_idx = idx[:int(0.8 * len(idx))]
test_idx = idx[int(0.8 * len(idx)):]

X_train = X[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# === 7. Build LSTM Autoencoder ===
input_dim = X.shape[2]
latent_dim = 64

inputs = Input(shape=(window_size, input_dim))
encoded = LSTM(latent_dim, activation='tanh')(inputs)
decoded = RepeatVector(window_size)(encoded)
decoded = LSTM(latent_dim, activation='tanh', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(input_dim))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
autoencoder.summary()

# === 8. Train model ===
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.1, verbose=2)

# === 9. Compute reconstruction error and threshold ===
train_pred = autoencoder.predict(X_train)
train_mse = np.mean((X_train - train_pred) ** 2, axis=(1, 2))
threshold = np.mean(train_mse) + 3 * np.std(train_mse)
print(f"Threshold (μ + 3σ): {threshold:.5f}")

# === 10. Evaluate on test data ===
test_pred = autoencoder.predict(X_test)
test_mse = np.mean((X_test - test_pred) ** 2, axis=(1, 2))
y_pred = (test_mse > threshold).astype(int)

print("\n=== Evaluation Report ===")
print(classification_report(y_test, y_pred, digits=4))
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("TN", tn, "FP", fp, "FN", fn, "TP", tp)

# === 11. Optional: MSE Histogram ===
plt.hist(test_mse[y_test==0], bins=50, alpha=0.6, label="Normal")
plt.hist(test_mse[y_test==1], bins=50, alpha=0.6, label="Anomaly")
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Test Reconstruction Error Distribution")

plt.legend()
plt.tight_layout()
plt.show()
