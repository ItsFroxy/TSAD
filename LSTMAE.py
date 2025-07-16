import json, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Bidirectional, LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
with open("synthetic_realistic_unlabeled_dataset.json") as f:
    patients = json.load(f)

# 2️⃣ Prepare windows
W = 60
timeseries, labels = [], []

for p in patients:
    df = pd.DataFrame(p['time_series']).sort_values('timestamp')
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[['heart_rate','spo2','temperature','steps','respiratory_rate','systolic_bp','diastolic_bp','hrv','eda','skin_temp']])
    label = 1 if not p.get('reviewed', True) else 0

    for i in range(len(X) - W + 1):
        timeseries.append(X[i:i+W])
        labels.append(label)

X = np.array(timeseries); y = np.array(labels)
idx = np.random.permutation(len(X)); n80 = int(0.8 * len(idx))
train_idx, test_idx = idx[:n80], idx[n80:]

X_train = X[train_idx][y[train_idx]==0]  # train only on normal
X_test, y_test = X[test_idx], y[test_idx]

# 3️⃣ Build Bi‑LSTM Autoencoder
n_features = X.shape[2]
latent = 64

inp = Input((W, n_features))
enc = Bidirectional(LSTM(latent, activation='tanh'))(inp)
dec = RepeatVector(W)(enc)
dec = Bidirectional(LSTM(latent, activation='tanh', return_sequences=True))(dec)
out = TimeDistributed(Dense(n_features))(dec)

model = Model(inp, out)
model.compile(optimizer='adam', loss='mse')
model.summary()

# 4️⃣ Train
model.fit(X_train, X_train, epochs=20, batch_size=64, validation_split=0.1, verbose=2)

# 5️⃣ Define Threshold using 95th percentile
train_pred = model.predict(X_train)
train_mse = np.mean((train_pred - X_train)**2, axis=(1,2))
threshold = np.percentile(train_mse, 95)
print(f"⚠️ Detection threshold set at 95th percentile: {threshold:.5f}")

# 6️⃣ Detect Anomalies
test_pred = model.predict(X_test)
mse_test = np.mean((test_pred - X_test)**2, axis=(1,2))
y_pred = (mse_test > threshold).astype(int)

# 7️⃣ Evaluate
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("TN", tn, "FP", fp, "FN", fn, "TP", tp)

# 8️⃣ Visualize Error Distribution
plt.hist(mse_test[y_test==0], bins=100, alpha=0.6, label='Normal')
plt.hist(mse_test[y_test==1], bins=100, alpha=0.6, label='Anomaly')
plt.axvline(threshold, color='r', linestyle='--')
plt.legend(); plt.title("Reconstruction Error (95th-centile threshold)"); plt.show()
