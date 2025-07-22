import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Load dataset
csv_path = "data/emotion_data.csv"
df = pd.read_csv(csv_path)

# ✅ Separate label
if 'label' not in df.columns:
    raise ValueError("❌ 'label' column not found in dataset.")
y = df['label']

# ✅ Select numeric features (excluding label)
X_raw = df.select_dtypes(include=[np.number])

if len(X_raw) < 1 or X_raw.shape[0] != y.shape[0]:
    raise ValueError("❌ Mismatch between features and labels or empty numeric data.")

# ✅ Add statistical features per row (row-wise mean, std, etc. if needed)
X = pd.DataFrame()
for col in X_raw.columns:
    X[f"{col}_mean"] = X_raw[col]
    # Optionally add other per-column transformations here

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save model, scaler, and feature list
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(scaler, "model/eeg_scaler.pkl")
joblib.dump(X.columns.tolist(), "model/eeg_features.pkl")

print("✅ EEG emotion model, scaler, and feature names saved!")
