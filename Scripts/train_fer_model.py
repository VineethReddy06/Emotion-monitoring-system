import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Path to the train/ folder
TRAIN_DIR = r'C:\Users\vinee\Downloads\archive (3)\train'
 # replace this with your actual path

# Map emotion names to labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Step 1: Load the images and labels
X, y = [], []

print("🔄 Loading images...")

for label in emotion_labels:
    emotion_dir = os.path.join(TRAIN_DIR, label)
    for img_file in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_file)
        try:
            # Load image in grayscale and resize
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            img_flat = img.flatten()  # flatten to 1D vector
            X.append(img_flat)
            y.append(label)
        except Exception as e:
            print(f"❌ Error reading {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} images.")

# Step 2: Normalize and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Training completed. Accuracy on train set: {model.score(X_train, y_train)*100:.2f}%")

# Step 4: Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/fer_model.pkl')
joblib.dump(scaler, 'model/fer_scaler.pkl')

print("🎉 Model and scaler saved as model/fer_model.pkl and model/fer_scaler.pkl")
