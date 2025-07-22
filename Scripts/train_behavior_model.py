import pandas as pd
import os
import joblib
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Step 1: Load the dataset ===
df = pd.read_csv("data/emotion_data.csv")
df['label'] = df['label'].str.lower().str.strip()

# === Step 2: Filter valid emotion labels ===
valid_emotions = ['happy', 'sad', 'angry']
df = df[df['label'].isin(valid_emotions)]

if df.empty:
    raise ValueError("No valid emotion labels found in the dataset.")

# === Step 3: Define realistic behavioral mappings for each emotion ===
behavior_map = {
    'happy': [
        ('Low', 'Normal', 'Moderate'),
        ('Medium', 'Active', 'Frequent'),
        ('High', 'Active', 'Frequent')
    ],
    'sad': [
        ('Low', 'Idle', 'Rare'),
        ('Medium', 'Idle', 'Rare'),
        ('Low', 'Normal', 'Rare')
    ],
    'angry': [
        ('High', 'Active', 'Frequent'),
        ('High', 'Normal', 'Frequent'),
        ('Medium', 'Active', 'Frequent')
    ]
}

# === Step 4: Build a balanced training dataset ===
rows = []
N = 100  # Add 100 examples per emotion for balance
for emotion, combos in behavior_map.items():
    for _ in range(N):
        behavior = random.choice(combos)
        rows.append({
            'label': emotion,
            'app_usage': behavior[0],
            'activity_level': behavior[1],
            'posting_freq': behavior[2]
        })

# === Step 5: Create DataFrame and encode ===
data = pd.DataFrame(rows)
le_app = LabelEncoder()
le_act = LabelEncoder()
le_post = LabelEncoder()
le_emotion = LabelEncoder()

X = pd.DataFrame({
    'app_usage': le_app.fit_transform(data['app_usage']),
    'activity_level': le_act.fit_transform(data['activity_level']),
    'posting_freq': le_post.fit_transform(data['posting_freq'])
})
y = le_emotion.fit_transform(data['label'])

# === Step 6: Train the model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Step 7: Save model and encoders ===
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/behavior_emotion_model.pkl")
joblib.dump(le_app, "model/le_app.pkl")
joblib.dump(le_act, "model/le_act.pkl")
joblib.dump(le_post, "model/le_post.pkl")
joblib.dump(le_emotion, "model/le_emotion.pkl")

print("\u2705 Model and encoders retrained and saved successfully.")
