from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import numpy as np
import pytesseract
from PIL import Image
import torch
import torch.nn as nn
import cv2
from transformers import pipeline
from insightface.app import FaceAnalysis
from deepface import DeepFace
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector

# 🟢 Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🟢 Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="2046",
    database="emotionmonitoringsystem"
)
cursor = db.cursor(dictionary=True)

# 🟢 Text emotion pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# 🟢 CNN model for FER
class CNNEmotionModel(nn.Module):
    def __init__(self):
        super(CNNEmotionModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNEmotionModel().to(device)
cnn_model.load_state_dict(torch.load("model/fer_cnn_model.pth", map_location=device))
cnn_model.eval()

# 🟢 Face detection
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# 🟢 Flask app config
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🟢 Landing
@app.route('/')
def landing():
    return render_template("landing.html")

# 🟢 Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user:
            flash('Email already exists. Please login.')
            return redirect(url_for('login'))

        hashed_pw = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)",
            (full_name, email, hashed_pw)
        )
        db.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template("register.html")

# 🟢 Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        print(f"👉 You typed: EMAIL = '{email}' | PASSWORD = '{password}'")

        cursor.execute(
            "SELECT * FROM users WHERE email = %s AND password = %s",
            (email, password)
        )
        user = cursor.fetchone()

        print(f"👉 DB returned: {user}")

        if user:
            session['username'] = user['full_name']
            session['history'] = []
            session['play_greeting'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('login'))

    return render_template("login.html")


# 🟢 Dashboard
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session.get('username')
    mood_data = session.get('history', [])
    play_greeting = session.pop('play_greeting', False)

    emotion_counts = {
        "Joy": 40, "Anger": 20, "Sadness": 15,
        "Surprise": 10, "Fear": 5, "Disgust": 5, "Neutral": 5
    }

    return render_template(
        "index.html",
        username=username,
        play_greeting=play_greeting,
        mood_data=mood_data,
        emotion_counts=emotion_counts
    )

# 🟢 Profile
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        flash('Please login first.')
        return redirect(url_for('login'))

    cursor.execute(
        "SELECT * FROM users WHERE full_name = %s",
        (session['username'],)
    )
    user = cursor.fetchone()

    cursor.execute(
        "SELECT * FROM emotion_history WHERE user_id = %s",
        (user['id'],)
    )
    history = cursor.fetchall()

    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']

        cursor.execute(
            "UPDATE users SET full_name = %s, email = %s, password = %s WHERE id = %s",
            (full_name, email, password, user['id'])
        )
        db.commit()
        session['username'] = full_name
        flash('Profile updated!')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user, history=history)

# 🟢 Logout
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

# 🟢 Predict
# 🟢 Predict
@app.route('/predict', methods=['POST'])
def predict():
    input_type = request.form['type']
    result, emoji, score = "", "", 0

    if input_type == "text":
        text = request.form['text_input']
        result, emoji, score = analyze_emotion(text)
    elif input_type in ["social", "image"]:
        file = request.files['social_file'] if input_type == "social" else request.files['image']
        if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            if input_type == "social":
                image = Image.open(filepath)
                extracted_text = pytesseract.image_to_string(image).strip()
                if extracted_text:
                    result, emoji, score = analyze_emotion(extracted_text)
                else:
                    result, emoji, score = analyze_image(filepath)
            else:
                result, emoji, score = analyze_image(filepath)
        else:
            result, emoji, score = "Invalid image format.", "⚠", 0

    # ✅ Keep local session history
    history = session.get('history', [])
    history.append(score)
    if len(history) > 10:
        history = history[-10:]
    session['history'] = history

    # ✅ ✅ ✅ NEW: Store in emotion_history table!
    if 'username' in session:
        cursor.execute(
            "SELECT id FROM users WHERE full_name = %s",
            (session['username'],)
        )
        user = cursor.fetchone()
        if user:
            cursor.execute(
                "INSERT INTO emotion_history (user_id, input_type, result, polarity) VALUES (%s, %s, %s, %s)",
                (user['id'], input_type, result, score)
            )
            db.commit()

    emotion_counts = {
        "Joy": 40, "Anger": 20, "Sadness": 15,
        "Surprise": 10, "Fear": 5, "Disgust": 5, "Neutral": 5
    }

    return render_template(
        "index.html",
        username=session.get('username', 'User'),
        result=result,
        emoji=emoji,
        play_greeting=False,
        mood_data=history,
        emotion_counts=emotion_counts
    )


# 🟢 Screen Time
@app.route('/screen-time', methods=['GET', 'POST'])
def screen_time_form():
    if request.method == 'POST':
        try:
            hours = None
            if 'screen_time' in request.form and request.form['screen_time']:
                hours = float(request.form['screen_time'])
            elif 'screen_image' in request.files:
                file = request.files['screen_image']
                if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                    file.save(path)
                    image = Image.open(path)
                    extracted_text = pytesseract.image_to_string(image)

                    import re
                    match = re.search(r"(\d+)\s*h(?:ours?)?\s*(\d+)?\s*m?", extracted_text.lower())
                    if match:
                        h = int(match.group(1))
                        m = int(match.group(2)) if match.group(2) else 0
                        hours = h + m / 60

            if hours is None:
                raise ValueError("No valid input detected.")

            if hours <= 2:
                emotion, emoji, msg = "Happy", "😊", "Good job managing your screen time!"
            elif 2 < hours <= 4:
                emotion, emoji, msg = "Sad", "😢", "Try to reduce your screen time a bit."
            else:
                emotion, emoji, msg = "Angry", "😠", "Warning: High screen time detected!"

            return render_template("screen_time.html",
                                   username=session.get('username', 'User'),
                                   result=emotion, emoji=emoji, message=msg)

        except Exception as e:
            return render_template("screen_time.html",
                                   username=session.get('username', 'User'),
                                   result="Error", emoji="⚠", message=str(e))

    return render_template("screen_time.html", username=session.get('username', 'User'))

# 🟢 Analyze text
def analyze_emotion(text):
    try:
        predictions = emotion_classifier(text)
        top_emotion = predictions[0][0]["label"].lower()
        score = predictions[0][0]["score"]

        emoji_map = {
            "joy": "😊", "anger": "😠", "surprise": "😲",
            "sadness": "😢", "fear": "😨", "disgust": "🤢", "neutral": "😐"
        }
        polarity = {
            "joy": 0.8, "surprise": 0.4, "neutral": 0,
            "sadness": -0.6, "anger": -0.8, "fear": -0.7, "disgust": -0.9
        }

        if score < 0.3:
            return "Neutral", "😐", 0

        return top_emotion.capitalize(), emoji_map.get(top_emotion, "😐"), polarity.get(top_emotion, 0)
    except Exception as e:
        return f"Error: {e}", "⚠", 0

# 🟢 Analyze image
def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Image not found.", "⚠", 0

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(img_rgb)

        if faces:
            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(gray, (48, 48))
            image_np = np.array(face_resized, dtype=np.float32) / 255.0
            image_np = np.expand_dims(image_np, axis=0)
            image_np = np.expand_dims(image_np, axis=0)
            image_tensor = torch.tensor(image_np, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = cnn_model(image_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                top_idx = torch.argmax(probs).item()
                top_score = probs[top_idx].item()

            top_emotion = emotion_labels[top_idx]
            emoji_map = {
                "happy": "😊", "sad": "😢", "angry": "😠",
                "fear": "😨", "surprise": "😲", "neutral": "😐", "disgust": "🤢"
            }
            polarity = {
                "happy": 0.8, "sad": -0.6, "angry": -0.8,
                "fear": -0.7, "surprise": 0.4, "neutral": 0, "disgust": -0.9
            }

            if top_score < 0.1:
                return "Neutral", "😐", 0

            return top_emotion.capitalize(), emoji_map.get(top_emotion, "😐"), polarity.get(top_emotion, 0)

        # Fallback: DeepFace
        results = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        if isinstance(results, dict):
            results = [results]
        emotions = [r['dominant_emotion'].lower() for r in results]
        final_emotion = max(set(emotions), key=emotions.count)

        emoji_map = {
            "happy": "😊", "sad": "😢", "angry": "😠",
            "fear": "😨", "surprise": "😲", "neutral": "😐", "disgust": "🤢"
        }
        polarity = {
            "happy": 0.8, "sad": -0.6, "angry": -0.8,
            "fear": -0.7, "surprise": 0.4, "neutral": 0, "disgust": -0.9
        }


        return final_emotion.capitalize(), emoji_map.get(final_emotion, "😐"), polarity.get(final_emotion, 0)

    except Exception as e:
        return f"Error processing image: {e}", "⚠", 0

# 🟢 Run
if __name__ == '__main__':
    app.run(debug=True)