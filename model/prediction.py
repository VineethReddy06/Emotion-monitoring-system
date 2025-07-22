import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from cnn_emotion_model import CNNEmotionModel
from insightface.app import FaceAnalysis
from deepface import DeepFace

# Ensure output directory
os.makedirs("output", exist_ok=True)

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

model = CNNEmotionModel().to(device)
model.load_state_dict(torch.load("model/fer_cnn_model.pth", map_location=device))
model.eval()

# Load image
image_path = "uploads/Training_976799.jpg"
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    print("❌ Could not read image.")
    exit()

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
faces = app.get(img_bgr)

fallback_used = False

if len(faces) == 0:
    print("😐 No face detected with InsightFace. Trying DeepFace...")
    fallback_used = True
else:
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_crop = img_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        image_np = np.expand_dims(np.expand_dims(resized, axis=0), axis=0) / 255.0
        image_tensor = torch.tensor(image_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            top_idx = torch.argmax(probs).item()
            top_score = probs[top_idx].item()

        if top_score < 0.25:
            print(f"🤔 Low confidence ({top_score:.2f}) from CNN. Trying DeepFace...")
            fallback_used = True
        else:
            top_emotion = emotion_labels[top_idx]
            print(f"😀 Detected Emotion: {top_emotion} ({top_score:.2f})")
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, f"{top_emotion} ({top_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# ---- DEEPFACE fallback ----
if fallback_used:
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)[0]
        deep_emotion = result['dominant_emotion']
        deep_score = result['emotion'][deep_emotion] / 100  # Convert percentage to 0–1 scale
        print(f"🔁 DeepFace detected: {deep_emotion} ({deep_score:.2f})")

        # Draw fallback label
        cv2.putText(img_bgr, f"{deep_emotion} ({deep_score:.2f})", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    except Exception as e:
        print(f"❌ DeepFace error: {e}")

# Save result
cv2.imwrite("output/emotion_result.jpg", img_bgr)
print("✅ Saved annotated result to output/emotion_result.jpg")
