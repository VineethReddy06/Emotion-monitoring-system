from deepface import DeepFace
import cv2
import os
from collections import Counter

# -----------------------------
# 1️⃣ File paths
# -----------------------------
uploads_dir = "uploads"
img_filename = "Group1.jpg"
img_path = os.path.join(uploads_dir, img_filename)

print(f"📂 Using image: {img_path}")

# -----------------------------
# 2️⃣ Load image
# -----------------------------
img = cv2.imread(img_path)
if img is None:
    print("❌ Failed to load image. Check path!")
    exit()

# -----------------------------
# 3️⃣ Analyze with DeepFace
# -----------------------------
results = DeepFace.analyze(
    img_path=img_path,
    actions=['emotion'],
    enforce_detection=False,
    detector_backend='retinaface'
)

# -----------------------------
# 4️⃣ Handle multi-face output
# -----------------------------
if isinstance(results, list):
    face_results = results
else:
    face_results = [results]

emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "fear": "😨", "surprise": "😲", "neutral": "😐", "disgust": "🤢"
}

# -----------------------------
# 5️⃣ Draw each face
# -----------------------------
all_emotions = []
for idx, face in enumerate(face_results):
    emotion = face['dominant_emotion']
    conf = face['emotion'][emotion]

    region = face['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']

    emoji = emoji_map.get(emotion.lower(), "❓")

    # Draw box + label
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{emotion} {emoji} ({conf:.1f}%)",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    all_emotions.append(emotion.lower())

    print(f"Face {idx+1}: {emotion} {emoji} - Confidence: {conf:.1f}%")

# -----------------------------
# 6️⃣ Get final group emotion
# -----------------------------
from collections import Counter
counter = Counter(all_emotions)
group_emotion = counter.most_common(1)[0][0]
group_emoji = emoji_map.get(group_emotion, "❓")

print(f"\n🎯 Final Group Emotion: {group_emotion} {group_emoji}")

# -----------------------------
# 7️⃣ Save output
# -----------------------------
output_file = "debug_group_result_deepface.jpg"
cv2.imwrite(output_file, img)
print(f"✅ Saved output: {output_file}")
