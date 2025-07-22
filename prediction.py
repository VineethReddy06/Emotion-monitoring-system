# prediction.py

import cv2
import torch
import numpy as np
from torchvision import transforms
from model.cnn_emotion_model import CNNEmotionModel  # Make sure this file exists
from PIL import Image

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load model
model = CNNEmotionModel()
model.load_state_dict(torch.load("model/fer_cnn_model.pth", map_location=torch.device("cpu")))
model.eval()

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image path to test
image_path = "uploads/test.jpg"  # Change this to your actual test image path
img = cv2.imread(image_path)

if img is None:
    print("❌ Could not read the image. Please check the path.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("😕 No face detected.")
else:
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_pil = Image.fromarray(roi_gray).convert("L")
        roi_transformed = transform(roi_pil)
        roi_transformed = roi_transformed.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(roi_transformed)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        print(f"😀 Detected Emotion: {emotion}")
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display result
    cv2.imshow('Emotion Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
