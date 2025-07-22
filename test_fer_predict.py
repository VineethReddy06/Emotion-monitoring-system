import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import os

# Define the same model architecture used during training
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),           # <- THIS IS REQUIRED
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc(x)
        return x
 

# Labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load model
device = torch.device("cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("model/fer_cnn_model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# Image path
image_path = "uploads/Cryinggirl.jpg"  # Change as needed

# Load image and detect face
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("❌ No face detected.")
    exit()

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    face_img = cv2.resize(roi_gray, (48, 48))
    pil_img = Image.fromarray(face_img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Save preprocessed image for debugging
    debug_path = 'debug_face.jpg'
    cv2.imwrite(debug_path, face_img)
    print(f"📎 Saved preprocessed face as '{debug_path}'")

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    top_pred = np.argmax(probs)
    print(f"\n🖼 Using image: {image_path}")
    print(f"\n✅ Predicted Emotion: {emotion_labels[top_pred]}")
    print(f"🎯 Confidence: {probs[top_pred]:.3f}")

    # Show all probabilities
    print(f"\n📊 All Emotion Probabilities:")
    for i, emotion in enumerate(emotion_labels):
        print(f"{emotion:>10}: {probs[i]:.3f}")

    break  # Only process the first detected face
