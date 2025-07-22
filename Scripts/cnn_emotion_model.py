import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
label2idx = {label: idx for idx, label in enumerate(emotion_labels)}

# Configuration
BATCH_SIZE = 64
EPOCHS = 30
IMG_SIZE = 48

# Correct raw paths
DATA_DIR = {
    "train": r"C:\\Users\\vinee\\Downloads\\archive (3)\\train",
    "test": r"C:\\Users\\vinee\\Downloads\\archive (3)\\test"
}

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append((img_path, label2idx[class_name]))

        print(f"🗂 Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# Transform
# Transform with data augmentation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Load datasets
train_dataset = EmotionDataset(DATA_DIR["train"], transform=transform)
test_dataset = EmotionDataset(DATA_DIR["test"], transform=transform)

# Debug checks
print("✅ Total training samples:", len(train_dataset))
print("✅ Total test samples:", len(test_dataset))

# Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Compute class weights
all_labels = [label for _, label in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Simple CNN model
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
            nn.Linear(128, 7)  # 7 classes: angry, disgust, fear, happy, sad, surprise, neutral
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Instantiate model
model = CNNEmotionModel().to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"✅ Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        print(f"🔥 Epoch {epoch+1} completed, Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/fer_cnn_model.pth")
    print("✅ Model saved to 'model/fer_cnn_model.pth'")

# Evaluation function
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"🧪 Test Accuracy: {100 * correct / total:.2f}%")

# Run training & evaluation
if __name__ == "__main__":
    train()
    evaluate()