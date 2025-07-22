import joblib

# Load the model
model = joblib.load("model/fer_model.pkl")

# Print class labels
print("FER Model Classes:", model.classes_)
