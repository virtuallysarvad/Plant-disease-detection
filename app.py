import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# -------------------------------
# Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Class Names
# -------------------------------
dataset = datasets.ImageFolder("dataset/color")
class_names = dataset.classes
num_classes = len(class_names)

# -------------------------------
# Load Model
# -------------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model.load_state_dict(torch.load("plant_model.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Image Transform (Must match training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 Plant Disease Detection System (PyTorch + GPU)")

st.write("Upload a leaf image to detect plant disease.")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100

    st.success(f"Prediction: {predicted_label}")
    st.info(f"Confidence: {confidence_score:.2f}%")