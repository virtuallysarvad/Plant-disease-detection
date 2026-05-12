import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image to detect plant disease.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------
# Load Class Names
# ----------------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ----------------------------------
# Load Model (Cached)
# ----------------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load("plant_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ----------------------------------
# Transform
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

    raw_label = class_names[predicted_class.item()]
    # Clean the label (e.g., convert "Tomato___Bacterial_spot" to "Tomato Bacterial Spot")
    clean_label = raw_label.replace("_", " ").replace("  ", " ").strip().title()
    confidence_score = confidence.item() * 100

    st.markdown("---")
    
    # Create two columns to display image and results side-by-side
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Diagnostic Results")
        
        # Display the full condition name without truncation
        st.markdown(f"### Detected Condition:")
        st.markdown(f"#### <span style='color:#FF4B4B'>{clean_label}</span>" if "healthy" not in clean_label.lower() else f"#### <span style='color:#00CC96'>{clean_label}</span>", unsafe_allow_html=True)
        
        st.markdown(f"**Confidence Score: {confidence_score:.2f}%**")
        st.progress(float(confidence.item()))
        
        if "healthy" in clean_label.lower():
            st.success("Great news! The plant appears to be healthy. 🌱")
        else:
            st.error(f"Attention! The plant shows signs of **{clean_label}**. ⚠️")