import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Get class names from dataset
dataset_path = "dataset/color"
data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(128,128),
    batch_size=32
)
class_names = data.class_names

st.title("🌿 Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((128,128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
