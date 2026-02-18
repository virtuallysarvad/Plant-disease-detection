# 🌿 Plant Disease Detection System

A Deep Learning based web application that detects plant leaf diseases from images.

This project uses Transfer Learning (MobileNetV2) for image classification and Streamlit for the web interface.

---

## 📌 What This Project Does

- Takes an image of a plant leaf
- Predicts the disease (or healthy condition)
- Displays prediction with confidence percentage
- Provides a simple web interface for testing

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Streamlit
- NumPy
- Pillow
- Matplotlib

---

## 📂 Project Structure

Plant-Disease-Detection/
│
├── dataset/
│   └── color/                 # PlantVillage color images (NOT included in repo)
│
├── train.py                   # Model training script
├── app.py                     # Streamlit web app
├── requirements.txt           # Required Python libraries
├── .gitignore
└── README.md

---

## 📦 Dataset Setup

This project uses the PlantVillage dataset from Kaggle.

Download it from:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

After downloading and extracting:

1. Open the extracted folder.
2. Copy ONLY the `color` folder.
3. Paste it inside:

dataset/

Final structure should look like:

dataset/color/Tomato___Healthy  
dataset/color/Potato___Early_blight  
etc.

⚠ The dataset is not included in this repository due to its large size.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

git clone https://github.com/yourusername/plant-disease-detection.git  
cd plant-disease-detection

---

### 2️⃣ Create virtual environment

Windows:

python -m venv venv  
venv\Scripts\activate  

---

### 3️⃣ Install dependencies

pip install -r requirements.txt

---

## 🧠 Train the Model

Before running the app, train the model:

python train.py

This will:
- Load dataset
- Train MobileNetV2 model
- Save the trained model as `plant_disease_model.h5`

---

## 🌐 Run the Web Application

After training is complete:

streamlit run app.py

Your browser will open automatically.

Upload a plant leaf image to test the prediction.

---

## ⚠ Common Issues

### 1. Model file not found
Make sure you ran:
python train.py

### 2. Dataset not loading
Ensure the folder structure is:
dataset/color/<class_folders>

### 3. Training is slow
You can reduce classes by keeping only one plant type inside the `color` folder.

---

## 🎯 Future Improvements

- Add treatment suggestions for each disease
- Fine-tune MobileNet for higher accuracy
- Deploy online using Streamlit Cloud
- Add real-time camera detection

---