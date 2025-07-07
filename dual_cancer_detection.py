import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 100

# --- Load both models once ---
print("📦 Loading models...")
brain_model = load_model("brain_tumor_cnn_model.h5")
lung_model = load_model("lung_cancer_cnn_model.h5")
print("✅ Models ready.")

def preprocess_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None

def predict(img_path, model, disease_name, labels):
    print(f"🔍 Predicting {disease_name}...")
    img = preprocess_image(img_path)
    if img is None:
        print("🚫 Unable to process image.")
        return
    prediction = model.predict(img)[0][0]
    label = labels[1] if prediction > 0.5 else labels[0]
    emoji = "🟥" if prediction > 0.5 else "🟩"
    print(f"📊 {disease_name} Result: {label} {emoji}")

# --- User Interaction ---
while True:
    print("\n🔸 Choose detection:")
    print("1. Brain Tumor")
    print("2. Lung Cancer")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        img_path = input("🧠 Enter path to Brain MRI image: ")
        predict(img_path, brain_model, "Brain Tumor", ["No Tumor", "Tumor Detected"])
    elif choice == '2':
        img_path = input("🫁 Enter path to Lung image: ")
        predict(img_path, lung_model, "Lung Cancer", ["Lung Normal", "Cancer Detected"])
    elif choice == '3':
        print("👋 Exiting. Stay healthy!")
        break
    else:
        print("❌ Invalid input. Try again.")
