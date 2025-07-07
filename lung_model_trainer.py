
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 100
DATA_PATH = "lung_image_sets"

# ✅ Updated categories for binary classification
CATEGORIES = {
    "lung_n": 0,
    "lung_aca": 1,
    "lung_scc": 1
}

def load_data():
    images, labels = [], []
    for category, label in CATEGORIES.items():
        folder = os.path.join(DATA_PATH, category)
        for img in os.listdir(folder)[:1000]:  # Limit to avoid overloading
            try:
                img_arr = cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                images.append(img_arr)
                labels.append(label)
            except:
                pass
    return np.array(images), np.array(labels)

print("📦 Loading Lung Cancer Data...")
X, y = load_data()
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("🫁 Training Lung Cancer CNN Model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("lung_cancer_cnn_model.h5")
print("✅ Model saved as lung_cancer_cnn_model.h5")
