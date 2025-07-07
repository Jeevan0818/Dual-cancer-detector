import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 100
DATA_PATH = "brain_tumor"
CATEGORIES = ["no", "yes"]

def load_data():
    images, labels = [], []
    for label, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_PATH, category)
        for img in os.listdir(path)[:1500]:
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                images.append(img_arr)
                labels.append(label)
            except:
                pass
    return np.array(images), np.array(labels)

print("🔄 Loading Brain Tumor Dataset...")
X, y = load_data()
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
print("🧠 Training Brain Tumor Model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("brain_tumor_cnn_model.h5")
print("✅ Brain Tumor Model Saved.")
