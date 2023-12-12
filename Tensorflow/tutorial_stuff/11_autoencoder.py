import tensorflow as tf
from tensorflow.keras import layers, models

input_shape = (512, 512, 3)  # Anpassen an die Dimensionen Ihrer Bilder
encoding_dim = 512  # Anpassen je nach gewünschter Dimensionalität des latenten Raums

import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Definieren Sie den Pfad zu Ihrem Ordner mit Bildern
image_folder_path = "./data/harry-potter/0001"

# Definieren Sie die Größe der Bilder, passen Sie sie an Ihre Bedürfnisse an
img_width, img_height = 512, 512

# Laden und normalisieren Sie die Bilder
def load_and_preprocess_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=(img_width, img_height))
        img = img_to_array(img) / 255.0  # Normalisierung auf den Bereich [0, 1]
        images.append(img)
    return np.array(images)

# Laden Sie die Bilder und erstellen Sie Trainingsdaten
images = load_and_preprocess_images(image_folder_path)

# Teilen Sie die Daten in Trainings- und Testsets auf
train_images, test_images = train_test_split(images, test_size=0.1, random_state=42)

# Überprüfen Sie die Form der Trainingsdaten
print("Shape of training data:", train_images.shape)
print("Shape of test data:", test_images.shape)







model = models.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(encoding_dim, activation='relu'))

model.add(layers.Dense(512 * 512 * 3, activation='sigmoid'))
model.add(layers.Reshape(input_shape))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_images, train_images, epochs=5, batch_size=4,
          validation_data = (test_images, test_images))