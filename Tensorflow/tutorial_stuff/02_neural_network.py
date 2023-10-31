# Es gibt die Sequential API und die Subclassing API (ähnlich zu Pytorch)
# Hier wird vor allem auf die Sequential API eingegangen

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' #um Warnmeldung loszuwerden
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)

# normalize : 0.255 -> 0.1
X_train, X_test = X_train/255.0, X_test/255.0

# Model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())