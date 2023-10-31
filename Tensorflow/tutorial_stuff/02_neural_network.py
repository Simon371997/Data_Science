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

# Model (1.Möglichkeit)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())

# 2.Möglichkeit
#model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(28,28)))
#model.add(keras.layers.Dense(128, activation='relu'))
#model.add(keras.layers.Dense(10))

# loss and optimzer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2) #verbose for loging


# evaluate
model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)


# predictions
    # 1.Option add Softmax to the model
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(X_test)
pred0 = predictions[0]
print(pred0)
label0 =np.argmax(pred0)
print(label0)

    # 2.Option model + softmax
predictions = model(X_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 =np.argmax(pred0)
print(label0)

pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)