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

# Model (1.Möglichkeit) feed forward neural net
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())

# loss and optimzer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# fit/training
batch_size = 64
epochs = 5

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2) #verbose for loging
print('Evaluate:')
model.evaluate(X_test, y_test, verbose=2)

# Saving and Loading the model

# Option 1: Saving the whole model:
model.save('nn.h5') #.h5 just saves the model
model.save('neural_net') # creating a whole folder for the model

new_model = keras.models.load_model('nn.h5')


# Option 2: Saving only the weights
model.save_weights('nn.weights.h5')

model.load_weight('nn.weights.h5')


# Option 3: Save only the architecture to json
json_string = model.to_json()

with open('nn_model', 'w') as f:
    f.write(json_string)

with open('nn_model', 'r') as f:
    loaded_json_string = f.read()

new_model = keras.model.model_from_json(loaded_json_string)
print(new_model.summary())

