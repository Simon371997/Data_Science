import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' #um Warnmeldung loszuwerden

import tensorflow as tf
from tensorflow import keras


mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
# 28, 28
# input_size = 28, seq_lenght = 28

# model
model = keras.models.Sequential()
model.add(keras.Input(shape=(28,28))) # seq_length, input_size
# model.add(keras.layers.SimpleRNN(units=128,return_sequences=True ,activation='relu')) # N, 28, 128
# model.add(keras.layers.GRU(units=128,return_sequences=True ,activation='relu')) # N, 28, 128
# model.add(keras.layers.LSTM(units=128,return_sequences=True ,activation='relu')) # N, 28, 128
model.add(keras.layers.SimpleRNN(units=128,return_sequences=False ,activation='relu')) # N, 128
model.add(keras.layers.Dense(10))

print(model.summary())

# loss and optimzer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']


model.compile(loss=loss, optimizer=optim, metrics=metrics)


# Training
batch_size = 64
epochs = 5

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

# Evaluation
model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)