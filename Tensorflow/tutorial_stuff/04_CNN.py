import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' #um Warnmeldung loszuwerden

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape) #50.000, 32, 32, 3
print(test_images.shape) #10.000, 32, 32, 3

#normalize 0,255 --> 0,1
train_images, test_images = train_images/255.0, test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                'frog', 'horse', 'ship', 'truck']

'''
def show():
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

show()
'''

# model
model = keras.models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3,3), strides=1, padding='valid', activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(32, kernel_size=(3,3), strides=1, padding='valid', activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10))

print(model.summary())

#import sys; sys.exit()

#loss & optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

model.compile(optimizer = optim, loss = loss)

#training
batch_size= 64
num_epochs = 5

history = model.fit(train_images, train_labels, epochs=num_epochs, 
                    batch_size=batch_size, verbose=1, validation_split=0.15)

#evaluation
model.evaluate(test_images, test_labels, batch_size=batch_size)

#plotting results

    #plott loss
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Trainig Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

    #plot accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()