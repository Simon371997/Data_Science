# nd -array
# GPU support
# Computational graph / Backpropagation
# immutable

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' #um Warnmeldung loszuwerden
import tensorflow as tf

x = tf.constant(4, shape=(1, 1), dtype =tf.float32)
y = tf.ones((3,3))
z = tf.eye(1)
print(x, y, z)

yz = tf.add(y,z)
print(yz)

#reshaping
a = tf.random.normal((2,3))
print(a)
new_a = tf.reshape(a, (3,2))
print(new_a)
new_new_a = tf.reshape(new_a, (-1))
print(new_new_a)

#numpy
a = a.numpy()
print(a, type(a))

a = tf.convert_to_tensor(a)
print(a, type(a))