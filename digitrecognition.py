import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = train_images[0].shape[1] * train_images[0].shape[1]

a = np.array([np.zeros(numin) for i in range(numout)])
image_size = train_images[0].shape[1] * train_images[0].shape[1]
train_images = np.array([np.reshape(x, (image_size, 1)) for x in train_images])
test_images = np.array([np.reshape(x, (image_size, 1)) for x in test_images])

train_images = np.array([a * i for i in train_images]) #/ 255.0

print(train_images)
test_images = np.array([a * i for i in test_images]) #/ 255.0
print(train_images.shape)
print(test_images.shape)


