#from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import time

import numpy as np


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0


image_size = train_images[0].shape[1] * train_images[0].shape[1]
samples = train_images.shape[0]



train_images = np.array([np.reshape(x, (1, image_size)) for x in train_images])

    

#train_images = np.array([np.mean(x[i:i+batch]) for x, i in train_images, range(len(train_images.shape[0] - 5))])

labels = np.array([np.zeros(10) for i in range(samples)])
for i in range(samples):
    labels[i][train_labels[i]] = 1

B = np.matrix(np.stack([x for x in labels]))

V = np.matrix(np.array([1 for i in range(samples)])).T

AA = np.matrix(train_images)

A = np.hstack((V, AA))


#at this point, A is a 60000 x 784 matrix - not ideal, so we'll just take the first 100 vals
A = A[4000:]
B = B[4000:]
start = time.time()
aTransp = A.T * A
bTransp = A.T * B


#ideally i want to solve for  the system Atransp * omega = Btransp, but Atransp is 
omega = np.linalg.solve(aTransp, bTransp)

alpha = np.matrix(omega[0]).T
beta = np.matrix(omega[1:]).T
end = time.time()
changeinTime = end - start


results = [alpha + (beta * np.matrix(a).T) for a in train_images[4000:]]
print(results)
labels = np.array([np.zeros(10) for i in range(samples)])
for i in range(samples):
    labels[i][train_labels[i]] = 1
print("1")
B = np.matrix(np.stack([x for x in labels]))
print(2)
V = np.matrix(np.array([1 for i in range(2000)])).T
print(3)
AA = np.matrix([x for x in results[0]])
print(4)
A = np.hstack((V, AA))
start = time.time()
aTransp = A.T * A
bTransp = A.T * B
omega = np.linalg.solve(aTransp, bTransp)

alpha = np.matrix(omega[0]).T
beta = np.matrix(omega[1:]).T
end = time.time()
changeinTime = end - start





#print("ALPHA")
#print(alpha)
#print("BETA")
#print(beta)



dif = 0

for i in range(test_images.shape[0]):
    value = test_labels[i]
    a = np.reshape(test_images[i], (1, image_size))
    a = np.matrix(a[0]).T /  255.0
    result = np.array(((beta * a) + alpha).T)[0]
    a = result.argmax()
    if(a != value):
        dif = dif+1

print(alpha)
print(beta)
print(test_images.shape)
print(dif)
print("Data finished: accuracy = %f %%" % (100 * ((test_images.shape[0]) - dif) / test_images.shape[0]))

print("Training Time taken: %f seconds" % (changeinTime))
