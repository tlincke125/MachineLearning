#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def importMNIST(numout):
    digit_mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()
    image_size = train_images[0].shape[1] * train_images[0].shape[1]
    
    numin = image_size
 
    a = np.array([np.zeros(numin) for i in range(numout)])
    i = 0
    j = 0
    while(i < numout and j < numin):
        for x in range(numin // numout):
            a[i][j+x] = 1
        i = i + 1
        j = j + numin // numout
    a = np.matrix(a)
    image_size = train_images[0].shape[1] * train_images[0].shape[1]
    train_images = np.array([np.reshape(x, (image_size, 1)) for x in train_images])
    test_images = np.array([np.reshape(x, (image_size, 1)) for x in test_images])
    
    train_images = np.array([a * i for i in train_images]) #/ 255.0
    
    print(train_images) 
    test_images = np.array([a * i for i in test_images]) #/ 255.0
    print(train_images.shape)
    print(test_images.shape)
    return train_images, train_labels, test_images, test_labels


def importMNIST():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    plt.figure(figsize = (10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+ 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()    
    image_size = train_images[0].shape[1] * train_images[0].shape[1]
    train_images = np.array([np.reshape(x, (image_size, 1)) for x in train_images])
    test_images = np.array([np.reshape(x, (image_size, 1)) for x in test_images])
    train_images = train_images / 1000.0 #PRODUCES SAME OUTPUT NO MATTER WHAT NUMBER
    test_images = test_images / 1000.0
    return train_images, train_labels, test_images, test_labels

def createAB(train_images, train_labels):
    image_size = train_images.shape[1]
    samples = train_images.shape[0]
    maxLab = max(train_labels)
    minLab = min(train_labels)
    outputSize = maxLab - minLab
        
    labels = np.array([np.zeros(outputSize + 1) for i in range(samples)])

    for i in range(samples):
        labels[i][train_labels[i] - minLab] = 1
    
    B = np.matrix(np.stack([x for x in labels]))
    A = np.hstack((np.matrix(np.array([1 for i in range(samples)])).T, np.matrix(train_images)))
    print(A[0])
    print(B[0])
    return A, B

def solveAB(A, B, startIndex):
    start = time.time()
    print(A.shape)
    A = A[:startIndex]
    B = B[:startIndex]
    print(A.shape)
    aTransp = A.T * A
    bTransp = A.T * B
    print(aTransp[0])
    print(bTransp[0])
    omega = np.linalg.solve(aTransp, bTransp)
    alpha = np.matrix(omega[0]).T
    beta = np.matrix(omega[1:]).T
    end = time.time()
    print("Time Taken to train model: %f seconds" % (end - start))
    print(alpha, beta)
    return alpha, beta

def testAccuracy(test_images, test_labels, alpha, beta):
    image_size = test_images.shape[1]
    dif = 0
    errorsa = np.array([], np.int32)
    errorsV = np.array([], np.int32)
    print(test_images.shape)
    for i in range(test_images.shape[0]):
        value = test_labels[i]
        a = np.reshape(test_images[i], (1, image_size))
        a = np.matrix(a[0]).T
        result = np.array(((beta * a) + alpha).T)[0]
        a = result.argmax()
        if(a != value ):
            errorsa = np.append(errorsa, int(a))
            errorsV = np.append(errorsV, int(value))
            dif = dif+1
    
    n, bins, patches = plt.hist(errorsa, bins = 10)
    plt.show()



    print("Accuracy: out of %d samples, %d where incorrect -- %f%% accuracy" % (test_images.shape[0], dif, (test_images.shape[0] - dif) / test_images.shape[0]))
    return (test_images.shape[0] - dif) / test_images.shape[0]
    

if __name__ == "__main__":    
    train_images, train_labels, test_images, test_labels = importMNIST()
    A, B = createAB(train_images, train_labels)
    alpha, beta = solveAB(A, B, 20000)
    testAccuracy(test_images, test_labels, alpha, beta)
