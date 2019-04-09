#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


'''
The Least squares approach to machine learning:
uses mnist database,
getting a ~78% accuracy with one matrix at ~.037 seconds and 5000 samples (used 2000 from another text databae, but the numpy.linalg.solve() method has inaccuracies

Mathematically, the matrix cannot be singular unless there are equivalent inputs, in which case it could just discard the zero rows, but linalg.solve() doesn't do that





NOTES:
5000 training images - 78% accuracy (pretty low, but the max I've gotten is 82% and that was using all 60000 samples, so the return is pretty small)

.03 - .05 seconds consistantly (not an accurate time trial)

Sources of error - 

non linear, tensorflow is going to have higher results because 
it has a nonlinear function (sigmoid) that has no foundation in math at all (pretty black box)

numpy.linalg.solve()
    solve function is pretty generic, LU decomposition would be better as well as treating variables as doubles rather than floats
    
'''





class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
DON'T USE THIS FUNCTION
(you can, but it just decreases accuracy)
This condenses the input vector by averagning chunks so that the size of the input vector is numout
'''
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


'''
Input mnist data

fashion_mnist is of shape
(60000, 28, 28)

this function condenses it to shape
(60000, 1, 784)
as well as shows a nice graph of the images
'''
def importMNIST():
    #import database
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    '''
    Comment this out if you don't want the nice graph at the beginning of the program
    '''
    plt.figure(figsize = (10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+ 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()    
    '''
    End of plot section
    '''
    
    #size of an input vector (784 in this case 28 * 28)
    image_size = train_images[0].shape[1] * train_images[0].shape[1]

    
    #THIS CAN BE OPTIMIZED A LOT, I DIDN'T SPEND MUCH TIME OPTIMIZING THIS
    #reshapes all input and output vectors and scales them by 1000
    train_images = np.array([np.reshape(x, (image_size, 1)) for x in train_images]) / 1000.0
    test_images = np.array([np.reshape(x, (image_size, 1)) for x in test_images]) / 1000.0 #PRODUCES SAME OUTPUT NO MATTER WHAT NUMBER, just a lot slower at higher numbers



    return train_images, train_labels, test_images, test_labels


'''
Creates A and B from the regression method
A is a one vector augmented with all the transposes of the input vectors
if we have m data vectors, A is m * (784 + 1)

B is just all the output vectors augmented transposed
B is m * 10 (10 outputs)
'''
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
    print("\n\n\n==============A (60000 x (784 + 1)===================")
    print(A)
    print("\n\n\n=============B (60000 x 10)====================")
    print(B)
    print("\n\n\n")
    return A, B



'''
Creates alpha and beta from A and B
startIndex is the number of samples you want to test (currently using 5000, 
anything below messes with numpy.linalg.solve, but I'm sure there's a way to fix this)
'''
def solveAB(A, B, startIndex):
    #time the creation of these two matrices
    start = time.time()
    A = A[:startIndex]
    B = B[:startIndex]
    aTransp = A.T * A
    bTransp = A.T * B

    print("\n\n\n=====A transpose A======")
    print(aTransp)
    print("\n\n\n=====A transpose B======")
    print(bTransp)
    omega = np.linalg.solve(aTransp, bTransp)
    alpha = np.matrix(omega[0]).T
    beta = np.matrix(omega[1:]).T
    end = time.time()
    print("\n\nTime Taken to train model: %f seconds" % (end - start))
    print("\n\n\n=============ALPHA================")
    print(alpha)
    print("\n\n\n=============BETA===============")
    print(beta)
    print("\n\n\n")
    return alpha, beta


#Inputs all 10000 test_images and labels and tests them on the trained model
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
    
    #PLOTS A HISTOGRAM OF THE COMMONLY MISSED ITEMS    
    n, bins, patches = plt.hist(errorsa, bins = 10)
    plt.show()



    print("Accuracy: out of %d samples, %d where incorrect -- %f%% accuracy" % (test_images.shape[0], dif, (test_images.shape[0] - dif) / test_images.shape[0]))
    return (test_images.shape[0] - dif) / test_images.shape[0]
    

if __name__ == "__main__":    
    train_images, train_labels, test_images, test_labels = importMNIST()
    A, B = createAB(train_images, train_labels)
    #sample size 5000
    alpha, beta = solveAB(A, B, 5000)
    testAccuracy(test_images, test_labels, alpha, beta)

