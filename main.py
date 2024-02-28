import os
import cv2 #for computer vision
import numpy as numpy #for multi-dimensional matrix math
import matplotlib.pyplot as plt 
import tensorflow as tf

#dataset for handwritten numbers -- already split between testing and training data
mnist = tf.keras.datasets.mnist 
#set tuples to training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
    normalization:
        grayscale pixel can have brightness from 0-255
        scaling down to interval of 0-1
'''
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#setting model to standard sequential neural network https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
model = tf.keras.models.Sequential()

'''
    adding layers
    Flatten layer => flatten 2d 28x28 drawing pad into one line of 784 nodes
    Dense layer => simple NN layer with 128 nodes and ReLU activation https://en.wikipedia.org/wiki/Activation_function
    Dense layer => simple NN layer with 128 nodes and ReLU activation https://en.wikipedia.org/wiki/Activation_function
    Output/Last Dense layer => 10 nodes for numbers 0-9, uses softmax activation function https://en.wikipedia.org/wiki/Softmax_function
'''
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

'''
    compile model:
        adam optimizer
        sparse categorical crossentropy loss function
        accuracy for metrics
'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, epochs=10)

#save model
model.save('handwritten.model.keras')