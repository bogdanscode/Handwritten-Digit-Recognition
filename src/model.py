import os  # Provides a way of using operating system dependent functionality
import cv2  # For computer vision tasks
import numpy as np  # For multi-dimensional matrix operations
import matplotlib.pyplot as plt  # For plotting graphs and displaying images
import tensorflow as tf  # For building and training neural network models

# Load the MNIST dataset, which contains handwritten digits. The dataset is already split between training and testing data
mnist = tf.keras.datasets.mnist 
# Assign the training and testing data to respective variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load the pre-trained model from the file 'handwritten.model.keras'
model = tf.keras.models.load_model("handwritten.model.keras")

image_number = 0  # Initialize the counter for image number
# Loop through the images in the 'digits' folder until there are no more images
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # Read the image, convert it to grayscale (by taking only one channel with [:,:,0])
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        # Invert the colors of the image and reshape for the model input
        img = np.invert(np.array([img]))
        # Predict the digit using the loaded model
        prediction = model.predict(img)
        # Print the predicted digit by finding the index of the highest confidence score
        print(f"This digit is probably a {np.argmax(prediction)}")  # Print the prediction
        # Show the image using matplotlib
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        # Print "Error!" if there are issues during processing the image or predicting
        print("Error!")
    finally:
        # Increment the image_number to move to the next image
        image_number += 1
