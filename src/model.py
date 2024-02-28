import os
import cv2 #for computer vision
import numpy as np #for multi-dimensional matrix math
import matplotlib.pyplot as plt 
import tensorflow as tf

#dataset for handwritten numbers -- already split between testing and training data
mnist = tf.keras.datasets.mnist 
#set tuples to training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model("handwritten.model.keras")

image_number = 0
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") #print the node with the highest confidence
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1




