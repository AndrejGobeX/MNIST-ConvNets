from TFNeuralNet import TFNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import imageio

model = TFNeuralNetwork(28, 10, "Checkpoints/cp.chpt")
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#print(training_images[0])

print("Load existing data? [Y/n]:", end=" ")
x=input()

if not x=='n':
    model.load()
    model.model.summary()
    #keras.models.save_model(model.model, "CNN.h5")

#tflitemodel = tf.lite.TFLiteConverter.from_keras_model(model.model).convert()
#open("net.tflite", "wb").write(tflitemodel)

quit_flag=''
while not quit_flag=='y':

    print("Train model? [y/N]:", end=" ")
    x=input()

    if x=='y':
        epochs=int(input("Number of epochs: "))
        model.fit(training_images, training_labels, epochs)

    predictions = model.predict(test_images)

    print("Evaluate model? [y/N]:", end=" ")
    x=input()

    if x=='y':
        model.evaluate(test_images, test_labels)

    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                    100*np.max(predictions_array),
                                    true_label),
                                    color=color)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    x=int(input("Input test case (-1 to quit): "))
    while x>-1 and x<len(test_images):
        i = x
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(1, 2, 2)
        plot_value_array(i, predictions[i], test_labels)
        plt.show()
        x=int(input("Input test case (-1 to quit): "))

    print("Custom image test? (-1 to quit):", end=" ")
    x=int(input())
    while x>-1:
        expected=x
        custom=[]

        path=input("Image name: ")
        image=imageio.imread("images/"+path+".bmp", as_gray=True)

        custom_input=np.ndarray((1, 28, 28, 1))
        for i in range(28):
            for j in range(28):
                custom_input[0][i][j][0]=(255-image[i][j])/255.0
        
        custom=custom_input
        result=model.predict(custom)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(0, result[0], [expected], custom)
        plt.subplot(1, 2, 2)
        plot_value_array(0, result[0], [expected])
        plt.show()
        print("Custom image test? (-1 to quit):", end=" ")
        x=int(input())
    print("Exit? [y/N]: ", end="")
    quit_flag=input()

x=input("Press any key to exit.")
