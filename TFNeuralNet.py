import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

class TFNeuralNetwork():

    def __init__(self, no_inputs, no_outputs, checkpoint_path):
        """self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(no_inputs, no_inputs)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(no_outputs)
        ])"""

        #784 - [32C5-P2] - [64C5-P2] - 128 - 10
        self.model = keras.Sequential([
            keras.layers.Conv2D(32,kernel_size=5,activation='relu',input_shape=(no_inputs,no_inputs,1)),
            keras.layers.MaxPool2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(64,kernel_size=5,activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(no_outputs, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        """self.probability_model = keras.Sequential([
            self.model,
            keras.layers.Softmax()
        ])"""

        self.path=checkpoint_path
        self.checkpoint_callback=keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=2
        )
    
    def fit(self, train_input, train_output, no_epochs):
        self.model.fit(train_input, train_output, epochs=no_epochs,
            validation_data=None,
            callbacks=[self.checkpoint_callback])

    def evaluate(self, test_input, test_output):
        self.model.evaluate(test_input, test_output, verbose=1)

    def predict(self, test_input):
        return self.model.predict(test_input)

    def summary(self):
        self.model.summary()

    def load(self):
        self.model.load_weights(self.path)