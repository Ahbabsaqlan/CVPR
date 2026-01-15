import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


print("Loading data...")
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train / 255.0, x_test / 255.0


model = models.Sequential([
    layers.Input(shape=(28, 28)),             
    layers.Flatten(),                         
    layers.Dense(128, activation='relu'),     
    layers.Dropout(0.2),                      
    layers.Dense(10, activation='softmax')    
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("Starting training... (This might take 10-30 seconds)")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc}")


model.save('mnist_mlp.h5')
print("\nSUCCESS: Model saved as 'mnist_mlp.h5'")