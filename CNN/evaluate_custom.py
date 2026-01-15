# evaluate_custom.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

# CONFIG
DATA_DIR = 'CNN/dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Load Model
model = tf.keras.models.load_model("CNN/transfer_face_model.keras")

# Load Data (Only Validation set)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Predict
print("Predicting...")
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1) # Top 1 guess
y_true = validation_generator.classes

# Calculate Accuracy
matches = (y_pred == y_true)
accuracy = np.sum(matches) / len(y_true)
print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

# Generate Report (Precision/Recall per person)
class_names = list(validation_generator.class_indices.keys())
print("\n--- Detailed Report ---")
# Only print first 10 classes to save space, remove slicing [:10] to see all
# Create a list of indices [0, 1, 2, ... 56]
all_labels = np.arange(len(class_names))

# Explicitly pass the labels so it doesn't crash if some are missing in predictions
print(classification_report(y_true, y_pred, target_names=class_names, labels=all_labels))