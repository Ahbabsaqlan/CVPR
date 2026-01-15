import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# CONFIG
DATA_DIR = 'CNN/dataset'
IMG_SIZE = 128
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2
)

print("Loading Data...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Training on {num_classes} classes.")

# 2. Load the Pre-Trained Model 
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')


base_model.trainable = False 

# 3. Add Custom Head
model = models.Sequential([
    base_model,                         
    layers.GlobalAveragePooling2D(),    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),                
    layers.Dense(num_classes, activation='softmax') 
])

# 4. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Callbacks
checkpoint = ModelCheckpoint("CNN/transfer_face_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 6. Train
print("Starting Transfer Learning...")
history = model.fit(
    train_generator,
    epochs=10, 
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# 7. Save Labels
class_names = list(train_generator.class_indices.keys())
with open("CNN/labels_transfer.txt", "w") as f:
    f.write("\n".join(class_names))

print("Done! Run 'webcam_transfer.py'")