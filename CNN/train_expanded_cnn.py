# train_expanded_cnn.py
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CONFIG
DATA_DIR = 'CNN/dataset'
IMG_SIZE = 128
BATCH_SIZE = 64 


train_datagen = ImageDataGenerator(
    rescale=1./255,
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
print(f"Training on {num_classes} people.")


model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 4 
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dropout(0.5), 
    
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 3. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Callbacks
checkpoint = ModelCheckpoint("CNN/final_face_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. Train
history = model.fit(
    train_generator,
    epochs=20, 
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# 6. Save Labels
class_names = list(train_generator.class_indices.keys())
with open("CNN/labels_final.txt", "w") as f:
    f.write("\n".join(class_names))

print("Done! Run your webcam script using 'final_face_model.keras'")