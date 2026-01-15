# generate_data.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- CONFIGURATION ---
DATA_DIR = 'CNN/dataset'
TARGET_COUNT = 500
IMG_SIZE = 128  
# ---------------------

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

print(f"--- STARTING DATA GENERATION (SAFE MODE) ---")

folders = sorted(os.listdir(DATA_DIR))

for person_name in folders:
    person_dir = os.path.join(DATA_DIR, person_name)
    
    # Validation checks
    if not os.path.isdir(person_dir): continue
    
    # Filter only valid images
    valid_extensions = ('.png', '.jpg', '.jpeg')
    existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(valid_extensions)]
    current_count = len(existing_files)
    
    # Clean up empty folders logic
    if current_count == 0:
        print(f"⚠️  WARNING: {person_name} is empty. Please delete this folder manually.")
        continue
    
    if current_count >= TARGET_COUNT:
        print(f"Skipping {person_name}: Has {current_count} images.")
        continue

    print(f"Processing {person_name}...")
    
    images = []
    for img_name in existing_files:
        try:
            img_path = os.path.join(person_dir, img_name)
            
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            images.append(x)
        except Exception as e:
            print(f"   - Corrupt file ignored: {img_name}")

    if len(images) > 0:
        try:
            image_batch = np.vstack(images)

            count = 0
            needed = TARGET_COUNT - current_count
            
            for batch in datagen.flow(image_batch, 
                                      batch_size=1, 
                                      save_to_dir=person_dir, 
                                      save_prefix='aug', 
                                      save_format='jpg'):
                count += 1
                if count >= needed:
                    break
            print(f"   - Success! Added {count} images.")
            
        except ValueError as e:
            print(f"   - CRITICAL ERROR in {person_name}: {e}")
            print("   - Suggestion: Delete this folder and retake photos.")

print("\n--- GENERATION COMPLETE ---")