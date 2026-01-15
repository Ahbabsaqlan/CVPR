# webcam_final.py
import cv2
import numpy as np
import tensorflow as tf

# CONFIG
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.60  # Only show name if 60% sure

# 1. Load Model & Labels
print("Loading Final Model...")
model = tf.keras.models.load_model('CNN/final_face_model.keras')

print("Loading Labels...")
try:
    with open("CNN/labels_final.txt", "r") as f:
        class_names = f.read().split("\n")
except FileNotFoundError:
    print("Error: labels_final.txt not found. Run training script first.")
    exit()

# 2. Setup Camera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting Face Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # --- Preprocessing (Must match Training exactly) ---
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # 1. Resize to 128x128
            roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            
            # 2. Convert BGR to RGB (Keras models are trained on RGB)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # 3. Normalize (0-1)
            roi = roi.astype("float") / 255.0
            
            # 4. Add Batch Dimension (1, 128, 128, 3)
            roi = np.expand_dims(roi, axis=0)

            # --- Prediction ---
            preds = model.predict(roi, verbose=0)[0]
            
            # Get Top Result
            class_idx = np.argmax(preds)      # The Index Number (0 to 70)
            confidence = preds[class_idx]     # The Probability (0.0 to 1.0)
            person_name = class_names[class_idx]

            # --- Visualization ---
            if confidence > CONFIDENCE_THRESHOLD:
                # Green Box for known people
                color = (0, 255, 0) 
                label_text = f"[{class_idx}] {person_name}"
                conf_text = f"{int(confidence * 100)}%"
            else:
                # Red Box for unknown/uncertain
                color = (0, 0, 255)
                label_text = f"Unknown"
                conf_text = f"{int(confidence * 100)}%"

            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw Background for Text (Readability)
            cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
            
            # Draw Name & Index
            cv2.putText(frame, label_text, (x + 5, y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw Confidence Percentage
            cv2.putText(frame, conf_text, (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            print(e)
            pass

    cv2.imshow("Final Face ID System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()