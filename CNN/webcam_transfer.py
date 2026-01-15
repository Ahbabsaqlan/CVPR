import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# CONFIG
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.70 

print("Loading Transfer Model...")
model = tf.keras.models.load_model('CNN/transfer_face_model.keras')

try:
    with open("CNN/labels_transfer.txt", "r") as f:
        class_names = f.read().split("\n")
except FileNotFoundError:
    print("Error: labels_transfer.txt not found.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # 1. Resize
            roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            
            # 2. Convert BGR to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # 3. Preprocess for MobileNet
            # Convert to float array
            roi = img_to_array(roi) 
            # Add Batch Dimension (1, 128, 128, 3)
            roi = np.expand_dims(roi, axis=0)
            # Apply MobileNet specific math (-1 to 1)
            roi = preprocess_input(roi)

            # 4. Predict
            preds = model.predict(roi, verbose=0)[0]
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]
            person_name = class_names[class_idx]

            # 5. Visualize
            if confidence > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)
                label_text = f"{person_name}"
            else:
                color = (0, 0, 255)
                label_text = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label_text} ({int(confidence*100)}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            pass

    cv2.imshow("Transfer Learning Face ID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()