import cv2
import numpy as np
import tensorflow as tf


print("Loading Model...")
model = tf.keras.models.load_model('mnist_mlp.h5')
print("Model Loaded!")

def empty(a):
    pass


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# 3. Create a Settings Window
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 300, 100)
cv2.createTrackbar("Threshold", "Settings", 110, 255, empty)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check camera permissions.")
        break

    # 4. Draw a Box in the Center
    height, width, _ = frame.shape
    x1, y1 = int(width/2) - 80, int(height/2) - 80
    x2, y2 = int(width/2) + 80, int(height/2) + 80
    

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 5. Extract the Image inside the box (Region of Interest - ROI)
    roi = frame[y1:y2, x1:x2]
    
    # 6. Image Processing
    # Step A: Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Step B: Thresholding (Convert to Black and White)
    thresh_val = cv2.getTrackbarPos("Threshold", "Settings")
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Step C: Resize to 28x28 pixels 
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Step D: Normalize and Reshape
    input_data = resized / 255.0
    input_data = np.reshape(input_data, (1, 28, 28))

    # 7. Prediction
    prediction = model.predict(input_data, verbose=0)
    class_index = np.argmax(prediction) # The predicted number (0-9)
    probability = np.max(prediction)    # How confident is the AI?

    # 8. Display Results
    cv2.imshow("AI Eye", cv2.resize(resized, (200, 200))) 

    # Only show prediction if confidence is high (> 60%)
    if probability > 0.6:
        text = f"Digit: {class_index}"
        color = (0, 255, 0)
    else:
        text = "???"
        color = (0, 0, 255)

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"Conf: {int(probability*100)}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Webcam Scanner", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()